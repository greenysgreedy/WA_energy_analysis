"""Parse raw AEMO CSVs into tidy parquet in ``data/processed/``, with schema checks.

Conventions applied to every dataset, per ``.claude/skills/aemo-wem-data``:

* timestamps are Perth time (AWST, UTC+8, no daylight saving) and mark the **start** of
  the interval. They are parsed with an explicit day-first format and stored UTC-naive;
  nothing in this pipeline localises them, because AWST has no transitions.
* the trading day runs 08:00 to 08:00, so ``trading_date = (interval_start - 8h).date()``
* raw rows are sorted newest first, so everything is re-sorted ascending
* ``Extracted At`` is populated on the first row only and is dropped
* ``Failure Reason`` is kept and counted; null prices are only tolerated on those rows

Checks are reported rather than silently repaired. Duplicate keys raise, because every
downstream revenue sum would be wrong; missing intervals are listed, because AEMO does
occasionally publish short trading days and filling them would invent data.

Usage::

    .venv\\Scripts\\python.exe -m src.clean                  # everything
    .venv\\Scripts\\python.exe -m src.clean --only prices
    .venv\\Scripts\\python.exe -m src.clean --only market_schedule --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.paths import (
    DT_HOURS,
    INTERVALS_PER_DAY_5MIN,
    INTERVALS_PER_DAY_30MIN,
    PROCESSED,
    RAW,
    REFORM_START,
    TRADING_DAY_START_HOUR,
)

#: AEMO publishes day-first timestamps. Never let pandas infer this: "01/02/2024" is
#: 1 February, and an inferred parse silently reads it as 2 January for days <= 12.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

#: First post-reform interval. The trading day starting 1 October 2023 begins at 08:00.
REFORM_FIRST_INTERVAL = pd.Timestamp(REFORM_START) + pd.Timedelta(hours=TRADING_DAY_START_HOUR)

FCESS_SERVICES = (
    "contingency_raise",
    "contingency_lower",
    "regulation_raise",
    "regulation_lower",
    "rocof",
)

PRICE_COLUMNS = {
    "Energy Clearing Price": "energy_price",
    "Contingency Raise Clearing Price": "contingency_raise_price",
    "Contingency Lower Clearing Price": "contingency_lower_price",
    "Regulation Raise Clearing Price": "regulation_raise_price",
    "Regulation Lower Clearing Price": "regulation_lower_price",
    "RoCoF Clearing Price": "rocof_price",
}

SCHEDULE_COLUMNS = {
    "Energy Schedule": "energy_mw",
    "Contingency Raise Schedule": "contingency_raise_mw",
    "Contingency Lower Schedule": "contingency_lower_mw",
    "Regulation Raise Schedule": "regulation_raise_mw",
    "Regulation Lower Schedule": "regulation_lower_mw",
    "RoCoF Schedule": "rocof_mw",
}


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def parse_interval(series: pd.Series) -> pd.Series:
    """Parse an AEMO timestamp column to UTC-naive AWST interval-start timestamps."""
    return pd.to_datetime(series, format=TIMESTAMP_FORMAT, errors="raise")


def trading_date(interval_start: pd.Series) -> pd.Series:
    """Trading date of each interval. The WEM trading day runs 08:00 to 08:00 AWST."""
    return (interval_start - pd.Timedelta(hours=TRADING_DAY_START_HOUR)).dt.normalize()


def read_aemo_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a raw AEMO CSV, dropping the ``Extracted At`` column.

    ``low_memory=False`` avoids a mixed-dtype warning on ``Extracted At``, which is
    populated on the first row only and dropped immediately.
    """
    kwargs.setdefault("low_memory", False)
    frame = pd.read_csv(path, **kwargs)
    return frame.drop(columns=[c for c in frame.columns if c.strip() == "Extracted At"])


def _tidy_intervals(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Rename the time column, add ``trading_date`` and sort ascending."""
    frame = frame.rename(columns={time_col: "interval_start"})
    frame["interval_start"] = parse_interval(frame["interval_start"])
    frame["trading_date"] = trading_date(frame["interval_start"])
    return frame


def post_reform_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep intervals from the first post-reform trading day onwards."""
    return frame.loc[frame["interval_start"] >= REFORM_FIRST_INTERVAL]


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of the schema checks for one cleaned table."""

    name: str
    rows: int
    first_interval: pd.Timestamp | None = None
    last_interval: pd.Timestamp | None = None
    trading_days: int = 0
    duplicate_keys: int = 0
    short_days: dict = field(default_factory=dict)
    long_days: dict = field(default_factory=dict)
    null_values: dict = field(default_factory=dict)
    failure_reason_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.duplicate_keys == 0 and not self.long_days

    def summary_row(self) -> dict:
        return {
            "table": self.name,
            "rows": self.rows,
            "first": self.first_interval,
            "last": self.last_interval,
            "trading_days": self.trading_days,
            "dup_keys": self.duplicate_keys,
            "short_days": len(self.short_days),
            "long_days": len(self.long_days),
            "failure_rows": self.failure_reason_rows,
            "null_cols": len(self.null_values),
        }

    def report(self) -> str:
        lines = [f"{self.name}: {self.rows:,} rows"]
        if self.first_interval is not None:
            lines.append(f"  interval range   {self.first_interval}  ..  {self.last_interval}")
        if self.trading_days:
            lines.append(f"  trading days     {self.trading_days:,}")
        lines.append(f"  duplicate keys   {self.duplicate_keys:,}")
        if self.failure_reason_rows:
            lines.append(f"  failure reasons  {self.failure_reason_rows:,} row(s)")
        if self.short_days:
            shown = list(self.short_days.items())[:8]
            detail = ", ".join(f"{d.date()}={n}" for d, n in shown)
            more = "" if len(self.short_days) <= 8 else f" (+{len(self.short_days) - 8} more)"
            lines.append(f"  short days       {len(self.short_days)}: {detail}{more}")
        if self.long_days:
            shown = list(self.long_days.items())[:8]
            detail = ", ".join(f"{d.date()}={n}" for d, n in shown)
            lines.append(f"  LONG days        {len(self.long_days)}: {detail}")
        for column, count in self.null_values.items():
            lines.append(f"  nulls            {column}: {count:,}")
        for note in self.notes:
            lines.append(f"  note             {note}")
        return "\n".join(lines)


def check_interval_table(
    frame: pd.DataFrame,
    name: str,
    *,
    per_day: int,
    key: list[str],
    value_columns: list[str] | None = None,
    group_column: str | None = None,
) -> CheckResult:
    """Run the standard checks on a cleaned interval table.

    Parameters
    ----------
    per_day
        Expected intervals per trading day: 288 at five minutes, 48 at thirty.
    key
        Columns that must uniquely identify a row.
    group_column
        For per-facility tables, interval counts are checked per facility per trading
        day rather than per trading day, since the facility register changes over time.
    """
    result = CheckResult(name=name, rows=len(frame))
    if frame.empty:
        result.notes.append("table is empty")
        return result

    result.first_interval = frame["interval_start"].min()
    result.last_interval = frame["interval_start"].max()
    result.trading_days = int(frame["trading_date"].nunique())
    result.duplicate_keys = int(frame.duplicated(subset=key).sum())

    group_keys = ["trading_date"] + ([group_column] if group_column else [])
    counts = frame.groupby(group_keys, observed=True)["interval_start"].nunique()

    # A facility present for only part of a day is normal (commissioning, outages), so
    # shortfalls are only flagged for whole-of-market tables.
    if group_column is None:
        short = counts[counts < per_day]
        result.short_days = {k: int(v) for k, v in short.items()}
    long = counts[counts > per_day]
    result.long_days = {k: int(v) for k, v in long.items()}

    if "failure_reason" in frame.columns:
        failed = frame["failure_reason"].notna()
        result.failure_reason_rows = int(failed.sum())
    else:
        failed = pd.Series(False, index=frame.index)

    for column in value_columns or []:
        nulls = int(frame.loc[~failed, column].isna().sum())
        if nulls:
            result.null_values[column] = nulls

    if result.duplicate_keys:
        raise ValueError(
            f"{name}: {result.duplicate_keys:,} duplicate rows on {key}. "
            "Every downstream revenue total would be wrong, so this is fatal."
        )
    return result


def price_extremes(frame: pd.DataFrame, column: str) -> dict:
    """Min, max and the number of intervals sitting at either extreme."""
    series = frame[column].dropna()
    if series.empty:
        return {}
    lo, hi = float(series.min()), float(series.max())
    return {
        "min": lo,
        "max": hi,
        "n_at_min": int((series == lo).sum()),
        "n_at_max": int((series == hi).sum()),
        "n_negative": int((series < 0).sum()),
    }


# --------------------------------------------------------------------------------------
# cleaners
# --------------------------------------------------------------------------------------


def clean_prices_5min() -> tuple[pd.DataFrame, CheckResult]:
    """5-minute energy and FCESS clearing prices, one row per dispatch interval."""
    files = sorted((RAW / "market-clearing-prices-csv").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/market-clearing-prices-csv")

    frames = [read_aemo_csv(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={**PRICE_COLUMNS, "Failure Reason": "failure_reason"})
    frame = _tidy_intervals(frame, "Dispatch Interval")
    frame = post_reform_only(frame).sort_values("interval_start", ignore_index=True)

    value_columns = list(PRICE_COLUMNS.values())
    frame = frame[["interval_start", "trading_date", *value_columns, "failure_reason"]]

    result = check_interval_table(
        frame,
        "prices_5min",
        per_day=INTERVALS_PER_DAY_5MIN,
        key=["interval_start"],
        value_columns=value_columns,
    )
    for column in value_columns:
        stats = price_extremes(frame, column)
        if stats:
            result.notes.append(
                f"{column}: min {stats['min']:.2f}, max {stats['max']:.2f}, "
                f"{stats['n_negative']:,} negative interval(s)"
            )
    return frame, result


def clean_prices_30min() -> tuple[pd.DataFrame, CheckResult]:
    """30-minute Reference Trading Price.

    Retained for the settlement-price comparison in ``docs/assumptions.md``. Whether it
    is used for revenue depends on which price the Rules settle energy on.
    """
    files = sorted((RAW / "reference-trading-price-csv").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/reference-trading-price-csv")

    frame = pd.concat([read_aemo_csv(p) for p in files], ignore_index=True)
    frame = frame.rename(
        columns={"Reference Trading Price": "reference_trading_price", "Failure Reason": "failure_reason"}
    )
    frame = _tidy_intervals(frame, "Trading Interval")
    frame = post_reform_only(frame).sort_values("interval_start", ignore_index=True)
    frame = frame[["interval_start", "trading_date", "reference_trading_price", "failure_reason"]]

    result = check_interval_table(
        frame,
        "prices_30min",
        per_day=INTERVALS_PER_DAY_30MIN,
        key=["interval_start"],
        value_columns=["reference_trading_price"],
    )
    stats = price_extremes(frame, "reference_trading_price")
    if stats:
        result.notes.append(
            f"reference_trading_price: min {stats['min']:.2f}, max {stats['max']:.2f}"
        )
    return frame, result


def _clean_monthly_facility_dataset(
    dataset: str,
    out_name: str,
    time_col: str,
    renames: dict[str, str],
    value_columns: list[str],
    *,
    force: bool = False,
) -> CheckResult:
    """Clean a large per-facility 5-minute dataset one month at a time.

    These datasets are gigabytes of CSV, so each monthly file is cleaned and written to
    its own parquet part. Checks are accumulated across parts.
    """
    files = sorted((RAW / dataset).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no files in data/raw/{dataset}")

    out_dir = PROCESSED / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = CheckResult(name=out_name, rows=0)
    long_days: dict = {}
    null_counts: dict[str, int] = {}
    failure_rows = 0
    first_interval: pd.Timestamp | None = None
    last_interval: pd.Timestamp | None = None
    trading_days: set = set()

    for path in files:
        part = out_dir / f"{path.stem}.parquet"
        if part.exists() and not force:
            existing = pd.read_parquet(part, columns=["interval_start", "trading_date"])
            if not existing.empty:
                totals.rows += len(existing)
                trading_days.update(existing["trading_date"].unique())
                lo, hi = existing["interval_start"].min(), existing["interval_start"].max()
                first_interval = lo if first_interval is None else min(first_interval, lo)
                last_interval = hi if last_interval is None else max(last_interval, hi)
            print(f"    skip   {path.name}")
            continue

        frame = read_aemo_csv(path)
        frame = frame.rename(
            columns={**renames, "Facility Code": "facility_code", "Failure Reason": "failure_reason"}
        )
        frame = _tidy_intervals(frame, time_col)
        frame = post_reform_only(frame)
        if frame.empty:
            print(f"    empty  {path.name} (no post-reform intervals)")
            continue

        frame = frame.sort_values(["interval_start", "facility_code"], ignore_index=True)
        keep = ["interval_start", "trading_date", "facility_code", *value_columns]
        if "failure_reason" in frame.columns:
            keep.append("failure_reason")
        frame = frame[keep]
        frame["facility_code"] = frame["facility_code"].astype("category")

        part_result = check_interval_table(
            frame,
            f"{out_name}:{path.stem}",
            per_day=INTERVALS_PER_DAY_5MIN,
            key=["interval_start", "facility_code"],
            value_columns=value_columns,
            group_column="facility_code",
        )

        totals.rows += part_result.rows
        long_days.update(part_result.long_days)
        failure_rows += part_result.failure_reason_rows
        for column, count in part_result.null_values.items():
            null_counts[column] = null_counts.get(column, 0) + count
        trading_days.update(frame["trading_date"].unique())
        lo, hi = part_result.first_interval, part_result.last_interval
        first_interval = lo if first_interval is None else min(first_interval, lo)
        last_interval = hi if last_interval is None else max(last_interval, hi)

        frame.to_parquet(part, index=False, compression="zstd")
        print(f"    wrote  {part.name} ({part_result.rows:,} rows, {part.stat().st_size / 1e6:.1f} MB)")

    totals.first_interval = first_interval
    totals.last_interval = last_interval
    totals.trading_days = len(trading_days)
    totals.long_days = long_days
    totals.null_values = null_counts
    totals.failure_reason_rows = failure_rows
    totals.notes.append(f"written as {len(list(out_dir.glob('*.parquet')))} monthly parquet part(s)")
    return totals


def clean_market_schedule(*, force: bool = False) -> CheckResult:
    """Per-facility energy and FCESS schedule quantities, five-minutely.

    This is the only public source of FCESS enablement by facility. The columns are
    *schedule* quantities from the dispatch run, which need not equal the enablement
    quantities used in settlement; that caveat is recorded in ``docs/assumptions.md``.
    """
    return _clean_monthly_facility_dataset(
        "market-schedule-csv",
        "market_schedule",
        "Dispatch Interval",
        SCHEDULE_COLUMNS,
        list(SCHEDULE_COLUMNS.values()),
        force=force,
    )


def clean_facility_scada(*, force: bool = False) -> CheckResult:
    """Per-facility metered output, five-minutely.

    ``Average MWh`` is stored unchanged as ``average_mwh``. Whether it is an average MW
    or an MWh per five minutes is settled empirically by
    :func:`diagnose_scada_unit` and recorded in ``docs/assumptions.md`` before any
    revenue is computed from it.
    """
    return _clean_monthly_facility_dataset(
        "facility-scada-csv",
        "facility_scada",
        "Dispatch Interval",
        {"Average MWh": "average_mwh"},
        ["average_mwh"],
        force=force,
    )


def parse_flexible_date(series: pd.Series) -> pd.Series:
    """Parse a date column that AEMO publishes inconsistently across years.

    The obligation-intervals files use ISO dates from 2024 onwards but day-first dates
    in the 2023 file. Both are tried explicitly rather than inferred, because an
    inferred parse would read 1 October as 10 January for the first twelve days of a
    month.
    """
    text = series.astype("string").str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    missing = iso.isna()
    if missing.any():
        iso.loc[missing] = pd.to_datetime(text[missing], format="%d/%m/%Y", errors="raise")
    return iso


def clean_esroi() -> tuple[pd.DataFrame, CheckResult]:
    """Electric Storage Resource Obligation Intervals, one row per obligation window.

    The published schema changed within the study period: files up to 2025 give a single
    market-wide window per trading day, while the 2026 file gives one row per
    Separately Certified Component. Rows without a component are market-wide and apply
    to every storage facility holding Capacity Credits.
    """
    files = sorted((RAW / "electric-storage-resource-obligation-intervals").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/electric-storage-resource-obligation-intervals")

    frame = pd.concat([read_aemo_csv(p) for p in files], ignore_index=True)
    frame = frame.rename(
        columns={
            "Capacity Year": "capacity_year",
            "Trading Date": "trading_date",
            "Component": "component",
            "ESROI Trading Interval Start": "esroi_start",
            "ESROI Trading Interval End": "esroi_end",
        }
    )
    if "component" not in frame.columns:
        frame["component"] = pd.NA

    frame["trading_date"] = parse_flexible_date(frame["trading_date"])

    # The component code is the facility code with an ESR unit suffix, e.g.
    # ALINTA_WGP_ESR1_ESR_01 -> ALINTA_WGP_ESR1. A null component means the window is
    # market-wide for that trading day.
    frame["facility_code"] = (
        frame["component"].astype("string").str.replace(r"_ESR_\d+$", "", regex=True)
    )
    frame["market_wide"] = frame["component"].isna()

    start = pd.to_timedelta(frame["esroi_start"].astype("string").str.strip() + ":00")
    end = pd.to_timedelta(frame["esroi_end"].astype("string").str.strip() + ":00")
    frame["esroi_start_dt"] = frame["trading_date"] + start
    frame["esroi_end_dt"] = frame["trading_date"] + end

    # The published end time is the start of the final obligation interval, so the
    # window spans ((end - start) / 30 min) + 1 Trading Intervals. The ESR Obligation
    # Duration is eight Trading Intervals throughout the study period
    # (docs/assumptions.md C2, D2), which this count is checked against below.
    frame["n_trading_intervals"] = ((end - start) / pd.Timedelta(minutes=30)).astype(int) + 1
    frame = frame.sort_values(["trading_date", "component"], ignore_index=True)

    result = CheckResult(name="esroi", rows=len(frame))
    result.first_interval = frame["esroi_start_dt"].min()
    result.last_interval = frame["esroi_end_dt"].max()
    result.trading_days = int(frame["trading_date"].nunique())
    result.duplicate_keys = int(frame.duplicated(subset=["trading_date", "component"]).sum())
    if result.duplicate_keys:
        raise ValueError(f"esroi: {result.duplicate_keys} duplicate (trading_date, component) rows")

    counts = frame["n_trading_intervals"].value_counts().sort_index()
    result.notes.append(
        "obligation window length in Trading Intervals: "
        + ", ".join(f"{k}:{v:,}" for k, v in counts.items())
    )
    windows = (
        frame.groupby("capacity_year", observed=True)[["esroi_start", "esroi_end"]]
        .agg(lambda s: s.mode().iat[0])
        .apply(lambda r: f"{r['esroi_start']}-{r['esroi_end']}", axis=1)
    )
    result.notes.append("window by capacity year: " + ", ".join(f"{k} {v}" for k, v in windows.items()))
    result.notes.append(
        f"market-wide rows: {int(frame['market_wide'].sum()):,}; "
        f"per-component rows: {int((~frame['market_wide']).sum()):,} "
        f"covering {frame['facility_code'].nunique()} facilities"
    )
    return frame, result


def clean_facilities() -> tuple[pd.DataFrame, CheckResult]:
    """Post-reform facility standing data."""
    path = RAW / "post-facilities" / "facilities.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = read_aemo_csv(path)
    frame = frame.rename(
        columns={
            "Participant Code": "participant_code",
            "Participant Name": "participant_name",
            "Facility Code": "facility_code",
            "Facility Class": "facility_class",
            "System Size (MW)": "registered_mw",
            "Remaining capacity from embedded generator (MW)": "embedded_remaining_mw",
        }
    )
    frame = frame.sort_values("facility_code", ignore_index=True)

    result = CheckResult(name="facilities", rows=len(frame))
    result.duplicate_keys = int(frame.duplicated(subset=["facility_code"]).sum())
    if result.duplicate_keys:
        raise ValueError("facilities: duplicate facility_code rows")
    result.notes.append(f"classes: {dict(frame['facility_class'].value_counts())}")
    return frame, result


def clean_fcess_accreditation() -> tuple[pd.DataFrame, CheckResult]:
    """FCESS accreditation standing data: maximum accredited quantities per facility.

    This is standing accreditation, not enablement. Enablement comes from the market
    schedule dataset.
    """
    path = RAW / "fcess" / "fcess.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = read_aemo_csv(path)
    frame = frame.rename(
        columns={
            "Participant Code": "participant_code",
            "Participant Name": "participant_name",
            "Facility Code": "facility_code",
            "Facility Class": "facility_class",
            "Max Accredited Regulation Raise": "max_regulation_raise_mw",
            "Max Accredited Regulation Lower": "max_regulation_lower_mw",
            "Max Accredited Contingency Raise": "max_contingency_raise_mw",
            "Max Accredited Contingency Lower": "max_contingency_lower_mw",
            "Max Accredited ROCOF": "max_rocof_mw",
            "Facility Speed Factor": "speed_factor",
            "RoCoF Ride-Through Capability": "rocof_ride_through",
        }
    )
    frame = frame.sort_values("facility_code", ignore_index=True)

    result = CheckResult(name="fcess_accreditation", rows=len(frame))
    result.duplicate_keys = int(frame.duplicated(subset=["facility_code"]).sum())
    if result.duplicate_keys:
        raise ValueError("fcess_accreditation: duplicate facility_code rows")
    accredited = frame["max_contingency_raise_mw"].notna().sum()
    result.notes.append(f"{accredited} facilities with a contingency raise accreditation")
    return frame, result


def clean_capacity_credits() -> tuple[pd.DataFrame, CheckResult]:
    """Capacity Credits per facility per capacity year, melted to long form."""
    files = sorted((RAW / "capacity-credits-since-market-start").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/capacity-credits-since-market-start")

    # Later files supersede earlier ones, so take the most recent vintage.
    frame = read_aemo_csv(files[-1])
    id_columns = ["Market Participant", "Facility"]
    year_columns = [c for c in frame.columns if c not in id_columns]

    long = frame.melt(
        id_vars=id_columns,
        value_vars=year_columns,
        var_name="capacity_year",
        value_name="capacity_credits",
    ).rename(columns={"Market Participant": "participant_name", "Facility": "facility_code"})
    long = long.dropna(subset=["capacity_credits"])
    long["capacity_credits"] = long["capacity_credits"].astype(float)
    long = long.sort_values(["capacity_year", "facility_code"], ignore_index=True)

    result = CheckResult(name="capacity_credits", rows=len(long))
    result.duplicate_keys = int(long.duplicated(subset=["facility_code", "capacity_year"]).sum())
    if result.duplicate_keys:
        raise ValueError("capacity_credits: duplicate (facility_code, capacity_year) rows")
    result.notes.append(f"source file: {files[-1].name}")
    result.notes.append(f"capacity years: {long['capacity_year'].min()} .. {long['capacity_year'].max()}")
    return long, result


#: Price type labels changed with the Reserve Capacity Reform: the floating price a new
#: merchant entrant would receive was "Reserve Capacity" up to the 2026 capacity year
#: and "Peak Reserve Capacity" from 2027. Both map to one canonical label so the
#: economics stage does not have to know when the rename happened.
RCP_TYPE_ALIASES = {
    "reserve capacity": "Reserve Capacity",
    "peak reserve capacity": "Reserve Capacity",
}


def clean_reserve_capacity_prices() -> tuple[pd.DataFrame, CheckResult]:
    """Reserve Capacity Price per capacity year, one file per year.

    The published schema drifts: the 2028 file omits the Capacity Credits column and
    the price-type labels were renamed from 2027. Only the capacity year, price type and
    price are taken, which are present and unambiguous in every file.
    """
    files = sorted((RAW / "reserve-capacity-prices").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/reserve-capacity-prices")

    frames = []
    for path in files:
        raw = read_aemo_csv(path)
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
        price_column = next(c for c in raw.columns if c.startswith("price_("))
        part = raw[["capacity_year", "price_type", price_column]].rename(
            columns={price_column: "price_per_mw_year"}
        )
        part["source_file"] = path.name
        frames.append(part)

    frame = pd.concat(frames, ignore_index=True)
    frame["capacity_year"] = frame["capacity_year"].astype(int)
    frame["price_type"] = frame["price_type"].astype("string").str.strip().str.strip('"')
    frame["price_per_mw_year"] = pd.to_numeric(frame["price_per_mw_year"], errors="coerce")
    frame["price_type_canonical"] = (
        frame["price_type"].str.lower().map(RCP_TYPE_ALIASES).fillna(frame["price_type"])
    )
    frame = frame.dropna(subset=["price_per_mw_year"]).sort_values(
        ["capacity_year", "price_type"], ignore_index=True
    )

    result = CheckResult(name="reserve_capacity_prices", rows=len(frame))
    result.duplicate_keys = int(frame.duplicated(subset=["capacity_year", "price_type"]).sum())
    if result.duplicate_keys:
        raise ValueError("reserve_capacity_prices: duplicate (capacity_year, price_type) rows")

    headline = frame.loc[frame["price_type_canonical"] == "Reserve Capacity"]
    result.notes.append(f"{len(files)} source file(s), capacity years {frame['capacity_year'].min()}"
                        f" to {frame['capacity_year'].max()}")
    result.notes.append(
        "Reserve Capacity Price $/MW-year: "
        + ", ".join(
            f"{int(r.capacity_year)}={r.price_per_mw_year:,.0f}"
            for r in headline.itertuples()
        )
    )
    return frame, result


def clean_market_requirements() -> tuple[pd.DataFrame, CheckResult]:
    """FCESS requirement quantities per interval."""
    files = sorted((RAW / "market-requirements-csv").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no files in data/raw/market-requirements-csv")

    frame = pd.concat([read_aemo_csv(p) for p in files], ignore_index=True)
    time_col = next(c for c in frame.columns if "Interval" in c)
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame.rename(columns={"Failure Reason": "failure_reason"})
    frame = _tidy_intervals(frame, time_col)
    frame = post_reform_only(frame).sort_values("interval_start", ignore_index=True)
    frame.columns = [
        c if c in {"interval_start", "trading_date", "failure_reason"}
        else c.strip().lower().replace(" ", "_")
        for c in frame.columns
    ]

    result = check_interval_table(
        frame,
        "market_requirements",
        per_day=INTERVALS_PER_DAY_5MIN,
        key=["interval_start"],
    )
    result.notes.append(f"columns: {list(frame.columns)}")
    return frame, result


# --------------------------------------------------------------------------------------
# SCADA unit diagnosis
# --------------------------------------------------------------------------------------


#: Confirmed by ``scripts/diagnose_scada_unit.py``: SCADA ``Average MWh`` is MWh per
#: five-minute interval, so power in MW is the column times twelve
#: (docs/assumptions.md D5).
SCADA_MWH_PER_INTERVAL = True
SCADA_MW_PER_MWH = 1.0 / DT_HOURS


def scada_mw(average_mwh: pd.Series) -> pd.Series:
    """Convert the SCADA energy column to MW."""
    return average_mwh * SCADA_MW_PER_MWH


def diagnose_scada_unit(sample_months: int = 3, top_n: int = 12) -> pd.DataFrame:
    """Decide empirically whether SCADA ``Average MWh`` is average MW or MWh per interval.

    For each large facility the observed peak is compared with its registered MW and with
    registered MW / 12. Whichever is closer identifies the unit. The verdict is recorded
    in ``docs/assumptions.md``; nothing in the pipeline converts the column until it is.
    """
    parts = sorted((PROCESSED / "facility_scada").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("clean facility_scada first")

    frames = [pd.read_parquet(p, columns=["facility_code", "average_mwh"]) for p in parts[:sample_months]]
    scada = pd.concat(frames, ignore_index=True)
    peaks = scada.groupby("facility_code", observed=True)["average_mwh"].max()

    facilities = pd.read_parquet(PROCESSED / "facilities.parquet")
    registered = facilities.set_index("facility_code")["registered_mw"].astype(float)

    joined = pd.concat([peaks.rename("observed_peak"), registered.rename("registered_mw")], axis=1)
    joined = joined.dropna()
    joined = joined.loc[joined["registered_mw"] >= 50].copy()

    joined["ratio_to_mw"] = joined["observed_peak"] / joined["registered_mw"]
    joined["ratio_to_mwh_per_interval"] = joined["observed_peak"] / (joined["registered_mw"] / 12.0)
    joined["err_if_mw"] = (joined["ratio_to_mw"] - 1.0).abs()
    joined["err_if_mwh"] = (joined["ratio_to_mwh_per_interval"] - 1.0).abs()
    joined["verdict"] = joined.apply(
        lambda row: "average MW" if row["err_if_mw"] < row["err_if_mwh"] else "MWh per interval",
        axis=1,
    )
    return joined.sort_values("registered_mw", ascending=False).head(top_n)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

SIMPLE_CLEANERS = {
    "prices_5min": clean_prices_5min,
    "prices_30min": clean_prices_30min,
    "market_requirements": clean_market_requirements,
    "esroi": clean_esroi,
    "facilities": clean_facilities,
    "fcess_accreditation": clean_fcess_accreditation,
    "capacity_credits": clean_capacity_credits,
    "reserve_capacity_prices": clean_reserve_capacity_prices,
}

PARTITIONED_CLEANERS = {
    "market_schedule": clean_market_schedule,
    "facility_scada": clean_facility_scada,
}

ALL_TABLES = tuple(SIMPLE_CLEANERS) + tuple(PARTITIONED_CLEANERS)


def run(only: list[str] | None = None, *, force: bool = False) -> pd.DataFrame:
    """Clean the requested tables and return the one-row-per-table summary."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    names = only or list(ALL_TABLES)
    results: list[CheckResult] = []

    for name in names:
        print(f"\n{name}")
        if name in SIMPLE_CLEANERS:
            out = PROCESSED / f"{name}.parquet"
            if out.exists() and not force:
                print("    skip (already built; pass --force to rebuild)")
                frame = pd.read_parquet(out)
                results.append(CheckResult(name=name, rows=len(frame), notes=["cached"]))
                continue
            frame, result = SIMPLE_CLEANERS[name]()
            frame.to_parquet(out, index=False, compression="zstd")
            print(f"    wrote  {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
            results.append(result)
        else:
            results.append(PARTITIONED_CLEANERS[name](force=force))
        print(results[-1].report())

    summary = pd.DataFrame([r.summary_row() for r in results])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=ALL_TABLES, help="clean only this table")
    parser.add_argument("--force", action="store_true", help="rebuild tables already present")
    args = parser.parse_args()

    summary = run(args.only, force=args.force)
    print("\n\nsummary")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
