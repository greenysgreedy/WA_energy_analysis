"""Load cleaned parquet from ``data/processed/`` into the shapes the analysis needs.

Everything downstream reads through here so the settlement-price convention is applied
in exactly one place: energy is settled on the thirty-minute Reference Trading Price
(``docs/assumptions.md`` E1) while FCESS is paid per five-minute Dispatch Interval
(F2). :func:`load_prices` therefore returns both the five-minute clearing prices and
the Reference Trading Price broadcast onto the five-minute grid, and
:func:`settlement_price_column` names the one the model should settle on.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.model import SERVICES
from src.paths import INTERVALS_PER_DAY_5MIN, PROCESSED, TRADING_DAY_START_HOUR

#: The price column energy revenue is settled on. See docs/assumptions.md E1.
SETTLEMENT_PRICE_COLUMN = "settlement_price"

#: Five-minute clearing price for energy, kept for the dispatch-versus-settlement
#: comparison and for the sensitivity that settles on five-minute prices instead.
CLEARING_PRICE_COLUMN = "energy_price"

FCESS_PRICE_COLUMNS = tuple(f"{service}_price" for service in SERVICES)
FCESS_SCHEDULE_COLUMNS = tuple(f"{service}_mw" for service in SERVICES)


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.download_data' then 'python -m src.clean'."
        )
    return path


def trading_interval_start(interval_start: pd.Series) -> pd.Series:
    """Start of the thirty-minute Trading Interval containing each dispatch interval."""
    return interval_start.dt.floor("30min")


@lru_cache(maxsize=1)
def _prices_cached() -> pd.DataFrame:
    five = pd.read_parquet(_require(PROCESSED / "prices_5min.parquet"))
    thirty = pd.read_parquet(_require(PROCESSED / "prices_30min.parquet"))

    five["trading_interval"] = trading_interval_start(five["interval_start"])
    thirty = thirty.rename(columns={"interval_start": "trading_interval"})

    merged = five.merge(
        thirty[["trading_interval", "reference_trading_price"]],
        on="trading_interval",
        how="left",
        validate="many_to_one",
    )
    merged[SETTLEMENT_PRICE_COLUMN] = merged["reference_trading_price"]

    # A handful of Trading Intervals have no published Reference Trading Price. Rather
    # than drop or fill them, fall back to the time-weighted average of the five-minute
    # clearing prices in the interval, which is how 7.11A.1(b) defines the price anyway.
    missing = merged[SETTLEMENT_PRICE_COLUMN].isna()
    if missing.any():
        derived = merged.groupby("trading_interval", observed=True)[CLEARING_PRICE_COLUMN].transform(
            "mean"
        )
        merged.loc[missing, SETTLEMENT_PRICE_COLUMN] = derived[missing]

    return merged


def load_prices(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Five-minute price table with the settlement price attached.

    Parameters
    ----------
    start, end
        Inclusive bounds on ``trading_date``.

    Returns
    -------
    DataFrame
        ``interval_start``, ``trading_date``, ``trading_interval``, the five FCESS
        clearing prices, the five-minute ``energy_price``, ``reference_trading_price``
        and ``settlement_price``.
    """
    frame = _prices_cached()
    if start is not None:
        frame = frame.loc[frame["trading_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["trading_date"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


def complete_trading_days(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only trading days with the full 288 dispatch intervals.

    Short days are reported by the cleaning step and excluded here rather than padded,
    because a padded day would either invent prices or silently shorten the dispatch
    horizon. The count of excluded days is reported by the caller.
    """
    counts = prices.groupby("trading_date", observed=True)["interval_start"].transform("size")
    return prices.loc[counts == INTERVALS_PER_DAY_5MIN].reset_index(drop=True)


def price_arrays(prices: pd.DataFrame, *, settle_on: str = SETTLEMENT_PRICE_COLUMN):
    """Reshape a complete-days price table into per-day arrays.

    Returns
    -------
    (dates, energy, fcess)
        ``dates`` has length ``n_days``; ``energy`` is ``(n_days, 288)``; ``fcess`` is
        ``(n_days, 5, 288)`` in :data:`src.model.SERVICES` order.
    """
    ordered = prices.sort_values("interval_start")
    dates = ordered["trading_date"].drop_duplicates().to_numpy()
    n_days = len(dates)

    if len(ordered) != n_days * INTERVALS_PER_DAY_5MIN:
        raise ValueError(
            "price_arrays needs complete trading days; "
            f"got {len(ordered):,} rows for {n_days:,} days. Call complete_trading_days first."
        )

    energy = ordered[settle_on].to_numpy(dtype=float).reshape(n_days, INTERVALS_PER_DAY_5MIN)
    fcess = np.stack(
        [
            ordered[column].to_numpy(dtype=float).reshape(n_days, INTERVALS_PER_DAY_5MIN)
            for column in FCESS_PRICE_COLUMNS
        ],
        axis=1,
    )
    return dates, energy, np.nan_to_num(fcess)


# --------------------------------------------------------------------------------------
# facilities and storage register
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_facilities() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED / "facilities.parquet"))


@lru_cache(maxsize=1)
def load_fcess_accreditation() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED / "fcess_accreditation.parquet"))


@lru_cache(maxsize=1)
def load_capacity_credits() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED / "capacity_credits.parquet"))


@lru_cache(maxsize=1)
def load_reserve_capacity_prices() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED / "reserve_capacity_prices.parquet"))


@lru_cache(maxsize=1)
def load_esroi() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED / "esroi.parquet"))


#: Facility codes are the only reliable marker of storage in the public standing data:
#: there is no technology-type column. Confirmed against the FCESS accreditation and
#: capacity credit registers, which list the same nine facilities.
STORAGE_CODE_PATTERN = r"_ESR\d*$|_BESS\d*$"


def storage_facilities() -> pd.DataFrame:
    """The SWIS storage register: code, participant, registered MW and accreditation."""
    facilities = load_facilities()
    storage = facilities.loc[
        facilities["facility_code"].str.contains(STORAGE_CODE_PATTERN, regex=True, na=False)
    ].copy()

    accreditation = load_fcess_accreditation().set_index("facility_code")
    for column in (
        "max_regulation_raise_mw",
        "max_regulation_lower_mw",
        "max_contingency_raise_mw",
        "max_contingency_lower_mw",
        "max_rocof_mw",
        "speed_factor",
    ):
        storage[column] = storage["facility_code"].map(accreditation[column])

    return storage.sort_values("facility_code", ignore_index=True)


def capacity_credits_for(facility_code: str) -> pd.Series:
    """Capacity Credits by capacity year for one facility."""
    credits = load_capacity_credits()
    rows = credits.loc[credits["facility_code"] == facility_code]
    return rows.set_index("capacity_year")["capacity_credits"].sort_index()


def reserve_capacity_price(capacity_year: int, price_type: str = "Reserve Capacity") -> float | None:
    """Published Reserve Capacity Price in $/MW-year for a capacity year.

    Capacity years are labelled by their starting calendar year in this dataset, so
    2024 means the year beginning 1 October 2024 (docs/assumptions.md C5, C6). The
    default price type is the floating price a new merchant entrant would receive,
    under its canonical label: it was published as "Reserve Capacity" up to the 2026
    capacity year and "Peak Reserve Capacity" from 2027.
    """
    prices = load_reserve_capacity_prices()
    rows = prices.loc[
        (prices["capacity_year"] == capacity_year)
        & (prices["price_type_canonical"] == price_type)
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["price_per_mw_year"])


# --------------------------------------------------------------------------------------
# large partitioned tables
# --------------------------------------------------------------------------------------


def _load_partitioned(
    table: str,
    columns: list[str] | None = None,
    facilities: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    directory = PROCESSED / table
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} not found. Run 'python -m src.clean --only {table}'."
        )

    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet parts in {directory}")

    frames = []
    for part in parts:
        frame = pd.read_parquet(part, columns=columns)
        if facilities is not None:
            frame = frame.loc[frame["facility_code"].isin(facilities)]
        if start is not None:
            frame = frame.loc[frame["trading_date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame["trading_date"] <= pd.Timestamp(end)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns or [])
    out = pd.concat(frames, ignore_index=True)
    out["facility_code"] = out["facility_code"].astype("string")
    return out.sort_values(["facility_code", "interval_start"], ignore_index=True)


def load_scada(
    facilities: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-facility five-minute metered energy.

    ``average_mwh`` is MWh per five-minute interval (``docs/assumptions.md`` D5), so it
    sums directly to MWh and multiplies by twelve to give MW.
    """
    return _load_partitioned(
        "facility_scada",
        columns=["interval_start", "trading_date", "facility_code", "average_mwh"],
        facilities=facilities,
        start=start,
        end=end,
    )


def load_schedule(
    facilities: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-facility five-minute energy and FCESS schedule quantities.

    The FCESS columns are the enablement quantities used for FCESS revenue
    (``docs/assumptions.md`` F7). AEMO leaves them blank rather than zero when a
    facility is not enabled for a service, so they are filled with zero here: a blank
    means no enablement and therefore no payment (F9).
    """
    frame = _load_partitioned(
        "market_schedule",
        columns=[
            "interval_start",
            "trading_date",
            "facility_code",
            "energy_mw",
            *FCESS_SCHEDULE_COLUMNS,
        ],
        facilities=facilities,
        start=start,
        end=end,
    )
    if not frame.empty:
        frame[list(FCESS_SCHEDULE_COLUMNS)] = frame[list(FCESS_SCHEDULE_COLUMNS)].fillna(0.0)
    return frame


# --------------------------------------------------------------------------------------
# ESROI masks
# --------------------------------------------------------------------------------------


def esroi_windows(facility_code: str | None = None) -> pd.DataFrame:
    """Obligation windows applying to a facility, one row per trading day.

    Rows published without a component are market-wide and apply to every storage
    facility holding Capacity Credits; per-component rows override them.
    """
    windows = load_esroi()
    if facility_code is None:
        return windows.loc[windows["market_wide"]].reset_index(drop=True)

    specific = windows.loc[windows["facility_code"] == facility_code]
    market_wide = windows.loc[windows["market_wide"]]
    combined = pd.concat([specific, market_wide], ignore_index=True)
    # Prefer the facility-specific row where both exist for a trading day.
    combined = combined.sort_values(["trading_date", "market_wide"])
    return combined.drop_duplicates(subset=["trading_date"], keep="first").reset_index(drop=True)


def esroi_masks(dates, facility_code: str | None = None) -> np.ndarray:
    """Boolean ``(n_days, 288)`` array marking obligation intervals for each trading day.

    Days with no published window get an all-false row, which switches the obligation
    constraint off for that day rather than guessing a window.
    """
    windows = esroi_windows(facility_code).set_index("trading_date")
    dates = pd.to_datetime(pd.Index(dates))
    mask = np.zeros((len(dates), INTERVALS_PER_DAY_5MIN), dtype=bool)

    for row, date in enumerate(dates):
        if date not in windows.index:
            continue
        window = windows.loc[date]
        # Dispatch interval index within the trading day, which starts at 08:00.
        day_start = date + pd.Timedelta(hours=TRADING_DAY_START_HOUR)
        first = int((window["esroi_start_dt"] - day_start) / pd.Timedelta(minutes=5))
        # The published end is the start of the final obligation Trading Interval, so
        # the window runs to the end of that half hour (docs/assumptions.md D2).
        last = int((window["esroi_end_dt"] - day_start) / pd.Timedelta(minutes=5)) + 6
        first, last = max(first, 0), min(last, INTERVALS_PER_DAY_5MIN)
        if last > first:
            mask[row, first:last] = True

    return mask


#: Column names in the cleaned market requirements table, in SERVICES order.
FCESS_REQUIREMENT_COLUMNS = tuple(f"{service}_requirement" for service in SERVICES)


@lru_cache(maxsize=1)
def load_market_requirements() -> pd.DataFrame:
    """Five-minute FCESS requirement quantities for the whole SWIS, in MW."""
    return pd.read_parquet(_require(PROCESSED / "market_requirements.parquet"))


def requirement_arrays(dates) -> np.ndarray:
    """FCESS requirement as ``(n_days, 5, 288)`` aligned to ``dates``.

    Days with no published requirement get zeros, which switches the market-share cap
    to zero for that day rather than silently allowing unlimited enablement.
    """
    requirements = load_market_requirements()
    wanted = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    subset = requirements.loc[requirements["trading_date"].isin(wanted)].sort_values(
        "interval_start"
    )

    out = np.zeros((len(wanted), len(FCESS_REQUIREMENT_COLUMNS), INTERVALS_PER_DAY_5MIN))
    position = {date: row for row, date in enumerate(wanted)}

    for date, group in subset.groupby("trading_date", observed=True):
        if len(group) != INTERVALS_PER_DAY_5MIN:
            continue
        row = position[date]
        for service, column in enumerate(FCESS_REQUIREMENT_COLUMNS):
            out[row, service] = group[column].to_numpy(dtype=float)

    return np.nan_to_num(out)


def accredited_fleet_mw() -> dict[str, float]:
    """Total accredited quantity across the SWIS for each FCESS service, in MW.

    Taken from the FCESS accreditation register, which is a current snapshot rather
    than a history. The fleet grew over the study period, so using today's totals
    understates an entrant's share in the earlier years; this is stated in
    ``docs/assumptions.md`` F11 and varied as a sensitivity.
    """
    register = load_fcess_accreditation()
    mapping = {
        "contingency_raise": "max_contingency_raise_mw",
        "contingency_lower": "max_contingency_lower_mw",
        "regulation_raise": "max_regulation_raise_mw",
        "regulation_lower": "max_regulation_lower_mw",
        "rocof": "max_rocof_mw",
    }
    return {service: float(register[column].sum()) for service, column in mapping.items()}


def entrant_share(accredited_mw: dict[str, float]) -> dict[str, float]:
    """Share of each FCESS requirement a new entrant of this size would win.

    A new battery adds its accredited quantity to an already accredited fleet, so its
    pro-rata share is ``own / (fleet + own)``. The SWIS is heavily oversupplied with
    accredited capability — about 2,000 MW of contingency raise against a requirement
    averaging under 200 MW — so this share is small, which is exactly why FCESS prices
    fell as far as they did.
    """
    fleet = accredited_fleet_mw()
    return {
        service: (own / (fleet.get(service, 0.0) + own)) if own > 0 else 0.0
        for service, own in accredited_mw.items()
    }


def fcess_caps(dates, accredited_mw: dict[str, float], *, share: dict[str, float] | None = None):
    """Per-interval enablement ceiling from the market requirement and the entrant's share.

    Returns ``(n_days, 5, 288)`` in :data:`src.model.SERVICES` order.
    """
    share = share if share is not None else entrant_share(accredited_mw)
    requirement = requirement_arrays(dates)
    multipliers = np.array([share.get(service, 0.0) for service in SERVICES]).reshape(1, -1, 1)
    return requirement * multipliers


def clear_caches() -> None:
    """Drop cached tables. Useful in notebooks after re-running the cleaning step."""
    for function in (
        _prices_cached,
        load_facilities,
        load_fcess_accreditation,
        load_capacity_credits,
        load_reserve_capacity_prices,
        load_esroi,
        load_market_requirements,
    ):
        function.cache_clear()
