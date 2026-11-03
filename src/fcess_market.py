"""Calibrate how much FCESS enablement a new battery could actually win.

This is the single most consequential modelling choice in the project, because FCESS is
the larger half of a SWIS battery's revenue. Three candidate ceilings were considered:

1. **Accreditation only.** The optimiser enables to the accredited quantity in every
   interval. For a 100 MW battery accredited at 50 MW per service that is a quarter of
   the entire SWIS contingency raise requirement, held continuously. It produces annual
   FCESS revenue an order of magnitude above anything any real battery has earned, so it
   is reported as an upper bound and nothing else.

2. **The published market requirement.** A battery cannot be paid for holding more
   reserve than the market procured — except that it demonstrably can: in 2025 the
   storage fleet's contingency lower enablement averaged 134 % of the published
   requirement, because at a clearing price near zero the dispatch engine enables more
   than the minimum at no cost. So the requirement is a reference point, not a ceiling.

3. **What the operating fleet actually achieved per accredited MW.** For each dispatch
   interval, the total enablement of storage facilities is divided by the accredited
   quantity of the storage facilities that were *operating* in that interval. A new
   entrant is assumed to win the same enablement per accredited MW. This is an
   observation rather than an assumption about the merit order, it is time-varying, and
   it reproduces the competitive pressure that drove FCESS prices from $82/MWh in 2023
   to near zero by 2026.

Option 3 is the central case. Options 1 and 2 are reported alongside it so the reader
can see how much of the answer rests on this choice.

Usage::

    .venv\\Scripts\\python.exe -m src.fcess_market --build
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd

from src import data
from src.model import SERVICES
from src.paths import INTERVALS_PER_DAY_5MIN, PROCESSED

OUT_PATH = PROCESSED / "fcess_fleet.parquet"

#: A facility counts as operating in an interval if it has a market schedule row with any
#: non-zero energy or enablement quantity. Commissioning gaps and outages therefore drop
#: out of the denominator rather than diluting it.
ACCREDITATION_COLUMNS = {
    "contingency_raise": "max_contingency_raise_mw",
    "contingency_lower": "max_contingency_lower_mw",
    "regulation_raise": "max_regulation_raise_mw",
    "regulation_lower": "max_regulation_lower_mw",
    "rocof": "max_rocof_mw",
}


def build() -> pd.DataFrame:
    """Build the per-interval fleet enablement and active accreditation table."""
    storage = data.storage_facilities()
    codes = storage["facility_code"].tolist()
    accreditation = (
        data.load_fcess_accreditation()
        .set_index("facility_code")[list(ACCREDITATION_COLUMNS.values())]
        .reindex(codes)
        .astype(float)
        .fillna(0.0)
    )

    schedule = data.load_schedule(facilities=codes)
    if schedule.empty:
        raise ValueError("no storage rows in the market schedule; clean market_schedule first")

    activity_columns = ["energy_mw", *data.FCESS_SCHEDULE_COLUMNS]
    schedule["operating"] = schedule[activity_columns].abs().sum(axis=1) > 0

    frames = {"interval_start": schedule["interval_start"], "trading_date": schedule["trading_date"]}
    for service in SERVICES:
        accredited = schedule["facility_code"].map(accreditation[ACCREDITATION_COLUMNS[service]])
        frames[f"{service}_enabled_mw"] = schedule[f"{service}_mw"]
        frames[f"{service}_active_accredited_mw"] = accredited.where(schedule["operating"], 0.0)

    per_interval = pd.DataFrame(frames).groupby(["interval_start", "trading_date"], as_index=False).sum()

    # Reindex onto the full five-minute grid. Before the fleet existed there are no
    # storage rows at all in the market schedule, and in the early months there are
    # intervals with none; without this those intervals would be missing rather than
    # zero, and every day containing one would be silently dropped downstream.
    grid = (
        data.load_prices()[["interval_start", "trading_date"]]
        .drop_duplicates()
        .sort_values("interval_start")
    )
    per_interval = grid.merge(per_interval, on=["interval_start", "trading_date"], how="left")
    value_columns = [c for c in per_interval.columns if c.endswith("_mw")]
    per_interval[value_columns] = per_interval[value_columns].fillna(0.0)

    for service in SERVICES:
        active = per_interval[f"{service}_active_accredited_mw"]
        enabled = per_interval[f"{service}_enabled_mw"]
        per_interval[f"{service}_utilisation"] = np.where(active > 0, enabled / active, 0.0)

    per_interval = per_interval.sort_values("interval_start", ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    per_interval.to_parquet(OUT_PATH, index=False, compression="zstd")
    return per_interval


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    if not OUT_PATH.exists():
        return build()
    return pd.read_parquet(OUT_PATH)


def utilisation_arrays(dates) -> np.ndarray:
    """Fleet enablement per accredited MW as ``(n_days, 5, 288)``, aligned to ``dates``."""
    fleet = load()
    wanted = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    subset = fleet.loc[fleet["trading_date"].isin(wanted)].sort_values("interval_start")

    out = np.zeros((len(wanted), len(SERVICES), INTERVALS_PER_DAY_5MIN))
    position = {date: row for row, date in enumerate(wanted)}

    for date, group in subset.groupby("trading_date", observed=True):
        if len(group) != INTERVALS_PER_DAY_5MIN:
            continue
        row = position[date]
        for index, service in enumerate(SERVICES):
            out[row, index] = group[f"{service}_utilisation"].to_numpy(dtype=float)

    return np.nan_to_num(out)


def entrant_caps(dates, accredited_mw: dict[str, float], *, scale: float = 1.0) -> np.ndarray:
    """Enablement ceiling for a new entrant, ``(n_days, 5, 288)``.

    ``scale`` multiplies the observed fleet utilisation, for the sensitivity that asks
    what happens if the entrant competes better or worse than the fleet average.
    """
    utilisation = utilisation_arrays(dates) * scale
    own = np.array([accredited_mw.get(service, 0.0) for service in SERVICES]).reshape(1, -1, 1)
    return np.minimum(utilisation * own, own)


MARKET_COST_PATH = PROCESSED / "fcess_market_cost.parquet"


def build_market_cost() -> pd.DataFrame:
    """Whole-market FCESS enablement cost per trading day, from schedules and prices.

    This is the strongest validation available in the project, because it does not
    depend on anything about storage: it multiplies **every** facility's enablement
    quantity by the clearing price and sums. AEMO publishes the same total each quarter,
    so the two can be compared directly, and agreement means the enablement and price
    series are both being read correctly.
    """
    from src.paths import DT_HOURS

    prices = data.load_prices()[["interval_start", "trading_date", *data.FCESS_PRICE_COLUMNS]]
    parts = sorted((PROCESSED / "market_schedule").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("clean market_schedule first")

    frames = []
    for part in parts:
        schedule = pd.read_parquet(
            part, columns=["interval_start", *data.FCESS_SCHEDULE_COLUMNS]
        )
        schedule[list(data.FCESS_SCHEDULE_COLUMNS)] = schedule[
            list(data.FCESS_SCHEDULE_COLUMNS)
        ].fillna(0.0)
        totals = schedule.groupby("interval_start", as_index=False).sum()
        frames.append(totals)

    enabled = pd.concat(frames, ignore_index=True)
    merged = enabled.merge(prices, on="interval_start", how="inner")

    for service in SERVICES:
        merged[f"{service}_cost"] = (
            merged[f"{service}_mw"] * merged[f"{service}_price"].fillna(0.0) * DT_HOURS
        )

    cost_columns = [f"{service}_cost" for service in SERVICES]
    daily = merged.groupby("trading_date", observed=True)[cost_columns].sum()
    daily["total_cost"] = daily.sum(axis=1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_parquet(MARKET_COST_PATH, index=False, compression="zstd")
    return daily


@lru_cache(maxsize=1)
def load_market_cost() -> pd.DataFrame:
    if not MARKET_COST_PATH.exists():
        return build_market_cost()
    return pd.read_parquet(MARKET_COST_PATH).set_index("trading_date")


def summary() -> pd.DataFrame:
    """Fleet enablement, active accreditation and utilisation by year."""
    fleet = load().copy()
    fleet["year"] = fleet["trading_date"].dt.year
    rows = []
    for year, group in fleet.groupby("year"):
        row = {"year": year}
        for service in SERVICES:
            row[f"{service} enabled MW"] = group[f"{service}_enabled_mw"].mean()
            row[f"{service} active accredited MW"] = group[f"{service}_active_accredited_mw"].mean()
            row[f"{service} utilisation"] = group[f"{service}_utilisation"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="rebuild the fleet table")
    args = parser.parse_args()

    frame = build() if args.build or not OUT_PATH.exists() else load()
    print(f"{len(frame):,} intervals written to {OUT_PATH}\n")

    if args.build or not MARKET_COST_PATH.exists():
        cost = build_market_cost()
        print(f"{len(cost):,} trading days written to {MARKET_COST_PATH}\n")

    table = summary()
    utilisation = table[[c for c in table.columns if c.endswith("utilisation")]]
    enabled = table[[c for c in table.columns if c.endswith("enabled MW")]]
    print("fleet enablement, mean MW:")
    print(enabled.round(2).to_string())
    print("\nenablement per accredited MW of operating storage:")
    print(utilisation.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
