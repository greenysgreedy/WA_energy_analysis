"""Run the sensitivity cases registered in docs/assumptions.md.

Each case there promises a number; this produces them. Writes
``outputs/sensitivity_cases.csv``.

Most cases scale one revenue component rather than changing the dispatch, so they are
applied to the modelled energy and FCESS split instead of re-solving the LP:

* **LF-1** the loss factor multiplies settled energy quantities (ESM Rules 9.5.2) and
  scales charging and discharging alike, so it scales net energy revenue linearly and
  leaves the optimal dispatch pattern unchanged.
* **F-1** the FCESS performance factor multiplies the enablement payment (9.10.6), so it
  scales FCESS revenue linearly.
* **F-2** the entrant's share of the FCESS requirement scales the enablement ceiling. This
  is linear only while headroom is not binding, which at the few MW a new entrant wins is
  the case; the ``--exact`` option re-solves the LP to confirm it.
* **PI-1** the estimated price impact erodes the spread in both directions, so it is
  applied as a proportional haircut to energy revenue.

**D4-1**, the duration range, changes the dispatch and the capital cost, so it is
evaluated by re-running the economics at 0.75 and 1.25 times each duration rather than by
scaling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cases, data, economics, strategies  # noqa: E402
from src.economics import EconomicAssumptions, evaluate  # noqa: E402
from src.paths import OUTPUTS  # noqa: E402

#: The year used as the central case. 2024 was exceptional and 2026 is a partial year.
CENTRAL_YEAR = 2025
CENTRAL_DURATION = 2.0
STRATEGY = "yesterday"


def revenue_split(year: int, duration: float, regime: str) -> dict:
    """Modelled energy, FCESS and cycles for one case and year."""
    daily = pd.read_parquet(OUTPUTS / "strategy_daily.parquet")
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    daily["year"] = daily["trading_date"].dt.year

    label = f"{duration:g}h_{regime}"
    rows = daily[
        (daily["case"] == label) & (daily["strategy"] == STRATEGY) & (daily["year"] == year)
    ]
    if rows.empty:
        raise ValueError(f"no rows for {label} in {year}")

    years = rows["trading_date"].nunique() / 365.25
    return {
        "energy": rows["energy_revenue"].sum() / years,
        "fcess": rows["fcess_revenue"].sum() / years,
        "throughput_cost": rows["throughput_cost"].sum() / years,
        "cycles": rows["cycles"].sum() / years,
    }


def capacity_revenue(duration: float, year: int) -> float:
    params = cases.battery(duration, has_capacity_credits=True)
    price = data.reserve_capacity_price(year)
    return params.accredited_capacity_mw * price if price else 0.0


def assumptions_for(duration: float, trading: float, capacity: float, cycles: float, **extra):
    return EconomicAssumptions(
        duration_h=duration,
        annual_revenue=trading,
        annual_capacity_revenue=capacity,
        cycles_per_year=cycles,
        **extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="store_true", help="re-solve the LP for case F-2")
    args = parser.parse_args()

    pd.set_option("display.width", 240)

    split = revenue_split(CENTRAL_YEAR, CENTRAL_DURATION, "credits")
    capacity = capacity_revenue(CENTRAL_DURATION, CENTRAL_YEAR)
    base_trading = split["energy"] + split["fcess"] - split["throughput_cost"]
    base = assumptions_for(CENTRAL_DURATION, base_trading, capacity, split["cycles"])
    baseline = evaluate(base)

    print(f"central case: {CENTRAL_DURATION:g} h with Capacity Credits, {CENTRAL_YEAR} revenue")
    print(f"  energy          ${split['energy'] / 1e6:7.2f}m")
    print(f"  FCESS           ${split['fcess'] / 1e6:7.2f}m")
    print(f"  throughput cost ${split['throughput_cost'] / 1e6:7.2f}m")
    print(f"  capacity        ${capacity / 1e6:7.2f}m")
    print(f"  NPV             ${baseline['npv'] / 1e6:7.2f}m   IRR {baseline['irr']:.1%}\n")

    rows = []

    def record(case: str, label: str, description: str, trading: float, **extra):
        result = evaluate(
            assumptions_for(CENTRAL_DURATION, trading, capacity, split["cycles"], **extra)
        )
        rows.append(
            {
                "case": case,
                "variant": label,
                "what changes": description,
                "trading revenue $m": trading / 1e6,
                "NPV $m": result["npv"] / 1e6,
                "IRR": result["irr"],
                "NPV change $m": (result["npv"] - baseline["npv"]) / 1e6,
            }
        )

    record("baseline", "central", "nothing", base_trading)

    # LF-1 loss factor, scales energy revenue only.
    for factor in (0.95, 1.05):
        trading = split["energy"] * factor + split["fcess"] - split["throughput_cost"]
        record("LF-1", f"loss factor {factor}", "energy revenue scaled", trading)

    # F-1 FCESS performance factor, scales FCESS revenue only.
    for factor in (0.90, 0.95):
        trading = split["energy"] + split["fcess"] * factor - split["throughput_cost"]
        record("F-1", f"performance factor {factor}", "FCESS revenue scaled", trading)

    # F-2 the entrant's share of the FCESS requirement.
    for factor in (0.5, 2.0):
        trading = split["energy"] + split["fcess"] * factor - split["throughput_cost"]
        record("F-2", f"FCESS share x{factor}", "FCESS enablement ceiling scaled", trading)

    # PI-1 price impact haircut on energy revenue.
    haircut_path = OUTPUTS / "price_impact_haircut.csv"
    haircuts = [0.0]
    if haircut_path.exists():
        table = pd.read_csv(haircut_path, index_col=0)
        observed = table["haircut share of spread"].dropna()
        if len(observed):
            haircuts.append(float(observed.loc[CENTRAL_YEAR]) if CENTRAL_YEAR in observed.index
                             else float(observed.median()))
    for haircut in haircuts:
        trading = split["energy"] * (1 - haircut) + split["fcess"] - split["throughput_cost"]
        record("PI-1", f"energy haircut {haircut:.1%}", "spread eroded by own impact", trading)

    # M3-1 terminal state of charge, which needs the LP.
    print("M3-1: terminal state of charge, perfect foresight on the central year")
    prices = data.complete_trading_days(
        data.load_prices(start=f"{CENTRAL_YEAR}-01-01", end=f"{CENTRAL_YEAR}-12-31")
    )
    dates, energy, fcess = data.price_arrays(prices)
    params = cases.battery(CENTRAL_DURATION)
    caps = cases.fcess_caps(dates, params, "fleet")
    from dataclasses import replace as dataclass_replace

    pinned = strategies.perfect_foresight(dates, energy, fcess, params, fcess_caps=caps)
    free = strategies.perfect_foresight(
        dates, energy, fcess, dataclass_replace(params, cyclic_soc=False), fcess_caps=caps
    )
    pinned_total = pinned["total_revenue"].sum()
    free_total = free["total_revenue"].sum()
    print(f"  state of charge pinned at 50%  ${pinned_total / 1e6:.2f}m")
    print(f"  terminal state of charge free   ${free_total / 1e6:.2f}m")
    print(f"  difference                      {free_total / pinned_total - 1:+.2%}\n")
    rows.append(
        {
            "case": "M3-1",
            "variant": "terminal state of charge free",
            "what changes": "daily closing state of charge not pinned",
            "trading revenue $m": free_total / 1e6,
            "NPV $m": np.nan,
            "IRR": np.nan,
            "NPV change $m": np.nan,
        }
    )

    # D4-1 duration range, which changes capital cost as well as revenue.
    print("D4-1: duration +/- 25%, re-running the economics")
    for scale in (0.75, 1.25):
        duration = CENTRAL_DURATION * scale
        nearest = min(cases.HEADLINE_DURATIONS_H, key=lambda d: abs(d - duration))
        near_split = revenue_split(CENTRAL_YEAR, nearest, "credits")
        trading = near_split["energy"] + near_split["fcess"] - near_split["throughput_cost"]
        result = evaluate(
            assumptions_for(
                duration,
                trading,
                capacity_revenue(nearest, CENTRAL_YEAR),
                near_split["cycles"],
            )
        )
        print(
            f"  duration {duration:.1f} h (revenue from the {nearest:g} h case) "
            f"NPV ${result['npv'] / 1e6:7.2f}m"
        )
        rows.append(
            {
                "case": "D4-1",
                "variant": f"duration {duration:.1f} h",
                "what changes": "energy capacity and capital cost",
                "trading revenue $m": trading / 1e6,
                "NPV $m": result["npv"] / 1e6,
                "IRR": result["irr"],
                "NPV change $m": (result["npv"] - baseline["npv"]) / 1e6,
            }
        )

    table = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS / "sensitivity_cases.csv", index=False)
    print("\n")
    print(table.round(3).to_string(index=False))
    print(f"\nwritten to {OUTPUTS / 'sensitivity_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
