"""Test how far the negative 2026 result depends on the throughput cost.

docs/results.md claims the qualitative finding — that there is almost nothing left to
capture in 2026 — is robust, but that the *sign* of realisable revenue is not. This
script checks that rather than asserting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cases, data, strategies  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--durations",
        default=",".join(f"{d:g}" for d in cases.HEADLINE_DURATIONS_H),
        help="comma-separated durations in hours",
    )
    parser.add_argument(
        "--costs", default="0,2,5,10,20", help="comma-separated throughput costs in $/MWh"
    )
    args = parser.parse_args()

    durations = [float(d) for d in args.durations.split(",")]
    costs = [float(c) for c in args.costs.split(",")]

    pd.set_option("display.width", 220)

    prices = data.complete_trading_days(data.load_prices(start="2026-01-01"))
    dates, energy, fcess = data.price_arrays(prices)
    print(f"{len(dates)} complete trading days in 2026\n")

    rows = []
    for duration in durations:
        for throughput_cost in costs:
            params = cases.battery(duration, c_thru=throughput_cost)
            caps = cases.fcess_caps(dates, params, "fleet")

            bound = strategies.perfect_foresight(
                dates, energy, fcess, params, fcess_caps=caps
            )
            outcome = strategies.rolling_horizon(
                dates,
                energy,
                fcess,
                params,
                strategies.frozen_for(2024, "yesterday"),
                fcess_caps=caps,
            )
            rows.append(
                {
                    "duration_h": duration,
                    "c_thru $/MWh": throughput_cost,
                    "perfect foresight $m": bound["total_revenue"].sum() / 1e6,
                    "realisable $m": outcome.total_revenue / 1e6,
                    "realisable energy $m": outcome.daily["energy_revenue"].sum() / 1e6,
                    "realisable FCESS $m": outcome.daily["fcess_revenue"].sum() / 1e6,
                    "cycles": outcome.daily["cycles"].sum(),
                }
            )
            print(f"  {duration:g}h c_thru={throughput_cost:5.1f} -> "
                  f"${rows[-1]['realisable $m']:6.2f}m", flush=True)

    table = pd.DataFrame(rows)
    print("\n")
    print(
        table.pivot_table(
            index="duration_h", columns="c_thru $/MWh", values="realisable $m", observed=True
        ).round(2).to_string()
    )
    print("\nrealisable revenue, $ million, 2026 to date, by throughput cost")
    print("\nIf the sign flips between columns, the loss is a throughput-cost artefact;")
    print("if every column is small, the finding that little remains to capture holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
