"""Print the economics grid and ranges, for quoting in the write-up."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import OUTPUTS  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    grid = pd.read_csv(OUTPUTS / "economics_grid.csv")
    columns = [
        "duration_h",
        "revenue_year",
        "trading $/MW-yr",
        "capacity $/MW-yr",
        "total $/MW-yr",
        "breakeven $/MW-yr",
        "NPV $m",
        "IRR",
        "cycles/yr",
        "augmentations",
    ]

    for regime in ("credits", "merchant"):
        subset = grid.loc[grid["regime"] == regime, columns]
        print(f"\n{regime.upper()}")
        print(subset.sort_values(["duration_h", "revenue_year"]).round(3).to_string(index=False))

    ranges_path = OUTPUTS / "economics_ranges.csv"
    if ranges_path.exists():
        print("\nRANGES across revenue years and a 5% to 10% real WACC")
        print(pd.read_csv(ranges_path).round(3).to_string(index=False))

    sensitivity_path = OUTPUTS / "sensitivity_cases.csv"
    if sensitivity_path.exists():
        print("\nREGISTERED SENSITIVITY CASES")
        print(pd.read_csv(sensitivity_path).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
