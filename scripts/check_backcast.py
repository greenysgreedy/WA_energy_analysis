"""Spot-check the revenue backcast on one facility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backcast, data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facility", nargs="?", default="KWINANA_ESR1")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    register = data.storage_facilities().set_index("facility_code")
    registered_mw = float(register.loc[args.facility, "registered_mw"])

    result = backcast.rebuild_facility(args.facility, registered_mw=registered_mw)

    print(f"{args.facility}: registered {registered_mw:.0f} MW")
    print(f"  usable energy estimated from SCADA   {result.usable_mwh_estimate:,.1f} MWh")
    print(f"  implied duration                     {result.duration_estimate_h:.2f} h")
    print(f"  peak output observed                 {result.daily['peak_mw'].max():,.1f} MW")
    print(f"  first full operating month           {result.first_full_month}")

    implied = backcast.duration_from_capacity_credits(args.facility, "2024-25")
    if implied is not None:
        print(f"  usable energy implied by credits     {implied:,.1f} MWh")

    window = result.restricted()
    print(f"\n{len(window):,} trading days from the first full operating month\n")
    annual = window[
        ["energy_revenue", "fcess_revenue", "total_revenue", "discharge_mwh", "charge_mwh"]
    ].resample("YS").sum()
    print(annual.round(0).to_string())

    print("\ncapacity revenue:")
    print(backcast.capacity_revenue(args.facility).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
