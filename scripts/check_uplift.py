"""Measure how much FCESS Uplift went to storage, and how much to everyone else.

Modelled revenue excludes Uplift because computing it needs offer data that is not
public (docs/assumptions.md E5, F6). Uplift is published per facility, though, so the
size of what is being excluded *for storage specifically* can be measured rather than
assumed. If storage received none, the exclusion costs the analysis nothing and the
lower-bound caveat can be stated precisely instead of vaguely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402
from src.paths import OUTPUTS, PROCESSED  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 220)

    path = PROCESSED / "fcess_uplift.parquet"
    if not path.exists():
        raise SystemExit(
            "run '.venv\\Scripts\\python.exe -m src.clean --only fcess_uplift' first"
        )

    uplift = pd.read_parquet(path)
    uplift["year"] = uplift["trading_date"].dt.year

    storage = set(data.storage_facilities()["facility_code"])
    uplift["is_storage"] = uplift["facility_code"].isin(storage)

    by_year = uplift.groupby(["year", "is_storage"])["uplift_total"].sum().unstack(fill_value=0.0)
    by_year.columns = ["non-storage", "storage"] if True in by_year.columns else ["non-storage"]
    by_year = by_year / 1e6

    print("FCESS Uplift paid, $ million by trading-day year")
    print(by_year.round(3).to_string())

    total = uplift["uplift_total"].sum() / 1e6
    storage_total = uplift.loc[uplift["is_storage"], "uplift_total"].sum() / 1e6
    print(f"\nwhole market  ${total:,.1f}m")
    print(f"storage       ${storage_total:,.3f}m  ({storage_total / total:.3%} of the total)")

    recipients = (
        uplift.groupby("facility_code")["uplift_total"].sum().sort_values(ascending=False) / 1e6
    )
    print("\nten largest recipients, $ million")
    print(recipients.head(10).round(2).to_string())

    if storage_total > 0:
        print("\nstorage facilities receiving uplift, $ million")
        print(
            (
                uplift.loc[uplift["is_storage"]]
                .groupby("facility_code")["uplift_total"]
                .sum()
                .sort_values(ascending=False)
                / 1e6
            ).round(4).to_string()
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(OUTPUTS / "fcess_uplift_by_year.csv")
    recipients.to_frame("uplift_$m").to_csv(OUTPUTS / "fcess_uplift_by_facility.csv")
    print(f"\nwritten to {OUTPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
