"""Decide whether SCADA ``Average MWh`` is an average MW or an MWh per five minutes.

Getting this wrong scales every backcast revenue figure by twelve, so it is settled
empirically against registered capacity before any revenue is computed. The verdict is
recorded in ``docs/assumptions.md`` as D5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import diagnose_scada_unit  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    table = diagnose_scada_unit(sample_months=6, top_n=20)
    columns = [
        "registered_mw",
        "observed_peak",
        "ratio_to_mw",
        "ratio_to_mwh_per_interval",
        "verdict",
    ]
    print(table[columns].to_string(float_format=lambda v: f"{v:10.4f}"))

    verdicts = table["verdict"].value_counts()
    print("\nverdict by facility count:")
    print(verdicts.to_string())

    winner = verdicts.index[0]
    share = verdicts.iloc[0] / verdicts.sum()
    print(f"\nconclusion: {winner} ({share:.0%} of the {verdicts.sum()} largest facilities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
