"""Build the per-facility loss factor table from AEMO's annual WEM Loss Factor Reports.

Loss factors are not published in the AEMO public data file portal, so they are parsed
out of the annual reports fetched by :mod:`src.download_docs` and written to
``data/reference/loss_factors.csv`` with the source report recorded on every row.

Two facts from the Rules shape this (see ``docs/assumptions.md`` L1 to L4):

* a Loss Factor is the product of a Transmission Loss Factor and a Distribution Loss
  Factor (2.27.3), and transmission-connected Measurement Points take a Distribution
  Loss Factor of 1 (2.27.4(c)). Every SWIS battery is a transmission-connected
  Scheduled Facility, so its Loss Factor is its Transmission Loss Factor.
* the loss factor adjusts the settled **quantity**, not the price (9.5.2).

Each report's Table 4.1 lists a TLF code, a description, and one or more value columns.
The **last** numeric column is always the value that applies for the period the report
covers: in the 2023/24 report that column is "To apply from the New WEM Commencement
Day", which begins on 1 October 2023, exactly the start of the study period.

Usage::

    .venv\\Scripts\\python.exe -m src.loss_factors --build
    .venv\\Scripts\\python.exe -m src.loss_factors --show
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from src.download_docs import DOCS_DIR
from src.paths import REFERENCE

OUT_PATH = REFERENCE / "loss_factors.csv"

#: Report stem -> (financial year label, first trading day the values apply from).
#: The 2023/24 report's final column applies from the New WEM Commencement Day rather
#: than from 1 July, which is why its start date is 1 October 2023.
REPORTS = {
    "loss_factor_report_2023-24": ("2023-24", "2023-10-01"),
    "loss_factor_report_2024-25": ("2024-25", "2024-07-01"),
    "loss_factor_report_2025-26": ("2025-26", "2025-07-01"),
    "loss_factor_report_2026-27": ("2026-27", "2026-07-01"),
}

#: Battery facility code -> TLF code, from the descriptions in the reports' Table 4.1.
#: Where a report predates a dedicated code for a facility, the fallback code is used;
#: these are judgement calls and are recorded as such in the output file.
FACILITY_TLF_CODES: dict[str, tuple[str, ...]] = {
    "COLLIE_ESR1": ("WCBS", "WCPS"),          # "Collie BESS 1"
    "COLLIE_BESS2": ("WCBT", "WCBS", "WCPS"),  # "Neoen Collie Stage 2 BESS"
    "COLLIE_ESR4": ("WCBD", "WCPS", "WCBS"),  # "Collie ESR4"
    "COLLIE_ESR5": ("WCBE", "WCPS", "WCBS"),  # "Collie ESR5"
    "KWINANA_ESR1": ("WKBS",),                # "Kwinana Battery Energy Storage System"
    "KWINANA_ESR2": ("WKBT", "WKBS"),         # "Kwinana BESS 2 (KBESS2)"
    "MERREDIN_ESR1": ("WMER",),               # "Merredin"
    "MUCHEA_ESR1": ("WMUC",),                 # "Muchea"
    "ALINTA_WGP_ESR1": ("WWGU", "WWGP"),      # "Wagerup Battery", else "Wagerup"
}

#: Loss factor used for the hypothetical 100 MW battery in the headline cases. The SWIS
#: battery fleet clusters close to unity, so unity is the neutral central case and
#: sensitivity LF-1 varies it. Recorded in docs/assumptions.md as assumption L4.
DEFAULT_MODEL_LOSS_FACTOR = 1.0

#: Table 4.1 rows look like "WCBS Collie BESS 1 0.9698 0.9719".
_ROW_RE = re.compile(
    r"^(?P<code>[A-Z]{4})\s+(?P<description>.+?)\s+(?P<values>(?:-?\d+\.\d{3,4}\s*)+)$"
)


def parse_report(stem: str) -> pd.DataFrame:
    """Parse the Transmission Loss Factors table out of one report's cached text."""
    path = DOCS_DIR / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.download_docs' then "
            "'python -m src.ruletext --extract'."
        )

    financial_year, applies_from = REPORTS[stem]
    rows = []
    seen: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW_RE.match(line.strip())
        if not match:
            continue
        values = [float(v) for v in match.group("values").split()]
        code = match.group("code")

        # Later sections of the reports repeat codes with extra columns (for example a
        # year-on-year change table). The first occurrence is Table 4.1.
        if code in seen:
            continue
        seen.add(code)

        rows.append(
            {
                "tlf_code": code,
                "description": match.group("description").strip(),
                "transmission_loss_factor": values[-1],
                "financial_year": financial_year,
                "applies_from": applies_from,
                "source_report": f"{stem}.pdf",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no Table 4.1 rows parsed from {path}")
    return frame


def parse_all_reports() -> pd.DataFrame:
    """Every report's loss factor table, concatenated."""
    return pd.concat([parse_report(stem) for stem in REPORTS], ignore_index=True)


def build() -> pd.DataFrame:
    """Build the per-facility loss factor table and write it to ``data/reference``."""
    table = parse_all_reports()
    lookup = table.set_index(["financial_year", "tlf_code"])

    rows = []
    for facility, candidates in FACILITY_TLF_CODES.items():
        for financial_year, applies_from in REPORTS.values():
            chosen = None
            for rank, code in enumerate(candidates):
                if (financial_year, code) in lookup.index:
                    chosen = (code, rank)
                    break

            if chosen is None:
                rows.append(
                    {
                        "facility_code": facility,
                        "financial_year": financial_year,
                        "applies_from": applies_from,
                        "tlf_code": "",
                        "description": "",
                        "loss_factor": 1.0,
                        "exact_match": False,
                        "source_report": "",
                        "note": "no TLF code found for this facility-year; 1.0 assumed",
                    }
                )
                continue

            code, rank = chosen
            record = lookup.loc[(financial_year, code)]
            rows.append(
                {
                    "facility_code": facility,
                    "financial_year": financial_year,
                    "applies_from": applies_from,
                    "tlf_code": code,
                    "description": record["description"],
                    # Distribution Loss Factor is 1 for transmission-connected points
                    # (ESM Rules 2.27.4(c)), so the Loss Factor is the TLF.
                    "loss_factor": float(record["transmission_loss_factor"]),
                    "exact_match": rank == 0,
                    "source_report": record["source_report"],
                    "note": (
                        ""
                        if rank == 0
                        else f"no dedicated code this year; using nearest site code {code}"
                    ),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["facility_code", "financial_year"], ignore_index=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    return frame


def load() -> pd.DataFrame:
    """Read the built loss factor table."""
    if not OUT_PATH.exists():
        return build()
    return pd.read_csv(OUT_PATH)


def loss_factor_for(facility_code: str, trading_date) -> float:
    """Loss factor applying to ``facility_code`` on ``trading_date``.

    Falls back to 1.0 for facilities with no entry, which is the neutral assumption
    recorded as L4 in ``docs/assumptions.md``.
    """
    table = load()
    rows = table.loc[table["facility_code"] == facility_code].copy()
    if rows.empty:
        return 1.0
    rows["applies_from"] = pd.to_datetime(rows["applies_from"])
    applicable = rows.loc[rows["applies_from"] <= pd.Timestamp(trading_date)]
    if applicable.empty:
        return float(rows.sort_values("applies_from").iloc[0]["loss_factor"])
    return float(applicable.sort_values("applies_from").iloc[-1]["loss_factor"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="rebuild the reference table")
    parser.add_argument("--show", action="store_true", help="print the table")
    args = parser.parse_args()

    frame = build() if args.build or not OUT_PATH.exists() else load()
    if args.show or args.build:
        pivot = frame.pivot(index="facility_code", columns="financial_year", values="loss_factor")
        print(pivot.to_string())
        inexact = frame.loc[~frame["exact_match"].astype(bool)]
        if not inexact.empty:
            print("\nfacility-years without a dedicated TLF code:")
            print(inexact[["facility_code", "financial_year", "tlf_code", "note"]].to_string(index=False))
        print(f"\nwritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
