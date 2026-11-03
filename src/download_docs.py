"""Download the rule and procedure documents the model's assumptions are sourced from.

Market-rule detail is never taken from memory. Every assumption in
``docs/assumptions.md`` is traced to a clause in one of the documents fetched here,
which are stored under ``data/reference/docs/`` so quotes can be re-checked later.

Usage::

    .venv\\Scripts\\python.exe -m src.download_docs            # fetch anything missing
    .venv\\Scripts\\python.exe -m src.download_docs --list      # show the registry
    .venv\\Scripts\\python.exe -m src.download_docs --force     # re-fetch everything
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import requests

from src.paths import REFERENCE

DOCS_DIR = REFERENCE / "docs"

#: Seconds between requests. The same courtesy delay used for the data files.
REQUEST_DELAY_S = 6.7
TIMEOUT_S = 60

#: Browser-like headers. wa.gov.au and aemo.com.au both reject the default
#: python-requests user agent with HTTP 403.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}


@dataclass(frozen=True)
class ReferenceDoc:
    """A primary-source document cited in ``docs/assumptions.md``."""

    key: str
    title: str
    url: str
    filename: str
    note: str

    @property
    def path(self):
        return DOCS_DIR / self.filename


DOCUMENTS: tuple[ReferenceDoc, ...] = (
    ReferenceDoc(
        key="esm_rules_current",
        title="Electricity System and Market Rules, 1 October 2026",
        url="https://www.wa.gov.au/system/files/2026-09/electricity_system_and_market_rules_-_1_october_2026.pdf",
        filename="esm_rules_2026-10-01.pdf",
        note="Current consolidated Rules. Renamed from WEM Rules on 6 February 2025.",
    ),
    ReferenceDoc(
        key="wem_rules_2023_10_01",
        title="Wholesale Electricity Market Rules, 1 October 2023",
        url="https://www.wa.gov.au/system/files/2023-09/wholesale_electricity_market_rules_-_1_october_2023.pdf",
        filename="wem_rules_2023-10-01.pdf",
        note="Version in force at the start of the study period, i.e. market commencement.",
    ),
    ReferenceDoc(
        key="wem_rules_2025_01_01",
        title="Wholesale Electricity Market Rules, 1 January 2025",
        url="https://www.wa.gov.au/system/files/2024-12/wholesale_electricity_market_rules_1_january_2025.pdf",
        filename="wem_rules_2025-01-01.pdf",
        note="Mid-period version, used to check whether cited clauses moved.",
    ),
    ReferenceDoc(
        key="esroi_procedure",
        title="WEM Procedure: Electric Storage Resource Obligation Intervals",
        url="https://aemo.com.au/-/media/files/electricity/wem/procedures/electric-storage-resource-obligation-intervals-wem-procedure.pdf",
        filename="wem_procedure_esroi.pdf",
        note="How AEMO sets the ESROI windows published in the obligation-intervals dataset.",
    ),
    ReferenceDoc(
        key="settlement_formulation",
        title="WEM Metering, Settlement and Prudential Calculations Formulation (certified)",
        url="https://www.aemo.com.au/-/media/files/electricity/wem/participant_information/guides-and-useful-information/wem-metering-settlement-and-prudential-calculations-formulation-v081-certified.pdf",
        filename="wem_settlement_formulation_v081.pdf",
        note="Independently certified settlement equations, including loss factor application.",
    ),
    # Annual loss factor reports. These are the only published source of per-facility
    # Transmission Loss Factors; the AEMO loss-factors index page rejects automated
    # requests, so the per-year media URLs are listed explicitly.
    ReferenceDoc(
        key="loss_factors_2023_24",
        title="WEM Loss Factor Report 2023/24",
        url="https://www.aemo.com.au/-/media/files/electricity/wem/data/loss-factors/2024/2023-24-loss-factor-report.pdf",
        filename="loss_factor_report_2023-24.pdf",
        note="Transmission and distribution loss factors effective 1 July 2023.",
    ),
    ReferenceDoc(
        key="loss_factors_2024_25",
        title="WEM Loss Factor Report 2024/25",
        url="https://www.aemo.com.au/-/media/files/electricity/wem/data/loss-factors/2025/2024-25-loss-factor-report.pdf",
        filename="loss_factor_report_2024-25.pdf",
        note="Transmission and distribution loss factors effective 1 July 2024.",
    ),
    ReferenceDoc(
        key="loss_factors_2025_26",
        title="WEM Loss Factor Report 2025/26",
        url="https://www.aemo.com.au/-/media/files/electricity/wem/data/loss-factors/2026/2025-26-loss-factor-report.pdf",
        filename="loss_factor_report_2025-26.pdf",
        note="Transmission and distribution loss factors effective 1 July 2025.",
    ),
    ReferenceDoc(
        key="loss_factors_2026_27",
        title="WEM Loss Factor Report 2026/27",
        url="https://www.aemo.com.au/-/media/files/electricity/wem/data/loss-factors/2027/2026-27-loss-factor-report.pdf",
        filename="loss_factor_report_2026-27.pdf",
        note="Transmission and distribution loss factors effective 1 July 2026.",
    ),
    # AEMO Quarterly Energy Dynamics. Used only to reconcile modelled FCESS totals
    # against published market costs, never as a rule source.
    ReferenceDoc(
        key="qed_2024_q4",
        title="AEMO Quarterly Energy Dynamics, Q4 2024",
        url="https://www.aemo.com.au/-/media/files/major-publications/qed/2024/qed-q4-2024.pdf",
        filename="qed_2024_q4.pdf",
        note="Published WEM Essential System Services costs for reconciliation.",
    ),
    ReferenceDoc(
        key="qed_2025_q4",
        title="AEMO Quarterly Energy Dynamics, Q4 2025",
        url="https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q4-2025.pdf",
        filename="qed_2025_q4.pdf",
        note="Published WEM Essential System Services costs for reconciliation.",
    ),
    ReferenceDoc(
        key="qed_2026_q3",
        title="AEMO Quarterly Energy Dynamics, Q3 2026",
        url="https://www.aemo.com.au/-/media/files/major-publications/qed/2026/qed-q3-2026.pdf",
        filename="qed_2026_q3.pdf",
        note="Published WEM Essential System Services costs for reconciliation.",
    ),
    ReferenceDoc(
        key="gencost_2025_26",
        title="CSIRO GenCost 2025-26 Final Report",
        url="https://www.csiro.au/-/media/Energy/GenCost-2025-26-Final/GenCost_2025-26_Final_Report_20260715.pdf",
        filename="gencost_2025-26_final.pdf",
        note="Capital and operating cost estimates for battery storage, used in the economics stage.",
    ),
    ReferenceDoc(
        key="market_design_summary",
        title="WEM Reform Market Design Summary, September 2023",
        url="https://www.aemo.com.au/-/media/files/initiatives/wem-reform-program/wem-reform-market-design-summary.pdf",
        filename="wem_reform_market_design_summary.pdf",
        note="Context only. Never cited as a rule source; used to locate the right clauses.",
    ),
)

BY_KEY = {doc.key: doc for doc in DOCUMENTS}


def fetch(doc: ReferenceDoc, *, force: bool = False) -> bool:
    """Download ``doc`` unless it is already present. Returns True if a request was made."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if doc.path.exists() and not force:
        print(f"  skip      {doc.filename} ({doc.path.stat().st_size / 1e6:.2f} MB)")
        return False

    response = requests.get(doc.url, headers=HEADERS, timeout=TIMEOUT_S)
    response.raise_for_status()
    doc.path.write_bytes(response.content)
    print(f"  retrieved {doc.filename} ({len(response.content) / 1e6:.2f} MB)")
    return True


def fetch_all(*, force: bool = False, keys: list[str] | None = None) -> dict[str, str]:
    """Fetch the registry. Failures are reported and returned rather than raised, so one
    dead link does not block the rest."""
    selected = [BY_KEY[k] for k in keys] if keys else list(DOCUMENTS)
    failures: dict[str, str] = {}

    for index, doc in enumerate(selected):
        print(f"{doc.key}:")
        try:
            requested = fetch(doc, force=force)
        except Exception as exc:  # noqa: BLE001 - a dead link must not stop the run
            print(f"  FAILED    {doc.filename}: {exc}")
            failures[doc.key] = str(exc)
            requested = True
        if requested and index < len(selected) - 1:
            time.sleep(REQUEST_DELAY_S)

    if failures:
        print(f"\n{len(failures)} document(s) could not be retrieved:")
        for key, message in failures.items():
            print(f"  {key}: {message}")
            print(f"    {BY_KEY[key].url}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch documents already present")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--key", action="append", help="fetch only this key (repeatable)")
    args = parser.parse_args()

    if args.list:
        for doc in DOCUMENTS:
            present = "present" if doc.path.exists() else "missing"
            print(f"{doc.key:26s} [{present}] {doc.title}\n  {doc.url}\n  {doc.note}\n")
        return 0

    failures = fetch_all(force=args.force, keys=args.key)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
