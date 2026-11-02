"""Download AEMO WEM public data files into ``data/raw/<dataset>/``.

Only the public data file portal is used; no developer APIs. Files are saved byte for
byte as published so the raw directory stays an auditable copy of the source.

Download etiquette, per ``.claude/skills/aemo-wem-data``:

* about 6.7 seconds between requests
* skip files already present, **except** the file covering the current period, which
  AEMO is still appending to
* ``raise_for_status()`` on every response, 30 second timeout

Usage::

    .venv\\Scripts\\python.exe -m src.download_data                      # everything needed
    .venv\\Scripts\\python.exe -m src.download_data --dataset market-schedule-csv
    .venv\\Scripts\\python.exe -m src.download_data --list
    .venv\\Scripts\\python.exe -m src.download_data --dry-run
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests

from src.paths import RAW, REFORM_START

BASE_URL = "https://data.wa.aemo.com.au/public/public-data/datafiles"

REQUEST_DELAY_S = 6.7
TIMEOUT_S = 30

#: First post-reform year. Yearly and monthly files are grouped by trading day, so the
#: 2023 files contain the first post-reform trading days from 1 October 2023.
FIRST_YEAR = int(REFORM_START[:4])
FIRST_MONTH = (FIRST_YEAR, int(REFORM_START[5:7]))


@dataclass(frozen=True)
class Dataset:
    """A dataset on the AEMO public data file portal.

    Attributes
    ----------
    name
        Portal folder name, which is also the ``data/raw`` subdirectory name.
    grain
        ``"monthly"``, ``"yearly"`` or ``"static"``. Drives both the post-reform filter
        and which file counts as the current period.
    purpose
        Why the analysis needs it. Shown by ``--list``.
    key_columns
        Columns the pipeline actually reads, as a cross-check when cleaning.
    """

    name: str
    grain: str
    purpose: str
    key_columns: tuple[str, ...] = field(default=())

    @property
    def directory(self) -> Path:
        return RAW / self.name


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="market-clearing-prices-csv",
        grain="yearly",
        purpose="5-minute energy and FCESS clearing prices. The revenue price series.",
        key_columns=(
            "Dispatch Interval",
            "Energy Clearing Price",
            "Contingency Raise Clearing Price",
            "Contingency Lower Clearing Price",
            "Regulation Raise Clearing Price",
            "Regulation Lower Clearing Price",
            "RoCoF Clearing Price",
        ),
    ),
    Dataset(
        name="reference-trading-price-csv",
        grain="yearly",
        purpose="30-minute Reference Trading Price. Used only if settlement uses it.",
        key_columns=("Trading Interval", "Reference Trading Price"),
    ),
    Dataset(
        name="market-schedule-csv",
        grain="monthly",
        purpose=(
            "Per-facility energy and FCESS schedule quantities every 5 minutes. "
            "The only public source of FCESS enablement by facility."
        ),
        key_columns=(
            "Dispatch Interval",
            "Facility Code",
            "Energy Schedule",
            "Contingency Raise Schedule",
            "Contingency Lower Schedule",
            "Regulation Raise Schedule",
            "Regulation Lower Schedule",
            "RoCoF Schedule",
        ),
    ),
    Dataset(
        name="market-requirements-csv",
        grain="yearly",
        purpose="FCESS requirement quantities per interval. Context for enablement scarcity.",
    ),
    Dataset(
        name="facility-scada-csv",
        grain="monthly",
        purpose="Per-facility metered output every 5 minutes. Drives the revenue backcast.",
        key_columns=("Dispatch Interval", "Facility Code", "Average MWh"),
    ),
    Dataset(
        name="electric-storage-resource-obligation-intervals",
        grain="yearly",
        purpose="ESROI windows a storage facility with Capacity Credits must be available in.",
        key_columns=(
            "Capacity Year",
            "Trading Date",
            "Component",
            "ESROI Trading Interval Start",
            "ESROI Trading Interval End",
        ),
    ),
    Dataset(
        name="post-facilities",
        grain="static",
        purpose="Post-reform facility standing data, including registered MW.",
        key_columns=("Facility Code", "Facility Class", "System Size (MW)"),
    ),
    Dataset(
        name="post-participants",
        grain="static",
        purpose="Post-reform participant register. Maps facilities to owners.",
    ),
    Dataset(
        name="fcess",
        grain="static",
        purpose="FCESS accreditation standing data: maximum accredited quantities per facility.",
    ),
    Dataset(
        name="capacity-credits-since-market-start",
        grain="static",
        purpose="Capacity Credits assigned per facility per capacity year. Wide format.",
    ),
    Dataset(
        name="reserve-capacity-prices",
        grain="static",
        purpose="Reserve Capacity Price per capacity year. Converts credits to revenue.",
    ),
    Dataset(
        name="fcess-uplift-payments",
        grain="static",
        purpose="FCESS uplift payments. Checks whether enablement revenue is the whole story.",
    ),
    Dataset(
        name="outages",
        grain="static",
        purpose="Facility outages. Explains gaps between actual and modelled revenue.",
    ),
)

BY_NAME = {dataset.name: dataset for dataset in DATASETS}

HEADERS = {"User-Agent": "WA_energy_analysis research pipeline (contact via repository)"}

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_MONTH_RE = re.compile(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])(?!\d)")


def list_files(dataset: str) -> list[str]:
    """Return absolute URLs of the CSV files on a dataset's index page."""
    folder = f"{BASE_URL}/{dataset}/"
    response = requests.get(folder, headers=HEADERS, timeout=TIMEOUT_S)
    response.raise_for_status()
    links = re.findall(r'href="([^"]+)"', response.text, flags=re.IGNORECASE)
    return sorted({urljoin(folder, link) for link in links if link.lower().endswith(".csv")})


def file_period(filename: str) -> tuple[int, int] | tuple[int] | None:
    """Period a filename covers: ``(year, month)``, ``(year,)`` or ``None`` for static files."""
    month = _MONTH_RE.search(filename)
    if month:
        return int(month.group(1)), int(month.group(2))
    year = _YEAR_RE.search(filename)
    if year:
        return (int(year.group(1)),)
    return None


def is_post_reform(filename: str) -> bool:
    """Whether a file can contain post-reform trading days.

    Files are grouped by trading day, so the file covering September 2023 cannot contain
    any post-reform day, while the October 2023 file is the first that can.
    """
    period = file_period(filename)
    if period is None:
        return True
    if len(period) == 1:
        return period[0] >= FIRST_YEAR
    return period >= FIRST_MONTH


def is_current_period(filename: str, *, today: date | None = None) -> bool:
    """Whether a file covers the period AEMO is still appending to."""
    today = today or date.today()
    period = file_period(filename)
    if period is None:
        return True  # static files are republished in place
    if len(period) == 1:
        return period[0] == today.year
    return period == (today.year, today.month)


def download(url: str, dataset: str, *, force: bool = False, dry_run: bool = False) -> bool:
    """Download one file. Returns True if a request was made."""
    name = url.rsplit("/", 1)[-1]
    out = BY_NAME[dataset].directory / name if dataset in BY_NAME else RAW / dataset / name

    if out.exists() and not force:
        print(f"    skip      {name}")
        return False

    reason = "refresh" if out.exists() else "new"
    if dry_run:
        print(f"    would get {name} ({reason})")
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
    response.raise_for_status()
    out.write_bytes(response.content)
    print(f"    retrieved {name} ({len(response.content) / 1e6:.1f} MB, {reason})")
    return True


def download_dataset(
    dataset: str,
    *,
    all_periods: bool = False,
    dry_run: bool = False,
    today: date | None = None,
) -> int:
    """Download a dataset's post-reform files. Returns the number of requests made."""
    print(f"{dataset}:")
    urls = list_files(dataset)
    if not all_periods:
        urls = [u for u in urls if is_post_reform(u.rsplit("/", 1)[-1])]

    if not urls:
        print("    no post-reform files on the index page")
        return 0

    requests_made = 1  # the index page itself
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        time.sleep(0 if dry_run else REQUEST_DELAY_S)
        if download(url, dataset, force=is_current_period(name, today=today), dry_run=dry_run):
            requests_made += 1
    return requests_made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        help="portal folder name; repeatable. Defaults to every dataset in the registry.",
    )
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--dry-run", action="store_true", help="report what would be fetched")
    parser.add_argument(
        "--all-periods",
        action="store_true",
        help="include pre-reform files, which the analysis otherwise ignores",
    )
    args = parser.parse_args()

    if args.list:
        for dataset in DATASETS:
            present = len(list(dataset.directory.glob("*.csv"))) if dataset.directory.exists() else 0
            print(f"{dataset.name:48s} {dataset.grain:8s} {present:3d} local file(s)")
            print(f"    {dataset.purpose}")
        return 0

    names = args.dataset or [d.name for d in DATASETS]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    total = 0
    for name in names:
        total += download_dataset(name, all_periods=args.all_periods, dry_run=args.dry_run)
    print(f"\n{total} request(s) made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
