"""Repository directory layout.

Every module resolves paths through here so notebooks and tests agree regardless of
the working directory they are launched from.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"

DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"

# Analysis window. The WEM reform commenced on 1 October 2023, so the first complete
# post-reform trading day starts at 08:00 AWST on 1 October 2023.
REFORM_START = "2023-10-01"

#: Length of a trading day in the WEM, which runs 08:00 to 08:00 AWST.
TRADING_DAY_START_HOUR = 8

#: Dispatch interval length in hours. The real-time market clears every five minutes.
DT_HOURS = 5.0 / 60.0

#: Intervals per trading day at five-minute and thirty-minute granularity.
INTERVALS_PER_DAY_5MIN = 288
INTERVALS_PER_DAY_30MIN = 48


def ensure_dirs() -> None:
    """Create the directories the pipeline writes to."""
    for path in (RAW, PROCESSED, REFERENCE, DOCS, OUTPUTS):
        path.mkdir(parents=True, exist_ok=True)


def raw_dir(dataset: str) -> Path:
    """Directory holding untouched AEMO files for ``dataset``."""
    return RAW / dataset


def processed_path(name: str) -> Path:
    """Path of a cleaned parquet table."""
    return PROCESSED / f"{name}.parquet"


def output_path(name: str) -> Path:
    """Path of a generated result table or figure."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / name
