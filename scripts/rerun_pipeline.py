"""Re-run every compute stage in order, after a change that invalidates the cleaned data.

Stages are run as subprocesses so a failure in one is visible and stops the chain.

Usage::

    .venv\\Scripts\\python.exe scripts\\rerun_pipeline.py
    .venv\\Scripts\\python.exe scripts\\rerun_pipeline.py --from strategies
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES: list[tuple[str, list[str]]] = [
    ("clean_prices", ["-m", "src.clean", "--only", "prices_5min", "--force"]),
    ("clean_schedule", ["-m", "src.clean", "--only", "market_schedule", "--force"]),
    ("clean_uplift", ["-m", "src.clean", "--only", "fcess_uplift", "--force"]),
    ("fcess_market", ["-m", "src.fcess_market", "--build"]),
    ("service_mapping_check", ["scripts/check_service_mapping.py"]),
    ("uplift_check", ["scripts/check_uplift.py"]),
    ("strategies", ["scripts/run_strategies.py", "--workers", "6"]),
    ("backcast", ["scripts/run_backcast.py"]),
    ("reconcile", ["-m", "src.reconcile"]),
    ("sensitivities", ["scripts/run_sensitivities.py"]),
]

NOTEBOOKS = [
    "01_download_and_clean.ipynb",
    "02_explore_prices.ipynb",
    "03_dispatch_model.ipynb",
    "04_strategies_and_capture.ipynb",
    "05_backcast_validation.ipynb",
    "06_price_impact.ipynb",
    "07_economics.ipynb",
]


def run(name: str, arguments: list[str]) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT)
    elapsed = (time.perf_counter() - started) / 60
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")
    print(f"\n{name} finished in {elapsed:.1f} min", flush=True)


def run_notebooks() -> None:
    for notebook in NOTEBOOKS:
        print(f"\n{'=' * 70}\nnotebook {notebook}\n{'=' * 70}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=3600",
                notebook,
            ],
            cwd=ROOT / "notebooks",
            capture_output=True,
            text=True,
        )
        elapsed = (time.perf_counter() - started) / 60
        if result.returncode != 0:
            print(result.stderr[-3000:])
            raise SystemExit(f"{notebook} failed")
        print(f"{notebook} finished in {elapsed:.1f} min", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default=None, help="first stage to run")
    parser.add_argument("--skip-notebooks", action="store_true")
    args = parser.parse_args()

    names = [name for name, _ in STAGES]
    start_index = names.index(args.start) if args.start else 0

    overall = time.perf_counter()
    for name, arguments in STAGES[start_index:]:
        run(name, arguments)

    if not args.skip_notebooks:
        run_notebooks()

    print(f"\npipeline complete in {(time.perf_counter() - overall) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
