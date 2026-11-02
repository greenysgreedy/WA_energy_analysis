"""Run perfect foresight and every rolling-horizon strategy over the whole period.

Writes ``outputs/strategy_daily.parquet`` (one row per case, strategy and trading day)
and ``outputs/strategy_summary.csv``. Combinations are independent, so they are run in
parallel across processes.

Usage::

    .venv\\Scripts\\python.exe scripts\\run_strategies.py
    .venv\\Scripts\\python.exe scripts\\run_strategies.py --start 2024-01-01 --end 2024-12-31
    .venv\\Scripts\\python.exe scripts\\run_strategies.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cases, data, strategies  # noqa: E402
from src.paths import OUTPUTS  # noqa: E402


def _run_one(task: dict) -> pd.DataFrame:
    """Run one (case, strategy) combination. Executed in a worker process."""
    label = task["label"]
    strategy = task["strategy"]
    params = task["params"]
    dates = task["dates"]
    energy = task["energy"]
    fcess = task["fcess"]
    caps = task["caps"]
    masks = task["masks"]

    if strategy == "perfect_foresight":
        daily = strategies.perfect_foresight(
            dates, energy, fcess, params, esroi_masks=masks, fcess_caps=caps
        )
        breaches, solves = 0, len(dates)
    else:
        outcome = strategies.rolling_horizon(
            dates,
            energy,
            fcess,
            params,
            strategies.frozen_for(task["design_year"], strategy),
            esroi_masks=masks,
            fcess_caps=caps,
        )
        daily = outcome.daily
        breaches, solves = outcome.obligation_breaches, outcome.solves

    daily = daily.reset_index()
    daily["case"] = label
    daily["strategy"] = strategy
    daily["obligation_breaches"] = breaches
    daily["solves"] = solves
    return daily


def build_tasks(start: str | None, end: str | None, strategy_names: list[str]) -> list[dict]:
    prices = data.complete_trading_days(data.load_prices(start=start, end=end))
    dates, energy, fcess = data.price_arrays(prices)
    masks = data.esroi_masks(dates)

    print(f"{len(dates):,} complete trading days: {pd.Timestamp(dates[0]).date()} to "
          f"{pd.Timestamp(dates[-1]).date()}")

    tasks = []
    for label, params in cases.headline_cases().items():
        caps = cases.fcess_caps(dates, params, "fleet")
        for strategy in strategy_names:
            tasks.append(
                {
                    "label": label,
                    "strategy": strategy,
                    "params": params,
                    "dates": dates,
                    "energy": energy,
                    "fcess": fcess,
                    "caps": caps,
                    "masks": masks if params.has_capacity_credits else None,
                    "design_year": strategies.DESIGN_YEAR,
                }
            )
    return tasks


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    """Annual totals and capture rates per case and strategy."""
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["trading_date"]).dt.year

    grouped = (
        frame.groupby(["case", "strategy", "year"], observed=True)
        .agg(
            days=("trading_date", "nunique"),
            total_revenue=("total_revenue", "sum"),
            energy_revenue=("energy_revenue", "sum"),
            fcess_revenue=("fcess_revenue", "sum"),
            throughput_cost=("throughput_cost", "sum"),
            discharge_mwh=("discharge_mwh", "sum"),
            cycles=("cycles", "sum"),
            obligation_breaches=("obligation_breaches", "max"),
        )
        .reset_index()
    )

    bound = grouped[grouped["strategy"] == "perfect_foresight"].set_index(["case", "year"])[
        "total_revenue"
    ]
    grouped["perfect_foresight"] = grouped.set_index(["case", "year"]).index.map(bound)
    grouped["capture_rate"] = grouped["total_revenue"] / grouped["perfect_foresight"]
    grouped["revenue_per_mw_year"] = grouped["total_revenue"] / cases.HEADLINE_MW
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument(
        "--strategies",
        default=",".join(("perfect_foresight",) + strategies.NAIVE_FORECASTS),
        help="comma-separated strategy names",
    )
    args = parser.parse_args()

    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tasks = build_tasks(args.start, args.end, strategy_names)
    print(f"{len(tasks)} combinations on {args.workers} worker(s)\n")

    started = time.perf_counter()
    frames = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            frame = future.result()
            frames.append(frame)
            elapsed = time.perf_counter() - started
            total = frame["total_revenue"].sum() / 1e6
            print(
                f"[{done:2d}/{len(tasks)}] {task['label']:14s} {task['strategy']:18s} "
                f"${total:7.2f}m  ({elapsed / 60:5.1f} min elapsed)",
                flush=True,
            )

    daily = pd.concat(frames, ignore_index=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUTS / "strategy_daily.parquet", index=False, compression="zstd")

    summary = summarise(daily)
    summary.to_csv(OUTPUTS / "strategy_summary.csv", index=False)

    print(f"\nwrote {len(daily):,} daily rows and {len(summary):,} summary rows to {OUTPUTS}")
    print(f"total time {(time.perf_counter() - started) / 60:.1f} min\n")

    view = summary.pivot_table(
        index=["case", "year"], columns="strategy", values="total_revenue", observed=True
    )
    print((view / 1e6).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
