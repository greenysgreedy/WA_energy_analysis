"""Check that the FCESS schedule and price columns line up with the market requirement.

Whole-market enablement in a service should track that service's requirement. If the
contingency raise column tracks the contingency lower requirement instead, the columns
are crossed somewhere and every per-service revenue figure is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402
from src.model import SERVICES  # noqa: E402
from src.paths import PROCESSED  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    parts = sorted((PROCESSED / "market_schedule").glob("*.parquet"))
    frames = []
    for part in parts:
        schedule = pd.read_parquet(part, columns=["interval_start", *data.FCESS_SCHEDULE_COLUMNS])
        schedule[list(data.FCESS_SCHEDULE_COLUMNS)] = schedule[
            list(data.FCESS_SCHEDULE_COLUMNS)
        ].fillna(0.0)
        frames.append(schedule.groupby("interval_start", as_index=False).sum())
    enabled = pd.concat(frames, ignore_index=True)

    requirements = data.load_market_requirements()
    prices = data.load_prices()

    merged = enabled.merge(
        requirements[["interval_start", "trading_date", *data.FCESS_REQUIREMENT_COLUMNS]],
        on="interval_start",
    ).merge(prices[["interval_start", *data.FCESS_PRICE_COLUMNS]], on="interval_start")
    merged["year"] = merged["trading_date"].dt.year

    rows = []
    for service in SERVICES:
        for year, group in merged.groupby("year"):
            rows.append(
                {
                    "service": service,
                    "year": year,
                    "enabled MW": group[f"{service}_mw"].mean(),
                    "requirement MW": group[f"{service}_requirement"].mean(),
                    "enabled / required": group[f"{service}_mw"].mean()
                    / group[f"{service}_requirement"].mean(),
                    "mean price": group[f"{service}_price"].mean(),
                }
            )

    table = pd.DataFrame(rows)
    print("whole-market enablement against the published requirement, by service and year")
    print(
        table.pivot_table(
            index="service", columns="year", values="enabled / required", observed=True
        ).round(3).to_string()
    )
    print("\nmean enabled MW")
    print(
        table.pivot_table(
            index="service", columns="year", values="enabled MW", observed=True
        ).round(1).to_string()
    )
    print("\nmean requirement MW")
    print(
        table.pivot_table(
            index="service", columns="year", values="requirement MW", observed=True
        ).round(1).to_string()
    )

    print("\ncross-correlation of enablement against each requirement, 2024 to 2025")
    window = merged[merged["year"].isin([2024, 2025])]
    correlations = pd.DataFrame(
        {
            service: {
                requirement: window[f"{service}_mw"].corr(window[f"{requirement}_requirement"])
                for requirement in SERVICES
            }
            for service in SERVICES
        }
    )
    correlations.index.name = "requirement"
    correlations.columns.name = "enablement"
    print(correlations.round(3).to_string())

    # After the relabelling applied in src/clean.py, each service's enablement must line
    # up with its own requirement. If this fails, either the source files changed or the
    # relabelling was removed, and every per-service figure in the project is suspect.
    #
    # The test is on levels, not on correlation. Correlation cannot separate regulation
    # raise from regulation lower, because the two requirements are near-identical series
    # (87.0 MW against 87.0 MW), so the ranking between them is arbitrary and means
    # nothing. Enablement must at least meet its own requirement and must not wildly
    # exceed it, which is what caught the transposed contingency pair: before the fix the
    # ratios were 0.34 and 3.99.
    failures = []
    ratios = table.set_index(["service", "year"])["enabled / required"]
    for service in SERVICES:
        for year in (2024, 2025):
            ratio = ratios.get((service, year))
            if ratio is None:
                continue
            if not 0.95 <= ratio <= 2.5:
                failures.append(
                    f"{service} {year}: enablement is {ratio:.2f} times its own requirement"
                )

    # Where two services have materially different requirement levels, enablement must
    # also track the right one. This is the check that distinguishes the contingency pair.
    levels = table.set_index(["service", "year"])["requirement MW"]
    for service in SERVICES:
        for other in SERVICES:
            if other == service:
                continue
            own, alternative = levels.get((service, 2024)), levels.get((other, 2024))
            if not own or not alternative or abs(own / alternative - 1.0) < 0.25:
                continue  # too similar to tell apart; correlation would be meaningless
            if correlations.loc[other, service] > correlations.loc[service, service] + 0.1:
                failures.append(
                    f"{service} enablement tracks the {other} requirement more closely"
                )

    print()
    if failures:
        print("SERVICE MAPPING CHECK FAILED")
        for failure in failures:
            print(f"  {failure}")
        print("\nSee assumption D9 in docs/assumptions.md and the mapping in src/clean.py.")
        return 1

    print("service mapping check passed: every service enables against its own requirement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
