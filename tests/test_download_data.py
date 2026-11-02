"""Filename period logic in the downloader.

Getting these wrong either silently drops post-reform data or re-downloads a gigabyte
of history on every run, so they are tested rather than eyeballed.
"""

from __future__ import annotations

from datetime import date

import pytest

from src import download_data as dd


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("FacilityScada-2023-10.csv", (2023, 10)),
        ("MarketSchedule-2026-01.csv", (2026, 1)),
        ("MarketClearingPrices-2024.csv", (2024,)),
        ("electric-storage-resource-obligation-intervals-2025.csv", (2025,)),
        ("facilities.csv", None),
        ("fcess.csv", None),
    ],
)
def test_file_period(filename: str, expected) -> None:
    assert dd.file_period(filename) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Files are grouped by trading day, so September 2023 cannot hold a post-reform day.
        ("FacilityScada-2023-09.csv", False),
        ("FacilityScada-2023-10.csv", True),
        ("FacilityScada-2022-12.csv", False),
        # The 2023 yearly files contain October to December 2023, so they are needed.
        ("MarketClearingPrices-2023.csv", True),
        ("MarketClearingPrices-2022.csv", False),
        # Static files carry no period and are always in scope.
        ("facilities.csv", True),
    ],
)
def test_is_post_reform(filename: str, expected: bool) -> None:
    assert dd.is_post_reform(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("FacilityScada-2026-11.csv", True),
        ("FacilityScada-2026-10.csv", False),
        ("MarketClearingPrices-2026.csv", True),
        ("MarketClearingPrices-2025.csv", False),
        ("facilities.csv", True),
    ],
)
def test_is_current_period(filename: str, expected: bool) -> None:
    assert dd.is_current_period(filename, today=date(2026, 11, 2)) is expected


def test_registry_covers_every_dataset_the_pipeline_reads() -> None:
    required = {
        "market-clearing-prices-csv",
        "market-schedule-csv",
        "facility-scada-csv",
        "electric-storage-resource-obligation-intervals",
        "post-facilities",
        "fcess",
        "capacity-credits-since-market-start",
        "reserve-capacity-prices",
    }
    assert required <= set(dd.BY_NAME)


def test_registry_grains_are_known() -> None:
    assert {d.grain for d in dd.DATASETS} <= {"monthly", "yearly", "static"}
