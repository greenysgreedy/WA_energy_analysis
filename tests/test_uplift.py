"""FCESS Uplift: the measurement that justifies excluding it for storage."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data
from src.paths import PROCESSED

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def uplift() -> pd.DataFrame:
    path = PROCESSED / "fcess_uplift.parquet"
    if not path.exists():
        pytest.skip("run 'python -m src.clean --only fcess_uplift' first")
    return pd.read_parquet(path)


def test_uplift_is_keyed_uniquely_by_interval_and_facility(uplift: pd.DataFrame) -> None:
    assert not uplift.duplicated(subset=["interval_start", "facility_code"]).any()


def test_the_service_shares_sum_to_the_total(uplift: pd.DataFrame) -> None:
    shares = [c for c in uplift.columns if c.startswith("uplift_") and c != "uplift_total"]
    difference = (uplift[shares].sum(axis=1) - uplift["uplift_total"]).abs()
    # The shares are published rounded, so allow a cent per row.
    assert difference.max() < 0.01


def test_storage_received_a_negligible_share_of_uplift(uplift: pd.DataFrame) -> None:
    """Assumption F12. If this ever fails, excluding Uplift stops being defensible."""
    storage = set(data.storage_facilities()["facility_code"])
    storage_total = uplift.loc[uplift["facility_code"].isin(storage), "uplift_total"].sum()
    market_total = uplift["uplift_total"].sum()

    assert market_total > 0
    assert storage_total / market_total < 0.01, (
        f"storage received {storage_total / market_total:.3%} of FCESS Uplift; "
        "modelled revenue can no longer ignore it"
    )


def test_uplift_collapsed_after_the_fcess_cost_review(uplift: pd.DataFrame) -> None:
    """AEMO attributes a roughly 96% fall to the November 2024 FCESS Cost Review."""
    annual = uplift.groupby(uplift["trading_date"].dt.year)["uplift_total"].sum()
    assert annual.loc[2025] < annual.loc[2024] * 0.2
