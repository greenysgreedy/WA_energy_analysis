"""The FCESS reconciliation against AEMO's published figures."""

from __future__ import annotations

import pandas as pd
import pytest

from src import reconcile

pytestmark = pytest.mark.slow


def test_published_figures_carry_their_source() -> None:
    """A published number without a citation is not evidence."""
    assert not reconcile.PUBLISHED.empty
    assert reconcile.PUBLISHED["source"].str.contains("QED").all()
    assert (reconcile.PUBLISHED["market_enablement_cost_m"] > 0).all()
    assert reconcile.PUBLISHED["battery_volume_share"].between(0, 1).all()


def test_whole_market_cost_matches_the_published_total() -> None:
    """The strongest check in the project: it does not depend on anything about storage.

    Multiplying every facility's published enablement by the clearing price should
    reproduce AEMO's published quarterly FCESS enablement cost.
    """
    from src import fcess_market

    cost = fcess_market.load_market_cost().copy()
    cost.index = pd.to_datetime(cost.index)
    quarterly = cost["total_cost"].resample("QS").sum() / 1e6
    quarterly.index = quarterly.index.to_period("Q").astype(str)

    for row in reconcile.PUBLISHED.itertuples():
        modelled = float(quarterly.loc[row.quarter])
        published = row.market_enablement_cost_m
        assert modelled == pytest.approx(published, rel=0.15), (
            f"{row.quarter}: modelled ${modelled:.1f}m against published ${published:.1f}m"
        )


def test_storage_share_reconciles_once_the_service_mix_is_even() -> None:
    """Q4 2025, when storage was spread across every service, should agree closely.

    Q4 2024 is deliberately not asserted: storage then held 11 per cent of contingency
    raise, the service carrying 54 per cent of market cost, so a share of *volume* is not
    a share of *cost*. That gap is explained in docs/results.md rather than closed.
    """
    table = reconcile.reconcile()
    row = table.loc["2025Q4"]
    assert row["ratio"] == pytest.approx(1.0, abs=0.25)


def test_service_mix_shows_why_the_earlier_quarter_reads_low() -> None:
    mix = reconcile.service_mix("2024Q4")
    raise_share = mix.loc["contingency_raise", "storage share of requirement"]
    raise_cost_weight = mix.loc["contingency_raise", "share of market cost"]

    # Storage held a small share of the service that carried most of the cost.
    assert raise_share < 0.5
    assert raise_cost_weight > 0.3


def test_the_report_names_the_source_for_every_quarter() -> None:
    text = reconcile.report()
    for quarter in reconcile.PUBLISHED["quarter"]:
        assert quarter in text
    assert "QED" in text
