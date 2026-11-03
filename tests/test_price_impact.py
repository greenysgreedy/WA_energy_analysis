"""The price-impact estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import price_impact


# --------------------------------------------------------------------------------------
# the regression
# --------------------------------------------------------------------------------------


def test_ols_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 50.0, size=2000)
    y = 3.0 + 0.25 * x + rng.normal(0.0, 1.0, size=2000)

    slope, error, r_squared = price_impact._ols(x, y)
    assert slope == pytest.approx(0.25, abs=0.01)
    assert error < 0.01
    assert r_squared > 0.95


def test_ols_reports_a_near_zero_slope_for_unrelated_series() -> None:
    rng = np.random.default_rng(1)
    slope, error, _ = price_impact._ols(
        rng.normal(size=2000), rng.normal(size=2000)
    )
    assert abs(slope) < 4 * error


# --------------------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------------------


def slopes_frame(slope: float, band: str = "0 to 100") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "band": "test",
                "price range": band,
                "slope $/MWh per MW": slope,
                "standard error": 0.001,
                "t": slope / 0.001,
                "r squared": 0.1,
                "intervals": 1000,
                "impact of 100 MW, $/MWh": slope * 100,
            }
        ]
    )


def test_a_small_impact_is_a_price_taker() -> None:
    # 100 MW moves the price $1 against a $50 level: two per cent.
    verdict = price_impact.price_taker_verdict(slopes_frame(0.01))
    assert verdict["impact as share of level"].iloc[0] == pytest.approx(0.02)
    assert bool(verdict["price taker"].iloc[0])


def test_a_large_impact_is_not_a_price_taker() -> None:
    # 100 MW moves the price $20 against a $50 level: forty per cent.
    verdict = price_impact.price_taker_verdict(slopes_frame(0.2))
    assert not bool(verdict["price taker"].iloc[0])


def test_the_verdict_uses_the_band_midpoint() -> None:
    verdict = price_impact.price_taker_verdict(slopes_frame(0.01, band="100 to 300"))
    assert verdict["band midpoint $/MWh"].iloc[0] == pytest.approx(200.0)


def test_the_threshold_is_adjustable() -> None:
    strict = price_impact.price_taker_verdict(slopes_frame(0.01), threshold_share=0.01)
    assert not bool(strict["price taker"].iloc[0])


# --------------------------------------------------------------------------------------
# event study
# --------------------------------------------------------------------------------------


def toy_metrics() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=900, freq="D")
    # A clean step down halfway through, with no seasonality, so the control window
    # should show no change and the event window should show the step.
    values = np.where(index < pd.Timestamp("2024-06-01"), 100.0, 60.0)
    return pd.DataFrame(
        {
            "capturable_spread_2h": values,
            "spread": values,
            "fcess_price_sum": values,
            "negative_share": np.full(len(index), 0.1),
        },
        index=index,
    )


def test_event_study_detects_a_step_and_the_control_does_not() -> None:
    metrics = toy_metrics()
    events = pd.Series({"BATTERY_1": pd.Timestamp("2024-06-01")})

    study = price_impact.event_study(metrics, events, window_days=90)
    event = study.loc[("BATTERY_1", "event"), "capturable_spread_2h change"]
    control = study.loc[("BATTERY_1", "previous year"), "capturable_spread_2h change"]

    assert event == pytest.approx(-40.0)
    assert control == pytest.approx(0.0)


def test_event_study_returns_empty_without_usable_windows() -> None:
    metrics = toy_metrics()
    events = pd.Series({"BATTERY_1": pd.Timestamp("2030-01-01")})
    assert price_impact.event_study(metrics, events).empty


# --------------------------------------------------------------------------------------
# capacity accumulation
# --------------------------------------------------------------------------------------


class FakeResult:
    def __init__(self, month, mw, mwh):
        self.first_full_month = pd.Period(month, freq="M")
        self.registered_mw = mw
        self.usable_mwh_estimate = mwh


def test_capacity_against_spread_steps_at_each_commissioning() -> None:
    metrics = toy_metrics()
    capacity = pd.DataFrame(
        {
            "facility_code": ["A", "B"],
            "from": [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-06-01")],
            "cumulative_mw": [100.0, 300.0],
            "cumulative_mwh": [200.0, 1000.0],
        }
    )
    joint = price_impact.capacity_against_spread(metrics, capacity)

    assert joint.loc["2023-01-01", "operating_storage_mw"] == 0.0
    assert joint.loc["2023-06-01", "operating_storage_mw"] == 100.0
    assert joint.loc["2024-06-01", "operating_storage_mw"] == 300.0
    assert joint.loc["2024-06-01", "operating_storage_mwh"] == 1000.0


def test_cumulative_capacity_accumulates_in_commissioning_order() -> None:
    fleet = {
        "B": FakeResult("2025-01", 200.0, 800.0),
        "A": FakeResult("2024-01", 100.0, 200.0),
    }
    capacity = price_impact.cumulative_storage_capacity(fleet)

    assert capacity["facility_code"].tolist() == ["A", "B"]
    assert capacity["cumulative_mw"].tolist() == [100.0, 300.0]
    assert capacity["cumulative_mwh"].tolist() == [200.0, 1000.0]


def test_facilities_that_never_operated_are_excluded() -> None:
    class NeverRan:
        first_full_month = None
        registered_mw = 200.0
        usable_mwh_estimate = 0.0

    capacity = price_impact.cumulative_storage_capacity(
        {"A": FakeResult("2024-01", 100.0, 200.0), "B": NeverRan()}
    )
    assert capacity["facility_code"].tolist() == ["A"]


@pytest.mark.slow
def test_the_estimated_slope_has_the_expected_sign() -> None:
    """More demand in the same five minutes should not lower the price."""
    slopes = price_impact.estimate_supply_slope("2024-01-01", "2025-12-31")
    all_intervals = slopes[slopes["band"] == "all intervals"]
    assert len(all_intervals) == 1
    assert all_intervals["slope $/MWh per MW"].iloc[0] > 0
