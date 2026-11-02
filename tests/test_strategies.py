"""Forecasts, rolling-horizon mechanics and the perfect-foresight bound."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import strategies
from src.model import SERVICES, BatteryParams

DAY = strategies.DAY


def toy_prices(n_days: int, seed: int = 0) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """A price series with a daily shape plus noise, so arbitrage is worth doing."""
    rng = np.random.default_rng(seed)
    shape = 60.0 + 90.0 * np.sin(np.linspace(0, 2 * np.pi, DAY, endpoint=False))
    energy = shape[None, :] + rng.normal(0.0, 25.0, size=(n_days, DAY))
    fcess = np.abs(rng.normal(8.0, 6.0, size=(n_days, len(SERVICES), DAY)))
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return dates, energy, fcess


def toy_battery(**overrides) -> BatteryParams:
    defaults = dict(p_max_mw=100.0, duration_h=2.0, round_trip_efficiency=0.88, c_thru=5.0)
    defaults.update(overrides)
    return BatteryParams(**defaults)


# --------------------------------------------------------------------------------------
# forecasts
# --------------------------------------------------------------------------------------


def test_persistence_holds_the_last_observed_price() -> None:
    actual = np.arange(100.0)
    forecast = strategies.forecast_persistence(actual, 50, 10)
    assert forecast == pytest.approx(np.full(10, 49.0))


def test_persistence_at_the_very_start_uses_the_first_price() -> None:
    actual = np.arange(10.0)
    assert strategies.forecast_persistence(actual, 0, 3) == pytest.approx([0.0, 0.0, 0.0])


def test_yesterday_reads_the_same_interval_a_day_earlier() -> None:
    actual = np.arange(3.0 * DAY)
    forecast = strategies.forecast_yesterday(actual, 2 * DAY, 5)
    assert forecast == pytest.approx(np.arange(DAY, DAY + 5, dtype=float))


def test_yesterday_falls_back_to_persistence_without_a_day_of_history() -> None:
    actual = np.arange(100.0)
    assert strategies.forecast_yesterday(actual, 50, 4) == pytest.approx(np.full(4, 49.0))


def test_last_week_reads_the_same_interval_seven_days_earlier() -> None:
    actual = np.arange(9.0 * DAY)
    forecast = strategies.forecast_last_week(actual, 8 * DAY, 3)
    assert forecast == pytest.approx(np.arange(DAY, DAY + 3, dtype=float))


def test_forecast_windows_past_the_end_are_padded_not_truncated() -> None:
    actual = np.arange(10.0)
    forecast = strategies.forecast_oracle(actual, 8, 5)
    assert len(forecast) == 5
    assert forecast == pytest.approx([8.0, 9.0, 9.0, 9.0, 9.0])


# --------------------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------------------


def test_flatten_preserves_order_for_two_and_three_dimensional_arrays() -> None:
    per_day = np.arange(2 * DAY, dtype=float).reshape(2, DAY)
    assert strategies.flatten(per_day) == pytest.approx(np.arange(2 * DAY, dtype=float))

    three = np.arange(2 * 5 * DAY, dtype=float).reshape(2, 5, DAY)
    flat = strategies.flatten(three)
    assert flat.shape == (5, 2 * DAY)
    assert flat[2, :DAY] == pytest.approx(three[0, 2])
    assert flat[2, DAY:] == pytest.approx(three[1, 2])


def test_flatten_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="cannot flatten"):
        strategies.flatten(np.zeros(5))


# --------------------------------------------------------------------------------------
# the bound
# --------------------------------------------------------------------------------------


def test_perfect_foresight_is_never_beaten_by_any_forecast() -> None:
    """The central inequality of the whole analysis."""
    dates, energy, fcess = toy_prices(14, seed=1)
    params = toy_battery()

    bound = strategies.perfect_foresight(dates, energy, fcess, params)
    for forecast in ("persistence", "yesterday", "last_week"):
        outcome = strategies.rolling_horizon(
            dates, energy, fcess, params, strategies.frozen_for(2024, forecast)
        )
        check = strategies.check_bound(outcome.daily, bound)
        assert check["bound_holds"], f"{forecast} beat the perfect-foresight bound"


def test_an_oracle_forecast_closes_most_of_the_gap_to_the_bound() -> None:
    """Separates forecast error from the cost of committing in small steps.

    An oracle rolling horizon still differs from daily perfect foresight because it
    carries the state of charge across days and does not close it, so it is not required
    to match exactly, only to beat every naive forecast comfortably.
    """
    dates, energy, fcess = toy_prices(10, seed=2)
    params = toy_battery()

    bound = strategies.perfect_foresight(dates, energy, fcess, params)
    oracle = strategies.rolling_horizon(
        dates, energy, fcess, params, strategies.StrategyConfig(forecast="oracle")
    )
    naive = strategies.rolling_horizon(
        dates, energy, fcess, params, strategies.frozen_for(2024, "persistence")
    )

    oracle_capture = oracle.total_revenue / bound["total_revenue"].sum()
    naive_capture = naive.total_revenue / bound["total_revenue"].sum()
    assert oracle_capture > naive_capture
    assert oracle_capture > 0.9


def test_capture_rate_aggregates_before_dividing() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    realisable = pd.DataFrame({"total_revenue": [10.0, 0.0, 40.0]}, index=index)
    bound = pd.DataFrame({"total_revenue": [20.0, 0.0, 80.0]}, index=index)

    rate = strategies.capture_rate(realisable, bound, freq="YS")
    assert rate["capture_rate"].iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------------------


def test_rolling_horizon_respects_power_and_state_of_charge_limits() -> None:
    dates, energy, fcess = toy_prices(7, seed=3)
    params = toy_battery(soc_min_frac=0.05, soc_max_frac=0.95)

    outcome = strategies.rolling_horizon(dates, energy, fcess, params)
    dispatch = outcome.dispatch

    assert dispatch["p_ch_mw"].max() <= params.p_max_mw + 1e-6
    assert dispatch["p_dis_mw"].max() <= params.p_max_mw + 1e-6
    assert dispatch["soc_mwh"].min() >= params.soc_min_mwh - 1e-6
    assert dispatch["soc_mwh"].max() <= params.soc_max_mwh + 1e-6


def test_rolling_horizon_solve_count_matches_the_commit_schedule() -> None:
    dates, energy, fcess = toy_prices(3, seed=4)
    config = strategies.StrategyConfig(commit_intervals=12)

    outcome = strategies.rolling_horizon(dates, energy, fcess, toy_battery(), config)
    assert outcome.solves == 3 * DAY // 12


def test_a_longer_commit_interval_cannot_beat_a_shorter_one_by_much() -> None:
    """Committing less often discards information, so it should not do better."""
    dates, energy, fcess = toy_prices(10, seed=5)
    params = toy_battery()

    short = strategies.rolling_horizon(
        dates, energy, fcess, params, strategies.StrategyConfig(commit_intervals=6)
    )
    long = strategies.rolling_horizon(
        dates, energy, fcess, params, strategies.StrategyConfig(commit_intervals=48)
    )
    assert short.total_revenue >= long.total_revenue * 0.95


def test_the_fcess_ceiling_is_respected_in_the_committed_dispatch() -> None:
    dates, energy, fcess = toy_prices(5, seed=6)
    caps = np.full((5, len(SERVICES), DAY), 7.0)

    outcome = strategies.rolling_horizon(
        dates, energy, fcess, toy_battery(), fcess_caps=caps
    )
    for service in SERVICES:
        assert outcome.dispatch[f"{service}_mw"].max() <= 7.0 + 1e-6


def test_the_obligation_prevents_charging_inside_the_window() -> None:
    dates, energy, fcess = toy_prices(6, seed=7)
    params = toy_battery(duration_h=4.0, has_capacity_credits=True)

    masks = np.zeros((6, DAY), dtype=bool)
    masks[:, 120:168] = True  # a four-hour window in every day

    outcome = strategies.rolling_horizon(dates, energy, fcess, params, esroi_masks=masks)
    flat_mask = strategies.flatten(masks.astype(float)) > 0.5

    charging_inside = outcome.dispatch.loc[flat_mask, "p_ch_mw"]
    assert charging_inside.max() <= 1e-6
    assert outcome.obligation_breaches == 0


def test_the_obligation_cannot_raise_realisable_revenue() -> None:
    dates, energy, fcess = toy_prices(8, seed=8)
    masks = np.zeros((8, DAY), dtype=bool)
    masks[:, 120:168] = True

    free = strategies.rolling_horizon(dates, energy, fcess, toy_battery(duration_h=4.0))
    obliged = strategies.rolling_horizon(
        dates,
        energy,
        fcess,
        toy_battery(duration_h=4.0, has_capacity_credits=True),
        esroi_masks=masks,
    )
    assert obliged.total_revenue <= free.total_revenue + 1e-6


def test_strategy_config_validates_its_parameters() -> None:
    with pytest.raises(ValueError, match="commit_intervals"):
        strategies.StrategyConfig(commit_intervals=0)
    with pytest.raises(ValueError, match="commit_intervals"):
        strategies.StrategyConfig(horizon_intervals=10, commit_intervals=20)
    with pytest.raises(ValueError, match="unknown forecast"):
        strategies.StrategyConfig(forecast="crystal ball")


def test_frozen_config_varies_only_the_forecast() -> None:
    """The no-tuning guard: a test year gets the design year's parameters."""
    design = strategies.frozen_for(strategies.DESIGN_YEAR, "persistence")
    test_year = strategies.frozen_for(2026, "persistence")

    assert design == test_year
    assert design.horizon_intervals == strategies.FROZEN_CONFIG.horizon_intervals
    assert design.commit_intervals == strategies.FROZEN_CONFIG.commit_intervals


def test_revenue_falls_as_the_throughput_cost_rises_under_a_forecast_too() -> None:
    dates, energy, fcess = toy_prices(8, seed=9)
    revenues = [
        strategies.rolling_horizon(dates, energy, fcess, toy_battery(c_thru=cost)).total_revenue
        for cost in (0.0, 10.0, 40.0, 120.0)
    ]
    assert revenues == sorted(revenues, reverse=True)
