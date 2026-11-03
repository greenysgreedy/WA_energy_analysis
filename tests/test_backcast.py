"""Backcast logic: discharge runs, usable-energy estimation and operating windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backcast


# --------------------------------------------------------------------------------------
# discharge runs
# --------------------------------------------------------------------------------------


def test_discharge_runs_splits_on_charging_intervals() -> None:
    series = pd.Series([1.0, 2.0, 0.0, -1.0, 3.0, 4.0, 5.0, 0.0])
    assert backcast.discharge_runs(series) == pytest.approx([3.0, 12.0])


def test_a_run_still_open_at_the_end_is_counted() -> None:
    series = pd.Series([1.0, 2.0, -1.0, 4.0, 4.0])
    assert backcast.discharge_runs(series) == pytest.approx([3.0, 8.0])


def test_no_discharge_gives_no_runs() -> None:
    assert backcast.discharge_runs(pd.Series([-1.0, -2.0, 0.0])) == []


def test_usable_energy_uses_a_high_quantile_not_the_maximum() -> None:
    """One anomalous run must not set the estimate."""
    normal = [10.0, -1.0] * 50
    with_outlier = pd.Series(normal + [1000.0])

    estimate = backcast.estimate_usable_energy(with_outlier, quantile=0.9)
    assert estimate == pytest.approx(10.0, abs=1e-6)


def test_usable_energy_of_an_idle_facility_is_zero() -> None:
    assert backcast.estimate_usable_energy(pd.Series([-1.0, -2.0])) == 0.0


# --------------------------------------------------------------------------------------
# operating window
# --------------------------------------------------------------------------------------


def frame_with_throughput(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"throughput_mwh": values}, index=index)


def test_first_full_month_skips_a_commissioning_month() -> None:
    """January barely runs, February runs on nearly every day."""
    january = [1.0] * 31  # well below the threshold
    february = [100.0] * 29
    daily = frame_with_throughput(january + february)

    first = backcast.first_full_operating_month(daily, usable_mwh=100.0)
    assert first == pd.Period("2024-02", freq="M")


def test_a_month_that_is_mostly_idle_does_not_count_as_full() -> None:
    january = [100.0] * 10 + [0.0] * 21  # only a third of days operating
    february = [100.0] * 29
    daily = frame_with_throughput(january + february)

    assert backcast.first_full_operating_month(daily, usable_mwh=100.0) == pd.Period(
        "2024-02", freq="M"
    )


def test_no_operating_month_returns_none() -> None:
    daily = frame_with_throughput([0.0] * 60)
    assert backcast.first_full_operating_month(daily, usable_mwh=100.0) is None


def test_an_empty_frame_returns_none() -> None:
    assert backcast.first_full_operating_month(pd.DataFrame(), usable_mwh=100.0) is None


# --------------------------------------------------------------------------------------
# duration from capacity credits
# --------------------------------------------------------------------------------------


def test_duration_from_credits_inverts_the_accreditation_rule(monkeypatch) -> None:
    """Credits are capacity sustainable over four hours, net of discharge losses."""
    monkeypatch.setattr(
        backcast.data,
        "capacity_credits_for",
        lambda code: pd.Series({"2023-24": 20.0, "2024-25": 50.0}),
    )
    # The largest allocation is used, because credits track demonstrated capability.
    assert backcast.duration_from_capacity_credits("X") == pytest.approx(50.0 * 4.0 / 0.938)
    assert backcast.duration_from_capacity_credits("X", "2023-24") == pytest.approx(
        20.0 * 4.0 / 0.938
    )


def test_duration_from_credits_is_none_without_credits(monkeypatch) -> None:
    monkeypatch.setattr(backcast.data, "capacity_credits_for", lambda code: pd.Series(dtype=float))
    assert backcast.duration_from_capacity_credits("X") is None

    monkeypatch.setattr(backcast.data, "capacity_credits_for", lambda code: pd.Series({"a": 0.0}))
    assert backcast.duration_from_capacity_credits("X") is None


def test_duration_from_credits_is_none_for_an_unknown_year(monkeypatch) -> None:
    monkeypatch.setattr(
        backcast.data, "capacity_credits_for", lambda code: pd.Series({"2024-25": 50.0})
    )
    assert backcast.duration_from_capacity_credits("X", "1999-00") is None


# --------------------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_fleet_rebuild_produces_plausible_durations() -> None:
    """AEMO states COLLIE_ESR1 as 200 MW / 800 MWh and KWINANA_ESR2 as 225 MW / 900 MWh."""
    from src import data

    register = data.storage_facilities().set_index("facility_code")
    stated = {"COLLIE_ESR1": (200.0, 800.0), "KWINANA_ESR2": (225.0, 900.0)}

    for code, (stated_mw, stated_mwh) in stated.items():
        result = backcast.rebuild_facility(
            code, registered_mw=float(register.loc[code, "registered_mw"])
        )
        assert result.observed_peak_mw == pytest.approx(stated_mw, rel=0.05)
        assert result.usable_mwh_estimate == pytest.approx(stated_mwh, rel=0.15)


@pytest.mark.slow
def test_registered_mw_is_about_twice_the_observed_peak() -> None:
    """Assumption D8: registered MW is a bidirectional range, not an injection rating."""
    from src import data

    register = data.storage_facilities().set_index("facility_code")
    ratios = []
    for code in ("COLLIE_ESR1", "KWINANA_ESR1", "COLLIE_BESS2"):
        registered = float(register.loc[code, "registered_mw"])
        result = backcast.rebuild_facility(code, registered_mw=registered)
        ratios.append(result.observed_peak_mw / registered)

    assert all(0.45 < ratio < 0.55 for ratio in ratios), ratios


@pytest.mark.slow
def test_no_real_battery_beats_its_own_perfect_foresight_bound() -> None:
    """The central check on the model. Reads the comparison table the runner writes."""
    from src.paths import OUTPUTS

    path = OUTPUTS / "backcast_model_comparison.csv"
    if not path.exists():
        pytest.skip("run scripts/run_backcast.py first")

    comparison = pd.read_csv(path, index_col=0)
    assert (comparison["actual_over_pf"] <= 1.0).all(), comparison["actual_over_pf"].to_dict()
