"""Economics: capital costs, degradation, NPV, IRR and the sensitivity machinery."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src import economics
from src.economics import EconomicAssumptions, cash_flows, evaluate


def base(**overrides) -> EconomicAssumptions:
    defaults = dict(
        p_max_mw=100.0,
        duration_h=2.0,
        real_wacc=0.07,
        project_life_years=20,
        annual_revenue=30e6,
        cycles_per_year=300.0,
    )
    defaults.update(overrides)
    return EconomicAssumptions(**defaults)


# --------------------------------------------------------------------------------------
# capital cost
# --------------------------------------------------------------------------------------


def test_capex_matches_the_gencost_table() -> None:
    """GenCost 2025-26 Apx Table B.4: a 2-hour battery is $525/kWh in 2025."""
    assumptions = base(duration_h=2.0, development_share=0.0)
    assert assumptions.energy_mwh == pytest.approx(200.0)
    assert assumptions.plant_capex == pytest.approx(525.0 * 200.0 * 1000.0)
    assert assumptions.total_capex == pytest.approx(105e6)


def test_capex_per_kw_rises_with_duration_but_per_kwh_falls() -> None:
    per_kw = {}
    per_kwh = {}
    for duration in (1.0, 2.0, 4.0, 8.0):
        assumptions = base(duration_h=duration, development_share=0.0)
        per_kw[duration] = assumptions.total_capex / (assumptions.p_max_mw * 1000.0)
        per_kwh[duration] = assumptions.total_capex / (assumptions.energy_mwh * 1000.0)

    assert list(per_kw.values()) == sorted(per_kw.values())
    assert list(per_kwh.values()) == sorted(per_kwh.values(), reverse=True)


def test_development_cost_is_added_on_top_of_the_plant() -> None:
    plain = base(development_share=0.0)
    loaded = base(development_share=0.10)
    assert loaded.total_capex == pytest.approx(plain.total_capex * 1.10)


def test_battery_component_cost_falls_over_time() -> None:
    assumptions = base(duration_h=4.0)
    costs = [assumptions.battery_component_per_kwh(offset) for offset in (0, 5, 10, 20)]
    assert costs == sorted(costs, reverse=True)
    # GenCost Apx Table B.5: the 4-hour battery component is $265/kWh in 2025.
    assert costs[0] == pytest.approx(265.0)


# --------------------------------------------------------------------------------------
# degradation and augmentation
# --------------------------------------------------------------------------------------


def test_capacity_declines_monotonically_without_augmentation() -> None:
    trajectory = economics.capacity_trajectory(base(allow_augmentation=False))
    retained = trajectory["retained_capacity"].to_numpy()
    assert np.all(np.diff(retained) <= 1e-12)
    assert retained[-1] < retained[0]


def test_more_cycling_degrades_faster() -> None:
    gentle = economics.capacity_trajectory(base(cycles_per_year=150.0, allow_augmentation=False))
    hard = economics.capacity_trajectory(base(cycles_per_year=600.0, allow_augmentation=False))
    assert hard["retained_capacity"].iloc[-1] < gentle["retained_capacity"].iloc[-1]


def test_augmentation_restores_capacity_and_costs_money() -> None:
    trajectory = economics.capacity_trajectory(base(cycles_per_year=600.0))
    assert trajectory["augmented"].any()
    assert trajectory.loc[trajectory["augmented"], "augmentation_cost"].min() > 0
    # After an augmentation the battery is back at nameplate.
    first = trajectory.index[trajectory["augmented"]][0]
    assert trajectory.loc[first, "retained_capacity"] == pytest.approx(1.0)


def test_a_battery_that_never_cycles_still_fades_on_the_calendar() -> None:
    trajectory = economics.capacity_trajectory(
        base(cycles_per_year=0.0, allow_augmentation=False, calendar_fade_per_year=0.015)
    )
    assert trajectory["retained_capacity"].iloc[9] == pytest.approx(1.0 - 0.15, abs=1e-9)


# --------------------------------------------------------------------------------------
# cash flows, NPV and IRR
# --------------------------------------------------------------------------------------


def test_year_zero_is_capital_only() -> None:
    flows = cash_flows(base())
    assert flows.loc[0, "capex"] == pytest.approx(-base().total_capex)
    assert flows.loc[0, "revenue"] == 0.0
    assert flows.loc[0, "net_cash_flow"] < 0


def test_cash_flow_table_covers_the_whole_project_life() -> None:
    assumptions = base(project_life_years=15)
    flows = cash_flows(assumptions)
    assert flows.index.tolist() == list(range(0, 16))


def test_npv_rises_with_revenue_and_falls_with_the_discount_rate() -> None:
    low = evaluate(base(annual_revenue=20e6))["npv"]
    high = evaluate(base(annual_revenue=40e6))["npv"]
    assert high > low

    cheap = evaluate(base(real_wacc=0.05))["npv"]
    dear = evaluate(base(real_wacc=0.10))["npv"]
    assert cheap > dear


def test_npv_is_negative_when_revenue_cannot_cover_capital() -> None:
    assert evaluate(base(annual_revenue=0.0))["npv"] < 0


def test_irr_is_nan_when_the_project_never_turns_positive() -> None:
    assert np.isnan(evaluate(base(annual_revenue=0.0))["irr"])


def test_irr_exceeds_the_wacc_exactly_when_the_npv_is_positive() -> None:
    for revenue in (5e6, 15e6, 25e6, 40e6, 60e6):
        result = evaluate(base(annual_revenue=revenue))
        if np.isnan(result["irr"]):
            assert result["npv"] < 0
            continue
        assert (result["irr"] > base().real_wacc) == (result["npv"] > 0)


def test_the_breakeven_revenue_brings_the_npv_to_approximately_zero() -> None:
    """The levelised revenue requirement is the headline comparison number."""
    assumptions = base(annual_revenue=0.0)
    requirement = evaluate(assumptions)["levelised_revenue_requirement"]

    at_breakeven = evaluate(replace(assumptions, annual_revenue=requirement * assumptions.p_max_mw))
    assert at_breakeven["npv"] == pytest.approx(0.0, abs=abs(assumptions.total_capex) * 1e-6)


def test_capacity_revenue_adds_to_the_npv() -> None:
    merchant = evaluate(base(annual_capacity_revenue=0.0))["npv"]
    credited = evaluate(base(annual_capacity_revenue=10e6))["npv"]
    assert credited > merchant


def test_revenue_scales_down_with_retained_capacity() -> None:
    """A degraded battery earns less, so the cash flow must decline without augmentation."""
    flows = cash_flows(base(allow_augmentation=False))
    revenue = flows["revenue"].iloc[1:].to_numpy()
    assert np.all(np.diff(revenue) < 0)


# --------------------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------------------


def test_tornado_orders_parameters_by_swing() -> None:
    table = economics.tornado(
        base(),
        {
            "annual_revenue": (20e6, 40e6),
            "real_wacc": (0.05, 0.10),
            "fom_share_of_capex": (0.01, 0.03),
        },
    )
    swings = table["swing"].to_numpy()
    assert np.all(np.diff(swings) >= 0)  # sorted ascending for a horizontal chart
    assert table["swing"].max() > 0


def test_tornado_baseline_is_the_unvaried_case() -> None:
    assumptions = base()
    table = economics.tornado(assumptions, {"real_wacc": (0.05, 0.10)})
    assert table["baseline"].iloc[0] == pytest.approx(evaluate(assumptions)["npv_per_mw"])


def test_scenario_grid_covers_every_combination() -> None:
    grid = economics.scenario_grid(
        base(),
        revenue_cases={"2024": 300_000.0, "2025": 80_000.0},
        capacity_cases={"merchant": 0.0, "credits": 50_000.0},
        durations=(1.0, 2.0, 4.0),
    )
    assert len(grid) == 3 * 2 * 2
    assert set(grid["duration_h"]) == {1.0, 2.0, 4.0}
    assert grid["npv_per_mw"].notna().all()


def test_a_higher_revenue_case_never_produces_a_lower_npv() -> None:
    grid = economics.scenario_grid(
        base(), revenue_cases={"low": 50_000.0, "high": 400_000.0}, durations=(2.0,)
    )
    low = grid[grid["revenue_case"] == "low"]["npv_per_mw"].iloc[0]
    high = grid[grid["revenue_case"] == "high"]["npv_per_mw"].iloc[0]
    assert high > low
