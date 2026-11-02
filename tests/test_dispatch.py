"""Toy-day dispatch tests with hand-calculated answers.

These must pass before the model is run on real data. Each case uses a two-interval or
four-interval day with round numbers so the optimal revenue can be worked out by hand
and written into the assertion, rather than being read back out of the solver.

Standing toy battery: 10 MW, 2 hours, so 20 MWh of usable energy, opening and closing
state of charge 10 MWh, and one-hour intervals so 1 MW for one interval is 1 MWh.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import (
    SERVICES,
    BatteryParams,
    DispatchLP,
    settle,
    solve_dispatch,
)

DT = 1.0  # one-hour intervals keep the arithmetic in whole MWh


def energy_only(**overrides) -> BatteryParams:
    """Toy battery that cannot offer FCESS, so only energy arbitrage is available."""
    defaults = dict(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.5,
        c_thru=0.0,
        loss_factor=1.0,
        enabled_services=(),
    )
    defaults.update(overrides)
    return BatteryParams(**defaults)


def fcess_prices_for(service: str, price: float, n: int) -> np.ndarray:
    prices = np.zeros((len(SERVICES), n))
    prices[SERVICES.index(service)] = price
    return prices


# --------------------------------------------------------------------------------------
# 1. A spread wider than the efficiency loss produces one full cycle
# --------------------------------------------------------------------------------------


def test_lossless_two_price_day_cycles_once_for_hand_calculated_revenue() -> None:
    """Charge 10 MWh at $0, discharge 10 MWh at $100. Revenue is exactly $1,000."""
    params = energy_only()
    result = solve_dispatch(np.array([0.0, 100.0]), params, dt=DT)

    assert result.p_ch == pytest.approx([10.0, 0.0], abs=1e-6)
    assert result.p_dis == pytest.approx([0.0, 10.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(1000.0, abs=1e-6)
    assert result.energy_revenue == pytest.approx(1000.0, abs=1e-6)


def test_lossy_two_price_day_revenue_is_scaled_by_discharge_efficiency() -> None:
    """With a 64 % round trip, 10 MWh in returns 6.4 MWh out, so revenue is $640.

    Charging 10 MWh at the cap raises the state of charge by 0.8 x 10 = 8 MWh. Returning
    to the opening state of charge discharges 8 x 0.8 = 6.4 MWh at $100.
    """
    params = energy_only(round_trip_efficiency=0.64)
    result = solve_dispatch(np.array([0.0, 100.0]), params, dt=DT)

    assert result.p_ch == pytest.approx([10.0, 0.0], abs=1e-6)
    assert result.p_dis == pytest.approx([0.0, 6.4], abs=1e-6)
    assert result.total_revenue == pytest.approx(640.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 2. A spread narrower than the efficiency loss produces no cycling
# --------------------------------------------------------------------------------------


def test_spread_below_the_efficiency_loss_leaves_the_battery_idle() -> None:
    """At 64 % round trip the break-even price ratio is 1/0.64 = 1.5625.

    Buying at $80 and selling at $100 is a ratio of 1.25, so cycling loses money:
    10 MWh costs $800 and returns 6.4 MWh for $640.
    """
    params = energy_only(round_trip_efficiency=0.64)
    result = solve_dispatch(np.array([80.0, 100.0]), params, dt=DT)

    assert result.p_ch == pytest.approx([0.0, 0.0], abs=1e-6)
    assert result.p_dis == pytest.approx([0.0, 0.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(0.0, abs=1e-6)


def test_throughput_cost_can_make_an_otherwise_profitable_spread_unprofitable() -> None:
    """A $0-to-$100 spread is worth $100/MWh; a $120/MWh throughput cost kills it.

    The cost is charged on half of total throughput, so a round trip of 10 MWh in and
    10 MWh out is charged on 10 MWh: $1,200 against $1,000 of revenue.
    """
    params = energy_only(c_thru=120.0)
    result = solve_dispatch(np.array([0.0, 100.0]), params, dt=DT)

    assert result.throughput_mwh == pytest.approx(0.0, abs=1e-6)
    assert result.total_revenue == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 3. A flat price earns nothing
# --------------------------------------------------------------------------------------


def test_flat_price_earns_nothing_and_does_not_cycle() -> None:
    params = energy_only(c_thru=1.0)  # a small cost breaks the degenerate tie
    result = solve_dispatch(np.full(4, 50.0), params, dt=DT)

    assert result.throughput_mwh == pytest.approx(0.0, abs=1e-6)
    assert result.total_revenue == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 4. Negative prices induce charging and are paid for
# --------------------------------------------------------------------------------------


def test_negative_price_interval_is_charged_through_and_paid_for() -> None:
    """Charging 10 MWh at -$50 earns $500; discharging 10 MWh at $10 earns $100."""
    params = energy_only()
    result = solve_dispatch(np.array([-50.0, 10.0]), params, dt=DT)

    assert result.p_ch == pytest.approx([10.0, 0.0], abs=1e-6)
    assert result.p_dis == pytest.approx([0.0, 10.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(600.0, abs=1e-6)


def test_a_day_that_is_negative_throughout_still_charges_within_soc_room() -> None:
    """With every interval negative the battery wants to absorb as much as it can.

    The cyclic state of charge forces it back to 10 MWh, so it can only shift energy
    from the most negative interval to the least negative one.
    """
    params = energy_only()
    result = solve_dispatch(np.array([-100.0, -10.0]), params, dt=DT)

    assert result.p_ch[0] == pytest.approx(10.0, abs=1e-6)
    assert result.p_dis[1] == pytest.approx(10.0, abs=1e-6)
    # Absorb at -$100 (earning $1,000), release at -$10 (costing $100).
    assert result.total_revenue == pytest.approx(900.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 5. FCESS with no energy spread is taken to the binding limit
# --------------------------------------------------------------------------------------


def test_fcess_enablement_is_taken_to_the_power_headroom_limit() -> None:
    """Flat energy price, $100/MWh contingency raise, nothing else binding.

    Power headroom allows 10 MW of raise in both intervals, so revenue is
    $100 x 10 MW x 1 h x 2 intervals = $2,000.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.5,
        enabled_services=("contingency_raise",),
        fcess_raise_duration_h=0.25,
    )
    result = solve_dispatch(
        np.zeros(2),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        dt=DT,
    )

    assert result.enablement["contingency_raise"] == pytest.approx([10.0, 10.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(2000.0, abs=1e-6)
    assert result.energy_revenue == pytest.approx(0.0, abs=1e-6)


def test_fcess_enablement_is_capped_by_the_stored_energy_headroom() -> None:
    """A nearly empty battery cannot promise raise it has no energy to deliver.

    The state of charge is held at 10 % of 20 MWh, so 2 MWh, by setting the ceiling to
    the opening level. With a one-hour required raise duration and a lossless discharge,
    enabled raise is limited to 2 MW even though 10 MW of power headroom is free.
    Revenue is $100 x 2 MW x 1 h x 2 intervals = $400.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_max_frac=0.1,
        soc_initial_frac=0.1,
        enabled_services=("contingency_raise",),
        fcess_raise_duration_h=1.0,
    )
    result = solve_dispatch(
        np.zeros(2),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        dt=DT,
    )

    assert result.enablement["contingency_raise"] == pytest.approx([2.0, 2.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(400.0, abs=1e-6)


def test_charging_can_buy_fcess_energy_headroom_when_the_ceiling_allows_it() -> None:
    """Co-optimisation lets the battery charge purely to be able to sell more raise.

    Same nearly empty battery, but now free to charge to 20 MWh. Charging ``x`` MWh in
    interval 0 lifts the state of charge to ``2 + x``, and the cyclic closing condition
    forces it to discharge ``x`` in interval 1, which consumes ``x`` of raise headroom
    there. So enabled raise is 2 MW in interval 0 and ``min(10 - x, 2 + x)`` in interval
    1, maximised at ``x = 4`` giving 6 MW. Revenue is $100 x (2 + 6) = $800.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.1,
        enabled_services=("contingency_raise",),
        fcess_raise_duration_h=1.0,
    )
    result = solve_dispatch(
        np.zeros(2),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        dt=DT,
    )

    assert result.enablement["contingency_raise"] == pytest.approx([2.0, 6.0], abs=1e-6)
    assert result.p_ch[0] == pytest.approx(4.0, abs=1e-6)
    assert result.total_revenue == pytest.approx(800.0, abs=1e-6)


def test_fcess_enablement_is_capped_by_the_accredited_quantity() -> None:
    """Accreditation of 4 MW binds before the 10 MW of power headroom does."""
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.5,
        enabled_services=("contingency_raise",),
        accredited_mw={"contingency_raise": 4.0},
        fcess_raise_duration_h=0.25,
    )
    result = solve_dispatch(
        np.zeros(2),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        dt=DT,
    )

    assert result.enablement["contingency_raise"] == pytest.approx([4.0, 4.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(800.0, abs=1e-6)


def test_a_market_requirement_cap_binds_before_the_accreditation() -> None:
    """A battery cannot be paid for holding more reserve than the market procured.

    Accreditation allows 10 MW, but the per-interval cap allows 3 MW then 7 MW, so
    revenue is $100 x (3 + 7) = $1,000 rather than $2,000.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        enabled_services=("contingency_raise",),
        fcess_raise_duration_h=0.25,
    )
    cap = np.zeros((len(SERVICES), 2))
    cap[SERVICES.index("contingency_raise")] = [3.0, 7.0]

    result = solve_dispatch(
        np.zeros(2),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        fcess_cap=cap,
        dt=DT,
    )

    assert result.enablement["contingency_raise"] == pytest.approx([3.0, 7.0], abs=1e-6)
    assert result.total_revenue == pytest.approx(1000.0, abs=1e-6)


def test_a_requirement_cap_can_only_reduce_revenue() -> None:
    rng = np.random.default_rng(11)
    prices = rng.normal(60.0, 120.0, size=96)
    fcess = np.abs(rng.normal(10.0, 15.0, size=(len(SERVICES), 96)))
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0)

    uncapped = solve_dispatch(prices, params, fcess_prices=fcess, dt=1.0 / 12.0)
    capped = solve_dispatch(
        prices,
        params,
        fcess_prices=fcess,
        fcess_cap=np.full((len(SERVICES), 96), 1.0),
        dt=1.0 / 12.0,
    )

    assert capped.total_revenue <= uncapped.total_revenue + 1e-6
    for quantity in capped.enablement.values():
        assert quantity.max() <= 1.0 + 1e-6


def test_rejects_a_malformed_requirement_cap() -> None:
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0)

    with pytest.raises(ValueError, match="fcess_cap must have shape"):
        DispatchLP(4, params, fcess_cap=np.zeros((len(SERVICES), 3)))
    with pytest.raises(ValueError, match="non-negative"):
        DispatchLP(4, params, fcess_cap=np.full((len(SERVICES), 4), -1.0))


def test_raise_enablement_competes_with_discharging_for_power_headroom() -> None:
    """Raise headroom and discharge share the 10 MW rating.

    Energy prices are $0 then $100, and contingency raise is $100/MWh throughout.
    Discharging 1 MW in interval 1 earns $100 and displaces 1 MW of raise worth $100,
    so the two are exactly indifferent there. Interval 0 can charge *and* sell raise,
    because charging consumes lower headroom, not raise headroom. The test asserts the
    headroom constraint holds rather than picking between tied optima.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=2.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.5,
        enabled_services=("contingency_raise",),
        fcess_raise_duration_h=0.25,
    )
    result = solve_dispatch(
        np.array([0.0, 100.0]),
        params,
        fcess_prices=fcess_prices_for("contingency_raise", 100.0, 2),
        dt=DT,
    )

    headroom = result.p_dis - result.p_ch + result.enablement["contingency_raise"]
    assert np.all(headroom <= params.p_max_mw + 1e-6)
    # Charging in interval 0 while selling 10 MW of raise, then releasing in interval 1,
    # is worth $1,000 of raise plus $1,000 of energy arbitrage.
    assert result.total_revenue == pytest.approx(2000.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 6. State of charge stays inside its bounds and returns to where it started
# --------------------------------------------------------------------------------------


def test_state_of_charge_stays_in_bounds_and_closes_where_it_opened() -> None:
    rng = np.random.default_rng(0)
    prices = rng.normal(50.0, 80.0, size=48)
    params = BatteryParams(
        p_max_mw=100.0,
        duration_h=4.0,
        round_trip_efficiency=0.88,
        soc_min_frac=0.05,
        soc_max_frac=0.95,
        soc_initial_frac=0.5,
        c_thru=5.0,
    )
    result = solve_dispatch(prices, params, dt=1.0 / 12.0)

    assert result.soc.min() >= params.soc_min_mwh - 1e-6
    assert result.soc.max() <= params.soc_max_mwh + 1e-6
    assert result.soc[0] == pytest.approx(params.soc_initial_mwh, abs=1e-6)
    assert result.soc[-1] == pytest.approx(params.soc_initial_mwh, abs=1e-6)
    assert result.energy_balance_error() < 1e-6


def test_free_terminal_state_of_charge_is_never_worse_than_a_pinned_one() -> None:
    """Relaxing the closing state of charge can only add revenue, never remove it."""
    prices = np.array([10.0, 20.0, 30.0, 200.0])
    pinned = solve_dispatch(np.array(prices), energy_only(), dt=DT)
    free = solve_dispatch(np.array(prices), energy_only(cyclic_soc=False), dt=DT)

    assert free.total_revenue >= pinned.total_revenue - 1e-6
    # Here the free case can simply sell its opening stock into the $200 interval.
    assert free.total_revenue > pinned.total_revenue


# --------------------------------------------------------------------------------------
# Monotonicity invariants
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("c_thru", [0.0, 5.0, 20.0, 60.0, 200.0])
def test_revenue_is_non_increasing_in_the_throughput_cost(c_thru: float) -> None:
    rng = np.random.default_rng(1)
    prices = rng.normal(60.0, 120.0, size=96)
    baseline = solve_dispatch(prices, energy_only(c_thru=0.0), dt=1.0 / 12.0)
    tested = solve_dispatch(prices, energy_only(c_thru=c_thru), dt=1.0 / 12.0)

    assert tested.total_revenue <= baseline.total_revenue + 1e-6


def test_revenue_is_non_decreasing_in_round_trip_efficiency() -> None:
    rng = np.random.default_rng(2)
    prices = rng.normal(60.0, 120.0, size=96)
    revenues = [
        solve_dispatch(prices, energy_only(round_trip_efficiency=eta), dt=1.0 / 12.0).total_revenue
        for eta in (0.70, 0.80, 0.88, 0.95, 1.00)
    ]

    assert revenues == sorted(revenues)


def test_revenue_is_non_decreasing_in_duration() -> None:
    rng = np.random.default_rng(3)
    prices = rng.normal(60.0, 120.0, size=288)
    revenues = [
        solve_dispatch(prices, energy_only(duration_h=h), dt=1.0 / 12.0).total_revenue
        for h in (1.0, 2.0, 4.0, 8.0)
    ]

    assert revenues == sorted(revenues)


def test_allowing_fcess_never_reduces_revenue() -> None:
    """Co-optimisation is a relaxation: the energy-only solution stays feasible."""
    rng = np.random.default_rng(4)
    prices = rng.normal(60.0, 120.0, size=96)
    fcess = np.abs(rng.normal(10.0, 15.0, size=(len(SERVICES), 96)))

    energy = solve_dispatch(prices, energy_only(), dt=1.0 / 12.0)
    both = solve_dispatch(
        prices,
        BatteryParams(p_max_mw=10.0, duration_h=2.0, round_trip_efficiency=1.0),
        fcess_prices=fcess,
        dt=1.0 / 12.0,
    )

    assert both.total_revenue >= energy.total_revenue - 1e-6


def test_no_simultaneous_charge_and_discharge_at_positive_prices() -> None:
    """The LP has no binary forbidding it, so confirm it does not happen anyway."""
    rng = np.random.default_rng(5)
    prices = np.abs(rng.normal(80.0, 40.0, size=288))
    result = solve_dispatch(prices, energy_only(c_thru=2.0), dt=1.0 / 12.0)

    overlap = np.minimum(result.p_ch, result.p_dis)
    assert overlap.max() < 1e-6


# --------------------------------------------------------------------------------------
# ESR obligation intervals
# --------------------------------------------------------------------------------------


def test_esr_obligation_forbids_charging_and_reserves_energy_in_the_window() -> None:
    """A 4-hour battery with Capacity Credits must hold its window through.

    Eight one-hour intervals, with the cheapest energy inside the obligation window so
    an unconstrained battery would charge there. The obligation must prevent it and must
    leave enough energy at the window start to discharge at the accredited capacity for
    the whole window.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=4.0,
        round_trip_efficiency=1.0,
        soc_initial_frac=0.5,
        enabled_services=(),
        has_capacity_credits=True,
        esr_duration_requirement_h=4.0,
    )
    prices = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0])
    mask = np.array([False, False, True, True, True, True, False, False])

    result = solve_dispatch(prices, params, dt=DT, esroi_mask=mask)

    assert result.p_ch[mask] == pytest.approx(np.zeros(4), abs=1e-6)
    # Accredited capacity is the full 10 MW (duration equals the requirement), and the
    # window is 4 hours, so 40 MWh must be available when it opens.
    assert result.soc[2] >= params.accredited_capacity_mw * 4.0 - 1e-6


def test_capacity_credit_accreditation_is_derated_by_duration() -> None:
    """A 1-hour battery can only be accredited for a quarter of a 4-hour obligation."""
    def battery(duration_h: float) -> BatteryParams:
        return BatteryParams(
            p_max_mw=100.0,
            duration_h=duration_h,
            round_trip_efficiency=1.0,
            has_capacity_credits=True,
        )

    assert battery(1.0).accredited_capacity_mw == pytest.approx(25.0)
    assert battery(2.0).accredited_capacity_mw == pytest.approx(50.0)
    assert battery(4.0).accredited_capacity_mw == pytest.approx(100.0)
    # Beyond the obligation duration the power rating binds instead.
    assert battery(8.0).accredited_capacity_mw == pytest.approx(100.0)
    # No Capacity Credits means no accreditation and no obligation.
    assert BatteryParams(p_max_mw=100.0, duration_h=4.0).accredited_capacity_mw == 0.0


def test_accreditation_is_also_derated_by_discharge_efficiency_and_soc_limits() -> None:
    """Accreditation is a sent-out capacity, so losses and a reserved floor both bite."""
    lossy = BatteryParams(
        p_max_mw=100.0,
        duration_h=4.0,
        round_trip_efficiency=0.88,
        has_capacity_credits=True,
    )
    # 400 MWh usable, discharged at sqrt(0.88) = 0.93808, spread over four hours.
    assert lossy.accredited_capacity_mw == pytest.approx(400.0 * np.sqrt(0.88) / 4.0)

    reserved = BatteryParams(
        p_max_mw=100.0,
        duration_h=4.0,
        round_trip_efficiency=1.0,
        soc_min_frac=0.1,
        soc_max_frac=0.9,
        has_capacity_credits=True,
    )
    assert reserved.accredited_capacity_mw == pytest.approx(80.0)


def test_the_obligation_cannot_increase_revenue() -> None:
    """Adding a constraint can only reduce the achievable energy and FCESS revenue."""
    rng = np.random.default_rng(6)
    prices = rng.normal(60.0, 120.0, size=48)
    mask = np.zeros(48, dtype=bool)
    mask[20:24] = True  # a four-hour window, matching the obligation duration

    def battery(credits: bool) -> BatteryParams:
        return BatteryParams(
            p_max_mw=10.0,
            duration_h=4.0,
            enabled_services=(),
            has_capacity_credits=credits,
        )

    free = solve_dispatch(prices, battery(False), dt=1.0)
    obliged = solve_dispatch(prices, battery(True), dt=1.0, esroi_mask=mask)

    assert obliged.total_revenue <= free.total_revenue + 1e-6


def test_an_obligation_window_longer_than_the_battery_stays_feasible() -> None:
    """A published window longer than the obligation duration must not break the LP."""
    params = BatteryParams(
        p_max_mw=10.0,
        duration_h=1.0,
        round_trip_efficiency=1.0,
        enabled_services=(),
        has_capacity_credits=True,
    )
    mask = np.zeros(12, dtype=bool)
    mask[4:12] = True  # eight hours, far longer than the one-hour battery

    result = solve_dispatch(np.linspace(0.0, 100.0, 12), params, dt=1.0, esroi_mask=mask)

    # The strongest obligation it can meet is to be full when the window opens.
    assert result.soc[4] == pytest.approx(params.soc_max_mwh, abs=1e-6)


# --------------------------------------------------------------------------------------
# Settlement on actuals, and re-solvable LPs
# --------------------------------------------------------------------------------------


def test_settling_an_optimal_dispatch_on_its_own_prices_reproduces_its_revenue() -> None:
    rng = np.random.default_rng(7)
    prices = rng.normal(60.0, 120.0, size=96)
    fcess = np.abs(rng.normal(10.0, 15.0, size=(len(SERVICES), 96)))
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0, c_thru=3.0)

    result = solve_dispatch(prices, params, fcess_prices=fcess, dt=1.0 / 12.0)
    settled = settle(
        result.p_ch, result.p_dis, result.enablement, prices, fcess, params, dt=1.0 / 12.0
    )

    assert settled["total_revenue"] == pytest.approx(result.total_revenue, rel=1e-9)


def test_settling_a_dispatch_on_different_prices_cannot_beat_optimising_on_them() -> None:
    """The core realisable-versus-theoretical inequality, in miniature."""
    rng = np.random.default_rng(8)
    forecast = rng.normal(60.0, 120.0, size=96)
    actual = forecast + rng.normal(0.0, 60.0, size=96)
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0, enabled_services=())

    on_forecast = solve_dispatch(forecast, params, dt=1.0 / 12.0)
    settled_on_actual = settle(
        on_forecast.p_ch,
        on_forecast.p_dis,
        on_forecast.enablement,
        actual,
        np.zeros((len(SERVICES), 96)),
        params,
        dt=1.0 / 12.0,
    )
    perfect = solve_dispatch(actual, params, dt=1.0 / 12.0)

    assert settled_on_actual["total_revenue"] <= perfect.total_revenue + 1e-6


def test_a_built_lp_can_be_resolved_with_new_prices_and_opening_soc() -> None:
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0, enabled_services=())
    lp = DispatchLP(2, params, dt=DT)

    # 10 MWh in, times the 88 % round trip, is 8.8 MWh out at $100.
    first = lp.solve(np.array([0.0, 100.0]))
    assert first.total_revenue == pytest.approx(100.0 * 10.0 * params.round_trip_efficiency, rel=1e-9)

    # Re-solve from an empty battery with a negative price second: it can only charge,
    # and a free terminal state of charge lets it end full and be paid for doing so.
    second = lp.solve(np.array([100.0, -10.0]), soc_initial_mwh=0.0, cyclic=False)
    assert second.p_dis == pytest.approx([0.0, 0.0], abs=1e-6)
    assert second.p_ch[1] == pytest.approx(10.0, abs=1e-6)
    assert second.total_revenue == pytest.approx(100.0, abs=1e-6)

    # And the first solve is reproducible after the bounds were changed.
    lp.solve(np.array([0.0, 100.0]), soc_initial_mwh=params.soc_initial_mwh, cyclic=True)


def test_rejects_mismatched_price_vector_lengths() -> None:
    params = BatteryParams(p_max_mw=10.0, duration_h=2.0)
    lp = DispatchLP(4, params, dt=DT)

    with pytest.raises(ValueError, match="energy_price must have shape"):
        lp.solve(np.zeros(3))
    with pytest.raises(ValueError, match="fcess_prices must have shape"):
        lp.solve(np.zeros(4), np.zeros((len(SERVICES), 3)))


def test_rejects_invalid_battery_parameters() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        BatteryParams(p_max_mw=0.0)
    with pytest.raises(ValueError, match="round_trip_efficiency"):
        BatteryParams(round_trip_efficiency=1.5)
    with pytest.raises(ValueError, match="state of charge fractions"):
        BatteryParams(soc_min_frac=0.6, soc_initial_frac=0.5)
    with pytest.raises(ValueError, match="unknown service"):
        BatteryParams(accredited_mw={"nonsense": 1.0})
