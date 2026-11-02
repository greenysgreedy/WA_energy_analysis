"""Real-terms NPV, IRR, degradation and sensitivity analysis.

Everything is in **real 2025 Australian dollars**, so the discount rate is a real rate
and no inflation assumption is needed. Revenue is taken from the modelled realisable
dispatch, degradation is driven by the **modelled throughput** rather than an assumed
cycle count, and augmentation is costed at the future year's battery-cell price.

Results are reported as ranges. A single NPV from a single revenue year would be
meaningless here: the revenue available to a SWIS battery fell by roughly three quarters
between 2024 and 2026, so which year is annualised matters far more than the discount
rate does.

Cost sources
------------
Capital costs are CSIRO GenCost 2025-26 Final Report, Appendix Tables B.4 and B.5
(pages 73 to 74), Current policies scenario, 2025 values, in $/kWh. The report splits
total cost into a battery component and a balance-of-plant component, which is exactly
what augmentation costing needs: replacing cells late in life buys the battery component
at that year's price, not the whole plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy_financial as npf
import pandas as pd

#: CSIRO GenCost 2025-26 Final Report, Apx Table B.4 and B.5, Current policies, 2025.
#: Values are $/kWh of energy capacity; multiply by duration to get $/kW of power.
GENCOST_2025 = {
    1.0: {"total_per_kwh": 778.0, "battery_per_kwh": 310.0, "bop_per_kwh": 467.0},
    2.0: {"total_per_kwh": 525.0, "battery_per_kwh": 290.0, "bop_per_kwh": 235.0},
    4.0: {"total_per_kwh": 385.0, "battery_per_kwh": 265.0, "bop_per_kwh": 120.0},
    8.0: {"total_per_kwh": 308.0, "battery_per_kwh": 245.0, "bop_per_kwh": 63.0},
}

#: Projected battery-component cost, $/kWh, Current policies scenario, from the same
#: tables. Used to price mid-life augmentation at the year it happens.
GENCOST_BATTERY_COMPONENT = {
    1.0: {2025: 310, 2030: 256, 2035: 219, 2040: 190, 2045: 186, 2050: 183},
    2.0: {2025: 290, 2030: 239, 2035: 204, 2040: 178, 2045: 173, 2050: 170},
    4.0: {2025: 265, 2030: 218, 2035: 185, 2040: 162, 2045: 158, 2050: 155},
    8.0: {2025: 245, 2030: 201, 2035: 171, 2040: 150, 2045: 146, 2050: 143},
}

GENCOST_CITATION = (
    "CSIRO GenCost 2025-26 Final Report, Apx Tables B.4 and B.5 (pp. 73-74), "
    "Current policies scenario, 2025 values"
)


@dataclass(frozen=True)
class EconomicAssumptions:
    """Every economic input, with its source recorded in docs/assumptions.md."""

    p_max_mw: float = 100.0
    duration_h: float = 2.0

    #: Real pre-tax weighted average cost of capital. UNCONFIRMED: no published WEM
    #: storage WACC was found, so 7 % is used as a central value and varied over 5 to 10 %.
    real_wacc: float = 0.07
    project_life_years: int = 20

    #: Fixed operating cost as a share of initial capital cost per year. UNCONFIRMED:
    #: GenCost does not tabulate battery fixed O&M in a form that could be extracted, so
    #: 2 % is used and varied over 1 to 3 %.
    fom_share_of_capex: float = 0.02

    #: Variable operating cost per MWh of discharge, excluding degradation, which is
    #: handled separately through augmentation.
    vom_per_mwh: float = 1.0

    #: Annual revenue in real dollars, from the modelled realisable dispatch.
    annual_revenue: float = 0.0

    #: Annual Reserve Capacity revenue, zero for a merchant battery.
    annual_capacity_revenue: float = 0.0

    #: Equivalent full cycles per year from the modelled dispatch. Drives degradation.
    cycles_per_year: float = 300.0

    #: Capacity retained after this many equivalent full cycles, before augmentation.
    #: A conventional lithium iron phosphate warranty point.
    cycle_life: float = 6000.0
    end_of_life_capacity_fraction: float = 0.70

    #: Calendar fade per year, independent of cycling.
    calendar_fade_per_year: float = 0.01

    #: Augment back to nameplate energy whenever retained capacity falls below this.
    augmentation_trigger: float = 0.80
    allow_augmentation: bool = True

    #: Capital cost overrides, otherwise taken from GenCost.
    capex_per_kwh: float | None = None

    #: Connection and development cost on top of the plant, as a share of plant capex.
    development_share: float = 0.10

    #: Terminal value as a share of initial capital cost. Zero by default: a 20-year-old
    #: battery's residual value is not something this project can support a number for.
    terminal_value_share: float = 0.0

    extras: dict = field(default_factory=dict)

    # -- derived -----------------------------------------------------------------------

    @property
    def energy_mwh(self) -> float:
        return self.p_max_mw * self.duration_h

    @property
    def gencost_entry(self) -> dict:
        """Nearest tabulated duration in GenCost."""
        nearest = min(GENCOST_2025, key=lambda d: abs(d - self.duration_h))
        return GENCOST_2025[nearest]

    @property
    def plant_capex(self) -> float:
        per_kwh = self.capex_per_kwh or self.gencost_entry["total_per_kwh"]
        return per_kwh * self.energy_mwh * 1000.0

    @property
    def total_capex(self) -> float:
        return self.plant_capex * (1.0 + self.development_share)

    @property
    def annual_fom(self) -> float:
        return self.fom_share_of_capex * self.total_capex

    def battery_component_per_kwh(self, year_offset: int) -> float:
        """Battery-cell cost in a future year, interpolated from the GenCost projection."""
        nearest = min(GENCOST_BATTERY_COMPONENT, key=lambda d: abs(d - self.duration_h))
        curve = GENCOST_BATTERY_COMPONENT[nearest]
        years = np.array(sorted(curve))
        values = np.array([curve[y] for y in years], dtype=float)
        return float(np.interp(2025 + year_offset, years, values))


# --------------------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------------------


def capacity_trajectory(assumptions: EconomicAssumptions) -> pd.DataFrame:
    """Retained energy capacity year by year, and the augmentations it triggers.

    Degradation has a cycling term driven by the **modelled** throughput and a calendar
    term that applies regardless. Augmentation restores the nameplate energy and is
    costed at the battery-component price of the year it happens, which is why a
    four-hour battery's augmentation is cheaper per MWh than a one-hour battery's.
    """
    rows = []
    retained = 1.0
    cumulative_cycles = 0.0
    fade_per_cycle = (1.0 - assumptions.end_of_life_capacity_fraction) / assumptions.cycle_life

    for year in range(1, assumptions.project_life_years + 1):
        cumulative_cycles += assumptions.cycles_per_year
        retained -= fade_per_cycle * assumptions.cycles_per_year
        retained -= assumptions.calendar_fade_per_year
        retained = max(retained, 0.0)

        augmentation_cost = 0.0
        augmented = False
        if assumptions.allow_augmentation and retained < assumptions.augmentation_trigger:
            # Buy back the lost energy at the battery-component price of this year.
            lost_mwh = (1.0 - retained) * assumptions.energy_mwh
            augmentation_cost = (
                lost_mwh * 1000.0 * assumptions.battery_component_per_kwh(year)
            )
            retained = 1.0
            augmented = True

        rows.append(
            {
                "year": year,
                "retained_capacity": retained,
                "cumulative_cycles": cumulative_cycles,
                "augmented": augmented,
                "augmentation_cost": augmentation_cost,
            }
        )

    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------------------
# cash flow, NPV and IRR
# --------------------------------------------------------------------------------------


def cash_flows(assumptions: EconomicAssumptions) -> pd.DataFrame:
    """Real cash flows over the project life, year 0 being construction."""
    trajectory = capacity_trajectory(assumptions)

    rows = [
        {
            "year": 0,
            "retained_capacity": 1.0,
            "revenue": 0.0,
            "capacity_revenue": 0.0,
            "fom": 0.0,
            "vom": 0.0,
            "capex": -assumptions.total_capex,
            "augmentation": 0.0,
            "terminal_value": 0.0,
        }
    ]

    for year, row in trajectory.iterrows():
        # Energy and FCESS revenue scale with retained energy capacity; a degraded
        # battery simply has less to trade. Capacity revenue scales too, because
        # accreditation is based on sustainable energy (assumption C1).
        scale = row["retained_capacity"]
        terminal = (
            assumptions.terminal_value_share * assumptions.total_capex
            if year == assumptions.project_life_years
            else 0.0
        )
        rows.append(
            {
                "year": year,
                "retained_capacity": scale,
                "revenue": assumptions.annual_revenue * scale,
                "capacity_revenue": assumptions.annual_capacity_revenue * scale,
                "fom": -assumptions.annual_fom,
                "vom": -assumptions.vom_per_mwh
                * assumptions.cycles_per_year
                * assumptions.energy_mwh
                * scale,
                "capex": 0.0,
                "augmentation": -row["augmentation_cost"],
                "terminal_value": terminal,
            }
        )

    frame = pd.DataFrame(rows).set_index("year")
    frame["net_cash_flow"] = frame[
        ["revenue", "capacity_revenue", "fom", "vom", "capex", "augmentation", "terminal_value"]
    ].sum(axis=1)
    return frame


def evaluate(assumptions: EconomicAssumptions) -> dict:
    """NPV, IRR and the headline per-MW metrics for one set of assumptions."""
    flows = cash_flows(assumptions)
    net = flows["net_cash_flow"].to_numpy(dtype=float)

    npv = float(npf.npv(assumptions.real_wacc, net))
    try:
        irr = float(npf.irr(net))
    except Exception:  # noqa: BLE001 - npf raises various things on no sign change
        irr = np.nan
    if not np.isfinite(irr):
        irr = np.nan

    discount = (1.0 + assumptions.real_wacc) ** -np.arange(len(net))
    discounted_revenue = float(
        ((flows["revenue"] + flows["capacity_revenue"]).to_numpy() * discount).sum()
    )

    return {
        "npv": npv,
        "npv_per_mw": npv / assumptions.p_max_mw,
        "irr": irr,
        "total_capex": assumptions.total_capex,
        "capex_per_kw": assumptions.total_capex / (assumptions.p_max_mw * 1000.0),
        "discounted_revenue": discounted_revenue,
        "augmentations": int(flows["augmentation"].lt(0).sum()),
        "augmentation_cost": float(-flows["augmentation"].sum()),
        "payback_year": _payback_year(net),
        "levelised_revenue_requirement": _levelised_requirement(assumptions, flows),
    }


def _payback_year(net: np.ndarray) -> float:
    cumulative = np.cumsum(net)
    positive = np.flatnonzero(cumulative > 0)
    return float(positive[0]) if len(positive) else np.nan


def _levelised_requirement(assumptions: EconomicAssumptions, flows: pd.DataFrame) -> float:
    """Annual revenue, in $/MW-year, that would bring the NPV to zero.

    More useful than the NPV itself for a reader who wants to know how far away viability
    is: it can be compared directly with the modelled revenue per MW-year.
    """
    costs = flows[["fom", "vom", "capex", "augmentation", "terminal_value"]].sum(axis=1)
    discount = (1.0 + assumptions.real_wacc) ** -np.arange(len(flows))
    present_cost = -float((costs.to_numpy() * discount).sum())

    scaling = flows["retained_capacity"].to_numpy().copy()
    scaling[0] = 0.0
    annuity = float((scaling * discount).sum())
    if annuity <= 0:
        return np.nan
    return present_cost / annuity / assumptions.p_max_mw


# --------------------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------------------


def tornado(
    base: EconomicAssumptions,
    variations: dict[str, tuple[float, float]],
    *,
    metric: str = "npv_per_mw",
) -> pd.DataFrame:
    """One-at-a-time sensitivity of ``metric`` to each named field.

    ``variations`` maps a field name to its low and high value. Fields are set directly,
    so the low and high values are absolute, not multipliers, which keeps the chart
    readable when a parameter's plausible range is asymmetric.
    """
    baseline = evaluate(base)[metric]
    rows = []

    for field_name, (low, high) in variations.items():
        low_value = evaluate(replace(base, **{field_name: low}))[metric]
        high_value = evaluate(replace(base, **{field_name: high}))[metric]
        rows.append(
            {
                "parameter": field_name,
                "low_input": low,
                "high_input": high,
                "low": low_value,
                "high": high_value,
                "baseline": baseline,
                "swing": abs(high_value - low_value),
                "low_delta": low_value - baseline,
                "high_delta": high_value - baseline,
            }
        )

    return pd.DataFrame(rows).sort_values("swing", ascending=True).set_index("parameter")


def scenario_grid(
    base: EconomicAssumptions,
    revenue_cases: dict[str, float],
    capacity_cases: dict[str, float] | None = None,
    durations: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> pd.DataFrame:
    """Evaluate every duration against every revenue case.

    Revenue cases are annual revenue per MW; they are scaled by the power rating here so
    a caller can pass the modelled ``$/MW-year`` figures directly.
    """
    capacity_cases = capacity_cases or {"merchant": 0.0}
    rows = []

    for duration in durations:
        for revenue_label, revenue_per_mw in revenue_cases.items():
            for capacity_label, capacity_per_mw in capacity_cases.items():
                assumptions = replace(
                    base,
                    duration_h=duration,
                    annual_revenue=revenue_per_mw * base.p_max_mw,
                    annual_capacity_revenue=capacity_per_mw * base.p_max_mw,
                )
                result = evaluate(assumptions)
                rows.append(
                    {
                        "duration_h": duration,
                        "revenue_case": revenue_label,
                        "capacity_case": capacity_label,
                        "revenue_per_mw_year": revenue_per_mw,
                        "capacity_revenue_per_mw_year": capacity_per_mw,
                        "capex_per_kw": result["capex_per_kw"],
                        "npv_per_mw": result["npv_per_mw"],
                        "irr": result["irr"],
                        "breakeven_revenue_per_mw_year": result["levelised_revenue_requirement"],
                        "augmentations": result["augmentations"],
                    }
                )

    return pd.DataFrame(rows)
