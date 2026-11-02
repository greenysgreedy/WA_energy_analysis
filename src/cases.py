"""The headline battery cases, in one place so every stage runs the same definitions.

The brief sets 100 MW at 1, 2 and 4 hours. Each duration is run with and without
Capacity Credits, because holding them adds Reserve Capacity revenue but imposes the
ESR availability obligation, and the trade-off is not obvious in advance.

Parameter sources are recorded in ``docs/assumptions.md``: the FCESS accreditation
ceiling in F10, the loss factor in L4, the performance factor in F4, and the
efficiency, state-of-charge and throughput-cost conventions in M2 to M4.
"""

from __future__ import annotations

from dataclasses import replace

from src.model import SERVICES, BatteryParams

#: Headline power rating and durations from the brief.
HEADLINE_MW = 100.0
HEADLINE_DURATIONS_H = (1.0, 2.0, 4.0)

#: Share of the power rating a battery can be accredited for in each FCESS service.
#: The observed ceiling across the accredited SWIS fleet (docs/assumptions.md F10).
FCESS_ACCREDITATION_FRACTION = 0.5

#: No SWIS battery holds a RoCoF Control Service accreditation, so the hypothetical
#: battery does not offer it either (docs/assumptions.md F10).
OFFERED_SERVICES = tuple(s for s in SERVICES if s != "rocof")

#: Round-trip AC-to-AC efficiency for a modern lithium-ion grid battery. Varied in the
#: tornado chart over 0.82 to 0.92.
ROUND_TRIP_EFFICIENCY = 0.88

#: Throughput cost standing in for degradation inside the optimisation, $/MWh. Charged
#: on half of total throughput, so one full cycle is charged once. The economics stage
#: recomputes degradation from the modelled throughput rather than reusing this number.
THROUGHPUT_COST = 5.0

#: Usable state-of-charge window. Commercial batteries reserve headroom at both ends.
SOC_MIN_FRAC = 0.05
SOC_MAX_FRAC = 0.95

#: Hours of energy an enabled raise or lower quantity must be deliverable for. Chosen as
#: fifteen minutes, long enough to cover a contingency response and the subsequent
#: regulation recovery. Varied as a sensitivity.
FCESS_DELIVERY_HOURS = 0.25


def battery(
    duration_h: float,
    *,
    p_max_mw: float = HEADLINE_MW,
    has_capacity_credits: bool = False,
    loss_factor: float = 1.0,
    **overrides,
) -> BatteryParams:
    """One headline case."""
    accredited = {
        service: p_max_mw * FCESS_ACCREDITATION_FRACTION for service in OFFERED_SERVICES
    }
    params = BatteryParams(
        p_max_mw=p_max_mw,
        duration_h=duration_h,
        round_trip_efficiency=ROUND_TRIP_EFFICIENCY,
        soc_min_frac=SOC_MIN_FRAC,
        soc_max_frac=SOC_MAX_FRAC,
        soc_initial_frac=0.5,
        cyclic_soc=True,
        c_thru=THROUGHPUT_COST,
        loss_factor=loss_factor,
        accredited_mw=accredited,
        enabled_services=OFFERED_SERVICES,
        fcess_raise_duration_h=FCESS_DELIVERY_HOURS,
        fcess_lower_duration_h=FCESS_DELIVERY_HOURS,
        has_capacity_credits=has_capacity_credits,
    )
    return replace(params, **overrides) if overrides else params


def headline_cases() -> dict[str, BatteryParams]:
    """The six headline cases, keyed by a short readable label."""
    cases = {}
    for duration in HEADLINE_DURATIONS_H:
        hours = f"{duration:g}h"
        cases[f"{hours}_merchant"] = battery(duration, has_capacity_credits=False)
        cases[f"{hours}_credits"] = battery(duration, has_capacity_credits=True)
    return cases


#: How enablement is capped. See src/fcess_market.py for why the fleet-calibrated
#: ceiling is the central case.
FCESS_CAP_MODES = ("fleet", "requirement", "accreditation")


def accredited_mw(params: BatteryParams) -> dict[str, float]:
    """Accredited quantity per service for a case, as a plain dictionary."""
    return {service: params.accredited_for(service) for service in SERVICES}


def fcess_caps(dates, params: BatteryParams, mode: str = "fleet", *, scale: float = 1.0):
    """Per-interval enablement ceiling for a case, shaped ``(n_days, 5, 288)``.

    ``"fleet"``        what the operating storage fleet achieved per accredited MW
    ``"requirement"``  the published whole-market requirement, pro-rata by accreditation
    ``"accreditation"`` no cap beyond the accredited quantity, an upper bound only
    """
    from src import data, fcess_market

    own = accredited_mw(params)
    if mode == "accreditation":
        return None
    if mode == "requirement":
        return data.fcess_caps(dates, own)
    if mode == "fleet":
        return fcess_market.entrant_caps(dates, own, scale=scale)
    raise ValueError(f"unknown FCESS cap mode {mode!r}; expected one of {FCESS_CAP_MODES}")


def label_parts(label: str) -> tuple[float, bool]:
    """Split a case label such as ``"4h_credits"`` into duration and credit flag."""
    hours, regime = label.split("_")
    return float(hours.rstrip("h")), regime == "credits"
