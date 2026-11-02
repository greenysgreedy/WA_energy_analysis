"""Rebuild what the real SWIS batteries actually earned, and compare with the model.

This is the validation step. If the model's realisable strategy comfortably beats what
every real battery achieved, either a constraint is missing or operators are optimising
for something the model does not represent. Either way the model is wrong until the gap
is explained, and the explanation goes in ``docs/results.md`` — the model is never tuned
to close it.

Revenue is rebuilt from first principles rather than read from any published total:

* **energy** — net sent-out MWh per Trading Interval, loss factor adjusted, valued at
  the Reference Trading Price (ESM Rules 9.9.4, 9.5.2). Charging is negative, so a
  charging interval is a cost (9.5.5).
* **FCESS** — the enablement quantity in each five-minute Dispatch Interval valued at
  that interval's clearing price (9.10.6). No loss factor applies.
* **capacity** — Capacity Credits held multiplied by the published Reserve Capacity
  Price for the capacity year (assumption C5).

Energy quantities come from SCADA, which is metered output; enablement comes from the
market schedule, which is the dispatch engine's enablement quantity (assumption F7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import data, loss_factors
from src.model import SERVICES
from src.paths import DT_HOURS

#: A facility counts as commissioned once a trading day shows throughput above this
#: share of its registered energy. Commissioning tests are short and intermittent, so a
#: threshold on daily throughput separates them from normal operation better than a
#: threshold on peak power does.
COMMISSIONING_THROUGHPUT_FRACTION = 0.25

#: A month counts as a full operating month if at least this share of its trading days
#: clear the commissioning threshold.
FULL_MONTH_DAY_FRACTION = 0.8


@dataclass
class FacilityRevenue:
    """Rebuilt revenue for one facility."""

    facility_code: str
    daily: pd.DataFrame
    registered_mw: float
    first_full_month: pd.Period | None
    usable_mwh_estimate: float
    duration_estimate_h: float
    duration_source: str

    def restricted(self) -> pd.DataFrame:
        """Daily rows from the first full operating month onwards."""
        if self.first_full_month is None:
            return self.daily.iloc[0:0]
        start = self.first_full_month.to_timestamp()
        return self.daily.loc[self.daily.index >= start]


# --------------------------------------------------------------------------------------
# duration
# --------------------------------------------------------------------------------------


def discharge_runs(series: pd.Series) -> list[float]:
    """Energy in each unbroken run of discharging, in MWh."""
    values = series.to_numpy(dtype=float)
    runs: list[float] = []
    total = 0.0
    for value in values:
        if value > 0:
            total += value
        elif total > 0:
            runs.append(total)
            total = 0.0
    if total > 0:
        runs.append(total)
    return runs


def estimate_usable_energy(scada: pd.Series, quantile: float = 0.98) -> float:
    """Usable energy implied by the largest sustained discharges observed.

    A high quantile of the per-run discharge energy rather than the maximum, so one
    anomalous run does not set the estimate. The result is a *lower* bound on usable
    energy, because a battery is not obliged to discharge fully.
    """
    runs = discharge_runs(scada)
    if not runs:
        return 0.0
    return float(np.quantile(runs, quantile))


def duration_from_capacity_credits(
    facility_code: str, capacity_year: str, efficiency_dis: float = 0.938
) -> float | None:
    """Usable energy implied by the Capacity Credits held.

    Inverts assumption C1: credits are the sent-out capacity sustainable over the
    four-hour ESR Obligation Duration, so usable energy is credits x 4 / efficiency.
    """
    credits = data.capacity_credits_for(facility_code)
    if capacity_year not in credits.index:
        return None
    return float(credits.loc[capacity_year]) * 4.0 / efficiency_dis


# --------------------------------------------------------------------------------------
# revenue rebuild
# --------------------------------------------------------------------------------------


def rebuild_facility(
    facility_code: str,
    *,
    registered_mw: float,
    start: str | None = None,
    end: str | None = None,
) -> FacilityRevenue:
    """Rebuild one facility's daily energy and FCESS revenue."""
    scada = data.load_scada(facilities=[facility_code], start=start, end=end)
    schedule = data.load_schedule(facilities=[facility_code], start=start, end=end)
    prices = data.load_prices(start=start, end=end)

    if scada.empty:
        raise ValueError(f"no SCADA rows for {facility_code}")

    merged = scada.merge(
        prices[
            [
                "interval_start",
                "trading_interval",
                data.SETTLEMENT_PRICE_COLUMN,
                *data.FCESS_PRICE_COLUMNS,
            ]
        ],
        on="interval_start",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        schedule[["interval_start", *data.FCESS_SCHEDULE_COLUMNS]],
        on="interval_start",
        how="left",
    )
    merged[list(data.FCESS_SCHEDULE_COLUMNS)] = merged[
        list(data.FCESS_SCHEDULE_COLUMNS)
    ].fillna(0.0)

    # Loss factors change by financial year, so they are applied per trading day.
    merged["loss_factor"] = [
        loss_factors.loss_factor_for(facility_code, day) for day in merged["trading_date"]
    ]

    # Energy settles on the Trading Interval net quantity at the Reference Trading Price.
    by_interval = merged.groupby(["trading_date", "trading_interval"], observed=True).agg(
        net_mwh=("average_mwh", "sum"),
        price=(data.SETTLEMENT_PRICE_COLUMN, "first"),
        loss_factor=("loss_factor", "first"),
    )
    by_interval["energy_revenue"] = (
        by_interval["net_mwh"] * by_interval["loss_factor"] * by_interval["price"]
    )
    energy_daily = by_interval.groupby("trading_date", observed=True)["energy_revenue"].sum()

    fcess_daily = {}
    for index, service in enumerate(SERVICES):
        payment = (
            merged[f"{service}_mw"] * merged[f"{service}_price"].fillna(0.0) * DT_HOURS
        )
        fcess_daily[f"fcess_{service}"] = payment.groupby(merged["trading_date"]).sum()

    throughput = merged.groupby("trading_date", observed=True)["average_mwh"].agg(
        discharge_mwh=lambda s: float(s[s > 0].sum()),
        charge_mwh=lambda s: float(-s[s < 0].sum()),
        peak_mw=lambda s: float(s.max() / DT_HOURS),
    )

    daily = pd.concat([energy_daily, pd.DataFrame(fcess_daily), throughput], axis=1)
    daily["fcess_revenue"] = daily[[f"fcess_{s}" for s in SERVICES]].sum(axis=1)
    daily["total_revenue"] = daily["energy_revenue"] + daily["fcess_revenue"]
    daily["throughput_mwh"] = daily["discharge_mwh"] + daily["charge_mwh"]

    usable = estimate_usable_energy(merged.set_index("interval_start")["average_mwh"])
    duration = usable / registered_mw if registered_mw else 0.0
    source = "SCADA discharge runs"

    first_full = first_full_operating_month(daily, usable)

    return FacilityRevenue(
        facility_code=facility_code,
        daily=daily,
        registered_mw=registered_mw,
        first_full_month=first_full,
        usable_mwh_estimate=usable,
        duration_estimate_h=duration,
        duration_source=source,
    )


def first_full_operating_month(daily: pd.DataFrame, usable_mwh: float) -> pd.Period | None:
    """First month in which the facility operated normally on most trading days."""
    if daily.empty or usable_mwh <= 0:
        return None

    threshold = COMMISSIONING_THROUGHPUT_FRACTION * usable_mwh
    operating = daily["throughput_mwh"] >= threshold
    by_month = operating.groupby(daily.index.to_period("M")).mean()
    qualifying = by_month[by_month >= FULL_MONTH_DAY_FRACTION]
    return None if qualifying.empty else qualifying.index[0]


# --------------------------------------------------------------------------------------
# capacity revenue
# --------------------------------------------------------------------------------------


def capacity_revenue(facility_code: str) -> pd.DataFrame:
    """Capacity revenue per capacity year from credits held and the published price."""
    credits = data.capacity_credits_for(facility_code)
    rows = []
    for label, quantity in credits.items():
        start_year = int(str(label)[:4])
        price = data.reserve_capacity_price(start_year)
        rows.append(
            {
                "capacity_year": label,
                "capacity_credits_mw": quantity,
                "reserve_capacity_price": price,
                "capacity_revenue": None if price is None else quantity * price,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# fleet view
# --------------------------------------------------------------------------------------


def rebuild_fleet(start: str | None = None, end: str | None = None) -> dict[str, FacilityRevenue]:
    """Rebuild every storage facility that appears in SCADA."""
    register = data.storage_facilities()
    results = {}
    for row in register.itertuples():
        try:
            results[row.facility_code] = rebuild_facility(
                row.facility_code, registered_mw=float(row.registered_mw), start=start, end=end
            )
        except ValueError as exc:
            print(f"  skipped {row.facility_code}: {exc}")
    return results


def fleet_summary(fleet: dict[str, FacilityRevenue]) -> pd.DataFrame:
    """One row per facility, restricted to full operating months."""
    rows = []
    for code, result in fleet.items():
        window = result.restricted()
        if window.empty:
            rows.append(
                {
                    "facility_code": code,
                    "registered_mw": result.registered_mw,
                    "first_full_month": None,
                    "days": 0,
                }
            )
            continue

        years = len(window) / 365.25
        rows.append(
            {
                "facility_code": code,
                "registered_mw": result.registered_mw,
                "usable_mwh": result.usable_mwh_estimate,
                "duration_h": result.duration_estimate_h,
                "first_full_month": str(result.first_full_month),
                "days": len(window),
                "discharge_mwh": window["discharge_mwh"].sum(),
                "charge_mwh": window["charge_mwh"].sum(),
                "cycles": (
                    window["throughput_mwh"].sum() / (2 * result.usable_mwh_estimate)
                    if result.usable_mwh_estimate
                    else np.nan
                ),
                "energy_revenue": window["energy_revenue"].sum(),
                "fcess_revenue": window["fcess_revenue"].sum(),
                "total_revenue": window["total_revenue"].sum(),
                "revenue_per_mw_year": window["total_revenue"].sum() / result.registered_mw / years,
                "revenue_per_mwh_year": (
                    window["total_revenue"].sum() / result.usable_mwh_estimate / years
                    if result.usable_mwh_estimate
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("facility_code")


def fleet_fcess_total_by_year(fleet: dict[str, FacilityRevenue]) -> pd.DataFrame:
    """Modelled FCESS payments to storage by calendar year, for reconciliation."""
    frames = []
    for code, result in fleet.items():
        frame = result.daily[[f"fcess_{s}" for s in SERVICES]].copy()
        frame["facility_code"] = code
        frames.append(frame)

    combined = pd.concat(frames)
    combined["year"] = combined.index.year
    totals = combined.groupby("year")[[f"fcess_{s}" for s in SERVICES]].sum()
    totals["total"] = totals.sum(axis=1)
    return totals
