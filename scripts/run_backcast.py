"""Rebuild every SWIS battery's revenue and compare it with the model.

Writes ``outputs/backcast_daily.parquet``, ``outputs/backcast_summary.csv`` and
``outputs/backcast_model_comparison.csv``.

For each battery the dispatch model is re-run at that battery's own MW and MWh over the
same trading days, under perfect foresight and under the frozen rolling-horizon
strategy, so the comparison is like for like.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backcast, cases, data, loss_factors, strategies  # noqa: E402
from src.model import BatteryParams  # noqa: E402
from src.paths import OUTPUTS  # noqa: E402


def model_for(result: backcast.FacilityRevenue) -> BatteryParams:
    """A model battery matched to a real facility's rating, energy and accreditation."""
    register = data.load_fcess_accreditation().set_index("facility_code")
    accredited = {}
    mapping = {
        "contingency_raise": "max_contingency_raise_mw",
        "contingency_lower": "max_contingency_lower_mw",
        "regulation_raise": "max_regulation_raise_mw",
        "regulation_lower": "max_regulation_lower_mw",
    }
    for service, column in mapping.items():
        value = register[column].get(result.facility_code, np.nan)
        accredited[service] = 0.0 if pd.isna(value) else float(value)

    rating = result.rating_mw
    duration = result.duration_estimate_h or 1.0
    factor = loss_factors.loss_factor_for(result.facility_code, "2025-01-01")

    return BatteryParams(
        p_max_mw=rating,
        duration_h=duration,
        round_trip_efficiency=cases.ROUND_TRIP_EFFICIENCY,
        soc_min_frac=cases.SOC_MIN_FRAC,
        soc_max_frac=cases.SOC_MAX_FRAC,
        soc_initial_frac=0.5,
        c_thru=cases.THROUGHPUT_COST,
        loss_factor=factor,
        accredited_mw=accredited,
        enabled_services=tuple(mapping),
        fcess_raise_duration_h=cases.FCESS_DELIVERY_HOURS,
        fcess_lower_duration_h=cases.FCESS_DELIVERY_HOURS,
        has_capacity_credits=False,
    )


def compare_with_model(result: backcast.FacilityRevenue) -> dict:
    """Re-run the model at this facility's rating over its own operating window."""
    window = result.restricted()
    if window.empty:
        return {}

    start, end = window.index.min(), window.index.max()
    prices = data.complete_trading_days(data.load_prices(start=start, end=end))
    if prices.empty:
        return {}

    dates, energy, fcess = data.price_arrays(prices)
    params = model_for(result)

    # The facility's own accredited quantities are used, and the enablement ceiling is
    # what it actually achieved, so the model is not given a market share the facility
    # did not have.
    actual = data.load_schedule(facilities=[result.facility_code], start=start, end=end)
    caps = _observed_caps(actual, dates)

    bound = strategies.perfect_foresight(dates, energy, fcess, params, fcess_caps=caps)
    realisable = strategies.rolling_horizon(
        dates, energy, fcess, params, strategies.frozen_for(2024, "yesterday"), fcess_caps=caps
    )

    aligned = window.loc[window.index.isin(bound.index)]
    actual_total = float(aligned["total_revenue"].sum())
    bound_total = float(bound["total_revenue"].sum())

    return {
        "days_compared": len(aligned),
        "actual_total": actual_total,
        "actual_energy": float(aligned["energy_revenue"].sum()),
        "actual_fcess": float(aligned["fcess_revenue"].sum()),
        "model_perfect_foresight": bound_total,
        "model_pf_energy": float(bound["energy_revenue"].sum()),
        "model_pf_fcess": float(bound["fcess_revenue"].sum()),
        "model_realisable": realisable.total_revenue,
        "actual_over_pf": actual_total / bound_total if bound_total else np.nan,
        "model_capture_rate": realisable.total_revenue / bound_total if bound_total else np.nan,
        "actual_over_model_realisable": (
            actual_total / realisable.total_revenue if realisable.total_revenue else np.nan
        ),
    }


def _observed_caps(schedule: pd.DataFrame, dates) -> np.ndarray | None:
    """The facility's own observed enablement, reshaped as a per-interval ceiling."""
    if schedule.empty:
        return None

    from src.model import SERVICES
    from src.paths import INTERVALS_PER_DAY_5MIN

    wanted = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    out = np.zeros((len(wanted), len(SERVICES), INTERVALS_PER_DAY_5MIN))
    position = {date: row for row, date in enumerate(wanted)}

    for date, group in schedule.groupby("trading_date", observed=True):
        if date not in position or len(group) != INTERVALS_PER_DAY_5MIN:
            continue
        group = group.sort_values("interval_start")
        row = position[date]
        for index, service in enumerate(SERVICES):
            column = f"{service}_mw"
            if column in group.columns:
                out[row, index] = group[column].to_numpy(dtype=float)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true", help="rebuild revenue only")
    args = parser.parse_args()

    started = time.perf_counter()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)

    print("rebuilding revenue from SCADA, schedules and prices\n")
    fleet = backcast.rebuild_fleet()

    frames = []
    for code, result in fleet.items():
        frame = result.daily.copy()
        frame["facility_code"] = code
        frames.append(frame.reset_index())
    daily = pd.concat(frames, ignore_index=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUTS / "backcast_daily.parquet", index=False, compression="zstd")

    summary = backcast.fleet_summary(fleet)
    summary.to_csv(OUTPUTS / "backcast_summary.csv")
    print(summary.round(2).to_string())

    if not args.skip_model:
        print("\nre-running the model at each facility's own rating\n")
        rows = []
        for code, result in fleet.items():
            comparison = compare_with_model(result)
            if comparison:
                comparison["facility_code"] = code
                rows.append(comparison)
                print(
                    f"  {code:18s} actual ${comparison['actual_total'] / 1e6:6.1f}m  "
                    f"model PF ${comparison['model_perfect_foresight'] / 1e6:6.1f}m  "
                    f"actual/PF {comparison['actual_over_pf']:5.1%}",
                    flush=True,
                )
        if rows:
            comparison_frame = pd.DataFrame(rows).set_index("facility_code")
            comparison_frame.to_csv(OUTPUTS / "backcast_model_comparison.csv")

    fcess_totals = backcast.fleet_fcess_total_by_year(fleet)
    fcess_totals.to_csv(OUTPUTS / "backcast_fcess_by_year.csv")
    print("\nmodelled FCESS payments to storage, by calendar year:")
    print((fcess_totals / 1e6).round(2).to_string())

    print(f"\ntotal time {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
