"""Perfect foresight and rolling-horizon dispatch strategies.

The question this module answers is how much of the theoretical maximum a battery could
realistically capture. Two things are needed for that to be an honest comparison:

* **Perfect foresight** solves each trading day knowing every price in advance. It is a
  genuine upper bound: no strategy can beat it on the same data, which is asserted in
  the tests and re-checked on real data in notebook 04.
* **Rolling horizon** re-optimises every half hour on a *forecast* of the next 24 hours,
  commits only the next half hour, and is then **settled on actual prices**. The forecast
  is deliberately naive, because a naive forecast is the honest floor: anything a trader
  does better than persistence is upside, not something this model should assume.

What is *not* forecast, and why:

* **FCESS enablement ceilings.** Whether a battery is enabled is AEMO's dispatch
  decision, not the operator's. The operator offers; the market clears. So the actual
  per-interval ceiling is used in every strategy, and the capture rate measures what the
  forecast costs on the *energy* side, which is where the operator's judgement bites.
* **ESR obligation windows.** AEMO publishes these a year ahead, so knowing them is not
  foresight.

Parameters are designed on 2024 and then frozen. :data:`FROZEN_CONFIG` records the
frozen values, and :func:`run_year` refuses to let a caller vary them for a test year.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.model import (
    SERVICES,
    BatteryParams,
    DispatchLP,
    DispatchResult,
    InfeasibleDispatch,
)
from src.paths import DT_HOURS, INTERVALS_PER_DAY_5MIN

DAY = INTERVALS_PER_DAY_5MIN


@dataclass(frozen=True)
class StrategyConfig:
    """Rolling-horizon parameters.

    horizon_intervals
        How far ahead the optimiser looks. One trading day by default, so the evening
        peak is always inside the window.
    commit_intervals
        How much of the solution is acted on before re-optimising. Six intervals is one
        thirty-minute Trading Interval, which is also the energy settlement period.
    forecast
        ``"persistence"``, ``"yesterday"`` or ``"last_week"``.
    """

    horizon_intervals: int = DAY
    commit_intervals: int = 6
    forecast: str = "persistence"

    def __post_init__(self) -> None:
        if self.commit_intervals < 1 or self.commit_intervals > self.horizon_intervals:
            raise ValueError("commit_intervals must be between 1 and horizon_intervals")
        if self.forecast not in FORECASTS:
            raise ValueError(f"unknown forecast {self.forecast!r}; expected one of {list(FORECASTS)}")


# --------------------------------------------------------------------------------------
# forecasts
# --------------------------------------------------------------------------------------


def _window(series: np.ndarray, start: int, length: int) -> np.ndarray:
    """Slice ``length`` values from ``start``, padding the tail with the last value."""
    end = start + length
    if start >= len(series):
        return np.full(length, series[-1] if len(series) else 0.0)
    taken = series[start:end]
    if len(taken) == length:
        return taken
    return np.concatenate([taken, np.full(length - len(taken), taken[-1])])


def forecast_persistence(actual: np.ndarray, start: int, length: int) -> np.ndarray:
    """The last observed price, held flat across the horizon."""
    last = actual[start - 1] if start > 0 else actual[0]
    return np.full(length, last)


def forecast_yesterday(actual: np.ndarray, start: int, length: int) -> np.ndarray:
    """The same interval of the previous trading day."""
    if start < DAY:
        return forecast_persistence(actual, start, length)
    return _window(actual, start - DAY, length)


def forecast_last_week(actual: np.ndarray, start: int, length: int) -> np.ndarray:
    """The same interval of the same weekday a week earlier."""
    if start < 7 * DAY:
        return forecast_yesterday(actual, start, length)
    return _window(actual, start - 7 * DAY, length)


FORECASTS = {
    "persistence": forecast_persistence,
    "yesterday": forecast_yesterday,
    "last_week": forecast_last_week,
}

#: A control that is not a strategy: it forecasts perfectly. Used only to confirm that
#: the rolling-horizon machinery converges on the perfect-foresight bound, which
#: separates forecast error from any loss caused by committing in small steps.
def forecast_oracle(actual: np.ndarray, start: int, length: int) -> np.ndarray:
    return _window(actual, start, length)


FORECASTS["oracle"] = forecast_oracle


def forecast_trailing_mean(
    actual: np.ndarray, start: int, length: int, lookback: int = 7 * DAY
) -> np.ndarray:
    """The mean of the preceding ``lookback`` intervals, held flat across the horizon.

    Used for FCESS prices rather than for energy. FCESS clears at exactly zero in a
    large share of intervals, so a persistence forecast taken at such an interval
    predicts zero for the whole horizon and the battery declines to offer at all. That
    is not how a price taker behaves: it offers its available capacity and is paid
    whatever clears, because being enabled for a few MW out of a hundred costs it almost
    no headroom. A trailing mean is causal, uses no information from the future, and
    keeps the battery offering whenever the service has recently had any value.
    """
    if start <= 0:
        return np.full(length, float(actual[0]))
    window = actual[max(0, start - lookback) : start]
    return np.full(length, float(window.mean()))


#: Lookback for the FCESS price forecast.
FCESS_FORECAST_LOOKBACK = 7 * DAY

#: Naive forecasts, excluding the oracle control. These are the strategies reported.
NAIVE_FORECASTS = ("persistence", "yesterday", "last_week")

#: Parameters designed on 2024 and then applied unchanged to 2025 and 2026. Recorded
#: here so the no-tuning claim is auditable: any change to these values is a change to
#: this constant, visible in the repository history.
FROZEN_CONFIG = StrategyConfig(horizon_intervals=DAY, commit_intervals=6, forecast="persistence")

#: The year the strategy parameters were designed on. Test years must not be tuned.
DESIGN_YEAR = 2024


# --------------------------------------------------------------------------------------
# flattening helpers
# --------------------------------------------------------------------------------------


def flatten(array: np.ndarray) -> np.ndarray:
    """Collapse per-day arrays into one continuous series.

    ``(n_days, 288)`` becomes ``(n_days * 288,)``; ``(n_days, 5, 288)`` becomes
    ``(5, n_days * 288)``.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        return array.reshape(-1)
    if array.ndim == 3:
        return array.transpose(1, 0, 2).reshape(array.shape[1], -1)
    raise ValueError(f"cannot flatten an array with {array.ndim} dimensions")


def _pad_columns(array: np.ndarray, start: int, length: int) -> np.ndarray:
    """Slice columns ``start:start+length`` from a 2-D array, padding with the last column."""
    end = start + length
    taken = array[:, start:end]
    if taken.shape[1] == length:
        return taken
    pad = np.repeat(taken[:, -1:], length - taken.shape[1], axis=1)
    return np.concatenate([taken, pad], axis=1)


# --------------------------------------------------------------------------------------
# perfect foresight
# --------------------------------------------------------------------------------------


def perfect_foresight(
    dates,
    energy: np.ndarray,
    fcess: np.ndarray,
    params: BatteryParams,
    *,
    esroi_masks: np.ndarray | None = None,
    fcess_caps: np.ndarray | None = None,
    dt: float = DT_HOURS,
) -> pd.DataFrame:
    """Solve each trading day independently with full knowledge of its prices.

    Each day is treated as self-contained, with the state of charge opening and closing
    at the same level, so no day borrows revenue from the next. That makes the bound
    slightly conservative, which is the right direction for an upper bound.
    """
    n_days = energy.shape[0]
    lp = DispatchLP(energy.shape[1], params, dt=dt)
    rows = []

    for index in range(n_days):
        lp.set_fcess_cap(None if fcess_caps is None else fcess_caps[index])
        lp.set_esroi_mask(None if esroi_masks is None else esroi_masks[index])
        result = lp.solve(energy[index], fcess[index])
        row = result.summary()
        row["trading_date"] = dates[index]
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("trading_date")
    frame.attrs["strategy"] = "perfect_foresight"
    return frame


# --------------------------------------------------------------------------------------
# rolling horizon
# --------------------------------------------------------------------------------------


@dataclass
class RollingOutcome:
    """Settled result of a rolling-horizon run."""

    daily: pd.DataFrame
    dispatch: pd.DataFrame
    solves: int
    obligation_breaches: int
    config: StrategyConfig

    @property
    def total_revenue(self) -> float:
        return float(self.daily["total_revenue"].sum())


def rolling_horizon(
    dates,
    energy: np.ndarray,
    fcess: np.ndarray,
    params: BatteryParams,
    config: StrategyConfig = FROZEN_CONFIG,
    *,
    esroi_masks: np.ndarray | None = None,
    fcess_caps: np.ndarray | None = None,
    dt: float = DT_HOURS,
) -> RollingOutcome:
    """Re-optimise on forecasts, commit the next few intervals, settle on actuals."""
    n_days, per_day = energy.shape
    horizon, commit = config.horizon_intervals, config.commit_intervals
    forecast = FORECASTS[config.forecast]

    actual_energy = flatten(energy)
    actual_fcess = flatten(fcess)
    caps = None if fcess_caps is None else flatten(fcess_caps)
    masks = None if esroi_masks is None else flatten(esroi_masks.astype(float)) > 0.5

    total = len(actual_energy)
    lp = DispatchLP(horizon, params, dt=dt)

    committed_ch = np.zeros(total)
    committed_dis = np.zeros(total)
    committed_enable = np.zeros((len(SERVICES), total))
    soc_trace = np.zeros(total + 1)

    soc = params.soc_initial_mwh
    soc_trace[0] = soc
    solves = 0
    breaches = 0

    for start in range(0, total, commit):
        lp.set_fcess_cap(None if caps is None else _pad_columns(caps, start, horizon))
        if masks is not None:
            window_mask = np.zeros(horizon, dtype=bool)
            taken = masks[start : start + horizon]
            window_mask[: len(taken)] = taken
            lp.set_esroi_mask(window_mask)

        forecast_energy = forecast(actual_energy, start, horizon)
        # FCESS prices get a trailing mean rather than the energy forecast rule; see
        # forecast_trailing_mean for why. The oracle control forecasts both perfectly.
        if config.forecast == "oracle":
            forecast_fcess = _pad_columns(actual_fcess, start, horizon)
        else:
            forecast_fcess = np.stack(
                [
                    forecast_trailing_mean(
                        actual_fcess[s], start, horizon, FCESS_FORECAST_LOOKBACK
                    )
                    for s in range(len(SERVICES))
                ]
            )

        try:
            result = lp.solve(forecast_energy, forecast_fcess, soc_initial_mwh=soc, cyclic=False)
        except InfeasibleDispatch:
            # The battery has carried forward too little energy to meet an obligation
            # window inside the horizon. A real operator would be in breach and would
            # keep dispatching; so does the model, with the breach recorded.
            breaches += 1
            lp.set_esroi_mask(None)
            result = lp.solve(forecast_energy, forecast_fcess, soc_initial_mwh=soc, cyclic=False)
            if masks is not None:
                window_mask = np.zeros(horizon, dtype=bool)
                taken = masks[start : start + horizon]
                window_mask[: len(taken)] = taken
                lp.set_esroi_mask(window_mask)
        solves += 1

        taken = min(commit, total - start)
        stop = start + taken
        committed_ch[start:stop] = result.p_ch[:taken]
        committed_dis[start:stop] = result.p_dis[:taken]
        for index, service in enumerate(SERVICES):
            committed_enable[index, start:stop] = result.enablement[service][:taken]

        soc_trace[start + 1 : stop + 1] = result.soc[1 : taken + 1]
        soc = float(result.soc[taken])

    daily, dispatch = _settle_committed(
        dates,
        per_day,
        actual_energy,
        actual_fcess,
        committed_ch,
        committed_dis,
        committed_enable,
        soc_trace,
        params,
        dt=dt,
    )
    daily.attrs["strategy"] = config.forecast
    return RollingOutcome(
        daily=daily,
        dispatch=dispatch,
        solves=solves,
        obligation_breaches=breaches,
        config=config,
    )


def _settle_committed(
    dates,
    per_day: int,
    actual_energy: np.ndarray,
    actual_fcess: np.ndarray,
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    enablement: np.ndarray,
    soc_trace: np.ndarray,
    params: BatteryParams,
    *,
    dt: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Value a committed dispatch at actual prices, aggregated by trading day."""
    n_days = len(dates)

    def per_day_sum(series: np.ndarray) -> np.ndarray:
        return series.reshape(n_days, per_day).sum(axis=1)

    energy_revenue = per_day_sum(actual_energy * (p_dis - p_ch)) * dt * params.loss_factor
    throughput_cost = per_day_sum(p_ch + p_dis) * dt / 2.0 * params.c_thru

    fcess_revenue = {}
    for index, service in enumerate(SERVICES):
        fcess_revenue[service] = (
            per_day_sum(actual_fcess[index] * enablement[index]) * dt * params.performance_factor
        )

    fcess_total = sum(fcess_revenue.values())
    discharge = per_day_sum(p_dis) * dt
    charge = per_day_sum(p_ch) * dt

    daily = pd.DataFrame(
        {
            "total_revenue": energy_revenue + fcess_total - throughput_cost,
            "energy_revenue": energy_revenue,
            "fcess_revenue": fcess_total,
            "throughput_cost": throughput_cost,
            "discharge_mwh": discharge,
            "charge_mwh": charge,
            "cycles": (discharge + charge) / (2.0 * params.energy_mwh),
            **{f"fcess_{service}": values for service, values in fcess_revenue.items()},
        },
        index=pd.Index(dates, name="trading_date"),
    )

    dispatch = pd.DataFrame(
        {
            "p_ch_mw": p_ch,
            "p_dis_mw": p_dis,
            "soc_mwh": soc_trace[:-1],
            "energy_price": actual_energy,
            **{f"{service}_mw": enablement[i] for i, service in enumerate(SERVICES)},
        }
    )
    return daily, dispatch


# --------------------------------------------------------------------------------------
# capture rates
# --------------------------------------------------------------------------------------


def capture_rate(realisable: pd.DataFrame, bound: pd.DataFrame, freq: str = "YS") -> pd.DataFrame:
    """Realisable revenue as a share of the perfect-foresight bound.

    Aggregated before dividing, because a day-by-day ratio is meaningless when the bound
    for a flat day is near zero.
    """
    aligned = bound.index.intersection(realisable.index)
    numerator = realisable.loc[aligned, "total_revenue"].resample(freq).sum()
    denominator = bound.loc[aligned, "total_revenue"].resample(freq).sum()

    out = pd.DataFrame(
        {
            "realisable": numerator,
            "perfect_foresight": denominator,
            "capture_rate": numerator / denominator,
        }
    )
    return out


def check_bound(realisable: pd.DataFrame, bound: pd.DataFrame, *, tolerance: float = 1.0) -> dict:
    """Confirm the perfect-foresight bound is not beaten.

    Perfect foresight closes the state of charge every day while the rolling horizon
    carries it across days, so an individual day can exceed the bound by borrowing from
    its neighbour. The period total cannot, and that is what is checked.
    """
    aligned = bound.index.intersection(realisable.index)
    realisable_total = float(realisable.loc[aligned, "total_revenue"].sum())
    bound_total = float(bound.loc[aligned, "total_revenue"].sum())
    return {
        "days": len(aligned),
        "realisable_total": realisable_total,
        "bound_total": bound_total,
        "bound_holds": realisable_total <= bound_total + tolerance,
        "days_above_bound": int(
            (realisable.loc[aligned, "total_revenue"] > bound.loc[aligned, "total_revenue"] + tolerance).sum()
        ),
    }


def frozen_for(year: int, forecast: str) -> StrategyConfig:
    """The frozen configuration, with only the forecast rule allowed to vary.

    Guards the no-tuning claim: a test year cannot be given different horizon or commit
    parameters from the design year, because this is the only way the pipeline builds a
    configuration for a run.
    """
    del year  # the configuration is deliberately independent of the year
    return replace(FROZEN_CONFIG, forecast=forecast)
