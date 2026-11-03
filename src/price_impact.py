"""Did batteries move SWIS prices, and would a new one be a price taker?

Two independent lines of evidence, because neither is conclusive alone.

**Event study.** How daily spreads, price duration curves and FCESS prices moved around
each battery's first full operating month. This measures what happened, but cannot
separate the batteries from everything else changing at the same time, so a same-period
previous-year comparison is reported alongside it as a crude control.

**Residual supply slope.** Post-reform Real-Time Market Submissions are not public —
`effective-balancing-submission` stops at September 2023 and the post-reform submissions
are available only to participants through WEMS, which this project does not use
(``docs/assumptions.md`` D3). So the supply curve cannot be read off observed offers.
Instead the local slope of price against dispatched quantity is estimated from the data
that *is* public, by first-differencing consecutive dispatch intervals: differencing
removes anything that moves slowly relative to five minutes, which is most confounders.
The result is an **estimated elasticity, not an observed offer curve**, and is labelled
as such wherever it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import data


@dataclass
class SlopeEstimate:
    """Local slope of price against dispatched quantity, in $/MWh per MW."""

    label: str
    slope: float
    standard_error: float
    observations: int
    price_band: tuple[float, float]
    r_squared: float

    @property
    def impact_of(self) -> float:
        """Price move, in $/MWh, from adding 100 MW of demand or supply."""
        return self.slope * 100.0

    def row(self) -> dict:
        return {
            "band": self.label,
            "price range": f"{self.price_band[0]:,.0f} to {self.price_band[1]:,.0f}",
            "slope $/MWh per MW": self.slope,
            "standard error": self.standard_error,
            "t": self.slope / self.standard_error if self.standard_error else np.nan,
            "r squared": self.r_squared,
            "intervals": self.observations,
            "impact of 100 MW, $/MWh": self.impact_of,
        }


# --------------------------------------------------------------------------------------
# residual supply slope
# --------------------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, standard error and R-squared of a through-the-origin-free simple OLS."""
    design = np.column_stack([np.ones_like(x), x])
    coefficients, residuals_sum, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    dof = max(len(x) - rank, 1)
    sigma_squared = float(residuals @ residuals) / dof
    covariance = sigma_squared * np.linalg.pinv(design.T @ design)
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else np.nan
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1])), r_squared


def estimate_supply_slope(
    start: str | None = None,
    end: str | None = None,
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    min_quantity_change: float = 5.0,
) -> pd.DataFrame:
    """Estimate the local price-quantity slope in each price band.

    The regression is of the change in settlement price between consecutive dispatch
    intervals on the change in dispatched energy requirement over the same pair. Pairs
    with a very small quantity change are dropped, because dividing a price move by a
    near-zero quantity move estimates nothing.
    """
    prices = data.load_prices(start=start, end=end)
    requirements = data.load_market_requirements()
    merged = prices[["interval_start", data.SETTLEMENT_PRICE_COLUMN]].merge(
        requirements[["interval_start", "energy_requirement"]],
        on="interval_start",
        how="inner",
    ).sort_values("interval_start")

    merged["d_price"] = merged[data.SETTLEMENT_PRICE_COLUMN].diff()
    merged["d_quantity"] = merged["energy_requirement"].diff()
    merged["level"] = merged[data.SETTLEMENT_PRICE_COLUMN]
    merged = merged.dropna(subset=["d_price", "d_quantity"])
    merged = merged.loc[merged["d_quantity"].abs() >= min_quantity_change]

    if bands is None:
        quantiles = merged["level"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).to_list()
        bands = {
            "lowest decile (charging)": (quantiles[0], quantiles[1]),
            "middle": (quantiles[1], quantiles[3]),
            "highest decile (discharging)": (quantiles[3], quantiles[4]),
            "all intervals": (quantiles[0], quantiles[4]),
        }

    rows = []
    for label, (low, high) in bands.items():
        subset = merged.loc[(merged["level"] >= low) & (merged["level"] <= high)]
        if len(subset) < 100:
            continue
        slope, error, r_squared = _ols(
            subset["d_quantity"].to_numpy(dtype=float), subset["d_price"].to_numpy(dtype=float)
        )
        rows.append(
            SlopeEstimate(
                label=label,
                slope=slope,
                standard_error=error,
                observations=len(subset),
                price_band=(low, high),
                r_squared=r_squared,
            ).row()
        )
    return pd.DataFrame(rows)


def price_taker_verdict(slopes: pd.DataFrame, *, threshold_share: float = 0.05) -> pd.DataFrame:
    """Is a 100 MW entrant a price taker?

    The test is whether the price move from adding 100 MW is small relative to the price
    level in that band. A move above ``threshold_share`` of the level means the entrant
    is eroding its own spread materially and is not a price taker.
    """
    out = slopes.copy()
    midpoints = []
    for band in out["price range"]:
        low, high = (float(v.replace(",", "")) for v in band.split(" to "))
        midpoints.append((low + high) / 2.0)
    out["band midpoint $/MWh"] = midpoints
    out["impact as share of level"] = (
        out["impact of 100 MW, $/MWh"].abs() / out["band midpoint $/MWh"].abs()
    )
    out["price taker"] = out["impact as share of level"] < threshold_share
    return out


# --------------------------------------------------------------------------------------
# event study
# --------------------------------------------------------------------------------------


def daily_metrics(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Per-trading-day price metrics a battery's revenue depends on."""
    prices = data.load_prices(start=start, end=end)
    grouped = prices.groupby("trading_date", observed=True)

    metrics = pd.DataFrame(
        {
            "spread": grouped[data.SETTLEMENT_PRICE_COLUMN].max()
            - grouped[data.SETTLEMENT_PRICE_COLUMN].min(),
            "mean_price": grouped[data.SETTLEMENT_PRICE_COLUMN].mean(),
            "negative_share": grouped[data.SETTLEMENT_PRICE_COLUMN].apply(lambda s: (s < 0).mean()),
        }
    )

    # The spread a 2-hour battery could reach: the best 24 intervals against the worst 24.
    def capturable(series: pd.Series) -> float:
        if len(series) < 48:
            return np.nan
        return float(series.nlargest(24).mean() - series.nsmallest(24).mean())

    metrics["capturable_spread_2h"] = grouped[data.SETTLEMENT_PRICE_COLUMN].apply(capturable)

    for column in data.FCESS_PRICE_COLUMNS:
        metrics[column] = grouped[column].mean()
    metrics["fcess_price_sum"] = metrics[list(data.FCESS_PRICE_COLUMNS)].sum(axis=1)
    return metrics


def commissioning_dates(fleet: dict) -> pd.Series:
    """First full operating month per facility, from the backcast."""
    return pd.Series(
        {
            code: result.first_full_month.to_timestamp()
            for code, result in fleet.items()
            if result.first_full_month is not None
        }
    ).sort_values()


def fleet_from_summary(path=None) -> pd.DataFrame:
    """The storage register from ``outputs/backcast_summary.csv``.

    The event study and the capacity series need only three columns per facility: when it
    started operating, its rating and its usable energy. Reading them from the summary the
    backcast already wrote avoids rebuilding every facility's revenue from 23 million rows
    of SCADA just to recover three numbers.
    """
    from src.paths import OUTPUTS

    path = path or OUTPUTS / "backcast_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/run_backcast.py' first."
        )

    summary = pd.read_csv(path, index_col=0)
    summary = summary.loc[summary["first_full_month"].notna()]
    summary = summary.loc[summary["first_full_month"] != "None"]
    summary["from"] = pd.PeriodIndex(summary["first_full_month"], freq="M").to_timestamp()
    return summary


def commissioning_dates_from_summary(path=None) -> pd.Series:
    """First full operating month per facility, read from the backcast summary."""
    summary = fleet_from_summary(path)
    return summary["from"].sort_values()


def cumulative_storage_capacity_from_summary(path=None) -> pd.DataFrame:
    """Operating storage MW and MWh over time, read from the backcast summary."""
    summary = fleet_from_summary(path).sort_values("from").reset_index()
    frame = summary[["facility_code", "from", "observed_peak_mw", "usable_mwh"]].copy()
    frame = frame.rename(columns={"observed_peak_mw": "rating_mw"})
    frame["cumulative_mw"] = frame["rating_mw"].cumsum()
    frame["cumulative_mwh"] = frame["usable_mwh"].cumsum()
    return frame


def event_study(
    metrics: pd.DataFrame,
    events: pd.Series,
    *,
    window_days: int = 90,
    columns: tuple[str, ...] = ("capturable_spread_2h", "spread", "fcess_price_sum"),
) -> pd.DataFrame:
    """Metrics in the windows before and after each event, with a previous-year control.

    The control is the same calendar window one year earlier. If the before-to-after
    change is similar in the control window, the change is seasonal rather than caused by
    the entrant.
    """
    rows = []
    for facility, date in events.items():
        date = pd.Timestamp(date)
        for label, offset in (("event", pd.Timedelta(0)), ("previous year", pd.Timedelta(days=-365))):
            centre = date + offset
            before = metrics.loc[centre - pd.Timedelta(days=window_days) : centre - pd.Timedelta(days=1)]
            after = metrics.loc[centre : centre + pd.Timedelta(days=window_days - 1)]
            if before.empty or after.empty:
                continue
            row = {"facility_code": facility, "window": label, "date": centre.date()}
            for column in columns:
                row[f"{column} before"] = before[column].median()
                row[f"{column} after"] = after[column].median()
                row[f"{column} change"] = after[column].median() - before[column].median()
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.set_index(["facility_code", "window"]).sort_index()


def cumulative_storage_capacity(fleet: dict) -> pd.DataFrame:
    """Operating storage MW over time, from each facility's first full operating month."""
    rows = []
    for code, result in fleet.items():
        if result.first_full_month is None:
            continue
        rows.append(
            {
                "facility_code": code,
                "from": result.first_full_month.to_timestamp(),
                "registered_mw": result.registered_mw,
                "usable_mwh": result.usable_mwh_estimate,
            }
        )
    frame = pd.DataFrame(rows).sort_values("from").reset_index(drop=True)
    frame["cumulative_mw"] = frame["registered_mw"].cumsum()
    frame["cumulative_mwh"] = frame["usable_mwh"].cumsum()
    return frame


def capacity_against_spread(metrics: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Monthly capturable spread beside operating storage capacity.

    The headline correlation for the price-impact question. A correlation over a period
    this short is suggestive, not causal, and is reported as such.
    """
    monthly = metrics.resample("MS").median()
    mw = pd.Series(0.0, index=monthly.index)
    mwh = pd.Series(0.0, index=monthly.index)

    # Column access by name rather than itertuples: "from" is a Python keyword, so
    # itertuples silently renames it to a positional label.
    ordered = capacity.sort_values("from")
    for start, cumulative_mw, cumulative_mwh in zip(
        ordered["from"], ordered["cumulative_mw"], ordered["cumulative_mwh"]
    ):
        mw.loc[mw.index >= start] = cumulative_mw
        mwh.loc[mwh.index >= start] = cumulative_mwh

    out = monthly[["capturable_spread_2h", "spread", "fcess_price_sum", "negative_share"]].copy()
    out["operating_storage_mw"] = mw
    out["operating_storage_mwh"] = mwh
    return out
