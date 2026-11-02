"""The data access layer, against the real cleaned parquet.

Marked slow because it reads ``data/processed/``. Run with ``--run-slow`` after cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data
from src.paths import INTERVALS_PER_DAY_5MIN

pytestmark = pytest.mark.slow


def test_prices_carry_a_settlement_price_for_every_interval() -> None:
    prices = data.load_prices()
    assert not prices.empty
    assert prices[data.SETTLEMENT_PRICE_COLUMN].notna().all()


def test_settlement_price_is_constant_within_a_trading_interval() -> None:
    """Energy settles on the thirty-minute price, so it must not vary within the half hour."""
    prices = data.load_prices(start="2024-06-01", end="2024-06-07")
    spread = prices.groupby("trading_interval", observed=True)[data.SETTLEMENT_PRICE_COLUMN].nunique()
    assert (spread == 1).all()


def test_settlement_price_matches_the_mean_of_its_five_minute_prices() -> None:
    """ESM Rules 7.11A.1(b): the Reference Trading Price is the time-weighted average."""
    prices = data.load_prices(start="2024-06-01", end="2024-06-30")
    grouped = prices.groupby("trading_interval", observed=True)
    derived = grouped[data.CLEARING_PRICE_COLUMN].mean()
    published = grouped[data.SETTLEMENT_PRICE_COLUMN].first()

    difference = (derived - published).abs()
    # A handful of intervals carry a Failure Reason and are revised, so allow a small
    # share of mismatches but require the bulk to agree to the cent.
    assert (difference < 0.01).mean() > 0.99


def test_complete_trading_days_yields_whole_days_only() -> None:
    prices = data.complete_trading_days(data.load_prices())
    counts = prices.groupby("trading_date", observed=True).size()
    assert (counts == INTERVALS_PER_DAY_5MIN).all()


def test_price_arrays_have_the_expected_shape() -> None:
    prices = data.complete_trading_days(data.load_prices(start="2024-01-01", end="2024-01-31"))
    dates, energy, fcess = data.price_arrays(prices)

    assert energy.shape == (len(dates), INTERVALS_PER_DAY_5MIN)
    assert fcess.shape == (len(dates), 5, INTERVALS_PER_DAY_5MIN)
    assert np.isfinite(energy).all()
    assert np.isfinite(fcess).all()


def test_storage_register_finds_the_swis_batteries() -> None:
    storage = data.storage_facilities()
    codes = set(storage["facility_code"])

    expected = {
        "ALINTA_WGP_ESR1",
        "COLLIE_BESS2",
        "COLLIE_ESR1",
        "COLLIE_ESR4",
        "COLLIE_ESR5",
        "KWINANA_ESR1",
        "KWINANA_ESR2",
        "MERREDIN_ESR1",
    }
    assert expected <= codes
    assert (storage["registered_mw"] > 0).all()


def test_esroi_masks_mark_eight_trading_intervals() -> None:
    """Every published window is eight Trading Intervals, so 48 dispatch intervals."""
    windows = data.esroi_windows()
    dates = windows["trading_date"].head(20)
    mask = data.esroi_masks(dates)

    assert mask.shape == (len(dates), INTERVALS_PER_DAY_5MIN)
    assert set(mask.sum(axis=1)) == {48}


def test_esroi_mask_sits_inside_the_trading_day() -> None:
    """The 2023-24 window is 16:30 to 20:00, which is 8.5 to 12 hours into the day."""
    mask = data.esroi_masks([pd.Timestamp("2023-10-02")])
    indices = np.flatnonzero(mask[0])

    # The trading day starts at 08:00, so 16:30 is interval (8.5 * 12) = 102.
    assert indices[0] == 102
    assert indices[-1] == 149  # 20:00 plus its half hour ends at interval 149


def test_reserve_capacity_price_is_published_for_the_study_years() -> None:
    for year in (2023, 2024, 2025, 2026):
        price = data.reserve_capacity_price(year)
        assert price is not None and price > 0


def test_schedule_fcess_columns_are_never_null() -> None:
    schedule = data.load_schedule(
        facilities=["KWINANA_ESR1"], start="2024-06-01", end="2024-06-07"
    )
    assert not schedule.empty
    assert schedule[list(data.FCESS_SCHEDULE_COLUMNS)].notna().all().all()


def test_scada_peaks_are_consistent_with_registered_capacity() -> None:
    """D5: average_mwh is MWh per interval, so MW is twelve times the column."""
    scada = data.load_scada(facilities=["COLLIE_G1"], start="2024-01-01", end="2024-03-31")
    peak_mw = scada["average_mwh"].max() * 12

    registered = data.load_facilities().set_index("facility_code").loc["COLLIE_G1", "registered_mw"]
    assert 0.9 < peak_mw / float(registered) < 1.1
