"""
Unit tests for regime detection module.

Tests the regime detection and structural break analysis functionality
without requiring database access (uses in-memory data structures).
"""

import datetime as dt

import pandas as pd

from otonom_trader.analytics.regime import regimes_to_dataframe, RegimePoint


def test_regimes_to_dataframe_basic():
    """Test that regimes_to_dataframe creates DataFrame with correct shape and columns."""
    points = [
        RegimePoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 1),
            regime_id=0,
            volatility=0.1,
            trend=0.01,
            is_structural_break=False,
        ),
        RegimePoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 2),
            regime_id=1,
            volatility=0.2,
            trend=-0.02,
            is_structural_break=True,
        ),
    ]

    df = regimes_to_dataframe(points)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert "regime_id" in df.columns
    assert "volatility" in df.columns
    assert "trend" in df.columns
    assert "is_structural_break" in df.columns


def test_regimes_to_dataframe_structural_break_flag():
    """Test that structural break flag is preserved correctly."""
    points = [
        RegimePoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 1),
            regime_id=0,
            volatility=0.1,
            trend=0.01,
            is_structural_break=False,
        ),
        RegimePoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 2),
            regime_id=1,
            volatility=0.2,
            trend=-0.02,
            is_structural_break=True,
        ),
    ]

    df = regimes_to_dataframe(points)

    # Check that is_structural_break is True for second point
    assert df.loc[dt.date(2020, 1, 2)]["is_structural_break"] == True
    assert df.loc[dt.date(2020, 1, 1)]["is_structural_break"] == False


def test_regimes_to_dataframe_empty():
    """Test that empty list returns empty DataFrame."""
    points = []
    df = regimes_to_dataframe(points)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 0


def test_regime_point_dataclass():
    """Test RegimePoint dataclass construction."""
    point = RegimePoint(
        symbol="BTC-USD",
        trade_date=dt.date(2023, 1, 1),
        regime_id=2,
        volatility=0.25,
        trend=0.05,
        is_structural_break=True,
    )

    assert point.symbol == "BTC-USD"
    assert point.trade_date == dt.date(2023, 1, 1)
    assert point.regime_id == 2
    assert point.volatility == 0.25
    assert point.trend == 0.05
    assert point.is_structural_break is True
