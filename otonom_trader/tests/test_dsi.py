"""
Unit tests for Data Health Index (DSI) module.

Tests DSI computation and dataframe conversion functionality
without requiring database access.
"""

import datetime as dt

from otonom_trader.analytics.dsi import DsiPoint, dsi_to_dataframe


def test_dsi_to_dataframe_range_and_columns():
    """Test that DSI values are in valid range [0, 1] and DataFrame has correct columns."""
    points = [
        DsiPoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 1),
            dsi=0.95,
            missing_ratio=0.0,
            outlier_ratio=0.0,
            volume_jump_ratio=0.0,
        ),
        DsiPoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 2),
            dsi=0.40,
            missing_ratio=0.2,
            outlier_ratio=0.1,
            volume_jump_ratio=0.1,
        ),
    ]

    df = dsi_to_dataframe(points)

    assert df.shape[0] == 2
    assert "dsi" in df.columns
    assert "missing_ratio" in df.columns
    assert "outlier_ratio" in df.columns
    assert "volume_jump_ratio" in df.columns

    # DSI should be in [0, 1]
    assert df["dsi"].min() >= 0.0
    assert df["dsi"].max() <= 1.0


def test_dsi_dataclass_construction():
    """Test DsiPoint dataclass construction."""
    point = DsiPoint(
        symbol="BTC-USD",
        trade_date=dt.date(2023, 6, 15),
        dsi=0.85,
        missing_ratio=0.05,
        outlier_ratio=0.02,
        volume_jump_ratio=0.03,
    )

    assert point.symbol == "BTC-USD"
    assert point.trade_date == dt.date(2023, 6, 15)
    assert point.dsi == 0.85
    assert point.missing_ratio == 0.05
    assert point.outlier_ratio == 0.02
    assert point.volume_jump_ratio == 0.03


def test_dsi_to_dataframe_empty():
    """Test that empty list returns empty DataFrame."""
    points = []
    df = dsi_to_dataframe(points)

    assert df.shape[0] == 0
    assert "dsi" in df.columns


def test_dsi_boundary_values():
    """Test DSI boundary values (0 and 1)."""
    points = [
        DsiPoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 1),
            dsi=1.0,  # Perfect data quality
            missing_ratio=0.0,
            outlier_ratio=0.0,
            volume_jump_ratio=0.0,
        ),
        DsiPoint(
            symbol="TEST",
            trade_date=dt.date(2020, 1, 2),
            dsi=0.0,  # Worst data quality
            missing_ratio=0.5,
            outlier_ratio=0.3,
            volume_jump_ratio=0.2,
        ),
    ]

    df = dsi_to_dataframe(points)

    assert df.loc[dt.date(2020, 1, 1)]["dsi"] == 1.0
    assert df.loc[dt.date(2020, 1, 2)]["dsi"] == 0.0


def test_dsi_penalty_components():
    """Test that penalty components (missing, outlier, volume_jump) are preserved."""
    point = DsiPoint(
        symbol="TEST",
        trade_date=dt.date(2020, 1, 1),
        dsi=0.70,
        missing_ratio=0.10,
        outlier_ratio=0.05,
        volume_jump_ratio=0.03,
    )

    df = dsi_to_dataframe([point])

    assert df.loc[dt.date(2020, 1, 1)]["missing_ratio"] == 0.10
    assert df.loc[dt.date(2020, 1, 1)]["outlier_ratio"] == 0.05
    assert df.loc[dt.date(2020, 1, 1)]["volume_jump_ratio"] == 0.03
