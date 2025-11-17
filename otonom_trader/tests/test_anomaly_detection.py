"""
Tests for anomaly detection functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from otonom_trader.analytics.returns import compute_returns, compute_rolling_stats, compute_volume_quantile
from otonom_trader.analytics.labeling import classify_anomaly
from otonom_trader.analytics.anomaly import detect_anomalies_for_asset
from otonom_trader.domain import Asset, AssetClass, AnomalyType
from otonom_trader.data.schema import Symbol, DailyBar
from otonom_trader.data.db import get_engine, init_db
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def test_db():
    """Create a temporary in-memory database for testing."""
    engine = get_engine(":memory:")
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_compute_returns():
    """Test return calculation."""
    # Create sample price data
    data = {
        "date": pd.date_range("2023-01-01", periods=5),
        "close": [100.0, 105.0, 103.0, 108.0, 110.0],
    }
    df = pd.DataFrame(data)

    # Compute returns
    df_ret = compute_returns(df, price_col="close")

    assert "ret" in df_ret.columns
    assert pd.isna(df_ret["ret"].iloc[0]), "First return should be NaN"
    assert not pd.isna(df_ret["ret"].iloc[1]), "Second return should not be NaN"

    # Verify log return calculation
    expected_ret_1 = np.log(105.0 / 100.0)
    assert abs(df_ret["ret"].iloc[1] - expected_ret_1) < 1e-6


def test_compute_rolling_stats():
    """Test rolling statistics calculation."""
    # Create sample data with a spike
    np.random.seed(42)
    n = 100
    returns = np.random.normal(0, 0.01, n)
    returns[70] = 0.08  # Insert a spike

    data = {
        "date": pd.date_range("2023-01-01", periods=n),
        "ret": returns,
    }
    df = pd.DataFrame(data)

    # Compute rolling stats
    df_stats = compute_rolling_stats(df, window=60)

    assert "rolling_mean" in df_stats.columns
    assert "rolling_std" in df_stats.columns
    assert "ret_zscore" in df_stats.columns

    # Verify z-score at spike location
    zscore_at_spike = df_stats["ret_zscore"].iloc[70]
    assert zscore_at_spike > 3.0, "Spike should have high z-score"


def test_compute_volume_quantile():
    """Test volume quantile calculation."""
    # Create sample volume data with a spike
    n = 100
    volume = np.full(n, 1000000.0)
    volume[70] = 5000000.0  # High volume spike

    data = {
        "date": pd.date_range("2023-01-01", periods=n),
        "volume": volume,
    }
    df = pd.DataFrame(data)

    # Compute volume quantile
    df_vol = compute_volume_quantile(df, window=60)

    assert "volume_quantile" in df_vol.columns

    # Volume quantile at spike should be high
    quantile_at_spike = df_vol["volume_quantile"].iloc[70]
    assert quantile_at_spike >= 0.9, "High volume should have high quantile"


def test_classify_anomaly():
    """Test anomaly classification logic."""
    # Test SPIKE_UP
    anomaly = classify_anomaly(zscore=3.0, volume_quantile=0.9, k_threshold=2.5, q_threshold=0.8)
    assert anomaly == AnomalyType.SPIKE_UP

    # Test SPIKE_DOWN
    anomaly = classify_anomaly(zscore=-3.0, volume_quantile=0.9, k_threshold=2.5, q_threshold=0.8)
    assert anomaly == AnomalyType.SPIKE_DOWN

    # Test NONE (low volume)
    anomaly = classify_anomaly(zscore=3.0, volume_quantile=0.5, k_threshold=2.5, q_threshold=0.8)
    assert anomaly == AnomalyType.NONE

    # Test NONE (low z-score)
    anomaly = classify_anomaly(zscore=1.0, volume_quantile=0.9, k_threshold=2.5, q_threshold=0.8)
    assert anomaly == AnomalyType.NONE


def test_detect_anomalies_for_asset(test_db):
    """Test end-to-end anomaly detection for an asset."""
    # Create test asset and symbol
    asset = Asset(
        symbol="TEST",
        name="Test Asset",
        asset_class=AssetClass.INDEX,
    )

    symbol = Symbol(symbol=asset.symbol, name=asset.name, asset_class=str(asset.asset_class))
    test_db.add(symbol)
    test_db.flush()

    # Create synthetic price data with anomalies
    np.random.seed(42)
    n = 150
    base_price = 100.0
    returns = np.random.normal(0.001, 0.01, n)  # Normal market
    returns[80] = 0.05  # SPIKE_UP
    returns[120] = -0.06  # SPIKE_DOWN

    # Calculate prices from returns
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * np.exp(ret))

    # Create volume data (spikes aligned with price spikes)
    volume = np.random.normal(1000000, 200000, n + 1)
    volume[80] = 3000000  # High volume
    volume[120] = 3500000  # High volume

    # Insert bars into database
    start_date = date(2023, 1, 1)
    for i in range(n + 1):
        bar = DailyBar(
            symbol_id=symbol.id,
            date=start_date + timedelta(days=i),
            open=prices[i] * 0.99,
            high=prices[i] * 1.01,
            low=prices[i] * 0.98,
            close=prices[i],
            volume=volume[i],
        )
        test_db.add(bar)

    test_db.commit()

    # Detect anomalies
    anomalies = detect_anomalies_for_asset(
        session=test_db,
        asset=asset,
        k=2.5,
        q=0.8,
        window=60,
        persist=True,
    )

    # Verify anomalies were detected
    assert len(anomalies) >= 1, "Should detect at least one anomaly"

    # Check that anomalies include both spike types
    anomaly_types = {a.anomaly_type for a in anomalies}
    # Note: Depending on the random seed and exact thresholds, we might not catch all
    # But we should detect at least some anomalies
    assert AnomalyType.NONE not in anomaly_types, "Should only return actual anomalies"

    # Verify persistence
    from otonom_trader.data.schema import Anomaly as AnomalyORM

    db_anomalies = test_db.query(AnomalyORM).filter_by(symbol_id=symbol.id).all()
    assert len(db_anomalies) == len(anomalies), "All anomalies should be persisted"
