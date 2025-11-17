"""
Tests for Patron rule-based decision engine.
"""
import pytest
import numpy as np
from datetime import date, timedelta

from otonom_trader.patron.rules import make_decision_for_anomaly, calculate_trend
from otonom_trader.domain import Asset, AssetClass, Anomaly, AnomalyType, SignalType
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


@pytest.fixture
def test_asset_and_symbol(test_db):
    """Create test asset and symbol in database."""
    asset = Asset(
        symbol="TEST",
        name="Test Asset",
        asset_class=AssetClass.INDEX,
    )

    symbol = Symbol(symbol=asset.symbol, name=asset.name, asset_class=str(asset.asset_class))
    test_db.add(symbol)
    test_db.flush()

    return asset, symbol


def create_price_bars(test_db, symbol_id, start_date, prices):
    """Helper to create price bars in database."""
    for i, price in enumerate(prices):
        bar = DailyBar(
            symbol_id=symbol_id,
            date=start_date + timedelta(days=i),
            open=price * 0.99,
            high=price * 1.01,
            low=price * 0.98,
            close=price,
            volume=1000000.0,
        )
        test_db.add(bar)
    test_db.commit()


def test_calculate_trend_uptrend(test_db, test_asset_and_symbol):
    """Test trend calculation for uptrend."""
    asset, symbol = test_asset_and_symbol

    # Create uptrending prices
    start_date = date(2023, 1, 1)
    prices = np.linspace(100, 120, 30)  # 20% increase over 30 days

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Calculate trend
    ref_date = start_date + timedelta(days=30)
    trend = calculate_trend(test_db, symbol.id, ref_date, window=20)

    assert trend == "UP", "Should detect uptrend"


def test_calculate_trend_downtrend(test_db, test_asset_and_symbol):
    """Test trend calculation for downtrend."""
    asset, symbol = test_asset_and_symbol

    # Create downtrending prices
    start_date = date(2023, 1, 1)
    prices = np.linspace(120, 100, 30)  # 16.7% decrease over 30 days

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Calculate trend
    ref_date = start_date + timedelta(days=30)
    trend = calculate_trend(test_db, symbol.id, ref_date, window=20)

    assert trend == "DOWN", "Should detect downtrend"


def test_calculate_trend_flat(test_db, test_asset_and_symbol):
    """Test trend calculation for flat market."""
    asset, symbol = test_asset_and_symbol

    # Create flat prices with small noise
    start_date = date(2023, 1, 1)
    np.random.seed(42)
    prices = 100 + np.random.normal(0, 0.5, 30)  # Small variations around 100

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Calculate trend
    ref_date = start_date + timedelta(days=30)
    trend = calculate_trend(test_db, symbol.id, ref_date, window=20)

    assert trend == "FLAT", "Should detect flat trend"


def test_decision_spike_down_uptrend(test_db, test_asset_and_symbol):
    """Test BUY decision for SPIKE_DOWN in uptrend."""
    asset, symbol = test_asset_and_symbol

    # Create uptrending prices
    start_date = date(2023, 1, 1)
    prices = np.linspace(100, 120, 40)

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Create anomaly (spike down)
    anomaly_date = start_date + timedelta(days=35)
    anomaly = Anomaly(
        asset_symbol=asset.symbol,
        date=anomaly_date,
        anomaly_type=AnomalyType.SPIKE_DOWN,
        abs_return=0.05,
        zscore=-3.5,
        volume_rank=0.95,
    )

    # Make decision
    decision = make_decision_for_anomaly(test_db, anomaly, persist=False)

    assert decision.signal == SignalType.BUY, "Should generate BUY signal"
    assert decision.confidence >= 0.5, "Should have medium/high confidence"
    assert "uptrend" in decision.reason.lower(), "Reason should mention uptrend"


def test_decision_spike_up_downtrend(test_db, test_asset_and_symbol):
    """Test SELL decision for SPIKE_UP in downtrend."""
    asset, symbol = test_asset_and_symbol

    # Create downtrending prices
    start_date = date(2023, 1, 1)
    prices = np.linspace(120, 100, 40)

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Create anomaly (spike up)
    anomaly_date = start_date + timedelta(days=35)
    anomaly = Anomaly(
        asset_symbol=asset.symbol,
        date=anomaly_date,
        anomaly_type=AnomalyType.SPIKE_UP,
        abs_return=0.04,
        zscore=3.2,
        volume_rank=0.92,
    )

    # Make decision
    decision = make_decision_for_anomaly(test_db, anomaly, persist=False)

    assert decision.signal == SignalType.SELL, "Should generate SELL signal"
    assert decision.confidence >= 0.5, "Should have medium/high confidence"
    assert "downtrend" in decision.reason.lower(), "Reason should mention downtrend"


def test_decision_spike_down_downtrend(test_db, test_asset_and_symbol):
    """Test HOLD decision for SPIKE_DOWN in downtrend (avoid catching falling knife)."""
    asset, symbol = test_asset_and_symbol

    # Create downtrending prices
    start_date = date(2023, 1, 1)
    prices = np.linspace(120, 100, 40)

    create_price_bars(test_db, symbol.id, start_date, prices)

    # Create anomaly (spike down)
    anomaly_date = start_date + timedelta(days=35)
    anomaly = Anomaly(
        asset_symbol=asset.symbol,
        date=anomaly_date,
        anomaly_type=AnomalyType.SPIKE_DOWN,
        abs_return=0.06,
        zscore=-4.0,
        volume_rank=0.93,
    )

    # Make decision
    decision = make_decision_for_anomaly(test_db, anomaly, persist=False)

    assert decision.signal == SignalType.HOLD, "Should generate HOLD signal (avoid catching knife)"
    assert "downtrend" in decision.reason.lower() or "falling" in decision.reason.lower()


def test_decision_persistence(test_db, test_asset_and_symbol):
    """Test that decisions are persisted to database."""
    asset, symbol = test_asset_and_symbol

    # Create some price data
    start_date = date(2023, 1, 1)
    prices = np.linspace(100, 110, 40)
    create_price_bars(test_db, symbol.id, start_date, prices)

    # Create anomaly
    anomaly_date = start_date + timedelta(days=35)
    anomaly = Anomaly(
        asset_symbol=asset.symbol,
        date=anomaly_date,
        anomaly_type=AnomalyType.SPIKE_DOWN,
        abs_return=0.04,
        zscore=-3.0,
        volume_rank=0.90,
    )

    # Make decision with persistence
    decision = make_decision_for_anomaly(test_db, anomaly, persist=True)

    # Verify decision was saved
    from otonom_trader.data.schema import Decision as DecisionORM

    db_decision = test_db.query(DecisionORM).filter_by(symbol_id=symbol.id).first()

    assert db_decision is not None, "Decision should be persisted"
    assert db_decision.signal == str(decision.signal)
    assert db_decision.confidence == decision.confidence
    assert db_decision.date == decision.date
