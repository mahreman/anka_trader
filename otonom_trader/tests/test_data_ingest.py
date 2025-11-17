"""
Tests for data ingestion functionality.
"""
import pytest
from datetime import date, timedelta
import pandas as pd

from otonom_trader.domain import Asset, AssetClass
from otonom_trader.data.ingest import fetch_daily_ohlcv, upsert_daily_bars
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
def test_asset():
    """Create a test asset (S&P 500)."""
    return Asset(
        symbol="^GSPC",
        name="S&P 500 Test",
        asset_class=AssetClass.INDEX,
        base_currency="USD",
    )


def test_fetch_daily_ohlcv(test_asset):
    """Test fetching OHLCV data from yfinance."""
    # Fetch last 30 days of data
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    df = fetch_daily_ohlcv(test_asset, start_date, end_date, retry=False)

    # Verify DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 10, "Should fetch at least 10 rows of data"

    # Verify columns
    required_cols = ["date", "open", "high", "low", "close", "volume", "adj_close"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Verify data types
    assert df["date"].dtype == object  # date objects
    assert df["close"].dtype in [float, "float64"]
    assert df["volume"].dtype in [float, "float64", int, "int64"]

    # Verify data sanity
    assert df["close"].notna().all(), "Close prices should not be NaN"
    assert (df["high"] >= df["low"]).all(), "High should be >= Low"
    assert (df["high"] >= df["close"]).all(), "High should be >= Close"
    assert (df["low"] <= df["close"]).all(), "Low should be <= Close"


def test_upsert_daily_bars(test_db, test_asset):
    """Test upserting daily bars to database."""
    # Create sample data
    data = {
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [103.0, 104.0, 105.0],
        "volume": [1000000.0, 1100000.0, 1200000.0],
        "adj_close": [103.0, 104.0, 105.0],
    }
    df = pd.DataFrame(data)

    # Upsert data
    count = upsert_daily_bars(df, test_asset, test_db)

    assert count == 3, "Should insert 3 bars"

    # Verify symbol was created
    symbol = test_db.query(Symbol).filter_by(symbol=test_asset.symbol).first()
    assert symbol is not None
    assert symbol.name == test_asset.name

    # Verify bars were inserted
    bars = test_db.query(DailyBar).filter_by(symbol_id=symbol.id).all()
    assert len(bars) == 3

    # Test idempotency - upsert again
    count2 = upsert_daily_bars(df, test_asset, test_db)
    assert count2 == 3, "Should update 3 bars"

    # Verify still only 3 bars (no duplicates)
    bars = test_db.query(DailyBar).filter_by(symbol_id=symbol.id).all()
    assert len(bars) == 3


def test_upsert_updates_existing(test_db, test_asset):
    """Test that upsert updates existing bars correctly."""
    # Initial data
    data1 = {
        "date": [date(2023, 1, 1)],
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [103.0],
        "volume": [1000000.0],
        "adj_close": [103.0],
    }
    df1 = pd.DataFrame(data1)
    upsert_daily_bars(df1, test_asset, test_db)

    # Updated data (same date, different values)
    data2 = {
        "date": [date(2023, 1, 1)],
        "open": [101.0],
        "high": [106.0],
        "low": [100.0],
        "close": [104.0],
        "volume": [1100000.0],
        "adj_close": [104.0],
    }
    df2 = pd.DataFrame(data2)
    upsert_daily_bars(df2, test_asset, test_db)

    # Verify bar was updated, not duplicated
    symbol = test_db.query(Symbol).filter_by(symbol=test_asset.symbol).first()
    bars = test_db.query(DailyBar).filter_by(symbol_id=symbol.id).all()

    assert len(bars) == 1, "Should have only 1 bar (updated, not duplicated)"
    assert bars[0].close == 104.0, "Close price should be updated"
    assert bars[0].volume == 1100000.0, "Volume should be updated"
