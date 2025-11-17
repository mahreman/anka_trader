"""
Data ingestion - Fetch and store OHLCV data.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..domain import Asset
from .schema import Symbol, DailyBar
from .symbols import get_p0_assets
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


def fetch_daily_ohlcv(
    asset: Asset, start: date, end: date, retry: bool = True
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for an asset using yfinance.

    Args:
        asset: Asset object with symbol
        start: Start date
        end: End date
        retry: Whether to retry on failure

    Returns:
        DataFrame with columns: [date, open, high, low, close, adj_close, volume]

    Raises:
        Exception: If data fetch fails after retries
    """
    logger.info(f"Fetching data for {asset.symbol} from {start} to {end}")

    def _fetch():
        ticker = yf.Ticker(asset.symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {asset.symbol}")

        # Rename columns to lowercase and standardize
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Select and rename required columns
        required_cols = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        # Handle adj close if available
        if "adj close" in df.columns:
            df["adj_close"] = df["adj close"]
        else:
            df["adj_close"] = df["close"]  # Fallback to close

        # Select columns
        final_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        df = df[final_cols].copy()

        # Convert date to date type (remove timestamp)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Drop rows with missing critical values
        df = df.dropna(subset=["close", "volume"])

        logger.info(f"Fetched {len(df)} rows for {asset.symbol}")
        return df

    if retry:
        return retry_with_backoff(_fetch, max_retries=3)
    else:
        return _fetch()


def upsert_daily_bars(df: pd.DataFrame, asset: Asset, session: Session) -> int:
    """
    Insert or update daily bars in database (idempotent).

    Args:
        df: DataFrame with OHLCV data
        asset: Asset object
        session: Database session

    Returns:
        Number of rows inserted/updated
    """
    logger.info(f"Upserting {len(df)} bars for {asset.symbol}")

    # Get or create symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=asset.symbol).first()
    if symbol_obj is None:
        symbol_obj = Symbol(
            symbol=asset.symbol,
            name=asset.name,
            asset_class=str(asset.asset_class),
        )
        session.add(symbol_obj)
        session.flush()  # Get the ID
        logger.info(f"Created new symbol: {asset.symbol}")

    count = 0
    for _, row in df.iterrows():
        # Check if bar already exists
        existing = (
            session.query(DailyBar)
            .filter_by(symbol_id=symbol_obj.id, date=row["date"])
            .first()
        )

        if existing:
            # Update existing bar
            existing.open = float(row["open"])
            existing.high = float(row["high"])
            existing.low = float(row["low"])
            existing.close = float(row["close"])
            existing.volume = float(row["volume"])
            existing.adj_close = float(row["adj_close"]) if pd.notna(row["adj_close"]) else None
        else:
            # Insert new bar
            bar = DailyBar(
                symbol_id=symbol_obj.id,
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adj_close=float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
            )
            session.add(bar)

        count += 1

    session.commit()
    logger.info(f"Upserted {count} bars for {asset.symbol}")
    return count


def ingest_all_assets(
    session: Session, start: date, end: date, assets: Optional[list[Asset]] = None
) -> dict[str, int]:
    """
    Ingest data for all P0 assets.

    Args:
        session: Database session
        start: Start date
        end: End date
        assets: List of assets to ingest. If None, uses all P0 assets.

    Returns:
        Dictionary mapping symbol to number of bars ingested
    """
    if assets is None:
        assets = get_p0_assets()

    results = {}

    for asset in assets:
        try:
            df = fetch_daily_ohlcv(asset, start, end)
            count = upsert_daily_bars(df, asset, session)
            results[asset.symbol] = count
        except Exception as e:
            logger.error(f"Failed to ingest {asset.symbol}: {e}")
            results[asset.symbol] = 0

    return results


def get_latest_bar_date(session: Session, symbol: str) -> Optional[date]:
    """
    Get the latest date for which we have data for a symbol.

    Args:
        session: Database session
        symbol: Asset symbol

    Returns:
        Latest date or None if no data exists
    """
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
    if symbol_obj is None:
        return None

    latest = (
        session.query(func.max(DailyBar.date))
        .filter(DailyBar.symbol_id == symbol_obj.id)
        .scalar()
    )

    return latest


def ingest_incremental(
    session: Session,
    days_back: int = 7,
    assets: Optional[list[Asset]] = None,
) -> dict[str, int]:
    """
    Incremental data ingest: fetch only recent data.

    For each asset:
    - If data exists: fetch from latest_date to today
    - If no data: fetch last N days

    Args:
        session: Database session
        days_back: Number of days to look back if no data exists
        assets: List of assets to ingest. If None, uses all P0 assets.

    Returns:
        Dictionary mapping symbol to number of bars ingested
    """
    if assets is None:
        assets = get_p0_assets()

    today = date.today()
    results = {}

    logger.info(f"Running incremental ingest (days_back={days_back})")

    for asset in assets:
        try:
            # Get latest bar date
            latest_date = get_latest_bar_date(session, asset.symbol)

            if latest_date:
                # We have data: fetch from latest_date to today
                # Add 1 day to avoid re-fetching the same day
                start = latest_date + timedelta(days=1)

                # Skip if we're already up to date
                if start > today:
                    logger.info(f"{asset.symbol}: Already up to date")
                    results[asset.symbol] = 0
                    continue

                logger.info(
                    f"{asset.symbol}: Incremental update from {start} (latest was {latest_date})"
                )
            else:
                # No data: fetch last N days
                start = today - timedelta(days=days_back)
                logger.info(f"{asset.symbol}: No existing data, fetching {days_back} days")

            # Fetch and upsert
            df = fetch_daily_ohlcv(asset, start, today)
            count = upsert_daily_bars(df, asset, session)
            results[asset.symbol] = count

        except Exception as e:
            logger.error(f"Failed incremental ingest for {asset.symbol}: {e}")
            results[asset.symbol] = 0

    return results
