"""
Yahoo Finance data provider for stocks, FX, indices.
"""
import logging
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import DataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider for equities, FX, indices.

    Uses yfinance library to fetch historical OHLCV data.
    """

    def __init__(self, config: dict):
        """
        Initialize YFinance provider.

        Args:
            config: Configuration dictionary (yfinance doesn't need API keys)
        """
        super().__init__(config)

    def get_name(self) -> str:
        return "yfinance"

    def fetch_data(
        self,
        start: date,
        end: Optional[date] = None,
        symbol: str = "SPY",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            start: Start date
            end: End date (default: today)
            symbol: Ticker symbol (e.g., "SPY", "EURUSD=X")
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns: [date, open, high, low, close, adj_close, volume, symbol]
        """
        if end is None:
            end = date.today()

        logger.info(f"Fetching yfinance data for {symbol} from {start} to {end}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=False)

            if df.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()

            # Rename columns to lowercase and standardize
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Convert date to date type (remove timestamp)
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # Handle adj close
            if "adj close" in df.columns:
                df["adj_close"] = df["adj close"]
            else:
                df["adj_close"] = df["close"]

            # Select columns
            final_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
            df = df[final_cols].copy()

            # Drop rows with missing critical values
            df = df.dropna(subset=["close", "volume"])

            # Add symbol column
            df["symbol"] = symbol

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            raise
