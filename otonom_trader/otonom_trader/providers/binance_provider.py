"""
Binance data provider for crypto OHLCV data.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .base import DataProvider

logger = logging.getLogger(__name__)


class BinanceProvider(DataProvider):
    """
    Binance data provider for crypto OHLCV.

    Uses Binance public API to fetch historical klines (candlestick) data.
    """

    def __init__(self, config: dict):
        """
        Initialize Binance provider.

        Args:
            config: Configuration dictionary with:
                - api_key: Binance API key (optional for public data)
                - api_secret: Binance API secret (optional for public data)
                - base_url: Binance API base URL (default: https://api.binance.com)
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.base_url = config.get("base_url", "https://api.binance.com")

    def get_name(self) -> str:
        return "binance"

    def fetch_data(
        self,
        start: date,
        end: Optional[date] = None,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.

        Args:
            start: Start date
            end: End date (default: today)
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (default: "1d")
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, symbol]
        """
        if end is None:
            end = date.today()

        logger.info(f"Fetching Binance data for {symbol} from {start} to {end}")

        # Convert dates to timestamps
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int(datetime.combine(end, datetime.max.time()).timestamp() * 1000)

        # Binance API endpoint
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,  # Max 1000 candles per request
        }

        all_data = []
        current_start = start_ts

        # Fetch data in chunks (Binance limits to 1000 candles per request)
        while current_start < end_ts:
            params["startTime"] = current_start

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)

                # Update start time for next batch
                # Last candle's close time + 1ms
                current_start = data[-1][6] + 1

                # If we got less than 1000, we're done
                if len(data) < 1000:
                    break

            except requests.exceptions.RequestException as e:
                logger.error(f"Binance API error for {symbol}: {e}")
                raise

        if not all_data:
            logger.warning(f"No data returned from Binance for {symbol}")
            return pd.DataFrame()

        # Parse response
        # Binance kline format:
        # [
        #   [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
        #    number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
        # ]
        df = pd.DataFrame(
            all_data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
        )

        # Convert timestamps to dates
        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date

        # Select and convert required columns
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["symbol"] = symbol

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
