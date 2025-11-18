"""
yfinance price data provider.

Fetches OHLCV data from Yahoo Finance using yfinance library.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Dict, Any

import pandas as pd
import yfinance as yf

from .base import PriceProvider, OHLCVBar, Quote, ProviderError

logger = logging.getLogger(__name__)


class YFinanceProvider(PriceProvider):
    """
    Yahoo Finance price provider.

    Supports:
    - Stocks (e.g., AAPL, TSLA)
    - ETFs (e.g., SPY, QQQ)
    - Indices (e.g., ^GSPC, ^DJI)
    - Crypto (e.g., BTC-USD, ETH-USD)
    - Forex (e.g., EURUSD=X)
    - Futures (e.g., ES=F, GC=F)

    This is a wrapper around the existing yfinance integration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize yfinance provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        self.auto_adjust = config.get("extra", {}).get("auto_adjust", False)
        self.prepost = config.get("extra", {}).get("prepost", False)

        logger.info(f"YFinanceProvider initialized (auto_adjust={self.auto_adjust})")

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date | datetime,
        end_date: date | datetime,
        interval: str = "1d",
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Yahoo Finance symbol (e.g., "AAPL", "BTC-USD")
            start_date: Start date
            end_date: End date
            interval: Interval (1d, 1h, 1m, etc.)

        Returns:
            List of OHLCV bars

        Example:
            >>> provider = YFinanceProvider({})
            >>> bars = provider.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        """
        logger.info(f"Fetching yfinance data: {symbol} from {start_date} to {end_date}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.auto_adjust,
                prepost=self.prepost,
            )

            if df.empty:
                logger.warning(f"No yfinance data for {symbol}")
                return []

            # Convert DataFrame to OHLCVBar objects
            bars = []
            df = df.reset_index()

            # Normalize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            for _, row in df.iterrows():
                # Get timestamp (handle both Date and Datetime columns)
                if "date" in df.columns:
                    ts_value = row["date"]
                elif "datetime" in df.columns:
                    ts_value = row["datetime"]
                else:
                    logger.error(f"No date column found in yfinance data for {symbol}")
                    continue

                bar_ts = pd.to_datetime(ts_value, utc=True).to_pydatetime()

                # Create bar
                bar = OHLCVBar(
                    symbol=symbol,
                    date=bar_ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    adj_close=float(row.get("adj_close", row["close"])),
                )

                bars.append(bar)

            logger.info(f"Fetched {len(bars)} yfinance bars for {symbol}")

            return bars

        except Exception as e:
            raise ProviderError(f"yfinance fetch failed for {symbol}: {e}")

    def fetch_latest_quote(self, symbol: str) -> Quote:
        """
        Fetch latest quote from Yahoo Finance.

        Args:
            symbol: Yahoo Finance symbol

        Returns:
            Latest quote

        Example:
            >>> quote = provider.fetch_latest_quote("AAPL")
            >>> print(f"AAPL: ${quote.last:.2f}")
        """
        logger.info(f"Fetching yfinance quote: {symbol}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get latest price data
            # Try fast_info first (faster API)
            try:
                last_price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
            except:
                # Fallback to info
                last_price = info.get("regularMarketPrice") or info.get("currentPrice")

            if last_price is None:
                # Try fetching 1-day history as last resort
                df = ticker.history(period="1d")
                if not df.empty:
                    last_price = float(df["Close"].iloc[-1])
                else:
                    raise ProviderError(f"Could not get price for {symbol}")

            # Construct quote (yfinance doesn't provide bid/ask easily)
            quote = Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=last_price * 0.9999,  # Estimate bid as 0.01% below last
                ask=last_price * 1.0001,  # Estimate ask as 0.01% above last
                last=float(last_price),
                volume=float(info.get("regularMarketVolume", 0.0)),
            )

            logger.debug(f"yfinance quote: {symbol} = ${quote.last:.2f}")

            return quote

        except Exception as e:
            raise ProviderError(f"yfinance quote fetch failed for {symbol}: {e}")

    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Note: yfinance supports millions of symbols but doesn't provide
        a comprehensive list. Returns empty list.

        Returns:
            Empty list (yfinance doesn't enumerate symbols)
        """
        logger.warning("yfinance doesn't provide symbol enumeration")
        return []
