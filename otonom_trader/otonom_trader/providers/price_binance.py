"""
Binance price data provider.

Fetches OHLCV data and real-time quotes from Binance Spot API.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any

import requests

from .base import PriceProvider, OHLCVBar, Quote, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


class BinanceProvider(PriceProvider):
    """
    Binance Spot API price provider.

    Supports:
    - Historical OHLCV data (klines)
    - Real-time quotes (24hr ticker)
    - All Binance Spot symbols

    API Docs: https://binance-docs.github.io/apidocs/spot/en/
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance provider.

        Args:
            config: Provider configuration with:
                - base_url: Binance API URL
                - api_key: Optional API key
                - use_testnet: Whether to use testnet
        """
        super().__init__(config)

        self.base_url = config.get("base_url", "https://api.binance.com")
        self.api_key = config.get("api_key")
        self.use_testnet = config.get("extra", {}).get("use_testnet", False)

        # Use testnet URL if configured
        if self.use_testnet:
            self.base_url = "https://testnet.binance.vision"

        # Binance interval mapping
        self.interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
        }

        logger.info(f"BinanceProvider initialized (testnet={self.use_testnet})")

    def _request(self, endpoint: str, params: Dict = None) -> Any:
        """
        Make API request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response

        Raises:
            ProviderError: If request fails
            RateLimitError: If rate limited
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}

        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.get("timeout_seconds", 30),
            )

            # Check for rate limiting
            if response.status_code == 429:
                raise RateLimitError("Binance rate limit exceeded")

            # Check for errors
            if response.status_code != 200:
                error_msg = response.json().get("msg", "Unknown error")
                raise ProviderError(f"Binance API error: {error_msg}")

            return response.json()

        except requests.RequestException as e:
            raise ProviderError(f"Binance request failed: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for Binance API.

        Args:
            symbol: Symbol (e.g., "BTC-USD" or "BTCUSDT")

        Returns:
            Binance format (e.g., "BTCUSDT")
        """
        # Remove dashes
        symbol = symbol.replace("-", "").upper()

        # Convert USD to USDT for crypto
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            # Check if it's a crypto pair (not a stock)
            # Heuristic: if it's a short symbol like BTCUSD, it's crypto
            if len(symbol) <= 7:
                symbol = symbol.replace("USD", "USDT")

        return symbol

    def _to_timestamp_ms(self, value: date | datetime, is_end: bool) -> int:
        """Convert a date/datetime to Binance milliseconds."""

        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.combine(
                value,
                datetime.max.time() if is_end else datetime.min.time(),
            )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return int(dt.timestamp() * 1000)

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date | datetime,
        end_date: date | datetime,
        interval: str = "1d",
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV data from Binance.

        Args:
            symbol: Symbol (e.g., "BTC-USD" or "BTCUSDT")
            start_date: Start date
            end_date: End date
            interval: Interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)

        Returns:
            List of OHLCV bars

        Example:
            >>> provider = BinanceProvider({})
            >>> bars = provider.fetch_ohlcv("BTC-USD", date(2024, 1, 1), date(2024, 1, 31))
            >>> print(f"Fetched {len(bars)} bars")
        """
        binance_symbol = self._normalize_symbol(symbol)
        binance_interval = self.interval_map.get(interval, "1d")

        # Convert dates to milliseconds timestamps
        start_ms = self._to_timestamp_ms(start_date, is_end=False)
        end_ms = self._to_timestamp_ms(end_date, is_end=True)

        # Fetch klines (OHLCV)
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,  # Max 1000 per request
        }

        logger.info(f"Fetching Binance klines: {binance_symbol} {start_date} to {end_date}")

        # Binance may require multiple requests for large date ranges
        all_bars = []
        current_start = start_ms

        while current_start < end_ms:
            params["startTime"] = current_start

            try:
                data = self._request("/api/v3/klines", params)
            except ProviderError as e:
                logger.error(f"Failed to fetch Binance data: {e}")
                raise

            if not data:
                break

            # Parse klines
            for kline in data:
                # Binance kline format:
                # [open_time, open, high, low, close, volume, close_time, ...]
                bar_time = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)

                bar = OHLCVBar(
                    symbol=symbol,  # Use original symbol
                    date=bar_time,
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    adj_close=float(kline[4]),  # Binance doesn't have adj_close
                )

                all_bars.append(bar)

            # Move to next batch
            if len(data) < 1000:
                break  # No more data

            # Update start time to last kline's close time + 1ms
            current_start = data[-1][6] + 1

            # Rate limiting: sleep briefly between requests
            time.sleep(0.1)

        logger.info(f"Fetched {len(all_bars)} Binance bars for {symbol}")

        return all_bars

    def fetch_latest_quote(self, symbol: str) -> Quote:
        """
        Fetch latest quote from Binance.

        Args:
            symbol: Symbol (e.g., "BTC-USD")

        Returns:
            Latest quote

        Example:
            >>> quote = provider.fetch_latest_quote("BTC-USD")
            >>> print(f"BTC: ${quote.last:,.2f}")
        """
        binance_symbol = self._normalize_symbol(symbol)

        # Get 24hr ticker (includes bid, ask, last price, volume)
        params = {"symbol": binance_symbol}

        try:
            data = self._request("/api/v3/ticker/24hr", params)
        except ProviderError as e:
            logger.error(f"Failed to fetch Binance quote: {e}")
            raise

        # Parse ticker
        quote = Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=float(data.get("bidPrice", 0.0)),
            ask=float(data.get("askPrice", 0.0)),
            last=float(data.get("lastPrice", 0.0)),
            volume=float(data.get("volume", 0.0)),
        )

        logger.debug(f"Binance quote: {symbol} = ${quote.last:.2f}")

        return quote

    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols from Binance.

        Returns:
            List of symbol strings

        Example:
            >>> symbols = provider.get_supported_symbols()
            >>> print(f"Binance supports {len(symbols)} symbols")
        """
        try:
            data = self._request("/api/v3/exchangeInfo")
        except ProviderError as e:
            logger.error(f"Failed to fetch Binance exchange info: {e}")
            return []

        # Extract symbols from exchange info
        symbols = []
        for symbol_info in data.get("symbols", []):
            if symbol_info.get("status") == "TRADING":
                symbols.append(symbol_info["symbol"])

        logger.info(f"Binance supports {len(symbols)} symbols")

        return symbols
