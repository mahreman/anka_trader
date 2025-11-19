"""
Dummy price provider for testing and fallback.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any
import random

from .base import PriceProvider, OHLCVBar, Quote

logger = logging.getLogger(__name__)


class DummyPriceProvider(PriceProvider):
    """
    Dummy price provider that generates synthetic data.

    Useful for:
    - Testing
    - Fallback when real providers fail
    - Development without API keys
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("DummyPriceProvider initialized (returns synthetic data)")

    def _ensure_datetime(self, value: date | datetime) -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.combine(value, datetime.min.time())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _interval_delta(self, interval: str) -> timedelta:
        interval = (interval or "1d").lower()
        try:
            qty = int(interval[:-1])
            unit = interval[-1]
        except ValueError:
            qty = 1
            unit = "d"

        if unit == "m":
            return timedelta(minutes=qty)
        if unit == "h":
            return timedelta(hours=qty)
        if unit == "w":
            return timedelta(weeks=qty)
        return timedelta(days=qty)

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date | datetime,
        end_date: date | datetime,
        interval: str = "1d",
    ) -> List[OHLCVBar]:
        """Generate synthetic OHLCV data."""
        logger.warning(f"DummyPriceProvider returning synthetic data for {symbol}")

        bars = []
        current_dt = self._ensure_datetime(start_date)
        end_dt = self._ensure_datetime(end_date)
        step = self._interval_delta(interval)
        price = 100.0  # Starting price

        while current_dt <= end_dt:
            # Generate random OHLCV
            open_price = price
            close_price = price * (1 + random.uniform(-0.03, 0.03))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
            volume = random.uniform(1000000, 10000000)

            bar = OHLCVBar(
                symbol=symbol,
                date=current_dt,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                adj_close=close_price,
            )

            bars.append(bar)

            current_dt += step
            price = close_price

        return bars

    def fetch_latest_quote(self, symbol: str) -> Quote:
        """Generate synthetic quote."""
        logger.warning(f"DummyPriceProvider returning synthetic quote for {symbol}")

        last_price = random.uniform(90.0, 110.0)

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=last_price * 0.999,
            ask=last_price * 1.001,
            last=last_price,
            volume=random.uniform(1000000, 10000000),
        )

    def get_supported_symbols(self) -> List[str]:
        """Return dummy symbol list."""
        return ["DUMMY-USD", "TEST-USD"]
