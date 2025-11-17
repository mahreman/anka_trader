"""
Base classes for data providers.

All data providers implement standardized interfaces for:
- Price data (OHLCV, quotes)
- News data (headlines, sentiment)
- Macro data (economic indicators)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(ProviderError):
    """Raised when API authentication fails."""
    pass


@dataclass
class OHLCVBar:
    """
    OHLCV price bar.

    Attributes:
        symbol: Asset symbol
        date: Bar date
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        adj_close: Adjusted close (optional)
    """
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close,
        }


@dataclass
class Quote:
    """
    Real-time quote.

    Attributes:
        symbol: Asset symbol
        timestamp: Quote timestamp
        bid: Bid price
        ask: Ask price
        last: Last trade price
        volume: 24h volume
    """
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float

    @property
    def mid(self) -> float:
        """Midpoint price."""
        return (self.bid + self.ask) / 2.0


@dataclass
class NewsArticle:
    """
    News article.

    Attributes:
        source: News source (e.g., "Reuters", "Bloomberg")
        title: Article title
        description: Article description/snippet
        url: Article URL
        published_at: Publication timestamp
        symbols: Related symbols (if available)
        sentiment: Sentiment score (-1 to +1, optional)
        author: Article author (optional)
    """
    source: str
    title: str
    description: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment: Optional[float] = None
    author: Optional[str] = None


@dataclass
class MacroIndicator:
    """
    Macroeconomic indicator value.

    Attributes:
        indicator_code: Indicator code (e.g., "GDP", "UNRATE", "DFF")
        name: Indicator name (e.g., "Unemployment Rate")
        date: Observation date
        value: Indicator value
        unit: Unit of measurement (e.g., "Percent", "Billions of Dollars")
        frequency: Data frequency (e.g., "Monthly", "Quarterly")
    """
    indicator_code: str
    name: str
    date: date
    value: float
    unit: str
    frequency: str


class PriceProvider(ABC):
    """
    Abstract base class for price data providers.

    Providers fetch OHLCV bars and real-time quotes from various sources:
    - Binance (crypto spot/futures)
    - Polygon (stocks, crypto, forex)
    - Alpha Vantage (stocks, forex)
    - yfinance (stocks, ETFs, indices)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider.

        Args:
            config: Provider configuration (API keys, base URLs, etc.)
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV bars for a symbol.

        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            interval: Time interval (e.g., "1d", "1h", "15m")

        Returns:
            List of OHLCV bars

        Raises:
            ProviderError: If fetch fails
            RateLimitError: If rate limit exceeded
        """
        pass

    @abstractmethod
    def fetch_latest_quote(self, symbol: str) -> Quote:
        """
        Fetch latest quote for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Latest quote

        Raises:
            ProviderError: If fetch fails
        """
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of symbol strings
        """
        pass


class NewsProvider(ABC):
    """
    Abstract base class for news data providers.

    Providers fetch news articles from various sources:
    - NewsAPI
    - Polygon News
    - Benzinga News API
    - Custom RSS feeds
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider.

        Args:
            config: Provider configuration (API keys, etc.)
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def fetch_news(
        self,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """
        Fetch news articles.

        Args:
            symbol: Filter by symbol (optional)
            query: Search query (optional)
            start_date: Start datetime (optional)
            end_date: End datetime (optional)
            limit: Max number of articles

        Returns:
            List of news articles

        Raises:
            ProviderError: If fetch fails
        """
        pass

    @abstractmethod
    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        """
        Fetch latest headlines.

        Args:
            symbol: Filter by symbol (optional)
            limit: Max number of headlines

        Returns:
            List of recent news articles
        """
        pass


class MacroProvider(ABC):
    """
    Abstract base class for macroeconomic data providers.

    Providers fetch economic indicators from:
    - FRED (Federal Reserve Economic Data)
    - World Bank
    - IMF
    - Trading Economics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider.

        Args:
            config: Provider configuration (API keys, etc.)
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def fetch_indicator(
        self,
        indicator_code: str,
        start_date: date,
        end_date: date,
    ) -> List[MacroIndicator]:
        """
        Fetch macroeconomic indicator time series.

        Args:
            indicator_code: Indicator code (e.g., "GDP", "UNRATE")
            start_date: Start date
            end_date: End date

        Returns:
            List of indicator observations

        Raises:
            ProviderError: If fetch fails
        """
        pass

    @abstractmethod
    def get_available_indicators(self) -> List[Dict[str, str]]:
        """
        Get list of available indicators.

        Returns:
            List of dicts with 'code', 'name', 'frequency'
        """
        pass

    @abstractmethod
    def fetch_latest_value(self, indicator_code: str) -> MacroIndicator:
        """
        Fetch latest value for an indicator.

        Args:
            indicator_code: Indicator code

        Returns:
            Latest indicator observation
        """
        pass
