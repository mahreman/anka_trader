"""
Data providers for price, news, and macro data.
"""
from .base import DataProvider
from .binance_provider import BinanceProvider
from .yfinance_provider import YFinanceProvider
from .newsapi_provider import NewsAPIProvider
from .fred_provider import FREDProvider

__all__ = [
    "DataProvider",
    "BinanceProvider",
    "YFinanceProvider",
    "NewsAPIProvider",
    "FREDProvider",
]
