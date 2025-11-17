"""
Data providers module - Integrate real data sources.

Supports multiple data types:
- Price data (OHLCV, real-time quotes)
- News data (headlines, sentiment)
- Macro data (economic indicators, interest rates)
"""

from .base import (
    PriceProvider,
    NewsProvider,
    MacroProvider,
    ProviderError,
    RateLimitError,
)
from .factory import create_price_provider, create_news_provider, create_macro_provider
from .config import ProviderConfig, get_provider_config

__all__ = [
    # Base classes
    "PriceProvider",
    "NewsProvider",
    "MacroProvider",
    "ProviderError",
    "RateLimitError",
    # Factory
    "create_price_provider",
    "create_news_provider",
    "create_macro_provider",
    # Config
    "ProviderConfig",
    "get_provider_config",
]
