"""
Provider factory - Create providers from configuration.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from .base import PriceProvider, NewsProvider, MacroProvider, ProviderError
from .config import (
    ProviderConfig,
    PriceProviderConfig,
    NewsProviderConfig,
    MacroProviderConfig,
    get_provider_config,
)

logger = logging.getLogger(__name__)


def create_price_provider(config: PriceProviderConfig) -> PriceProvider:
    """
    Create price provider from configuration.

    Args:
        config: Price provider configuration

    Returns:
        PriceProvider instance

    Raises:
        ProviderError: If provider type is unknown

    Example:
        >>> from otonom_trader.providers.config import PriceProviderConfig
        >>> config = PriceProviderConfig(provider_type="binance")
        >>> provider = create_price_provider(config)
    """
    provider_type = config.provider_type.lower()

    # Convert config to dict for provider initialization
    provider_config = {
        "api_key": config.api_key,
        "api_secret": config.api_secret,
        "base_url": config.base_url,
        "timeout_seconds": config.timeout_seconds,
        "extra": config.extra,
    }

    if provider_type == "binance":
        from .price_binance import BinanceProvider
        return BinanceProvider(provider_config)

    elif provider_type == "yfinance":
        from .price_yfinance import YFinanceProvider
        return YFinanceProvider(provider_config)

    elif provider_type == "polygon":
        # TODO: Implement PolygonProvider
        logger.warning("PolygonProvider not implemented yet, using dummy")
        from .price_dummy import DummyPriceProvider
        return DummyPriceProvider(provider_config)

    elif provider_type == "alphavantage":
        # TODO: Implement AlphaVantageProvider
        logger.warning("AlphaVantageProvider not implemented yet, using dummy")
        from .price_dummy import DummyPriceProvider
        return DummyPriceProvider(provider_config)

    else:
        raise ProviderError(f"Unknown price provider type: {provider_type}")


def create_news_provider(config: NewsProviderConfig) -> NewsProvider:
    """
    Create news provider from configuration.

    Args:
        config: News provider configuration

    Returns:
        NewsProvider instance

    Raises:
        ProviderError: If provider type is unknown

    Example:
        >>> from otonom_trader.providers.config import NewsProviderConfig
        >>> config = NewsProviderConfig(provider_type="newsapi", api_key="...")
        >>> provider = create_news_provider(config)
    """
    provider_type = config.provider_type.lower()

    provider_config = {
        "api_key": config.api_key,
        "timeout_seconds": config.timeout_seconds,
        "extra": config.extra,
    }

    if provider_type == "newsapi":
        from .news_newsapi import NewsAPIProvider
        return NewsAPIProvider(provider_config)

    elif provider_type in {"yfinance", "yfinance_news"}:
        from .news_yfinance import YFinanceNewsProvider
        return YFinanceNewsProvider(provider_config)

    elif provider_type == "polygon_news":
        # TODO: Implement PolygonNewsProvider
        logger.warning("PolygonNewsProvider not implemented yet, using dummy")
        from .news_dummy import DummyNewsProvider
        return DummyNewsProvider(provider_config)

    elif provider_type == "rss":
        from .news_rss import RSSNewsProvider
        return RSSNewsProvider(provider_config)

    else:
        raise ProviderError(f"Unknown news provider type: {provider_type}")


def create_macro_provider(config: MacroProviderConfig) -> MacroProvider:
    """
    Create macro provider from configuration.

    Args:
        config: Macro provider configuration

    Returns:
        MacroProvider instance

    Raises:
        ProviderError: If provider type is unknown

    Example:
        >>> from otonom_trader.providers.config import MacroProviderConfig
        >>> config = MacroProviderConfig(provider_type="fred", api_key="...")
        >>> provider = create_macro_provider(config)
    """
    provider_type = config.provider_type.lower()

    provider_config = {
        "api_key": config.api_key,
        "base_url": config.extra.get("base_url") if config.extra else None,
        "timeout_seconds": config.timeout_seconds,
        "extra": config.extra,
    }

    if provider_type == "fred":
        from .macro_fred import FREDProvider
        return FREDProvider(provider_config)

    elif provider_type == "worldbank":
        # TODO: Implement WorldBankProvider
        logger.warning("WorldBankProvider not implemented yet, using dummy")
        from .macro_dummy import DummyMacroProvider
        return DummyMacroProvider(provider_config)

    elif provider_type == "tradingeconomics":
        # TODO: Implement TradingEconomicsProvider
        logger.warning("TradingEconomicsProvider not implemented yet, using dummy")
        from .macro_dummy import DummyMacroProvider
        return DummyMacroProvider(provider_config)

    else:
        raise ProviderError(f"Unknown macro provider type: {provider_type}")


def create_all_providers(
    config_path: str = "config/providers.yaml",
) -> tuple[List[PriceProvider], List[NewsProvider], List[MacroProvider]]:
    """
    Create all enabled providers from configuration file.

    Args:
        config_path: Path to providers.yaml

    Returns:
        Tuple of (price_providers, news_providers, macro_providers)

    Example:
        >>> price, news, macro = create_all_providers()
        >>> print(f"Created {len(price)} price providers")
    """
    config = get_provider_config(config_path)

    # Create price providers
    price_providers = []
    for p_config in config.get_enabled_price_providers():
        try:
            provider = create_price_provider(p_config)
            price_providers.append(provider)
            logger.info(f"Created price provider: {p_config.provider_type}")
        except Exception as e:
            logger.error(f"Failed to create price provider {p_config.provider_type}: {e}")

    # Create news providers
    news_providers = []
    for n_config in config.get_enabled_news_providers():
        try:
            provider = create_news_provider(n_config)
            news_providers.append(provider)
            logger.info(f"Created news provider: {n_config.provider_type}")
        except Exception as e:
            logger.error(f"Failed to create news provider {n_config.provider_type}: {e}")

    # Create macro providers
    macro_providers = []
    for m_config in config.get_enabled_macro_providers():
        try:
            provider = create_macro_provider(m_config)
            macro_providers.append(provider)
            logger.info(f"Created macro provider: {m_config.provider_type}")
        except Exception as e:
            logger.error(f"Failed to create macro provider {m_config.provider_type}: {e}")

    logger.info(
        f"Created {len(price_providers)} price, "
        f"{len(news_providers)} news, "
        f"{len(macro_providers)} macro providers"
    )

    return price_providers, news_providers, macro_providers


def get_primary_price_provider(config_path: str = "config/providers.yaml") -> Optional[PriceProvider]:
    """
    Get primary (first enabled) price provider.

    Args:
        config_path: Path to providers.yaml

    Returns:
        Primary price provider or None

    Example:
        >>> provider = get_primary_price_provider()
        >>> bars = provider.fetch_ohlcv("BTC-USD", start, end)
    """
    config = get_provider_config(config_path)
    primary_config = config.get_primary_price_provider()

    if primary_config is None:
        logger.warning("No enabled price providers")
        return None

    return create_price_provider(primary_config)


def get_primary_news_provider(config_path: str = "config/providers.yaml") -> Optional[NewsProvider]:
    """
    Get primary (first enabled) news provider.

    Args:
        config_path: Path to providers.yaml

    Returns:
        Primary news provider or None
    """
    config = get_provider_config(config_path)
    primary_config = config.get_primary_news_provider()

    if primary_config is None:
        logger.warning("No enabled news providers")
        return None

    return create_news_provider(primary_config)


def get_primary_macro_provider(config_path: str = "config/providers.yaml") -> Optional[MacroProvider]:
    """
    Get primary (first enabled) macro provider.

    Args:
        config_path: Path to providers.yaml

    Returns:
        Primary macro provider or None
    """
    config = get_provider_config(config_path)
    primary_config = config.get_primary_macro_provider()

    if primary_config is None:
        logger.warning("No enabled macro providers")
        return None

    return create_macro_provider(primary_config)
