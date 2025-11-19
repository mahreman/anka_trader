"""
Provider configuration management.

Loads provider configs from YAML files with API keys and settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PriceProviderConfig:
    """Configuration for price data provider."""

    provider_type: str  # "binance", "polygon", "alpaca", "yfinance"
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class NewsProviderConfig:
    """Configuration for news data provider."""

    provider_type: str  # "newsapi", "polygon", "benzinga"
    enabled: bool = True
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class MacroProviderConfig:
    """Configuration for macro data provider."""

    provider_type: str  # "fred", "worldbank", "imf"
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class ProviderConfig:
    """
    Complete provider configuration.

    Attributes:
        price_providers: List of price provider configs
        news_providers: List of news provider configs
        macro_providers: List of macro provider configs
        fallback_enabled: Enable fallback to secondary providers
        cache_enabled: Enable caching of provider responses
    """

    price_providers: list[PriceProviderConfig]
    news_providers: list[NewsProviderConfig]
    macro_providers: list[MacroProviderConfig]
    fallback_enabled: bool = True
    cache_enabled: bool = True

    def get_enabled_price_providers(self) -> list[PriceProviderConfig]:
        """Get list of enabled price providers."""
        return [p for p in self.price_providers if p.enabled]

    def get_enabled_news_providers(self) -> list[NewsProviderConfig]:
        """Get list of enabled news providers."""
        return [p for p in self.news_providers if p.enabled]

    def get_enabled_macro_providers(self) -> list[MacroProviderConfig]:
        """Get list of enabled macro providers."""
        return [p for p in self.macro_providers if p.enabled]

    def get_primary_price_provider(self) -> Optional[PriceProviderConfig]:
        """Get primary (first enabled) price provider."""
        enabled = self.get_enabled_price_providers()
        return enabled[0] if enabled else None

    def get_primary_news_provider(self) -> Optional[NewsProviderConfig]:
        """Get primary (first enabled) news provider."""
        enabled = self.get_enabled_news_providers()
        return enabled[0] if enabled else None

    def get_primary_macro_provider(self) -> Optional[MacroProviderConfig]:
        """Get primary (first enabled) macro provider."""
        enabled = self.get_enabled_macro_providers()
        return enabled[0] if enabled else None


def load_provider_config_from_yaml(config_path: str | Path) -> ProviderConfig:
    """
    Load provider configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        ProviderConfig object

    Example:
        >>> config = load_provider_config_from_yaml("config/providers.yaml")
        >>> print(config.price_providers[0].provider_type)
        'binance'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Provider config not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse price providers
    price_providers = []
    for p in data.get("price_providers", []):
        price_providers.append(
            PriceProviderConfig(
                provider_type=p["type"],
                enabled=p.get("enabled", True),
                api_key=p.get("api_key"),
                api_secret=p.get("api_secret"),
                base_url=p.get("base_url"),
                rate_limit_per_minute=p.get("rate_limit_per_minute", 60),
                timeout_seconds=p.get("timeout_seconds", 30),
                extra=p.get("extra", {}),
            )
        )

    # Parse news providers
    news_providers = []
    for p in data.get("news_providers", []):
        news_providers.append(
            NewsProviderConfig(
                provider_type=p["type"],
                enabled=p.get("enabled", True),
                api_key=p.get("api_key"),
                rate_limit_per_minute=p.get("rate_limit_per_minute", 60),
                timeout_seconds=p.get("timeout_seconds", 30),
                extra=p.get("extra", {}),
            )
        )

    # Parse macro providers
    macro_providers = []
    for p in data.get("macro_providers", []):
        macro_providers.append(
            MacroProviderConfig(
                provider_type=p["type"],
                enabled=p.get("enabled", True),
                api_key=p.get("api_key"),
                base_url=p.get("base_url"),
                rate_limit_per_minute=p.get("rate_limit_per_minute", 60),
                timeout_seconds=p.get("timeout_seconds", 30),
                extra=p.get("extra", {}),
            )
        )

    # Global settings
    fallback_enabled = data.get("fallback_enabled", True)
    cache_enabled = data.get("cache_enabled", True)

    config = ProviderConfig(
        price_providers=price_providers,
        news_providers=news_providers,
        macro_providers=macro_providers,
        fallback_enabled=fallback_enabled,
        cache_enabled=cache_enabled,
    )

    logger.info(
        f"Loaded provider config: {len(price_providers)} price, "
        f"{len(news_providers)} news, {len(macro_providers)} macro"
    )

    return config


def get_provider_config(config_path: str = "config/providers.yaml") -> ProviderConfig:
    """
    Get provider configuration (convenience function).

    Args:
        config_path: Path to config file

    Returns:
        ProviderConfig object

    Example:
        >>> config = get_provider_config()
        >>> binance = config.get_primary_price_provider()
    """
    return load_provider_config_from_yaml(config_path)
