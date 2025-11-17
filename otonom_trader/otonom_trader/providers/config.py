"""
Provider configuration loader.
"""
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_provider_config(config_path: str = "config/providers.yaml") -> Dict[str, Any]:
    """
    Load provider configuration from YAML file.

    Args:
        config_path: Path to providers.yaml configuration file

    Returns:
        Dictionary with provider configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Provider config not found at: {config_path}\n"
            f"Create this file with provider settings (API keys, etc.)"
        )

    logger.info(f"Loading provider config from: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")

    return config


def get_price_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract price data configuration.

    Args:
        config: Full provider configuration

    Returns:
        Dictionary with price provider settings
    """
    return config.get("price", {})


def get_news_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract news data configuration.

    Args:
        config: Full provider configuration

    Returns:
        Dictionary with news provider settings
    """
    return config.get("news", {})


def get_macro_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract macro data configuration.

    Args:
        config: Full provider configuration

    Returns:
        Dictionary with macro provider settings
    """
    return config.get("macro", {})


def get_provider_credentials(config: Dict[str, Any], provider_name: str) -> Dict[str, Any]:
    """
    Get credentials for a specific provider.

    Args:
        config: Full provider configuration
        provider_name: Name of provider (e.g., "binance", "newsapi", "fred")

    Returns:
        Dictionary with provider credentials
    """
    return config.get(provider_name, {})
