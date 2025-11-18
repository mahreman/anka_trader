"""
Broker factory for loading and building broker instances.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .base import Broker
from .binance import BinanceBroker, BinanceBrokerConfig

logger = logging.getLogger(__name__)


def load_broker_config(path: str | Path) -> Dict[str, Any]:
    """
    Load broker configuration from YAML file.
    
    Args:
        path: Path to broker config YAML
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_broker_config("config/broker.yaml")
        >>> config["kind"]
        'binance'
    """
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"Broker config not found: {path}")
    
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    logger.info(f"Loaded broker config from {path}")
    return data


def build_broker(path: str | Path) -> Broker:
    """
    Build broker instance from configuration file.
    
    Args:
        path: Path to broker config YAML
        
    Returns:
        Broker instance
        
    Raises:
        ValueError: If broker kind is unknown
        FileNotFoundError: If config file doesn't exist
        
    Example:
        >>> broker = build_broker("config/broker.yaml")
        >>> broker.ping()
        True
    """
    data = load_broker_config(path)
    kind = data.get("kind", "binance").lower()

    logger.info(f"Building broker: {kind}")

    if kind == "binance":
        cfg = BinanceBrokerConfig(
            name="binance",
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            base_url=data.get("base_url", "https://api.binance.com"),
            testnet=data.get("testnet", True),
            recv_window=data.get("recv_window", 5000),
            timeout=data.get("timeout", 10),
        )
        return BinanceBroker(cfg)

    # Add more broker types here as needed
    # elif kind == "bybit":
    #     ...

    raise ValueError(f"Unknown broker kind: {kind}")
