"""
Broker factory for creating configured brokers.

Creates broker instances based on configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Broker
from .binance import BinanceBroker
from .config import BrokerConfig, get_broker_config
from .dummy import DummyBroker
from .risk_guardrails import GuardedBroker

logger = logging.getLogger(__name__)


def create_broker(
    config: Optional[BrokerConfig] = None,
    config_path: str = "config/broker.yaml",
) -> Broker:
    """
    Create broker instance from configuration.

    Args:
        config: Optional BrokerConfig instance (loads from file if not provided)
        config_path: Path to broker.yaml (used if config not provided)

    Returns:
        Broker instance (Dummy, Binance, etc.) wrapped with risk guardrails

    Raises:
        ValueError: If broker type is not supported

    Example:
        >>> broker = create_broker()  # Uses config/broker.yaml
        >>> result = broker.place_order(OrderRequest(...))

    Example with custom config:
        >>> config = BrokerConfig(broker_type="binance", api_key="...", api_secret="...")
        >>> broker = create_broker(config=config)
    """
    # Load config if not provided
    if config is None:
        config = get_broker_config(config_path)

    # Create base broker based on type
    if config.broker_type == "dummy" or config.shadow_mode:
        # Shadow mode always uses dummy broker
        base_broker = DummyBroker()
        logger.info("Using DummyBroker (shadow mode)")

    elif config.broker_type == "binance":
        if not config.api_key or not config.api_secret:
            logger.error("Binance API credentials not provided, falling back to DummyBroker")
            base_broker = DummyBroker()
        else:
            base_broker = BinanceBroker(
                api_key=config.api_key,
                api_secret=config.api_secret,
                base_url=config.base_url,
                use_testnet=config.use_testnet,
            )
            logger.info(f"Using BinanceBroker (testnet={config.use_testnet})")

    elif config.broker_type == "alpaca":
        # TODO: Implement AlpacaBroker
        logger.warning("Alpaca broker not implemented, using DummyBroker")
        base_broker = DummyBroker()

    elif config.broker_type == "ibkr":
        # TODO: Implement IBKRBroker
        logger.warning("IBKR broker not implemented, using DummyBroker")
        base_broker = DummyBroker()

    else:
        raise ValueError(f"Unsupported broker type: {config.broker_type}")

    # Wrap with risk guardrails
    guardrails = config.get_risk_guardrails()
    guarded_broker = GuardedBroker(
        underlying_broker=base_broker,
        guardrails=guardrails,
    )

    logger.info(
        f"Broker created: type={config.broker_type}, "
        f"shadow_mode={config.shadow_mode}, "
        f"guardrails=enabled"
    )

    return guarded_broker
