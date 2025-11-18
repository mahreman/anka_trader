"""
Broker abstraction layer for unified trading interface.

Provides a unified interface for different brokers (Binance, Bybit, etc.)
with support for:
- Order placement and cancellation
- Position and balance queries
- Guardrails and kill-switch
- Shadow mode (paper + real broker parallel)
"""

from .base import (
    Broker,
    BrokerConfig,
    OrderRequest,
    OrderFill,
    Position,
    Balance,
    OrderSide,
    OrderType,
    TimeInForce,
    BrokerError,
)
from .factory import build_broker, load_broker_config
from .guarded import GuardedBroker, GuardrailConfig

__all__ = [
    "Broker",
    "BrokerConfig",
    "OrderRequest",
    "OrderFill",
    "Position",
    "Balance",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "BrokerError",
    "build_broker",
    "load_broker_config",
    "GuardedBroker",
    "GuardrailConfig",
]
