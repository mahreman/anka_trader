"""
Broker abstraction layer for order execution.

Supports both real and shadow (paper) trading modes.
"""

from .base import Broker, OrderRequest, OrderResult
from .binance import BinanceBroker
from .config import BrokerConfig, RiskGuardrails, get_broker_config
from .dummy import DummyBroker
from .factory import create_broker
from .risk_guardrails import GuardedBroker

__all__ = [
    "Broker",
    "OrderRequest",
    "OrderResult",
    "DummyBroker",
    "BinanceBroker",
    "BrokerConfig",
    "RiskGuardrails",
    "get_broker_config",
    "create_broker",
    "GuardedBroker",
]
