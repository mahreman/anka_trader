"""
Broker abstraction layer for order execution.

Supports both real and shadow (paper) trading modes.
"""

from .base import Broker, OrderRequest, OrderResult
from .dummy import DummyBroker

__all__ = [
    "Broker",
    "OrderRequest",
    "OrderResult",
    "DummyBroker",
]
