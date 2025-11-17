"""
Domain enumerations for asset classes, signal types, and anomaly types.
"""
from enum import Enum


class AssetClass(str, Enum):
    """Asset classification types."""

    COMMODITY = "COMMODITY"
    INDEX = "INDEX"
    CRYPTO = "CRYPTO"

    def __str__(self) -> str:
        return self.value


class SignalType(str, Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __str__(self) -> str:
        return self.value


class AnomalyType(str, Enum):
    """Price anomaly classification types."""

    SPIKE_UP = "SPIKE_UP"
    SPIKE_DOWN = "SPIKE_DOWN"
    NONE = "NONE"

    def __str__(self) -> str:
        return self.value
