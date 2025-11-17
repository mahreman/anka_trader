"""
Domain models - Core business entities as dataclasses.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .enums import AssetClass, SignalType, AnomalyType


@dataclass
class Asset:
    """
    Represents a tradeable asset.

    Attributes:
        symbol: Trading symbol (e.g., 'BTC-USD', 'GC=F')
        name: Human-readable name (e.g., 'Bitcoin', 'Gold')
        asset_class: Classification of the asset
        base_currency: Base currency for pricing (e.g., 'USD')
    """

    symbol: str
    name: str
    asset_class: AssetClass
    base_currency: str = "USD"

    def __post_init__(self):
        """Validate asset fields."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.name:
            raise ValueError("Name cannot be empty")


@dataclass
class Anomaly:
    """
    Represents a detected price anomaly (spike or crash).

    Attributes:
        asset_symbol: Symbol of the asset
        date: Date when anomaly occurred
        anomaly_type: Type of anomaly detected
        abs_return: Absolute return value (e.g., 0.05 for 5%)
        zscore: Z-score of the return
        volume_rank: Volume percentile/quantile (0-1)
        comment: Optional manual comment/note
    """

    asset_symbol: str
    date: date
    anomaly_type: AnomalyType
    abs_return: float
    zscore: float
    volume_rank: float
    comment: Optional[str] = None

    def __post_init__(self):
        """Validate anomaly fields."""
        if not self.asset_symbol:
            raise ValueError("Asset symbol cannot be empty")
        if not isinstance(self.date, date):
            raise ValueError("Date must be a date object")
        if self.volume_rank < 0 or self.volume_rank > 1:
            raise ValueError("Volume rank must be between 0 and 1")


@dataclass
class Decision:
    """
    Represents a trading decision made by the Patron.

    Attributes:
        asset_symbol: Symbol of the asset
        date: Date of the decision
        signal: Trading signal (BUY/SELL/HOLD)
        confidence: Confidence level (0-1)
        reason: Human-readable explanation of the decision
    """

    asset_symbol: str
    date: date
    signal: SignalType
    confidence: float
    reason: str

    def __post_init__(self):
        """Validate decision fields."""
        if not self.asset_symbol:
            raise ValueError("Asset symbol cannot be empty")
        if not isinstance(self.date, date):
            raise ValueError("Date must be a date object")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not self.reason:
            raise ValueError("Reason cannot be empty")
