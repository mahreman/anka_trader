"""
Base broker interface and data classes.

All broker implementations must implement the Broker abstract class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class TimeInForce(str, Enum):
    """Time in force for limit orders."""
    GTC = "GTC"   # Good till cancel
    IOC = "IOC"   # Immediate or cancel
    FOK = "FOK"   # Fill or kill


@dataclass
class BrokerConfig:
    """Base broker configuration."""
    name: str
    api_key: str
    api_secret: str
    base_url: str
    testnet: bool = True


@dataclass
class OrderRequest:
    """
    Order request to be sent to broker.
    
    Attributes:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        side: BUY or SELL
        quantity: Order quantity
        order_type: MARKET, LIMIT, etc.
        price: Limit price (required for LIMIT orders)
        time_in_force: GTC, IOC, FOK
        client_order_id: Custom order ID
        meta: Additional metadata
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderFill:
    """
    Order fill confirmation from broker.
    
    Attributes:
        order_id: Broker order ID
        symbol: Trading pair symbol
        side: BUY or SELL
        quantity: Filled quantity
        price: Fill price
        ts: Execution timestamp
        commission: Trading commission
        commission_asset: Asset used for commission
    """
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0
    commission_asset: str = ""


@dataclass
class Position:
    """
    Open position.
    
    Attributes:
        symbol: Trading pair symbol
        quantity: Position quantity (positive for long, negative for short)
        entry_price: Average entry price
        unrealized_pnl: Unrealized P&L
        leverage: Leverage (1.0 for spot)
    """
    symbol: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: float = 1.0


@dataclass
class Balance:
    """
    Asset balance.
    
    Attributes:
        asset: Asset symbol (e.g., "BTC", "USDT")
        free: Available balance
        locked: Locked balance (in orders)
    """
    asset: str
    free: float
    locked: float
    
    @property
    def total(self) -> float:
        """Total balance (free + locked)."""
        return self.free + self.locked


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class Broker(ABC):
    """
    Abstract broker interface.
    
    All broker implementations (Binance, Bybit, etc.) must implement this interface.
    """

    def __init__(self, cfg: BrokerConfig):
        """
        Initialize broker.
        
        Args:
            cfg: Broker configuration
        """
        self.cfg = cfg

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderFill:
        """
        Place an order.
        
        Args:
            req: Order request
            
        Returns:
            Order fill confirmation
            
        Raises:
            BrokerError: If order fails
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Raises:
            BrokerError: If cancellation fails
        """
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
            
        Raises:
            BrokerError: If query fails
        """
        raise NotImplementedError

    @abstractmethod
    def get_balances(self) -> List[Balance]:
        """
        Get account balances.
        
        Returns:
            List of asset balances
            
        Raises:
            BrokerError: If query fails
        """
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        """
        Check broker connectivity.
        
        Returns:
            True if connection is OK, False otherwise
        """
        raise NotImplementedError
