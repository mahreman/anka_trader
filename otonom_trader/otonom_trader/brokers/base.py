"""
Base broker abstraction for order execution.

Provides a unified interface for both real and shadow (paper) trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OrderRequest:
    """
    Order request specification.

    Attributes:
        symbol: Asset symbol (e.g., "BTC-USD")
        side: Order side ("BUY" or "SELL")
        qty: Order quantity
        price: Limit price (None for market orders)
        order_type: Order type ("MARKET" or "LIMIT")

    Example:
        >>> order = OrderRequest(
        ...     symbol="BTC-USD",
        ...     side="BUY",
        ...     qty=0.1,
        ...     price=None,  # Market order
        ...     order_type="MARKET"
        ... )
    """

    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    price: Optional[float] = None
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"


@dataclass
class OrderResult:
    """
    Order execution result.

    Attributes:
        ok: Whether order was successful
        order_id: Broker-assigned order ID
        message: Optional status or error message
        filled_qty: Quantity filled (for partial fills)
        avg_price: Average fill price

    Example:
        >>> result = OrderResult(
        ...     ok=True,
        ...     order_id="12345",
        ...     message="Order filled",
        ...     filled_qty=0.1,
        ...     avg_price=50000.0
        ... )
    """

    ok: bool
    order_id: Optional[str] = None
    message: Optional[str] = None
    filled_qty: Optional[float] = None
    avg_price: Optional[float] = None


class Broker:
    """
    Abstract broker interface.

    Subclass this to implement real broker integrations (e.g., Alpaca, IBKR).
    Use DummyBroker for shadow mode (paper trading without real orders).
    """

    def place_order(self, req: OrderRequest) -> OrderResult:
        """
        Place an order.

        Args:
            req: Order request specification

        Returns:
            OrderResult with execution status

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement place_order()")

    def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an existing order.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            OrderResult with cancellation status

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement cancel_order()")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Get open (unfilled) orders.

        Args:
            symbol: Optional symbol filter (None = all symbols)

        Returns:
            List of open orders

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_open_orders()")

    def get_positions(self) -> List[dict]:
        """
        Get current positions.

        Returns:
            List of position dictionaries with symbol, qty, avg_entry_price

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_positions()")

    def get_account_balance(self) -> dict:
        """
        Get account balance and equity.

        Returns:
            Dictionary with cash, equity, buying_power

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_account_balance()")
