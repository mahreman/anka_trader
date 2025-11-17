"""
Dummy broker for shadow mode (paper trading).

Logs orders without executing them on a real broker.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from .base import Broker, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class DummyBroker(Broker):
    """
    Shadow mode broker that logs orders without real execution.

    Useful for:
    - Paper trading (dry-run mode)
    - Testing strategies without capital
    - Validating order flow before going live

    Example:
        >>> broker = DummyBroker()
        >>> req = OrderRequest(symbol="BTC-USD", side="BUY", qty=0.1)
        >>> result = broker.place_order(req)
        >>> print(result.ok)  # True (always succeeds in shadow mode)
        True
    """

    def __init__(self):
        """Initialize dummy broker with empty order tracking."""
        self._open_orders = {}  # order_id -> OrderRequest
        self._positions = {}  # symbol -> qty
        self._cash = 100000.0  # Starting cash for paper account

        logger.info("DummyBroker initialized (shadow mode)")

    def place_order(self, req: OrderRequest) -> OrderResult:
        """
        Log order without real execution.

        Args:
            req: Order request

        Returns:
            OrderResult with dummy order ID
        """
        order_id = f"dummy-{uuid.uuid4().hex[:8]}"

        logger.info(
            f"[SHADOW] Order placed: {req.side} {req.qty} {req.symbol} "
            f"@ {req.price or 'MARKET'} (order_id={order_id})"
        )

        # Track open order
        self._open_orders[order_id] = req

        return OrderResult(
            ok=True,
            order_id=order_id,
            message="Shadow order logged (not executed)",
            filled_qty=req.qty,  # Assume immediate fill in shadow mode
            avg_price=req.price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel a shadow order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResult with cancellation status
        """
        if order_id in self._open_orders:
            req = self._open_orders.pop(order_id)
            logger.info(f"[SHADOW] Order canceled: {order_id} ({req.side} {req.symbol})")
            return OrderResult(
                ok=True,
                order_id=order_id,
                message="Shadow order canceled"
            )
        else:
            logger.warning(f"[SHADOW] Order not found for cancellation: {order_id}")
            return OrderResult(
                ok=False,
                order_id=order_id,
                message="Order not found"
            )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Get tracked open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open order results
        """
        results = []

        for order_id, req in self._open_orders.items():
            if symbol is None or req.symbol == symbol:
                results.append(
                    OrderResult(
                        ok=True,
                        order_id=order_id,
                        message="Open",
                        filled_qty=0.0,
                        avg_price=req.price,
                    )
                )

        return results

    def get_positions(self) -> List[dict]:
        """
        Get tracked paper positions.

        Returns:
            List of position dictionaries
        """
        positions = []

        for symbol, qty in self._positions.items():
            if qty != 0:
                positions.append({
                    "symbol": symbol,
                    "qty": qty,
                    "avg_entry_price": None,  # Not tracked in shadow mode
                    "market_value": None,
                })

        return positions

    def get_account_balance(self) -> dict:
        """
        Get paper account balance.

        Returns:
            Dictionary with cash, equity, buying_power
        """
        return {
            "cash": self._cash,
            "equity": self._cash,  # Simplified: not tracking position values
            "buying_power": self._cash,
        }
