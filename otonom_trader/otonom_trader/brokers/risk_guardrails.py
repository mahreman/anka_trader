"""
Risk guardrails wrapper for broker.

Enforces risk limits before order execution.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .base import Broker, OrderRequest, OrderResult
from .config import RiskGuardrails

logger = logging.getLogger(__name__)


class GuardedBroker(Broker):
    """
    Broker wrapper with risk guardrails.

    Enforces risk limits before passing orders to underlying broker:
    - Maximum notional per order
    - Maximum open risk
    - Symbol blacklist
    - Kill switch

    Example:
        >>> from .dummy import DummyBroker
        >>> from .config import RiskGuardrails
        >>>
        >>> guardrails = RiskGuardrails(max_notional_per_order=5000)
        >>> base_broker = DummyBroker()
        >>> broker = GuardedBroker(base_broker, guardrails)
        >>>
        >>> # This will be rejected
        >>> req = OrderRequest(symbol="BTC-USD", side="BUY", qty=1.0)  # $50k
        >>> result = broker.place_order(req, current_price=50000)
        >>> print(result.ok)  # False
    """

    def __init__(
        self,
        underlying_broker: Broker,
        guardrails: RiskGuardrails,
        alert_callback: Optional[callable] = None,
    ):
        """
        Initialize guarded broker.

        Args:
            underlying_broker: Actual broker (e.g., BinanceBroker, DummyBroker)
            guardrails: Risk guardrails configuration
            alert_callback: Optional callback for alerts (e.g., send_alert)
        """
        self.broker = underlying_broker
        self.guardrails = guardrails
        self.alert_callback = alert_callback

        logger.info("GuardedBroker initialized with risk limits")

    def _send_alert(self, level: str, message: str):
        """Send alert if callback is configured."""
        if self.alert_callback:
            try:
                self.alert_callback(level, message)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            logger.warning(f"[ALERT {level}] {message}")

    def place_order(
        self,
        req: OrderRequest,
        current_price: Optional[float] = None,
        current_equity: Optional[float] = None,
        current_positions: Optional[int] = None,
    ) -> OrderResult:
        """
        Place order with risk checks.

        Args:
            req: Order request
            current_price: Current market price (for notional calculation)
            current_equity: Current account equity (for position sizing)
            current_positions: Current number of open positions

        Returns:
            OrderResult (rejected if risk check fails)

        Example:
            >>> req = OrderRequest(symbol="BTC-USD", side="BUY", qty=0.1)
            >>> result = broker.place_order(req, current_price=50000, current_equity=100000)
        """
        # Calculate order notional
        if current_price is None:
            # Try to get price from broker if available
            if hasattr(self.broker, "get_ticker_price"):
                current_price = self.broker.get_ticker_price(req.symbol)

            if current_price is None:
                current_price = req.price if req.price else 0.0

        notional = req.qty * current_price

        # Default equity and positions if not provided
        if current_equity is None:
            # Try to get from broker
            if hasattr(self.broker, "get_account_balance"):
                balance = self.broker.get_account_balance()
                current_equity = balance.get("equity", 100000.0)
            else:
                current_equity = 100000.0  # Default assumption

        if current_positions is None:
            # Try to get from broker
            if hasattr(self.broker, "get_positions"):
                positions = self.broker.get_positions()
                current_positions = len(positions)
            else:
                current_positions = 0

        # Run risk checks
        is_ok, error_message = self.guardrails.check_order(
            symbol=req.symbol,
            notional=notional,
            current_equity=current_equity,
            current_positions=current_positions,
        )

        if not is_ok:
            logger.warning(f"Order rejected by risk guardrails: {error_message}")
            self._send_alert("WARN", f"Order rejected: {error_message}")

            return OrderResult(
                ok=False,
                message=f"RISK REJECTED: {error_message}",
            )

        # Risk checks passed, execute order
        logger.info(
            f"Risk checks passed: notional=${notional:.2f}, "
            f"equity=${current_equity:.2f}, positions={current_positions}"
        )

        return self.broker.place_order(req)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> OrderResult:
        """Pass through to underlying broker."""
        return self.broker.cancel_order(order_id, symbol)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Pass through to underlying broker."""
        return self.broker.get_open_orders(symbol)

    def get_positions(self) -> List[dict]:
        """Pass through to underlying broker."""
        return self.broker.get_positions()

    def get_account_balance(self) -> dict:
        """Pass through to underlying broker."""
        return self.broker.get_account_balance()
