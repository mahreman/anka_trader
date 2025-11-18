"""
Guarded broker with kill-switch and risk guardrails.

Wraps any broker implementation and adds safety checks:
- Maximum daily loss limit
- Maximum drawdown limit
- Consecutive loss limit
- Position size limits

When guardrails are triggered, the kill-switch prevents new orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from .base import (
    Balance,
    Broker,
    BrokerConfig,
    BrokerError,
    OrderFill,
    OrderRequest,
    Position,
)
from ..data import get_session
from ..data.schema import PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """
    Guardrail configuration for risk management.
    
    Attributes:
        max_daily_loss_pct: Maximum daily loss as % of starting equity
        max_drawdown_pct: Maximum drawdown from peak as %
        max_consecutive_losses: Maximum number of consecutive losing trades
        enabled: Whether guardrails are enabled
    """
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 40.0
    max_consecutive_losses: int = 5
    enabled: bool = True


class GuardedBroker(Broker):
    """
    Broker wrapper with kill-switch and guardrails.
    
    This wraps any broker implementation and adds safety checks.
    When guardrails are triggered, it prevents new orders from being placed.
    
    Example:
        >>> raw_broker = BinanceBroker(config)
        >>> guarded = GuardedBroker(raw_broker, GuardrailConfig(max_daily_loss_pct=5.0))
        >>> guarded.place_order(order)  # Will check guardrails first
    """

    def __init__(self, inner: Broker, guard_cfg: GuardrailConfig):
        """
        Initialize guarded broker.
        
        Args:
            inner: Underlying broker implementation
            guard_cfg: Guardrail configuration
        """
        super().__init__(inner.cfg)
        self.inner = inner
        self.guard_cfg = guard_cfg
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        
        logger.info(f"Initialized GuardedBroker with config: {guard_cfg}")

    # ==================== Risk Checks ====================

    def _check_daily_loss(self) -> tuple[bool, str]:
        """
        Check if daily loss exceeded limit.
        
        Returns:
            (triggered, reason) tuple
        """
        if not self.guard_cfg.enabled:
            return False, ""

        try:
            with get_session() as session:
                today = datetime.utcnow().date()
                yesterday = today - timedelta(days=1)
                
                # Get today's snapshots
                today_snaps = (
                    session.query(PortfolioSnapshot)
                    .filter(PortfolioSnapshot.timestamp >= yesterday)
                    .order_by(PortfolioSnapshot.timestamp.asc())
                    .all()
                )
                
                if len(today_snaps) < 2:
                    return False, ""
                
                start_equity = today_snaps[0].equity
                current_equity = today_snaps[-1].equity
                
                if start_equity == 0:
                    return False, ""
                
                daily_loss_pct = (current_equity - start_equity) / start_equity * 100.0
                
                if daily_loss_pct <= -self.guard_cfg.max_daily_loss_pct:
                    reason = (
                        f"Daily loss {daily_loss_pct:.2f}% exceeded limit "
                        f"{self.guard_cfg.max_daily_loss_pct}%"
                    )
                    logger.warning(f"GUARDRAIL TRIGGERED: {reason}")
                    return True, reason
                
                return False, ""
                
        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return False, ""

    def _check_max_drawdown(self) -> tuple[bool, str]:
        """
        Check if max drawdown exceeded limit.
        
        Returns:
            (triggered, reason) tuple
        """
        if not self.guard_cfg.enabled:
            return False, ""

        try:
            with get_session() as session:
                # Get all snapshots
                snaps = (
                    session.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.asc())
                    .all()
                )
                
                if len(snaps) < 2:
                    return False, ""
                
                # Calculate max drawdown
                equity_series = [float(s.equity) for s in snaps]
                peak = max(equity_series)
                current = equity_series[-1]
                
                if peak == 0:
                    return False, ""
                
                dd_pct = (current - peak) / peak * 100.0
                
                if dd_pct <= -self.guard_cfg.max_drawdown_pct:
                    reason = (
                        f"Max drawdown {dd_pct:.2f}% exceeded limit "
                        f"{self.guard_cfg.max_drawdown_pct}%"
                    )
                    logger.warning(f"GUARDRAIL TRIGGERED: {reason}")
                    return True, reason
                
                return False, ""
                
        except Exception as e:
            logger.error(f"Error checking max drawdown: {e}")
            return False, ""

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        """
        Check if consecutive losses exceeded limit.
        
        Returns:
            (triggered, reason) tuple
        """
        if not self.guard_cfg.enabled:
            return False, ""

        try:
            with get_session() as session:
                # Import Trade here to avoid circular import
                from ..data.schema import Trade
                
                # Get last N trades
                last_trades = (
                    session.query(Trade)
                    .filter(Trade.exit_date.isnot(None))
                    .order_by(Trade.exit_date.desc())
                    .limit(self.guard_cfg.max_consecutive_losses)
                    .all()
                )
                
                if len(last_trades) < self.guard_cfg.max_consecutive_losses:
                    return False, ""
                
                # Count consecutive losses
                consecutive_losses = 0
                for trade in last_trades:
                    if trade.pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                
                if consecutive_losses >= self.guard_cfg.max_consecutive_losses:
                    reason = (
                        f"Consecutive losses {consecutive_losses} exceeded limit "
                        f"{self.guard_cfg.max_consecutive_losses}"
                    )
                    logger.warning(f"GUARDRAIL TRIGGERED: {reason}")
                    return True, reason
                
                return False, ""
                
        except Exception as e:
            logger.error(f"Error checking consecutive losses: {e}")
            return False, ""

    def _is_kill_switch_triggered(self) -> bool:
        """
        Check all guardrails and determine if kill-switch should be active.
        
        Returns:
            True if any guardrail is triggered, False otherwise
        """
        if not self.guard_cfg.enabled:
            return False

        # Check all guardrails
        checks = [
            self._check_daily_loss(),
            self._check_max_drawdown(),
            self._check_consecutive_losses(),
        ]
        
        for triggered, reason in checks:
            if triggered:
                self._kill_switch_active = True
                self._kill_switch_reason = reason
                return True
        
        # Reset kill-switch if all checks pass
        if self._kill_switch_active:
            logger.info("Guardrails clear - resetting kill-switch")
            self._kill_switch_active = False
            self._kill_switch_reason = ""
        
        return False

    # ==================== Broker Interface ====================

    def place_order(self, req: OrderRequest) -> OrderFill:
        """
        Place order with guardrail checks.
        
        Args:
            req: Order request
            
        Returns:
            Order fill confirmation
            
        Raises:
            BrokerError: If guardrails are triggered or order fails
        """
        # Check kill-switch
        if self._is_kill_switch_triggered():
            error_msg = f"KILL-SWITCH ACTIVE: {self._kill_switch_reason}"
            logger.error(error_msg)
            raise BrokerError(error_msg)
        
        # Forward to underlying broker
        return self.inner.place_order(req)

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """
        Cancel order (always allowed, even if kill-switch is active).
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
        """
        return self.inner.cancel_order(symbol, order_id)

    def get_open_positions(self) -> List[Position]:
        """
        Get open positions.
        
        Returns:
            List of open positions
        """
        return self.inner.get_open_positions()

    def get_balances(self) -> List[Balance]:
        """
        Get account balances.
        
        Returns:
            List of asset balances
        """
        return self.inner.get_balances()

    def ping(self) -> bool:
        """
        Check broker connectivity.
        
        Returns:
            True if connection is OK
        """
        return self.inner.ping()

    def get_kill_switch_status(self) -> tuple[bool, str]:
        """
        Get kill-switch status.
        
        Returns:
            (active, reason) tuple
        """
        self._is_kill_switch_triggered()  # Refresh status
        return self._kill_switch_active, self._kill_switch_reason
