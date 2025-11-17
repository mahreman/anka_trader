"""
Portfolio constraint management for risk control (P3 preparation).

This module enforces portfolio-level constraints:
1. Turnover limits (max trades per day, max notional per day)
2. Cooldown periods (prevent flipping symbols too quickly)
3. Correlation netting (reduce exposure to correlated assets)

Example:
    >>> constraints = PortfolioConstraints(max_daily_trades=10)
    >>> if constraints.can_trade(symbol, "BUY"):
    ...     # Execute trade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Set, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """
    Simple trade record for constraint tracking.

    Attributes:
        symbol: Asset symbol
        timestamp: Trade timestamp
        action: BUY, SELL, HOLD
        notional: Trade notional value
    """
    symbol: str
    timestamp: datetime
    action: str
    notional: float


@dataclass
class CooldownRecord:
    """
    Cooldown record for a symbol.

    Attributes:
        symbol: Asset symbol
        last_trade_time: Timestamp of last trade
        last_action: Last action taken (BUY/SELL)
        cooldown_until: Timestamp when cooldown expires
    """
    symbol: str
    last_trade_time: datetime
    last_action: str
    cooldown_until: datetime


# Asset correlation groups for netting
# In production, this would be computed dynamically from returns correlation matrix
CORRELATION_GROUPS = {
    "CRYPTO": ["BTC-USD", "ETH-USD", "BNB-USD"],
    "GOLD": ["GC=F", "SI=F"],  # Gold, Silver
    "EQUITY": ["^GSPC", "^IXIC", "^DJI"],  # S&P 500, NASDAQ, Dow
}


class PortfolioConstraints:
    """
    Portfolio constraint enforcement engine.

    Enforces:
    - Daily turnover limits (trade count + notional)
    - Cooldown periods after flips
    - Correlation-based position netting

    Example:
        >>> constraints = PortfolioConstraints(
        ...     max_daily_trades=10,
        ...     max_daily_notional=50000.0,
        ...     cooldown_hours=24,
        ... )
        >>> constraints.record_trade("BTC-USD", "BUY", 10000.0)
        >>> can_trade = constraints.can_trade("BTC-USD", "SELL")
    """

    def __init__(
        self,
        max_daily_trades: int = 20,
        max_daily_notional: float = 100000.0,
        cooldown_hours: int = 24,
        max_correlated_exposure: float = 0.5,  # 50% of portfolio
    ):
        """
        Initialize portfolio constraints.

        Args:
            max_daily_trades: Maximum trades per day
            max_daily_notional: Maximum notional value per day
            cooldown_hours: Hours to wait after a flip (BUY→SELL or SELL→BUY)
            max_correlated_exposure: Maximum exposure to correlated assets (as fraction)
        """
        self.max_daily_trades = max_daily_trades
        self.max_daily_notional = max_daily_notional
        self.cooldown_hours = cooldown_hours
        self.max_correlated_exposure = max_correlated_exposure

        # Internal state
        self.trade_history: list[TradeRecord] = []
        self.cooldown_records: Dict[str, CooldownRecord] = {}
        self.current_positions: Dict[str, str] = {}  # symbol -> action (BUY/SELL)

        logger.info(
            f"PortfolioConstraints initialized: max_daily_trades={max_daily_trades}, "
            f"max_daily_notional=${max_daily_notional:,.0f}, cooldown={cooldown_hours}h"
        )

    def _get_today_trades(self) -> list[TradeRecord]:
        """Get trades from today."""
        today = date.today()
        return [
            t for t in self.trade_history
            if t.timestamp.date() == today
        ]

    def _get_today_trade_count(self) -> int:
        """Get number of trades today."""
        return len(self._get_today_trades())

    def _get_today_notional(self) -> float:
        """Get total notional traded today."""
        return sum(t.notional for t in self._get_today_trades())

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.cooldown_records:
            return False

        cooldown = self.cooldown_records[symbol]
        return datetime.utcnow() < cooldown.cooldown_until

    def _is_flip(self, symbol: str, action: str) -> bool:
        """Check if this trade would be a flip (BUY→SELL or SELL→BUY)."""
        if symbol not in self.current_positions:
            return False

        current_action = self.current_positions[symbol]

        # Flip conditions
        if current_action == "BUY" and action == "SELL":
            return True
        if current_action == "SELL" and action == "BUY":
            return True

        return False

    def _get_correlation_group(self, symbol: str) -> Optional[str]:
        """Get correlation group for a symbol."""
        for group_name, symbols in CORRELATION_GROUPS.items():
            if symbol in symbols:
                return group_name
        return None

    def _get_correlated_exposure(
        self, session: Session, symbol: str, portfolio_value: float
    ) -> float:
        """
        Calculate current exposure to correlated assets.

        Args:
            session: Database session
            symbol: Symbol to check
            portfolio_value: Total portfolio value

        Returns:
            Fraction of portfolio exposed to correlated assets (0-1)
        """
        group = self._get_correlation_group(symbol)
        if not group:
            return 0.0

        # Get current positions in same group
        from .paper_trader import PaperTrader
        from ..data.schema import Symbol, DailyBar

        correlated_symbols = CORRELATION_GROUPS[group]
        total_exposure = 0.0

        for sym in correlated_symbols:
            if sym in self.current_positions:
                # Get latest price
                symbol_obj = session.query(Symbol).filter_by(symbol=sym).first()
                if symbol_obj:
                    latest_bar = (
                        session.query(DailyBar)
                        .filter_by(symbol_id=symbol_obj.id)
                        .order_by(DailyBar.date.desc())
                        .first()
                    )
                    if latest_bar:
                        # Simplified: assume 1% risk per position
                        total_exposure += portfolio_value * 0.01

        return total_exposure / portfolio_value if portfolio_value > 0 else 0.0

    def can_trade(
        self,
        symbol: str,
        action: str,
        notional: float,
        session: Optional[Session] = None,
        portfolio_value: float = 100000.0,
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed under current constraints.

        Args:
            symbol: Asset symbol
            action: Trade action (BUY/SELL/HOLD)
            notional: Trade notional value
            session: Database session (for correlation check)
            portfolio_value: Current portfolio value (for correlation check)

        Returns:
            Tuple of (can_trade: bool, reason: str)

        Example:
            >>> can_trade, reason = constraints.can_trade("BTC-USD", "BUY", 5000.0)
            >>> if not can_trade:
            ...     print(f"Trade blocked: {reason}")
        """
        # HOLD is always allowed
        if action == "HOLD":
            return True, "HOLD action allowed"

        # Check daily trade limit
        if self._get_today_trade_count() >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # Check daily notional limit
        today_notional = self._get_today_notional()
        if today_notional + notional > self.max_daily_notional:
            return False, (
                f"Daily notional limit reached "
                f"(${today_notional:,.0f} + ${notional:,.0f} > ${self.max_daily_notional:,.0f})"
            )

        # Check cooldown for flips
        if self._is_flip(symbol, action):
            if self._is_in_cooldown(symbol):
                cooldown = self.cooldown_records[symbol]
                remaining = (cooldown.cooldown_until - datetime.utcnow()).total_seconds() / 3600
                return False, f"Symbol in cooldown (flip detected, {remaining:.1f}h remaining)"

        # Check correlated exposure
        if session:
            correlated_exposure = self._get_correlated_exposure(session, symbol, portfolio_value)
            if correlated_exposure > self.max_correlated_exposure:
                return False, (
                    f"Correlated exposure limit reached "
                    f"({correlated_exposure:.1%} > {self.max_correlated_exposure:.0%})"
                )

        return True, "Trade allowed"

    def record_trade(
        self,
        symbol: str,
        action: str,
        notional: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a trade for constraint tracking.

        Args:
            symbol: Asset symbol
            action: Trade action (BUY/SELL/HOLD)
            notional: Trade notional value
            timestamp: Trade timestamp (default: now)

        Example:
            >>> constraints.record_trade("BTC-USD", "BUY", 10000.0)
        """
        if action == "HOLD":
            return  # Don't record HOLD

        if timestamp is None:
            timestamp = datetime.utcnow()

        # Record trade
        trade = TradeRecord(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            notional=notional,
        )
        self.trade_history.append(trade)

        # Update current position
        old_action = self.current_positions.get(symbol)
        self.current_positions[symbol] = action

        # Set cooldown if flip
        if old_action and old_action != action:
            cooldown_until = timestamp + timedelta(hours=self.cooldown_hours)
            self.cooldown_records[symbol] = CooldownRecord(
                symbol=symbol,
                last_trade_time=timestamp,
                last_action=action,
                cooldown_until=cooldown_until,
            )
            logger.info(
                f"Flip detected for {symbol}: {old_action} → {action}. "
                f"Cooldown until {cooldown_until.isoformat()}"
            )

        logger.debug(f"Recorded trade: {symbol} {action} ${notional:,.2f}")

    def get_constraint_summary(self) -> dict:
        """
        Get summary of current constraint state.

        Returns:
            Dictionary with constraint metrics

        Example:
            >>> summary = constraints.get_constraint_summary()
            >>> print(f"Trades today: {summary['today_trades']}/{summary['max_daily_trades']}")
        """
        today_trades = self._get_today_trade_count()
        today_notional = self._get_today_notional()
        active_cooldowns = sum(
            1 for c in self.cooldown_records.values()
            if datetime.utcnow() < c.cooldown_until
        )

        return {
            "today_trades": today_trades,
            "max_daily_trades": self.max_daily_trades,
            "today_notional": today_notional,
            "max_daily_notional": self.max_daily_notional,
            "utilization_trades": today_trades / self.max_daily_trades if self.max_daily_trades > 0 else 0,
            "utilization_notional": today_notional / self.max_daily_notional if self.max_daily_notional > 0 else 0,
            "active_cooldowns": active_cooldowns,
            "total_positions": len(self.current_positions),
        }

    def cleanup_old_records(self, days_to_keep: int = 7):
        """
        Clean up old trade records and cooldowns.

        Args:
            days_to_keep: Number of days to keep in history

        Example:
            >>> constraints.cleanup_old_records(days_to_keep=7)
        """
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)

        # Clean trade history
        old_count = len(self.trade_history)
        self.trade_history = [t for t in self.trade_history if t.timestamp >= cutoff]
        new_count = len(self.trade_history)

        # Clean expired cooldowns
        expired_symbols = [
            symbol for symbol, cooldown in self.cooldown_records.items()
            if cooldown.cooldown_until < datetime.utcnow()
        ]
        for symbol in expired_symbols:
            del self.cooldown_records[symbol]

        logger.info(
            f"Cleanup: removed {old_count - new_count} old trades, "
            f"{len(expired_symbols)} expired cooldowns"
        )
