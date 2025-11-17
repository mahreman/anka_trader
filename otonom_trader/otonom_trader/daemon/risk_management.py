"""
Risk management module for stop-loss and take-profit (P3 preparation).

This module provides position-level risk management:
1. Stop-loss: Exit when loss exceeds threshold
2. Take-profit: Exit when profit reaches target
3. Trailing stop: Dynamic stop-loss that follows price

Example:
    >>> risk_mgr = RiskManager(stop_loss_pct=5.0, take_profit_pct=10.0)
    >>> if risk_mgr.should_exit_position(position, current_price):
    ...     # Exit the position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskLevels:
    """
    Risk levels for a position.

    Attributes:
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        trailing_stop: Trailing stop price (optional)
        entry_price: Entry price of position
        current_price: Current market price
    """
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    entry_price: float
    current_price: float

    def __repr__(self) -> str:
        return (
            f"RiskLevels(entry={self.entry_price:.2f}, "
            f"SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, "
            f"current={self.current_price:.2f})"
        )


@dataclass
class ExitSignal:
    """
    Exit signal for a position.

    Attributes:
        should_exit: Whether to exit
        reason: Reason for exit
        exit_price: Suggested exit price
        pnl_pct: PnL percentage at exit
    """
    should_exit: bool
    reason: str
    exit_price: float
    pnl_pct: float


class RiskManager:
    """
    Position risk management engine.

    Implements stop-loss, take-profit, and trailing stop logic.

    Example:
        >>> risk_mgr = RiskManager(
        ...     stop_loss_pct=5.0,
        ...     take_profit_pct=10.0,
        ...     use_trailing_stop=True,
        ...     trailing_stop_distance_pct=3.0,
        ... )
        >>> levels = risk_mgr.calculate_risk_levels(entry_price=100.0, direction="BUY")
        >>> print(levels)  # RiskLevels(entry=100.00, SL=95.00, TP=110.00, current=100.00)
    """

    def __init__(
        self,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 10.0,
        use_trailing_stop: bool = False,
        trailing_stop_distance_pct: float = 3.0,
    ):
        """
        Initialize risk manager.

        Args:
            stop_loss_pct: Stop-loss percentage (e.g., 5.0 for 5%)
            take_profit_pct: Take-profit percentage (e.g., 10.0 for 10%)
            use_trailing_stop: Enable trailing stop
            trailing_stop_distance_pct: Distance for trailing stop (e.g., 3.0 for 3%)
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_distance_pct = trailing_stop_distance_pct

        # Track trailing stops per symbol
        self.trailing_stops: dict[str, float] = {}

        logger.info(
            f"RiskManager initialized: SL={stop_loss_pct}%, TP={take_profit_pct}%, "
            f"trailing={'ON' if use_trailing_stop else 'OFF'}"
        )

    def calculate_risk_levels(
        self,
        entry_price: float,
        direction: str,
        current_price: Optional[float] = None,
    ) -> RiskLevels:
        """
        Calculate stop-loss and take-profit levels for a position.

        Args:
            entry_price: Entry price of position
            direction: Position direction (BUY/SELL)
            current_price: Current market price (default: entry_price)

        Returns:
            RiskLevels with SL/TP prices

        Example:
            >>> levels = risk_mgr.calculate_risk_levels(100.0, "BUY")
            >>> print(f"Stop loss at: ${levels.stop_loss:.2f}")
        """
        if current_price is None:
            current_price = entry_price

        if direction == "BUY":
            # Long position
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.take_profit_pct / 100)

            # Trailing stop (follows price up)
            if self.use_trailing_stop and current_price > entry_price:
                trailing_stop = current_price * (1 - self.trailing_stop_distance_pct / 100)
                trailing_stop = max(stop_loss, trailing_stop)  # Never below initial SL
            else:
                trailing_stop = None

        else:  # SELL (short position)
            # Short position (inverse logic)
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.take_profit_pct / 100)

            # Trailing stop (follows price down)
            if self.use_trailing_stop and current_price < entry_price:
                trailing_stop = current_price * (1 + self.trailing_stop_distance_pct / 100)
                trailing_stop = min(stop_loss, trailing_stop)  # Never above initial SL
            else:
                trailing_stop = None

        return RiskLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            entry_price=entry_price,
            current_price=current_price,
        )

    def check_exit_conditions(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        direction: str,
    ) -> ExitSignal:
        """
        Check if position should be exited based on risk levels.

        Args:
            symbol: Asset symbol
            entry_price: Entry price of position
            current_price: Current market price
            direction: Position direction (BUY/SELL)

        Returns:
            ExitSignal indicating whether to exit

        Example:
            >>> signal = risk_mgr.check_exit_conditions("BTC-USD", 100.0, 92.0, "BUY")
            >>> if signal.should_exit:
            ...     print(f"Exit reason: {signal.reason}")
        """
        # Calculate risk levels
        levels = self.calculate_risk_levels(entry_price, direction, current_price)

        # Calculate PnL
        if direction == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # Check stop-loss
        if direction == "BUY":
            if current_price <= levels.stop_loss:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Stop-loss hit (${current_price:.2f} <= ${levels.stop_loss:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

            # Check trailing stop
            if levels.trailing_stop and current_price <= levels.trailing_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Trailing stop hit (${current_price:.2f} <= ${levels.trailing_stop:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

            # Check take-profit
            if current_price >= levels.take_profit:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Take-profit hit (${current_price:.2f} >= ${levels.take_profit:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

        else:  # SELL (short)
            if current_price >= levels.stop_loss:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Stop-loss hit (${current_price:.2f} >= ${levels.stop_loss:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

            # Check trailing stop
            if levels.trailing_stop and current_price >= levels.trailing_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Trailing stop hit (${current_price:.2f} >= ${levels.trailing_stop:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

            # Check take-profit
            if current_price <= levels.take_profit:
                return ExitSignal(
                    should_exit=True,
                    reason=f"Take-profit hit (${current_price:.2f} <= ${levels.take_profit:.2f})",
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                )

        # No exit signal
        return ExitSignal(
            should_exit=False,
            reason="Position within risk bounds",
            exit_price=current_price,
            pnl_pct=pnl_pct,
        )

    def update_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        direction: str,
    ):
        """
        Update trailing stop for a position.

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            direction: Position direction (BUY/SELL)

        Example:
            >>> risk_mgr.update_trailing_stop("BTC-USD", 100.0, 105.0, "BUY")
        """
        if not self.use_trailing_stop:
            return

        levels = self.calculate_risk_levels(entry_price, direction, current_price)

        if levels.trailing_stop:
            old_trailing = self.trailing_stops.get(symbol)

            # Update only if new trailing stop is better
            if old_trailing is None:
                self.trailing_stops[symbol] = levels.trailing_stop
                logger.debug(f"Set trailing stop for {symbol}: ${levels.trailing_stop:.2f}")
            elif direction == "BUY" and levels.trailing_stop > old_trailing:
                self.trailing_stops[symbol] = levels.trailing_stop
                logger.debug(
                    f"Updated trailing stop for {symbol}: "
                    f"${old_trailing:.2f} → ${levels.trailing_stop:.2f}"
                )
            elif direction == "SELL" and levels.trailing_stop < old_trailing:
                self.trailing_stops[symbol] = levels.trailing_stop
                logger.debug(
                    f"Updated trailing stop for {symbol}: "
                    f"${old_trailing:.2f} → ${levels.trailing_stop:.2f}"
                )

    def remove_trailing_stop(self, symbol: str):
        """
        Remove trailing stop for a symbol (e.g., after exit).

        Args:
            symbol: Asset symbol

        Example:
            >>> risk_mgr.remove_trailing_stop("BTC-USD")
        """
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
            logger.debug(f"Removed trailing stop for {symbol}")

    def get_risk_summary(self) -> dict:
        """
        Get summary of risk management state.

        Returns:
            Dictionary with risk metrics

        Example:
            >>> summary = risk_mgr.get_risk_summary()
            >>> print(f"Active trailing stops: {summary['active_trailing_stops']}")
        """
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "use_trailing_stop": self.use_trailing_stop,
            "trailing_stop_distance_pct": self.trailing_stop_distance_pct,
            "active_trailing_stops": len(self.trailing_stops),
        }


class VolatilityScaledRisk(RiskManager):
    """
    Risk manager with volatility-based scaling.

    Adjusts stop-loss and take-profit based on asset volatility.

    Example:
        >>> vol_risk = VolatilityScaledRisk(
        ...     base_stop_loss_pct=5.0,
        ...     vol_multiplier=1.5,
        ... )
        >>> # For high volatility asset (vol=0.03):
        >>> # Actual stop-loss = 5% * (0.03 / 0.02) * 1.5 = 11.25%
    """

    def __init__(
        self,
        base_stop_loss_pct: float = 5.0,
        base_take_profit_pct: float = 10.0,
        vol_multiplier: float = 1.5,
        reference_vol: float = 0.02,  # 2% daily vol as reference
        **kwargs,
    ):
        """
        Initialize volatility-scaled risk manager.

        Args:
            base_stop_loss_pct: Base stop-loss percentage
            base_take_profit_pct: Base take-profit percentage
            vol_multiplier: Volatility scaling factor
            reference_vol: Reference volatility level
            **kwargs: Additional arguments for base RiskManager
        """
        super().__init__(
            stop_loss_pct=base_stop_loss_pct,
            take_profit_pct=base_take_profit_pct,
            **kwargs,
        )
        self.base_stop_loss_pct = base_stop_loss_pct
        self.base_take_profit_pct = base_take_profit_pct
        self.vol_multiplier = vol_multiplier
        self.reference_vol = reference_vol

    def scale_risk_by_volatility(self, volatility: float):
        """
        Scale stop-loss and take-profit based on volatility.

        Args:
            volatility: Asset volatility (e.g., 0.03 for 3%)

        Example:
            >>> vol_risk.scale_risk_by_volatility(0.03)  # 3% volatility
            >>> print(f"Scaled SL: {vol_risk.stop_loss_pct:.2f}%")
        """
        # Scale factor = (current_vol / reference_vol) * multiplier
        scale = (volatility / self.reference_vol) * self.vol_multiplier

        # Clamp to reasonable range [0.5, 3.0]
        scale = max(0.5, min(3.0, scale))

        # Apply scaling
        self.stop_loss_pct = self.base_stop_loss_pct * scale
        self.take_profit_pct = self.base_take_profit_pct * scale

        logger.debug(
            f"Volatility scaling: vol={volatility:.4f}, scale={scale:.2f}, "
            f"SL={self.stop_loss_pct:.2f}%, TP={self.take_profit_pct:.2f}%"
        )
