"""
Portfolio-based backtest skeleton (P2.5/P3).

Uses event-based backtest output to:
- Open/close positions
- Calculate equity, cash, and drawdown
- Write results to PortfolioSnapshot table

This is a minimal skeleton. Extend with:
- Position sizing logic
- Multi-position portfolio management
- Risk management rules
- Margin and leverage handling
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from ..data.schema import DailyBar, Symbol
from ..data.schema_intraday_and_portfolio import (
    PortfolioPosition,
    PortfolioSnapshot,
)


@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio backtest.

    Attributes:
        initial_cash: Starting cash balance
        risk_per_trade_pct: Percentage of equity to risk per trade
    """

    initial_cash: float = 100000.0
    risk_per_trade_pct: float = 1.0  # % of equity to risk per trade


class SimplePortfolio:
    """
    Simple portfolio manager for backtesting.

    Tracks cash, equity, positions, and drawdown.
    """

    def __init__(self, cfg: PortfolioConfig):
        """
        Initialize portfolio with configuration.

        Args:
            cfg: Portfolio configuration
        """
        self.cfg = cfg
        self.cash = cfg.initial_cash
        self.equity = cfg.initial_cash
        self.positions: List[PortfolioPosition] = []
        self.max_equity = cfg.initial_cash

    def update_equity(self, session: Session, current_date: date) -> None:
        """
        Update portfolio equity based on current market prices.

        Mark-to-market all open positions using latest bar close prices.

        Args:
            session: Database session
            current_date: Date to use for pricing
        """
        total_pos_value = 0.0

        for pos in self.positions:
            # Get latest bar for this symbol
            bar = (
                session.query(DailyBar)
                .filter(
                    DailyBar.symbol_id == pos.symbol_id,
                    DailyBar.date == current_date,
                )
                .one_or_none()
            )

            if bar is None:
                # No bar available - use entry price as fallback
                total_pos_value += pos.qty * pos.entry_price
                continue

            # Mark to market
            total_pos_value += pos.qty * bar.close

        # Update equity and max equity
        self.equity = self.cash + total_pos_value
        self.max_equity = max(self.max_equity, self.equity)

    @property
    def drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity.

        Returns:
            Drawdown as decimal (negative value, e.g., -0.15 for 15% drawdown)
        """
        if self.max_equity <= 0:
            return 0.0
        return (self.equity - self.max_equity) / self.max_equity

    def open_position(
        self,
        session: Session,
        symbol_id: int,
        qty: float,
        entry_price: float,
        opened_at: datetime,
    ) -> None:
        """
        Open a new position.

        Args:
            session: Database session
            symbol_id: Symbol ID
            qty: Quantity (positive for long, negative for short)
            entry_price: Entry price
            opened_at: Timestamp when position was opened
        """
        cost = abs(qty * entry_price)

        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need {cost}, have {self.cash}")

        # Deduct from cash
        self.cash -= cost

        # Create position
        pos = PortfolioPosition(
            symbol_id=symbol_id,
            opened_at=opened_at,
            qty=qty,
            entry_price=entry_price,
        )
        self.positions.append(pos)
        session.add(pos)

    def close_position(
        self,
        session: Session,
        pos: PortfolioPosition,
        exit_price: float,
    ) -> float:
        """
        Close an existing position.

        Args:
            session: Database session
            pos: Position to close
            exit_price: Exit price

        Returns:
            PnL from closing the position
        """
        # Calculate PnL
        pnl = pos.qty * (exit_price - pos.entry_price)

        # Add proceeds to cash
        proceeds = pos.qty * exit_price
        self.cash += proceeds

        # Remove position
        self.positions.remove(pos)
        session.delete(pos)

        return pnl


def record_snapshot(
    session: Session,
    timestamp: datetime,
    pf: SimplePortfolio,
) -> None:
    """
    Record portfolio snapshot to database.

    Args:
        session: Database session
        timestamp: Snapshot timestamp
        pf: Portfolio instance
    """
    snap = PortfolioSnapshot(
        ts=timestamp,
        equity=pf.equity,
        cash=pf.cash,
        max_drawdown=pf.drawdown,
    )
    session.add(snap)
    session.flush()


def run_portfolio_backtest(
    session: Session,
    config: PortfolioConfig,
) -> None:
    """
    Run portfolio-based backtest (skeleton).

    TODO: Integrate with event-based backtest results:
    - Load HypothesisResult records
    - Simulate position opening/closing
    - Track equity curve
    - Record snapshots

    Args:
        session: Database session
        config: Portfolio configuration
    """
    pf = SimplePortfolio(config)

    # TODO: Implement trade replay logic
    # For each trade from event backtest:
    #   - Open position at entry
    #   - Close position at exit
    #   - Update equity
    #   - Record snapshot

    # Example skeleton:
    # trades = session.query(HypothesisResult).order_by(HypothesisResult.entry_date).all()
    # for trade in trades:
    #     pf.open_position(session, trade.symbol_id, qty, trade.entry_price, trade.entry_date)
    #     # ... wait until exit_date
    #     pf.close_position(session, pos, trade.exit_price)
    #     pf.update_equity(session, trade.exit_date)
    #     record_snapshot(session, trade.exit_date, pf)

    pass
