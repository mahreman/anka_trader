"""
Portfolio-based backtest implementation (P2.5).

Uses event-based backtest output to:
- Open/close positions
- Calculate equity, cash, and drawdown
- Write results to PortfolioSnapshot table
- Generate portfolio performance metrics

Features:
- Position sizing based on risk per trade
- Equity curve tracking
- Drawdown monitoring
- Win rate and PnL statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from collections import defaultdict
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..data.schema import DailyBar, Symbol, HypothesisResult, Hypothesis
from ..data.schema_intraday_and_portfolio import (
    PortfolioPosition,
    PortfolioSnapshot,
)

logger = logging.getLogger(__name__)


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


@dataclass
class PortfolioMetrics:
    """
    Summary metrics for portfolio backtest.

    Attributes:
        total_trades: Total number of trades
        total_pnl: Total profit/loss
        final_equity: Final portfolio equity
        max_drawdown: Maximum drawdown from peak
        win_rate: Percentage of winning trades
        avg_win: Average win amount
        avg_loss: Average loss amount
        sharpe_ratio: Sharpe ratio (if applicable)
    """

    total_trades: int
    total_pnl: float
    final_equity: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: Optional[float] = None


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


def calculate_portfolio_position_size(
    pf: SimplePortfolio,
    entry_price: float,
    config: PortfolioConfig,
) -> float:
    """
    Calculate position size based on risk management rules.

    Simple approach: Risk a percentage of equity per trade.
    Position size = (equity * risk_pct) / entry_price

    Args:
        pf: Portfolio instance
        entry_price: Entry price for the trade
        config: Portfolio configuration

    Returns:
        Position size (number of units)
    """
    if entry_price <= 0:
        return 0.0

    # Risk amount in dollars
    risk_amount = pf.equity * (config.risk_per_trade_pct / 100.0)

    # Position size
    qty = risk_amount / entry_price

    return qty


def run_portfolio_backtest(
    session: Session,
    hypothesis_id: int,
    config: PortfolioConfig,
) -> PortfolioMetrics:
    """
    Run portfolio-based backtest using event-based backtest results.

    Replays all trades from HypothesisResult table and simulates
    portfolio behavior with position sizing and equity tracking.

    Args:
        session: Database session
        hypothesis_id: Hypothesis ID to backtest
        config: Portfolio configuration

    Returns:
        PortfolioMetrics with summary statistics

    Raises:
        ValueError: If hypothesis not found or no trades available
    """
    # Verify hypothesis exists
    hypothesis = session.query(Hypothesis).filter(Hypothesis.id == hypothesis_id).one_or_none()
    if hypothesis is None:
        raise ValueError(f"Hypothesis ID {hypothesis_id} not found")

    logger.info("Starting portfolio backtest for hypothesis: %s", hypothesis.name)

    # Load all trades for this hypothesis, sorted by entry date
    trades = (
        session.query(HypothesisResult)
        .filter(HypothesisResult.hypothesis_id == hypothesis_id)
        .order_by(HypothesisResult.entry_date)
        .all()
    )

    if not trades:
        raise ValueError(f"No trades found for hypothesis ID {hypothesis_id}")

    logger.info("Found %d trades to replay", len(trades))

    # Initialize portfolio
    pf = SimplePortfolio(config)

    # Track trades for metrics
    trade_pnls = []
    trade_positions = {}  # Map symbol_id -> position for each trade

    # Replay trades chronologically
    for trade in trades:
        logger.debug(
            "Processing trade: symbol_id=%d, entry=%s, exit=%s",
            trade.symbol_id,
            trade.entry_date,
            trade.exit_date,
        )

        # Calculate position size
        qty = calculate_portfolio_position_size(pf, trade.entry_price, config)

        if qty <= 0:
            logger.warning("Skipping trade: calculated qty=%.2f <= 0", qty)
            continue

        # Check if we have enough cash
        required_cash = qty * trade.entry_price
        if required_cash > pf.cash:
            logger.warning(
                "Insufficient cash for trade: need %.2f, have %.2f",
                required_cash,
                pf.cash,
            )
            continue

        # Open position at entry
        try:
            pf.open_position(
                session,
                symbol_id=trade.symbol_id,
                qty=qty,
                entry_price=trade.entry_price,
                opened_at=datetime.combine(trade.entry_date, datetime.min.time()),
            )

            # Store position reference for this trade
            trade_key = f"{trade.symbol_id}_{trade.entry_date}"
            trade_positions[trade_key] = pf.positions[-1]  # Last added position

            # Record snapshot at entry
            record_snapshot(
                session,
                datetime.combine(trade.entry_date, datetime.min.time()),
                pf,
            )

        except Exception as e:
            logger.error("Failed to open position: %s", e)
            continue

        # Close position at exit
        try:
            pos = trade_positions[trade_key]

            # Close the position
            realized_pnl = pf.close_position(session, pos, trade.exit_price)
            trade_pnls.append(realized_pnl)

            # Update equity at exit
            pf.update_equity(session, trade.exit_date)

            # Record snapshot at exit
            record_snapshot(
                session,
                datetime.combine(trade.exit_date, datetime.min.time()),
                pf,
            )

            logger.debug("Trade PnL: %.2f (qty=%.2f)", realized_pnl, qty)

        except Exception as e:
            logger.error("Failed to close position: %s", e)
            continue

    # Calculate metrics
    metrics = _calculate_metrics(pf, trade_pnls, config)

    logger.info("Portfolio backtest completed:")
    logger.info("  Total trades: %d", metrics.total_trades)
    logger.info("  Total PnL: %.2f", metrics.total_pnl)
    logger.info("  Final equity: %.2f", metrics.final_equity)
    logger.info("  Max drawdown: %.2f%%", metrics.max_drawdown * 100)
    logger.info("  Win rate: %.2f%%", metrics.win_rate * 100)

    return metrics


def _calculate_metrics(
    pf: SimplePortfolio,
    trade_pnls: List[float],
    config: PortfolioConfig,
) -> PortfolioMetrics:
    """
    Calculate portfolio performance metrics.

    Args:
        pf: Portfolio instance
        trade_pnls: List of realized PnLs from all trades
        config: Portfolio configuration

    Returns:
        PortfolioMetrics object
    """
    if not trade_pnls:
        return PortfolioMetrics(
            total_trades=0,
            total_pnl=0.0,
            final_equity=config.initial_cash,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
        )

    # Basic metrics
    total_trades = len(trade_pnls)
    total_pnl = sum(trade_pnls)
    final_equity = pf.equity

    # Win/loss metrics
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    # Drawdown (already tracked in portfolio)
    max_drawdown = pf.drawdown

    return PortfolioMetrics(
        total_trades=total_trades,
        total_pnl=total_pnl,
        final_equity=final_equity,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def get_portfolio_summary(
    session: Session,
    hypothesis_id: int,
) -> Dict[str, Any]:
    """
    Get summary statistics for a portfolio backtest.

    Queries PortfolioSnapshot table for equity curve data.

    Args:
        session: Database session
        hypothesis_id: Hypothesis ID

    Returns:
        Dictionary with summary statistics
    """
    # Get all snapshots
    snapshots = (
        session.query(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.ts)
        .all()
    )

    if not snapshots:
        return {
            "total_snapshots": 0,
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "max_equity": 0.0,
            "max_drawdown": 0.0,
        }

    initial_equity = snapshots[0].equity
    final_equity = snapshots[-1].equity
    max_equity = max(s.equity for s in snapshots)
    max_drawdown = min(s.max_drawdown for s in snapshots)

    return {
        "total_snapshots": len(snapshots),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "max_equity": max_equity,
        "max_drawdown": max_drawdown,
        "total_return_pct": ((final_equity - initial_equity) / initial_equity * 100)
        if initial_equity > 0
        else 0.0,
    }
