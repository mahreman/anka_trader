"""
Unified backtest runner with single entry/single exit API.

Provides a clean, standardized interface for running backtests:
- Input: StrategyConfig + date range
- Output: Comprehensive BacktestReport

Usage:
    from otonom_trader.research.backtest_runner import run_backtest_for_strategy
    from otonom_trader.strategy.config import load_strategy_config

    cfg = load_strategy_config("strategies/baseline_v1.0.yaml")
    report = run_backtest_for_strategy(
        strategy_cfg=cfg,
        start_date=date(2018, 1, 1),
        end_date=date(2024, 1, 1),
    )

    print(f"CAGR: {report.metrics.cagr:.2f}%")
    print(f"Sharpe: {report.metrics.sharpe:.2f}")
    print(f"Total Trades: {len(report.trades)}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..data import get_session
from ..data.schema import (
    Trade as TradeORM,
    PortfolioSnapshot,
    Regime as RegimeORM,
    Symbol,
    DailyBar,
)
from ..strategy.config import StrategyConfig
from ..eval.performance_report import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics.

    Attributes:
        cagr: Compound Annual Growth Rate (%)
        sharpe: Sharpe ratio (annualized)
        sortino: Sortino ratio (annualized)
        max_drawdown: Maximum drawdown (%)
        win_rate: Win rate (0-1)
        avg_r_multiple: Average R-multiple
        profit_factor: Profit factor (gross profit / gross loss)
        total_trades: Total number of trades
        total_pnl: Total P&L ($)
    """
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0


@dataclass
class TradeRecord:
    """
    Individual trade record with risk metrics.

    Attributes:
        symbol: Asset symbol (ticker)
        entry_date: Entry date (ISO format)
        exit_date: Exit date (ISO format)
        direction: Trade direction (BUY/SELL)
        pnl: Absolute P&L ($)
        pnl_pct: Percentage P&L (%)
        r_multiple: Reward-to-risk ratio
        holding_days: Days held
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position size
    """
    symbol: str
    entry_date: str
    exit_date: str
    direction: str
    pnl: float
    pnl_pct: float
    r_multiple: float
    holding_days: int
    entry_price: float
    exit_price: float
    quantity: float


@dataclass
class RegimePerformance:
    """
    Performance breakdown by market regime.

    Attributes:
        regime_id: Regime ID (0=low vol, 1=normal, 2=high vol)
        regime_name: Regime name
        trades: Number of trades
        cagr: CAGR in this regime
        sharpe: Sharpe ratio in this regime
        max_drawdown: Max drawdown in this regime
        win_rate: Win rate in this regime
        avg_r_multiple: Average R-multiple in this regime
    """
    regime_id: int
    regime_name: str
    trades: int
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_r_multiple: float


@dataclass
class BacktestReport:
    """
    Comprehensive backtest report.

    This is the single output format for all backtests.

    Attributes:
        strategy_name: Strategy name
        strategy_version: Strategy version
        start_date: Backtest start date (ISO format)
        end_date: Backtest end date (ISO format)
        universe: List of symbols tested
        initial_capital: Starting capital
        final_equity: Final equity
        metrics: Performance metrics
        trades: List of trade records
        equity_curve: Equity curve [{date, equity, drawdown}]
        regime_breakdown: Performance by regime
        raw: Additional metadata
    """
    strategy_name: str
    strategy_version: str
    start_date: str
    end_date: str
    universe: List[str]
    initial_capital: float
    final_equity: float

    metrics: PerformanceMetrics
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    regime_breakdown: List[RegimePerformance] = field(default_factory=list)

    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Saved report to: {path}")


def _run_core_backtest(
    session: Session,
    strategy_cfg: StrategyConfig,
    start_date: date,
    end_date: date,
    universe: List[str],
) -> None:
    """
    Run core backtest engine.

    This is a placeholder that should connect to your existing backtest engine.
    For now, it's a simplified implementation that assumes decisions already exist
    in the database and simulates trades based on them.

    Args:
        session: Database session
        strategy_cfg: Strategy configuration
        start_date: Start date
        end_date: End date
        universe: List of symbols to trade
    """
    # TODO: Connect to your actual backtest engine
    # For now, this is a placeholder that assumes:
    # 1. Decisions already exist in the database
    # 2. We simulate trades based on decisions
    # 3. We write Trade and PortfolioSnapshot records

    logger.info(f"Running backtest: {strategy_cfg.name} v{strategy_cfg.version}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Universe: {universe}")

    # This would call your existing backtest engine
    # Example: from ..eval.portfolio_backtest import run_backtest
    pass


def _collect_trades(
    session: Session,
    strategy_name: str,
    strategy_version: str,
    start_date: date,
    end_date: date,
) -> List[TradeRecord]:
    """
    Collect trades from database.

    Args:
        session: Database session
        strategy_name: Strategy name filter
        strategy_version: Strategy version filter
        start_date: Start date
        end_date: End date

    Returns:
        List of TradeRecord objects
    """
    # Query trades
    query = session.query(TradeORM).join(Symbol)

    # Filter by strategy if provided
    if strategy_name:
        query = query.filter(TradeORM.strategy_name == strategy_name)
    if strategy_version:
        query = query.filter(TradeORM.strategy_version == strategy_version)

    # Filter by date
    query = query.filter(
        TradeORM.entry_date >= start_date,
        TradeORM.exit_date <= end_date,
        TradeORM.exit_date.isnot(None),  # Only closed trades
    )

    # Execute query
    trades = query.all()

    # Convert to TradeRecord
    records = []
    for t in trades:
        records.append(
            TradeRecord(
                symbol=t.symbol_obj.symbol,
                entry_date=t.entry_date.isoformat(),
                exit_date=t.exit_date.isoformat() if t.exit_date else "",
                direction=t.direction,
                pnl=float(t.pnl) if t.pnl is not None else 0.0,
                pnl_pct=float(t.pnl_pct) if t.pnl_pct is not None else 0.0,
                r_multiple=float(t.r_multiple) if t.r_multiple is not None else 0.0,
                holding_days=int(t.holding_days) if t.holding_days is not None else 0,
                entry_price=float(t.entry_price),
                exit_price=float(t.exit_price) if t.exit_price is not None else 0.0,
                quantity=float(t.quantity),
            )
        )

    logger.info(f"Collected {len(records)} trades from database")
    return records


def _collect_equity_curve(
    session: Session,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Collect equity curve from database.

    Args:
        session: Database session
        start_date: Start date
        end_date: End date

    Returns:
        List of equity points [{date, equity, drawdown}]
    """
    # Query portfolio snapshots
    snapshots = (
        session.query(PortfolioSnapshot)
        .filter(
            PortfolioSnapshot.timestamp >= start_date,
            PortfolioSnapshot.timestamp <= end_date,
        )
        .order_by(PortfolioSnapshot.timestamp.asc())
        .all()
    )

    if not snapshots:
        logger.warning("No portfolio snapshots found in database")
        return []

    # Convert to list
    curve = []
    for s in snapshots:
        curve.append({
            "date": s.timestamp.date().isoformat(),
            "equity": float(s.equity),
            "drawdown": float(s.max_drawdown) if s.max_drawdown is not None else 0.0,
            "cash": float(s.cash),
            "positions_value": float(s.positions_value),
        })

    logger.info(f"Collected equity curve with {len(curve)} points")
    return curve


def _compute_metrics(
    trades: List[TradeRecord],
    equity_curve: List[Dict[str, Any]],
    initial_capital: float,
    start_date: date,
    end_date: date,
) -> PerformanceMetrics:
    """
    Compute performance metrics from trades and equity curve.

    Args:
        trades: List of trade records
        equity_curve: Equity curve data
        initial_capital: Starting capital
        start_date: Start date
        end_date: End date

    Returns:
        PerformanceMetrics object
    """
    if not trades and not equity_curve:
        logger.warning("No trades or equity curve data, returning empty metrics")
        return PerformanceMetrics()

    # Calculate years
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        years = 1.0

    # Total P&L and trade stats
    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_r = sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # CAGR
    if equity_curve:
        final_equity = equity_curve[-1]["equity"]
        cagr = calculate_cagr(initial_capital, final_equity, years)
    else:
        final_equity = initial_capital + total_pnl
        cagr = calculate_cagr(initial_capital, final_equity, years)

    # Sharpe and Sortino (from equity curve returns)
    sharpe = 0.0
    sortino = 0.0
    max_dd = 0.0

    if equity_curve and len(equity_curve) > 1:
        # Calculate daily returns
        equity_series = pd.Series([p["equity"] for p in equity_curve])
        returns = equity_series.pct_change().dropna()

        if len(returns) > 0:
            sharpe = calculate_sharpe_ratio(returns)
            sortino = calculate_sortino_ratio(returns)

        # Max drawdown
        max_dd = calculate_max_drawdown(equity_series)

    return PerformanceMetrics(
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
        avg_r_multiple=float(avg_r),
        profit_factor=float(profit_factor),
        total_trades=len(trades),
        total_pnl=float(total_pnl),
    )


def _compute_regime_breakdown(
    session: Session,
    trades: List[TradeRecord],
    start_date: date,
    end_date: date,
) -> List[RegimePerformance]:
    """
    Compute performance breakdown by market regime.

    Args:
        session: Database session
        trades: List of trade records
        start_date: Start date
        end_date: End date

    Returns:
        List of RegimePerformance objects
    """
    if not trades:
        return []

    # Group trades by regime
    # For each trade, find the regime at entry date
    regime_trades: Dict[int, List[TradeRecord]] = {0: [], 1: [], 2: []}

    for trade in trades:
        # Parse entry date
        entry = date.fromisoformat(trade.entry_date)

        # Find regime for this symbol at entry date
        symbol = session.query(Symbol).filter_by(symbol=trade.symbol).first()
        if not symbol:
            continue

        regime = (
            session.query(RegimeORM)
            .filter(
                RegimeORM.symbol_id == symbol.id,
                RegimeORM.date == entry,
            )
            .first()
        )

        if regime:
            regime_id = regime.regime_id
            if regime_id in regime_trades:
                regime_trades[regime_id].append(trade)

    # Compute metrics for each regime
    regime_names = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}
    breakdown = []

    for regime_id, regime_trade_list in regime_trades.items():
        if not regime_trade_list:
            continue

        # Calculate metrics
        wins = [t for t in regime_trade_list if t.pnl > 0]
        win_rate = len(wins) / len(regime_trade_list) if regime_trade_list else 0.0
        avg_r = sum(t.r_multiple for t in regime_trade_list) / len(regime_trade_list)

        # Simple CAGR/Sharpe calculation (simplified)
        total_pnl = sum(t.pnl for t in regime_trade_list)
        pnl_list = [t.pnl for t in regime_trade_list]

        # Calculate Sharpe from P&L list
        if len(pnl_list) > 1:
            mean_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list)
            sharpe = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        breakdown.append(
            RegimePerformance(
                regime_id=regime_id,
                regime_name=regime_names.get(regime_id, f"Regime {regime_id}"),
                trades=len(regime_trade_list),
                cagr=0.0,  # Simplified
                sharpe=float(sharpe),
                max_drawdown=0.0,  # Simplified
                win_rate=float(win_rate),
                avg_r_multiple=float(avg_r),
            )
        )

    return breakdown


def run_backtest_for_strategy(
    strategy_cfg: StrategyConfig,
    start_date: date,
    end_date: date,
    universe_override: Optional[List[str]] = None,
    run_backtest: bool = True,
) -> BacktestReport:
    """
    Run backtest for a strategy configuration.

    This is the single entry point for all backtests.

    Args:
        strategy_cfg: Strategy configuration (StrategyConfig)
        start_date: Backtest start date
        end_date: Backtest end date
        universe_override: Override universe symbols (optional)
        run_backtest: Whether to run the core backtest engine (default: True)
                      Set to False if you just want to collect existing results

    Returns:
        BacktestReport with comprehensive results

    Example:
        >>> from otonom_trader.strategy.config import load_strategy_config
        >>> cfg = load_strategy_config("strategies/baseline_v1.0.yaml")
        >>> report = run_backtest_for_strategy(
        ...     strategy_cfg=cfg,
        ...     start_date=date(2018, 1, 1),
        ...     end_date=date(2024, 1, 1),
        ... )
        >>> print(f"CAGR: {report.metrics.cagr:.2f}%")
        >>> print(f"Sharpe: {report.metrics.sharpe:.2f}")
    """
    logger.info("=" * 60)
    logger.info(f"Running backtest: {strategy_cfg.name} v{strategy_cfg.version}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Determine universe
    universe = universe_override or strategy_cfg.get_symbols()
    initial_capital = strategy_cfg.get_initial_capital()

    logger.info(f"Universe: {universe}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")

    with get_session() as session:
        # Step 1: Run core backtest (if requested)
        if run_backtest:
            logger.info("Step 1: Running core backtest engine...")
            _run_core_backtest(session, strategy_cfg, start_date, end_date, universe)
        else:
            logger.info("Step 1: Skipping core backtest (collecting existing results)")

        # Step 2: Collect trades
        logger.info("Step 2: Collecting trades from database...")
        trades = _collect_trades(
            session,
            strategy_cfg.name,
            strategy_cfg.version,
            start_date,
            end_date,
        )

        # Step 3: Collect equity curve
        logger.info("Step 3: Collecting equity curve...")
        equity_curve = _collect_equity_curve(session, start_date, end_date)

        # Step 4: Compute metrics
        logger.info("Step 4: Computing performance metrics...")
        metrics = _compute_metrics(
            trades,
            equity_curve,
            initial_capital,
            start_date,
            end_date,
        )

        # Step 5: Compute regime breakdown
        logger.info("Step 5: Computing regime breakdown...")
        regime_breakdown = _compute_regime_breakdown(session, trades, start_date, end_date)

        # Step 6: Create report
        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital

        report = BacktestReport(
            strategy_name=strategy_cfg.name,
            strategy_version=strategy_cfg.version,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            universe=universe,
            initial_capital=initial_capital,
            final_equity=final_equity,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            regime_breakdown=regime_breakdown,
            raw={"config": strategy_cfg.to_dict() if hasattr(strategy_cfg, "to_dict") else {}},
        )

    logger.info("=" * 60)
    logger.info("Backtest complete!")
    logger.info(f"CAGR: {metrics.cagr:.2f}%")
    logger.info(f"Sharpe: {metrics.sharpe:.2f}")
    logger.info(f"Max DD: {metrics.max_drawdown:.2f}%")
    logger.info(f"Win Rate: {metrics.win_rate:.1%}")
    logger.info(f"Total Trades: {metrics.total_trades}")
    logger.info("=" * 60)

    return report


def save_report_as_json(report: BacktestReport, path: str | Path) -> None:
    """
    Save backtest report as JSON file.

    Args:
        report: BacktestReport to save
        path: Output path

    Example:
        >>> save_report_as_json(report, "reports/baseline_v1.0_2018_2024.json")
    """
    report.save_json(path)
