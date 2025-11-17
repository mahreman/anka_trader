"""
Backtest runner wrapper for experiments.

Provides a clean interface for running train/test backtests with
standardized metrics output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..eval.portfolio_backtest import run_backtest
from ..eval.performance_report import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """
    Standardized backtest metrics.

    Attributes:
        cagr: Compound annual growth rate (%)
        sharpe: Sharpe ratio
        sortino: Sortino ratio
        max_dd: Maximum drawdown (%)
        win_rate: Win rate (%)
        total_trades: Total number of trades
        profit_factor: Profit factor (gross profit / gross loss)
    """
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    win_rate: float
    total_trades: int
    profit_factor: float

    def __repr__(self) -> str:
        return (
            f"BacktestMetrics(CAGR={self.cagr:.2f}%, Sharpe={self.sharpe:.2f}, "
            f"MaxDD={self.max_dd:.2f}%, WinRate={self.win_rate:.1f}%)"
        )


def run_backtest_for_strategy(
    session: Session,
    symbol: str,
    strategy_cfg: Dict[str, Any],
    start_date: date,
    end_date: date,
) -> BacktestMetrics:
    """
    Run backtest for a strategy configuration.

    Args:
        session: Database session
        symbol: Asset symbol to backtest
        strategy_cfg: Strategy configuration dictionary
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        BacktestMetrics with performance results

    Example:
        >>> metrics = run_backtest_for_strategy(
        ...     session,
        ...     "BTC-USD",
        ...     strategy_config,
        ...     date(2020, 1, 1),
        ...     date(2021, 1, 1)
        ... )
        >>> print(f"CAGR: {metrics.cagr:.2f}%")
    """
    # Extract parameters from strategy config
    initial_capital = strategy_cfg.get("execution", {}).get("initial_capital", 100000.0)
    risk_per_trade = strategy_cfg.get("risk_management", {}).get(
        "position_sizing", {}
    ).get("risk_per_trade_pct", 1.0)
    use_ensemble = strategy_cfg.get("ensemble", {}).get("enabled", True)

    # Run backtest
    result = run_backtest(
        session=session,
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_cash=initial_capital,
        risk_per_trade=risk_per_trade / 100,
        use_ensemble=use_ensemble,
    )

    if not result or not result.get("equity_curve"):
        logger.warning(f"No backtest results for {symbol} {start_date} to {end_date}")
        # Return zero metrics
        return BacktestMetrics(
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_dd=0.0,
            win_rate=0.0,
            total_trades=0,
            profit_factor=0.0,
        )

    # Calculate metrics
    metrics_dict = calculate_metrics(
        equity_curve=result["equity_curve"],
        trades=result.get("trades", []),
        initial_capital=initial_capital,
    )

    # Convert to BacktestMetrics
    return BacktestMetrics(
        cagr=metrics_dict["cagr"],
        sharpe=metrics_dict["sharpe_ratio"],
        sortino=metrics_dict["sortino_ratio"],
        max_dd=metrics_dict["max_drawdown"],
        win_rate=metrics_dict["win_rate"],
        total_trades=metrics_dict["total_trades"],
        profit_factor=metrics_dict["profit_factor"],
    )


def run_train_test_backtest(
    session: Session,
    symbol: str,
    strategy_cfg: Dict[str, Any],
    train_start: date,
    train_end: date,
    test_start: Optional[date] = None,
    test_end: Optional[date] = None,
) -> Dict[str, BacktestMetrics]:
    """
    Run train/test split backtest for a strategy configuration.

    Args:
        session: Database session
        symbol: Asset symbol to backtest
        strategy_cfg: Strategy configuration dictionary
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date (optional)
        test_end: Test period end date (optional)

    Returns:
        Dictionary with "train" and "test" BacktestMetrics

    Example:
        >>> metrics = run_train_test_backtest(
        ...     session,
        ...     "BTC-USD",
        ...     strategy_config,
        ...     train_start=date(2017, 1, 1),
        ...     train_end=date(2022, 12, 31),
        ...     test_start=date(2023, 1, 1),
        ...     test_end=date(2025, 1, 1),
        ... )
        >>> print(f"Train Sharpe: {metrics['train'].sharpe:.2f}")
        >>> print(f"Test Sharpe: {metrics['test'].sharpe:.2f}")
    """
    logger.info(
        f"Running train backtest: {symbol} from {train_start} to {train_end}"
    )

    # Run training backtest
    train_metrics = run_backtest_for_strategy(
        session=session,
        symbol=symbol,
        strategy_cfg=strategy_cfg,
        start_date=train_start,
        end_date=train_end,
    )

    # Run test backtest (if specified)
    if test_start is not None and test_end is not None:
        logger.info(
            f"Running test backtest: {symbol} from {test_start} to {test_end}"
        )

        test_metrics = run_backtest_for_strategy(
            session=session,
            symbol=symbol,
            strategy_cfg=strategy_cfg,
            start_date=test_start,
            end_date=test_end,
        )
    else:
        # No test period specified
        test_metrics = BacktestMetrics(
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_dd=0.0,
            win_rate=0.0,
            total_trades=0,
            profit_factor=0.0,
        )

    return {
        "train": train_metrics,
        "test": test_metrics,
    }


def run_multi_symbol_backtest(
    session: Session,
    symbols: list[str],
    strategy_cfg: Dict[str, Any],
    start_date: date,
    end_date: date,
) -> Dict[str, BacktestMetrics]:
    """
    Run backtest across multiple symbols and aggregate results.

    Args:
        session: Database session
        symbols: List of symbols to backtest
        strategy_cfg: Strategy configuration dictionary
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Dictionary mapping symbols to BacktestMetrics

    Example:
        >>> results = run_multi_symbol_backtest(
        ...     session,
        ...     ["BTC-USD", "ETH-USD"],
        ...     strategy_config,
        ...     date(2020, 1, 1),
        ...     date(2021, 1, 1),
        ... )
        >>> for symbol, metrics in results.items():
        ...     print(f"{symbol}: CAGR={metrics.cagr:.2f}%")
    """
    results = {}

    for symbol in symbols:
        logger.info(f"Running backtest for {symbol}")

        try:
            metrics = run_backtest_for_strategy(
                session=session,
                symbol=symbol,
                strategy_cfg=strategy_cfg,
                start_date=start_date,
                end_date=end_date,
            )
            results[symbol] = metrics

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
            # Return zero metrics on failure
            results[symbol] = BacktestMetrics(
                cagr=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_dd=0.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
            )

    return results


def aggregate_backtest_metrics(
    metrics_list: list[BacktestMetrics],
) -> BacktestMetrics:
    """
    Aggregate multiple backtest metrics (e.g., across symbols).

    Takes simple average of all metrics.

    Args:
        metrics_list: List of BacktestMetrics to aggregate

    Returns:
        Aggregated BacktestMetrics

    Example:
        >>> btc_metrics = BacktestMetrics(cagr=20.0, sharpe=1.5, ...)
        >>> eth_metrics = BacktestMetrics(cagr=15.0, sharpe=1.2, ...)
        >>> avg_metrics = aggregate_backtest_metrics([btc_metrics, eth_metrics])
        >>> print(f"Average CAGR: {avg_metrics.cagr:.2f}%")
    """
    if not metrics_list:
        return BacktestMetrics(
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_dd=0.0,
            win_rate=0.0,
            total_trades=0,
            profit_factor=0.0,
        )

    n = len(metrics_list)

    return BacktestMetrics(
        cagr=sum(m.cagr for m in metrics_list) / n,
        sharpe=sum(m.sharpe for m in metrics_list) / n,
        sortino=sum(m.sortino for m in metrics_list) / n,
        max_dd=sum(m.max_dd for m in metrics_list) / n,
        win_rate=sum(m.win_rate for m in metrics_list) / n,
        total_trades=sum(m.total_trades for m in metrics_list),
        profit_factor=sum(m.profit_factor for m in metrics_list) / n,
    )
