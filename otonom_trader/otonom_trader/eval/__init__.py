"""
Evaluation layer - Backtesting and hypothesis tracking.
"""
from .backtest import (
    BacktestConfig,
    create_or_get_hypothesis,
    run_event_backtest,
    get_backtest_summary,
)

__all__ = [
    "BacktestConfig",
    "create_or_get_hypothesis",
    "run_event_backtest",
    "get_backtest_summary",
]
