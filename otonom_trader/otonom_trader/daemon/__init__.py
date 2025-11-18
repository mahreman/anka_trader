"""
Daemon module - Autonomous paper trading.
P3 preparation: Full pipeline automation.
"""
from .paper_trader import PaperTrader, PortfolioState
from .daemon import run_daemon_cycle, DaemonConfig, get_or_create_paper_trader

__all__ = [
    "PaperTrader",
    "PortfolioState",
    "run_daemon_cycle",
    "DaemonConfig",
    "get_or_create_paper_trader",
]
