"""
Configuration management for strategies and system settings.
"""
import os
from pathlib import Path

from .strategy_loader import (
    StrategyConfig,
    UniverseConfig,
    RiskConfig,
    FiltersConfig,
    EnsembleConfig,
    ExecutionConfig,
    load_strategy,
    validate_strategy_config,
)

# System-wide configuration constants
# (Previously in otonom_trader/config.py, now consolidated here)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Database settings
DB_PATH = os.getenv("OTONOM_DB_PATH", str(PROJECT_ROOT / "otonom_trader.db"))

# Data fetching settings
DEFAULT_START_DATE = "2013-01-01"  # 10+ years of data
DEFAULT_END_DATE = None  # None means today

# Anomaly detection parameters
ANOMALY_ZSCORE_THRESHOLD = 2.5  # k parameter
ANOMALY_VOLUME_QUANTILE = 0.8  # q parameter
ANOMALY_ROLLING_WINDOW = 60  # days

# Patron (decision engine) parameters
TREND_WINDOW = 20  # days for trend calculation
TREND_THRESHOLD = 0.02  # 2% difference to consider as trend

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

__all__ = [
    # Strategy configuration
    "StrategyConfig",
    "UniverseConfig",
    "RiskConfig",
    "FiltersConfig",
    "EnsembleConfig",
    "ExecutionConfig",
    "load_strategy",
    "validate_strategy_config",
    # System constants
    "PROJECT_ROOT",
    "DB_PATH",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "ANOMALY_ZSCORE_THRESHOLD",
    "ANOMALY_VOLUME_QUANTILE",
    "ANOMALY_ROLLING_WINDOW",
    "TREND_WINDOW",
    "TREND_THRESHOLD",
    "LOG_LEVEL",
]
