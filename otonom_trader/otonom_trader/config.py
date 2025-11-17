"""
Configuration settings for Otonom Trader P0.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

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
