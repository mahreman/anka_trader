"""
Analytics layer - Returns calculation, anomaly detection, labeling, and regime detection.
"""
from .returns import compute_returns
from .anomaly import detect_anomalies_for_asset, detect_anomalies_all_assets
from .labeling import classify_anomaly
from .regime import (
    compute_regimes_for_symbol,
    compute_regimes_all_symbols,
    regimes_to_dataframe,
    persist_regimes,
    RegimePoint,
)

__all__ = [
    "compute_returns",
    "detect_anomalies_for_asset",
    "detect_anomalies_all_assets",
    "classify_anomaly",
    "compute_regimes_for_symbol",
    "compute_regimes_all_symbols",
    "regimes_to_dataframe",
    "persist_regimes",
    "RegimePoint",
]
