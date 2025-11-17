"""
Analytics layer - Returns calculation, anomaly detection, and labeling.
"""
from .returns import compute_returns
from .anomaly import detect_anomalies_for_asset, detect_anomalies_all_assets
from .labeling import classify_anomaly

__all__ = [
    "compute_returns",
    "detect_anomalies_for_asset",
    "detect_anomalies_all_assets",
    "classify_anomaly",
]
