"""
Anomaly type classification and labeling.
"""
from ..domain import AnomalyType


def classify_anomaly(
    zscore: float,
    volume_quantile: float,
    k_threshold: float = 2.5,
    q_threshold: float = 0.8,
) -> AnomalyType:
    """
    Classify anomaly type based on z-score and volume quantile.

    Args:
        zscore: Return z-score
        volume_quantile: Volume percentile (0-1)
        k_threshold: Z-score threshold for anomaly
        q_threshold: Volume quantile threshold

    Returns:
        AnomalyType enum value

    Logic:
        - SPIKE_UP: zscore > k AND volume_quantile > q
        - SPIKE_DOWN: zscore < -k AND volume_quantile > q
        - NONE: otherwise
    """
    # Check if we have high volume
    high_volume = volume_quantile >= q_threshold

    if high_volume:
        if zscore >= k_threshold:
            return AnomalyType.SPIKE_UP
        elif zscore <= -k_threshold:
            return AnomalyType.SPIKE_DOWN

    return AnomalyType.NONE


def is_anomaly(anomaly_type: AnomalyType) -> bool:
    """
    Check if anomaly type represents an actual anomaly.

    Args:
        anomaly_type: AnomalyType enum

    Returns:
        True if SPIKE_UP or SPIKE_DOWN, False if NONE
    """
    return anomaly_type in (AnomalyType.SPIKE_UP, AnomalyType.SPIKE_DOWN)
