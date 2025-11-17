"""
Reporting and formatting utilities for decisions and anomalies.
"""
from typing import List

from ..domain import Decision as DecisionDomain, Anomaly as AnomalyDomain, Asset


def format_decision(
    decision: DecisionDomain, anomaly: AnomalyDomain = None, asset: Asset = None
) -> str:
    """
    Format a single decision as a human-readable string.

    Args:
        decision: Decision object
        anomaly: Optional anomaly that triggered the decision
        asset: Optional asset metadata

    Returns:
        Formatted string (one line summary)

    Example:
        "2023-05-10 | SUGAR | SPIKE_DOWN | BUY | conf=0.62 | reason='20d uptrend + high volume'"
    """
    parts = [
        f"{decision.date}",
        f"{decision.asset_symbol:>10}",
        f"{decision.signal.value:>4}",
        f"conf={decision.confidence:.2f}",
    ]

    if anomaly:
        parts.insert(2, f"{anomaly.anomaly_type.value:>10}")

    # Truncate reason for single line display
    reason = decision.reason
    if len(reason) > 80:
        reason = reason[:77] + "..."

    parts.append(f"{reason}")

    return " | ".join(parts)


def format_decision_summary(decisions: List[DecisionDomain]) -> str:
    """
    Format a summary of multiple decisions.

    Args:
        decisions: List of decisions

    Returns:
        Multi-line formatted string
    """
    if not decisions:
        return "No decisions to display"

    # Group by signal type
    buy_count = sum(1 for d in decisions if d.signal.value == "BUY")
    sell_count = sum(1 for d in decisions if d.signal.value == "SELL")
    hold_count = sum(1 for d in decisions if d.signal.value == "HOLD")

    # Calculate average confidence by signal type
    buy_conf = (
        sum(d.confidence for d in decisions if d.signal.value == "BUY") / buy_count
        if buy_count > 0
        else 0
    )
    sell_conf = (
        sum(d.confidence for d in decisions if d.signal.value == "SELL") / sell_count
        if sell_count > 0
        else 0
    )
    hold_conf = (
        sum(d.confidence for d in decisions if d.signal.value == "HOLD") / hold_count
        if hold_count > 0
        else 0
    )

    summary = f"""
Decision Summary
================
Total decisions: {len(decisions)}

By Signal Type:
  BUY:  {buy_count:3d} (avg conf: {buy_conf:.2f})
  SELL: {sell_count:3d} (avg conf: {sell_conf:.2f})
  HOLD: {hold_count:3d} (avg conf: {hold_conf:.2f})
"""
    return summary.strip()


def format_anomaly(anomaly: AnomalyDomain) -> str:
    """
    Format a single anomaly as a human-readable string.

    Args:
        anomaly: Anomaly object

    Returns:
        Formatted string

    Example:
        "2023-05-10 | BTC-USD | SPIKE_DOWN | ret=-0.08 | z=-3.2 | vol_q=0.92"
    """
    return (
        f"{anomaly.date} | {anomaly.asset_symbol:>10} | "
        f"{anomaly.anomaly_type.value:>10} | "
        f"ret={anomaly.abs_return:+.4f} | "
        f"z={anomaly.zscore:+.2f} | "
        f"vol_q={anomaly.volume_rank:.2f}"
    )


def format_anomaly_list(anomalies: List[AnomalyDomain], limit: int = None) -> str:
    """
    Format a list of anomalies as a table.

    Args:
        anomalies: List of anomalies
        limit: Maximum number to display

    Returns:
        Formatted table string
    """
    if not anomalies:
        return "No anomalies to display"

    display_anomalies = anomalies[:limit] if limit else anomalies

    lines = [
        "Date       | Symbol     | Type       | Return   | Z-Score | Vol Quantile",
        "-" * 75,
    ]

    for a in display_anomalies:
        lines.append(format_anomaly(a))

    if limit and len(anomalies) > limit:
        lines.append(f"\n... and {len(anomalies) - limit} more")

    return "\n".join(lines)
