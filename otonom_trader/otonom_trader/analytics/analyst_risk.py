"""
Analist-3: Regime/DSI risk analyst module (P2).

This module provides risk assessment based on regime detection and DSI scores,
then generates position sizing recommendations and trading signals.

Pipeline:
1. Get regime context (volatility, trend, structural breaks)
2. Get DSI context (fear/greed levels)
3. Assess risk-on/risk-off mode
4. Calculate position size multiplier
5. Return AnalystSignal with risk-adjusted recommendation

Example:
    >>> signal = generate_risk_analyst_signal(session, anomaly)
    >>> print(f"{signal.direction}: {signal.reasoning}")
    >>> print(f"Position size multiplier: {signal.confidence}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ..domain import Anomaly, AnomalyType
from .resolvers import resolve_regime, resolve_dsi

logger = logging.getLogger(__name__)


@dataclass
class RiskMode:
    """
    Risk mode assessment.

    Attributes:
        mode: "RISK_ON", "RISK_OFF", or "NEUTRAL"
        score: Risk score from -1 (extreme risk-off) to +1 (extreme risk-on)
        regime_contribution: Contribution from regime analysis
        dsi_contribution: Contribution from DSI analysis
        summary: Human-readable summary
    """
    mode: str
    score: float
    regime_contribution: float
    dsi_contribution: float
    summary: str


@dataclass
class PositionSizingRec:
    """
    Position sizing recommendation.

    Attributes:
        multiplier: Position size multiplier (0.0 to 2.0)
        reasoning: Explanation of sizing decision
        risk_factors: List of risk factors considered
    """
    multiplier: float
    reasoning: str
    risk_factors: list[str]


def assess_regime_risk(
    regime_data: Optional[Tuple[int, float, float, bool]]
) -> Tuple[float, str]:
    """
    Assess risk from regime data.

    Args:
        regime_data: Tuple of (regime_id, volatility, trend, is_structural_break)

    Returns:
        Tuple of (risk_score, explanation)
        risk_score: -1 (high risk) to +1 (low risk)

    Example:
        >>> risk, reason = assess_regime_risk((2, 0.025, 0.15, False))
        >>> print(f"Risk: {risk:.2f}, Reason: {reason}")
    """
    if regime_data is None:
        return 0.0, "No regime data"

    regime_id, volatility, trend, is_break = regime_data

    # Base risk from volatility (higher vol = higher risk)
    # Typical crypto vol ranges: 0.01 (low) to 0.05+ (high)
    if volatility < 0.015:
        vol_risk = 0.5  # Low vol = safer
    elif volatility < 0.03:
        vol_risk = 0.0  # Normal vol
    else:
        vol_risk = -0.5  # High vol = riskier

    # Trend contribution (strong trend = safer for trend-following)
    if abs(trend) > 0.1:
        trend_risk = 0.3  # Strong trend = safer
    elif abs(trend) > 0.05:
        trend_risk = 0.1  # Moderate trend
    else:
        trend_risk = -0.2  # Weak/choppy = riskier

    # Structural break penalty
    break_risk = -0.4 if is_break else 0.0

    # Combine
    total_risk = vol_risk + trend_risk + break_risk
    total_risk = max(-1.0, min(1.0, total_risk))  # Clamp to [-1, 1]

    # Explanation
    parts = []
    if is_break:
        parts.append("STRUCTURAL BREAK detected")
    parts.append(f"Vol={volatility:.4f}")
    parts.append(f"Trend={trend:+.4f}")

    explanation = f"Regime risk: {total_risk:+.2f} ({', '.join(parts)})"

    return total_risk, explanation


def assess_dsi_risk(dsi_data: Optional[Tuple[float, float, float, float]]) -> Tuple[float, str]:
    """
    Assess risk from DSI data.

    Args:
        dsi_data: Tuple of (dsi_score, rsi, stoch, williams)

    Returns:
        Tuple of (risk_score, explanation)
        risk_score: -1 (extreme fear, contrarian bullish) to +1 (extreme greed, contrarian bearish)

    Example:
        >>> risk, reason = assess_dsi_risk((0.25, 35, 20, -75))
        >>> print(f"Risk: {risk:.2f}, Reason: {reason}")
    """
    if dsi_data is None:
        return 0.0, "No DSI data"

    dsi_score, rsi, stoch, williams = dsi_data

    # DSI interpretation (contrarian):
    # DSI < 0.3: Extreme fear → bullish contrarian → risk-on
    # DSI > 0.7: Extreme greed → bearish contrarian → risk-off
    # DSI 0.4-0.6: Neutral

    if dsi_score < 0.2:
        dsi_risk = 0.8  # Extreme fear = buy opportunity
        mode = "EXTREME FEAR (contrarian buy)"
    elif dsi_score < 0.3:
        dsi_risk = 0.5  # Strong fear
        mode = "FEAR (bullish)"
    elif dsi_score < 0.4:
        dsi_risk = 0.2  # Mild fear
        mode = "Mild fear"
    elif dsi_score < 0.6:
        dsi_risk = 0.0  # Neutral
        mode = "NEUTRAL"
    elif dsi_score < 0.7:
        dsi_risk = -0.2  # Mild greed
        mode = "Mild greed"
    elif dsi_score < 0.8:
        dsi_risk = -0.5  # Strong greed
        mode = "GREED (bearish)"
    else:
        dsi_risk = -0.8  # Extreme greed = sell signal
        mode = "EXTREME GREED (contrarian sell)"

    explanation = f"DSI={dsi_score:.2f} → {mode}"

    return dsi_risk, explanation


def determine_risk_mode(
    session: Session,
    symbol: str,
    query_date: date,
) -> RiskMode:
    """
    Determine overall risk mode for a symbol on a given date.

    Args:
        session: Database session
        symbol: Asset symbol
        query_date: Date to assess

    Returns:
        RiskMode with overall assessment

    Example:
        >>> mode = determine_risk_mode(session, "BTC-USD", date(2025, 1, 15))
        >>> print(f"{mode.mode}: {mode.summary}")
    """
    # Get regime and DSI data
    regime_data = resolve_regime(session, symbol, query_date)
    dsi_data = resolve_dsi(session, symbol, query_date)

    # Assess individual risks
    regime_risk, regime_reason = assess_regime_risk(regime_data)
    dsi_risk, dsi_reason = assess_dsi_risk(dsi_data)

    # Combine (weight: regime 60%, DSI 40%)
    combined_score = 0.6 * regime_risk + 0.4 * dsi_risk

    # Determine mode
    if combined_score > 0.3:
        mode = "RISK_ON"
    elif combined_score < -0.3:
        mode = "RISK_OFF"
    else:
        mode = "NEUTRAL"

    # Summary
    summary = f"{mode} (score: {combined_score:+.2f}) | {regime_reason} | {dsi_reason}"

    return RiskMode(
        mode=mode,
        score=combined_score,
        regime_contribution=regime_risk,
        dsi_contribution=dsi_risk,
        summary=summary,
    )


def calculate_position_sizing(
    risk_mode: RiskMode,
    anomaly: Anomaly,
) -> PositionSizingRec:
    """
    Calculate position sizing recommendation based on risk mode.

    Args:
        risk_mode: Risk mode assessment
        anomaly: Anomaly being traded

    Returns:
        PositionSizingRec with multiplier and reasoning

    Example:
        >>> sizing = calculate_position_sizing(risk_mode, anomaly)
        >>> print(f"Multiplier: {sizing.multiplier:.2f}x")
        >>> print(f"Reasoning: {sizing.reasoning}")
    """
    risk_factors = []

    # Base multiplier from risk mode
    if risk_mode.mode == "RISK_ON":
        base_multiplier = 1.5  # Increase size in favorable conditions
        risk_factors.append(f"{risk_mode.mode} environment (+50%)")
    elif risk_mode.mode == "RISK_OFF":
        base_multiplier = 0.5  # Reduce size in unfavorable conditions
        risk_factors.append(f"{risk_mode.mode} environment (-50%)")
    else:
        base_multiplier = 1.0  # Normal size
        risk_factors.append("NEUTRAL environment (no adjustment)")

    # Adjust for anomaly strength
    if abs(anomaly.zscore) > 3.0:
        strength_adj = 1.2
        risk_factors.append(f"Strong anomaly (zscore={anomaly.zscore:.2f}, +20%)")
    elif abs(anomaly.zscore) < 2.0:
        strength_adj = 0.8
        risk_factors.append(f"Weak anomaly (zscore={anomaly.zscore:.2f}, -20%)")
    else:
        strength_adj = 1.0

    # Adjust for volume confirmation
    if anomaly.volume_rank > 0.9:
        volume_adj = 1.1
        risk_factors.append(f"High volume confirmation (rank={anomaly.volume_rank:.2f}, +10%)")
    elif anomaly.volume_rank < 0.5:
        volume_adj = 0.9
        risk_factors.append(f"Low volume (rank={anomaly.volume_rank:.2f}, -10%)")
    else:
        volume_adj = 1.0

    # Combine multipliers
    final_multiplier = base_multiplier * strength_adj * volume_adj

    # Clamp to reasonable range [0.25, 2.0]
    final_multiplier = max(0.25, min(2.0, final_multiplier))

    # Reasoning
    reasoning = (
        f"Position size: {final_multiplier:.2f}x base | "
        f"Risk mode: {risk_mode.mode} ({risk_mode.score:+.2f})"
    )

    return PositionSizingRec(
        multiplier=final_multiplier,
        reasoning=reasoning,
        risk_factors=risk_factors,
    )


def generate_risk_analyst_signal(
    session: Session,
    anomaly: Anomaly,
) -> Optional[dict]:
    """
    Generate Analist-3 signal using regime/DSI risk assessment.

    Pipeline:
    1. Determine risk mode (RISK_ON/RISK_OFF/NEUTRAL)
    2. Calculate position sizing recommendation
    3. Generate trading signal with risk adjustment
    4. Return signal dict with reasoning

    Args:
        session: Database session
        anomaly: Anomaly to analyze

    Returns:
        Dict with direction, p_up, confidence (position size multiplier), reasoning

    Example:
        >>> signal = generate_risk_analyst_signal(session, anomaly)
        >>> if signal:
        ...     print(f"{signal['direction']}: {signal['reasoning']}")
        ...     print(f"Position size: {signal['confidence']:.2f}x")
    """
    logger.info(f"Generating Analist-3 (Risk) signal for {anomaly.asset_symbol} on {anomaly.date}")

    # Step 1: Determine risk mode
    risk_mode = determine_risk_mode(session, anomaly.asset_symbol, anomaly.date)

    logger.debug(f"Risk mode: {risk_mode.summary}")

    # Step 2: Calculate position sizing
    sizing = calculate_position_sizing(risk_mode, anomaly)

    logger.debug(f"Position sizing: {sizing.reasoning}")

    # Step 3: Generate signal based on risk mode and anomaly type
    # Base direction from anomaly
    if anomaly.anomaly_type in (AnomalyType.SURGE, AnomalyType.BREAKOUT):
        base_direction = "BUY"
        base_p_up = 0.6
    elif anomaly.anomaly_type in (AnomalyType.DROP, AnomalyType.BREAKDOWN):
        base_direction = "SELL"
        base_p_up = 0.4
    else:
        base_direction = "HOLD"
        base_p_up = 0.5

    # Adjust based on risk mode
    if risk_mode.mode == "RISK_OFF" and base_direction == "BUY":
        # In risk-off mode, be more cautious about buys
        adjusted_direction = "HOLD"
        adjusted_p_up = 0.5
        adjustment_reason = "RISK_OFF mode → downgrade BUY to HOLD"
    elif risk_mode.mode == "RISK_ON" and base_direction == "SELL":
        # In risk-on mode, be more cautious about sells
        adjusted_direction = "HOLD"
        adjusted_p_up = 0.5
        adjustment_reason = "RISK_ON mode → downgrade SELL to HOLD"
    else:
        # No adjustment needed
        adjusted_direction = base_direction
        adjusted_p_up = base_p_up
        adjustment_reason = f"{risk_mode.mode} mode → no adjustment"

    # Build reasoning
    reasoning_parts = [
        f"Risk mode: {risk_mode.mode} (score: {risk_mode.score:+.2f})",
        f"Base signal: {base_direction} (from {anomaly.anomaly_type.value})",
        adjustment_reason,
        sizing.reasoning,
    ]

    reasoning = " | ".join(reasoning_parts)

    # Return signal dict
    signal = {
        "direction": adjusted_direction,
        "p_up": adjusted_p_up,
        "confidence": sizing.multiplier,  # Use position size multiplier as confidence
        "reasoning": reasoning,
    }

    logger.info(
        f"Analist-3 signal: {signal['direction']} "
        f"(p_up={signal['p_up']:.2f}, size={signal['confidence']:.2f}x)"
    )

    return signal
