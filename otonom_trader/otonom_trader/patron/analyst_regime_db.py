"""
Analist-1 (Enhanced): Regime and data quality-aware technical analyst.

This analyst combines:
- Market regime detection (volatility, trend, structural breaks)
- Data quality assessment (DSI - Data Health Index)
- Anomaly detection (spikes/crashes)

P1 Feature: Regime-aware trading signals
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from ..data.schema import Regime, DataHealthIndex, Symbol
from .ensemble import AnalystSignal

logger = logging.getLogger(__name__)


def build_regime_analyst_signal(
    session: Session,
    symbol_ticker: str,
    on_date: date,
) -> AnalystSignal:
    """
    Build trading signal based on market regime and data quality.

    Regime interpretation:
    - Low volatility (regime_id=0): More aggressive signals
    - Normal volatility (regime_id=1): Standard signals
    - High volatility (regime_id=2): Conservative/defensive

    Trend integration:
    - Uptrend + low vol = BUY bias
    - Downtrend + high vol = SELL bias

    Data quality (DSI):
    - High DSI (>0.8): Full confidence
    - Medium DSI (0.5-0.8): Reduced confidence
    - Low DSI (<0.5): Very low confidence

    Args:
        session: Database session
        symbol_ticker: Trading symbol
        on_date: Analysis date

    Returns:
        AnalystSignal for regime-based trading decision

    Example:
        >>> signal = build_regime_analyst_signal(session, "BTCUSDT", date.today())
        >>> print(signal.direction)
        'BUY'  # Low vol uptrend
    """
    # Get symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol_ticker).first()
    if symbol_obj is None:
        logger.warning(f"Symbol {symbol_ticker} not found in database")
        return AnalystSignal(
            name="Analist-1 (Regime)",
            direction="HOLD",
            p_up=0.5,
            confidence=0.1,
            weight=1.0,
        )

    # Get regime for this date
    regime = (
        session.query(Regime)
        .filter(
            Regime.symbol_id == symbol_obj.id,
            Regime.date == on_date,
        )
        .first()
    )

    if regime is None:
        logger.warning(f"No regime data for {symbol_ticker} on {on_date}")
        return AnalystSignal(
            name="Analist-1 (Regime)",
            direction="HOLD",
            p_up=0.5,
            confidence=0.2,
            weight=1.0,
        )

    # Get data quality (DSI)
    dsi_record = (
        session.query(DataHealthIndex)
        .filter(
            DataHealthIndex.symbol_id == symbol_obj.id,
            DataHealthIndex.date == on_date,
        )
        .first()
    )

    dsi_score = dsi_record.dsi if dsi_record else 0.7  # Default to moderate quality

    # Analyze regime
    regime_id = regime.regime_id
    volatility = regime.volatility
    trend = regime.trend
    is_structural_break = bool(regime.is_structural_break)

    # Initialize scoring
    signal_score = 0.0  # -1 (bearish) to +1 (bullish)

    # Factor 1: Regime volatility
    if regime_id == 0:  # Low volatility
        # Low vol = more stable, slightly bullish bias
        signal_score += 0.1
    elif regime_id == 2:  # High volatility
        # High vol = risky, slightly bearish bias
        signal_score -= 0.1

    # Factor 2: Trend
    if trend > 0.02:  # Strong uptrend
        signal_score += 0.3
    elif trend > 0.01:  # Moderate uptrend
        signal_score += 0.15
    elif trend < -0.02:  # Strong downtrend
        signal_score -= 0.3
    elif trend < -0.01:  # Moderate downtrend
        signal_score -= 0.15

    # Factor 3: Structural break
    if is_structural_break:
        # Structural break = uncertainty, reduce signal strength
        signal_score *= 0.7

    # Factor 4: Combine regime and trend
    # Low vol + uptrend = strong BUY
    # High vol + downtrend = strong SELL
    if regime_id == 0 and trend > 0.02:
        signal_score += 0.2  # Bonus for low-vol uptrend
    elif regime_id == 2 and trend < -0.02:
        signal_score -= 0.2  # Penalty for high-vol downtrend

    # Clamp signal score
    signal_score = max(-1.0, min(1.0, signal_score))

    # Convert to p_up
    p_up = 0.5 + signal_score * 0.3  # ±0.3 from neutral
    p_up = max(0.2, min(0.8, p_up))

    # Determine direction
    if p_up > 0.6:
        direction = "BUY"
    elif p_up < 0.4:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Confidence based on data quality and regime clarity
    base_confidence = 0.6

    # Adjust by DSI
    dsi_factor = dsi_score  # 0-1
    confidence = base_confidence * dsi_factor

    # Reduce confidence for high volatility
    if regime_id == 2:
        confidence *= 0.8

    # Reduce confidence for structural breaks
    if is_structural_break:
        confidence *= 0.7

    # Clamp confidence
    confidence = max(0.2, min(0.9, confidence))

    logger.info(
        f"Regime signal for {symbol_ticker}: {direction} "
        f"(p_up={p_up:.2f}, conf={confidence:.2f}, "
        f"regime={regime_id}, trend={trend:.3f}, dsi={dsi_score:.2f})"
    )

    return AnalystSignal(
        name="Analist-1 (Regime)",
        direction=direction,
        p_up=p_up,
        confidence=confidence,
        weight=1.0,
    )


def get_regime_summary(
    session: Session,
    symbol_ticker: str,
    on_date: date,
) -> Optional[str]:
    """
    Get human-readable regime summary for a symbol.

    Args:
        session: Database session
        symbol_ticker: Trading symbol
        on_date: Analysis date

    Returns:
        Formatted string with regime details, or None if no data

    Example:
        >>> summary = get_regime_summary(session, "BTCUSDT", date.today())
        >>> print(summary)
        Regime: Low Volatility (0)
        Trend: +1.2% (Uptrend)
        Volatility: 0.15
        Structural Break: No
        Data Quality: 0.87 (Good)
    """
    # Get symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol_ticker).first()
    if symbol_obj is None:
        return None

    # Get regime
    regime = (
        session.query(Regime)
        .filter(
            Regime.symbol_id == symbol_obj.id,
            Regime.date == on_date,
        )
        .first()
    )

    if regime is None:
        return None

    # Get DSI
    dsi_record = (
        session.query(DataHealthIndex)
        .filter(
            DataHealthIndex.symbol_id == symbol_obj.id,
            DataHealthIndex.date == on_date,
        )
        .first()
    )

    dsi_score = dsi_record.dsi if dsi_record else None

    # Format regime
    regime_labels = {
        0: "Low Volatility",
        1: "Normal Volatility",
        2: "High Volatility",
    }
    regime_label = regime_labels.get(regime.regime_id, "Unknown")

    # Format trend
    trend_pct = regime.trend * 100
    if trend_pct > 1.0:
        trend_label = "Uptrend"
    elif trend_pct < -1.0:
        trend_label = "Downtrend"
    else:
        trend_label = "Neutral"

    # Format structural break
    break_label = "Yes" if regime.is_structural_break else "No"

    # Format DSI
    if dsi_score is not None:
        if dsi_score > 0.8:
            dsi_label = f"{dsi_score:.2f} (Good)"
        elif dsi_score > 0.5:
            dsi_label = f"{dsi_score:.2f} (Fair)"
        else:
            dsi_label = f"{dsi_score:.2f} (Poor)"
    else:
        dsi_label = "N/A"

    summary = f"""Regime: {regime_label} ({regime.regime_id})
Trend: {trend_pct:+.1f}% ({trend_label})
Volatility: {regime.volatility:.3f}
Structural Break: {break_label}
Data Quality: {dsi_label}"""

    return summary
