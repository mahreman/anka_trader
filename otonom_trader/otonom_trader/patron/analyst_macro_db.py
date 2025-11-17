"""
Analist-3: Macro-based analyst using economic indicators.

This analyst fetches macroeconomic indicators from the database
(e.g., interest rates, inflation, GDP) and produces risk-adjusted signals.

P2 Feature: Alternative data integration (macro analysis)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..data.schema import MacroIndicator
from .ensemble import AnalystSignal

logger = logging.getLogger(__name__)


def _get_macro_state(
    session: Session,
    series_id: str,
    on_date: date,
    lookback_days: int = 30,
) -> Optional[float]:
    """
    Get the most recent value of a macro indicator.

    Args:
        session: Database session
        series_id: FRED series ID (e.g., "DGS10", "CPIAUCSL")
        on_date: Reference date
        lookback_days: Days to look back for latest value

    Returns:
        Latest indicator value, or None if not found
    """
    cutoff = on_date - timedelta(days=lookback_days)

    rows = (
        session.query(MacroIndicator)
        .filter(
            MacroIndicator.series_id == series_id,
            MacroIndicator.date >= cutoff,
            MacroIndicator.date <= on_date,
        )
        .order_by(MacroIndicator.date.desc())
        .limit(1)
        .all()
    )

    if not rows:
        logger.warning(f"No data found for {series_id} around {on_date}")
        return None

    return float(rows[0].value)


def _get_macro_trend(
    session: Session,
    series_id: str,
    on_date: date,
    window_days: int = 90,
) -> Optional[str]:
    """
    Calculate trend direction for a macro indicator.

    Args:
        session: Database session
        series_id: FRED series ID
        on_date: Reference date
        window_days: Days to look back for trend calculation

    Returns:
        "UP", "DOWN", "FLAT", or None if insufficient data
    """
    cutoff = on_date - timedelta(days=window_days)

    rows = (
        session.query(MacroIndicator)
        .filter(
            MacroIndicator.series_id == series_id,
            MacroIndicator.date >= cutoff,
            MacroIndicator.date <= on_date,
        )
        .order_by(MacroIndicator.date.asc())
        .all()
    )

    if len(rows) < 2:
        return None

    # Compare first and last values
    first_value = float(rows[0].value)
    last_value = float(rows[-1].value)

    pct_change = (last_value - first_value) / first_value if first_value != 0 else 0

    if pct_change > 0.05:  # 5% increase
        return "UP"
    elif pct_change < -0.05:  # 5% decrease
        return "DOWN"
    else:
        return "FLAT"


def build_macro_risk_signal(
    session: Session,
    symbol_ticker: str,
    on_date: date,
) -> AnalystSignal:
    """
    Build trading signal based on macroeconomic risk assessment.

    Analyzes multiple macro indicators to determine risk environment:
    - 10Y Treasury yield (DGS10): Rising rates = risk-off
    - 2Y Treasury yield (DGS2): Short-term rate pressure
    - Fed Funds Rate (FEDFUNDS): Monetary policy tightness
    - CPI (CPIAUCSL): Inflation pressure

    Args:
        session: Database session
        symbol_ticker: Trading symbol (for context, may adjust by asset class)
        on_date: Analysis date

    Returns:
        AnalystSignal for macro-based trading decision

    Example:
        >>> signal = build_macro_risk_signal(session, "BTCUSDT", date.today())
        >>> print(signal.direction)
        'SELL'  # Risk-off environment
    """
    # Fetch key macro indicators
    y10 = _get_macro_state(session, "DGS10", on_date)  # 10Y Treasury
    y2 = _get_macro_state(session, "DGS2", on_date)   # 2Y Treasury
    fed_funds = _get_macro_state(session, "FEDFUNDS", on_date)  # Fed Funds Rate

    # Get trends
    y10_trend = _get_macro_trend(session, "DGS10", on_date)
    fed_trend = _get_macro_trend(session, "FEDFUNDS", on_date)

    # Calculate risk score (-1 to 1)
    # Negative = risk-off, Positive = risk-on
    risk_score = 0.0
    num_factors = 0

    # Factor 1: 10Y yield level
    # High yields (>4.5%) = risk-off for risk assets
    if y10 is not None:
        if y10 > 4.5:
            risk_score -= 0.3
        elif y10 < 3.0:
            risk_score += 0.2
        num_factors += 1

    # Factor 2: 10Y yield trend
    # Rising yields = risk-off
    if y10_trend is not None:
        if y10_trend == "UP":
            risk_score -= 0.2
        elif y10_trend == "DOWN":
            risk_score += 0.2
        num_factors += 1

    # Factor 3: Yield curve (2Y - 10Y)
    # Inverted curve = recession risk
    if y10 is not None and y2 is not None:
        curve = y10 - y2
        if curve < -0.5:  # Deeply inverted
            risk_score -= 0.3
        elif curve > 1.0:  # Steep curve
            risk_score += 0.2
        num_factors += 1

    # Factor 4: Fed Funds trend
    # Rising rates = tightening = risk-off
    if fed_trend is not None:
        if fed_trend == "UP":
            risk_score -= 0.2
        elif fed_trend == "DOWN":
            risk_score += 0.3  # Easing = risk-on
        num_factors += 1

    # Normalize risk score
    if num_factors == 0:
        # No macro data available
        logger.warning("No macro indicators available for risk assessment")
        return AnalystSignal(
            name="Analist-3 (Macro)",
            direction="HOLD",
            p_up=0.5,
            confidence=0.2,
            weight=1.0,
        )

    normalized_risk = risk_score / num_factors  # -1 to 1

    # Convert to p_up (0-1)
    # Risk-on (positive score) = higher p_up
    # Risk-off (negative score) = lower p_up
    p_up = 0.5 + normalized_risk * 0.3  # ±0.3 from neutral
    p_up = max(0.2, min(0.8, p_up))  # Clamp to [0.2, 0.8]

    # Determine direction
    if p_up > 0.6:
        direction = "BUY"  # Risk-on environment
    elif p_up < 0.4:
        direction = "SELL"  # Risk-off environment
    else:
        direction = "HOLD"  # Neutral

    # Confidence based on data availability
    confidence = 0.5 + (num_factors / 8.0)  # More factors = higher confidence
    confidence = max(0.3, min(0.9, confidence))

    logger.info(
        f"Macro risk signal for {symbol_ticker}: {direction} "
        f"(p_up={p_up:.2f}, conf={confidence:.2f}, risk_score={normalized_risk:.2f})"
    )

    return AnalystSignal(
        name="Analist-3 (Macro)",
        direction=direction,
        p_up=p_up,
        confidence=confidence,
        weight=1.0,
    )


def get_macro_summary(
    session: Session,
    on_date: date,
) -> Dict[str, Optional[float]]:
    """
    Get summary of key macro indicators.

    Args:
        session: Database session
        on_date: Reference date

    Returns:
        Dictionary with latest values of key indicators

    Example:
        >>> summary = get_macro_summary(session, date.today())
        >>> print(summary)
        {
            'DGS10': 4.23,
            'DGS2': 4.58,
            'FEDFUNDS': 5.33,
            'CPIAUCSL': 3.2,
            ...
        }
    """
    key_series = [
        "DGS10",      # 10Y Treasury
        "DGS2",       # 2Y Treasury
        "FEDFUNDS",   # Fed Funds Rate
        "DFF",        # Fed Funds Effective Rate
        "CPIAUCSL",   # CPI
        "UNRATE",     # Unemployment Rate
        "GDP",        # GDP
    ]

    summary = {}
    for series_id in key_series:
        value = _get_macro_state(session, series_id, on_date)
        summary[series_id] = value

    return summary


def format_macro_summary(summary: Dict[str, Optional[float]]) -> str:
    """
    Format macro summary as human-readable string.

    Args:
        summary: Dictionary from get_macro_summary()

    Returns:
        Formatted string
    """
    lines = ["Current Macro Environment:"]

    labels = {
        "DGS10": "10Y Treasury",
        "DGS2": "2Y Treasury",
        "FEDFUNDS": "Fed Funds Rate",
        "DFF": "Fed Funds (Effective)",
        "CPIAUCSL": "CPI (YoY %)",
        "UNRATE": "Unemployment Rate",
        "GDP": "GDP (Billions $)",
    }

    for series_id, value in summary.items():
        label = labels.get(series_id, series_id)
        if value is not None:
            lines.append(f"  - {label}: {value:.2f}")
        else:
            lines.append(f"  - {label}: N/A")

    return "\n".join(lines)
