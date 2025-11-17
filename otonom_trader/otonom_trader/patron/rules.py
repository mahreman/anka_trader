"""
Rule-based trading decision engine (Patron).

P0 Rules:
- SPIKE_DOWN + uptrend = BUY (mean reversion bet)
- SPIKE_UP + downtrend = SELL (trend continuation)
- No anomaly = HOLD (low confidence)
"""
import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..domain import Anomaly as AnomalyDomain, Decision as DecisionDomain, SignalType, AnomalyType
from ..data.schema import Anomaly as AnomalyORM, Decision as DecisionORM, DailyBar, Symbol
from ..config import TREND_WINDOW, TREND_THRESHOLD

logger = logging.getLogger(__name__)


def calculate_trend(
    session: Session, symbol_id: int, ref_date: date, window: int = TREND_WINDOW
) -> Optional[str]:
    """
    Calculate price trend (UP/DOWN/FLAT) for the period before ref_date.

    Args:
        session: Database session
        symbol_id: Symbol ID
        ref_date: Reference date
        window: Number of days to look back

    Returns:
        'UP', 'DOWN', 'FLAT', or None if insufficient data
    """
    start_date = ref_date - timedelta(days=window + 10)  # Extra buffer for gaps

    bars = (
        session.query(DailyBar)
        .filter(
            DailyBar.symbol_id == symbol_id,
            DailyBar.date < ref_date,
            DailyBar.date >= start_date,
        )
        .order_by(DailyBar.date.desc())
        .limit(window)
        .all()
    )

    if len(bars) < window // 2:  # Need at least half the window
        logger.warning(f"Insufficient data for trend calculation: {len(bars)} bars")
        return None

    # Calculate simple moving average
    prices = [bar.close for bar in bars]
    avg_price = sum(prices) / len(prices)
    current_price = bars[0].close  # Most recent price

    # Calculate percent difference
    pct_diff = (current_price - avg_price) / avg_price

    if pct_diff > TREND_THRESHOLD:
        return "UP"
    elif pct_diff < -TREND_THRESHOLD:
        return "DOWN"
    else:
        return "FLAT"


def make_decision_for_anomaly(
    session: Session, anomaly: AnomalyDomain, persist: bool = True
) -> DecisionDomain:
    """
    Generate trading decision based on anomaly and trend.

    Rules:
        1. SPIKE_DOWN + UP trend → BUY (confidence: 0.6)
           Reasoning: Mean reversion after crash in uptrend

        2. SPIKE_UP + DOWN trend → SELL (confidence: 0.6)
           Reasoning: Trend continuation after spike in downtrend

        3. Other combinations → HOLD (confidence: 0.3)
           Reasoning: Unclear signal

    Args:
        session: Database session
        anomaly: Anomaly domain object
        persist: Whether to save decision to database

    Returns:
        Decision domain object
    """
    # Get symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=anomaly.asset_symbol).first()
    if symbol_obj is None:
        raise ValueError(f"Symbol {anomaly.asset_symbol} not found")

    # Calculate trend
    trend = calculate_trend(session, symbol_obj.id, anomaly.date)

    # Apply rules
    signal = SignalType.HOLD
    confidence = 0.3
    reason = "No clear signal"

    if anomaly.anomaly_type == AnomalyType.SPIKE_DOWN:
        if trend == "UP":
            signal = SignalType.BUY
            confidence = 0.6
            reason = (
                f"{TREND_WINDOW}d uptrend + spike down (zscore={anomaly.zscore:.2f}), "
                f"high volume (q={anomaly.volume_rank:.2f}). Mean reversion play."
            )
        elif trend == "DOWN":
            signal = SignalType.HOLD
            confidence = 0.4
            reason = (
                f"Spike down in downtrend - avoid catching falling knife. "
                f"Wait for reversal signal."
            )
        else:  # FLAT or None
            signal = SignalType.HOLD
            confidence = 0.35
            reason = f"Spike down but trend unclear. Monitor for opportunity."

    elif anomaly.anomaly_type == AnomalyType.SPIKE_UP:
        if trend == "DOWN":
            signal = SignalType.SELL
            confidence = 0.6
            reason = (
                f"{TREND_WINDOW}d downtrend + spike up (zscore={anomaly.zscore:.2f}), "
                f"high volume (q={anomaly.volume_rank:.2f}). Likely dead cat bounce."
            )
        elif trend == "UP":
            signal = SignalType.HOLD
            confidence = 0.4
            reason = (
                f"Spike up in uptrend - could be continuation or exhaustion. "
                f"Wait for confirmation."
            )
        else:  # FLAT or None
            signal = SignalType.HOLD
            confidence = 0.35
            reason = f"Spike up but trend unclear. Avoid premature short."

    # Create decision
    decision = DecisionDomain(
        asset_symbol=anomaly.asset_symbol,
        date=anomaly.date,
        signal=signal,
        confidence=confidence,
        reason=reason,
    )

    # Persist to database
    if persist:
        decision_orm = DecisionORM(
            symbol_id=symbol_obj.id,
            date=decision.date,
            signal=str(decision.signal),
            confidence=decision.confidence,
            reason=decision.reason,
        )
        session.add(decision_orm)
        session.commit()

    return decision


def make_ensemble_decision_for_anomaly(
    session: Session,
    anomaly: AnomalyDomain,
    use_ensemble: bool = True,
    persist: bool = True,
) -> DecisionDomain:
    """
    Generate trading decision using ensemble of analysts (P1 feature).

    This is an enhanced version of make_decision_for_anomaly that:
    1. Gets signal from technical analyst (existing rules)
    2. Gets signal from other analysts (news, macro - stub for P1)
    3. Combines them using weighted ensemble
    4. Applies disagreement penalty

    Args:
        session: Database session
        anomaly: Anomaly domain object
        use_ensemble: If True, uses ensemble; if False, falls back to basic rules
        persist: Whether to save decision to database

    Returns:
        Decision domain object with ensemble-enhanced confidence

    Example:
        >>> # With ensemble
        >>> decision = make_ensemble_decision_for_anomaly(session, anomaly, use_ensemble=True)
        >>> # Confidence adjusted based on analyst disagreement
    """
    from .ensemble import AnalystSignal, combine_signals, apply_disagreement_penalty

    # Get basic technical analysis (existing rules)
    base_decision = make_decision_for_anomaly(session, anomaly, persist=False)

    if not use_ensemble:
        # Fallback to basic rules
        if persist:
            # Persist the basic decision
            symbol_obj = session.query(Symbol).filter_by(symbol=anomaly.asset_symbol).first()
            if symbol_obj:
                decision_orm = DecisionORM(
                    symbol_id=symbol_obj.id,
                    date=base_decision.date,
                    signal=str(base_decision.signal),
                    confidence=base_decision.confidence,
                    reason=base_decision.reason,
                )
                session.add(decision_orm)
                session.commit()
        return base_decision

    # P1: Build ensemble of analysts
    signals = []

    # Technical analyst (from existing rules)
    tech_direction = base_decision.signal.value
    # Approximate p_up from signal and confidence
    if tech_direction == "BUY":
        tech_p_up = 0.5 + base_decision.confidence / 2.0
    elif tech_direction == "SELL":
        tech_p_up = 0.5 - base_decision.confidence / 2.0
    else:
        tech_p_up = 0.5

    signals.append(
        AnalystSignal(
            name="Technical",
            direction=tech_direction,
            p_up=tech_p_up,
            confidence=base_decision.confidence,
            weight=1.0,
        )
    )

    # TODO P2: Add news analyst
    # TODO P2: Add macro analyst
    # For now, we only have technical, so ensemble just wraps it

    # Combine signals
    ensemble = combine_signals(signals)

    # Apply disagreement penalty
    adjusted_confidence = apply_disagreement_penalty(
        base_decision.confidence,
        ensemble.disagreement,
        threshold=0.5,
    )

    # Create enhanced decision
    enhanced_reason = (
        base_decision.reason
        + f"\n[Ensemble: p_up={ensemble.p_up:.2f}, disagreement={ensemble.disagreement:.2f}]"
    )

    decision = DecisionDomain(
        asset_symbol=anomaly.asset_symbol,
        date=anomaly.date,
        signal=base_decision.signal,  # Keep same signal
        confidence=adjusted_confidence,  # Adjusted by disagreement
        reason=enhanced_reason,
    )

    # Persist
    if persist:
        symbol_obj = session.query(Symbol).filter_by(symbol=anomaly.asset_symbol).first()
        if symbol_obj:
            decision_orm = DecisionORM(
                symbol_id=symbol_obj.id,
                date=decision.date,
                signal=str(decision.signal),
                confidence=decision.confidence,
                reason=decision.reason,
            )
            session.add(decision_orm)
            session.commit()

    return decision


def run_daily_decision_pass(
    session: Session, days_back: int = 30, use_ensemble: bool = False
) -> dict[str, List[DecisionDomain]]:
    """
    Run Patron decision engine on recent anomalies.

    Args:
        session: Database session
        days_back: Number of days to look back for anomalies
        use_ensemble: If True, uses ensemble mode (P1 feature)

    Returns:
        Dictionary mapping symbol to list of decisions
    """
    logger.info(f"Running Patron decision pass for last {days_back} days")
    if use_ensemble:
        logger.info("Using ensemble mode (P1)")

    cutoff_date = date.today() - timedelta(days=days_back)

    # Get recent anomalies
    anomalies = (
        session.query(AnomalyORM, Symbol.symbol)
        .join(Symbol)
        .filter(AnomalyORM.date >= cutoff_date)
        .order_by(AnomalyORM.date.desc())
        .all()
    )

    logger.info(f"Found {len(anomalies)} anomalies to process")

    results = {}

    for anomaly_orm, symbol in anomalies:
        # Convert to domain object
        anomaly = AnomalyDomain(
            asset_symbol=symbol,
            date=anomaly_orm.date,
            anomaly_type=AnomalyType(anomaly_orm.anomaly_type),
            abs_return=anomaly_orm.abs_return,
            zscore=anomaly_orm.zscore,
            volume_rank=anomaly_orm.volume_rank,
            comment=anomaly_orm.comment,
        )

        # Generate decision
        try:
            if use_ensemble:
                decision = make_ensemble_decision_for_anomaly(
                    session, anomaly, use_ensemble=True, persist=True
                )
            else:
                decision = make_decision_for_anomaly(session, anomaly, persist=True)

            if symbol not in results:
                results[symbol] = []
            results[symbol].append(decision)

        except Exception as e:
            logger.error(f"Failed to make decision for {symbol} on {anomaly.date}: {e}")

    return results
