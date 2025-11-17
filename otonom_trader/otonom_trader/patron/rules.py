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
    Generate trading decision using ensemble of analysts (P2 feature).

    This is an enhanced version of make_decision_for_anomaly that:
    1. Gets signal from Analist-1 (technical analyst - existing rules)
    2. Gets signal from Analist-2 (news/macro + LLM)
    3. Gets signal from Analist-3 (regime/DSI context)
    4. Combines them using weighted ensemble
    5. Applies disagreement penalty

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
    import json
    from .ensemble import AnalystSignal, combine_signals, apply_disagreement_penalty
    from ..analytics import (
        generate_news_analyst_signal,
        generate_risk_analyst_signal,
    )

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

    # Build ensemble of analysts
    signals = []
    analyst_details = []

    # ============================================================
    # Analist-1: Technical (from existing rules)
    # ============================================================
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
            name="Analist-1 (Technical)",
            direction=tech_direction,
            p_up=tech_p_up,
            confidence=base_decision.confidence,
            weight=1.0,
        )
    )
    analyst_details.append({
        "name": "Analist-1 (Technical)",
        "direction": tech_direction,
        "p_up": tech_p_up,
        "confidence": base_decision.confidence,
    })

    # ============================================================
    # Analist-2: News/Macro + LLM (using analyst_news module)
    # ============================================================
    try:
        llm_signal = generate_news_analyst_signal(session, anomaly)

        if llm_signal:
            signals.append(
                AnalystSignal(
                    name="Analist-2 (News/Macro/LLM)",
                    direction=llm_signal.direction,
                    p_up=llm_signal.p_up,
                    confidence=llm_signal.confidence,
                    weight=1.2,  # Slightly higher weight for LLM
                )
            )
            analyst_details.append({
                "name": "Analist-2 (News/Macro/LLM)",
                "direction": llm_signal.direction,
                "p_up": llm_signal.p_up,
                "confidence": llm_signal.confidence,
                "reasoning": llm_signal.reasoning,
            })

    except Exception as e:
        logger.warning(f"Analist-2 (News/Macro/LLM) failed: {e}")
        # Continue without LLM signal

    # ============================================================
    # Analist-3: Risk Assessment (using analyst_risk module)
    # ============================================================
    try:
        risk_signal = generate_risk_analyst_signal(session, anomaly)

        if risk_signal:
            signals.append(
                AnalystSignal(
                    name="Analist-3 (Risk)",
                    direction=risk_signal["direction"],
                    p_up=risk_signal["p_up"],
                    confidence=risk_signal["confidence"],  # Position size multiplier
                    weight=0.8,
                )
            )
            analyst_details.append({
                "name": "Analist-3 (Risk)",
                "direction": risk_signal["direction"],
                "p_up": risk_signal["p_up"],
                "confidence": risk_signal["confidence"],
                "reasoning": risk_signal["reasoning"],
            })

    except Exception as e:
        logger.warning(f"Analist-3 (Risk) failed: {e}")
        # Continue without risk signal

    # ============================================================
    # Combine signals
    # ============================================================
    ensemble = combine_signals(signals)

    # Apply disagreement penalty
    adjusted_confidence = apply_disagreement_penalty(
        base_decision.confidence,
        ensemble.disagreement,
        threshold=0.5,
    )

    # Convert ensemble direction to SignalType
    if ensemble.direction == "BUY":
        final_signal = SignalType.BUY
    elif ensemble.direction == "SELL":
        final_signal = SignalType.SELL
    else:
        final_signal = SignalType.HOLD

    # ============================================================
    # Calculate uncertainty
    # ============================================================
    # Uncertainty is based on:
    # 1. Analyst disagreement (high disagreement = high uncertainty)
    # 2. Number of analysts (fewer analysts = higher uncertainty)
    # 3. Data availability (missing analysts = higher uncertainty)

    num_analysts = len(signals)
    max_analysts = 3  # We expect 3 analysts

    # Base uncertainty from disagreement
    uncertainty = ensemble.disagreement

    # Penalty for missing analysts
    if num_analysts < max_analysts:
        missing_penalty = (max_analysts - num_analysts) * 0.15
        uncertainty = min(1.0, uncertainty + missing_penalty)

    # ============================================================
    # Build detailed multi-analyst explanation
    # ============================================================
    analyst_explanations = []

    for detail in analyst_details:
        name = detail["name"].split("(")[1].rstrip(")")  # Extract short name
        direction = detail["direction"]

        # Add reasoning if available
        if "reasoning" in detail:
            reasoning = detail["reasoning"]
            # Truncate long reasoning
            if len(reasoning) > 80:
                reasoning = reasoning[:77] + "..."
            analyst_explanations.append(f"{name}: {direction} ({reasoning})")
        else:
            analyst_explanations.append(f"{name}: {direction}")

    # Join all analyst explanations
    multi_analyst_reason = " | ".join(analyst_explanations)

    # Create enhanced decision with detailed explanation
    enhanced_reason = (
        f"MULTI-ANALYST DECISION:\n"
        f"{multi_analyst_reason}\n"
        f"→ ENSEMBLE: {ensemble.direction} (p_up={ensemble.p_up:.2f}, "
        f"disagreement={ensemble.disagreement:.2f}, uncertainty={uncertainty:.2f})"
    )

    decision = DecisionDomain(
        asset_symbol=anomaly.asset_symbol,
        date=anomaly.date,
        signal=final_signal,  # Use ensemble direction
        confidence=adjusted_confidence,  # Adjusted by disagreement
        reason=enhanced_reason,
        p_up=ensemble.p_up,
        disagreement=ensemble.disagreement,
        uncertainty=uncertainty,
        analyst_signals=json.dumps(analyst_details),
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
                p_up=decision.p_up,
                disagreement=decision.disagreement,
                uncertainty=decision.uncertainty,
                analyst_signals=decision.analyst_signals,
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
