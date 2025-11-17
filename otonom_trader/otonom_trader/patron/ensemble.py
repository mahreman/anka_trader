"""
Weighted ensemble and disagreement mode for Patron v1.

This module does NOT replace existing rules.py.
Instead, rules.py can call into this to aggregate multiple
AnalystSignal objects (technical/news/macro) into a final signal.

P1 Feature: Multi-analyst consensus with disagreement tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Dict

import numpy as np

from ..domain import SignalType


Direction = Literal["BUY", "SELL", "HOLD"]


@dataclass
class AnalystSignal:
    """
    Signal from a single analyst (technical, news, macro, etc.).

    Attributes:
        name: Analyst name/identifier
        direction: Trading direction
        p_up: Model probability that price will go up (0-1)
        confidence: Subjective confidence of the analyst (0-1)
        weight: Base weight for this analyst (default: 1.0)
    """
    name: str
    direction: Direction
    p_up: float          # Probability price goes up
    confidence: float    # Analyst confidence (0-1)
    weight: float = 1.0  # Base weight


@dataclass
class EnsembleDecision:
    """
    Aggregated decision from multiple analysts.

    Attributes:
        direction: Consensus direction (BUY/SELL/HOLD)
        p_up: Weighted probability of price going up
        disagreement: Disagreement metric (0 = consensus, 1 = chaos)
        explanation: Human-readable explanation
    """
    direction: Direction
    p_up: float
    disagreement: float  # 0 = full agreement, 1 = max disagreement
    explanation: str


def _direction_to_sign(direction: Direction) -> int:
    """
    Convert direction to signed integer.

    Args:
        direction: Trading direction

    Returns:
        +1 for BUY, -1 for SELL, 0 for HOLD
    """
    if direction == "BUY":
        return +1
    if direction == "SELL":
        return -1
    return 0


def combine_signals(
    signals: List[AnalystSignal],
) -> EnsembleDecision:
    """
    Combine multiple AnalystSignal objects into a single decision.

    Algorithm:
    1. Convert each p_up to signed score in [-1, 1] based on direction
    2. Compute weighted average of signed scores
    3. Convert back to p_up probability
    4. Calculate disagreement as 1 - mean(|signed_score|)
    5. Determine final direction based on p_up thresholds

    Args:
        signals: List of AnalystSignal objects

    Returns:
        EnsembleDecision with consensus direction and disagreement metric

    Example:
        >>> tech = AnalystSignal("Technical", "BUY", 0.7, 0.8)
        >>> news = AnalystSignal("News", "HOLD", 0.5, 0.6)
        >>> result = combine_signals([tech, news])
        >>> print(result.direction)
        'BUY'
    """
    if not signals:
        return EnsembleDecision(
            direction="HOLD",
            p_up=0.5,
            disagreement=1.0,
            explanation="No analyst signal; default HOLD.",
        )

    # Normalize p_up into signed score in [-1, 1]
    scores = []
    weights = []

    for s in signals:
        sign = _direction_to_sign(s.direction)
        # Map p_up from [0,1] to signed [-1,1] with direction
        # p_up=0.7, direction=BUY -> signed_p = (0.7-0.5)*2*1 = 0.4
        # p_up=0.3, direction=SELL -> signed_p = (0.3-0.5)*2*(-1) = 0.4
        signed_p = (s.p_up - 0.5) * 2.0 * sign
        w = s.weight * s.confidence
        scores.append(signed_p)
        weights.append(w)

    scores_arr = np.array(scores)
    weights_arr = np.array(weights)

    # Handle zero weights
    if weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr)

    # Weighted average of signed scores
    mean_score = float(np.average(scores_arr, weights=weights_arr))

    # Back to probability of up
    p_up = 0.5 + mean_score / 2.0
    p_up = max(0.0, min(1.0, p_up))

    # Disagreement: if signs are mixed, mean of |signed_score| is low
    mean_abs = float(np.average(np.abs(scores_arr), weights=weights_arr))
    disagreement = 1.0 - mean_abs  # [0,1], 0 = consensus, 1 = chaos

    # Determine direction based on p_up thresholds
    if p_up > 0.55:
        direction: Direction = "BUY"
    elif p_up < 0.45:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Human-readable explanation
    parts = []
    for s in signals:
        parts.append(
            f"{s.name}: {s.direction} (p_up={s.p_up:.2f}, conf={s.confidence:.2f})"
        )

    explanation = (
        f"Ensemble: direction={direction}, p_up={p_up:.2f}, "
        f"disagreement={disagreement:.2f}\n"
        + " | ".join(parts)
    )

    return EnsembleDecision(
        direction=direction,
        p_up=p_up,
        disagreement=disagreement,
        explanation=explanation,
    )


def apply_disagreement_penalty(
    base_confidence: float,
    disagreement: float,
    threshold: float = 0.5,
) -> float:
    """
    Apply penalty to confidence based on disagreement.

    If analysts disagree (disagreement > threshold), reduce confidence.

    Args:
        base_confidence: Original confidence (0-1)
        disagreement: Disagreement metric (0-1)
        threshold: Disagreement threshold (default: 0.5)

    Returns:
        Adjusted confidence

    Example:
        >>> apply_disagreement_penalty(0.8, 0.7, 0.5)
        0.4  # High disagreement cuts confidence
    """
    if disagreement > threshold:
        # Penalty proportional to excess disagreement
        penalty_factor = 1.0 - (disagreement - threshold) / (1.0 - threshold)
        return base_confidence * penalty_factor
    return base_confidence


def get_analyst_weights(
    performance_history: Dict[str, float] = None,
    default_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Get analyst weights based on historical performance.

    P1: Simple static weights
    P2: Dynamic weights based on backtest performance

    Args:
        performance_history: Dict mapping analyst name to win rate
        default_weight: Default weight if no history

    Returns:
        Dict mapping analyst name to weight

    Example:
        >>> perf = {"Technical": 0.65, "News": 0.52}
        >>> weights = get_analyst_weights(perf)
        >>> weights["Technical"] > weights["News"]
        True
    """
    if performance_history is None:
        return {}

    weights = {}
    for name, win_rate in performance_history.items():
        # Simple: weight proportional to win rate
        # Win rate 0.6 -> weight 1.2
        # Win rate 0.5 -> weight 1.0
        # Win rate 0.4 -> weight 0.8
        weights[name] = max(0.1, (win_rate - 0.5) * 4.0 + 1.0)

    return weights
