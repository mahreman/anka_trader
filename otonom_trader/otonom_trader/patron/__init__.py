"""
Patron - Rule-based decision engine for trading signals.
"""
from .rules import make_decision_for_anomaly, run_daily_decision_pass
from .reporter import format_decision, format_decision_summary
from .ensemble import (
    AnalystSignal,
    EnsembleDecision,
    combine_signals,
    apply_disagreement_penalty,
    get_analyst_weights,
)

__all__ = [
    "make_decision_for_anomaly",
    "run_daily_decision_pass",
    "format_decision",
    "format_decision_summary",
    "AnalystSignal",
    "EnsembleDecision",
    "combine_signals",
    "apply_disagreement_penalty",
    "get_analyst_weights",
]
