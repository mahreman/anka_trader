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
from .analyst_regime_db import build_regime_analyst_signal, get_regime_summary
from .analyst_news_db import (
    build_news_analyst_signal_from_db,
    get_recent_news_summary,
    calculate_news_sentiment_score,
)
from .analyst_macro_db import (
    build_macro_risk_signal,
    get_macro_summary,
    format_macro_summary,
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
    "build_regime_analyst_signal",
    "get_regime_summary",
    "build_news_analyst_signal_from_db",
    "get_recent_news_summary",
    "calculate_news_sentiment_score",
    "build_macro_risk_signal",
    "get_macro_summary",
    "format_macro_summary",
]
