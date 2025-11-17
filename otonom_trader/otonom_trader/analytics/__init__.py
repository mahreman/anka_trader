"""
Analytics layer - Returns calculation, anomaly detection, labeling, regime detection, and data health.

P2 additions: News, calendar, LLM agents, and resolver utilities.
"""
from .returns import compute_returns
from .anomaly import detect_anomalies_for_asset, detect_anomalies_all_assets
from .labeling import classify_anomaly
from .regime import (
    compute_regimes_for_symbol,
    compute_regimes_all_symbols,
    regimes_to_dataframe,
    persist_regimes,
    RegimePoint,
)
from .dsi import (
    compute_dsi_for_symbol,
    compute_dsi_all_symbols,
    dsi_to_dataframe,
    persist_dsi,
    DsiPoint,
)
from .news_client import (
    NewsItem,
    load_news_from_csv,
    load_news_from_json,
    get_recent_news,
    aggregate_sentiment,
)
from .calendar_client import (
    MacroEvent,
    EventImpact,
    load_calendar_from_csv,
    load_calendar_from_json,
    get_upcoming_events,
    get_recent_events,
    compute_macro_bias,
)
from .llm_agent import (
    LLMSignal,
    get_llm_signal,
    get_llm_signal_with_fallback,
)
from .resolvers import (
    resolve_regime,
    resolve_dsi,
)

__all__ = [
    "compute_returns",
    "detect_anomalies_for_asset",
    "detect_anomalies_all_assets",
    "classify_anomaly",
    "compute_regimes_for_symbol",
    "compute_regimes_all_symbols",
    "regimes_to_dataframe",
    "persist_regimes",
    "RegimePoint",
    "compute_dsi_for_symbol",
    "compute_dsi_all_symbols",
    "dsi_to_dataframe",
    "persist_dsi",
    "DsiPoint",
    # P2: News and calendar
    "NewsItem",
    "load_news_from_csv",
    "load_news_from_json",
    "get_recent_news",
    "aggregate_sentiment",
    "MacroEvent",
    "EventImpact",
    "load_calendar_from_csv",
    "load_calendar_from_json",
    "get_upcoming_events",
    "get_recent_events",
    "compute_macro_bias",
    # P2: LLM
    "LLMSignal",
    "get_llm_signal",
    "get_llm_signal_with_fallback",
    # P2: Resolvers
    "resolve_regime",
    "resolve_dsi",
]
