"""
Analytics layer - Returns calculation, anomaly detection, labeling, regime detection, and data health.

P2 additions: News, calendar, LLM agents, resolver utilities, and analyst modules.
"""
from .returns import compute_returns
from .anomaly import (
    detect_anomalies_for_asset,
    detect_anomalies_all_assets,
    detect_anomalies_for_universe,
)
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
from .analyst_news import (
    NewsContext,
    MacroContext,
    generate_news_analyst_signal,
)
from .analyst_risk import (
    RiskMode,
    PositionSizingRec,
    generate_risk_analyst_signal,
)

__all__ = [
    "compute_returns",
    "detect_anomalies_for_asset",
    "detect_anomalies_all_assets",
    "detect_anomalies_for_universe",
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
    # P2: Analyst modules
    "NewsContext",
    "MacroContext",
    "generate_news_analyst_signal",
    "RiskMode",
    "PositionSizingRec",
    "generate_risk_analyst_signal",
]
