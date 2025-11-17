"""
Analist-2: News/Calendar + LLM analyst module (P2).

This module provides news and macro context for anomalies,
then generates trading signals via LLM analysis.

Pipeline:
1. Get news/calendar events around anomaly time
2. Build context (anomaly + news + macro + regime/DSI)
3. Call LLM to generate signal
4. Return AnalystSignal with reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from ..domain import Anomaly
from .news_client import NewsItem, get_recent_news, aggregate_sentiment
from .calendar_client import MacroEvent, get_recent_events, compute_macro_bias
from .llm_agent import get_llm_signal_with_fallback, LLMSignal
from .resolvers import resolve_regime, resolve_dsi

logger = logging.getLogger(__name__)


@dataclass
class NewsContext:
    """
    News context for an anomaly.

    Attributes:
        news_items: Relevant news items
        sentiment: Aggregated sentiment (-1 to +1)
        top_headlines: Top 3 headlines
        summary: Human-readable summary
    """
    news_items: List[NewsItem]
    sentiment: float
    top_headlines: List[str]
    summary: str


@dataclass
class MacroContext:
    """
    Macro context for an anomaly.

    Attributes:
        events: Relevant macro events
        bias: Macro bias (-1 to +1)
        key_events: Top 3 event names
        summary: Human-readable summary
    """
    events: List[MacroEvent]
    bias: float
    key_events: List[str]
    summary: str


def get_news_context_for_anomaly(
    symbol: str,
    anomaly_date: date,
    lookback_days: int = 3,
    data_dir: str = "data/news",
) -> NewsContext:
    """
    Get news context around an anomaly.

    Args:
        symbol: Asset symbol
        anomaly_date: Date of anomaly
        lookback_days: Days to look back from anomaly (default: 3)
        data_dir: Directory containing news files

    Returns:
        NewsContext with news items and sentiment

    Example:
        >>> ctx = get_news_context_for_anomaly("BTC-USD", date(2025, 1, 15))
        >>> print(f"Sentiment: {ctx.sentiment:.2f}")
        >>> print(f"Headlines: {ctx.top_headlines}")
    """
    # Get news from [anomaly_date - lookback_days, anomaly_date]
    end_date = anomaly_date
    start_date = end_date - timedelta(days=lookback_days)

    # Load news (Note: get_recent_news uses "days_back" from today, so we need custom range)
    # For now, we'll load recent news and filter by date
    all_news = get_recent_news(symbol, days_back=lookback_days + 1, data_dir=data_dir)

    # Filter to our date range
    news_items = [
        n for n in all_news
        if start_date <= n.date <= end_date
    ]

    # Aggregate sentiment
    sentiment = aggregate_sentiment(news_items) if news_items else 0.0

    # Top headlines
    top_headlines = [n.headline for n in news_items[:3]]

    # Summary
    if not news_items:
        summary = f"No news in last {lookback_days} days"
    else:
        summary = f"{len(news_items)} news items, sentiment: {sentiment:+.2f}"
        if top_headlines:
            summary += f" | Top: {top_headlines[0][:50]}..."

    logger.debug(f"News context for {symbol} on {anomaly_date}: {summary}")

    return NewsContext(
        news_items=news_items,
        sentiment=sentiment,
        top_headlines=top_headlines,
        summary=summary,
    )


def get_macro_context_for_anomaly(
    anomaly_date: date,
    lookback_days: int = 7,
    region: str = "US",
    data_dir: str = "data/calendar",
) -> MacroContext:
    """
    Get macro context around an anomaly.

    Args:
        anomaly_date: Date of anomaly
        lookback_days: Days to look back from anomaly (default: 7)
        region: Region filter (default: US)
        data_dir: Directory containing calendar files

    Returns:
        MacroContext with events and bias

    Example:
        >>> ctx = get_macro_context_for_anomaly(date(2025, 1, 15))
        >>> print(f"Macro bias: {ctx.bias:.2f}")
        >>> print(f"Key events: {ctx.key_events}")
    """
    # Get events from [anomaly_date - lookback_days, anomaly_date]
    end_date = anomaly_date
    start_date = end_date - timedelta(days=lookback_days)

    # Load events (Note: get_recent_events uses "days_back" from today)
    all_events = get_recent_events(days_back=lookback_days + 1, region=region, data_dir=data_dir)

    # Filter to our date range
    events = [
        e for e in all_events
        if start_date <= e.date <= end_date
    ]

    # Compute bias
    bias = compute_macro_bias(events) if events else 0.0

    # Key events
    key_events = [e.event_name for e in events[:3]]

    # Summary
    if not events:
        summary = f"No macro events in last {lookback_days} days"
    else:
        summary = f"{len(events)} events, bias: {bias:+.2f}"
        if key_events:
            summary += f" | Key: {key_events[0][:40]}..."

    logger.debug(f"Macro context for {anomaly_date}: {summary}")

    return MacroContext(
        events=events,
        bias=bias,
        key_events=key_events,
        summary=summary,
    )


def generate_news_analyst_signal(
    session: Session,
    anomaly: Anomaly,
    news_lookback: int = 3,
    macro_lookback: int = 7,
) -> Optional[LLMSignal]:
    """
    Generate Analist-2 signal using news/macro + LLM.

    Pipeline:
    1. Get news context (last N days)
    2. Get macro context (last M days)
    3. Get regime/DSI context
    4. Build prompt and call LLM
    5. Return LLMSignal

    Args:
        session: Database session
        anomaly: Anomaly to analyze
        news_lookback: Days to look back for news (default: 3)
        macro_lookback: Days to look back for macro (default: 7)

    Returns:
        LLMSignal with direction, p_up, confidence, reasoning

    Example:
        >>> signal = generate_news_analyst_signal(session, anomaly)
        >>> if signal:
        ...     print(f"{signal.direction}: {signal.reasoning}")
    """
    logger.info(f"Generating Analist-2 signal for {anomaly.asset_symbol} on {anomaly.date}")

    # Get news context
    try:
        news_ctx = get_news_context_for_anomaly(
            anomaly.asset_symbol,
            anomaly.date,
            lookback_days=news_lookback,
        )
    except Exception as e:
        logger.warning(f"Failed to get news context: {e}")
        news_ctx = NewsContext(
            news_items=[],
            sentiment=0.0,
            top_headlines=[],
            summary="No news available",
        )

    # Get macro context
    try:
        macro_ctx = get_macro_context_for_anomaly(
            anomaly.date,
            lookback_days=macro_lookback,
        )
    except Exception as e:
        logger.warning(f"Failed to get macro context: {e}")
        macro_ctx = MacroContext(
            events=[],
            bias=0.0,
            key_events=[],
            summary="No macro data available",
        )

    # Get regime/DSI context
    regime_context = ""
    try:
        regime_data = resolve_regime(session, anomaly.asset_symbol, anomaly.date)
        dsi_data = resolve_dsi(session, anomaly.asset_symbol, anomaly.date)

        if regime_data:
            regime_id, vol, trend, is_break = regime_data
            regime_context = f"Regime: {regime_id}, Vol: {vol:.4f}, Trend: {trend:.4f}"
            if is_break:
                regime_context += " (structural break!)"

        if dsi_data:
            dsi_score, _, _, _ = dsi_data
            regime_context += f" | DSI: {dsi_score:.2f}"

    except Exception as e:
        logger.warning(f"Failed to get regime/DSI context: {e}")

    # Build anomaly context
    anomaly_context = (
        f"{anomaly.anomaly_type.value} detected on {anomaly.date}, "
        f"zscore={anomaly.zscore:.2f}, volume_rank={anomaly.volume_rank:.2f}"
    )

    # Build news summary
    news_summary = news_ctx.summary
    if news_ctx.top_headlines:
        news_summary += f"\nTop headlines: {'; '.join(news_ctx.top_headlines[:2])}"

    # Build macro summary
    macro_summary = macro_ctx.summary
    if macro_ctx.key_events:
        macro_summary += f"\nKey events: {'; '.join(macro_ctx.key_events[:2])}"

    # Call LLM
    try:
        llm_signal = get_llm_signal_with_fallback(
            symbol=anomaly.asset_symbol,
            anomaly_context=anomaly_context,
            news_summary=news_summary,
            macro_summary=macro_summary,
            regime_context=regime_context,
        )

        logger.info(
            f"Analist-2 signal: {llm_signal.direction} "
            f"(p_up={llm_signal.p_up:.2f}, conf={llm_signal.confidence:.2f})"
        )

        return llm_signal

    except Exception as e:
        logger.error(f"Failed to generate LLM signal: {e}")
        return None
