"""
Analist-2: News-based analyst using LLM agent.

This analyst fetches recent news articles from the database,
analyzes them using the LLM agent, and produces trading signals.

P2 Feature: Alternative data integration (news sentiment)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from ..data.schema import NewsArticle
from ..research.llm_agent import analyze_events
from .ensemble import AnalystSignal

logger = logging.getLogger(__name__)


def build_news_analyst_signal_from_db(
    session: Session,
    symbol_ticker: str,
    as_of: datetime,
    window_hours: int = 24,
    use_llm: bool = False,
) -> AnalystSignal:
    """
    Build trading signal from news articles in database.

    Fetches recent news articles related to the symbol and uses
    LLM agent to analyze sentiment and generate signal.

    Args:
        session: Database session
        symbol_ticker: Trading symbol (e.g., "BTCUSDT", "bitcoin")
        as_of: Current timestamp
        window_hours: Hours to look back for news (default: 24)
        use_llm: Whether to use real LLM API (default: False)

    Returns:
        AnalystSignal for news-based trading decision

    Example:
        >>> from otonom_trader.data import get_session
        >>> with next(get_session()) as session:
        ...     signal = build_news_analyst_signal_from_db(
        ...         session, "bitcoin", datetime.now()
        ...     )
        ...     print(signal.direction)
        'BUY'
    """
    cutoff = as_of - timedelta(hours=window_hours)

    # Query news articles
    # Note: NewsArticle.symbols is comma-separated string, use LIKE
    # Match both exact ticker and partial matches (e.g., "bitcoin" matches "bitcoin,ethereum")
    articles: List[NewsArticle] = (
        session.query(NewsArticle)
        .filter(
            NewsArticle.symbols.like(f"%{symbol_ticker}%"),
            NewsArticle.published_at >= cutoff,
            NewsArticle.published_at <= as_of,
        )
        .order_by(NewsArticle.published_at.desc())
        .limit(50)  # Limit to 50 most recent articles
        .all()
    )

    logger.info(
        f"Found {len(articles)} news articles for {symbol_ticker} "
        f"in last {window_hours} hours"
    )

    # Analyze articles using LLM agent
    opinion = analyze_events(
        symbol=symbol_ticker,
        news_articles=articles,
        macro_events=[],  # Future: integrate macro events
        use_llm=use_llm,
    )

    # Convert to AnalystSignal
    signal = AnalystSignal(
        name="Analist-2 (News/LLM)",
        direction=opinion.direction_hint,
        p_up=opinion.p_up,
        confidence=opinion.confidence,
        weight=1.0,
    )

    logger.info(
        f"News analyst signal for {symbol_ticker}: "
        f"{signal.direction} (p_up={signal.p_up:.2f}, conf={signal.confidence:.2f})"
    )

    return signal


def get_recent_news_summary(
    session: Session,
    symbol_ticker: str,
    as_of: datetime,
    window_hours: int = 24,
    max_articles: int = 5,
) -> str:
    """
    Get human-readable summary of recent news for a symbol.

    Args:
        session: Database session
        symbol_ticker: Trading symbol
        as_of: Current timestamp
        window_hours: Hours to look back
        max_articles: Maximum articles to include in summary

    Returns:
        Formatted string with news summary

    Example:
        >>> summary = get_recent_news_summary(session, "bitcoin", datetime.now())
        >>> print(summary)
        Recent news for bitcoin (last 24 hours):
        - Bitcoin surges past $50K (CoinDesk, 2h ago)
        - Federal Reserve signals rate cuts (Reuters, 5h ago)
        ...
    """
    cutoff = as_of - timedelta(hours=window_hours)

    articles = (
        session.query(NewsArticle)
        .filter(
            NewsArticle.symbols.like(f"%{symbol_ticker}%"),
            NewsArticle.published_at >= cutoff,
            NewsArticle.published_at <= as_of,
        )
        .order_by(NewsArticle.published_at.desc())
        .limit(max_articles)
        .all()
    )

    if not articles:
        return f"No recent news for {symbol_ticker} in last {window_hours} hours."

    lines = [f"Recent news for {symbol_ticker} (last {window_hours} hours):"]

    for article in articles:
        hours_ago = (as_of - article.published_at).total_seconds() / 3600
        hours_str = f"{int(hours_ago)}h ago" if hours_ago >= 1 else f"{int(hours_ago * 60)}m ago"

        source = article.source_name or "Unknown"
        title = article.title[:80] + "..." if len(article.title) > 80 else article.title

        lines.append(f"- {title} ({source}, {hours_str})")

    return "\n".join(lines)


def calculate_news_sentiment_score(
    session: Session,
    symbol_ticker: str,
    as_of: datetime,
    window_hours: int = 24,
) -> Optional[float]:
    """
    Calculate aggregated news sentiment score for a symbol.

    Args:
        session: Database session
        symbol_ticker: Trading symbol
        as_of: Current timestamp
        window_hours: Hours to look back

    Returns:
        Sentiment score (-1 to 1), or None if no articles

    Note:
        Uses NewsArticle.sentiment_score if available,
        otherwise falls back to LLM analysis
    """
    cutoff = as_of - timedelta(hours=window_hours)

    articles = (
        session.query(NewsArticle)
        .filter(
            NewsArticle.symbols.like(f"%{symbol_ticker}%"),
            NewsArticle.published_at >= cutoff,
            NewsArticle.published_at <= as_of,
        )
        .all()
    )

    if not articles:
        return None

    # Check if sentiment_score is available
    scored_articles = [a for a in articles if a.sentiment_score is not None]

    if scored_articles:
        # Use pre-computed sentiment scores
        scores = [a.sentiment_score for a in scored_articles]
        avg_score = sum(scores) / len(scores)
        return avg_score
    else:
        # Fall back to LLM analysis
        opinion = analyze_events(
            symbol=symbol_ticker,
            news_articles=articles,
            macro_events=[],
        )
        # Convert p_up to sentiment score (-1 to 1)
        # p_up=0.7 -> sentiment=0.4
        # p_up=0.3 -> sentiment=-0.4
        sentiment = (opinion.p_up - 0.5) * 2.0
        return sentiment
