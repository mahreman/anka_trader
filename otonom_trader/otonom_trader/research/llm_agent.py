"""
LLM-based event analysis agent for news and macro events.

P2 Feature: Uses LLM to analyze news articles and macro events,
producing trading signals with probabilities and confidence scores.

Future: Integrate with OpenAI/Anthropic APIs for real LLM analysis.
For now, uses simple heuristic-based sentiment analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventOpinion:
    """
    Opinion from LLM event analysis.

    Attributes:
        direction_hint: Suggested trading direction
        p_up: Probability that price will go up (0-1)
        confidence: Confidence in this opinion (0-1)
        reasoning: Explanation of the opinion
    """
    direction_hint: str  # "BUY", "SELL", "HOLD"
    p_up: float
    confidence: float
    reasoning: str


def analyze_events(
    symbol: str,
    news_articles: List,
    macro_events: List,
    use_llm: bool = False,
) -> EventOpinion:
    """
    Analyze news articles and macro events to produce trading opinion.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "SPY")
        news_articles: List of NewsArticle ORM objects
        macro_events: List of macro event objects (future)
        use_llm: Whether to use real LLM API (default: False, uses heuristics)

    Returns:
        EventOpinion with direction, probability, and confidence

    Implementation:
        P2: Simple heuristic-based sentiment (keyword matching)
        P3: Real LLM integration (OpenAI/Anthropic)
    """
    if use_llm:
        # Future: Real LLM integration
        return _analyze_with_llm(symbol, news_articles, macro_events)
    else:
        # Current: Heuristic-based sentiment
        return _analyze_with_heuristics(symbol, news_articles, macro_events)


def _analyze_with_heuristics(
    symbol: str,
    news_articles: List,
    macro_events: List,
) -> EventOpinion:
    """
    Heuristic-based sentiment analysis using keyword matching.

    Positive keywords: rally, surge, breakout, bullish, buy, upgrade, growth
    Negative keywords: crash, plunge, bearish, sell, downgrade, decline, risk

    Args:
        symbol: Trading symbol
        news_articles: List of NewsArticle objects
        macro_events: List of macro events

    Returns:
        EventOpinion based on keyword sentiment
    """
    if not news_articles:
        logger.info(f"No news articles for {symbol}, returning neutral opinion")
        return EventOpinion(
            direction_hint="HOLD",
            p_up=0.5,
            confidence=0.3,
            reasoning="No recent news articles available",
        )

    # Positive and negative keywords
    positive_keywords = [
        "rally", "surge", "breakout", "bullish", "buy", "upgrade",
        "growth", "gain", "rise", "soar", "jump", "up", "positive",
        "record", "all-time high", "ath", "moon", "pump"
    ]

    negative_keywords = [
        "crash", "plunge", "bearish", "sell", "downgrade", "decline",
        "fall", "drop", "dump", "down", "negative", "risk", "fear",
        "panic", "collapse", "tank", "slump"
    ]

    # Count sentiment across all articles
    positive_score = 0
    negative_score = 0
    total_articles = len(news_articles)

    for article in news_articles:
        # Combine title and description for analysis
        text = ""
        if hasattr(article, 'title') and article.title:
            text += article.title.lower() + " "
        if hasattr(article, 'description') and article.description:
            text += article.description.lower()

        # Count positive keywords
        for keyword in positive_keywords:
            if keyword in text:
                positive_score += 1

        # Count negative keywords
        for keyword in negative_keywords:
            if keyword in text:
                negative_score += 1

    # Calculate net sentiment
    net_sentiment = positive_score - negative_score

    # Normalize to probability (0-1)
    # Use sigmoid-like scaling
    max_score = max(abs(positive_score), abs(negative_score), 1)
    normalized_sentiment = net_sentiment / (2 * max_score)  # -0.5 to 0.5

    # Convert to p_up (0-1)
    p_up = 0.5 + normalized_sentiment
    p_up = max(0.1, min(0.9, p_up))  # Clamp to [0.1, 0.9]

    # Determine direction
    if p_up > 0.6:
        direction = "BUY"
    elif p_up < 0.4:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Calculate confidence based on article count and sentiment strength
    # More articles + stronger sentiment = higher confidence
    article_factor = min(total_articles / 10.0, 1.0)  # Cap at 10 articles
    sentiment_strength = abs(normalized_sentiment) * 2  # 0-1
    confidence = (article_factor * 0.5 + sentiment_strength * 0.5)
    confidence = max(0.3, min(0.9, confidence))  # Clamp to [0.3, 0.9]

    reasoning = (
        f"Analyzed {total_articles} news articles. "
        f"Positive signals: {positive_score}, Negative signals: {negative_score}. "
        f"Net sentiment: {net_sentiment} → {direction}"
    )

    logger.info(f"News sentiment for {symbol}: {direction} (p_up={p_up:.2f}, conf={confidence:.2f})")

    return EventOpinion(
        direction_hint=direction,
        p_up=p_up,
        confidence=confidence,
        reasoning=reasoning,
    )


def _analyze_with_llm(
    symbol: str,
    news_articles: List,
    macro_events: List,
) -> EventOpinion:
    """
    LLM-based event analysis using OpenAI/Anthropic API.

    Future implementation: Send articles to LLM for sophisticated analysis.

    Args:
        symbol: Trading symbol
        news_articles: List of NewsArticle objects
        macro_events: List of macro events

    Returns:
        EventOpinion from LLM analysis
    """
    # TODO: Implement real LLM integration
    # Example:
    # - Format news articles into prompt
    # - Send to OpenAI/Anthropic API
    # - Parse response for direction, probability, confidence
    # - Return EventOpinion

    logger.warning("LLM integration not yet implemented, falling back to heuristics")
    return _analyze_with_heuristics(symbol, news_articles, macro_events)


def format_news_for_llm_prompt(news_articles: List) -> str:
    """
    Format news articles into LLM prompt.

    Args:
        news_articles: List of NewsArticle objects

    Returns:
        Formatted prompt string

    Example output:
        Article 1 (2024-01-15):
        Title: Bitcoin surges past $50K
        Source: CoinDesk
        Summary: Bitcoin rallied 10% today...

        Article 2 (2024-01-14):
        ...
    """
    if not news_articles:
        return "No recent news articles available."

    prompt_parts = []
    for i, article in enumerate(news_articles, 1):
        prompt_parts.append(f"Article {i} ({article.published_at}):")
        if hasattr(article, 'title') and article.title:
            prompt_parts.append(f"Title: {article.title}")
        if hasattr(article, 'source_name') and article.source_name:
            prompt_parts.append(f"Source: {article.source_name}")
        if hasattr(article, 'description') and article.description:
            prompt_parts.append(f"Summary: {article.description}")
        prompt_parts.append("")  # Empty line between articles

    return "\n".join(prompt_parts)
