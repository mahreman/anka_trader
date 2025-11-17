"""
News client for Analist-2 (P2).

This module provides news data for fundamental analysis.
Initial version uses simple CSV/JSON log format.
Can be upgraded to real news APIs later.

P2 Features:
- Load news from CSV/JSON files
- Filter news by date and symbol
- Sentiment scoring (simple keyword-based for now)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """
    Represents a single news item.

    Attributes:
        date: Publication date
        symbol: Related asset symbol (or "GLOBAL" for macro news)
        headline: News headline
        summary: Brief summary (optional)
        sentiment: Sentiment score (-1 to +1, where -1=bearish, +1=bullish)
        source: News source identifier
        url: Optional URL to full article
    """
    date: date
    symbol: str
    headline: str
    summary: str = ""
    sentiment: float = 0.0
    source: str = "unknown"
    url: Optional[str] = None


def _simple_sentiment(text: str) -> float:
    """
    Ultra-simple keyword-based sentiment analysis.

    P2: Simple baseline, can be replaced with LLM later.

    Args:
        text: Text to analyze (headline + summary)

    Returns:
        Sentiment score in [-1, 1]
    """
    text_lower = text.lower()

    # Bullish keywords
    bullish = [
        "bullish", "rally", "surge", "breakout", "all-time high", "ath",
        "strong", "growth", "gains", "positive", "optimistic", "buy",
        "upgrade", "beat expectations", "exceeded", "recovery"
    ]

    # Bearish keywords
    bearish = [
        "bearish", "crash", "plunge", "collapse", "selloff", "sell-off",
        "weak", "decline", "losses", "negative", "pessimistic", "sell",
        "downgrade", "missed expectations", "warning", "crisis"
    ]

    bullish_count = sum(1 for word in bullish if word in text_lower)
    bearish_count = sum(1 for word in bearish if word in text_lower)

    total = bullish_count + bearish_count
    if total == 0:
        return 0.0

    # Normalize to [-1, 1]
    score = (bullish_count - bearish_count) / total
    return max(-1.0, min(1.0, score))


def load_news_from_csv(
    csv_path: Path | str,
    symbol: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[NewsItem]:
    """
    Load news items from CSV file.

    CSV Format:
        date,symbol,headline,summary,sentiment,source,url
        2025-01-15,BTC-USD,"Bitcoin rallies to new highs","...",0.8,reuters,https://...

    Args:
        csv_path: Path to CSV file
        symbol: Filter by symbol (optional)
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)

    Returns:
        List of NewsItem objects

    Example:
        >>> news = load_news_from_csv("data/news.csv", symbol="BTC-USD")
        >>> print(f"Loaded {len(news)} news items")
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.warning(f"News CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Apply filters
        if symbol:
            df = df[(df["symbol"] == symbol) | (df["symbol"] == "GLOBAL")]

        if start_date:
            df = df[df["date"] >= start_date]

        if end_date:
            df = df[df["date"] <= end_date]

        # Convert to NewsItem objects
        items = []
        for _, row in df.iterrows():
            # Auto-compute sentiment if not provided
            sentiment = row.get("sentiment", None)
            if pd.isna(sentiment):
                text = f"{row.get('headline', '')} {row.get('summary', '')}"
                sentiment = _simple_sentiment(text)

            items.append(
                NewsItem(
                    date=row["date"],
                    symbol=row["symbol"],
                    headline=row["headline"],
                    summary=row.get("summary", ""),
                    sentiment=float(sentiment),
                    source=row.get("source", "unknown"),
                    url=row.get("url", None),
                )
            )

        logger.info(f"Loaded {len(items)} news items from {csv_path}")
        return items

    except Exception as e:
        logger.error(f"Failed to load news CSV: {e}")
        return []


def load_news_from_json(
    json_path: Path | str,
    symbol: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[NewsItem]:
    """
    Load news items from JSON file.

    JSON Format:
        [
            {
                "date": "2025-01-15",
                "symbol": "BTC-USD",
                "headline": "Bitcoin rallies...",
                "summary": "...",
                "sentiment": 0.8,
                "source": "reuters",
                "url": "https://..."
            },
            ...
        ]

    Args:
        json_path: Path to JSON file
        symbol: Filter by symbol (optional)
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)

    Returns:
        List of NewsItem objects
    """
    json_path = Path(json_path)

    if not json_path.exists():
        logger.warning(f"News JSON not found: {json_path}")
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        items = []
        for item in data:
            item_date = pd.to_datetime(item["date"]).date()
            item_symbol = item["symbol"]

            # Apply filters
            if symbol and item_symbol != symbol and item_symbol != "GLOBAL":
                continue

            if start_date and item_date < start_date:
                continue

            if end_date and item_date > end_date:
                continue

            # Auto-compute sentiment if not provided
            sentiment = item.get("sentiment", None)
            if sentiment is None:
                text = f"{item.get('headline', '')} {item.get('summary', '')}"
                sentiment = _simple_sentiment(text)

            items.append(
                NewsItem(
                    date=item_date,
                    symbol=item_symbol,
                    headline=item["headline"],
                    summary=item.get("summary", ""),
                    sentiment=float(sentiment),
                    source=item.get("source", "unknown"),
                    url=item.get("url", None),
                )
            )

        logger.info(f"Loaded {len(items)} news items from {json_path}")
        return items

    except Exception as e:
        logger.error(f"Failed to load news JSON: {e}")
        return []


def get_recent_news(
    symbol: str,
    days_back: int = 7,
    data_dir: Path | str = "data/news",
) -> List[NewsItem]:
    """
    Get recent news for a symbol from all available sources.

    Args:
        symbol: Asset symbol
        days_back: Number of days to look back
        data_dir: Directory containing news files

    Returns:
        List of NewsItem objects sorted by date (newest first)

    Example:
        >>> news = get_recent_news("BTC-USD", days_back=7)
        >>> if news:
        ...     print(f"Latest: {news[0].headline}")
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    data_dir = Path(data_dir)
    all_news = []

    # Try to load from CSV
    csv_path = data_dir / "news.csv"
    if csv_path.exists():
        all_news.extend(
            load_news_from_csv(csv_path, symbol=symbol, start_date=start_date, end_date=end_date)
        )

    # Try to load from JSON
    json_path = data_dir / "news.json"
    if json_path.exists():
        all_news.extend(
            load_news_from_json(json_path, symbol=symbol, start_date=start_date, end_date=end_date)
        )

    # Sort by date (newest first)
    all_news.sort(key=lambda n: n.date, reverse=True)

    return all_news


def aggregate_sentiment(
    news_items: List[NewsItem],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Aggregate sentiment from multiple news items.

    Args:
        news_items: List of NewsItem objects
        weights: Optional weights for each item (e.g., by recency)

    Returns:
        Aggregated sentiment score in [-1, 1]

    Example:
        >>> news = get_recent_news("BTC-USD", days_back=3)
        >>> sentiment = aggregate_sentiment(news)
        >>> print(f"3-day sentiment: {sentiment:.2f}")
    """
    if not news_items:
        return 0.0

    if weights is None:
        # Default: equal weights
        weights = [1.0] * len(news_items)

    if len(weights) != len(news_items):
        raise ValueError("Number of weights must match number of news items")

    weighted_sum = sum(n.sentiment * w for n, w in zip(news_items, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight
