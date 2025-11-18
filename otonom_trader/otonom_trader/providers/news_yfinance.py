"""News provider that fetches ticker headlines from yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency for news fetching
    yf = None  # type: ignore

from .base import NewsProvider, NewsArticle, ProviderError

logger = logging.getLogger(__name__)


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    """Parse the various timestamp formats returned by yfinance news."""

    if raw is None:
        return None

    # Numeric epoch seconds
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):  # pragma: no cover - defensive
            return None

    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if not isinstance(raw, str):
        return None

    value = raw.strip()
    if not value:
        return None

    try:
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


class YFinanceNewsProvider(NewsProvider):
    """Fetch symbol-specific headlines through yfinance."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        extra = config.get("extra") or {}
        self.default_days_back = int(extra.get("days_back", 7))
        self.default_limit = int(extra.get("default_limit", 50))

    def _ensure_client(self) -> None:
        if yf is None:
            raise ProviderError(
                "yfinance is required for YFinanceNewsProvider but is not installed"
            )

    def fetch_news(
        self,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        self._ensure_client()

        ticker_key = symbol or query
        if not ticker_key:
            raise ProviderError("YFinanceNewsProvider requires a symbol or query")

        # Normalize window
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        elif end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = end_date.astimezone(timezone.utc)

        if start_date is None:
            start_date = end_date - timedelta(days=self.default_days_back)
        elif start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        else:
            start_date = start_date.astimezone(timezone.utc)

        max_articles = limit or self.default_limit

        try:
            ticker = yf.Ticker(ticker_key)
            raw_items = getattr(ticker, "news", None) or []
        except Exception as exc:  # pragma: no cover - network errors
            raise ProviderError(f"Failed to fetch yfinance news for {ticker_key}: {exc}")

        logger.info(
            "YFinanceNewsProvider pulled %d raw news items for %s",
            len(raw_items),
            ticker_key,
        )

        articles: List[NewsArticle] = []
        for item in raw_items:
            content = item.get("content") or {}

            title = content.get("title") or item.get("title") or ""
            if not title:
                continue

            description = (
                content.get("summary")
                or content.get("description")
                or item.get("summary")
                or item.get("content")
                or title
            )

            provider_info = content.get("provider") or {}
            source = provider_info.get("displayName") or item.get("publisher") or "Yahoo Finance"

            canonical = content.get("canonicalUrl") or {}
            click = content.get("clickThroughUrl") or {}
            url = (
                (click.get("url") if isinstance(click, dict) else None)
                or (canonical.get("url") if isinstance(canonical, dict) else None)
                or item.get("link")
                or item.get("url")
            )
            if not url:
                continue

            published_raw = (
                content.get("pubDate")
                or content.get("displayTime")
                or item.get("providerPublishTime")
            )
            published_at = _parse_timestamp(published_raw)
            if not published_at:
                continue

            if published_at < start_date or published_at > end_date:
                continue

            article = NewsArticle(
                source=source,
                title=title,
                description=description,
                url=url,
                published_at=published_at.astimezone(timezone.utc).replace(tzinfo=None),
                symbols=[symbol] if symbol else ([ticker_key] if ticker_key else []),
                sentiment=None,
                author=None,
            )
            articles.append(article)

            if len(articles) >= max_articles:
                break

        logger.info(
            "YFinanceNewsProvider returning %d filtered articles for %s",
            len(articles),
            ticker_key,
        )
        return articles

    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        return self.fetch_news(symbol=symbol, limit=limit)
