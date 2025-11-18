"""News provider backed by yfinance symbol headlines."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency is part of runtime requirements
    yf = None  # type: ignore

from .base import NewsProvider, NewsArticle, ProviderError
from ..utils import ensure_aware, utc_now

logger = logging.getLogger(__name__)


class YFinanceNewsProvider(NewsProvider):
    """Fetch symbol-specific news via yfinance."""

    def __init__(self, config: dict):
        super().__init__(config)
        extra = config.get("extra") or {}
        self.max_lookback_days = int(extra.get("max_lookback_days", 30))

    def _ensure_client(self) -> None:
        if yf is None:
            raise ProviderError(
                "yfinance is required for YFinanceNewsProvider but is not installed"
            )

    def _should_include(
        self,
        published_at: datetime,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> bool:
        published_at = ensure_aware(published_at)  # type: ignore[arg-type]
        start_dt = ensure_aware(start) if start else None
        end_dt = ensure_aware(end) if end else None

        if start_dt and published_at < start_dt:
            return False
        if end_dt and published_at > end_dt:
            return False
        cutoff = utc_now() - timedelta(days=self.max_lookback_days)
        if published_at < cutoff:
            return False
        return True

    def _transform_article(self, item: dict, symbol: str) -> Optional[NewsArticle]:
        link = item.get("link") or item.get("url")
        if not link:
            return None

        publish_ts = item.get("providerPublishTime") or item.get("published_at")
        if publish_ts is None:
            return None

        if isinstance(publish_ts, (int, float)):
            published_at = datetime.fromtimestamp(publish_ts, tz=timezone.utc)
        elif isinstance(publish_ts, datetime):
            published_at = ensure_aware(publish_ts)
        else:
            return None

        source = item.get("provider") or item.get("publisher") or "yfinance"
        description = item.get("summary") or item.get("content") or ""
        symbols = item.get("relatedTickers") or ([symbol] if symbol else [])

        return NewsArticle(
            source=source,
            title=item.get("title", ""),
            description=description,
            url=link,
            published_at=published_at,
            symbols=symbols,
            sentiment=None,
            author=item.get("author"),
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

        if not symbol:
            raise ProviderError("YFinanceNewsProvider requires a symbol for fetch_news")

        ticker = yf.Ticker(symbol)
        try:
            items = ticker.news or []
        except Exception as exc:  # pragma: no cover - network errors
            raise ProviderError(f"Failed to fetch yfinance news for {symbol}: {exc}") from exc

        normalized_start = ensure_aware(start_date) if start_date else None
        normalized_end = ensure_aware(end_date) if end_date else None

        sorted_items = sorted(
            items,
            key=lambda item: item.get("providerPublishTime") or 0,
            reverse=True,
        )

        articles: List[NewsArticle] = []
        for item in sorted_items:
            article = self._transform_article(item, symbol)
            if article is None:
                continue
            if not self._should_include(article.published_at, normalized_start, normalized_end):
                continue
            articles.append(article)
            if len(articles) >= limit:
                break

        return articles

    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        return self.fetch_news(symbol=symbol, limit=limit)
