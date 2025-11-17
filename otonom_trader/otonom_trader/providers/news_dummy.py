"""
Dummy news provider for testing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base import NewsProvider, NewsArticle

logger = logging.getLogger(__name__)


class DummyNewsProvider(NewsProvider):
    """Dummy news provider that returns synthetic news."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("DummyNewsProvider initialized (returns synthetic news)")

    def fetch_news(
        self,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """Generate synthetic news articles."""
        logger.warning("DummyNewsProvider returning synthetic news")

        articles = []
        for i in range(min(limit, 10)):
            article = NewsArticle(
                source="Dummy News",
                title=f"Dummy Article {i+1} about {symbol or query or 'markets'}",
                description=f"This is a dummy article for testing purposes.",
                url=f"https://example.com/article-{i+1}",
                published_at=datetime.now() - timedelta(hours=i),
                symbols=[symbol] if symbol else [],
                sentiment=0.0,
            )
            articles.append(article)

        return articles

    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        """Generate synthetic headlines."""
        return self.fetch_news(symbol=symbol, limit=limit)
