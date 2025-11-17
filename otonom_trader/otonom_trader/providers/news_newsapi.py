"""
NewsAPI provider for news articles.

Uses NewsAPI.org to fetch financial news.
API Docs: https://newsapi.org/docs
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import requests

from .base import NewsProvider, NewsArticle, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


class NewsAPIProvider(NewsProvider):
    """
    NewsAPI.org news provider.

    Free tier: 100 requests/day
    Paid tiers: Higher limits

    API Key: Get from https://newsapi.org/
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            logger.warning("NewsAPI: No API key configured")

        self.base_url = "https://newsapi.org/v2"
        self.language = config.get("extra", {}).get("language", "en")
        self.country = config.get("extra", {}).get("country", "us")

        logger.info(f"NewsAPIProvider initialized (language={self.language})")

    def _request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make NewsAPI request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response

        Raises:
            ProviderError: If request fails
            RateLimitError: If rate limited
        """
        url = f"{self.base_url}/{endpoint}"

        # Add API key to headers
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.get("timeout_seconds", 30),
            )

            # Check rate limiting
            if response.status_code == 429:
                raise RateLimitError("NewsAPI rate limit exceeded")

            # Check authentication
            if response.status_code == 401:
                raise ProviderError("NewsAPI authentication failed (check API key)")

            # Check for errors
            data = response.json()
            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown error")
                raise ProviderError(f"NewsAPI error: {error_msg}")

            return data

        except requests.RequestException as e:
            raise ProviderError(f"NewsAPI request failed: {e}")

    def fetch_news(
        self,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """
        Fetch news articles from NewsAPI.

        Args:
            symbol: Not used (NewsAPI doesn't support symbol filtering)
            query: Search query (e.g., "Bitcoin", "Tesla")
            start_date: Start datetime (max 1 month back for free tier)
            end_date: End datetime
            limit: Max articles (max 100)

        Returns:
            List of news articles

        Example:
            >>> provider = NewsAPIProvider({"api_key": "..."})
            >>> articles = provider.fetch_news(query="Bitcoin", limit=20)
        """
        # Build query
        if symbol and not query:
            # Convert symbol to search query
            query = symbol.replace("-", " ").replace("USD", "").strip()

        if not query:
            query = "finance OR stock OR crypto OR market"

        # Date range (NewsAPI free tier: max 1 month)
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=7)  # Default: last 7 days

        # Prepare parameters
        params = {
            "q": query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": self.language,
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),  # NewsAPI max: 100
        }

        logger.info(f"Fetching NewsAPI articles: query='{query}' from {start_date.date()} to {end_date.date()}")

        try:
            data = self._request("everything", params)
        except ProviderError as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            raise

        # Parse articles
        articles = []
        for item in data.get("articles", []):
            # Parse published date
            published_str = item.get("publishedAt")
            if published_str:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            else:
                published_at = datetime.now()

            # Create article
            article = NewsArticle(
                source=item.get("source", {}).get("name", "Unknown"),
                title=item.get("title", ""),
                description=item.get("description", ""),
                url=item.get("url", ""),
                published_at=published_at,
                symbols=[symbol] if symbol else [],
                author=item.get("author"),
            )

            articles.append(article)

        logger.info(f"Fetched {len(articles)} NewsAPI articles")

        return articles

    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        """
        Fetch latest headlines from NewsAPI.

        Args:
            symbol: Not used
            limit: Max headlines (max 100)

        Returns:
            List of recent news articles

        Example:
            >>> headlines = provider.fetch_latest_headlines(limit=10)
        """
        params = {
            "country": self.country,
            "category": "business",
            "pageSize": min(limit, 100),
        }

        logger.info(f"Fetching NewsAPI top headlines (country={self.country})")

        try:
            data = self._request("top-headlines", params)
        except ProviderError as e:
            logger.error(f"NewsAPI headlines fetch failed: {e}")
            raise

        # Parse articles
        articles = []
        for item in data.get("articles", []):
            published_str = item.get("publishedAt")
            if published_str:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            else:
                published_at = datetime.now()

            article = NewsArticle(
                source=item.get("source", {}).get("name", "Unknown"),
                title=item.get("title", ""),
                description=item.get("description", ""),
                url=item.get("url", ""),
                published_at=published_at,
                symbols=[],
                author=item.get("author"),
            )

            articles.append(article)

        logger.info(f"Fetched {len(articles)} NewsAPI headlines")

        return articles
