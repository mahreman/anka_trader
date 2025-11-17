"""
NewsAPI data provider for financial news.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .base import DataProvider

logger = logging.getLogger(__name__)


class NewsAPIProvider(DataProvider):
    """
    NewsAPI data provider for financial news articles.

    Uses NewsAPI.org to fetch news articles related to specific tickers.
    """

    def __init__(self, config: dict):
        """
        Initialize NewsAPI provider.

        Args:
            config: Configuration dictionary with:
                - api_key: NewsAPI.org API key (required)
                - languages: List of language codes (default: ["en"])
                - max_per_query: Max articles per query (default: 100)
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("NewsAPI requires an api_key in config")

        self.languages = config.get("languages", ["en"])
        self.max_per_query = config.get("max_per_query", 100)
        self.base_url = "https://newsapi.org/v2/everything"

    def get_name(self) -> str:
        return "newsapi"

    def fetch_data(
        self,
        start: date,
        end: Optional[date] = None,
        query: str = "bitcoin",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch news articles from NewsAPI.

        Args:
            start: Start date
            end: End date (default: today)
            query: Search query (e.g., "bitcoin", "ethereum", "stock market")
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns: [published_at, title, description, url, source_name, query]
        """
        if end is None:
            end = date.today()

        # NewsAPI free tier only allows queries up to 1 month back
        # Adjust start date if needed
        one_month_ago = date.today() - timedelta(days=30)
        if start < one_month_ago:
            logger.warning(
                f"NewsAPI free tier limits queries to last 30 days. "
                f"Adjusting start from {start} to {one_month_ago}"
            )
            start = one_month_ago

        logger.info(f"Fetching NewsAPI data for query='{query}' from {start} to {end}")

        params = {
            "apiKey": self.api_key,
            "q": query,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "language": ",".join(self.languages),
            "sortBy": "publishedAt",
            "pageSize": min(self.max_per_query, 100),  # NewsAPI max is 100
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown error")
                logger.error(f"NewsAPI error: {error_msg}")
                raise ValueError(f"NewsAPI error: {error_msg}")

            articles = data.get("articles", [])

            if not articles:
                logger.warning(f"No articles returned for query='{query}'")
                return pd.DataFrame()

            # Parse articles
            records = []
            for article in articles:
                records.append({
                    "published_at": pd.to_datetime(article["publishedAt"]),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source_name": article.get("source", {}).get("name", ""),
                    "query": query,
                })

            df = pd.DataFrame(records)

            # Sort by published date
            df = df.sort_values("published_at").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} articles for query='{query}'")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request error for query='{query}': {e}")
            raise
