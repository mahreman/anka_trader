"""
RSS news provider for financial news feeds.

Parses RSS/Atom feeds from various financial news sources.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

import requests

from .base import NewsProvider, NewsArticle, ProviderError

logger = logging.getLogger(__name__)


class RSSNewsProvider(NewsProvider):
    """
    RSS feed news provider.

    Supports:
    - CNBC RSS feeds
    - Bloomberg RSS feeds
    - Reuters RSS feeds
    - Any standard RSS/Atom feed

    No API key required.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get RSS feed URLs from config
        self.feeds = config.get("extra", {}).get("feeds", [])

        if not self.feeds:
            # Default feeds
            self.feeds = [
                {
                    "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                    "name": "CNBC Top News",
                },
                {
                    "url": "https://feeds.bloomberg.com/markets/news.rss",
                    "name": "Bloomberg Markets",
                },
            ]

        logger.info(f"RSSNewsProvider initialized with {len(self.feeds)} feeds")

    def _fetch_rss_feed(self, feed_url: str, feed_name: str) -> List[NewsArticle]:
        """
        Fetch and parse RSS feed.

        Args:
            feed_url: RSS feed URL
            feed_name: Feed name for attribution

        Returns:
            List of news articles from feed

        Raises:
            ProviderError: If fetch fails
        """
        try:
            response = requests.get(
                feed_url,
                timeout=self.config.get("timeout_seconds", 30),
                headers={"User-Agent": "anka_trader/1.0"},
            )

            if response.status_code != 200:
                raise ProviderError(f"RSS fetch failed: HTTP {response.status_code}")

            # Parse XML
            root = ET.fromstring(response.content)

            # Determine feed type (RSS vs Atom)
            if root.tag == "{http://www.w3.org/2005/Atom}feed":
                return self._parse_atom_feed(root, feed_name)
            else:
                return self._parse_rss_feed(root, feed_name)

        except requests.RequestException as e:
            raise ProviderError(f"RSS request failed for {feed_url}: {e}")
        except ET.ParseError as e:
            raise ProviderError(f"RSS parse failed for {feed_url}: {e}")

    def _parse_rss_feed(self, root: ET.Element, feed_name: str) -> List[NewsArticle]:
        """
        Parse RSS 2.0 feed.

        Args:
            root: XML root element
            feed_name: Feed name

        Returns:
            List of news articles
        """
        articles = []

        # Find all <item> elements
        for item in root.findall(".//item"):
            # Extract fields
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            link = item.findtext("link", "")
            pub_date_str = item.findtext("pubDate", "")
            author = item.findtext("author", "") or item.findtext("dc:creator", "")

            # Parse publication date
            if pub_date_str:
                try:
                    # RFC 822 format: "Mon, 01 Jan 2024 12:00:00 GMT"
                    published_at = datetime.strptime(
                        pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except ValueError:
                    # Try alternative formats
                    try:
                        published_at = datetime.strptime(
                            pub_date_str, "%a, %d %b %Y %H:%M:%S %z"
                        )
                    except ValueError:
                        published_at = datetime.now()
            else:
                published_at = datetime.now()

            # Create article
            article = NewsArticle(
                source=feed_name,
                title=title,
                description=description or title,  # Fallback to title if no description
                url=link,
                published_at=published_at,
                symbols=[],  # RSS feeds don't tag symbols
                author=author if author else None,
            )

            articles.append(article)

        return articles

    def _parse_atom_feed(self, root: ET.Element, feed_name: str) -> List[NewsArticle]:
        """
        Parse Atom feed.

        Args:
            root: XML root element
            feed_name: Feed name

        Returns:
            List of news articles
        """
        articles = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Find all <entry> elements
        for entry in root.findall("atom:entry", ns):
            # Extract fields
            title = entry.findtext("atom:title", "", ns)
            summary = entry.findtext("atom:summary", "", ns)

            # Get link
            link_elem = entry.find("atom:link[@rel='alternate']", ns)
            if link_elem is None:
                link_elem = entry.find("atom:link", ns)

            link = link_elem.get("href", "") if link_elem is not None else ""

            # Parse published date
            published_str = entry.findtext("atom:published", "", ns)
            if not published_str:
                published_str = entry.findtext("atom:updated", "", ns)

            if published_str:
                try:
                    # ISO 8601 format
                    published_at = datetime.fromisoformat(
                        published_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    published_at = datetime.now()
            else:
                published_at = datetime.now()

            # Get author
            author_elem = entry.find("atom:author/atom:name", ns)
            author = author_elem.text if author_elem is not None else None

            # Create article
            article = NewsArticle(
                source=feed_name,
                title=title,
                description=summary or title,
                url=link,
                published_at=published_at,
                symbols=[],
                author=author,
            )

            articles.append(article)

        return articles

    def fetch_news(
        self,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """
        Fetch news from RSS feeds.

        Args:
            symbol: Not used (RSS feeds don't support filtering)
            query: Not used (RSS feeds don't support search)
            start_date: Filter articles after this date
            end_date: Filter articles before this date
            limit: Max articles

        Returns:
            List of news articles

        Example:
            >>> provider = RSSNewsProvider({})
            >>> articles = provider.fetch_news(limit=50)
        """
        all_articles = []

        # Fetch from all configured feeds
        for feed in self.feeds:
            feed_url = feed.get("url")
            feed_name = feed.get("name", feed_url)

            logger.info(f"Fetching RSS feed: {feed_name}")

            try:
                articles = self._fetch_rss_feed(feed_url, feed_name)
                all_articles.extend(articles)
            except ProviderError as e:
                logger.warning(f"Failed to fetch RSS feed {feed_name}: {e}")
                continue

        # Filter by date range if specified
        if start_date or end_date:
            filtered = []
            for article in all_articles:
                if start_date and article.published_at < start_date:
                    continue
                if end_date and article.published_at > end_date:
                    continue
                filtered.append(article)

            all_articles = filtered

        # Sort by published date (newest first)
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        # Apply limit
        all_articles = all_articles[:limit]

        logger.info(f"Fetched {len(all_articles)} RSS articles from {len(self.feeds)} feeds")

        return all_articles

    def fetch_latest_headlines(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsArticle]:
        """
        Fetch latest headlines from RSS feeds.

        Args:
            symbol: Not used
            limit: Max headlines

        Returns:
            List of recent news articles
        """
        return self.fetch_news(limit=limit)
