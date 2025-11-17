# Data Provider Integration Guide

Complete guide for integrating real data sources into anka_trader.

## Overview

The anka_trader system now supports multiple real data sources:

1. **Price Data Providers**: Binance, yfinance, Polygon, Alpha Vantage
2. **News Data Providers**: NewsAPI, Polygon News, RSS feeds
3. **Macro Data Providers**: FRED, World Bank, Trading Economics

## Architecture

```
otonom_trader/
├── providers/
│   ├── base.py              # Abstract base classes
│   ├── config.py            # Provider configuration
│   ├── factory.py           # Provider factory
│   ├── price_binance.py     # Binance price provider
│   ├── price_yfinance.py    # yfinance provider
│   ├── news_newsapi.py      # NewsAPI provider
│   ├── news_rss.py          # RSS feed provider
│   ├── macro_fred.py        # FRED macro provider
│   └── *_dummy.py           # Dummy providers for testing
├── data/
│   ├── ingest.py            # Original yfinance ingestion
│   ├── ingest_providers.py  # Enhanced provider-based ingestion
│   └── schema.py            # Database models (+ NewsArticle, MacroIndicator)
└── config/
    └── providers.yaml       # Provider configuration
```

## Configuration

### Provider Configuration (`config/providers.yaml`)

```yaml
# Global settings
fallback_enabled: true  # Use fallback providers if primary fails
cache_enabled: true     # Cache provider responses

# Price Data Providers
price_providers:
  - type: binance
    enabled: true
    api_key: ""  # Optional for public endpoints
    api_secret: ""
    base_url: "https://api.binance.com"
    rate_limit_per_minute: 1200
    timeout_seconds: 30
    extra:
      use_testnet: false

  - type: yfinance
    enabled: true
    rate_limit_per_minute: 2000
    timeout_seconds: 30

# News Data Providers
news_providers:
  - type: newsapi
    enabled: false
    api_key: ""  # Get from newsapi.org (free tier: 100 req/day)
    rate_limit_per_minute: 100

  - type: rss
    enabled: true
    rate_limit_per_minute: 10
    extra:
      feeds:
        - url: "https://www.cnbc.com/id/100003114/device/rss/rss.html"
          name: "CNBC Top News"
        - url: "https://feeds.bloomberg.com/markets/news.rss"
          name: "Bloomberg Markets"

# Macro Data Providers
macro_providers:
  - type: fred
    enabled: true
    api_key: ""  # Get free API key from fred.stlouisfed.org
    base_url: "https://api.stlouisfed.org/fred"
    rate_limit_per_minute: 120
    extra:
      default_indicators:
        - "DFF"      # Federal Funds Rate
        - "UNRATE"   # Unemployment Rate
        - "CPIAUCSL" # CPI (inflation)
```

## Provider Types

### 1. Price Data Providers

#### Binance Provider

**Supports:**
- Crypto spot pairs (BTCUSDT, ETHUSDT, etc.)
- Historical OHLCV data (klines)
- Real-time quotes (24hr ticker)

**API Key:**
- Not required for public endpoints (historical data, quotes)
- Required for authenticated endpoints (if added later)

**Rate Limits:**
- ~1200 requests/minute
- Auto-throttling implemented

**Example:**
```python
from otonom_trader.providers import get_primary_price_provider

provider = get_primary_price_provider()
bars = provider.fetch_ohlcv("BTC-USD", date(2024, 1, 1), date(2024, 1, 31))
quote = provider.fetch_latest_quote("BTC-USD")

print(f"Fetched {len(bars)} bars")
print(f"Current price: ${quote.last:,.2f}")
```

#### yfinance Provider

**Supports:**
- Stocks (AAPL, TSLA, MSFT)
- ETFs (SPY, QQQ, VTI)
- Indices (^GSPC, ^DJI, ^IXIC)
- Crypto (BTC-USD, ETH-USD)
- Forex (EURUSD=X)
- Futures (ES=F, GC=F)

**API Key:**
- Not required (uses Yahoo Finance API)

**Rate Limits:**
- Generous (2000+ req/min)

**Example:**
```python
from otonom_trader.providers.price_yfinance import YFinanceProvider

provider = YFinanceProvider({})
bars = provider.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))
quote = provider.fetch_latest_quote("AAPL")
```

### 2. News Data Providers

#### NewsAPI Provider

**Supports:**
- 80,000+ news sources
- Search by keyword
- Filter by date range
- Multiple languages

**API Key:**
- Required (get from https://newsapi.org/)
- Free tier: 100 requests/day
- Paid tiers: Higher limits

**Rate Limits:**
- Free: 100 req/day
- Developer: 250 req/day
- Business: 1000 req/day

**Example:**
```python
from otonom_trader.providers.news_newsapi import NewsAPIProvider

provider = NewsAPIProvider({"api_key": "your_api_key"})
articles = provider.fetch_news(
    query="Bitcoin",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    limit=50
)

for article in articles:
    print(f"{article.published_at}: {article.title}")
```

#### RSS Feed Provider

**Supports:**
- CNBC RSS feeds
- Bloomberg RSS feeds
- Reuters RSS feeds
- Any standard RSS/Atom feed

**API Key:**
- Not required

**Rate Limits:**
- Self-imposed (10 req/min to be respectful)

**Example:**
```python
from otonom_trader.providers.news_rss import RSSNewsProvider

provider = RSSNewsProvider({
    "extra": {
        "feeds": [
            {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "name": "CNBC"},
            {"url": "https://feeds.bloomberg.com/markets/news.rss", "name": "Bloomberg"},
        ]
    }
})

articles = provider.fetch_latest_headlines(limit=20)
```

### 3. Macro Data Providers

#### FRED Provider

**Supports:**
- 800,000+ economic time series
- Interest rates (DFF, GS10, T10Y2Y)
- Inflation (CPIAUCSL, PCEPI)
- Employment (UNRATE, PAYEMS)
- GDP (GDP, GDPC1)
- And many more

**API Key:**
- Required (get free key from https://fred.stlouisfed.org/)
- Free tier: Unlimited requests (120 req/min limit)

**Rate Limits:**
- 120 requests/minute

**Example:**
```python
from otonom_trader.providers.macro_fred import FREDProvider

provider = FREDProvider({"api_key": "your_api_key"})

# Fetch unemployment rate
unrate = provider.fetch_indicator("UNRATE", date(2023, 1, 1), date(2024, 1, 1))
print(f"Fetched {len(unrate)} unemployment rate observations")

# Fetch latest value
latest = provider.fetch_latest_value("DFF")
print(f"Current Federal Funds Rate: {latest.value}%")

# Fetch multiple indicators
codes = ["DFF", "UNRATE", "CPIAUCSL"]
data = provider.fetch_multiple_indicators(codes, date(2023, 1, 1), date(2024, 1, 1))
```

## Data Ingestion

### Enhanced Ingestion Functions

```python
from otonom_trader.data import get_session
from otonom_trader.data.ingest_providers import (
    ingest_price_data,
    ingest_news_data,
    ingest_macro_data,
    ingest_all_data_types,
    ingest_incremental_all,
)

with get_session() as session:
    # Ingest price data
    price_count = ingest_price_data(
        session, "BTC-USD",
        date(2024, 1, 1), date(2024, 1, 31)
    )

    # Ingest news data
    news_count = ingest_news_data(
        session, symbol="BTC-USD",
        start_date=datetime(2024, 1, 1),
        limit=100
    )

    # Ingest macro data
    macro_count = ingest_macro_data(
        session, "UNRATE",
        date(2024, 1, 1), date(2024, 1, 31)
    )

    # Ingest all data types at once
    counts = ingest_all_data_types(
        session, "BTC-USD",
        date(2024, 1, 1), date(2024, 1, 31),
        macro_indicators=["DFF", "UNRATE", "CPIAUCSL"]
    )
    print(counts)  # {'price': 31, 'news': 42, 'macro': 3}

    # Incremental ingestion (last 7 days)
    results = ingest_incremental_all(
        session,
        ["BTC-USD", "ETH-USD"],
        days_back=7
    )
```

### Database Schema

#### NewsArticle Table

```python
class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    source = Column(String(100))  # NewsAPI, Bloomberg, etc.
    title = Column(Text)
    description = Column(Text)
    url = Column(Text, unique=True)
    published_at = Column(DateTime, index=True)
    author = Column(String(200))

    sentiment = Column(Float)  # -1 (bearish) to +1 (bullish)
    sentiment_source = Column(String(50))  # LLM model used

    symbols = Column(String(500))  # Comma-separated (e.g., "BTC-USD,ETH-USD")
```

#### MacroIndicator Table

```python
class MacroIndicator(Base):
    __tablename__ = "macro_indicators"

    id = Column(Integer, primary_key=True)
    indicator_code = Column(String(50), index=True)  # e.g., "GDP", "UNRATE"
    indicator_name = Column(String(200))  # e.g., "Unemployment Rate"
    date = Column(Date, index=True)

    value = Column(Float)
    unit = Column(String(50))  # e.g., "Percent"
    frequency = Column(String(20))  # e.g., "Monthly"

    provider = Column(String(50))  # e.g., "FRED"
```

## Usage Examples

### Example 1: Fetch Latest Bitcoin Price

```python
from otonom_trader.providers import get_primary_price_provider

provider = get_primary_price_provider()
quote = provider.fetch_latest_quote("BTC-USD")

print(f"BTC Price: ${quote.last:,.2f}")
print(f"Bid: ${quote.bid:,.2f}")
print(f"Ask: ${quote.ask:,.2f}")
print(f"24h Volume: {quote.volume:,.0f}")
```

### Example 2: Fetch Latest News

```python
from otonom_trader.providers import get_primary_news_provider

provider = get_primary_news_provider()
headlines = provider.fetch_latest_headlines(limit=10)

for article in headlines:
    print(f"{article.published_at.strftime('%Y-%m-%d %H:%M')}: {article.title}")
    print(f"  Source: {article.source}")
    print(f"  URL: {article.url}\n")
```

### Example 3: Fetch Economic Indicators

```python
from otonom_trader.providers import get_primary_macro_provider
from datetime import date

provider = get_primary_macro_provider()

# Latest unemployment rate
latest_unrate = provider.fetch_latest_value("UNRATE")
print(f"Current Unemployment Rate: {latest_unrate.value}%")

# Historical GDP
gdp = provider.fetch_indicator("GDP", date(2020, 1, 1), date(2024, 1, 1))
print(f"GDP observations: {len(gdp)}")
```

### Example 4: Complete Data Ingestion Pipeline

```python
from otonom_trader.data import get_session
from otonom_trader.data.ingest_providers import ingest_all_data_types
from datetime import date

# Ingest all data for BTC over the past year
with get_session() as session:
    counts = ingest_all_data_types(
        session,
        symbol="BTC-USD",
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        macro_indicators=["DFF", "UNRATE", "CPIAUCSL", "T10Y2Y"],
    )

    print(f"Ingested:")
    print(f"  - {counts['price']} price bars")
    print(f"  - {counts['news']} news articles")
    print(f"  - {counts['macro']} macro observations")
```

## Obtaining API Keys

### NewsAPI

1. Go to https://newsapi.org/
2. Click "Get API Key"
3. Sign up for free (100 requests/day)
4. Copy API key to `config/providers.yaml`

### FRED (Federal Reserve Economic Data)

1. Go to https://fred.stlouisfed.org/
2. Create free account
3. Request API key from https://fred.stlouisfed.org/docs/api/api_key.html
4. Copy API key to `config/providers.yaml`

### Binance (optional for crypto)

1. Go to https://www.binance.com/
2. Sign up and enable API access
3. Generate API key and secret
4. For testing: Use testnet at https://testnet.binance.vision/

## Provider Priority and Fallbacks

When multiple providers are enabled:

1. **Primary provider** = First enabled provider in config
2. **Fallback providers** = Subsequent enabled providers
3. **Automatic failover** if `fallback_enabled: true`

Example:
```yaml
price_providers:
  - type: binance    # Primary (used first)
    enabled: true
  - type: yfinance   # Fallback (used if Binance fails)
    enabled: true
```

## Rate Limiting

All providers implement rate limiting to respect API limits:

- Automatic throttling between requests
- Sleep intervals when approaching limits
- Error handling for 429 (Too Many Requests)

**Default rate limits:**
- Binance: 1200 req/min
- yfinance: 2000 req/min
- NewsAPI: 100 req/day (free tier)
- FRED: 120 req/min

## Caching

When `cache_enabled: true`, provider responses are cached:

- **OHLCV data**: 1 hour (for recent data)
- **News**: 5 minutes
- **Macro**: 24 hours (macro data updates slowly)

Cache is in-memory and cleared on restart.

## Error Handling

All providers raise standardized exceptions:

- `ProviderError`: General provider error
- `RateLimitError`: Rate limit exceeded
- `AuthenticationError`: API key authentication failed

```python
from otonom_trader.providers import ProviderError, RateLimitError

try:
    bars = provider.fetch_ohlcv("BTC-USD", start, end)
except RateLimitError:
    logger.warning("Rate limit hit, waiting...")
    time.sleep(60)
    bars = provider.fetch_ohlcv("BTC-USD", start, end)
except ProviderError as e:
    logger.error(f"Provider failed: {e}")
    # Fall back to alternative provider
```

## Integration with Analysts

### News/LLM Analyst (Analist-2)

```python
from otonom_trader.data import get_session, NewsArticleORM

# Query recent news for sentiment analysis
with get_session() as session:
    recent_news = (
        session.query(NewsArticleORM)
        .filter(NewsArticleORM.symbols.like("%BTC-USD%"))
        .order_by(NewsArticleORM.published_at.desc())
        .limit(20)
        .all()
    )

    # Analyze sentiment with LLM
    for article in recent_news:
        # TODO: Pass to LLM for sentiment analysis
        # article.sentiment = llm_analyze(article.title, article.description)
        pass
```

### Risk/Regime Analyst (Analist-3)

```python
from otonom_trader.data import get_session, MacroIndicatorORM

# Query macro indicators for regime detection
with get_session() as session:
    # Get latest Federal Funds Rate
    dff = (
        session.query(MacroIndicatorORM)
        .filter_by(indicator_code="DFF")
        .order_by(MacroIndicatorORM.date.desc())
        .first()
    )

    # Get latest unemployment rate
    unrate = (
        session.query(MacroIndicatorORM)
        .filter_by(indicator_code="UNRATE")
        .order_by(MacroIndicatorORM.date.desc())
        .first()
    )

    # Detect regime based on macro conditions
    if dff.value > 5.0 and unrate.value < 4.0:
        regime = "expansionary"
    elif dff.value < 2.0 and unrate.value > 6.0:
        regime = "recessionary"
    else:
        regime = "neutral"
```

## Deployment Workflow

### Phase 1: Local Development (Free Tiers)

1. **Configure free providers:**
   ```yaml
   price_providers:
     - type: yfinance
       enabled: true

   news_providers:
     - type: rss
       enabled: true

   macro_providers:
     - type: fred
       enabled: true
       api_key: "your_free_fred_api_key"
   ```

2. **Test ingestion:**
   ```bash
   python -c "
   from otonom_trader.data import get_session
   from otonom_trader.data.ingest_providers import ingest_all_data_types
   from datetime import date

   with get_session() as session:
       counts = ingest_all_data_types(
           session, 'BTC-USD',
           date(2024, 1, 1), date(2024, 1, 31)
       )
       print(counts)
   "
   ```

### Phase 2: Production (Paid Tiers)

1. **Upgrade to paid providers:**
   - NewsAPI Developer tier
   - Polygon.io (for more reliable data)
   - Consider Alpha Vantage premium

2. **Enable multiple providers:**
   ```yaml
   price_providers:
     - type: binance
       enabled: true
     - type: polygon
       enabled: true
     - type: yfinance
       enabled: true
   ```

3. **Set up automated ingestion:**
   ```python
   # In daemon loop
   with get_session() as session:
       ingest_incremental_all(
           session,
           symbols=["BTC-USD", "ETH-USD", "AAPL", "TSLA"],
           days_back=1,  # Daily incremental update
       )
   ```

## Troubleshooting

### Issue: Rate Limit Errors

**Symptom:**
```
RateLimitError: NewsAPI rate limit exceeded
```

**Solution:**
- Check daily quota remaining
- Reduce `limit` parameter in fetch calls
- Enable caching to reduce API calls
- Upgrade to paid tier

### Issue: No Data Returned

**Symptom:**
```
No bars returned for BTC-USD
```

**Solution:**
- Verify symbol format (Binance uses "BTCUSDT", yfinance uses "BTC-USD")
- Check date range (some providers have limits)
- Verify API key is valid
- Check provider status/uptime

### Issue: Authentication Failed

**Symptom:**
```
AuthenticationError: FRED authentication failed
```

**Solution:**
- Verify API key in `config/providers.yaml`
- Check API key permissions
- Ensure API key hasn't expired
- Request new API key if needed

## Summary

**Key Features:**
1. ✅ Multi-provider support (price, news, macro)
2. ✅ Automatic failover with fallback providers
3. ✅ Rate limiting and error handling
4. ✅ Database integration with ORM models
5. ✅ Incremental ingestion
6. ✅ Flexible configuration (YAML)

**Supported Data Sources:**
- **Price**: Binance, yfinance, Polygon, Alpha Vantage
- **News**: NewsAPI, RSS feeds, Polygon News
- **Macro**: FRED, World Bank, Trading Economics

**Ready for production** with comprehensive data integration! 🚀
