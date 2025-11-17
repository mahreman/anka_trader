# Data Ingestion Guide

This guide explains how to set up and run the real-time data ingestion pipeline for Anka Trader.

## Overview

The data ingestion system fetches data from multiple sources:
- **Price Data**: Binance (crypto) and Yahoo Finance (stocks/FX)
- **News Data**: NewsAPI.org
- **Macro Data**: FRED (Federal Reserve Economic Data)

## Architecture

```
providers/
├── base.py                   # Abstract provider interface
├── binance_provider.py       # Binance crypto data
├── yfinance_provider.py      # Yahoo Finance data
├── newsapi_provider.py       # News articles
├── fred_provider.py          # Macroeconomic indicators
├── config.py                 # Config loader utilities
└── ingest_providers.py       # Orchestration logic

scripts/
├── run_initial_ingest.py     # One-time full sync
└── run_incremental_ingest.py # Periodic incremental updates

config/
└── providers.yaml            # API keys and configuration
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r otonom_trader/requirements.txt
```

Required packages:
- `pyyaml`: Config file parsing
- `requests`: HTTP requests for APIs
- `yfinance`: Yahoo Finance data
- `pandas`, `sqlalchemy`: Data handling

### 2. Configure API Keys

Edit `config/providers.yaml` and add your API keys:

```yaml
# Binance (optional, for crypto data)
binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_API_SECRET"

# NewsAPI (required for news)
news:
  newsapi:
    api_key: "YOUR_NEWSAPI_KEY"

# FRED (required for macro data)
macro:
  fred:
    api_key: "YOUR_FRED_API_KEY"
```

#### Getting API Keys

**Binance** (optional, free tier available):
1. Sign up at https://www.binance.com
2. Go to Account > API Management
3. Create new API key (read-only permissions are sufficient)

**NewsAPI** (required, free tier: 100 requests/day):
1. Sign up at https://newsapi.org/register
2. Copy your API key from the dashboard
3. Note: Free tier only allows queries up to 1 month back

**FRED** (required, completely free):
1. Sign up at https://fred.stlouisfed.org
2. Request API key at https://fred.stlouisfed.org/docs/api/api_key.html
3. Check your email for the key

### 3. Configure Data Sources

Edit `config/providers.yaml` to customize which assets and series to track:

```yaml
price:
  primary: "binance"  # or "yfinance"
  universes:
    crypto:
      symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    fx:
      symbols: ["EURUSD=X", "GBPUSD=X"]
    equity:
      symbols: ["SPY", "QQQ", "AAPL"]

news:
  newsapi:
    tickers: ["bitcoin", "ethereum", "stock market"]
    max_per_symbol: 50

macro:
  fred:
    series: ["DGS10", "CPIAUCSL", "GDP", "UNRATE"]
```

## Running Data Ingestion

### Initial Full Sync

Run once to bootstrap your database with historical data:

```bash
# Default: crypto universe, from 2020-01-01 to today
python scripts/run_initial_ingest.py

# Custom date range
python scripts/run_initial_ingest.py --start-date 2018-01-01 --end-date 2023-12-31

# Different universe
python scripts/run_initial_ingest.py --universe all

# All options
python scripts/run_initial_ingest.py \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --universe crypto \
  --config config/providers.yaml
```

**Expected output:**
```
================================================================================
ANKA TRADER - INITIAL DATA INGESTION
================================================================================
Start date: 2020-01-01
End date: 2024-12-31
Universe: crypto
Config: config/providers.yaml
================================================================================
...
================================================================================
INGESTION COMPLETED SUCCESSFULLY
================================================================================
Price data: {'BTCUSDT': 1825, 'ETHUSDT': 1825, 'SOLUSDT': 1825}
News articles: 150
Macro indicators: 1200
================================================================================
```

### Incremental Updates

Run periodically to fetch recent data:

```bash
# Default: last 7 days
python scripts/run_incremental_ingest.py

# Custom lookback period
python scripts/run_incremental_ingest.py --days-back 30

# Different universe
python scripts/run_incremental_ingest.py --universe all
```

**Scheduling with cron:**

```bash
# Run every 15 minutes
*/15 * * * * cd /path/to/anka_trader && /path/to/.venv/bin/python scripts/run_incremental_ingest.py >> logs/incremental_ingest.log 2>&1

# Run every hour
0 * * * * cd /path/to/anka_trader && /path/to/.venv/bin/python scripts/run_incremental_ingest.py >> logs/incremental_ingest.log 2>&1

# Run daily at 9 AM
0 9 * * * cd /path/to/anka_trader && /path/to/.venv/bin/python scripts/run_incremental_ingest.py >> logs/incremental_ingest.log 2>&1
```

## Data Schema

### Price Data (DailyBar table)

```python
symbol_id: int       # Foreign key to symbols table
date: date           # Trading date
open: float          # Opening price
high: float          # High price
low: float           # Low price
close: float         # Closing price
volume: float        # Trading volume
adj_close: float     # Adjusted close (optional)
```

### News Data (NewsArticle table)

```python
provider: str        # "newsapi", etc.
title: str           # Article title
description: str     # Article summary
url: str             # Article URL (unique)
published_at: datetime  # Publication timestamp
source_name: str     # News source
symbols: str         # Related tickers (comma-separated)
sentiment_score: float  # Optional: -1 to 1
```

### Macro Data (MacroIndicator table)

```python
provider: str        # "fred", etc.
series_id: str       # FRED series ID (e.g., "DGS10")
series_name: str     # Human-readable name
date: date           # Observation date
value: float         # Indicator value
units: str           # Units (e.g., "Percent")
```

## Troubleshooting

### Error: "Provider config not found"
- Ensure `config/providers.yaml` exists
- Check the `--config` argument path

### Error: "NewsAPI requires an api_key"
- Add your NewsAPI key to `config/providers.yaml`
- Verify the key is valid at https://newsapi.org/account

### Error: "FRED API error"
- Add your FRED API key to `config/providers.yaml`
- Verify the series IDs are valid at https://fred.stlouisfed.org

### Error: "Binance API error"
- Check your Binance API key and secret
- Verify API key has read permissions
- Check if you're hitting rate limits

### NewsAPI free tier limitations
- Free tier: 100 requests/day, 1 month history
- Consider upgrading or using multiple tickers strategically
- Script automatically limits lookback to 30 days

### Rate limits
- Binance: 1200 requests/minute (weight-based)
- NewsAPI: 100 requests/day (free tier)
- FRED: No strict limits (be reasonable)

## Integration with Analysts

Once data is ingested, analysts can access it:

```python
from otonom_trader.data import get_session, DailyBar, NewsArticle, MacroIndicator

# Fetch price data
with next(get_session()) as session:
    bars = session.query(DailyBar).filter_by(symbol_id=1).all()

# Fetch news
with next(get_session()) as session:
    news = session.query(NewsArticle).filter(
        NewsArticle.symbols.like("%bitcoin%")
    ).all()

# Fetch macro data
with next(get_session()) as session:
    indicators = session.query(MacroIndicator).filter_by(
        series_id="DGS10"
    ).all()
```

## Next Steps

After setting up data ingestion:
1. Run initial sync to populate historical data
2. Set up cron job for incremental updates
3. Integrate with Analist-2 (news sentiment) and Analist-3 (macro)
4. Monitor logs for errors: `tail -f logs/incremental_ingest.log`

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review `config/providers.yaml` for misconfigurations
- Verify API keys are valid and have proper permissions
