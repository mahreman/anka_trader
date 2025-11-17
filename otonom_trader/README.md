# Otonom Trader - P0 (Core Version)

Autonomous trading system with anomaly detection and rule-based decision making.

## Overview

Otonom Trader P0 is the core version of an autonomous trading system that:

1. Fetches and stores historical daily OHLCV data for multiple assets
2. Detects price anomalies (spikes and crashes) using statistical methods
3. Generates trading signals (BUY/SELL/HOLD) using rule-based decision engine ("Patron")
4. Maintains full audit trail of all decisions and reasoning

### P0 Assets

- **SUGAR** (SB=F): Sugar #11 Futures
- **WTI** (CL=F): West Texas Intermediate Crude Oil
- **GOLD** (GC=F): Gold Futures
- **SP500** (^GSPC): S&P 500 Index
- **BTC** (BTC-USD): Bitcoin
- **ETH** (ETH-USD): Ethereum

### Technology Stack

- **Language**: Python 3.11+
- **Database**: SQLite (single-file, portable)
- **ORM**: SQLAlchemy 2.0+
- **Data Processing**: pandas, numpy
- **Data Source**: yfinance (Yahoo Finance API)
- **CLI**: typer
- **Testing**: pytest

## Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Setup

1. Clone the repository:
```bash
cd otonom_trader
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python -m otonom_trader.cli init
```

## Usage

### 1. Initialize Database

Create the database schema:

```bash
python -m otonom_trader.cli init
```

To force reinitialize (drops all tables):

```bash
python -m otonom_trader.cli init --force
```

### 2. Ingest Historical Data

Fetch and store historical OHLCV data:

```bash
# Ingest all P0 assets (default: from 2013-01-01 to today)
python -m otonom_trader.cli ingest-data

# Specify date range
python -m otonom_trader.cli ingest-data --start 2020-01-01 --end 2023-12-31

# Ingest specific asset
python -m otonom_trader.cli ingest-data --symbol BTC-USD --start 2020-01-01
```

### 3. Detect Anomalies

Run anomaly detection on price data:

```bash
# Detect anomalies for all assets (default parameters: k=2.5, q=0.8, window=60)
python -m otonom_trader.cli detect-anomalies

# Custom thresholds
python -m otonom_trader.cli detect-anomalies --k 3.0 --q 0.9 --window 90

# Detect for specific asset
python -m otonom_trader.cli detect-anomalies --symbol BTC-USD
```

**Parameters:**
- `--k`: Z-score threshold (default: 2.5)
- `--q`: Volume quantile threshold (default: 0.8)
- `--window`: Rolling window size in days (default: 60)

### 4. List Detected Anomalies

View detected anomalies:

```bash
# List recent anomalies (default: 20 most recent)
python -m otonom_trader.cli list-anomalies

# Filter by symbol
python -m otonom_trader.cli list-anomalies --symbol SUGAR --limit 10
```

### 5. Run Patron (Decision Engine)

Generate trading signals based on anomalies:

```bash
# Run Patron on anomalies from last 30 days
python -m otonom_trader.cli run-patron

# Custom lookback period
python -m otonom_trader.cli run-patron --days 60
```

**Patron Rules (P0):**
- **SPIKE_DOWN + Uptrend** → BUY (mean reversion play)
- **SPIKE_UP + Downtrend** → SELL (dead cat bounce)
- **Other combinations** → HOLD (unclear signal)

### 6. View Trading Decisions

Display generated trading decisions:

```bash
# Show recent decisions
python -m otonom_trader.cli show-decisions

# Filter by symbol
python -m otonom_trader.cli show-decisions --symbol BTC-USD --limit 10

# Filter by signal type
python -m otonom_trader.cli show-decisions --signal BUY
```

### 7. Check Database Status

View database statistics:

```bash
python -m otonom_trader.cli status
```

## Complete Workflow Example

Here's a complete workflow from setup to trading signals:

```bash
# 1. Initialize database
python -m otonom_trader.cli init

# 2. Ingest 10 years of data for all P0 assets
python -m otonom_trader.cli ingest-data --start 2013-01-01

# 3. Detect anomalies
python -m otonom_trader.cli detect-anomalies

# 4. View detected anomalies
python -m otonom_trader.cli list-anomalies --limit 20

# 5. Generate trading signals
python -m otonom_trader.cli run-patron

# 6. View BUY signals
python -m otonom_trader.cli show-decisions --signal BUY

# 7. Check database status
python -m otonom_trader.cli status
```

## Project Structure

```
otonom_trader/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── otonom_trader/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── cli.py                  # Command-line interface
│   ├── domain/                 # Domain models and enums
│   │   ├── enums.py           # AssetClass, SignalType, AnomalyType
│   │   └── models.py          # Asset, Anomaly, Decision dataclasses
│   ├── data/                   # Data layer
│   │   ├── symbols.py         # P0 asset definitions
│   │   ├── db.py              # Database connection
│   │   ├── schema.py          # SQLAlchemy ORM models
│   │   ├── ingest.py          # Data fetching and storage
│   │   └── utils.py           # Utility functions
│   ├── analytics/              # Analytics layer
│   │   ├── returns.py         # Return calculations
│   │   ├── anomaly.py         # Anomaly detection
│   │   └── labeling.py        # Anomaly classification
│   └── patron/                 # Decision engine
│       ├── rules.py           # Rule-based trading logic
│       └── reporter.py        # Formatting and reporting
└── tests/                      # Test suite
    ├── test_data_ingest.py
    ├── test_anomaly_detection.py
    └── test_patron_rules.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=otonom_trader --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detection.py

# Run with verbose output
pytest -v
```

## Configuration

Configuration settings can be found in `otonom_trader/config.py`:

- **Database**: `DB_PATH` (default: `./otonom_trader.db`)
- **Data Range**: `DEFAULT_START_DATE` (default: `2013-01-01`)
- **Anomaly Detection**:
  - `ANOMALY_ZSCORE_THRESHOLD` (default: 2.5)
  - `ANOMALY_VOLUME_QUANTILE` (default: 0.8)
  - `ANOMALY_ROLLING_WINDOW` (default: 60 days)
- **Patron**:
  - `TREND_WINDOW` (default: 20 days)
  - `TREND_THRESHOLD` (default: 2%)

Environment variables:
- `OTONOM_DB_PATH`: Override database path
- `LOG_LEVEL`: Set logging level (default: INFO)

## Anomaly Detection Algorithm

P0 uses a simple but robust statistical approach:

1. **Calculate Returns**: Log returns for each day
2. **Rolling Statistics**: 60-day rolling mean and standard deviation
3. **Z-Score**: Standardized return = (return - rolling_mean) / rolling_std
4. **Volume Rank**: Percentile rank of volume within 60-day window
5. **Classification**:
   - **SPIKE_UP**: z-score > 2.5 AND volume_rank > 0.8
   - **SPIKE_DOWN**: z-score < -2.5 AND volume_rank > 0.8

## Patron Decision Rules

The Patron engine uses simple, transparent rules:

| Anomaly Type | Trend    | Signal | Confidence | Reasoning |
|-------------|----------|--------|------------|-----------|
| SPIKE_DOWN  | UP       | BUY    | 0.6        | Mean reversion after crash in uptrend |
| SPIKE_UP    | DOWN     | SELL   | 0.6        | Dead cat bounce in downtrend |
| SPIKE_DOWN  | DOWN     | HOLD   | 0.4        | Avoid catching falling knife |
| SPIKE_UP    | UP       | HOLD   | 0.4        | Could be continuation or exhaustion |
| SPIKE_DOWN  | FLAT     | HOLD   | 0.35       | Trend unclear |
| SPIKE_UP    | FLAT     | HOLD   | 0.35       | Trend unclear |

**Trend Calculation**: 20-day simple moving average compared to current price (±2% threshold)

## Roadmap

### P0 (Current)
- ✓ 6 assets, daily data
- ✓ Basic anomaly detection
- ✓ Rule-based Patron
- ✓ Manual event labeling support
- ✓ CLI interface

### P1 (Future)
- TODO: News/event integration
- TODO: Sentiment analysis
- TODO: Enhanced Patron rules
- TODO: Backtesting framework
- TODO: Performance metrics

### P2 (Future)
- TODO: Machine learning models
- TODO: Multi-timeframe analysis
- TODO: Portfolio management
- TODO: Real-time data feeds
- TODO: Web dashboard

## Contributing

This is a P0 (core) version. Code is intentionally simple and readable. Future enhancements will be built on this foundation.

## License

MIT License (or specify your license)

## Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without proper testing, risk management, and understanding of financial markets. Trading involves risk of loss.

## Contact

For questions or feedback, please open an issue on GitHub.
