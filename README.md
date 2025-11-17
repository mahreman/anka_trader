# Otonom Trader

An autonomous cryptocurrency and asset trading system powered by multi-analyst ensemble decision-making, regime detection, and LLM-enhanced market analysis.

## Overview

Otonom Trader is a research-grade algorithmic trading system that combines:
- **Analist-1**: Technical anomaly detection (spikes, crashes, breakouts)
- **Analist-2**: News/macro sentiment analysis with LLM integration
- **Analist-3**: Regime/DSI-based risk assessment
- **Portfolio constraints**: Turnover limits, cooldown periods, correlation netting
- **Risk management**: Stop-loss, take-profit, volatility-scaled position sizing
- **Monitoring dashboard**: Real-time Streamlit interface

## Architecture

```
otonom_trader/
├── data/           # Database schema and data ingestion
├── analytics/      # Anomaly detection, regime analysis, LLM agents
├── domain/         # Core business models
├── patron/         # Multi-analyst decision engine
├── daemon/         # Production daemon and paper trader
├── eval/           # Backtesting and performance evaluation
└── config/         # Strategy configuration loader

strategies/         # YAML strategy definitions
scripts/            # Research and backtest runners
reports/            # Generated performance reports
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/anka_trader.git
cd anka_trader

# Install dependencies
pip install -e otonom_trader/

# Install dashboard dependencies (optional)
pip install -r requirements-dashboard.txt
```

## Quick Start

### 1. Run Paper Trading Daemon

```bash
# Single cycle (cron-friendly)
otonom-trader daemon-once --db trader.db

# Continuous loop (15-minute intervals)
otonom-trader daemon-loop --db trader.db --interval 900
```

### 2. Launch Monitoring Dashboard

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to view:
- Portfolio equity curve and drawdown
- Recent anomalies and decisions
- Multi-analyst reasoning
- Daemon health metrics

### 3. Run Backtests

```bash
# Run baseline_v1 strategy backtest
python scripts/run_research_backtests.py --strategy baseline_v1

# Backtest specific symbols
python scripts/run_research_backtests.py --strategy baseline_v1 --symbols BTC-USD ETH-USD
```

Results saved to `reports/baseline_v1_<timestamp>.html`

## Baseline v1 Strategy Performance

**Strategy**: `baseline_v1` - Multi-analyst ensemble with risk management

**Configuration**:
- Analist-1: Technical (weight 1.0)
- Analist-2: News/Macro/LLM (weight 1.2)
- Analist-3: Regime/DSI Risk (weight 0.8)
- Risk per trade: 1%
- Stop-loss: 5%, Take-profit: 10%
- Max daily trades: 10

### Backtest Results

| Symbol  | Period       | CAGR   | Sharpe | Max DD  | Win Rate | Trades |
|---------|--------------|--------|--------|---------|----------|--------|
| BTC-USD | 2017-2025    | TBD    | TBD    | TBD     | TBD      | TBD    |
| ETH-USD | 2017-2025    | TBD    | TBD    | TBD     | TBD      | TBD    |
| GC=F    | 2008-2025    | TBD    | TBD    | TBD     | TBD      | TBD    |
| ^GSPC   | 2008-2025    | TBD    | TBD    | TBD     | TBD      | TBD    |

*Note: Run `scripts/run_research_backtests.py` to generate results.*

## Strategy Configuration

Strategies are defined in YAML files in the `strategies/` directory.

Example (`strategies/baseline_v1.yaml`):

```yaml
name: "baseline_v1"
description: "Multi-analyst ensemble strategy with risk management"
version: "1.0.0"

# Analist weights
analist_1:
  enabled: true
  weight: 1.0
analist_2:
  enabled: true
  weight: 1.2
analist_3:
  enabled: true
  weight: 0.8

# Risk management
risk_management:
  position_sizing:
    risk_per_trade_pct: 1.0
  stop_loss:
    enabled: true
    percentage: 5.0
  take_profit:
    enabled: true
    percentage: 10.0

# Portfolio constraints
portfolio_constraints:
  turnover_limits:
    max_daily_trades: 10
  cooldown:
    hours_after_flip: 24
```

See `strategies/baseline_v1.yaml` for full configuration.

## CLI Commands

### Daemon Operations

```bash
# Run once (for cron)
otonom-trader daemon-once --db trader.db

# Run in loop
otonom-trader daemon-loop --db trader.db --interval 900

# Check status
otonom-trader status --db trader.db
```

### Data Ingestion

```bash
# Ingest price data
otonom-trader ingest --db trader.db --symbols BTC-USD ETH-USD

# Detect anomalies
otonom-trader anomaly --db trader.db

# Compute regimes
otonom-trader regime --db trader.db
```

## Development

### Running Tests

```bash
pytest otonom_trader/tests/
```

### Code Structure

- **Data Layer**: SQLAlchemy ORM models in `otonom_trader/data/schema.py`
- **Analytics**: Anomaly detection, regime analysis, LLM agents
- **Patron**: Multi-analyst ensemble decision engine
- **Daemon**: Production trading daemon with paper trader
- **Evaluation**: Backtesting and performance metrics

### Adding a New Analyst

1. Create module in `otonom_trader/analytics/analyst_*.py`
2. Implement `generate_*_analyst_signal()` function
3. Add to ensemble in `otonom_trader/patron/rules.py`
4. Update strategy YAML configuration

## Roadmap

### Phase 1: Foundation ✅
- [x] Multi-analyst ensemble (Analist-1/2/3)
- [x] Regime detection and DSI
- [x] LLM integration (DeepSeek, OpenAI, Ollama)
- [x] Paper trading daemon
- [x] Monitoring dashboard

### Phase 2: Research Infrastructure (Current)
- [x] Strategy YAML configuration
- [x] Backtest runner with performance metrics
- [ ] Experiment tracking database
- [ ] Grid search for parameter optimization
- [ ] Statistical significance testing

### Phase 3: Production (Planned)
- [ ] Broker abstraction layer
- [ ] Shadow mode (parallel live/paper)
- [ ] Kill-switch and guardrails
- [ ] Alert engine (email, Telegram)
- [ ] Packaging and distribution

## License

MIT License

## Acknowledgments

Built with:
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Plotly](https://plotly.com/) - Interactive charts
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data

## Contact

For questions or contributions, please open an issue on GitHub.
