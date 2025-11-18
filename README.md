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
│   ├── schema.py              # Core ORM models (Symbol, DailyBar, Trade, etc.)
│   └── schema_experiments.py  # Experiment tracking (Experiment, ExperimentRun)
├── analytics/      # Anomaly detection, regime analysis, LLM agents
├── domain/         # Core business models
├── patron/         # Multi-analyst decision engine
├── daemon/         # Production daemon and paper trader
├── eval/           # Backtesting and performance evaluation
│   ├── backtest.py            # Core backtest engine
│   ├── performance_report.py  # Metrics calculation (CAGR, Sharpe, etc.)
│   └── significance.py        # Statistical comparison of experiments
├── research/       # Research infrastructure
│   └── backtest_runner.py     # Unified backtest API
├── strategy/       # Strategy configuration
│   └── config.py              # StrategyConfig loader
└── broker/         # Broker abstraction (Binance, etc.)

strategies/         # YAML strategy definitions
experiments/        # Experiment templates (param sweeps, ablations)
scripts/            # Research and backtest runners
├── run_grid_search.py              # Grid search automation
└── promote_experiment_to_strategy.py  # Promote best runs to new versions
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
# Run baseline_v1 strategy backtest using unified API
python -m otonom_trader.cli backtest run \
  --strategy strategies/baseline_v1.0.yaml \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --output reports/baseline_v1_backtest.json

# View backtest report
python -m otonom_trader.cli backtest report \
  --file reports/baseline_v1_backtest.json
```

### 4. Run Parameter Optimization Experiments

```bash
# Parameter sweep experiment (486 combinations)
python scripts/run_grid_search.py \
  --config experiments/param_sweep_baseline.yaml

# Analyst ablation study (16 combinations)
python scripts/run_grid_search.py \
  --config experiments/ablation_analysts.yaml

# View experiment results (SQL)
sqlite3 data/trader.db "SELECT run_index, test_sharpe, test_cagr FROM experiment_runs WHERE experiment_id = 1 ORDER BY test_sharpe DESC LIMIT 10"
```

### 5. Promote Best Experiment to New Strategy Version

```bash
# Auto-select best run and promote to v1.1
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0

# View promotion history
cat STRATEGY_LOG.md
```

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

## Research Workflow

### Experiment-Driven Strategy Optimization

Otonom Trader provides a complete research infrastructure for systematic strategy optimization:

```
1. Design Experiment
   ↓
2. Run Grid Search
   ↓
3. Analyze Results
   ↓
4. Promote Best Run
   ↓
5. Deploy to Production
```

### 1. Design Experiment

Create experiment config in `experiments/`:

```yaml
# experiments/my_experiment.yaml
name: "baseline_v1_param_sweep"
description: "Optimize risk and analyst weights"

base_strategy: "strategies/baseline_v1.0.yaml"

search_method: "grid"  # or "random"
random_samples: 100    # if random

split:
  train_start: "2018-01-01"
  train_end: "2021-12-31"
  test_start: "2022-01-01"
  test_end: "2024-12-31"

parameters:
  risk.risk_pct:
    values: [0.5, 1.0, 1.5]
  risk.stop_loss_pct:
    values: [3.0, 5.0, 8.0]
  ensemble.analyst_weights.tech:
    values: [0.8, 1.0, 1.2]
```

### 2. Run Grid Search

```bash
python scripts/run_grid_search.py --config experiments/my_experiment.yaml
```

Results saved to `experiments` and `experiment_runs` database tables.

### 3. Analyze Results

```python
# Python analysis
from otonom_trader.data import get_session
from otonom_trader.data.schema_experiments import Experiment
from otonom_trader.eval.significance import compare_top_two

with get_session() as session:
    exp = session.query(Experiment).filter_by(name="baseline_v1_param_sweep").first()

    # Compare top 2 runs
    result = compare_top_two(exp.runs, n_obs=252*3)

    if result.is_significant:
        print(f"✓ Run #{result.run_a.run_index} significantly better (z={result.z_score:.2f})")
    else:
        print(f"✗ No significant difference")
```

### 4. Promote Best Run

```bash
# Promote to v1.1
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0
```

This will:
- Select best run by test Sharpe
- Apply parameter overrides to base strategy
- Save new strategy YAML
- Log promotion to `STRATEGY_LOG.md`

### 5. Deploy to Production

```bash
# Backtest new version
python -m otonom_trader.cli backtest run \
  --strategy strategies/baseline_v1.1.yaml \
  --start 2018-01-01 --end 2024-12-31

# Deploy to paper daemon
otonom-trader daemon-once --strategy baseline_v1.1
```

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

### Phase 2: Research Infrastructure ✅
- [x] Strategy YAML configuration
- [x] Backtest runner with performance metrics
- [x] Experiment tracking database
- [x] Grid search for parameter optimization
- [x] Statistical significance testing
- [x] Promotion workflow (v1 → v1.1/v2.0)

### Phase 3: Production ✅
- [x] Broker abstraction layer (Binance + extensible)
- [x] Shadow mode (PAPER/SHADOW/LIVE execution modes)
- [x] Kill-switch and guardrails (daily loss, max DD, consecutive losses)
- [x] Alert engine (email, Telegram)
- [x] Packaging and distribution (pyproject.toml)

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
