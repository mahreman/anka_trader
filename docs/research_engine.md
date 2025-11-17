# Research Engine Documentation

The Research Engine transforms the backtest runner into a comprehensive research tool that supports different backtest types and generates detailed reports.

## Overview

The Research Engine provides:

1. **Multiple Backtest Types**
   - Smoke Test: Quick validation (30 days, 1-2 symbols)
   - Full Historical: Complete backtest with train/val/test splits
   - Scenario: Specific market period testing

2. **Comprehensive Reports**
   - Performance metrics (CAGR, Sharpe, Sortino, Max DD, Win Rate, etc.)
   - Equity curve with daily values
   - Detailed trade list with R-multiples
   - Regime breakdown (performance by market regime)

3. **Easy CLI Interface**
   - Single command to generate reports
   - Flexible configuration
   - Multiple output formats (JSON, CSV)

## Usage

### Smoke Test

Quick sanity check to verify strategy works:

```bash
# Run smoke test with default config
otonom-trader research smoke baseline_v1

# Run smoke test with specific symbols
otonom-trader research smoke baseline_v1 --symbols BTC-USD,ETH-USD
```

**Smoke Test Configuration:**
- Duration: Last 30 days
- Symbols: First 1-2 from config (or specified)
- Purpose: Verify strategy doesn't crash, basic functionality works

### Full Historical Backtest

Complete backtest with train/validation/test splits:

```bash
# Run full backtest with default dates from config
otonom-trader research full baseline_v1

# Run full backtest with custom period
otonom-trader research full baseline_v1 \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --symbols BTC-USD,ETH-USD

# Save reports to custom directory
otonom-trader research full baseline_v1 --output reports/my_backtest
```

**Full Backtest Configuration:**
- Splits: 60% train / 20% validation / 20% test
- Symbols: From config or specified
- Purpose: Evaluate strategy performance, check for overfitting

**Output:**
- Three reports (Train, Validation, Test)
- Overfitting analysis (Train/Test Sharpe ratio)
- JSON + CSV files for each period

### Scenario Backtest

Test strategy on specific market conditions:

```bash
# Run predefined scenario
otonom-trader research scenario baseline_v1 --scenario covid_crash

# Run custom scenario period
otonom-trader research scenario baseline_v1 \
  --start 2020-03-01 \
  --end 2020-04-30

# Run all predefined scenarios
otonom-trader research scenario baseline_v1

# List available scenarios
otonom-trader research list-scenarios
```

**Available Scenarios:**
- `covid_crash`: February-April 2020 market crash
- `crypto_bear_2022`: 2022 crypto bear market
- `high_inflation`: 2021-2023 high inflation period
- `bull_run_2021`: Q4 2020 - Q4 2021 bull market

## Report Format

### Performance Metrics

Each report includes:

```yaml
metrics:
  cagr: 25.3              # Compound Annual Growth Rate (%)
  sharpe_ratio: 1.8       # Risk-adjusted returns
  sortino_ratio: 2.1      # Downside risk-adjusted returns
  max_drawdown: -15.2     # Maximum drawdown (%)
  win_rate: 58.5          # Percentage of winning trades
  profit_factor: 2.3      # Gross profit / Gross loss
  total_trades: 142       # Total number of trades
  avg_win_loss_ratio: 1.5 # Average win / Average loss
```

### Trade List

Each trade record includes:

```yaml
- symbol: BTC-USD
  entry_date: 2023-01-15
  exit_date: 2023-01-20
  direction: BUY
  entry_price: 21000.0
  exit_price: 22500.0
  quantity: 0.5
  pnl: 750.0
  pnl_pct: 7.14
  holding_days: 5
  r_multiple: 3.57        # Reward-to-risk ratio
  regime_id: 1            # Market regime at entry
```

### Regime Breakdown

Performance broken down by market regime:

```
Regime 0: 45 trades, 62.2% WR, 2.15R avg, $12,450 PnL
Regime 1: 67 trades, 55.2% WR, 1.82R avg, $18,230 PnL
Regime 2: 30 trades, 50.0% WR, 1.23R avg, $4,820 PnL
```

### Equity Curve

Daily portfolio values for visualization:

```csv
date,equity
2023-01-01,100000.0
2023-01-02,101250.0
2023-01-03,100800.0
...
```

## Programmatic Usage

You can also use the Research Engine programmatically:

```python
from otonom_trader.data import get_session
from otonom_trader.eval.research_engine import (
    BacktestType,
    BacktestPeriod,
    run_research_backtest,
)

# Run smoke test
with get_session() as session:
    reports = run_research_backtest(
        session=session,
        strategy_path="strategies/baseline_v1.yaml",
        backtest_type=BacktestType.SMOKE,
    )

    for report in reports:
        print(report.summary())

        # Access metrics
        print(f"CAGR: {report.metrics['cagr']:.2f}%")
        print(f"Sharpe: {report.metrics['sharpe_ratio']:.2f}")

        # Access trades
        for trade in report.trades:
            print(f"{trade.symbol}: {trade.pnl:.2f} ({trade.r_multiple:.2f}R)")

# Run full backtest
with get_session() as session:
    reports = run_research_backtest(
        session=session,
        strategy_path="strategies/baseline_v1.yaml",
        backtest_type=BacktestType.FULL,
        symbols=["BTC-USD", "ETH-USD"],
    )

    # reports[0] = Train
    # reports[1] = Validation
    # reports[2] = Test

    train_sharpe = reports[0].metrics["sharpe_ratio"]
    test_sharpe = reports[2].metrics["sharpe_ratio"]

    print(f"Overfitting ratio: {train_sharpe / test_sharpe:.2f}")

# Run custom scenario
from datetime import date

with get_session() as session:
    custom_period = BacktestPeriod(
        name="My Custom Period",
        start_date=date(2020, 3, 1),
        end_date=date(2020, 4, 30),
        description="March-April 2020"
    )

    reports = run_research_backtest(
        session=session,
        strategy_path="strategies/baseline_v1.yaml",
        backtest_type=BacktestType.SCENARIO,
        custom_period=custom_period,
    )
```

## Output Files

Reports are saved to the output directory (default: `reports/`):

```
reports/
├── smoke_smoke_test_20240115_143022.json          # Smoke test report
├── smoke_smoke_test_20240115_143022_trades.csv    # Trade list
├── smoke_smoke_test_20240115_143022_equity.csv    # Equity curve
├── full_train_20240115_143500.json                # Full backtest - train
├── full_train_20240115_143500_trades.csv
├── full_train_20240115_143500_equity.csv
├── full_validation_20240115_143500.json           # Full backtest - validation
├── full_validation_20240115_143500_trades.csv
├── full_validation_20240115_143500_equity.csv
├── full_test_20240115_143500.json                 # Full backtest - test
├── full_test_20240115_143500_trades.csv
├── full_test_20240115_143500_equity.csv
└── scenario_covid_crash_20240115_144000.json      # Scenario backtest
```

## R-Multiple Calculation

R-multiple measures how much you made (or lost) relative to your initial risk:

```
R-multiple = Actual PnL / Initial Risk
```

Where:
- **Initial Risk** = Stop loss distance × position size
- For this implementation, we use the strategy's `risk_per_trade_pct` as initial risk

Example:
- Entry: $100, Exit: $105, Risk: 2% → PnL = 5%, R-multiple = 5% / 2% = 2.5R
- Entry: $100, Exit: $98, Risk: 2% → PnL = -2%, R-multiple = -2% / 2% = -1.0R

**Good R-multiples:**
- Average R > 1.0: Strategy makes more than it risks
- Average R > 2.0: Excellent risk/reward
- Average R < 0.5: Strategy needs improvement

## Workflow

Typical research workflow:

1. **Smoke Test** - Verify strategy works
   ```bash
   otonom-trader research smoke my_strategy
   ```

2. **Full Backtest** - Evaluate performance
   ```bash
   otonom-trader research full my_strategy --output reports/my_strategy
   ```

3. **Check Results** - Review metrics and overfitting
   - Look at Train/Test Sharpe ratio
   - Aim for ratio < 1.5 (good generalization)
   - Check regime breakdown for robustness

4. **Scenario Testing** - Test edge cases
   ```bash
   otonom-trader research scenario my_strategy --scenario covid_crash
   otonom-trader research scenario my_strategy --scenario crypto_bear_2022
   ```

5. **Iterate** - Modify strategy based on results

## Advanced Usage

### Custom Backtest Period

```python
from datetime import date
from otonom_trader.eval.research_engine import BacktestPeriod

# Define custom period
period = BacktestPeriod(
    name="Bull Market 2017",
    start_date=date(2017, 1, 1),
    end_date=date(2017, 12, 31),
    description="Bitcoin bull run 2017"
)

# Run backtest with custom period
reports = run_research_backtest(
    session=session,
    strategy_path="strategies/my_strategy.yaml",
    backtest_type=BacktestType.SCENARIO,
    custom_period=period,
)
```

### Analyze Regime Performance

```python
# Get regime breakdown
for regime_id, perf in report.regime_breakdown.items():
    print(f"Regime {regime_id}:")
    print(f"  Trades: {perf.total_trades}")
    print(f"  Win Rate: {perf.win_rate:.1f}%")
    print(f"  Avg R: {perf.avg_r_multiple:.2f}")
    print(f"  Total PnL: ${perf.total_pnl:,.2f}")
```

### Export to JSON

```python
import json

# Convert report to dict
report_dict = report.to_dict()

# Save to file
with open("report.json", "w") as f:
    json.dump(report_dict, f, indent=2)
```

## Best Practices

1. **Always run smoke test first** - Catch bugs early
2. **Use full backtest for evaluation** - Get proper train/test splits
3. **Monitor overfitting** - Train/Test Sharpe ratio should be < 1.5
4. **Test scenarios** - Ensure strategy works in different market conditions
5. **Check regime breakdown** - Strategy should work across regimes
6. **Review R-multiples** - Average R > 1.0 is minimum, > 2.0 is excellent
7. **Analyze trade list** - Look for patterns in winners/losers

## Troubleshooting

### No results generated

- Check that you have data for the symbols and date range
- Run data ingestion: `otonom-trader ingest-all`
- Verify strategy YAML is valid

### All trades losing money

- Review strategy parameters
- Check anomaly detection thresholds
- Run scenario tests to see which periods work/don't work

### High overfitting (Train/Test ratio > 2.0)

- Strategy is likely curve-fitted to training data
- Simplify strategy parameters
- Use longer test period
- Consider ensemble approach

### No regime data

- Regime breakdown requires P1 regime detection
- Run regime analysis first
- Regime data may not be available for all symbols/dates

## See Also

- [Strategy Configuration](strategy_config.md)
- [Backtest Engine](backtest_engine.md)
- [Performance Metrics](performance_metrics.md)
- [Experiment Engine](experiment_engine.md)
