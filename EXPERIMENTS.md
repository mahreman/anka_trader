# Experiment System Guide

This guide explains how to use the experiment system to optimize trading strategies through systematic parameter search.

## Overview

The experiment system allows you to:
1. **Grid Search**: Test many parameter combinations systematically
2. **Track Results**: Store all runs in database with metrics
3. **Promote Best**: Extract best parameters → new strategy version
4. **Compare**: Baseline v1 → v2 → v3 evolution

## Workflow

```
┌─────────────────┐
│ 1. Grid Search  │  Test parameter combinations
│   (CLI)         │  Store results in DB
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 2. Promote Best │  Extract best run parameters
│   (Script)      │  Create baseline_v2.yaml
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 3. Backtest v2  │  Validate new strategy
│   (CLI)         │  Compare to v1
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 4. Deploy       │  Use v2 in daemon/live
│                 │  Or iterate with new grid
└─────────────────┘
```

## 1. Grid Search

### Setup

**Grid File**: `grids/baseline_grid.yaml`

```yaml
name: "baseline_grid_v1"
search_method: "grid"  # or "random"

# Train/test split
split:
  train_start: "2017-01-01"
  train_end: "2022-12-31"
  test_start: "2023-01-01"
  test_end: "2025-01-17"

# Parameters to optimize
parameters:
  risk_management.position_sizing.risk_per_trade_pct:
    values: [0.5, 1.0, 2.0]

  risk_management.stop_loss.percentage:
    values: [3.0, 5.0, 8.0]

  risk_management.take_profit.percentage:
    values: [8.0, 10.0, 15.0]

  analist_1.weight:
    values: [0.8, 1.0, 1.2]

  # ... more parameters

# Optimization objective
optimization:
  primary_metric: "test_sharpe"
  direction: "maximize"
```

### Run Grid Search

```bash
# Basic grid search
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_grid \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/baseline_grid.yaml

# With custom description
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_grid_crypto \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/baseline_grid.yaml \
  --description "Baseline v1 param sweep, crypto universe"

# Random search (faster, samples N combinations)
# Edit grids/baseline_grid.yaml: search_method: "random"
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_random \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/baseline_grid.yaml
```

**What happens:**
1. Loads base strategy (`baseline_v1.yaml`)
2. Generates all parameter combinations from grid
3. For each combination:
   - Creates a unique `ExperimentRun`
   - Runs backtest on train period
   - Runs backtest on test period
   - Stores metrics (Sharpe, CAGR, max DD, win rate)
4. Saves all results to database

**Example output:**
```
Experiment: baseline_v1_grid (ID: 1)
Parameter combinations: 243 (3×3×3×3×...)

Run 1/243: Testing params={risk_pct: 0.5, stop_loss: 3.0, ...}
  Train: Sharpe=1.23, CAGR=18.5%, MaxDD=-12.3%
  Test:  Sharpe=0.98, CAGR=15.2%, MaxDD=-15.8%
  ✓ PASSED constraints

Run 2/243: Testing params={risk_pct: 0.5, stop_loss: 5.0, ...}
  ...

================================================================================
EXPERIMENT COMPLETE
================================================================================
Total runs: 243
Successful: 241
Failed: 2
Best run (by test_sharpe): Run #87
  Train Sharpe: 1.45, CAGR: 22.3%
  Test Sharpe: 1.12, CAGR: 18.7%
  Params: {risk_pct: 1.0, stop_loss: 5.0, take_profit: 10.0, ...}
```

### View Experiment Results

```bash
# List all experiments
python -m otonom_trader.cli experiments list

# Show experiment details
python -m otonom_trader.cli experiments show --experiment-id 1

# Show top N runs
python -m otonom_trader.cli experiments show --experiment-id 1 --top 10

# Filter by constraints
python -m otonom_trader.cli experiments show \
  --experiment-id 1 \
  --min-sharpe 0.8 \
  --max-dd -20.0
```

## 2. Promote Best Run to baseline_v2

Once grid search completes, promote the best run to a new strategy:

```bash
# Promote best run (by default: test Sharpe + CAGR)
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v2.yaml

# Rank by CAGR only
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v2.yaml \
  --rank-by cagr

# Rank by Sharpe only
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v2.yaml \
  --rank-by sharpe
```

**Output:**
```
================================================================================
PROMOTING EXPERIMENT TO STRATEGY
================================================================================
Experiment: baseline_v1_grid (ID: 1)
Base Strategy: baseline_v1
Description: Baseline v1 param sweep, crypto universe
================================================================================

Found 241 completed runs for experiment 1
✅ Best run selected:
   Run ID: 87
   Run Index: 87
   Status: done

📊 Training Metrics:
   Sharpe: 1.450
   CAGR: 22.30%
   Max DD: -18.20%
   Win Rate: 58.30%
   Total Trades: 127

📊 Test Metrics:
   Sharpe: 1.120
   CAGR: 18.70%
   Max DD: -21.50%
   Win Rate: 55.20%
   Total Trades: 89

🔧 Parameter Values:
   risk_management.position_sizing.risk_per_trade_pct: 1.0
   risk_management.stop_loss.percentage: 5.0
   risk_management.take_profit.percentage: 10.0
   analist_1.weight: 1.0
   analist_2.weight: 1.2
   analist_3.weight: 0.8
   analist_1.anomaly_detection.zscore_threshold: 2.0
   portfolio_constraints.turnover_limits.max_daily_trades: 10
   portfolio_constraints.cooldown.hours_after_flip: 24

📄 Loading base strategy: strategies/baseline_v1.yaml
⚙️  Applying parameter overrides...

================================================================================
✅ SUCCESS: New strategy created!
================================================================================
Output: strategies/baseline_v2.yaml
Strategy Name: baseline_v2

Next steps:
1. Review the strategy: cat strategies/baseline_v2.yaml
2. Backtest it: python -m otonom_trader.cli backtest --strategy strategies/baseline_v2.yaml
3. Compare to baseline: Compare metrics in backtest reports
================================================================================
```

**Result**: `strategies/baseline_v2.yaml` created with optimized parameters.

## 3. Validate baseline_v2

Run backtests to validate the new strategy:

```bash
# Backtest v2 on same period as grid search
python -m otonom_trader.cli backtest \
  --strategy strategies/baseline_v2.yaml \
  --start 2023-01-01 --end 2025-01-17 \
  --output reports/baseline_v2_test.html

# Backtest v1 for comparison
python -m otonom_trader.cli backtest \
  --strategy strategies/baseline_v1.yaml \
  --start 2023-01-01 --end 2025-01-17 \
  --output reports/baseline_v1_test.html

# Full period backtest (train + test)
python -m otonom_trader.cli backtest \
  --strategy strategies/baseline_v2.yaml \
  --start 2017-01-01 --end 2025-01-17 \
  --output reports/baseline_v2_full.html
```

**Compare:**
| Metric | baseline_v1 | baseline_v2 | Improvement |
|--------|-------------|-------------|-------------|
| Sharpe | 0.92 | 1.12 | +21.7% |
| CAGR | 14.5% | 18.7% | +29.0% |
| Max DD | -25.3% | -21.5% | +15.0% |
| Win Rate | 52.1% | 55.2% | +5.9% |

## 4. Deploy baseline_v2

### Use in Daemon

```bash
# Start daemon with v2
python -m otonom_trader.cli daemon start \
  --strategy strategies/baseline_v2.yaml \
  --mode paper

# Or edit daemon config
# config/daemon.yaml:
#   strategy: strategies/baseline_v2.yaml
```

### Make v2 the Default

```bash
# Option 1: Overwrite v1 (backup first!)
cp strategies/baseline_v1.yaml strategies/baseline_v1_original.yaml
cp strategies/baseline_v2.yaml strategies/baseline_v1.yaml

# Option 2: Update references
# Update all scripts/configs to use baseline_v2.yaml
```

## Advanced Usage

### Custom Grid Search

Create your own grid file:

```yaml
# grids/my_custom_grid.yaml
name: "custom_grid"
search_method: "random"  # Random search
random_samples: 100  # Sample 100 combinations

split:
  train_start: "2018-01-01"
  train_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: "2025-01-17"

parameters:
  # Only optimize risk parameters
  risk_management.position_sizing.risk_per_trade_pct:
    values: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

  risk_management.stop_loss.percentage:
    range: [2.0, 10.0, 0.5]  # min, max, step

  risk_management.take_profit.percentage:
    range: [5.0, 20.0, 1.0]

optimization:
  primary_metric: "test_sharpe"
  direction: "maximize"

constraints:
  test_sharpe_min: 0.7
  test_max_dd_max: -30.0
```

### Multi-Stage Optimization

```bash
# Stage 1: Coarse grid (fast)
python -m otonom_trader.cli experiments grid-search \
  --experiment-name stage1_coarse \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/coarse_grid.yaml

# Promote stage 1
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 1 \
  --output-path strategies/baseline_v1.5.yaml

# Stage 2: Fine grid around best params
# Edit grids/fine_grid.yaml to search near stage 1 optimum
python -m otonom_trader.cli experiments grid-search \
  --experiment-name stage2_fine \
  --strategy-path strategies/baseline_v1.5.yaml \
  --grid-path grids/fine_grid.yaml

# Promote stage 2 to v2
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 2 \
  --output-path strategies/baseline_v2.yaml
```

### Walk-Forward Optimization

```bash
# Period 1: Train 2018-2020, Test 2021
# Edit grid: train_start="2018-01-01", train_end="2020-12-31"
#            test_start="2021-01-01", test_end="2021-12-31"
python -m otonom_trader.cli experiments grid-search \
  --experiment-name wf_period1 \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/wf_grid_p1.yaml

# Period 2: Train 2019-2021, Test 2022
# Similar for periods 2, 3, 4...

# Analyze stability: Check if same parameters win across periods
```

## Troubleshooting

### Grid search runs forever
- **Cause**: Too many combinations (e.g., 10^6 combinations)
- **Solution**: Use random search instead of exhaustive grid
  ```yaml
  search_method: "random"
  random_samples: 100
  ```

### All runs fail constraints
- **Cause**: Constraints too strict
- **Solution**: Relax constraints in `grids/baseline_grid.yaml`:
  ```yaml
  constraints:
    test_sharpe_min: 0.3  # Instead of 0.5
    test_max_dd_max: -50.0  # Instead of -40.0
  ```

### Overfitting detected
- **Symptom**: Train Sharpe >> Test Sharpe (e.g., 2.5 vs 0.8)
- **Solution**:
  1. Use longer test period
  2. Add regularization (simpler strategies)
  3. Walk-forward validation

### No improvement over baseline
- **Cause**: Grid doesn't explore right parameter space
- **Solution**: Expand grid range or try different parameters
  ```yaml
  parameters:
    # Add new parameters
    analist_1.lookback_days:
      values: [20, 30, 50, 100]
  ```

## Database Schema

Experiments are stored in two tables:

**experiments** table:
- id, name, description
- base_strategy_name
- param_grid_path
- created_at

**experiment_runs** table:
- id, experiment_id, run_index
- param_values_json (JSON string)
- train/test metrics (Sharpe, CAGR, max DD, win rate)
- status (pending/running/done/failed)

Query example:
```python
from otonom_trader.data import get_session
from otonom_trader.data.schema_experiments import Experiment, ExperimentRun

with next(get_session()) as session:
    # Get best run
    best_run = (
        session.query(ExperimentRun)
        .filter_by(experiment_id=1, status="done")
        .order_by(ExperimentRun.test_sharpe.desc())
        .first()
    )
    print(best_run.param_values_json)
```

## Next Steps

1. **Run first grid search**: Start with `baseline_grid.yaml`
2. **Promote to v2**: Use promote script
3. **Validate**: Backtest v2 vs v1
4. **Iterate**: Create v3, v4 with refined grids
5. **Deploy**: Use best version in daemon

See also:
- `grids/baseline_grid.yaml` - Example grid configuration
- `scripts/promote_experiment_to_strategy.py` - Promotion script
- Experiment CLI: `python -m otonom_trader.cli experiments --help`
