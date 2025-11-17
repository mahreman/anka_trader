# Experiment to Production Workflow

Complete workflow for running experiments and promoting results to production.

## Overview

The experiment workflow follows these phases:

1. **Run Grid Search**: Test many parameter combinations
2. **Analyze Results**: Find best-performing parameters
3. **Promote to Strategy**: Create new strategy version with best parameters
4. **Validate**: Backtest and validate new strategy
5. **Deploy**: Use new strategy in production

## Step-by-Step Workflow

### Phase 1: Run Grid Search Experiment

```bash
# 1. Define your grid search (grids/baseline_grid.yaml)
# See experiments/baseline_v1_grid.yaml for example

# 2. Run grid search experiment
python -m otonom_trader.cli experiments grid-search \
  --name baseline_v1_grid \
  --strategy-path strategies/baseline_v1.yaml \
  --grid-path grids/baseline_grid.yaml \
  --train-start 2018-01-01 \
  --train-end 2022-12-31 \
  --test-start 2023-01-01 \
  --test-end 2024-12-31

# This will:
# - Create Experiment record in database (e.g., id=3)
# - Run all parameter combinations
# - Store results in ExperimentRun table
# - Show progress with rich console output
```

**Example output:**
```
✓ Experiment 'baseline_v1_grid' created (ID: 3)
Running grid search: 160000 total combinations

Progress: ████████████████████ 100%
Completed: 160000/160000 runs
Time: 2h 15m 30s

Top 5 results (by test Sharpe):
  Run #42: Sharpe=2.15, CAGR=45.2%, MaxDD=-12.3%
  Run #127: Sharpe=2.10, CAGR=43.8%, MaxDD=-13.1%
  ...
```

### Phase 2: Analyze Experiment Results

```bash
# View experiment summary
python -m otonom_trader.cli experiments show 3

# Export detailed HTML report
python -m otonom_trader.cli experiments show 3 --export html

# This creates:
# - reports/experiment_3_report.html (interactive report)
# - Shows top runs, parameter distributions, overfitting warnings
```

**Example report sections:**
- **Summary Statistics**: Total runs, success rate, time taken
- **Top 10 Results**: Best performers sorted by test Sharpe
- **Parameter Analysis**: Which parameters had most impact
- **Overfitting Detection**: Runs with train/test Sharpe gap > 1.0
- **Distribution Plots**: Parameter vs. performance correlations

### Phase 3: Promote Best Run to Strategy

**Option A: High-Level (Recommended)**

Uses `promote_experiment_to_strategy()` which handles everything:

```bash
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --output-name baseline_v2 \
  --base-strategy strategies/baseline_v1.yaml \
  --expected-impact "+15% CAGR, +0.3 Sharpe on test set"
```

**What it does:**
1. Finds best run (by test Sharpe)
2. Extracts parameter values
3. Creates `strategies/baseline_v2.yaml`
4. Adds metadata (parent strategy, experiment ID, performance metrics)
5. Updates `STRATEGY_LOG.md` with change history

**Example output:**
```
✅ Successfully promoted experiment 3
📄 New strategy: strategies/baseline_v2.yaml
📝 Updated log: STRATEGY_LOG.md

Best run metrics:
  - Test Sharpe: 2.15
  - Test CAGR: 45.2%
  - Test MaxDD: -12.3%

Parameter changes:
  - risk.risk_pct: 1.0 → 1.5
  - ensemble.analyst_weights.news: 1.0 → 1.2
  - risk.stop_loss_pct: 5.0 → 8.0
```

**Option B: Low-Level (More Control)**

Uses simple script for manual control:

```bash
python scripts/promote_experiment_simple.py \
  --experiment-id 3 \
  --output-path strategies/baseline_v2.yaml
```

This gives you more control and transparency over the process.

### Phase 4: Validate New Strategy

**4.1 Review Generated Strategy**

```bash
# Review new strategy file
cat strategies/baseline_v2.yaml

# Check what changed
diff strategies/baseline_v1.yaml strategies/baseline_v2.yaml
```

**4.2 Backtest New Strategy**

```bash
# Backtest on same test period used in experiment
python -m otonom_trader.cli backtest \
  --strategy-path strategies/baseline_v2.yaml \
  --symbol BTC-USD \
  --start-date 2023-01-01 \
  --end-date 2024-12-31

# Verify results match experiment predictions
```

**4.3 Compare with Baseline**

```bash
# TODO: Implement comparison CLI command
python -m otonom_trader.cli compare \
  --baseline strategies/baseline_v1.yaml \
  --candidate strategies/baseline_v2.yaml \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

**Expected validation results:**
- Test Sharpe should match experiment result (~2.15)
- CAGR should match (~45.2%)
- MaxDD should match (~-12.3%)

**If results differ significantly:**
- Check data quality (DSI threshold)
- Verify date ranges match
- Check for overfitting (train/test Sharpe gap)

### Phase 5: Deploy to Production

**5.1 Update Daemon Configuration**

```python
# In daemon configuration
strategy_path = "strategies/baseline_v2.yaml"  # Updated from baseline_v1
```

**5.2 Run Shadow Mode Test**

```bash
# Run daemon in shadow mode for 7-30 days
# Monitor performance vs. baseline_v1

python examples/daemon_with_broker_example.py loop 3600
```

**5.3 Monitor Performance**

Track key metrics:
- **Sharpe Ratio**: Should stay close to test results
- **Drawdown**: Should not exceed test MaxDD significantly
- **Win Rate**: Should match test win rate
- **Slippage**: Monitor for execution issues

**5.4 Gradual Rollout**

1. **Day 1-7**: Shadow mode (paper + broker logging)
2. **Day 8-14**: Live mode with 10% capital allocation
3. **Day 15-30**: Live mode with 50% capital allocation
4. **Day 31+**: Full capital allocation if metrics hold

### Phase 6: Track Strategy Evolution

```bash
# View strategy evolution history
cat STRATEGY_LOG.md
```

**Example STRATEGY_LOG.md entry:**

```markdown
## baseline_v2

**Date**: 2025-01-15 14:23:45 UTC
**Parent**: baseline_v1
**Experiment**: #3
**Reason**: Best run from experiment #3: Test Sharpe=2.15, CAGR=45.2%, MaxDD=-12.3%

### Parameter Changes

- `risk.risk_pct`: 1.5
- `ensemble.analyst_weights.news`: 1.2
- `risk.stop_loss_pct`: 8.0

**Expected Impact**: +15% CAGR, +0.3 Sharpe on test set

**Actual Impact**: (Update after 30-day test)

---
```

## Advanced Workflows

### Workflow 1: Multi-Objective Optimization

Find different strategies optimizing for different objectives:

```bash
# 1. Run grid search
python -m otonom_trader.cli experiments grid-search --name exp1 ...

# 2. Promote multiple strategies
# Best Sharpe
python scripts/promote_experiment_to_strategy.py --experiment-id 1 \
  --output-name high_sharpe_v1 --select-by sharpe

# Best CAGR
python scripts/promote_experiment_to_strategy.py --experiment-id 1 \
  --output-name high_return_v1 --select-by cagr

# Best Sortino (TODO: add Sortino to metrics)
python scripts/promote_experiment_to_strategy.py --experiment-id 1 \
  --output-name low_downside_v1 --select-by sortino
```

### Workflow 2: Ensemble of Strategies

Run multiple strategies in parallel:

```python
# In daemon
strategies = [
    load_strategy("strategies/high_sharpe_v1.yaml"),
    load_strategy("strategies/high_return_v1.yaml"),
    load_strategy("strategies/low_downside_v1.yaml"),
]

# Allocate capital: 40% / 30% / 30%
allocations = [0.4, 0.3, 0.3]
```

### Workflow 3: Iterative Improvement

Build a chain of improving strategies:

```bash
# baseline_v1 → baseline_v2
python scripts/promote_experiment_to_strategy.py --experiment-id 1 \
  --output-name baseline_v2 --base-strategy strategies/baseline_v1.yaml

# baseline_v2 → baseline_v3 (using v2 as new base)
python -m otonom_trader.cli experiments grid-search \
  --strategy-path strategies/baseline_v2.yaml \
  --grid-path grids/refinement_grid.yaml  # Smaller grid around v2 params

python scripts/promote_experiment_to_strategy.py --experiment-id 2 \
  --output-name baseline_v3 --base-strategy strategies/baseline_v2.yaml

# Continue iteration...
```

## Best Practices

### 1. Experiment Design

**Good Grid Search:**
```yaml
# Focus on impactful parameters
params:
  risk.risk_pct: [0.5, 1.0, 1.5, 2.0]          # 4 values
  risk.stop_loss_pct: [3.0, 5.0, 8.0, 10.0]    # 4 values
  ensemble.analyst_weights.news: [0.5, 1.0, 1.5] # 3 values

# Total: 4 × 4 × 3 = 48 combinations (manageable)
```

**Bad Grid Search:**
```yaml
# Too many parameters, too many values
params:
  risk.risk_pct: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]              # 6 values
  risk.stop_loss_pct: [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]      # 6 values
  risk.take_profit_pct: [10.0, 15.0, 20.0, 25.0, 30.0]       # 5 values
  ensemble.analyst_weights.tech: [0.5, 1.0, 1.5, 2.0]        # 4 values
  ensemble.analyst_weights.news: [0.5, 1.0, 1.5, 2.0]        # 4 values

# Total: 6 × 6 × 5 × 4 × 4 = 2,880 combinations (too many!)
```

**Solution for large grids:**
- Use random search instead of grid search
- Start with coarse grid, then refine around best regions

### 2. Train/Test Split

**Good Split:**
```yaml
split:
  train_start: "2017-01-01"
  train_end: "2022-12-31"   # 6 years training
  test_start: "2023-01-01"
  test_end: "2024-12-31"     # 2 years testing
```

**Ratio:** 75% train / 25% test is standard

**Walk-Forward (Advanced):**
```bash
# Year 1-4: Train on 2017-2020, test on 2021
# Year 2-5: Train on 2018-2021, test on 2022
# Year 3-6: Train on 2019-2022, test on 2023
# Etc...
```

### 3. Overfitting Detection

**Warning Signs:**
- Train Sharpe >> Test Sharpe (gap > 1.0)
- Many parameters with tiny improvements
- Perfect training results, poor testing results

**Solutions:**
- Reduce parameter count
- Use regularization (e.g., penalty for complexity)
- Increase train/test split ratio
- Use walk-forward validation

### 4. Parameter Selection

**Start with:**
- Risk parameters (risk_pct, stop_loss, take_profit)
- Ensemble weights (analyst_weights)
- Entry/exit thresholds

**Avoid tuning:**
- Data quality thresholds (dsi_threshold)
- Lookback windows (too sensitive)
- Many correlated parameters at once

## Troubleshooting

### Issue: No Successful Runs

**Symptom:**
```
❌ No 'done' runs for experiment 3
```

**Solutions:**
- Check experiment status: `python -m otonom_trader.cli experiments show 3`
- Look for error messages in runs
- Verify strategy config is valid
- Check data availability for date range

### Issue: All Runs Have Similar Performance

**Symptom:**
```
Top 10 runs all have Sharpe between 1.48-1.52
```

**Solutions:**
- Parameters may not have much impact
- Try different parameters
- Increase parameter ranges
- Check if strategy is too constrained

### Issue: Best Run Shows Overfitting

**Symptom:**
```
Best run: Train Sharpe=3.5, Test Sharpe=1.2 (gap=2.3)
```

**Solutions:**
- Don't promote this run
- Look for run with lower train/test gap
- Simplify strategy
- Add regularization

### Issue: Promoted Strategy Underperforms

**Symptom:**
```
Experiment predicted: Sharpe=2.1
Live trading results: Sharpe=1.5
```

**Solutions:**
- Check for data drift (market regime changed)
- Verify execution slippage is reasonable
- Check if DSI threshold is filtering too much
- Consider shorter retraining cycle

## Summary

**Complete Workflow:**

```bash
# 1. Run experiment
python -m otonom_trader.cli experiments grid-search \\
  --name baseline_v1_grid \\
  --strategy-path strategies/baseline_v1.yaml \\
  --grid-path grids/baseline_grid.yaml

# 2. View results
python -m otonom_trader.cli experiments show 3 --export html

# 3. Promote to strategy
python scripts/promote_experiment_to_strategy.py \\
  --experiment-id 3 \\
  --output-name baseline_v2 \\
  --base-strategy strategies/baseline_v1.yaml

# 4. Validate
python -m otonom_trader.cli backtest \\
  --strategy-path strategies/baseline_v2.yaml \\
  --start-date 2023-01-01

# 5. Deploy
# Update daemon config to use baseline_v2.yaml
```

**Key Metrics to Track:**
- ✅ Test Sharpe Ratio (primary)
- ✅ Test CAGR (secondary)
- ✅ Test Max Drawdown
- ✅ Win Rate
- ✅ Train/Test Sharpe Gap (overfitting indicator)

Ready for systematic strategy improvement! 🚀
