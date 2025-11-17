# Experiment Templates Documentation

Systematic experiment design for strategy optimization and analysis.

## Overview

The experiment system provides **4 types of standardized experiments**:

1. **Parameter Sweep** - Find optimal configuration
2. **Ablation** - Measure component contribution
3. **Robustness** - Test parameter sensitivity
4. **Regime-Specific** - Evaluate performance by market condition

Each template is a YAML file that defines the parameter grid, optimization objective, and analysis method.

## Experiment Types

### 1. Parameter Sweep

**Goal**: "Is there a better tuning of the same strategy?"

**File**: `grids/param_sweep_baseline.yaml`

**What it tests**:
- Risk per trade (0.5% → 2.0%)
- Stop-loss and take-profit levels
- Analyst weights
- Anomaly detection thresholds
- Portfolio constraints

**Usage**:
```bash
# Run parameter sweep
otonom-trader experiments grid-search \
  --name "baseline_param_sweep" \
  --grid-path grids/param_sweep_baseline.yaml \
  --train-start 2018-01-01 --train-end 2022-12-31

# Show results
otonom-trader experiments show 1 --top 10

# Export report
otonom-trader experiments show 1 --export reports/param_sweep.html
```

**Typical Results**:
- Top 5 configurations with best Sharpe ratio
- Overfitting analysis (Train vs Test performance)
- Parameter sensitivity patterns

**Look for**:
- Best configuration (highest test Sharpe)
- Consistency (similar params → similar results?)
- Robustness (small change → big impact?)

---

### 2. Ablation Study

**Goal**: "Which analyst actually adds value?"

**File**: `grids/ablation_analysts.yaml`

**What it tests**:
- All analysts ON (baseline)
- Tech only
- Tech + News
- Tech + Risk
- News + Risk
- Individual analysts

**Usage**:
```bash
# Run ablation experiment
otonom-trader experiments grid-search \
  --name "analyst_ablation" \
  --grid-path grids/ablation_analysts.yaml

# Analyze ablation results
otonom-trader experiments ablation 2

# Export ablation report
otonom-trader experiments ablation 2 --export reports/ablation.md
```

**Example Output**:
```
Configuration    Sharpe  CAGR   Max DD  Contribution
All Analysts      1.50   25.0%  -12.0%  baseline
Tech + News       1.35   22.0%  -14.0%  -0.15
Tech + Risk       1.45   24.0%  -11.0%  -0.05
Tech only         1.20   18.0%  -16.0%  -0.30
News only         0.85   12.0%  -22.0%  -0.65
Risk only         0.60    8.0%  -28.0%  -0.90
```

**Insights**:
- **Contribution** = Performance vs baseline
  - Negative = Removing this component hurts performance
  - Large negative = Component is critical
- If removing X barely changes anything → X is redundant
- If any single analyst beats ensemble → Ensemble logic broken

---

### 3. Robustness/Sensitivity Analysis

**Goal**: "How fragile is this strategy?"

**File**: `grids/robustness_risk.yaml`

**What it tests**:
- Baseline configuration ± small perturbations
- Risk per trade: 1.0% ± 20-30%
- Stop-loss: 5.0% ± 20-30%
- Take-profit: 10.0% ± 20-30%

**Usage**:
```bash
# Run robustness experiment
otonom-trader experiments grid-search \
  --name "robustness_analysis" \
  --grid-path grids/robustness_risk.yaml

# Analyze robustness
otonom-trader experiments robustness 3 \
  --baseline-risk 1.0 \
  --baseline-sl 5.0 \
  --baseline-tp 10.0

# Export report
otonom-trader experiments robustness 3 --export reports/robustness.md
```

**Example Output**:
```
Parameter     Baseline  Sharpe Range     Std Dev  Sensitivity  Status
risk_pct      1.00      [1.20, 1.80]     0.18     0.40         ⚠ Fragile
stop_loss     5.00      [1.40, 1.60]     0.08     0.13         ✓ Robust
take_profit   10.00     [1.35, 1.65]     0.10     0.20         ✓ Robust
```

**Metrics**:
- **Sensitivity** = (max_sharpe - min_sharpe) / baseline_sharpe
  - < 0.3 = Robust ✓
  - > 0.5 = Fragile ⚠
- **Std Dev** = Standard deviation across perturbations
  - Low = Stable performance
  - High = Unpredictable

**Good Robustness**:
- Sharpe variance < 0.3
- ±20% param change → <20% performance change
- Smooth performance surface (no cliffs)

**Bad Robustness**:
- Sharpe variance > 0.5
- Small param change → huge performance swing
- Performance "cliff edges" (overfit)

---

### 4. Regime-Specific Analysis

**Goal**: "Where does the strategy work? Where does it fail?"

**File**: `grids/regime_focus_crisis.yaml`

**What it tests**:
- Covid Crash (Feb-Apr 2020)
- Crypto Bear 2022
- High Inflation (2021-2023)
- Bull Run 2021
- Sideways Markets (2019, 2023)

**Usage**:
```bash
# Run regime-specific experiment
otonom-trader experiments grid-search \
  --name "regime_analysis" \
  --grid-path grids/regime_focus_crisis.yaml

# Note: Use research engine for better regime analysis
otonom-trader research scenario baseline_v1 --scenario covid_crash
otonom-trader research scenario baseline_v1 --scenario crypto_bear_2022
```

**Example Output**:
```
Regime           Period           Sharpe  CAGR   MaxDD   Trades  Notes
Covid Crash      Feb-Apr'20       -0.50   -15%   -25%    12      Failed badly
Crypto Bear'22   2022             0.80    5%     -18%    45      OK
Bull Run 2021    Oct'20-Nov'21    2.10    45%    -12%    67      Excellent
Sideways 2019    2019             1.20    12%    -8%     34      Good
```

**Key Insights**:
1. **Regime breakdown**: Which regimes are profitable?
2. **Regime dependency**: Only works in bull markets?
3. **Crisis performance**: Does it blow up in crashes?
4. **Optimal allocation**: Run 24/7 or only in certain regimes?

---

## CLI Commands

### Run Experiments

```bash
# Parameter sweep (grid search)
otonom-trader experiments grid-search \
  --name "my_experiment" \
  --grid-path grids/param_sweep_baseline.yaml

# Random search (faster, samples N combinations)
otonom-trader experiments random-search \
  --name "my_random_experiment" \
  --grid-path grids/param_sweep_baseline.yaml
```

### Analyze Results

```bash
# List all experiments
otonom-trader experiments list --limit 20

# Show experiment details
otonom-trader experiments show 1 --top 10

# Ablation analysis
otonom-trader experiments ablation 2

# Robustness analysis
otonom-trader experiments robustness 3 --baseline-risk 1.0 --baseline-sl 5.0

# Compare multiple experiments
otonom-trader experiments compare 1,2,3 --metric test_sharpe
```

### Export Reports

```bash
# Export to HTML
otonom-trader experiments show 1 --export reports/experiment_1.html

# Export ablation to Markdown
otonom-trader experiments ablation 2 --export reports/ablation.md

# Export robustness to HTML
otonom-trader experiments robustness 3 --export reports/robustness.html
```

---

## YAML Template Structure

All experiment templates follow this structure:

```yaml
name: "experiment_name"
description: "What this experiment tests"
experiment_type: "param_sweep | ablation | robustness | regime_specific"

# Search method
search_method: "grid"  # or "random"
random_samples: 100    # For random search

# Train/test split
split:
  train_start: "2018-01-01"
  train_end: "2022-12-31"
  test_start: "2023-01-01"
  test_end: "2025-01-17"

# Symbols (empty = use strategy default)
symbols: []

# Parameter grid
parameters:
  risk_management.position_sizing.risk_per_trade_pct:
    values: [0.5, 1.0, 2.0]
    description: "Risk per trade %"

  risk_management.stop_loss.percentage:
    values: [3.0, 5.0, 8.0]
    description: "Stop-loss %"

# Optimization objective
optimization:
  primary_metric: "test_sharpe"  # test_sharpe, test_cagr, test_sortino
  direction: "maximize"

  secondary_metrics:
    - "test_cagr"
    - "test_max_dd"
    - "test_win_rate"

  overfitting_check:
    enabled: true
    train_test_ratio_threshold: 1.5

# Constraints (minimum acceptable values)
constraints:
  test_sharpe_min: 0.5
  test_max_dd_max: -40.0
  test_min_trades: 10

# Execution settings
execution:
  parallel_workers: 1
  save_equity_curves: false
  save_trade_lists: true
  verbose: true
```

---

## Workflow

Typical experiment workflow:

### 1. Parameter Sweep - Find Best Configuration

```bash
# Run param sweep
otonom-trader experiments grid-search \
  --name "baseline_optimization" \
  --grid-path grids/param_sweep_baseline.yaml

# Check results
otonom-trader experiments show 1 --top 5

# Export top configs
otonom-trader experiments show 1 --export reports/top_configs.html
```

**Look for**:
- Top 5 configurations
- Train/Test Sharpe ratio (< 1.5 = good)
- Consistent parameter patterns

### 2. Ablation - Validate Components

```bash
# Run ablation study
otonom-trader experiments grid-search \
  --name "analyst_contribution" \
  --grid-path grids/ablation_analysts.yaml

# Analyze contributions
otonom-trader experiments ablation 2 --export reports/ablation.md
```

**Look for**:
- Which analysts are critical?
- Which are redundant?
- Marginal contribution of each component

### 3. Robustness - Test Stability

```bash
# Run robustness test
otonom-trader experiments grid-search \
  --name "parameter_sensitivity" \
  --grid-path grids/robustness_risk.yaml

# Analyze sensitivity
otonom-trader experiments robustness 3 --export reports/robustness.md
```

**Look for**:
- Fragile parameters (sensitivity > 0.5)
- Robust parameters (sensitivity < 0.3)
- Performance variance

### 4. Regime Analysis - Understand Edge

```bash
# Test different market conditions
otonom-trader research scenario baseline_v1 --scenario covid_crash
otonom-trader research scenario baseline_v1 --scenario bull_run_2021
otonom-trader research scenario baseline_v1 --scenario sideways_2019
```

**Look for**:
- Which regimes are profitable?
- Crisis performance (max DD)
- When to turn strategy OFF

### 5. Compare & Decide

```bash
# Compare all experiments
otonom-trader experiments compare 1,2,3,4 --metric test_sharpe

# Pick best configuration
# Run final validation backtest
otonom-trader research full baseline_v1_optimized
```

---

## Interpreting Results

### Parameter Sweep

**Good Result**:
```
Run  Train Sharpe  Test Sharpe  Test CAGR  Overfitting Ratio
1    2.1           1.9          28%        1.10  ✓ Good
2    2.0           1.8          26%        1.11  ✓ Good
3    1.9           1.7          24%        1.12  ✓ Good
```
- Test Sharpe > 1.5
- Train/Test ratio < 1.3
- Consistent results across top configs

**Bad Result**:
```
Run  Train Sharpe  Test Sharpe  Test CAGR  Overfitting Ratio
1    3.5           1.2          18%        2.92  ⚠ Overfit
2    3.2           0.8          10%        4.00  ⚠ Overfit
3    2.9           0.5          5%         5.80  ⚠ Overfit
```
- Train/Test ratio > 2.0 = Overfit
- Test performance poor despite high train
- Need to simplify strategy

### Ablation

**Good Result**:
```
Configuration    Contribution
All Analysts     baseline
Tech + News      -0.10      (10% drop without risk analyst)
Tech + Risk      -0.05      (5% drop without news analyst)
News + Risk      -0.35      (35% drop without tech analyst)
```
- Each analyst contributes positively
- Ensemble > individual analysts
- Balanced contribution

**Bad Result**:
```
Configuration    Contribution
All Analysts     baseline
Tech only        +0.20      (Better without news/risk?!)
News only        -0.80      (News analyst hurts!)
```
- Removing components improves performance
- Suggests ensemble logic is broken
- Some analysts are harmful

### Robustness

**Good Result**:
```
Parameter     Sensitivity  Status
risk_pct      0.15         ✓ Robust
stop_loss     0.22         ✓ Robust
take_profit   0.18         ✓ Robust
```
- All parameters < 0.3 sensitivity
- Strategy is stable
- Safe to deploy

**Bad Result**:
```
Parameter     Sensitivity  Status
risk_pct      0.65         ⚠ Fragile
stop_loss     0.82         ⚠ Fragile
take_profit   0.45         ⚠ Fragile
```
- High sensitivity = overfit
- Small param changes → huge impact
- Need to simplify or use ensemble

### Regime Analysis

**Good Result**:
```
Regime           Sharpe
Covid Crash      0.5    (Survived, didn't blow up)
Crypto Bear'22   0.8    (Made money in bear)
Bull Run 2021    2.1    (Great in bull)
Sideways 2019    1.2    (Works in low vol)
```
- Positive across all regimes
- Robust strategy
- Safe to run 24/7

**Bad Result**:
```
Regime           Sharpe
Covid Crash      -2.5   (Blew up in crisis)
Crypto Bear'22   -0.5   (Lost money)
Bull Run 2021    3.5    (Only works in bull!)
Sideways 2019    -0.2   (Loses in sideways)
```
- Only works in one regime (bull)
- Not robust
- Turn OFF during crisis/bear/sideways

---

## Best Practices

1. **Always run all 4 experiment types**:
   - Param sweep → Find best config
   - Ablation → Validate components
   - Robustness → Test stability
   - Regime → Understand limits

2. **Start simple, add complexity**:
   - Start with small grid (fewer params)
   - If robust, expand grid
   - If fragile, simplify strategy

3. **Watch for overfitting**:
   - Train/Test Sharpe ratio < 1.5
   - Similar performance across splits
   - Robust to parameter changes

4. **Validate on multiple periods**:
   - Different market conditions
   - Multiple symbols
   - Out-of-sample testing

5. **Document findings**:
   - Export reports after each experiment
   - Compare results over time
   - Track what works/doesn't work

---

## See Also

- [Research Engine Documentation](research_engine.md)
- [Strategy Configuration](strategy_config.md)
- [Experiment Engine Architecture](experiment_engine.md)
