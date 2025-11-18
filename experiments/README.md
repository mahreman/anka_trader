# Experiment Templates

This directory contains standardized experiment templates for systematic strategy research using the new simplified StrategyConfig format.

## Available Templates

### 1. Parameter Sweep (`param_sweep_baseline.yaml`)

**Goal**: Find optimal hyperparameter tuning for baseline strategy

**What it tests**:
- Risk per trade: 0.5%, 1.0%, 1.5%
- Stop-loss: 3%, 5%, 8%
- Take-profit: 6%, 10%, 15%
- DSI threshold: 0.4, 0.5, 0.6
- Analyst weights: tech (0.8-1.2), news (0.5-1.0), macro (0.5-1.0)

**Total combinations**: 486 (or use random sampling with 100)

**Usage**:
```bash
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_param_sweep \
  --strategy-path strategies/baseline_v1.0.yaml \
  --grid-path experiments/param_sweep_baseline.yaml \
  --train-start 2018-01-01 --train-end 2021-12-31 \
  --test-start 2022-01-01 --test-end 2024-12-31 \
  --description "Baseline v1 parameter sweep"
```

**Expected insights**:
- Optimal risk level (conservative vs aggressive)
- Best stop-loss/take-profit ratio
- DSI threshold impact on trade quality
- Analyst weight sensitivity
- Parameter robustness

---

### 2. Ablation Study (`ablation_analysts.yaml`)

**Goal**: Measure individual analyst contribution by systematically removing them

**What it tests**:
- All 16 combinations of (tech, news, macro, rl) ON/OFF
- Single analysts: tech only, news only, macro only, rl only
- Pairs: tech+news, tech+macro, news+macro, etc.
- Full ensemble: all analysts enabled

**Total combinations**: 16 runs (fast!)

**Usage**:
```bash
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_ablation \
  --strategy-path strategies/baseline_v1.0.yaml \
  --grid-path experiments/ablation_analysts.yaml \
  --train-start 2018-01-01 --train-end 2021-12-31 \
  --test-start 2022-01-01 --test-end 2024-12-31 \
  --description "Analyst ablation study"
```

**Expected insights**:
- Which analyst is critical vs redundant?
- Can any single analyst beat the ensemble?
- Best 2-analyst combination
- Marginal contribution of each analyst
- Synergy between analyst pairs

**Analysis workflow**:
1. Run experiment
2. Compare test Sharpe across configurations
3. Calculate performance degradation when removing each analyst:
   - Baseline (all ON): Sharpe = 1.5
   - Remove tech: Sharpe = 1.2 → tech contributes +0.3
   - Remove news: Sharpe = 1.4 → news contributes +0.1
   - Remove macro: Sharpe = 1.3 → macro contributes +0.2

4. Identify optimal analyst subset

---

## New vs Legacy Format

These templates use the **NEW simplified StrategyConfig format**:

### NEW Format (✓ Use this)
```yaml
parameters:
  risk.risk_pct: [0.5, 1.0, 1.5]
  risk.stop_loss_pct: [3.0, 5.0]
  filters.dsi_threshold: [0.4, 0.5]
  ensemble.analyst_weights.tech: [0.8, 1.0]
  ensemble.analyst_weights.news: [1.0, 1.2]
  ensemble.analyst_weights.macro: [0.8, 1.0]
  ensemble.analyst_weights.rl: [0.0, 1.0]
```

### Legacy Format (✗ Old, but still supported)
```yaml
parameters:
  risk_management.position_sizing.risk_per_trade_pct: [0.5, 1.0]
  risk_management.stop_loss.percentage: [3.0, 5.0]
  analist_1.weight: [0.8, 1.0]  # tech
  analist_2.weight: [1.0, 1.2]  # news
  analist_3.weight: [0.8, 1.0]  # macro
```

---

## Configuration Schema

All experiment templates follow this structure:

```yaml
name: "experiment_name"
description: "What this experiment does"
experiment_type: "param_sweep" | "ablation" | "robustness" | "regime"

base_strategy: "strategies/baseline_v1.0.yaml"

search_method: "grid" | "random"
random_samples: 100  # If search_method == "random"

split:
  train_start: "2018-01-01"
  train_end: "2021-12-31"
  test_start: "2022-01-01"
  test_end: "2024-12-31"

universe_override: []  # Empty = use strategy default

parameters:
  param.path.key:
    values: [val1, val2, val3]
    description: "What this parameter does"

optimization:
  primary_metric: "test_sharpe"
  direction: "maximize"
  secondary_metrics: [...]

constraints:
  test_sharpe_min: 0.8
  test_max_dd_max: -40.0
  test_min_trades: 20

execution:
  parallel_workers: 1
  save_equity_curves: false
  save_trade_lists: true
  verbose: true
```

---

## Workflow

1. **Design experiment** → Choose template and customize parameters
2. **Run experiment** → Use CLI command
3. **Monitor progress** → Check logs and database
4. **Analyze results** → Use experiment analysis CLI
5. **Select best configs** → Pick top 3-5 by test Sharpe
6. **Promote to new version** → Use strategy promotion workflow

---

## Tips

### Parameter Sweep
- Start with small grid (3-5 values per param) to test
- Use random search for large grids (>100 combinations)
- Focus on most impactful parameters first
- Check for overfitting: train_sharpe / test_sharpe > 1.5

### Ablation Study
- Always include "all ON" baseline for comparison
- Test single analysts to find standalone performance
- Test pairs to find synergistic combinations
- Calculate marginal contribution: Sharpe(A+B) - Sharpe(A)

### Performance
- Use `parallel_workers: 4` for faster execution (if you have 4 CPUs)
- Set `save_equity_curves: false` for large experiments (saves space)
- Use `random_samples: 50-100` for initial exploration

---

## See Also

- [Strategy Configuration](../docs/strategy_config.md) - StrategyConfig contract
- [Experiment Analysis](../docs/experiment_analysis.md) - How to analyze results
- [Strategy Promotion](../docs/strategy_promotion.md) - Promoting best configs
