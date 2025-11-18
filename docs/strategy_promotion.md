# Strategy Promotion Workflow Documentation

Systematic strategy versioning and promotion ritual.

## Overview

The strategy promotion system provides a **6-step ritual** for evolving strategies in a controlled, documented manner:

1. **Run Experiment** - Grid search, ablation, robustness
2. **Select Best** - Top 1-3 runs by Sharpe + DD + robustness
3. **Promote** - Generate new versioned strategy YAML
4. **Document** - Log changes and rationale in STRATEGY_LOG.md
5. **Validate** - Full backtest on train+test, regression check
6. **Paper Trade** - Champion/challenger parallel testing

---

## Versioning Scheme

### Semantic Versioning: `major.minor`

**Major version** (X.0): Behavioral changes
- New analyst added/removed
- Risk model changed
- Fundamental logic change
- Example: `baseline_v1.0` → `baseline_v2.0`

**Minor version** (X.Y): Parameter tuning
- Risk % adjusted
- Weights tuned
- Thresholds optimized
- Example: `baseline_v1.0` → `baseline_v1.1`

### File Naming Convention

```
strategies/
├── baseline_v1.0.yaml    # Initial working version
├── baseline_v1.1.yaml    # Parameter tuning from grid search
├── baseline_v1.2.yaml    # Further optimization
└── baseline_v2.0.yaml    # RL analyst added (major change)
```

---

## Promotion Criteria

A strategy must meet these criteria to be promoted:

### Absolute Requirements

```python
min_test_sharpe = 1.2           # Minimum Sharpe on test set
max_test_max_dd = -30.0         # Max acceptable drawdown
min_test_trades = 50            # Minimum trades for significance
```

### Improvement Requirements (vs previous version)

```python
min_sharpe_improvement = 0.05   # 5% improvement minimum
max_dd_degradation = 5.0        # Max 5% DD degradation
```

### Regime Requirements

- Must not blow up in crisis periods (Sharpe > -1.0)
- Should work across multiple regimes
- Robustness score ≥ 70% (for robustness experiments)

### Overfitting Check

- Train/Test Sharpe ratio < 1.5 (preferably < 1.3)
- Similar performance across train/val/test splits

---

## 6-Step Promotion Ritual

### Step 1: Run Experiment

Run relevant experiment template:

```bash
# Parameter sweep
otonom-trader experiments grid-search \
  --name "baseline_opt" \
  --grid-path grids/param_sweep_baseline.yaml

# Ablation study
otonom-trader experiments grid-search \
  --name "analyst_ablation" \
  --grid-path grids/ablation_analysts.yaml

# Robustness test
otonom-trader experiments grid-search \
  --name "robustness" \
  --grid-path grids/robustness_risk.yaml
```

### Step 2: Select Best Runs

Extract promotion candidates:

```bash
# Show top 5 candidates
otonom-trader strategy candidates 5 --top 5

# Output:
Rank  Run  Test Sharpe  Test CAGR  Max DD   Overfit
1     12   1.85         28.0%      -14.5%   1.12     ← Best
2     7    1.80         26.5%      -15.0%   1.15
3     15   1.75         25.0%      -16.0%   1.20
```

**Selection criteria**:
- Highest test Sharpe
- Acceptable max DD (< -30%)
- Low overfitting (ratio < 1.5)
- Sufficient trades (> 50)

### Step 3: Promote

Generate new strategy version:

```bash
# Auto-promote best run
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml

# Promote specific run
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --run 12

# Force major version
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --type major

# With custom description
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml \
  --changes "Increased risk from 1.0% to 1.5%, tightened stop-loss" \
  --rationale "Grid search showed 15% Sharpe improvement with acceptable DD"
```

**Output**:
```
✓ Promotion complete!
New version: baseline v1.1
Type: Minor
Performance: Sharpe=1.85, CAGR=28.0%, MaxDD=-14.5%

Next steps:
  1. Run full backtest: otonom-trader research full baseline_v1.1
  2. Compare with previous: otonom-trader strategy compare v1.0 v1.1
  3. Deploy to paper daemon for champion/challenger test
```

### Step 4: Document

Promotion is automatically documented in `STRATEGY_LOG.md`:

```markdown
## v1.0 → v1.1 (Minor) - 2024-01-15

**Experiment**: #5
**Performance**: Sharpe=1.85, CAGR=28.0%, MaxDD=-14.5%

**Changes**:
• risk_management.position_sizing.risk_per_trade_pct: 1.0 → 1.5
• risk_management.stop_loss.percentage: 5.0 → 4.5
• analist_1.weight: 1.0 → 1.2

**Rationale**: Grid search experiment showed 15% Sharpe improvement
with tighter risk controls. Overfitting ratio 1.12 (acceptable).
Test performance validated on 145 trades with consistent regime performance.

---
```

### Step 5: Validate

Run comprehensive validation:

```bash
# Full backtest (train + val + test)
otonom-trader research full baseline_v1.1 --output reports/v1.1/

# Check for regression on different symbols
otonom-trader research full baseline_v1.1 \
  --symbols BTC-USD,ETH-USD,SOL-USD

# Scenario testing
otonom-trader research scenario baseline_v1.1 --scenario covid_crash
otonom-trader research scenario baseline_v1.1 --scenario crypto_bear_2022
```

**Validation checklist**:
- [ ] Test Sharpe ≥ 1.2
- [ ] Max DD ≤ -30%
- [ ] Train/Test ratio < 1.5
- [ ] Works across multiple symbols
- [ ] Survives crisis scenarios
- [ ] No unexpected behavior

### Step 6: Paper Trade (Champion/Challenger)

Deploy to paper daemon for parallel testing:

```bash
# Start paper daemon with both versions
otonom-trader paper start --champion baseline_v1.0 --challenger baseline_v1.1

# Monitor performance
otonom-trader paper status

# Compare live results after 30 days
otonom-trader strategy compare baseline_v1.0 baseline_v1.1

# Replace champion if challenger wins
otonom-trader paper promote-challenger
```

**Champion/Challenger criteria**:
- Run both in parallel for ≥ 30 days
- Compare on same data
- Replace if challenger:
  - Sharpe improvement > 5%
  - DD improvement or degradation ≤ 2%
  - Consistent outperformance

---

## CLI Commands

### Initialize Strategy Log

```bash
# Create STRATEGY_LOG.md template
otonom-trader strategy init-log

# Custom output path
otonom-trader strategy init-log --output docs/STRATEGY_LOG.md
```

### Find Latest Version

```bash
# Show latest version of strategy
otonom-trader strategy latest baseline

# Output:
Latest version: baseline v1.2
  File: baseline_v1.2.yaml
```

### Show Promotion Candidates

```bash
# Show top 5 candidates from experiment
otonom-trader strategy candidates 5 --top 5

# Show top 10
otonom-trader strategy candidates 5 --top 10
```

### Promote Strategy

```bash
# Auto-promote best run
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml

# Promote specific run
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --run 12

# Force promotion type
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --type major

# With custom metadata
otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml \
  --changes "Added RL analyst for entry timing" \
  --rationale "Ablation showed 20% Sharpe lift"
```

### Compare Strategies

```bash
# Compare champion vs challenger
otonom-trader strategy compare baseline_v1.0 baseline_v1.1

# Output:
Metric          Champion  Challenger  Improvement
Sharpe Ratio    1.50      1.65        +10.0%
CAGR            25.0%     27.5%       +10.0%
Max Drawdown    -15.0%    -14.0%      +1.0% better

Recommendation: REPLACE - Challenger beats champion on all metrics
```

### Run Full Workflow

```bash
# Manual selection
otonom-trader strategy workflow 5 strategies/baseline_v1.0.yaml

# Auto-select best
otonom-trader strategy workflow 5 strategies/baseline_v1.0.yaml --auto
```

---

## Examples

### Example 1: Parameter Tuning (Minor)

```bash
# 1. Run parameter sweep
otonom-trader experiments grid-search \
  --name "baseline_params" \
  --grid-path grids/param_sweep_baseline.yaml

# 2. Show candidates
otonom-trader strategy candidates 1 --top 3

# 3. Promote best
otonom-trader strategy promote 1 strategies/baseline_v1.0.yaml

# 4. Validate
otonom-trader research full baseline_v1.1

# 5. Paper trade
# (manual deployment to paper daemon)
```

**Result**: `baseline_v1.0.yaml` → `baseline_v1.1.yaml`

---

### Example 2: Adding RL Analyst (Major)

```bash
# 1. Run ablation to validate RL contribution
otonom-trader experiments grid-search \
  --name "rl_ablation" \
  --grid-path grids/ablation_analysts.yaml

# 2. Analyze ablation
otonom-trader experiments ablation 2

# Output shows RL adds +0.25 Sharpe

# 3. Promote with major version bump
otonom-trader strategy promote 2 strategies/baseline_v1.2.yaml --type major

# 4. Document in STRATEGY_LOG.md (auto)

# 5. Full validation
otonom-trader research full baseline_v2.0
otonom-trader research scenario baseline_v2.0 --scenario covid_crash

# 6. Champion/challenger test
# (30 days paper trading)
```

**Result**: `baseline_v1.2.yaml` → `baseline_v2.0.yaml`

---

### Example 3: Robustness Validation

```bash
# 1. Run robustness test
otonom-trader experiments grid-search \
  --name "robustness_v1.1" \
  --grid-path grids/robustness_risk.yaml

# 2. Analyze robustness
otonom-trader experiments robustness 3 \
  --baseline-risk 1.5 \
  --baseline-sl 4.5

# Output:
Parameter    Sensitivity  Status
risk_pct     0.20         ✓ Robust
stop_loss    0.15         ✓ Robust
take_profit  0.25         ✓ Robust

# All robust! Safe to promote.

# 3. Promote
otonom-trader strategy promote 3 strategies/baseline_v1.1.yaml
```

---

## Best Practices

### 1. Always Run Experiments First

Never promote without validation:
- ❌ Manual parameter changes without testing
- ✅ Grid search → select best → promote

### 2. Document Everything

Every promotion should have:
- Clear rationale (why?)
- Performance metrics (how much better?)
- Changes description (what changed?)
- Validation results (does it work?)

### 3. Test Across Regimes

Before promoting, validate in:
- Crisis periods (covid, crypto crash)
- Bull markets
- Sideways/low-vol periods

### 4. Monitor Overfitting

Red flags:
- Train/Test Sharpe ratio > 2.0
- Works only on specific symbols
- Breaks in new scenarios

### 5. Champion/Challenger Testing

Never replace production immediately:
- Run both in paper for ≥ 30 days
- Compare on same data
- Look for consistent outperformance

### 6. Version Control Integration

```bash
# After promotion
git add strategies/baseline_v1.1.yaml STRATEGY_LOG.md
git commit -m "feat(Strategy): Promote baseline v1.0 → v1.1

Experiment #5 showed 15% Sharpe improvement with tighter risk controls.

Changes:
- risk_per_trade: 1.0% → 1.5%
- stop_loss: 5.0% → 4.5%

Performance: Sharpe=1.85, CAGR=28%, MaxDD=-14.5%"

git push
```

---

## Troubleshooting

### No candidates pass criteria

**Problem**: All candidates fail promotion criteria

**Solutions**:
1. Lower criteria temporarily (min_sharpe = 1.0 instead of 1.2)
2. Review experiment setup (bad parameter grid?)
3. Strategy might be fundamentally flawed
4. Try different experiment type (ablation vs grid)

### High overfitting ratio

**Problem**: Train Sharpe >> Test Sharpe

**Solutions**:
1. Simplify strategy (fewer parameters)
2. Longer test period
3. Cross-validation
4. Ensemble approach

### DD degradation

**Problem**: New version has worse drawdown

**Solutions**:
1. Check if Sharpe improvement worth it
2. Adjust max_dd_degradation threshold
3. Investigate regime-specific issues
4. Consider ensemble with old version

### Inconsistent regime performance

**Problem**: Works in bull, fails in bear

**Solutions**:
1. Add regime-aware logic
2. Train on mixed regimes
3. Separate strategies per regime
4. Dynamic allocation

---

## See Also

- [Experiment Templates](experiment_templates.md)
- [Research Engine](research_engine.md)
- [Strategy Configuration](strategy_config.md)
- [STRATEGY_LOG.md](../STRATEGY_LOG.md)
