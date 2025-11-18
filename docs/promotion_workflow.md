# Strategy Promotion Workflow

Complete guide to the 6-step promotion ritual for evolving trading strategies.

## Overview

The promotion workflow ensures systematic strategy evolution:

1. **Run Experiment** → Test parameter variations
2. **Select Best** → Choose top configurations
3. **Promote** → Create new strategy version
4. **Document** → Log changes and rationale
5. **Validate** → Run full backtest
6. **Deploy** → Paper trade (champion/challenger)

## Step 1: Run Experiment

### Option A: Parameter Sweep

Test multiple parameter combinations to find optimal tuning:

```bash
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_param_sweep \
  --strategy-path strategies/baseline_v1.0.yaml \
  --grid-path experiments/param_sweep_baseline.yaml \
  --train-start 2018-01-01 --train-end 2021-12-31 \
  --test-start 2022-01-01 --test-end 2024-12-31 \
  --description "Baseline v1 parameter sweep"
```

**What it tests**:
- Risk per trade: 0.5%, 1.0%, 1.5%
- Stop-loss: 3%, 5%, 8%
- Take-profit: 6%, 10%, 15%
- DSI threshold: 0.4, 0.5, 0.6
- Analyst weights: various combinations

**Total runs**: ~486 (or use random sampling)

### Option B: Ablation Study

Test analyst contributions by systematically turning them on/off:

```bash
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_ablation \
  --strategy-path strategies/baseline_v1.0.yaml \
  --grid-path experiments/ablation_analysts.yaml \
  --train-start 2018-01-01 --train-end 2021-12-31 \
  --test-start 2022-01-01 --test-end 2024-12-31 \
  --description "Analyst ablation study"
```

**What it tests**:
- All 16 combinations of (tech, news, macro, rl) ON/OFF
- Single analysts: tech only, news only, etc.
- Pairs: tech+news, tech+macro, etc.
- Full ensemble: all analysts enabled

**Total runs**: 16 (fast!)

### Monitor Experiment

Check progress:
```bash
# View experiment status
python -m otonom_trader.cli experiments list

# View specific experiment
python -m otonom_trader.cli experiments show --id 3

# View top runs
python -m otonom_trader.cli experiments top --id 3 --metric test_sharpe
```

---

## Step 2: Select Best Configuration

### View Candidates

```bash
# Show top 5 candidates by test Sharpe
python -m otonom_trader.cli strategy candidates \
  --experiment-id 3 \
  --top-n 5
```

**Output**:
```
Top 5 Promotion Candidates
==========================

Run #42: Sharpe=1.52, CAGR=28.3%, MaxDD=-18.2%, Overfit=1.35
  risk.risk_pct: 1.2
  risk.stop_loss_pct: 4.0
  ensemble.analyst_weights.tech: 1.1

Run #127: Sharpe=1.48, CAGR=26.1%, MaxDD=-19.5%, Overfit=1.42
  risk.risk_pct: 1.0
  risk.stop_loss_pct: 5.0
  ensemble.analyst_weights.news: 0.8

...
```

### Selection Criteria

**Primary**:
- Test Sharpe ratio (higher is better)
- Test CAGR (higher is better)
- Test Max Drawdown (less negative is better)

**Secondary**:
- Overfitting ratio (train/test Sharpe < 1.5)
- Win rate (higher is better)
- Total trades (need at least 50 for statistical significance)
- Robustness (small param changes → small performance changes)

---

## Step 3: Promote to New Version

### Automatic Promotion

Use the promotion script to automatically:
1. Select best run
2. Apply parameter overrides
3. Bump version
4. Save new strategy YAML
5. Document in STRATEGY_LOG.md

```bash
# Auto-select best run and promote to v1.1
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0
```

**Output**:
```
Experiment: baseline_v1_param_sweep
Base strategy: strategies/baseline_v1.0.yaml

Selected run: #42
  Test Sharpe: 1.52
  Test CAGR: 28.30%
  Test Max DD: -18.20%
  Test Win Rate: 62.5%
  Test Trades: 127

Current version: baseline v1.0
Promotion type: minor
New version: baseline v1.1
Output path: strategies/baseline_v1.1.yaml

Changes:
• risk.risk_pct: 1.0 → 1.2
• risk.stop_loss_pct: 5.0 → 4.0
• ensemble.analyst_weights.tech: 1.0 → 1.1
• ensemble.analyst_weights.news: 1.2 → 0.8

✓ Saved new strategy: strategies/baseline_v1.1.yaml
✓ Updated strategy log: STRATEGY_LOG.md

PROMOTION COMPLETE!
```

### Manual Selection

Specify a specific run:
```bash
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --run-index 42 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0
```

### Auto-Versioning

Let the script detect whether it should be a major or minor bump:
```bash
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --auto-version
```

**Version bump logic**:
- **Minor** (1.0 → 1.1): Only parameter values changed
- **Major** (1.0 → 2.0): Structural changes (analysts added/removed, risk model changed)

### Dry Run

Preview what would happen without actually doing it:
```bash
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0 \
  --dry-run
```

---

## Step 4: Documentation (Automatic)

The promotion script automatically:
1. Compares old vs new configuration
2. Generates change summary
3. Appends to STRATEGY_LOG.md

**Example log entry**:
```markdown
## v1.0 → v1.1 (Minor) - 2025-11-17

**Experiment**: #3 (baseline_v1_param_sweep)
**Performance**: Sharpe=1.52, CAGR=28.3%, MaxDD=-18.2%

**Changes**:
• risk.risk_pct: 1.0 → 1.2
• risk.stop_loss_pct: 5.0 → 4.0
• ensemble.analyst_weights.tech: 1.0 → 1.1
• ensemble.analyst_weights.news: 1.2 → 0.8

**Rationale**: Promoted from experiment #3 (baseline_v1_param_sweep). 
Best run #42 achieved test Sharpe=1.52, CAGR=28.30%, MaxDD=-18.20%.
Grid search showed that slightly higher risk (1.2%) combined with tighter 
stop-loss (4%) improves risk-adjusted returns. Reducing news analyst weight 
helps prevent over-reaction to noise in crisis periods.

---
```

---

## Step 5: Validate with Full Backtest

Run a comprehensive backtest on the new strategy:

```bash
# Full historical backtest
python -m otonom_trader.cli backtest run \
  --strategy strategies/baseline_v1.1.yaml \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --output reports/baseline_v1.1_full_backtest.json
```

**Check for**:
1. **No regression**: v1.1 should not be worse than v1.0
2. **Consistency**: Performance should be similar to experiment results
3. **Robustness**: Test across different market regimes
4. **Trade count**: Sufficient trades for statistical significance

### Compare Versions

```bash
# Compare v1.0 vs v1.1
python -m otonom_trader.cli strategy compare \
  --champion reports/baseline_v1.0_full_backtest.json \
  --challenger reports/baseline_v1.1_full_backtest.json
```

**Output**:
```
Champion vs Challenger Comparison
==================================

Metric          Champion (v1.0)  Challenger (v1.1)  Change
---------------------------------------------------------------
Sharpe          1.32             1.52              +15.2% ✓
CAGR            23.5%            28.3%             +20.4% ✓
Max DD          -22.1%           -18.2%            +17.6% ✓
Win Rate        58.3%            62.5%             +7.2%  ✓
Total Trades    142              127               -10.6%

Recommendation: REPLACE - Challenger beats champion on all metrics
```

---

## Step 6: Deploy to Paper Daemon

### Champion/Challenger Setup

Run both v1.0 (champion) and v1.1 (challenger) in parallel:

**Terminal 1 - Champion (v1.0)**:
```bash
python -m otonom_trader.cli daemon-loop \
  --strategy strategies/baseline_v1.0.yaml \
  --initial-cash 100000 \
  --risk-pct 1.0 \
  --interval 900
```

**Terminal 2 - Challenger (v1.1)**:
```bash
python -m otonom_trader.cli daemon-loop \
  --strategy strategies/baseline_v1.1.yaml \
  --initial-cash 100000 \
  --risk-pct 1.0 \
  --interval 900
```

### Monitor Performance

```bash
# View paper trades
python -m otonom_trader.cli show-paper-trades --limit 50

# View portfolio snapshots
python -m otonom_trader.cli daemon-status --limit 20

# Check system status
python -m otonom_trader.cli status
```

### Decision Criteria

After 30-60 days of paper trading:

**Replace champion if**:
- Challenger Sharpe > Champion Sharpe + 0.1
- Challenger Max DD < Champion Max DD (less negative)
- No unexpected behavior or edge cases

**Keep champion if**:
- Challenger underperforms
- Challenger shows instability
- Edge cases discovered

**Continue testing if**:
- Results are inconclusive
- Need more data
- Market conditions changed

---

## Complete Example: v1.0 → v1.1

### 1. Run Experiment
```bash
python -m otonom_trader.cli experiments grid-search \
  --experiment-name baseline_v1_param_sweep \
  --strategy-path strategies/baseline_v1.0.yaml \
  --grid-path experiments/param_sweep_baseline.yaml \
  --train-start 2018-01-01 --train-end 2021-12-31 \
  --test-start 2022-01-01 --test-end 2024-12-31
```

### 2. View Candidates
```bash
python -m otonom_trader.cli strategy candidates --experiment-id 3 --top-n 5
```

### 3. Promote
```bash
python scripts/promote_experiment_to_strategy.py \
  --experiment-id 3 \
  --output-path strategies/baseline_v1.1.yaml \
  --new-version 1.1.0
```

### 4. Validate
```bash
python -m otonom_trader.cli backtest run \
  --strategy strategies/baseline_v1.1.yaml \
  --start 2018-01-01 --end 2024-12-31 \
  --output reports/baseline_v1.1_full.json
```

### 5. Compare
```bash
python -m otonom_trader.cli strategy compare \
  --champion reports/baseline_v1.0_full.json \
  --challenger reports/baseline_v1.1_full.json
```

### 6. Deploy
```bash
# Terminal 1: Champion
python -m otonom_trader.cli daemon-loop \
  --strategy strategies/baseline_v1.0.yaml \
  --interval 900

# Terminal 2: Challenger
python -m otonom_trader.cli daemon-loop \
  --strategy strategies/baseline_v1.1.yaml \
  --interval 900
```

---

## Tips and Best Practices

### Experiment Design
- Start with small grids (3-5 values per param)
- Use random search for large grids (>100 combinations)
- Focus on most impactful parameters first
- Always include baseline config as one of the runs

### Selection
- Don't just pick highest Sharpe - check for overfitting
- Look at multiple metrics: Sharpe, DD, Win Rate, Trade Count
- Prefer robust configurations over fragile optimizations
- Check performance across different market regimes

### Promotion
- Document WHY you're promoting (not just WHAT changed)
- Include experiment insights in rationale
- Always validate on full historical data
- Never skip the validation step

### Paper Trading
- Run champion/challenger for at least 30 days
- Monitor daily for unexpected behavior
- Check edge cases (extreme moves, data gaps, etc.)
- Document any issues or surprises

### Rollback Plan
- Always keep previous versions
- Document how to revert if needed
- Monitor closely for first 7 days after deployment
- Have alerting for performance degradation

---

## See Also

- [Strategy Configuration](strategy_config.md) - Config format and validation
- [Experiment Templates](../experiments/README.md) - Available experiment types
- [Strategy Versioning](../otonom_trader/otonom_trader/strategy/versioning.py) - Version management
- [Promotion Functions](../otonom_trader/otonom_trader/strategy/promotion.py) - Promotion API
