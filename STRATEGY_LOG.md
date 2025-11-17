# Strategy Promotion Log

This document tracks all strategy versions and promotions.

## Promotion Criteria

A strategy must meet these criteria to be promoted:

### Absolute Requirements
- Test Sharpe ≥ 1.2
- Test Max Drawdown ≤ -30%
- Test Trades ≥ 50

### Improvement Requirements (vs previous version)
- Test Sharpe improvement ≥ 5%
- Max DD degradation ≤ 5%

### Regime Requirements
- Must not blow up in crisis periods (Sharpe > -1.0)
- Should work across multiple regimes

## Versioning Scheme

**Major version** (X.0): Behavioral changes
- New analyst added/removed
- Risk model changed
- Fundamental logic change

**Minor version** (X.Y): Parameter tuning
- Risk % adjusted
- Weights tuned
- Thresholds optimized

## Promotion Workflow

1. **Run Experiment**: Grid search, ablation, or robustness test
2. **Select Best**: Pick top 1-3 runs based on Sharpe + DD + robustness
3. **Promote**: Generate new strategy YAML with updated version
4. **Document**: Add entry to this log with changes and rationale
5. **Validate**: Run full backtest on train+test, check regression
6. **Paper Trade**: Deploy to paper daemon (champion/challenger)

---

## Promotion History

(Promotions will be automatically added below)

