# Strategy Evolution Log

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

### Robustness Requirements
- Train/Test Sharpe ratio ≤ 1.5 (overfitting check)
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

### baseline v1.0 (Initial Version) - 2025-11-17

**Experiment**: N/A (Initial baseline)

**Performance**: 
- Test Sharpe: N/A
- Test CAGR: N/A
- Test MaxDD: N/A

**Configuration**:
- Risk per trade: 1.0%
- Stop-loss: 5.0%
- Take-profit: 10.0%
- DSI threshold: 0.5
- Tech weight: 1.0
- News weight: 1.2
- Macro weight: 0.8
- RL weight: 0.0

**Rationale**: Initial working version with multi-analyst ensemble (tech + news + macro)

---

<!-- New promotions will be added below by the promotion script -->
