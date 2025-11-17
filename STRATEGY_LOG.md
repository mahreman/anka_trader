# Strategy Evolution Log

Track of all strategy versions and their evolution over time.

## Purpose

This log documents:
- Strategy version changes
- Parameter modifications
- Experiment results that motivated changes
- Expected vs. actual performance impact

## Template

When creating a new version, include:
- **Date**: When version was created
- **Parent**: Previous version
- **Experiment**: Experiment ID that motivated change (if any)
- **Reason**: Why this change was made
- **Parameter Changes**: List of changed parameters
- **Expected Impact**: Predicted performance change
- **Actual Impact**: Measured performance change (update after deployment)

---

## baseline_v1

**Date**: 2025-01-17 (initial version)
**Parent**: None (baseline)
**Reason**: Initial strategy implementation

### Parameters

- `risk.risk_pct`: 1.0
- `risk.stop_loss_pct`: 5.0
- `risk.take_profit_pct`: 10.0
- `filters.dsi_threshold`: 0.5
- `filters.min_regime_vol`: 0.01
- `ensemble.analyst_weights.tech`: 1.0
- `ensemble.analyst_weights.news`: 1.0
- `ensemble.analyst_weights.risk`: 1.0

### Performance (2023-2025 test period)

- Test Sharpe: TBD
- Test CAGR: TBD
- Test MaxDD: TBD

---

## Example Future Entry

## baseline_v2

**Date**: 2025-02-01
**Parent**: baseline_v1
**Experiment**: #42
**Reason**: Grid search showed higher news weight improves Sharpe ratio by 0.3

### Parameter Changes

- `risk.risk_pct`: 1.0 → 1.5
- `ensemble.analyst_weights.news`: 1.0 → 1.2

**Expected Impact**: +10% CAGR, +0.3 Sharpe
**Actual Impact**: (Update after 30-day live test)

---
