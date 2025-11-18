# Strategy Configuration Contract

Standardized configuration format for all trading strategies.

## Overview

All strategies must follow the **StrategyConfig contract** defined in `otonom_trader/strategy/config.py`.

This provides:
- **Standardized YAML schema** - All strategies use same structure
- **Automatic validation** - Invalid configs are rejected
- **Backward compatibility** - Legacy YAMLs still work
- **Type safety** - Python dataclasses with type hints

---

## Required Fields

Every strategy YAML must have:

```yaml
name: "my_strategy"       # Strategy name (required)
description: "..."        # Description (optional)
version: "1.0.0"          # Version (semantic)

universe:                 # Required: which symbols to trade
  symbols: ["BTC-USD"]
```

---

## Configuration Sections

### 1. Universe (Required)

Defines which symbols to trade:

```yaml
universe:
  symbols:                # List of symbols
    - "BTC-USD"
    - "ETH-USD"
    - "^GSPC"

  universe_tags:          # Optional tags for filtering
    - "crypto"
    - "traditional"
```

**Validation**:
- At least `symbols` or `universe_tags` must be specified
- Empty lists are invalid

---

### 2. Risk (Optional - defaults provided)

Risk management parameters:

```yaml
risk:
  risk_pct: 1.0           # % of equity to risk per trade (0-10)
  stop_loss_pct: 5.0      # Stop-loss % (0-50)
  take_profit_pct: 10.0   # Take-profit %
  max_drawdown_pct: 40.0  # Max acceptable DD (alarm)
```

**Defaults**:
```python
risk_pct = 1.0
stop_loss_pct = 5.0
take_profit_pct = 10.0
max_drawdown_pct = 40.0
```

**Validation**:
- `risk_pct`: (0, 10]
- `stop_loss_pct`: (0, 50]
- `take_profit_pct`: > 0
- `max_drawdown_pct`: > 0

---

### 3. Filters (Optional - all None by default)

Signal filtering parameters:

```yaml
filters:
  dsi_threshold: 0.5      # DSI threshold (0-1)
  min_regime_vol: 0.02    # Min regime volatility
  max_regime_vol: 0.10    # Max regime volatility
  min_price: 1.0          # Min price filter
  min_volume: 1000000.0   # Min volume filter
```

**Defaults**:
```python
# All None = no filtering
dsi_threshold = None
min_regime_vol = None
max_regime_vol = None
min_price = None
min_volume = None
```

**Validation**:
- `dsi_threshold`: [0, 1] if specified
- All others: >= 0 if specified

---

### 4. Ensemble (Optional - defaults provided)

Analyst weights:

```yaml
ensemble:
  analyst_weights:
    tech: 1.0             # Technical analyst
    news: 1.2             # News/Macro/LLM
    risk: 0.8             # Regime/Risk
    rl: 0.0               # RL agent (disabled)

  disagreement_threshold: 0.5  # Penalty threshold
```

**Defaults**:
```python
tech = 1.0
news = 1.0
risk = 1.0
rl = 0.0
disagreement_threshold = 0.5
```

**Validation**:
- All weights >= 0
- `disagreement_threshold`: [0, 1]
- Warning if all weights are 0

**Analyst Naming**:
- `tech` = Technical (analist_1)
- `news` = News/Macro/LLM (analist_2)
- `risk` = Regime/Risk (analist_3)
- `rl` = RL agent

---

### 5. Execution (Optional - defaults provided)

Execution parameters:

```yaml
execution:
  bar_type: "D1"          # Bar type (D1, H1, M15, etc.)
  slippage_bps: 10.0      # Slippage in basis points (10 bps = 0.1%)
  max_trades_per_day: 10  # Max trades per day
  initial_capital: 100000.0  # Starting capital
```

**Defaults**:
```python
bar_type = "D1"
slippage_bps = 10.0  # 0.1%
max_trades_per_day = 10
initial_capital = 100000.0
```

**Validation**:
- `bar_type`: Warning if non-standard (D1, H1, M15, M5, M1, W1, MN)
- `slippage_bps`: >= 0
- `max_trades_per_day`: > 0
- `initial_capital`: > 0

---

## Python Usage

### Loading Config

```python
from otonom_trader.strategy.config import load_strategy_config

# Load and validate
config = load_strategy_config("strategies/baseline_v1.0.yaml")

# Access fields
print(config.name)                    # "baseline"
print(config.version)                 # "1.0"
print(config.risk.risk_pct)           # 1.0
print(config.ensemble.tech)           # 1.0
print(config.execution.initial_capital)  # 100000.0
```

### Accessing Nested Values

```python
# Using dot notation
risk_pct = config.risk.risk_pct

# Using get() method (dot notation string)
risk_pct = config.get("risk.risk_pct")
tech_weight = config.get("ensemble.analyst_weights.tech")
```

### Helper Methods

```python
# Get symbols
symbols = config.get_symbols()  # ["BTC-USD", "ETH-USD"]

# Get initial capital
capital = config.get_initial_capital()  # 100000.0

# Get risk per trade
risk = config.get_risk_per_trade_pct()  # 1.0

# Get backtest dates
start = config.get_backtest_start_date("crypto")  # "2017-01-01"
end = config.get_backtest_end_date("crypto")      # "2025-01-17"
```

---

## Backward Compatibility

The config system supports **legacy YAMLs** with old structure:

### Legacy Format (Still Supported)

```yaml
# Old format (baseline_v1.yaml)
data_sources:
  price_data:
    symbols: ["BTC-USD", "ETH-USD"]

risk_management:
  position_sizing:
    risk_per_trade_pct: 1.0
  stop_loss:
    percentage: 5.0
  take_profit:
    percentage: 10.0

analist_1:
  enabled: true
  weight: 1.0

analist_2:
  enabled: true
  weight: 1.2

analist_3:
  enabled: true
  weight: 0.8
```

This is **automatically converted** to the new format!

### New Simplified Format (Recommended)

```yaml
# New format (baseline_v1.0.yaml)
universe:
  symbols: ["BTC-USD", "ETH-USD"]

risk:
  risk_pct: 1.0
  stop_loss_pct: 5.0
  take_profit_pct: 10.0

ensemble:
  analyst_weights:
    tech: 1.0
    news: 1.2
    risk: 0.8
```

**Both work!** The new format is cleaner and easier to read.

---

## Validation

### Automatic Validation

Validation happens in dataclass `__post_init__`:

```python
config = load_strategy_config("strategies/my_strategy.yaml")
# Raises ValueError if invalid
```

### Common Validation Errors

**Missing required fields**:
```yaml
# ERROR: No name
description: "..."
```
→ `ValueError: Strategy YAML must have 'name' field`

**Invalid risk_pct**:
```yaml
risk:
  risk_pct: 15.0  # Too high!
```
→ `ValueError: risk_pct must be in (0, 10], got 15.0`

**Empty universe**:
```yaml
universe:
  symbols: []
  universe_tags: []
```
→ `ValueError: Either symbols or universe_tags must be specified`

**All analysts disabled**:
```yaml
ensemble:
  analyst_weights:
    tech: 0.0
    news: 0.0
    risk: 0.0
    rl: 0.0
```
→ `ValueError: At least one analyst must be enabled (weight > 0)`

---

## Complete Example

```yaml
---
# Complete strategy configuration example

name: "my_optimized_strategy"
description: "Optimized multi-analyst strategy from experiment #42"
version: "1.5"

# Universe
universe:
  symbols:
    - "BTC-USD"
    - "ETH-USD"
    - "SOL-USD"
  universe_tags:
    - "crypto"

# Risk Management
risk:
  risk_pct: 1.5              # 1.5% per trade (optimized from grid search)
  stop_loss_pct: 4.5         # Tighter stop (from robustness test)
  take_profit_pct: 12.0      # Higher TP (better R/R)
  max_drawdown_pct: 35.0     # Tighter DD tolerance

# Filters
filters:
  dsi_threshold: 0.4         # Filter extreme fear/greed
  min_regime_vol: 0.01       # Skip ultra-low vol
  max_regime_vol: 0.15       # Skip ultra-high vol
  min_volume: 1000000.0      # Min daily volume
  min_price: 1.0             # Skip penny stocks

# Ensemble Weights
ensemble:
  analyst_weights:
    tech: 1.2                # Boost technical (ablation showed +0.15 Sharpe)
    news: 1.0                # Standard weight
    risk: 0.9                # Slight reduction (less conservative)
    rl: 0.5                  # RL agent partially enabled

  disagreement_threshold: 0.6  # Higher tolerance for disagreement

# Execution
execution:
  bar_type: "D1"
  slippage_bps: 12.0         # Realistic slippage for crypto
  max_trades_per_day: 15     # Slightly more active
  initial_capital: 150000.0  # Higher starting capital

# Backtest Settings
backtest:
  date_ranges:
    crypto:
      start: "2018-01-01"
      end: "2025-01-17"
```

---

## Migration Guide

### From Legacy to Simplified

**Before (legacy)**:
```yaml
data_sources:
  price_data:
    symbols: ["BTC-USD"]

risk_management:
  position_sizing:
    risk_per_trade_pct: 1.0
  stop_loss:
    percentage: 5.0

analist_1:
  weight: 1.0
analist_2:
  weight: 1.2
```

**After (simplified)**:
```yaml
universe:
  symbols: ["BTC-USD"]

risk:
  risk_pct: 1.0
  stop_loss_pct: 5.0

ensemble:
  analyst_weights:
    tech: 1.0
    news: 1.2
```

**Benefits**:
- 50% less YAML
- Clearer structure
- Easier to read/modify
- Faster validation

---

## Best Practices

### 1. Use Simplified Format for New Strategies

✅ **Do**:
```yaml
risk:
  risk_pct: 1.0
  stop_loss_pct: 5.0
```

❌ **Don't**:
```yaml
risk_management:
  position_sizing:
    risk_per_trade_pct: 1.0
  stop_loss:
    percentage: 5.0
```

### 2. Document Changes in version/description

```yaml
name: "baseline"
version: "1.2"
description: "Increased risk to 1.5% and tightened stop to 4.5% based on experiment #42"
```

### 3. Validate Before Promoting

```python
from otonom_trader.strategy.config import load_strategy_config, validate_strategy_config

config = load_strategy_config("strategies/my_new_strategy.yaml")
validate_strategy_config(config)  # Raises ValueError if invalid
```

### 4. Use get() for Optional Fields

```python
# Safe: returns None if not specified
dsi = config.get("filters.dsi_threshold")

# Unsafe: raises AttributeError if missing
dsi = config.filters.dsi_threshold  # Only if you know it exists
```

---

## See Also

- [Strategy Promotion Workflow](strategy_promotion.md)
- [Experiment Templates](experiment_templates.md)
- [Research Engine](research_engine.md)
