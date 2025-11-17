# RL Agent Integration Guide

Quick guide for using RL agent as 4th analyst in the ensemble.

## Overview

The RL agent is trained using **behavior cloning** to imitate the ensemble's decision-making, then deployed as the 4th analyst alongside:

1. **Analist-1 (Technical)**: Technical indicators and patterns
2. **Analist-2 (News/LLM)**: News sentiment analysis
3. **Analist-3 (Risk/Regime)**: Market regime and risk assessment
4. **Analist-4 (RL)**: Learned policy from ensemble history ⭐

## Quick Start

### 1. Train RL Agent

```bash
# Generate offline dataset from backtest history
python -c "
from otonom_trader.data import get_session
from otonom_trader.research.offline_dataset import OfflineDatasetGenerator
from datetime import date

with get_session() as session:
    gen = OfflineDatasetGenerator(session)
    exp = gen.generate_dataset('BTC-USD', date(2020,1,1), date(2024,1,1))
    gen.save_dataset(exp, 'data/offline_dataset.npz')
"

# Train behavior cloning model
python -m otonom_trader.cli rl train-bc \
  --dataset-path data/offline_dataset.npz \
  --output-dir models/rl_bc \
  --max-epochs 20
```

### 2. View Model Info

```bash
python -m otonom_trader.cli rl info models/rl_bc/policy_best.pt
```

**Output:**
```
Model: models/rl_bc/policy_best.pt
┌─────────────────────┬────────┐
│ Property            │ Value  │
├─────────────────────┼────────┤
│ State Dimension     │ 34     │
│ Device              │ cpu    │
│ Validation Loss     │ 0.4892 │
│ Val Action Accuracy │ 0.812  │
└─────────────────────┴────────┘
```

### 3. Use in Ensemble

**Option A: Python API**

```python
from otonom_trader.patron import (
    RlAnalyst,
    RlAnalystConfig,
    combine_signals,
)
from otonom_trader.research.rl_state_builder import RlStateBuilder

# Create RL analyst
rl_analyst = RlAnalyst(
    RlAnalystConfig(
        model_path="models/rl_bc/policy_best.pt",
        enabled=True,
        weight=1.0,
        confidence_threshold=0.4,
    )
)

# Build state from market data
with get_session() as session:
    state_builder = RlStateBuilder(session)
    state = state_builder.build_state(
        symbol="BTC-USD",
        current_date=date.today(),
        current_position=0.0,
        portfolio_equity=10000.0,
    )

# Get RL signal
rl_signal = rl_analyst.infer(state)

# Combine with other analysts
ensemble_decision = combine_signals([
    tech_signal,   # Analist-1
    news_signal,   # Analist-2
    risk_signal,   # Analist-3
    rl_signal,     # Analist-4 (RL)
])

print(f"Ensemble says: {ensemble_decision.direction}")
print(f"P(up): {ensemble_decision.p_up:.3f}")
print(f"Disagreement: {ensemble_decision.disagreement:.3f}")
```

**Option B: Strategy Config**

```yaml
# strategies/baseline_v2.yaml
ensemble:
  enabled: true
  analyst_weights:
    tech: 1.0
    news: 1.0
    risk: 1.0
    rl: 1.0        # Enable RL agent

  rl_policy:
    enabled: true
    model_path: "models/rl_bc/policy_best.pt"
    device: "cpu"
    confidence_threshold: 0.4
```

## CLI Commands

### Training

```bash
# Basic training
python -m otonom_trader.cli rl train-bc \
  --dataset-path data/offline_dataset.npz \
  --output-dir models/rl_bc

# Advanced training
python -m otonom_trader.cli rl train-bc \
  --dataset-path data/offline_dataset.npz \
  --output-dir models/rl_large \
  --hidden-dim 256 \
  --num-layers 3 \
  --dropout 0.2 \
  --max-epochs 50 \
  --lr 0.0005
```

### Model Management

```bash
# View model info
python -m otonom_trader.cli rl info models/rl_bc/policy_best.pt

# Test prediction
python -m otonom_trader.cli rl predict \
  --model-path models/rl_bc/policy_best.pt
```

## RL Analyst Configuration

```python
from otonom_trader.patron import RlAnalystConfig

cfg = RlAnalystConfig(
    model_path="models/rl_bc/policy_best.pt",  # Path to trained model
    device="cuda",                              # or "cpu"
    enabled=True,                               # Enable/disable
    weight=1.0,                                 # Base weight in ensemble
    confidence_threshold=0.4,                   # Min confidence for BUY/SELL
)
```

**Configuration Options:**

- `model_path`: Path to trained .pt model file
- `device`: "cuda" or "cpu" (auto-detects if not specified)
- `enabled`: Set to False to disable RL analyst without removing code
- `weight`: Base weight in ensemble (multiplied by model's strength output)
- `confidence_threshold`: Minimum confidence to take non-HOLD action (0-1)

## State Representation (34 Features)

The RL agent sees a 34-dimensional state vector:

1. **Price Returns** (20): Historical log returns
2. **Technical Indicators** (4): RSI, MACD, Bollinger Bands, Volume
3. **Portfolio** (4): Position, leverage, cash%, days since trade
4. **Regime** (3): Market regime one-hot encoding
5. **Data Quality** (1): DSI (Data Health Index)
6. **Sentiment** (2): News and macro sentiment

## Output

The RL analyst produces an `AnalystSignal`:

```python
@dataclass
class AnalystSignal:
    name: str            # "Analist-4 (RL)"
    direction: Direction # "BUY", "SELL", or "HOLD"
    p_up: float         # Probability price goes up (0-1)
    confidence: float   # Action confidence (0-1)
    weight: float       # Effective weight (base_weight * strength)
```

**Interpretation:**

- `direction`: Predicted action (HOLD/BUY/SELL)
- `p_up`: Probability that price will go up
  - BUY: p_up > 0.5 (bullish)
  - SELL: p_up < 0.5 (bearish)
  - HOLD: p_up ≈ 0.5 (neutral)
- `confidence`: Model confidence in action (max softmax probability)
- `weight`: Dynamic weight = base_weight × model_strength

## Examples

### Example 1: RL in Ensemble

```bash
python examples/rl_ensemble_integration.py
# Choose option 1
```

This demonstrates:
- Loading RL analyst
- Building state from market data
- Getting RL signal
- Combining with other analysts
- Viewing ensemble decision

### Example 2: RL Standalone

```bash
python examples/rl_ensemble_integration.py
# Choose option 2
```

This demonstrates:
- Using RL agent without other analysts
- Direct signal interpretation
- Confidence-based decision logic

### Example 3: Multiple Symbols

```bash
python examples/rl_ensemble_integration.py
# Choose option 3
```

This demonstrates:
- Batch inference for multiple symbols
- Error handling for missing data

## Training Workflow

```
1. Run Backtest
   └─> Generates Decision history in database

2. Extract Offline Dataset
   └─> Converts Decision history to (state, action, strength) tuples
   └─> Saves as .npz file

3. Train Behavior Cloning
   └─> Learns to imitate ensemble decisions
   └─> Saves best model checkpoint

4. Deploy to Ensemble
   └─> Load trained model
   └─> Use as 4th analyst
   └─> Combine with Tech/News/Risk signals
```

## Performance Metrics

**Training Metrics:**
- Validation action accuracy: 70-85% (typical)
- Training time: ~5-10 minutes for 10k samples
- Inference latency: < 10ms per prediction

**Production Metrics to Monitor:**
- Action distribution (should match training)
- Confidence levels (higher is better)
- Ensemble disagreement (lower with RL is good)
- Sharpe ratio improvement (RL should help)

## Best Practices

### Training

1. **Dataset Quality**: Use diverse market conditions
2. **Validation**: Check train/val accuracy gap < 10%
3. **Hyperparameters**: Start with defaults, tune if needed
4. **Overfitting**: Use dropout and weight decay

### Deployment

1. **Confidence Threshold**: Set to 0.4-0.5 for safety
2. **Weight**: Start with 1.0, adjust based on performance
3. **Monitoring**: Track action distribution and metrics
4. **Retraining**: Retrain monthly or when accuracy drops

### Integration

1. **Fallback**: Ensemble should work even if RL fails
2. **Logging**: Log RL decisions for analysis
3. **A/B Testing**: Compare ensemble with/without RL
4. **Gradual Rollout**: Start with low weight, increase gradually

## Troubleshooting

### Issue: Low Validation Accuracy

**Solution:**
- Generate more training data (longer backtest)
- Increase model capacity (hidden_dim, num_layers)
- Check feature quality (missing values, normalization)

### Issue: RL Agent Always Predicts HOLD

**Solution:**
- Lower confidence_threshold (default: 0.4)
- Check class balance in training data
- Increase action_weight in loss function

### Issue: RL Disagrees with Ensemble

**Solution:**
- This is expected! RL learns different patterns
- Monitor if disagreement improves performance
- If not, retrain with better data or disable

## Advanced Features

### Fine-tuning with RL

After behavior cloning, fine-tune with reinforcement learning:

```python
# Use trained policy as initialization
# Fine-tune with PPO/A2C on trading environment
# This can improve performance beyond ensemble
```

### Ensemble of RL Policies

Train multiple RL agents and ensemble:

```python
policies = [
    load_rl_policy("models/rl_seed_42/policy_best.pt"),
    load_rl_policy("models/rl_seed_123/policy_best.pt"),
    load_rl_policy("models/rl_seed_456/policy_best.pt"),
]

# Majority voting or probability averaging
```

### Custom State Features

Add domain-specific features:

```python
# Extend RlState with custom features
# Retrain RL agent with enhanced state
# Better features → better decisions
```

## Related Documentation

- `RL_TRAINING.md`: Complete training guide (661 lines)
- `RL_README.md`: RL architecture and approach
- `EXPERIMENT_WORKFLOW.md`: Systematic improvement process
- `DATA_PROVIDERS.md`: Real data integration

## Summary

**Key Commands:**
```bash
# Train RL agent
python -m otonom_trader.cli rl train-bc --dataset-path data/offline_dataset.npz

# View model
python -m otonom_trader.cli rl info models/rl_bc/policy_best.pt

# Test
python examples/rl_ensemble_integration.py
```

**Key Classes:**
- `RlAnalyst`: Main integration class
- `RlAnalystConfig`: Configuration
- `AnalystSignal`: Output format for ensemble

**Integration Point:**
```python
from otonom_trader.patron import RlAnalyst, combine_signals

rl_analyst = RlAnalyst(config)
rl_signal = rl_analyst.infer(state)
decision = combine_signals([tech, news, risk, rl_signal])
```

RL agent is ready to trade! 🤖📈
