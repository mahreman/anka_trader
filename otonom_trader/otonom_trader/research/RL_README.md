# RL Agent Integration

## Overview

The RL (Reinforcement Learning) agent is designed to be integrated as a **4th analyst** in the ensemble, learning from historical trading decisions and market outcomes.

## Architecture

### 1. State Representation (RlState)

Complete market and portfolio state encoding:

**Market Features:**
- `returns_history`: Last N normalized log returns
- `technical_indicators`: RSI, MACD, Bollinger Band width, volume ratio

**Portfolio Features:**
- `portfolio_position`: Current position (-1 to +1)
- `portfolio_leverage`: Leverage ratio
- `portfolio_cash_pct`: Cash percentage
- `days_since_trade`: Time since last trade

**Regime & Quality:**
- `regime`: Market regime ID (0=low_vol, 1=high_vol, 2=crisis)
- `dsi`: Data health score (0-1)

**Sentiment:**
- `news_sentiment`: News sentiment (-1 to +1)
- `macro_sentiment`: Macro sentiment (-1 to +1)

**Feature Vector:** ~30 dimensions (20 returns + 10 other features)

### 2. Action Representation (RlAction)

Trading decision output:

- `direction`: "BUY", "SELL", or "HOLD"
- `strength`: Confidence level (0 to 1)
- `target_position`: Target position size (-1 to +1)

### 3. Training Approach: Behavior Cloning

**Phase 1: Supervised Learning (Behavior Cloning)**

Instead of learning from scratch, the RL agent starts by **imitating the ensemble**:

1. **Extract Historical Decisions:**
   ```python
   from otonom_trader.research import generate_offline_dataset

   dataset = generate_offline_dataset(
       session, "BTC-USD",
       start_date=date(2023, 1, 1),
       end_date=date(2024, 1, 1)
   )
   # Returns: [(state, action, reward), ...]
   ```

2. **Train Supervised Model:**
   - Input: RlState.to_vector() → 30-dim feature vector
   - Output: Action logits (BUY/SELL/HOLD) + strength
   - Loss: Cross-entropy (direction) + MSE (strength)
   - Model: Simple MLP (128-64-32 hidden layers)

3. **Benefits:**
   - Starts from "ensemble teacher" knowledge
   - Avoids random exploration phase
   - Faster convergence
   - Lower risk of catastrophic failures

**Phase 2: Fine-tuning with RL (Future)**

After behavior cloning, improve with RL:

- **Environment:** Backtest engine
- **Reward:** Sharpe-adjusted returns
- **Algorithm:** PPO or SAC
- **Exploration:** ε-greedy or entropy regularization

### 4. Integration as 4th Analyst

The RL agent outputs signals like other analysts:

```python
from otonom_trader.research import RlAgent
from otonom_trader.domain import AnalystSignal

# Create RL agent
rl_agent = RlAgent(config={
    "model_path": "models/rl_agent_v1.pth",
    "learning_rate": 0.001,
})

# Use in patron
state = build_rl_state(session, symbol, current_date)
action = rl_agent.act(state, deterministic=True)

# Convert to analyst signal
rl_signal = AnalystSignal(
    name="RL",
    strength=action.strength,
    direction=action.direction,
    confidence=action.strength,
    reasoning=f"RL prediction based on {state.regime} regime"
)

# Ensemble combines with other analysts
ensemble_decision = combine_signals([
    tech_signal,
    news_signal,
    risk_signal,
    rl_signal  # 4th analyst
])
```

## Offline Dataset Generation

### Usage

```python
from otonom_trader.research import OfflineDatasetGenerator
from otonom_trader.data import get_session

generator = OfflineDatasetGenerator(lookback_window=20)

with get_session() as session:
    # Generate dataset
    dataset = generator.generate_dataset(
        session=session,
        symbol="BTC-USD",
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        min_dsi=0.3  # Filter low-quality data
    )

    # Save for later use
    generator.save_dataset(dataset, "data/btc_training_data.npz")

    print(f"Generated {len(dataset)} training examples")
```

### Data Format

Saved as `.npz` file with:
- `states`: (N, 30) array of state vectors
- `actions`: (N, 2) array of [direction, strength]
- `rewards`: (N,) array of rewards
- `dones`: (N,) array of episode termination flags

## Training Pipeline

### 1. Generate Offline Dataset

```bash
# TODO: Add CLI command
python -m otonom_trader.research.offline_dataset \
  --symbol BTC-USD \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output data/btc_offline_dataset.npz
```

### 2. Train Behavior Cloning Model

```python
# TODO: Implement trainer
from otonom_trader.research import BehaviorCloningTrainer

trainer = BehaviorCloningTrainer(
    model_type="mlp",
    hidden_layers=[128, 64, 32],
    learning_rate=0.001,
)

# Load offline dataset
trainer.load_dataset("data/btc_offline_dataset.npz")

# Train
trainer.train(
    epochs=100,
    batch_size=64,
    val_split=0.2,
)

# Save model
trainer.save("models/rl_agent_bc_v1.pth")
```

### 3. Evaluate Model

```python
# Backtest with RL agent
from otonom_trader.research import RlAgent

agent = RlAgent(config={})
agent.load("models/rl_agent_bc_v1.pth")

# Use in backtest...
```

### 4. Deploy as 4th Analyst

```python
# Update ensemble weights in strategy YAML
ensemble:
  analyst_weights:
    tech: 1.0
    news: 1.0
    risk: 1.0
    rl: 0.5  # Start with lower weight
```

## Roadmap

### ✅ Completed

- [x] RlState with full features (returns, portfolio, sentiment, regime, DSI)
- [x] RlStateBuilder to construct state from database
- [x] OfflineDatasetGenerator to extract training data
- [x] RlAction representation
- [x] Feature vector encoding with normalization

### 🚧 In Progress

- [ ] Behavior cloning trainer (supervised learning)
- [ ] RL agent integration as 4th analyst
- [ ] Backtest evaluation with RL agent

### 📋 Future Work

- [ ] PPO/SAC fine-tuning after behavior cloning
- [ ] Online learning during live trading
- [ ] Multi-asset RL agent
- [ ] Attention-based state encoder (Transformer)
- [ ] Ensemble distillation (compress 3 analysts → 1 RL agent)

## Key Design Decisions

1. **Start with Behavior Cloning, Not Random RL:**
   - Avoids costly random exploration
   - Learns from proven ensemble decisions
   - Safer for production deployment

2. **State Builder Separation:**
   - Decouples state construction from agent logic
   - Easier to add new features
   - Can be reused for different RL algorithms

3. **Offline Dataset First:**
   - Train on historical data before live deployment
   - Validate performance in backtest
   - Reduce risk of live trading failures

4. **4th Analyst, Not Replacement:**
   - RL complements existing analysts
   - Ensemble can adaptively weight RL vs. traditional
   - Graceful degradation if RL fails

## Example: End-to-End Workflow

```python
from otonom_trader.data import get_session
from otonom_trader.research import (
    generate_offline_dataset,
    RlAgent,
    build_rl_state,
)

# Step 1: Generate training data
with get_session() as session:
    dataset = generate_offline_dataset(
        session, "BTC-USD",
        date(2020, 1, 1), date(2023, 12, 31),
        output_path="data/btc_dataset.npz"
    )

# Step 2: Train behavior cloning model
# TODO: Implement BehaviorCloningTrainer

# Step 3: Load trained agent
agent = RlAgent(config={})
agent.load("models/rl_agent_v1.pth")

# Step 4: Use in live trading
with get_session() as session:
    state = build_rl_state(
        session, "BTC-USD", datetime.now(),
        portfolio_position=0.5,
        portfolio_equity=100000
    )

    action = agent.act(state, deterministic=True)
    print(f"RL Agent: {action.direction} with strength {action.strength:.2f}")
```

## References

- **Behavior Cloning:** Learning from expert demonstrations
- **Offline RL:** Training from historical data without environment interaction
- **PPO:** Proximal Policy Optimization for online fine-tuning
- **SAC:** Soft Actor-Critic for continuous action spaces

## Notes

- This is a **research-grade** implementation
- Requires PyTorch or TensorFlow for neural networks (not yet added as dependency)
- Start with simple MLPs before trying complex architectures
- Always backtest before live deployment
- Monitor RL agent performance closely in production
