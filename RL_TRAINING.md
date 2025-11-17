# RL Agent Training Guide

Complete guide for training the RL agent using behavior cloning from ensemble decisions.

## Overview

The RL agent learns to imitate the ensemble's decision-making through **behavior cloning**:

1. **Offline Dataset Generation**: Extract (state, action, strength) tuples from backtest history
2. **Behavior Cloning**: Train neural network to predict ensemble actions
3. **Deployment**: Use trained policy as 4th analyst in ensemble

**Why Behavior Cloning?**
- Safe: Learn from proven ensemble, not random exploration
- Fast: Supervised learning is faster than RL
- Interpretable: Can analyze what the agent learned
- Foundation: Can fine-tune with RL later if needed

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     RL Agent Pipeline                         │
└──────────────────────────────────────────────────────────────┘

1. Offline Dataset Generation:
   Backtest History → RlStateBuilder → (state, action, strength) tuples

2. Behavior Cloning Training:
   Dataset → PolicyNet → Trained Model

3. Deployment:
   Current State → Trained Policy → Action + Strength
```

### State Representation (34 features)

The state vector captures:

1. **Market Features** (20 features):
   - Price returns history (20 days)

2. **Technical Indicators** (4 features):
   - RSI (normalized 0-1)
   - MACD signal
   - Bollinger Band width
   - Volume ratio

3. **Portfolio Features** (4 features):
   - Current position (-1 to +1)
   - Leverage
   - Cash percentage
   - Days since last trade (tanh normalized)

4. **Regime Features** (3 features):
   - Regime one-hot encoding [low_vol, normal, high_vol]

5. **Data Quality** (1 feature):
   - DSI (Data Health Index, 0-1)

6. **Sentiment** (2 features):
   - News sentiment (-1 to +1)
   - Macro sentiment (-1 to +1)

### Policy Network Architecture

```
Input (34 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU, Dropout)
    ↓
Hidden Layer 2 (128 neurons, ReLU, Dropout)
    ↓
    ├─→ Action Head (3 outputs: HOLD/BUY/SELL)
    └─→ Strength Head (1 output: position strength 0-1)
```

**Output:**
- **Action**: Softmax over 3 classes (HOLD=0, BUY=1, SELL=2)
- **Strength**: Sigmoid output (position size as fraction of capital)

**Loss Function:**
```
Total Loss = α * CrossEntropy(action) + β * MSE(strength)
```

Default: α = β = 1.0 (equal weighting)

## Workflow

### Step 1: Generate Offline Dataset

First, run backtests to generate decision history:

```bash
# Run backtest with ensemble (generates Decision records)
python -m otonom_trader.cli backtest \
  --strategy-path strategies/baseline_v1.yaml \
  --symbol BTC-USD \
  --start-date 2020-01-01 \
  --end-date 2024-01-01
```

Then extract offline dataset:

```python
from otonom_trader.data import get_session
from otonom_trader.research.offline_dataset import OfflineDatasetGenerator
from datetime import date

with get_session() as session:
    generator = OfflineDatasetGenerator(session)

    # Generate dataset from backtest history
    experiences = generator.generate_dataset(
        symbol="BTC-USD",
        start_date=date(2020, 1, 1),
        end_date=date(2024, 1, 1),
    )

    # Save as .npz file
    generator.save_dataset(
        experiences,
        output_path="data/offline_dataset.npz"
    )

print(f"Generated {len(experiences)} training samples")
```

**Expected output:**
```
Generated 1453 training samples
Saved to: data/offline_dataset.npz

Action distribution:
  HOLD: 872 (60.0%)
  BUY: 312 (21.5%)
  SELL: 269 (18.5%)
```

### Step 2: Train Behavior Cloning Model

**Option A: Command-line script**

```bash
python scripts/train_rl_agent.py \
  --dataset data/offline_dataset.npz \
  --output models/rl_bc \
  --epochs 20 \
  --batch-size 256 \
  --lr 0.001
```

**Option B: Python API**

```python
from otonom_trader.research.rl_training import (
    TrainingConfig,
    BehaviorCloningTrainer,
)

# Create config
cfg = TrainingConfig(
    dataset_path="data/offline_dataset.npz",
    output_dir="models/rl_bc",
    batch_size=256,
    lr=1e-3,
    max_epochs=20,
)

# Train
trainer = BehaviorCloningTrainer(cfg)
metrics = trainer.train()

print(f"Best val loss: {metrics.best_val_loss:.4f}")
print(f"Val action accuracy: {metrics.val_action_acc:.3f}")
```

**Training output:**
```
============================================================
RL Behavior Cloning Training
============================================================
Dataset: data/offline_dataset.npz
Loaded dataset: 1453 samples, 34 features
Action distribution: HOLD=872, BUY=312, SELL=269
Train size: 1308, Val size: 145
Model parameters: 54,595
Device: cuda
============================================================

[Epoch 1/20] train_loss=1.2345 val_loss=1.1234 train_acc=0.567 val_acc=0.589
[Epoch 2/20] train_loss=0.9876 val_loss=0.9234 train_acc=0.645 val_acc=0.658
✓ Saved best model (epoch 2, val_loss=0.9234)
...
[Epoch 20/20] train_loss=0.4567 val_loss=0.5123 train_acc=0.823 val_acc=0.796

============================================================
Training Complete!
============================================================
Best validation loss: 0.4892
Best epoch: 15
Val action accuracy: 0.812
Model saved to: models/rl_bc/policy_best.pt
============================================================
```

### Step 3: Evaluate Trained Model

```python
from otonom_trader.research.rl_inference import load_rl_policy
from otonom_trader.research.rl_state_builder import RlStateBuilder
from otonom_trader.data import get_session
from datetime import date

# Load trained policy
policy = load_rl_policy("models/rl_bc/policy_best.pt")

# Build state from recent data
with get_session() as session:
    state_builder = RlStateBuilder(session)
    state = state_builder.build_state(
        symbol="BTC-USD",
        current_date=date.today(),
        current_position=0.0,
        portfolio_equity=10000.0,
    )

# Predict action
action, strength, probs = policy.predict(state, return_probs=True)

print(f"Predicted action: {action.name}")
print(f"Position strength: {strength:.2f}")
print(f"Action probabilities:")
print(f"  HOLD: {probs[0]:.3f}")
print(f"  BUY:  {probs[1]:.3f}")
print(f"  SELL: {probs[2]:.3f}")
```

**Example output:**
```
Predicted action: BUY
Position strength: 0.75
Action probabilities:
  HOLD: 0.123
  BUY:  0.789
  SELL: 0.088
```

### Step 4: Integrate into Ensemble

Update strategy config to use RL agent as 4th analyst:

```yaml
# strategies/baseline_v1.yaml
ensemble:
  enabled: true
  analyst_weights:
    tech: 1.0      # Technical analyst
    news: 1.0      # News/LLM analyst
    risk: 1.0      # Risk/Regime analyst
    rl: 1.0        # RL agent (new!)

  rl_policy:
    enabled: true
    model_path: "models/rl_bc/policy_best.pt"
    device: "cpu"
```

Then use in backtest or daemon:

```python
from otonom_trader.research.rl_inference import load_rl_policy

# Load RL policy
rl_policy = load_rl_policy("models/rl_bc/policy_best.pt")

# In ensemble decision loop:
# 1. Get signals from Tech, News, Risk analysts
# 2. Get RL agent signal
rl_action, rl_strength = rl_policy.predict(current_state)

# 3. Aggregate with weighted voting
final_decision = aggregate_signals(
    tech_signal, news_signal, risk_signal, rl_action,
    weights=[1.0, 1.0, 1.0, 1.0]
)
```

## Training Configuration

### Hyperparameter Tuning

**Batch Size:**
- Small datasets (< 5k): 64-128
- Medium datasets (5k-50k): 256
- Large datasets (> 50k): 512

**Learning Rate:**
- Start with 1e-3
- If loss plateaus: reduce to 1e-4 or 1e-5
- If loss oscillates: reduce to 5e-4

**Hidden Dimension:**
- Small datasets: 64
- Medium datasets: 128
- Large datasets: 256-512

**Number of Layers:**
- Simple patterns: 1-2 layers
- Complex patterns: 3-4 layers
- Very complex: 4+ layers (risk of overfitting)

**Dropout:**
- No overfitting: 0.05-0.1
- Some overfitting: 0.1-0.2
- Heavy overfitting: 0.2-0.4

### Overfitting Detection

**Signs of overfitting:**
- Train loss << Val loss (gap > 0.2)
- Train accuracy >> Val accuracy (gap > 10%)
- Val loss increases after initial decrease

**Solutions:**
1. **Increase regularization:**
   ```yaml
   dropout: 0.2          # Increase from 0.1
   weight_decay: 0.0001  # Increase from 1e-5
   ```

2. **Reduce model capacity:**
   ```yaml
   hidden_dim: 64        # Reduce from 128
   num_layers: 1         # Reduce from 2
   ```

3. **Get more data:**
   - Run backtests on more symbols
   - Extend date range
   - Use data augmentation (noise injection)

4. **Early stopping:**
   - Training stops automatically at best val loss

### Class Imbalance Handling

If action distribution is very imbalanced (e.g., 80% HOLD, 10% BUY, 10% SELL):

**Option 1: Weighted loss**

```python
# In training code
action_counts = [872, 312, 269]  # HOLD, BUY, SELL
total = sum(action_counts)
weights = [total / (3 * c) for c in action_counts]
weights = torch.FloatTensor(weights)

ce_loss = nn.CrossEntropyLoss(weight=weights)
```

**Option 2: Oversample minority classes**

```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
sample_weights = [action_weights[action] for action in actions]
sampler = WeightedRandomSampler(sample_weights, len(dataset))

train_loader = DataLoader(train_ds, sampler=sampler, batch_size=256)
```

**Option 3: Focal loss**

Use focal loss to focus on hard examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()
```

## Advanced Techniques

### 1. Fine-tuning with RL

After behavior cloning, fine-tune with reinforcement learning:

```python
# Start with BC checkpoint
policy = load_rl_policy("models/rl_bc/policy_best.pt")

# Fine-tune with PPO/A2C
from stable_baselines3 import PPO

# Wrap policy for RL
rl_env = TradingEnv(policy)
rl_model = PPO("MlpPolicy", rl_env, learning_rate=1e-4)
rl_model.learn(total_timesteps=100000)
```

### 2. Ensemble of RL Policies

Train multiple policies and ensemble:

```python
# Train 3 policies with different seeds
policies = [
    load_rl_policy("models/rl_bc_seed42/policy_best.pt"),
    load_rl_policy("models/rl_bc_seed123/policy_best.pt"),
    load_rl_policy("models/rl_bc_seed456/policy_best.pt"),
]

# Ensemble prediction
action_votes = []
strengths = []

for policy in policies:
    action, strength = policy.predict(state)
    action_votes.append(action)
    strengths.append(strength)

# Majority voting for action
final_action = max(set(action_votes), key=action_votes.count)
final_strength = np.mean(strengths)
```

### 3. Multi-Task Learning

Train policy to predict multiple targets:

- Action (HOLD/BUY/SELL)
- Strength (position size)
- Expected return (regression)
- Risk level (classification)

```python
class MultiTaskPolicyNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.head_action = nn.Linear(128, 3)
        self.head_strength = nn.Linear(128, 1)
        self.head_return = nn.Linear(128, 1)     # New
        self.head_risk = nn.Linear(128, 3)       # New
```

### 4. Attention Mechanism

Add attention to focus on important features:

```python
class AttentionPolicyNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.attention = nn.Linear(state_dim, state_dim)
        self.backbone = nn.Sequential(...)

    def forward(self, x):
        # Attention weights
        attn = torch.softmax(self.attention(x), dim=-1)
        x_attended = x * attn

        h = self.backbone(x_attended)
        ...
```

## Deployment

### Production Checklist

- [ ] **Model validated**: Val accuracy > 70%
- [ ] **No overfitting**: Train/Val loss gap < 0.2
- [ ] **Backtested**: RL agent improves ensemble Sharpe
- [ ] **Stress tested**: Works in different market regimes
- [ ] **Inference speed**: < 10ms per prediction
- [ ] **Model versioned**: Saved with timestamp and metrics
- [ ] **Fallback**: Ensemble works without RL if model fails

### Monitoring in Production

Track these metrics:

1. **Action distribution**: Should match training distribution
2. **Strength values**: Should be in reasonable range (0.5-1.0)
3. **Prediction confidence**: High-confidence predictions should perform better
4. **Performance impact**: Track Sharpe with/without RL agent

### Retraining Schedule

Retrain RL agent:

- **Monthly**: For fast-changing markets (crypto)
- **Quarterly**: For stable markets (stocks)
- **After regime change**: When market dynamics shift
- **When performance degrades**: If RL accuracy drops < 60%

## Troubleshooting

### Issue: Low validation accuracy (< 60%)

**Possible causes:**
- Dataset too small
- Features not informative
- Model too simple
- Class imbalance

**Solutions:**
- Generate more training data (longer backtest)
- Add more features to state representation
- Increase model capacity (hidden_dim, num_layers)
- Use weighted loss or oversampling

### Issue: Overfitting (train acc >> val acc)

**Possible causes:**
- Model too complex
- Too little data
- No regularization

**Solutions:**
- Reduce hidden_dim and num_layers
- Increase dropout (0.2-0.3)
- Add weight_decay (1e-4)
- Get more training data

### Issue: Training loss not decreasing

**Possible causes:**
- Learning rate too high
- Learning rate too low
- Bad initialization

**Solutions:**
- Reduce learning rate (1e-4 or 1e-5)
- Try different optimizer (AdamW instead of Adam)
- Check gradient norms (may be exploding)

### Issue: RL agent performs poorly in backtest

**Possible causes:**
- Train/test distribution mismatch
- Overfitting to training period
- Missing important features

**Solutions:**
- Use walk-forward validation
- Train on diverse market regimes
- Add more regime-specific features

## Examples

### Example 1: Train with Custom Architecture

```bash
python scripts/train_rl_agent.py \
  --dataset data/offline_dataset.npz \
  --output models/rl_large \
  --hidden-dim 256 \
  --num-layers 3 \
  --dropout 0.2 \
  --epochs 50 \
  --lr 0.0005
```

### Example 2: Train Multiple Seeds for Ensemble

```bash
# Train 3 models with different seeds
for seed in 42 123 456; do
  python scripts/train_rl_agent.py \
    --dataset data/offline_dataset.npz \
    --output models/rl_seed_${seed} \
    --seed ${seed}
done
```

### Example 3: Evaluate Model on Test Set

```python
from otonom_trader.research.rl_inference import load_rl_policy
from otonom_trader.research.offline_dataset import OfflineDatasetGenerator
from sklearn.metrics import accuracy_score, classification_report

# Load policy
policy = load_rl_policy("models/rl_bc/policy_best.pt")

# Load test dataset
import numpy as np
data = np.load("data/offline_dataset_test.npz")
states = data["states"]
true_actions = data["actions"]

# Predict
pred_actions = []
for state_vec in states:
    # Convert to RlState (simplified)
    state = RlState(...)  # Build from state_vec
    action, _ = policy.predict(state)
    pred_actions.append(action.value)

# Evaluate
accuracy = accuracy_score(true_actions, pred_actions)
print(f"Test accuracy: {accuracy:.3f}")

# Detailed report
print(classification_report(
    true_actions, pred_actions,
    target_names=["HOLD", "BUY", "SELL"]
))
```

## Summary

**Complete Training Pipeline:**

```bash
# 1. Generate offline dataset
python -c "
from otonom_trader.data import get_session
from otonom_trader.research.offline_dataset import OfflineDatasetGenerator
from datetime import date

with get_session() as session:
    gen = OfflineDatasetGenerator(session)
    exp = gen.generate_dataset('BTC-USD', date(2020,1,1), date(2024,1,1))
    gen.save_dataset(exp, 'data/offline_dataset.npz')
"

# 2. Train behavior cloning model
python scripts/train_rl_agent.py \
  --dataset data/offline_dataset.npz \
  --output models/rl_bc \
  --epochs 20

# 3. Evaluate and deploy
python -c "
from otonom_trader.research.rl_inference import load_rl_policy

policy = load_rl_policy('models/rl_bc/policy_best.pt')
metadata = policy.get_metadata()
print(f'Val accuracy: {metadata[\"val_action_acc\"]:.3f}')
"
```

**Key Metrics:**
- ✅ Validation action accuracy > 70%
- ✅ No overfitting (train/val gap < 10%)
- ✅ Reasonable action distribution
- ✅ Improves ensemble performance in backtest

Ready to train your RL agent! 🚀
