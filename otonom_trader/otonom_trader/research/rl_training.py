"""
RL behavior cloning trainer.

Trains policy network to imitate ensemble decisions using offline dataset.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

# Action encoding
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


class OfflineTradingDataset(Dataset):
    """
    Offline RL dataset from .npz file.

    Expected fields:
      - states: (N, D) float32 - State features
      - actions: (N,) int64 - Action labels (0=HOLD, 1=BUY, 2=SELL)
      - strengths: (N,) float32 - Position strength [0, 1]
      - (optional) rewards: (N,) float32 - Rewards
    """

    def __init__(self, path: str | Path):
        """
        Load dataset from .npz file.

        Args:
            path: Path to .npz file
        """
        data = np.load(path)

        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.strengths = data["strengths"].astype(np.float32)

        # Optional rewards
        self.rewards = (
            data["rewards"].astype(np.float32)
            if "rewards" in data
            else np.zeros(len(self.states), dtype=np.float32)
        )

        assert (
            self.states.shape[0]
            == self.actions.shape[0]
            == self.strengths.shape[0]
            == self.rewards.shape[0]
        ), "Dataset size mismatch"

        self.n, self.d = self.states.shape

        logger.info(f"Loaded dataset: {self.n} samples, {self.d} features")
        logger.info(
            f"Action distribution: "
            f"HOLD={np.sum(self.actions == ACTION_HOLD)}, "
            f"BUY={np.sum(self.actions == ACTION_BUY)}, "
            f"SELL={np.sum(self.actions == ACTION_SELL)}"
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return self.n

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single sample.

        Returns:
            (state, action, strength, reward) tuple
        """
        s = torch.from_numpy(self.states[idx])
        a = torch.tensor(self.actions[idx], dtype=torch.long)
        st = torch.tensor(self.strengths[idx], dtype=torch.float32)
        r = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return s, a, st, r


class PolicyNet(nn.Module):
    """
    Policy network for behavior cloning.

    Architecture:
      - Input: D-dimensional state vector
      - Backbone: MLP with ReLU activations
      - Output heads:
        - action_logits: (batch, 3) - Action logits (HOLD/BUY/SELL)
        - strength: (batch,) - Position strength [0, 1]
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize policy network.

        Args:
            state_dim: State dimension (e.g., 34)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()

        # Build backbone layers
        layers = []
        in_dim = state_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.head_action = nn.Linear(hidden_dim, 3)  # HOLD, BUY, SELL
        self.head_strength = nn.Linear(hidden_dim, 1)  # Position strength

        logger.info(
            f"PolicyNet: state_dim={state_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: State tensor (batch, state_dim)

        Returns:
            (logits, strength) tuple
              - logits: (batch, 3) - Action logits
              - strength: (batch,) - Position strength [0, 1]
        """
        h = self.backbone(x)

        logits = self.head_action(h)
        strength = torch.sigmoid(self.head_strength(h)).squeeze(-1)

        return logits, strength

    def predict_action(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action from state.

        Args:
            x: State tensor (batch, state_dim)

        Returns:
            (action_probs, action_ids, strengths) tuple
              - action_probs: (batch, 3) - Action probabilities
              - action_ids: (batch,) - Predicted action IDs
              - strengths: (batch,) - Position strengths
        """
        logits, strength = self.forward(x)

        action_probs = torch.softmax(logits, dim=-1)
        action_ids = torch.argmax(action_probs, dim=-1)

        return action_probs, action_ids, strength


@dataclass
class TrainingConfig:
    """
    Behavior cloning training configuration.

    Attributes:
        dataset_path: Path to offline dataset (.npz)
        output_dir: Output directory for models and metrics
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization weight
        max_epochs: Maximum training epochs
        val_ratio: Validation set ratio
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout probability
        action_weight: Weight for action loss
        strength_weight: Weight for strength loss
        seed: Random seed
        device: Device (cuda/cpu)
    """

    dataset_path: str
    output_dir: str = "models/rl_bc"
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    val_ratio: float = 0.1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    action_weight: float = 1.0
    strength_weight: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """
    Training metrics.

    Attributes:
        best_val_loss: Best validation loss
        best_epoch: Best epoch number
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        train_action_acc: Final training action accuracy
        val_action_acc: Final validation action accuracy
        train_action_loss: Final training action loss
        val_action_loss: Final validation action loss
        train_strength_loss: Final training strength loss
        val_strength_loss: Final validation strength loss
    """

    best_val_loss: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    train_action_acc: float
    val_action_acc: float
    train_action_loss: float
    val_action_loss: float
    train_strength_loss: float
    val_strength_loss: float


class BehaviorCloningTrainer:
    """
    Behavior cloning trainer.

    Trains policy network to imitate ensemble decisions from offline dataset.
    """

    def __init__(self, cfg: TrainingConfig):
        """
        Initialize trainer.

        Args:
            cfg: Training configuration
        """
        self.cfg = cfg

        # Set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Load dataset
        ds = OfflineTradingDataset(cfg.dataset_path)
        val_len = int(len(ds) * cfg.val_ratio)
        train_len = len(ds) - val_len
        self.train_ds, self.val_ds = random_split(
            ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed)
        )

        logger.info(f"Train size: {train_len}, Val size: {val_len}")

        self.state_dim = ds.d

        # Create model
        self.device = torch.device(cfg.device)
        self.model = PolicyNet(
            state_dim=self.state_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Data loaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False
        )

        logger.info(f"Device: {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def _step(
        self, batch, train: bool
    ) -> Tuple[float, float, float, float]:
        """
        Training/validation step.

        Args:
            batch: (states, actions, strengths, rewards) batch
            train: Whether to perform backward pass

        Returns:
            (total_loss, action_loss, strength_loss, action_acc) tuple
        """
        states, actions, strengths, rewards = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        strengths = strengths.to(self.device)

        # Forward pass
        logits, pred_strength = self.model(states)

        # Action classification loss
        loss_action = self.ce_loss(logits, actions)

        # Strength regression loss
        loss_strength = self.mse_loss(pred_strength, strengths)

        # Total loss
        loss = (
            self.cfg.action_weight * loss_action
            + self.cfg.strength_weight * loss_strength
        )

        # Action accuracy
        pred_actions = torch.argmax(logits, dim=-1)
        action_acc = (pred_actions == actions).float().mean()

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return (
            float(loss.item()),
            float(loss_action.item()),
            float(loss_strength.item()),
            float(action_acc.item()),
        )

    def train(self) -> TrainingMetrics:
        """
        Train policy network.

        Returns:
            Training metrics
        """
        best_val_loss = math.inf
        best_epoch = -1

        # Create output directory
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Starting behavior cloning training...")

        for epoch in range(1, self.cfg.max_epochs + 1):
            # Training
            self.model.train()
            train_losses = []
            train_action_losses = []
            train_strength_losses = []
            train_action_accs = []

            for batch in self.train_loader:
                loss, action_loss, strength_loss, action_acc = self._step(
                    batch, train=True
                )
                train_losses.append(loss)
                train_action_losses.append(action_loss)
                train_strength_losses.append(strength_loss)
                train_action_accs.append(action_acc)

            # Validation
            self.model.eval()
            val_losses = []
            val_action_losses = []
            val_strength_losses = []
            val_action_accs = []

            with torch.no_grad():
                for batch in self.val_loader:
                    loss, action_loss, strength_loss, action_acc = self._step(
                        batch, train=False
                    )
                    val_losses.append(loss)
                    val_action_losses.append(action_loss)
                    val_strength_losses.append(strength_loss)
                    val_action_accs.append(action_acc)

            # Average metrics
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            train_action_loss = (
                float(np.mean(train_action_losses)) if train_action_losses else 0.0
            )
            val_action_loss = (
                float(np.mean(val_action_losses)) if val_action_losses else 0.0
            )
            train_strength_loss = (
                float(np.mean(train_strength_losses))
                if train_strength_losses
                else 0.0
            )
            val_strength_loss = (
                float(np.mean(val_strength_losses)) if val_strength_losses else 0.0
            )
            train_action_acc = (
                float(np.mean(train_action_accs)) if train_action_accs else 0.0
            )
            val_action_acc = (
                float(np.mean(val_action_accs)) if val_action_accs else 0.0
            )

            logger.info(
                f"[Epoch {epoch}/{self.cfg.max_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_action_acc:.3f} val_acc={val_action_acc:.3f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": asdict(self.cfg),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_action_acc": val_action_acc,
                        "state_dim": self.state_dim,
                    },
                    str(Path(self.cfg.output_dir) / "policy_best.pt"),
                )

                logger.info(
                    f"✓ Saved best model (epoch {epoch}, val_loss={val_loss:.4f})"
                )

        # Final metrics
        metrics = TrainingMetrics(
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            train_action_acc=train_action_acc,
            val_action_acc=val_action_acc,
            train_action_loss=train_action_loss,
            val_action_loss=val_action_loss,
            train_strength_loss=train_strength_loss,
            val_strength_loss=val_strength_loss,
        )

        # Save metrics
        metrics_path = Path(self.cfg.output_dir) / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

        logger.info(f"Training complete! Best epoch: {best_epoch}")
        logger.info(f"Metrics saved to: {metrics_path}")

        return metrics


def train_behavior_cloning(cfg_dict: Dict[str, Any]) -> TrainingMetrics:
    """
    Train behavior cloning model.

    Args:
        cfg_dict: Configuration dictionary

    Returns:
        Training metrics

    Example:
        >>> cfg = {
        ...     "dataset_path": "data/offline_dataset.npz",
        ...     "output_dir": "models/rl_bc",
        ...     "max_epochs": 20,
        ... }
        >>> metrics = train_behavior_cloning(cfg)
        >>> print(f"Best val loss: {metrics.best_val_loss:.4f}")
    """
    cfg = TrainingConfig(**cfg_dict)
    trainer = BehaviorCloningTrainer(cfg)
    return trainer.train()
