"""
RL analyst adapter for ensemble integration.

Integrates trained RL policy as 4th analyst in the ensemble.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..research.rl_training import PolicyNet, ACTION_HOLD, ACTION_BUY, ACTION_SELL
from ..research.rl_agent import RlState
from .ensemble import AnalystSignal, Direction

logger = logging.getLogger(__name__)


@dataclass
class RlAnalystConfig:
    """
    Configuration for RL analyst.

    Attributes:
        model_path: Path to trained policy model (.pt file)
        device: Device to use (cuda/cpu)
        enabled: Whether RL analyst is enabled
        weight: Base weight for RL analyst in ensemble
        confidence_threshold: Minimum confidence for non-HOLD actions
    """

    model_path: str = "models/rl_bc/policy_best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enabled: bool = True
    weight: float = 1.0
    confidence_threshold: float = 0.4  # Min confidence for BUY/SELL


class RlAnalyst:
    """
    RL analyst for ensemble decision-making.

    Wraps trained policy network and provides AnalystSignal interface
    for integration with ensemble.
    """

    def __init__(self, cfg: RlAnalystConfig):
        """
        Initialize RL analyst.

        Args:
            cfg: RL analyst configuration

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails
        """
        self.cfg = cfg

        if not cfg.enabled:
            logger.info("RL analyst disabled by config")
            return

        # Validate model exists
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RL model not found: {cfg.model_path}")

        # Load checkpoint
        logger.info(f"Loading RL model from: {cfg.model_path}")
        ckpt = torch.load(cfg.model_path, map_location=cfg.device)

        # Extract config
        config = ckpt.get("config", {})
        state_dim = ckpt.get("state_dim")
        if state_dim is None:
            # Fallback: try to get from config
            state_dim = config.get("state_dim", 34)

        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)

        # Create model
        self.model = PolicyNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Load weights
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(cfg.device)
        self.model.eval()

        self.device = cfg.device
        self.state_dim = state_dim

        # Store training metrics
        self.val_loss = ckpt.get("val_loss")
        self.val_action_acc = ckpt.get("val_action_acc")

        logger.info(
            f"RL analyst initialized: state_dim={state_dim}, "
            f"hidden_dim={hidden_dim}, device={self.device}"
        )
        if self.val_action_acc is not None:
            logger.info(f"Model validation accuracy: {self.val_action_acc:.3f}")

    @torch.no_grad()
    def infer(self, state: RlState) -> AnalystSignal:
        """
        Generate signal from RL state.

        Args:
            state: RL state object

        Returns:
            AnalystSignal for ensemble

        Example:
            >>> rl_analyst = RlAnalyst(RlAnalystConfig())
            >>> signal = rl_analyst.infer(state)
            >>> print(f"RL says: {signal.direction} (conf={signal.confidence:.2f})")
        """
        if not self.cfg.enabled:
            # Return neutral signal if disabled
            return AnalystSignal(
                name="Analist-4 (RL)",
                direction="HOLD",
                p_up=0.5,
                confidence=0.0,
                weight=0.0,
            )

        # Convert state to tensor
        state_vec = state.to_vector()
        x = torch.from_numpy(state_vec.astype(np.float32)).to(self.device).unsqueeze(0)

        # Forward pass
        logits, strength = self.model(x)

        # Get action probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        strength_val = float(strength.cpu().numpy()[0])

        # Determine action
        action_idx = int(probs.argmax())
        max_prob = float(probs[action_idx])

        # Convert to direction
        if action_idx == ACTION_BUY:
            direction: Direction = "BUY"
        elif action_idx == ACTION_SELL:
            direction: Direction = "SELL"
        else:
            direction: Direction = "HOLD"

        # Apply confidence threshold
        if direction != "HOLD" and max_prob < self.cfg.confidence_threshold:
            logger.debug(
                f"RL action {direction} below confidence threshold "
                f"({max_prob:.3f} < {self.cfg.confidence_threshold}), using HOLD"
            )
            direction = "HOLD"
            max_prob = probs[ACTION_HOLD]

        # Convert to p_up probability
        # p_up is probability that price will go up
        # If BUY: p_up should be high (> 0.5)
        # If SELL: p_up should be low (< 0.5)
        # If HOLD: p_up around 0.5
        if direction == "BUY":
            p_up = 0.5 + (max_prob - 0.33) * 0.5  # Map [0.33, 1.0] → [0.5, 0.83]
        elif direction == "SELL":
            p_up = 0.5 - (max_prob - 0.33) * 0.5  # Map [0.33, 1.0] → [0.5, 0.17]
        else:
            p_up = 0.5

        # Confidence is the max probability
        confidence = max_prob

        # Weight is combination of base weight and strength
        # Strength from model indicates position size preference
        effective_weight = self.cfg.weight * strength_val

        logger.debug(
            f"RL inference: action={direction}, p_up={p_up:.3f}, "
            f"conf={confidence:.3f}, strength={strength_val:.2f}"
        )

        return AnalystSignal(
            name="Analist-4 (RL)",
            direction=direction,
            p_up=p_up,
            confidence=confidence,
            weight=effective_weight,
        )

    def infer_from_vector(self, state_vec: np.ndarray) -> AnalystSignal:
        """
        Generate signal from raw state vector.

        This is useful when you have the state vector directly
        and don't want to reconstruct RlState.

        Args:
            state_vec: State vector (shape: (state_dim,))

        Returns:
            AnalystSignal for ensemble

        Example:
            >>> state_vec = np.random.randn(34)
            >>> signal = rl_analyst.infer_from_vector(state_vec)
        """
        if not self.cfg.enabled:
            return AnalystSignal(
                name="Analist-4 (RL)",
                direction="HOLD",
                p_up=0.5,
                confidence=0.0,
                weight=0.0,
            )

        # Validate shape
        if state_vec.shape[0] != self.state_dim:
            raise ValueError(
                f"State vector dimension mismatch: "
                f"expected {self.state_dim}, got {state_vec.shape[0]}"
            )

        # Create temporary RlState for consistency
        # (This is a hack, ideally we'd refactor to avoid this)
        # For now, we'll just do the inference directly

        x = torch.from_numpy(state_vec.astype(np.float32)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, strength = self.model(x)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        strength_val = float(strength.cpu().numpy()[0])

        action_idx = int(probs.argmax())
        max_prob = float(probs[action_idx])

        if action_idx == ACTION_BUY:
            direction: Direction = "BUY"
        elif action_idx == ACTION_SELL:
            direction: Direction = "SELL"
        else:
            direction: Direction = "HOLD"

        if direction != "HOLD" and max_prob < self.cfg.confidence_threshold:
            direction = "HOLD"
            max_prob = probs[ACTION_HOLD]

        if direction == "BUY":
            p_up = 0.5 + (max_prob - 0.33) * 0.5
        elif direction == "SELL":
            p_up = 0.5 - (max_prob - 0.33) * 0.5
        else:
            p_up = 0.5

        confidence = max_prob
        effective_weight = self.cfg.weight * strength_val

        return AnalystSignal(
            name="Analist-4 (RL)",
            direction=direction,
            p_up=p_up,
            confidence=confidence,
            weight=effective_weight,
        )

    def get_info(self) -> dict:
        """
        Get RL analyst information.

        Returns:
            Dictionary with model metadata

        Example:
            >>> info = rl_analyst.get_info()
            >>> print(f"State dim: {info['state_dim']}")
        """
        return {
            "enabled": self.cfg.enabled,
            "model_path": self.cfg.model_path,
            "state_dim": self.state_dim if self.cfg.enabled else None,
            "device": str(self.device) if self.cfg.enabled else None,
            "val_accuracy": self.val_action_acc if self.cfg.enabled else None,
            "base_weight": self.cfg.weight,
            "confidence_threshold": self.cfg.confidence_threshold,
        }


def create_rl_analyst(
    model_path: str = "models/rl_bc/policy_best.pt",
    enabled: bool = True,
    weight: float = 1.0,
    device: Optional[str] = None,
) -> RlAnalyst:
    """
    Create RL analyst with default configuration.

    Args:
        model_path: Path to trained model
        enabled: Whether to enable RL analyst
        weight: Base weight in ensemble
        device: Device to use (auto-detect if None)

    Returns:
        RlAnalyst instance

    Example:
        >>> rl_analyst = create_rl_analyst("models/rl_bc_v1/policy_best.pt")
        >>> signal = rl_analyst.infer(state)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = RlAnalystConfig(
        model_path=model_path,
        device=device,
        enabled=enabled,
        weight=weight,
    )

    return RlAnalyst(cfg)
