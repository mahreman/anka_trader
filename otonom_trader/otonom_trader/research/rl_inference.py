"""
RL policy inference.

Load trained policy and make predictions for deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from .rl_training import PolicyNet, ACTION_HOLD, ACTION_BUY, ACTION_SELL
from .rl_agent import RlState, RlAction

logger = logging.getLogger(__name__)


class RlPolicyInference:
    """
    RL policy inference wrapper.

    Loads trained policy network and provides inference interface.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """
        Initialize inference wrapper.

        Args:
            model_path: Path to trained model (.pt file)
            device: Device to use (cuda/cpu)

        Example:
            >>> policy = RlPolicyInference("models/rl_bc/policy_best.pt")
            >>> action, strength = policy.predict(state)
        """
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract config
        config = checkpoint.get("config", {})
        state_dim = checkpoint.get("state_dim", 34)  # Default to 34 features
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)

        # Create model
        self.model = PolicyNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Store metadata
        self.state_dim = state_dim
        self.val_loss = checkpoint.get("val_loss", None)
        self.val_action_acc = checkpoint.get("val_action_acc", None)

        logger.info(f"Loaded RL policy from: {model_path}")
        logger.info(f"State dim: {state_dim}, Hidden dim: {hidden_dim}")
        if self.val_loss is not None:
            logger.info(f"Val loss: {self.val_loss:.4f}")
        if self.val_action_acc is not None:
            logger.info(f"Val action acc: {self.val_action_acc:.3f}")

    def predict(
        self, state: RlState, return_probs: bool = False
    ) -> Tuple[RlAction, float] | Tuple[RlAction, float, np.ndarray]:
        """
        Predict action from state.

        Args:
            state: RL state
            return_probs: Whether to return action probabilities

        Returns:
            (action, strength) or (action, strength, action_probs) tuple

        Example:
            >>> action, strength = policy.predict(state)
            >>> print(f"Action: {action.name}, Strength: {strength:.2f}")
        """
        # Convert state to tensor
        state_vec = state.to_vector()
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            action_probs, action_ids, strengths = self.model.predict_action(
                state_tensor
            )

        # Extract results
        action_id = int(action_ids[0].item())
        strength = float(strengths[0].item())
        probs = action_probs[0].cpu().numpy()

        # Convert action ID to RlAction enum
        if action_id == ACTION_HOLD:
            action = RlAction.HOLD
        elif action_id == ACTION_BUY:
            action = RlAction.BUY
        elif action_id == ACTION_SELL:
            action = RlAction.SELL
        else:
            logger.warning(f"Unknown action ID: {action_id}, defaulting to HOLD")
            action = RlAction.HOLD

        if return_probs:
            return action, strength, probs
        else:
            return action, strength

    def predict_batch(
        self, states: list[RlState]
    ) -> Tuple[list[RlAction], list[float]]:
        """
        Predict actions for batch of states.

        Args:
            states: List of RL states

        Returns:
            (actions, strengths) tuple

        Example:
            >>> actions, strengths = policy.predict_batch([state1, state2, state3])
        """
        # Convert states to tensor
        state_vecs = np.array([s.to_vector() for s in states])
        state_tensor = torch.from_numpy(state_vecs).to(self.device)

        # Predict
        with torch.no_grad():
            action_probs, action_ids, strengths = self.model.predict_action(
                state_tensor
            )

        # Convert results
        actions = []
        for action_id in action_ids.cpu().numpy():
            if action_id == ACTION_HOLD:
                actions.append(RlAction.HOLD)
            elif action_id == ACTION_BUY:
                actions.append(RlAction.BUY)
            elif action_id == ACTION_SELL:
                actions.append(RlAction.SELL)
            else:
                actions.append(RlAction.HOLD)

        strengths_list = strengths.cpu().numpy().tolist()

        return actions, strengths_list

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model metadata

        Example:
            >>> metadata = policy.get_metadata()
            >>> print(f"State dim: {metadata['state_dim']}")
        """
        return {
            "state_dim": self.state_dim,
            "val_loss": self.val_loss,
            "val_action_acc": self.val_action_acc,
            "device": str(self.device),
        }


def load_rl_policy(model_path: str | Path, device: str = "cpu") -> RlPolicyInference:
    """
    Load trained RL policy.

    Args:
        model_path: Path to trained model (.pt file)
        device: Device to use (cuda/cpu)

    Returns:
        RlPolicyInference instance

    Example:
        >>> policy = load_rl_policy("models/rl_bc/policy_best.pt")
        >>> action, strength = policy.predict(state)
    """
    return RlPolicyInference(model_path, device=device)
