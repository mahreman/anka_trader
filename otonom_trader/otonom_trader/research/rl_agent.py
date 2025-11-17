"""
Reinforcement Learning (RL) agent for autonomous trading.

This module provides a skeleton for RL-based trading agents that can:
- Learn optimal trading policies from historical data
- Adapt to changing market conditions
- Balance exploration vs. exploitation

Potential algorithms:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- A3C (Asynchronous Advantage Actor-Critic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RlState:
    """
    RL agent state representation.

    Encodes complete market and portfolio state for RL agent input.

    Attributes:
        returns_history: Recent returns (normalized log returns)
        technical_indicators: Technical indicator values (RSI, MACD, etc.)
        portfolio_position: Current position size (-1 to 1)
        portfolio_leverage: Current leverage ratio
        portfolio_cash_pct: Cash as % of equity
        days_since_trade: Days since last trade
        regime: Market regime ID (0=low_vol, 1=high_vol, 2=crisis)
        dsi: Data health index (0-1)
        news_sentiment: News sentiment score (-1 to +1)
        macro_sentiment: Macro sentiment score (-1 to +1)

    Example:
        >>> state = RlState(
        ...     returns_history=np.array([0.01, -0.02, 0.015]),
        ...     technical_indicators={"rsi": 55.0, "macd": 0.5},
        ...     portfolio_position=0.5,
        ...     portfolio_leverage=0.5,
        ...     portfolio_cash_pct=0.5,
        ...     days_since_trade=3,
        ...     regime=0,
        ...     dsi=0.85,
        ...     news_sentiment=0.2,
        ...     macro_sentiment=0.1
        ... )
    """

    # Market features
    returns_history: np.ndarray  # Shape: (lookback_window,)
    technical_indicators: Dict[str, float]

    # Portfolio features
    portfolio_position: float  # -1 (max short) to +1 (max long)
    portfolio_leverage: float  # Current leverage ratio
    portfolio_cash_pct: float  # Cash as % of equity
    days_since_trade: int  # Days since last trade

    # Market regime and data quality
    regime: int  # Regime ID
    dsi: float  # Data health score

    # Sentiment features
    news_sentiment: float  # -1 (bearish) to +1 (bullish)
    macro_sentiment: float  # -1 (bearish) to +1 (bullish)

    def to_vector(self) -> np.ndarray:
        """
        Convert state to feature vector for RL agent.

        Returns:
            Flattened feature vector

        Example:
            >>> state = RlState(...)
            >>> vector = state.to_vector()
            >>> print(vector.shape)  # (30,) for lookback_window=20
        """
        # Flatten all components into single vector
        features = []

        # Returns history (already array)
        features.extend(self.returns_history.tolist())

        # Technical indicators (fixed order)
        features.extend([
            self.technical_indicators.get("rsi", 0.0) / 100.0,  # Normalize to [0,1]
            self.technical_indicators.get("macd", 0.0),
            self.technical_indicators.get("bb_width", 0.0),
            self.technical_indicators.get("volume_ratio", 1.0),
        ])

        # Portfolio features
        features.extend([
            self.portfolio_position,  # Already [-1, 1]
            self.portfolio_leverage,
            self.portfolio_cash_pct,
            np.tanh(self.days_since_trade / 30.0),  # Normalize with tanh
        ])

        # Regime (one-hot encoding for 3 regimes)
        regime_one_hot = [0.0, 0.0, 0.0]
        if 0 <= self.regime < 3:
            regime_one_hot[self.regime] = 1.0
        features.extend(regime_one_hot)

        # Data quality
        features.append(self.dsi)

        # Sentiment
        features.extend([
            self.news_sentiment,  # Already [-1, 1]
            self.macro_sentiment,  # Already [-1, 1]
        ])

        return np.array(features, dtype=np.float32)


@dataclass
class RlAction:
    """
    RL agent action output.

    Attributes:
        direction: Trading direction ("BUY", "SELL", "HOLD")
        strength: Action strength/confidence (0.0 to 1.0)
        target_position: Target position size (-1.0 to 1.0, where 0=neutral)

    Example:
        >>> action = RlAction(
        ...     direction="BUY",
        ...     strength=0.75,
        ...     target_position=0.5
        ... )
    """

    direction: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0.0 to 1.0
    target_position: float = 0.0  # -1.0 (max short) to 1.0 (max long)

    def __post_init__(self):
        """Validate action parameters."""
        assert self.direction in ["BUY", "SELL", "HOLD"], f"Invalid direction: {self.direction}"
        assert 0.0 <= self.strength <= 1.0, f"Strength must be in [0, 1]: {self.strength}"
        assert -1.0 <= self.target_position <= 1.0, f"Target position must be in [-1, 1]: {self.target_position}"


class RlAgent:
    """
    Reinforcement Learning trading agent (skeleton).

    This is a placeholder for future RL implementation. Currently returns
    HOLD actions. Subclass this to implement real RL algorithms.

    Attributes:
        config: Agent configuration
        model: RL model (e.g., neural network policy)
        training_mode: Whether agent is in training vs. inference mode

    Example:
        >>> config = {"learning_rate": 0.001, "gamma": 0.99}
        >>> agent = RlAgent(config)
        >>> state = RlState(...)
        >>> action = agent.act(state)
        >>> print(action.direction)
        'HOLD'
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RL agent.

        Args:
            config: Configuration dictionary with hyperparameters

        Example:
            >>> config = {
            ...     "model_type": "dqn",
            ...     "learning_rate": 0.001,
            ...     "gamma": 0.99,
            ...     "epsilon": 0.1,
            ...     "hidden_layers": [128, 64],
            ... }
            >>> agent = RlAgent(config)
        """
        self.config = config
        self.model = None  # Placeholder for RL model
        self.training_mode = False

        logger.info(f"RlAgent initialized with config: {config}")

    def act(self, state: RlState, deterministic: bool = False) -> RlAction:
        """
        Select action given current state.

        Args:
            state: Current market and portfolio state
            deterministic: If True, return greedy action; if False, sample from policy

        Returns:
            RlAction specifying trading decision

        Example:
            >>> state = RlState(...)
            >>> action = agent.act(state, deterministic=True)
        """
        # TODO: Implement real RL policy
        # For now, return HOLD action

        logger.debug("RlAgent.act() called (returning HOLD - not implemented)")

        return RlAction(
            direction="HOLD",
            strength=0.0,
            target_position=0.0,
        )

    def train(
        self,
        states: list[RlState],
        actions: list[RlAction],
        rewards: list[float],
        next_states: list[RlState],
        dones: list[bool],
    ) -> Dict[str, float]:
        """
        Train agent on batch of experiences.

        Args:
            states: List of states
            actions: List of actions taken
            rewards: List of rewards received
            next_states: List of next states
            dones: List of episode termination flags

        Returns:
            Training metrics (e.g., loss, q_value)

        Example:
            >>> metrics = agent.train(states, actions, rewards, next_states, dones)
            >>> print(f"Loss: {metrics['loss']:.4f}")
        """
        # TODO: Implement training loop
        logger.warning("RlAgent.train() not implemented")

        return {
            "loss": 0.0,
            "q_value": 0.0,
            "epsilon": self.config.get("epsilon", 0.1),
        }

    def save(self, path: str) -> None:
        """
        Save agent model to disk.

        Args:
            path: File path to save model

        Example:
            >>> agent.save("models/rl_agent_v1.pth")
        """
        # TODO: Implement model saving
        logger.warning(f"RlAgent.save() not implemented (path={path})")

    def load(self, path: str) -> None:
        """
        Load agent model from disk.

        Args:
            path: File path to load model from

        Example:
            >>> agent.load("models/rl_agent_v1.pth")
        """
        # TODO: Implement model loading
        logger.warning(f"RlAgent.load() not implemented (path={path})")

    def set_training_mode(self, training: bool) -> None:
        """
        Set agent to training or inference mode.

        Args:
            training: True for training mode, False for inference

        Example:
            >>> agent.set_training_mode(True)  # Enable training
            >>> agent.set_training_mode(False)  # Disable training (inference only)
        """
        self.training_mode = training
        logger.info(f"RlAgent mode: {'training' if training else 'inference'}")
