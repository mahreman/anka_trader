"""
Offline dataset generation for RL training.

Extracts (state, action, reward) tuples from backtest history for behavior cloning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from .rl_agent import RlState, RlAction
from .rl_state_builder import RlStateBuilder

logger = logging.getLogger(__name__)


@dataclass
class RlExperience:
    """
    Single RL experience tuple.

    Attributes:
        state: State at time t
        action: Action taken at time t
        reward: Reward received at time t+1
        next_state: State at time t+1
        done: Whether episode terminated
    """

    state: RlState
    action: RlAction
    reward: float
    next_state: Optional[RlState]
    done: bool


class OfflineDatasetGenerator:
    """
    Generate offline RL dataset from backtest history.

    Extracts (state, action, reward) tuples from Decision table for
    behavior cloning and offline RL training.

    Example:
        >>> generator = OfflineDatasetGenerator()
        >>> dataset = generator.generate_dataset(
        ...     session=session,
        ...     symbol="BTC-USD",
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2024, 1, 1)
        ... )
        >>> print(f"Generated {len(dataset)} experiences")
    """

    def __init__(self, lookback_window: int = 20):
        """
        Initialize dataset generator.

        Args:
            lookback_window: Lookback window for state builder
        """
        self.state_builder = RlStateBuilder(lookback_window=lookback_window)

    def generate_dataset(
        self,
        session: Session,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        min_dsi: float = 0.3,
    ) -> List[RlExperience]:
        """
        Generate offline dataset from backtest history.

        Args:
            session: Database session
            symbol: Symbol to generate dataset for
            start_date: Start date
            end_date: End date
            min_dsi: Minimum DSI threshold (filter low-quality data)

        Returns:
            List of RlExperience tuples

        Example:
            >>> dataset = generator.generate_dataset(
            ...     session, "BTC-USD",
            ...     date(2023, 1, 1), date(2024, 1, 1)
            ... )
        """
        from ..data import Decision

        # Get all decisions in date range
        decisions = (
            session.query(Decision)
            .filter(
                Decision.symbol == symbol,
                Decision.timestamp >= start_date,
                Decision.timestamp <= end_date,
            )
            .order_by(Decision.timestamp.asc())
            .all()
        )

        logger.info(f"Found {len(decisions)} decisions for {symbol}")

        experiences = []

        for i, decision in enumerate(decisions):
            # Build state at time t
            try:
                state = self.state_builder.build_state(
                    session=session,
                    symbol=symbol,
                    current_date=decision.timestamp,
                    portfolio_position=0.0,  # TODO: Track actual position
                    portfolio_equity=100000.0,  # TODO: Track actual equity
                )

                # Filter by DSI
                if state.dsi < min_dsi:
                    continue

                # Extract action from decision
                action = self._decision_to_action(decision)

                # Calculate reward (use next decision or price change)
                reward = self._calculate_reward(session, symbol, decision, decisions, i)

                # Build next state (if available)
                next_state = None
                done = False

                if i + 1 < len(decisions):
                    next_decision = decisions[i + 1]
                    next_state = self.state_builder.build_state(
                        session=session,
                        symbol=symbol,
                        current_date=next_decision.timestamp,
                        portfolio_position=0.0,  # TODO: Track actual position
                        portfolio_equity=100000.0,  # TODO: Track actual equity
                    )
                else:
                    done = True

                # Create experience
                experience = RlExperience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                experiences.append(experience)

            except Exception as e:
                logger.warning(f"Failed to create experience for decision {decision.id}: {e}")
                continue

        logger.info(f"Generated {len(experiences)} experiences (filtered by DSI >= {min_dsi})")

        return experiences

    def _decision_to_action(self, decision) -> RlAction:
        """
        Convert Decision to RlAction.

        Args:
            decision: Decision record

        Returns:
            RlAction
        """
        # Map decision direction to action
        direction = decision.direction  # "BUY" or "SELL"
        strength = decision.strength  # 0.0 to 1.0

        # Calculate target position
        if direction == "BUY":
            target_position = strength
        elif direction == "SELL":
            target_position = -strength
        else:
            target_position = 0.0

        return RlAction(
            direction=direction,
            strength=strength,
            target_position=target_position,
        )

    def _calculate_reward(
        self,
        session: Session,
        symbol: str,
        current_decision,
        all_decisions: List,
        current_index: int,
    ) -> float:
        """
        Calculate reward for decision.

        Uses next-day return as reward signal.

        Args:
            session: Database session
            symbol: Symbol
            current_decision: Current decision
            all_decisions: All decisions
            current_index: Index of current decision

        Returns:
            Reward (can be positive or negative)
        """
        from ..data import DailyBar

        # Get current day price
        current_bar = (
            session.query(DailyBar)
            .filter(
                DailyBar.symbol == symbol,
                DailyBar.date <= current_decision.timestamp,
            )
            .order_by(DailyBar.date.desc())
            .first()
        )

        if not current_bar:
            return 0.0

        # Get next day price
        if current_index + 1 < len(all_decisions):
            next_decision = all_decisions[current_index + 1]
            next_bar = (
                session.query(DailyBar)
                .filter(
                    DailyBar.symbol == symbol,
                    DailyBar.date <= next_decision.timestamp,
                )
                .order_by(DailyBar.date.desc())
                .first()
            )

            if next_bar and next_bar.id != current_bar.id:
                # Calculate return
                price_return = (next_bar.close - current_bar.close) / current_bar.close

                # Reward is return * position direction
                if current_decision.direction == "BUY":
                    reward = price_return * current_decision.strength
                elif current_decision.direction == "SELL":
                    reward = -price_return * current_decision.strength
                else:
                    reward = 0.0

                return reward

        return 0.0

    def save_dataset(self, experiences: List[RlExperience], output_path: str) -> None:
        """
        Save dataset to disk.

        Args:
            experiences: List of experiences
            output_path: Output file path (npz format)

        Example:
            >>> generator.save_dataset(dataset, "data/rl_dataset_btc.npz")
        """
        # Convert to arrays
        states = np.array([exp.state.to_vector() for exp in experiences])
        actions = np.array([
            [1.0 if exp.action.direction == "BUY" else (-1.0 if exp.action.direction == "SELL" else 0.0),
             exp.action.strength]
            for exp in experiences
        ])
        rewards = np.array([exp.reward for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        # Save
        np.savez(
            output_path,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        logger.info(f"Saved {len(experiences)} experiences to {output_path}")


def generate_offline_dataset(
    session: Session,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_path: Optional[str] = None,
) -> List[RlExperience]:
    """
    Convenience function to generate offline dataset.

    Args:
        session: Database session
        symbol: Symbol
        start_date: Start date
        end_date: End date
        output_path: Optional output path to save dataset

    Returns:
        List of experiences

    Example:
        >>> from otonom_trader.data import get_session
        >>> with get_session() as session:
        ...     dataset = generate_offline_dataset(
        ...         session, "BTC-USD",
        ...         date(2023, 1, 1), date(2024, 1, 1),
        ...         output_path="data/btc_dataset.npz"
        ...     )
    """
    generator = OfflineDatasetGenerator()
    dataset = generator.generate_dataset(session, symbol, start_date, end_date)

    if output_path:
        generator.save_dataset(dataset, output_path)

    return dataset
