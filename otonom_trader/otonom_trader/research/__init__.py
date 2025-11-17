"""
Research module for advanced analytics and machine learning.

Includes:
- RL (Reinforcement Learning) agents
- State builders for RL
- Offline dataset generation
- Behavior cloning (future)
"""

from .rl_agent import RlAgent, RlAction, RlState
from .rl_state_builder import RlStateBuilder, build_rl_state
from .offline_dataset import (
    OfflineDatasetGenerator,
    RlExperience,
    generate_offline_dataset,
)

__all__ = [
    # RL Agent
    "RlAgent",
    "RlAction",
    "RlState",
    # State Builder
    "RlStateBuilder",
    "build_rl_state",
    # Offline Dataset
    "OfflineDatasetGenerator",
    "RlExperience",
    "generate_offline_dataset",
]
