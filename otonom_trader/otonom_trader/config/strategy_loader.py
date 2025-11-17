"""
Strategy configuration loader.

Loads and validates strategy YAML files.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StrategyConfig:
    """
    Strategy configuration dataclass.

    Attributes:
        name: Strategy name
        description: Strategy description
        version: Strategy version
        raw_config: Full YAML config dict
    """
    name: str
    description: str
    version: str
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., "analist_1.weight")
            default: Default value if not found

        Returns:
            Config value or default

        Example:
            >>> config.get("analist_1.weight", 1.0)
            1.0
        """
        keys = path.split(".")
        value = self.raw_config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_symbols(self) -> List[str]:
        """Get list of symbols from config."""
        return self.get("data_sources.price_data.symbols", [])

    def get_backtest_start_date(self, asset_class: str = "crypto") -> str:
        """Get backtest start date for asset class."""
        return self.get(f"backtest.date_ranges.{asset_class}.start", "2017-01-01")

    def get_backtest_end_date(self, asset_class: str = "crypto") -> str:
        """Get backtest end date for asset class."""
        return self.get(f"backtest.date_ranges.{asset_class}.end", "2025-01-17")

    def get_initial_capital(self) -> float:
        """Get initial capital for backtesting."""
        return self.get("execution.initial_capital", 100000.0)

    def get_risk_per_trade_pct(self) -> float:
        """Get risk per trade percentage."""
        return self.get("risk_management.position_sizing.risk_per_trade_pct", 1.0)

    def get_stop_loss_pct(self) -> float:
        """Get stop-loss percentage."""
        return self.get("risk_management.stop_loss.percentage", 5.0)

    def get_take_profit_pct(self) -> float:
        """Get take-profit percentage."""
        return self.get("risk_management.take_profit.percentage", 10.0)

    def get_max_daily_trades(self) -> int:
        """Get max daily trades."""
        return self.get("portfolio_constraints.turnover_limits.max_daily_trades", 10)

    def is_analist_enabled(self, analist_num: int) -> bool:
        """Check if analist is enabled."""
        return self.get(f"analist_{analist_num}.enabled", True)

    def get_analist_weight(self, analist_num: int) -> float:
        """Get analist weight."""
        return self.get(f"analist_{analist_num}.weight", 1.0)


def load_strategy(path: str | Path) -> StrategyConfig:
    """
    Load strategy configuration from YAML file.

    Args:
        path: Path to strategy YAML file

    Returns:
        StrategyConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid

    Example:
        >>> config = load_strategy("strategies/baseline_v1.yaml")
        >>> print(config.name)
        baseline_v1
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Invalid strategy file: expected dict, got {type(raw_config)}")

    # Extract metadata
    name = raw_config.get("name", path.stem)
    description = raw_config.get("description", "")
    version = raw_config.get("version", "1.0.0")

    return StrategyConfig(
        name=name,
        description=description,
        version=version,
        raw_config=raw_config,
    )


def list_strategies(directory: str | Path = "strategies") -> List[Path]:
    """
    List all strategy YAML files in directory.

    Args:
        directory: Directory to search

    Returns:
        List of strategy file paths

    Example:
        >>> strategies = list_strategies()
        >>> for strat in strategies:
        ...     config = load_strategy(strat)
        ...     print(config.name)
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
