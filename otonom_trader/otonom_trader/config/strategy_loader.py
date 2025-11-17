"""
Strategy configuration loader.

Loads and validates strategy YAML files with standardized schema.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UniverseConfig:
    """
    Universe configuration for strategy.

    Attributes:
        symbols: List of trading symbols (e.g., ["BTC-USD", "ETH-USD"])
        universe_tags: Optional tags for symbol groups (e.g., ["crypto", "fx"])
    """
    symbols: List[str] = field(default_factory=list)
    universe_tags: List[str] = field(default_factory=list)


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Attributes:
        risk_pct: Percentage of equity to risk per trade (0-10)
        stop_loss_pct: Stop-loss percentage (0-50)
        take_profit_pct: Take-profit percentage
        max_drawdown_pct: Optional maximum drawdown alarm threshold
    """
    risk_pct: float = 1.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_drawdown_pct: Optional[float] = None


@dataclass
class FiltersConfig:
    """
    Signal filtering configuration.

    Attributes:
        dsi_threshold: DSI (fear/greed) threshold for filtering
        regime_vol_min: Minimum regime volatility
        regime_vol_max: Maximum regime volatility
        min_volume: Minimum volume filter
        min_price: Minimum price filter
    """
    dsi_threshold: Optional[float] = None
    regime_vol_min: Optional[float] = None
    regime_vol_max: Optional[float] = None
    min_volume: Optional[float] = None
    min_price: Optional[float] = None


@dataclass
class EnsembleConfig:
    """
    Ensemble analyst weights and configuration.

    Attributes:
        tech_weight: Technical analyst weight (Analist-1)
        news_weight: News/Macro/LLM analyst weight (Analist-2)
        risk_weight: Regime/Risk analyst weight (Analist-3)
        rl_weight: RL agent weight (if enabled)
        disagreement_threshold: Threshold for high disagreement (triggers HOLD)
    """
    tech_weight: float = 1.0
    news_weight: float = 1.0
    risk_weight: float = 1.0
    rl_weight: float = 0.0
    disagreement_threshold: Optional[float] = None


@dataclass
class ExecutionConfig:
    """
    Execution configuration.

    Attributes:
        bar_type: Bar/candle type (e.g., "D1", "M15", "H1")
        slippage_pct: Slippage assumption percentage
        max_trades_per_day: Maximum number of trades per day
    """
    bar_type: str = "D1"
    slippage_pct: float = 0.1
    max_trades_per_day: int = 10


@dataclass
class StrategyConfig:
    """
    Strategy configuration dataclass.

    Attributes:
        name: Strategy name
        description: Strategy description
        version: Strategy version
        universe: Universe configuration
        risk: Risk management configuration
        filters: Signal filtering configuration
        ensemble: Ensemble analyst weights
        execution: Execution configuration
        raw_config: Full YAML config dict (for backward compatibility)
    """
    name: str
    description: str
    version: str
    universe: UniverseConfig
    risk: RiskConfig
    filters: FiltersConfig
    ensemble: EnsembleConfig
    execution: ExecutionConfig
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


def _extract_universe_config(raw: Dict[str, Any]) -> UniverseConfig:
    """Extract universe config from raw YAML."""
    universe_dict = raw.get("universe", {})

    # Support both new format and old format
    if "symbols" in universe_dict:
        symbols = universe_dict.get("symbols", [])
        tags = universe_dict.get("universe_tags", [])
    else:
        # Fallback to old format
        symbols = raw.get("data_sources", {}).get("price_data", {}).get("symbols", [])
        tags = []

    return UniverseConfig(symbols=symbols, universe_tags=tags)


def _extract_risk_config(raw: Dict[str, Any]) -> RiskConfig:
    """Extract risk config from raw YAML."""
    risk_dict = raw.get("risk", {})

    # Support both new format and old format
    if "risk_pct" in risk_dict:
        return RiskConfig(
            risk_pct=risk_dict.get("risk_pct", 1.0),
            stop_loss_pct=risk_dict.get("stop_loss_pct", 5.0),
            take_profit_pct=risk_dict.get("take_profit_pct", 10.0),
            max_drawdown_pct=risk_dict.get("max_drawdown_pct"),
        )
    else:
        # Fallback to old format
        rm = raw.get("risk_management", {})
        return RiskConfig(
            risk_pct=rm.get("position_sizing", {}).get("risk_per_trade_pct", 1.0),
            stop_loss_pct=rm.get("stop_loss", {}).get("percentage", 5.0),
            take_profit_pct=rm.get("take_profit", {}).get("percentage", 10.0),
            max_drawdown_pct=None,
        )


def _extract_filters_config(raw: Dict[str, Any]) -> FiltersConfig:
    """Extract filters config from raw YAML."""
    filters_dict = raw.get("filters", {})

    # Support both new format and old format
    if filters_dict:
        return FiltersConfig(
            dsi_threshold=filters_dict.get("dsi_threshold"),
            regime_vol_min=filters_dict.get("regime_vol_min"),
            regime_vol_max=filters_dict.get("regime_vol_max"),
            min_volume=filters_dict.get("min_volume"),
            min_price=filters_dict.get("min_price"),
        )
    else:
        # Extract from old format if available
        dsi_config = raw.get("analist_3", {}).get("dsi_analysis", {})
        return FiltersConfig(
            dsi_threshold=(
                dsi_config.get("extreme_fear_threshold")
                if dsi_config.get("contrarian_mode")
                else None
            ),
        )


def _extract_ensemble_config(raw: Dict[str, Any]) -> EnsembleConfig:
    """Extract ensemble config from raw YAML."""
    ensemble_dict = raw.get("ensemble", {})

    # Support both new format and old format
    if "tech_weight" in ensemble_dict or "analyst_weights" in ensemble_dict:
        weights = ensemble_dict.get("analyst_weights", {})
        return EnsembleConfig(
            tech_weight=weights.get("tech", ensemble_dict.get("tech_weight", 1.0)),
            news_weight=weights.get("news", ensemble_dict.get("news_weight", 1.0)),
            risk_weight=weights.get("risk", ensemble_dict.get("risk_weight", 1.0)),
            rl_weight=weights.get("rl", ensemble_dict.get("rl_weight", 0.0)),
            disagreement_threshold=ensemble_dict.get("disagreement_threshold"),
        )
    else:
        # Fallback to old format (analist_1.weight, analist_2.weight, etc.)
        return EnsembleConfig(
            tech_weight=raw.get("analist_1", {}).get("weight", 1.0),
            news_weight=raw.get("analist_2", {}).get("weight", 1.0),
            risk_weight=raw.get("analist_3", {}).get("weight", 1.0),
            rl_weight=0.0,
            disagreement_threshold=raw.get("ensemble", {}).get("disagreement_penalty", {}).get("threshold"),
        )


def _extract_execution_config(raw: Dict[str, Any]) -> ExecutionConfig:
    """Extract execution config from raw YAML."""
    exec_dict = raw.get("execution", {})

    # Support both new format and old format
    if "bar_type" in exec_dict:
        return ExecutionConfig(
            bar_type=exec_dict.get("bar_type", "D1"),
            slippage_pct=exec_dict.get("slippage_pct", 0.1),
            max_trades_per_day=exec_dict.get("max_trades_per_day", 10),
        )
    else:
        # Fallback to old format
        return ExecutionConfig(
            bar_type=raw.get("data_sources", {}).get("price_data", {}).get("frequency", "daily"),
            slippage_pct=exec_dict.get("slippage", {}).get("percentage", 0.1),
            max_trades_per_day=raw.get("portfolio_constraints", {}).get("turnover_limits", {}).get("max_daily_trades", 10),
        )


def validate_strategy_config(config: StrategyConfig) -> None:
    """
    Validate strategy configuration.

    Args:
        config: Strategy configuration to validate

    Raises:
        ValueError: If configuration is invalid

    Validation rules:
        - risk_pct: 0 < value <= 10
        - stop_loss_pct: 0 < value <= 50
        - take_profit_pct: value > 0
        - universe.symbols: not empty
        - ensemble total weight: > 0
    """
    # Risk validation
    if config.risk.risk_pct <= 0 or config.risk.risk_pct > 10:
        raise ValueError(
            f"risk.risk_pct must be between 0 and 10, got {config.risk.risk_pct}"
        )

    if config.risk.stop_loss_pct <= 0 or config.risk.stop_loss_pct > 50:
        raise ValueError(
            f"risk.stop_loss_pct must be between 0 and 50, got {config.risk.stop_loss_pct}"
        )

    if config.risk.take_profit_pct <= 0:
        raise ValueError(
            f"risk.take_profit_pct must be > 0, got {config.risk.take_profit_pct}"
        )

    # Universe validation
    if not config.universe.symbols:
        raise ValueError("universe.symbols cannot be empty")

    # Ensemble weights validation
    total_weight = (
        config.ensemble.tech_weight +
        config.ensemble.news_weight +
        config.ensemble.risk_weight +
        config.ensemble.rl_weight
    )

    if total_weight <= 0:
        raise ValueError(
            f"Total ensemble weights must be > 0, got {total_weight}"
        )

    # Execution validation
    if config.execution.max_trades_per_day <= 0:
        raise ValueError(
            f"execution.max_trades_per_day must be > 0, got {config.execution.max_trades_per_day}"
        )

    if config.execution.slippage_pct < 0:
        raise ValueError(
            f"execution.slippage_pct must be >= 0, got {config.execution.slippage_pct}"
        )


def load_strategy(path: str | Path, validate: bool = True) -> StrategyConfig:
    """
    Load strategy configuration from YAML file.

    Args:
        path: Path to strategy YAML file
        validate: Whether to validate config after loading (default: True)

    Returns:
        StrategyConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If validation fails

    Example:
        >>> config = load_strategy("strategies/baseline_v1.yaml")
        >>> print(config.name)
        baseline_v1
        >>> print(config.risk.risk_pct)
        1.0
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

    # Extract standardized configurations
    universe = _extract_universe_config(raw_config)
    risk = _extract_risk_config(raw_config)
    filters = _extract_filters_config(raw_config)
    ensemble = _extract_ensemble_config(raw_config)
    execution = _extract_execution_config(raw_config)

    config = StrategyConfig(
        name=name,
        description=description,
        version=version,
        universe=universe,
        risk=risk,
        filters=filters,
        ensemble=ensemble,
        execution=execution,
        raw_config=raw_config,
    )

    # Validate if requested
    if validate:
        validate_strategy_config(config)

    return config


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
