"""
Standardized strategy configuration contract.

Provides a clean, validated interface for strategy parameters.
Supports both legacy (detailed) and simplified YAML formats.

Usage:
    from otonom_trader.strategy.config import load_strategy_config

    config = load_strategy_config("strategies/baseline_v1.0.yaml")
    print(config.risk.risk_pct)
    print(config.ensemble.tech)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# =================================
# Dataclass Definitions
# =================================

@dataclass
class UniverseConfig:
    """
    Universe configuration.

    Attributes:
        symbols: List of symbols to trade
        universe_tags: Tags for universe selection (e.g., ["crypto", "fx"])
    """
    symbols: List[str] = field(default_factory=list)
    universe_tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Ensure at least one is specified
        if not self.symbols and not self.universe_tags:
            raise ValueError("Either symbols or universe_tags must be specified")


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Attributes:
        risk_pct: Percentage of equity to risk per trade (0-10)
        stop_loss_pct: Stop-loss percentage (0-50)
        take_profit_pct: Take-profit percentage
        max_drawdown_pct: Maximum acceptable drawdown (for alarms)
    """
    risk_pct: float = 1.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_drawdown_pct: float = 40.0

    def __post_init__(self):
        if not (0.0 < self.risk_pct <= 10.0):
            raise ValueError(f"risk_pct must be in (0, 10], got {self.risk_pct}")

        if not (0.0 < self.stop_loss_pct <= 50.0):
            raise ValueError(f"stop_loss_pct must be in (0, 50], got {self.stop_loss_pct}")

        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")

        if self.max_drawdown_pct <= 0:
            raise ValueError(f"max_drawdown_pct must be > 0, got {self.max_drawdown_pct}")


@dataclass
class FiltersConfig:
    """
    Signal filtering configuration.

    Attributes:
        dsi_threshold: DSI threshold for filtering (0-1)
        min_regime_vol: Minimum regime volatility
        max_regime_vol: Maximum regime volatility
        min_price: Minimum price filter
        min_volume: Minimum volume filter
    """
    dsi_threshold: Optional[float] = None
    min_regime_vol: Optional[float] = None
    max_regime_vol: Optional[float] = None
    min_price: Optional[float] = None
    min_volume: Optional[float] = None

    def __post_init__(self):
        if self.dsi_threshold is not None:
            if not (0.0 <= self.dsi_threshold <= 1.0):
                raise ValueError(f"dsi_threshold must be in [0, 1], got {self.dsi_threshold}")

        if self.min_regime_vol is not None and self.min_regime_vol < 0:
            raise ValueError(f"min_regime_vol must be >= 0, got {self.min_regime_vol}")

        if self.max_regime_vol is not None and self.max_regime_vol < 0:
            raise ValueError(f"max_regime_vol must be >= 0, got {self.max_regime_vol}")

        if self.min_price is not None and self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")

        if self.min_volume is not None and self.min_volume < 0:
            raise ValueError(f"min_volume must be >= 0, got {self.min_volume}")


@dataclass
class EnsembleWeightsConfig:
    """
    Ensemble analyst weights configuration.

    Attributes:
        tech: Technical analyst weight
        news: News/Macro/LLM analyst weight
        risk: Regime/Risk analyst weight
        rl: RL agent weight
        disagreement_threshold: Disagreement penalty threshold (0-1)
    """
    tech: float = 1.0
    news: float = 1.0
    risk: float = 1.0
    rl: float = 0.0
    disagreement_threshold: float = 0.5

    def __post_init__(self):
        if not (0.0 <= self.disagreement_threshold <= 1.0):
            raise ValueError(f"disagreement_threshold must be in [0, 1], got {self.disagreement_threshold}")

        # Warn if all weights are zero
        if self.tech == 0 and self.news == 0 and self.risk == 0 and self.rl == 0:
            logger.warning("All analyst weights are zero - strategy will not generate signals")


@dataclass
class ExecutionConfig:
    """
    Execution configuration.

    Attributes:
        bar_type: Bar type ("D1", "H1", "M15", etc.)
        slippage_bps: Slippage in basis points
        max_trades_per_day: Maximum trades per day
        initial_capital: Initial capital for backtesting
    """
    bar_type: str = "D1"
    slippage_bps: float = 10.0  # 10 bps = 0.1%
    max_trades_per_day: int = 10
    initial_capital: float = 100000.0

    def __post_init__(self):
        valid_bar_types = {"D1", "H1", "M15", "M5", "M1", "W1", "MN"}
        if self.bar_type not in valid_bar_types:
            logger.warning(f"Non-standard bar_type: {self.bar_type} (valid: {valid_bar_types})")

        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")

        if self.max_trades_per_day <= 0:
            raise ValueError(f"max_trades_per_day must be > 0, got {self.max_trades_per_day}")

        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration.

    This is the standardized contract that all strategies must follow.

    Attributes:
        name: Strategy name (required)
        description: Strategy description
        version: Strategy version (semantic: major.minor.patch)
        universe: Universe configuration
        risk: Risk management configuration
        filters: Signal filtering configuration
        ensemble: Ensemble weights configuration
        execution: Execution configuration
        raw: Raw YAML data (for debugging/logging)
    """
    name: str
    description: str = ""
    version: str = "1.0.0"

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    ensemble: EnsembleWeightsConfig = field(default_factory=EnsembleWeightsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    raw: Dict[str, Any] = field(default_factory=dict)  # Original YAML for debugging

    def __repr__(self) -> str:
        return f"StrategyConfig(name={self.name}, version={self.version})"

    def get_initial_capital(self) -> float:
        """Get initial capital for backtesting."""
        return self.execution.initial_capital

    def get_risk_per_trade_pct(self) -> float:
        """Get risk per trade percentage."""
        return self.risk.risk_pct

    def get_symbols(self) -> List[str]:
        """Get list of symbols to trade."""
        return self.universe.symbols

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get nested config value using dot notation.

        Example:
            >>> config.get("risk.risk_pct")
            1.0
            >>> config.get("ensemble.tech")
            1.0
        """
        keys = key.split(".")
        value = self.raw

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_backtest_start_date(self, asset_class: str = "crypto") -> str:
        """
        Get backtest start date for asset class.

        Args:
            asset_class: "crypto" or "traditional"

        Returns:
            Start date string (YYYY-MM-DD)
        """
        # Try new simplified format first
        if "backtest" in self.raw:
            backtest = self.raw["backtest"]
            if "date_ranges" in backtest:
                ranges = backtest["date_ranges"]
                if asset_class in ranges:
                    return ranges[asset_class].get("start", "2017-01-01")

        # Fallback to defaults
        if asset_class == "crypto":
            return "2017-01-01"
        else:
            return "2008-01-01"

    def get_backtest_end_date(self, asset_class: str = "crypto") -> str:
        """
        Get backtest end date for asset class.

        Args:
            asset_class: "crypto" or "traditional"

        Returns:
            End date string (YYYY-MM-DD)
        """
        # Try new simplified format first
        if "backtest" in self.raw:
            backtest = self.raw["backtest"]
            if "date_ranges" in backtest:
                ranges = backtest["date_ranges"]
                if asset_class in ranges:
                    return ranges[asset_class].get("end", "2025-01-17")

        # Fallback to today
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


# =================================
# Loader Functions
# =================================

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Strategy YAML not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Strategy YAML must be a dictionary at top level")

    return data


def _parse_universe(data: Dict[str, Any]) -> UniverseConfig:
    """Parse universe configuration from YAML."""
    # Try simplified format first
    if "universe" in data:
        u = data["universe"] or {}
        return UniverseConfig(
            symbols=list(u.get("symbols", []) or []),
            universe_tags=list(u.get("universe_tags", []) or []),
        )

    # Fallback: extract from data_sources (legacy format)
    if "data_sources" in data:
        price_data = data["data_sources"].get("price_data", {})
        symbols = price_data.get("symbols", [])
        if symbols:
            return UniverseConfig(symbols=symbols, universe_tags=[])

    # No universe found - error
    raise ValueError("No universe configuration found (need 'universe' or 'data_sources.price_data.symbols')")


def _parse_risk(data: Dict[str, Any]) -> RiskConfig:
    """Parse risk configuration from YAML."""
    # Try simplified format first
    if "risk" in data:
        r = data["risk"] or {}
        return RiskConfig(
            risk_pct=float(r.get("risk_pct", 1.0)),
            stop_loss_pct=float(r.get("stop_loss_pct", 5.0)),
            take_profit_pct=float(r.get("take_profit_pct", 10.0)),
            max_drawdown_pct=float(r.get("max_drawdown_pct", 40.0)),
        )

    # Fallback: legacy format (risk_management section)
    if "risk_management" in data:
        rm = data["risk_management"]

        # Position sizing
        pos_sizing = rm.get("position_sizing", {})
        risk_pct = float(pos_sizing.get("risk_per_trade_pct", 1.0))

        # Stop loss
        stop_loss = rm.get("stop_loss", {})
        stop_loss_pct = float(stop_loss.get("percentage", 5.0))

        # Take profit
        take_profit = rm.get("take_profit", {})
        take_profit_pct = float(take_profit.get("percentage", 10.0))

        return RiskConfig(
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_drawdown_pct=40.0,  # Default
        )

    # Use defaults
    return RiskConfig()


def _parse_filters(data: Dict[str, Any]) -> FiltersConfig:
    """Parse filters configuration from YAML."""
    if "filters" in data:
        f = data["filters"] or {}
        return FiltersConfig(
            dsi_threshold=f.get("dsi_threshold"),
            min_regime_vol=f.get("min_regime_vol") or f.get("regime_vol_min"),
            max_regime_vol=f.get("max_regime_vol") or f.get("regime_vol_max"),
            min_price=f.get("min_price"),
            min_volume=f.get("min_volume"),
        )

    # Use defaults (all None = no filtering)
    return FiltersConfig()


def _parse_ensemble(data: Dict[str, Any]) -> EnsembleWeightsConfig:
    """Parse ensemble configuration from YAML."""
    # Try simplified format first
    if "ensemble" in data:
        e = data["ensemble"] or {}

        # Get weights
        weights = e.get("analyst_weights", {})

        # Try both old and new naming
        tech = float(weights.get("tech", weights.get("technical", 1.0)))
        news = float(weights.get("news", weights.get("macro", 1.0)))
        risk_weight = float(weights.get("risk", weights.get("regime", 1.0)))
        rl = float(weights.get("rl", 0.0))

        disagreement = float(e.get("disagreement_threshold", 0.5))

        return EnsembleWeightsConfig(
            tech=tech,
            news=news,
            risk=risk_weight,
            rl=rl,
            disagreement_threshold=disagreement,
        )

    # Fallback: legacy format (analist_1, analist_2, analist_3)
    analist_1 = data.get("analist_1", {})
    analist_2 = data.get("analist_2", {})
    analist_3 = data.get("analist_3", {})

    tech = float(analist_1.get("weight", 1.0)) if analist_1.get("enabled", True) else 0.0
    news = float(analist_2.get("weight", 1.0)) if analist_2.get("enabled", True) else 0.0
    risk_weight = float(analist_3.get("weight", 1.0)) if analist_3.get("enabled", True) else 0.0

    return EnsembleWeightsConfig(
        tech=tech,
        news=news,
        risk=risk_weight,
        rl=0.0,
        disagreement_threshold=0.5,
    )


def _parse_execution(data: Dict[str, Any]) -> ExecutionConfig:
    """Parse execution configuration from YAML."""
    if "execution" in data:
        ex = data["execution"] or {}

        bar_type = str(ex.get("bar_type", "D1"))

        # Slippage: support both bps and percentage
        slippage_bps = ex.get("slippage_bps")
        if slippage_bps is None:
            # Try percentage format
            slippage_pct = ex.get("slippage_pct", 0.1)
            slippage_bps = float(slippage_pct) * 100  # Convert % to bps

        max_trades = int(ex.get("max_trades_per_day", 10))
        initial_capital = float(ex.get("initial_capital", 100000.0))

        return ExecutionConfig(
            bar_type=bar_type,
            slippage_bps=float(slippage_bps),
            max_trades_per_day=max_trades,
            initial_capital=initial_capital,
        )

    # Use defaults
    return ExecutionConfig()


def load_strategy_config(path: str | Path) -> StrategyConfig:
    """
    Load and validate strategy configuration from YAML.

    Supports both simplified and legacy formats.

    Args:
        path: Path to strategy YAML file

    Returns:
        StrategyConfig instance

    Raises:
        FileNotFoundError: If YAML file not found
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_strategy_config("strategies/baseline_v1.0.yaml")
        >>> print(config.name, config.version)
        >>> print(config.risk.risk_pct)
    """
    data = _load_yaml(path)

    # Extract required fields
    name = data.get("name")
    if not name:
        raise ValueError("Strategy YAML must have 'name' field")

    description = str(data.get("description", ""))
    version = str(data.get("version", "1.0.0"))

    # Parse sections
    try:
        universe = _parse_universe(data)
        risk = _parse_risk(data)
        filters = _parse_filters(data)
        ensemble = _parse_ensemble(data)
        execution = _parse_execution(data)
    except Exception as e:
        raise ValueError(f"Failed to parse strategy config: {e}") from e

    # Create config
    config = StrategyConfig(
        name=name,
        description=description,
        version=version,
        universe=universe,
        risk=risk,
        filters=filters,
        ensemble=ensemble,
        execution=execution,
        raw=data,
    )

    logger.info(f"Loaded strategy config: {config.name} v{config.version}")

    return config


def validate_strategy_config(config: StrategyConfig) -> None:
    """
    Validate strategy configuration.

    Args:
        config: StrategyConfig to validate

    Raises:
        ValueError: If configuration is invalid

    Note:
        Most validation happens in dataclass __post_init__,
        but this function can add additional cross-field validation.
    """
    # Check that at least one analyst is enabled
    if (config.ensemble.tech == 0 and
        config.ensemble.news == 0 and
        config.ensemble.risk == 0 and
        config.ensemble.rl == 0):
        raise ValueError("At least one analyst must be enabled (weight > 0)")

    # Check risk/reward ratio
    if config.risk.take_profit_pct <= config.risk.stop_loss_pct:
        logger.warning(
            f"Take-profit ({config.risk.take_profit_pct}%) <= "
            f"Stop-loss ({config.risk.stop_loss_pct}%) - "
            "Risk/reward ratio < 1.0"
        )

    # Check universe
    if not config.universe.symbols and not config.universe.universe_tags:
        raise ValueError("Universe must have at least symbols or tags")

    logger.info(f"Strategy config validated: {config.name}")
