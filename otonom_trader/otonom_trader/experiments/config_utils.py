"""
Configuration utilities for experiment parameter overrides.

Provides utilities for loading, saving, and modifying strategy configurations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import yaml


def load_strategy_config(path: str) -> Dict[str, Any]:
    """
    Load strategy configuration from YAML file.

    Args:
        path: Path to strategy YAML file

    Returns:
        Dictionary with strategy configuration

    Example:
        >>> config = load_strategy_config("strategies/baseline_v1.yaml")
        >>> print(config["name"])
        baseline_v1
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_strategy_config(cfg: Dict[str, Any], path: str) -> None:
    """
    Save strategy configuration to YAML file.

    Args:
        cfg: Strategy configuration dictionary
        path: Output path for YAML file

    Example:
        >>> config = {"name": "test", "risk_pct": 1.0}
        >>> save_strategy_config(config, "strategies/test.yaml")
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get_nested(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Get nested value from config using dot notation.

    Args:
        config: Configuration dictionary
        dotted_key: Dot-separated key path (e.g., "risk.stop_loss_pct")
        default: Default value if key not found

    Returns:
        Value at the specified path, or default if not found

    Example:
        >>> config = {"risk": {"stop_loss_pct": 5.0}}
        >>> get_nested(config, "risk.stop_loss_pct")
        5.0
        >>> get_nested(config, "risk.nonexistent", default=0)
        0
    """
    parts = dotted_key.split(".")
    cur = config

    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default

    return cur


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Set nested value in config using dot notation.

    Creates intermediate dictionaries if they don't exist.

    Args:
        config: Configuration dictionary (modified in-place)
        dotted_key: Dot-separated key path (e.g., "risk.stop_loss_pct")
        value: Value to set

    Example:
        >>> config = {}
        >>> set_nested(config, "risk.stop_loss_pct", 5.0)
        >>> config
        {'risk': {'stop_loss_pct': 5.0}}

        >>> config = {"risk": {"take_profit_pct": 10.0}}
        >>> set_nested(config, "risk.stop_loss_pct", 5.0)
        >>> config
        {'risk': {'take_profit_pct': 10.0, 'stop_loss_pct': 5.0}}
    """
    parts = dotted_key.split(".")
    cur = config

    # Navigate/create intermediate dicts
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]

    # Set final value
    cur[parts[-1]] = value


def apply_param_overrides(
    base_cfg: Dict[str, Any],
    param_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply parameter overrides to base configuration.

    Creates a deep copy of base config and applies overrides.
    Param names can use dot notation for nested access.

    Args:
        base_cfg: Base configuration dictionary
        param_values: Parameter overrides (dot notation keys allowed)

    Returns:
        New configuration with overrides applied

    Example:
        >>> base = {
        ...     "name": "baseline_v1",
        ...     "risk": {
        ...         "stop_loss_pct": 5.0,
        ...         "take_profit_pct": 10.0,
        ...     }
        ... }
        >>> overrides = {
        ...     "risk.stop_loss_pct": 8.0,
        ...     "risk.take_profit_pct": 15.0,
        ... }
        >>> new_cfg = apply_param_overrides(base, overrides)
        >>> new_cfg["risk"]["stop_loss_pct"]
        8.0
        >>> new_cfg["risk"]["take_profit_pct"]
        15.0
    """
    # Deep copy to avoid modifying original
    cfg = deepcopy(base_cfg)

    # Apply each override
    for key, val in param_values.items():
        set_nested(cfg, key, val)

    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"d": 4}, "e": 5}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 4}, 'e': 5}
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def validate_param_path(config: Dict[str, Any], dotted_key: str) -> bool:
    """
    Check if a parameter path exists in config.

    Args:
        config: Configuration dictionary
        dotted_key: Dot-separated key path

    Returns:
        True if path exists, False otherwise

    Example:
        >>> config = {"risk": {"stop_loss_pct": 5.0}}
        >>> validate_param_path(config, "risk.stop_loss_pct")
        True
        >>> validate_param_path(config, "risk.nonexistent")
        False
    """
    parts = dotted_key.split(".")
    cur = config

    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False

    return True


def extract_param_values(config: Dict[str, Any], param_paths: list[str]) -> Dict[str, Any]:
    """
    Extract parameter values from config using dot notation paths.

    Args:
        config: Configuration dictionary
        param_paths: List of dot-separated parameter paths

    Returns:
        Dictionary mapping paths to values

    Example:
        >>> config = {
        ...     "risk": {"stop_loss_pct": 5.0, "take_profit_pct": 10.0},
        ...     "ensemble": {"enabled": True}
        ... }
        >>> paths = ["risk.stop_loss_pct", "ensemble.enabled"]
        >>> extract_param_values(config, paths)
        {'risk.stop_loss_pct': 5.0, 'ensemble.enabled': True}
    """
    result = {}

    for path in param_paths:
        value = get_nested(config, path)
        if value is not None:
            result[path] = value

    return result
