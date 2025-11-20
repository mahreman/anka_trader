"""
Strategy versioning utilities.

Manages strategy evolution and version tracking.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

from ..utils import utc_now


def create_strategy_version(
    base_strategy_path: str,
    new_version_name: str,
    param_changes: Dict[str, any],
    output_dir: str = "strategies",
    reason: Optional[str] = None,
) -> str:
    """
    Create new strategy version from base strategy with parameter changes.

    Args:
        base_strategy_path: Path to base strategy YAML
        new_version_name: Name for new version (e.g., "baseline_v2")
        param_changes: Dictionary of parameter changes (dot notation)
        output_dir: Output directory for new strategy
        reason: Reason for version change (for changelog)

    Returns:
        Path to new strategy file

    Example:
        >>> new_path = create_strategy_version(
        ...     base_strategy_path="strategies/baseline_v1.yaml",
        ...     new_version_name="baseline_v2",
        ...     param_changes={"risk.risk_pct": 1.5, "ensemble.analyst_weights.news": 1.2},
        ...     reason="Increased risk per trade and news weight based on exp_42 results"
        ... )
        >>> print(f"Created: {new_path}")
    """
    from .config_utils import set_nested

    # Load base strategy
    base_path = Path(base_strategy_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base strategy not found: {base_strategy_path}")

    with open(base_path, "r") as f:
        strategy_config = yaml.safe_load(f)

    # Update strategy name and version
    old_name = strategy_config.get("name", "unknown")
    strategy_config["name"] = new_version_name

    # Increment version if present
    old_version = strategy_config.get("version", "1.0")
    try:
        major, minor = old_version.split(".")
        new_version = f"{major}.{int(minor) + 1}"
    except:
        new_version = "1.1"

    strategy_config["version"] = new_version

    # Apply parameter changes
    for param_key, param_value in param_changes.items():
        set_nested(strategy_config, param_key, param_value)
        logger.info(f"Updated {param_key}: {param_value}")

    # Add metadata about versioning
    if "metadata" not in strategy_config:
        strategy_config["metadata"] = {}

    strategy_config["metadata"]["parent_strategy"] = old_name
    strategy_config["metadata"]["parent_version"] = old_version
    strategy_config["metadata"]["created_at"] = utc_now().isoformat()
    strategy_config["metadata"]["reason"] = reason or "Parameter tuning"
    strategy_config["metadata"]["param_changes"] = param_changes

    # Save new strategy
    output_path = Path(output_dir) / f"{new_version_name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(strategy_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created new strategy version: {output_path}")

    return str(output_path)


def append_to_strategy_log(
    log_path: str,
    version_name: str,
    parent_version: str,
    param_changes: Dict[str, any],
    reason: str,
    experiment_id: Optional[int] = None,
    expected_impact: Optional[str] = None,
) -> None:
    """
    Append strategy version change to STRATEGY_LOG.md.

    Args:
        log_path: Path to STRATEGY_LOG.md
        version_name: New version name
        parent_version: Parent version name
        param_changes: Parameter changes
        reason: Reason for change
        experiment_id: Optional experiment ID that motivated change
        expected_impact: Expected performance impact

    Example:
        >>> append_to_strategy_log(
        ...     log_path="STRATEGY_LOG.md",
        ...     version_name="baseline_v2",
        ...     parent_version="baseline_v1",
        ...     param_changes={"risk.risk_pct": 1.5},
        ...     reason="Exp #42 showed better Sharpe with higher risk",
        ...     experiment_id=42,
        ...     expected_impact="+10% CAGR, +0.2 Sharpe"
        ... )
    """
    log_file = Path(log_path)

    # Create log file if it doesn't exist
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("# Strategy Evolution Log\n\n")
            f.write("Track of all strategy versions and changes.\n\n")

    # Append new entry
    with open(log_file, "a") as f:
        f.write(f"## {version_name}\n\n")
        f.write(f"**Date**: {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"**Parent**: {parent_version}\n")

        if experiment_id:
            f.write(f"**Experiment**: #{experiment_id}\n")

        f.write(f"**Reason**: {reason}\n\n")

        f.write("### Parameter Changes\n\n")
        for key, value in param_changes.items():
            f.write(f"- `{key}`: {value}\n")
        f.write("\n")

        if expected_impact:
            f.write(f"**Expected Impact**: {expected_impact}\n\n")

        f.write("---\n\n")

    logger.info(f"Updated strategy log: {log_path}")


def promote_experiment_to_strategy(
    experiment_id: int,
    session,
    new_strategy_name: str,
    base_strategy_path: str,
    output_dir: str = "strategies",
    log_path: str = "STRATEGY_LOG.md",
    expected_impact: Optional[str] = None,
) -> str:
    """
    Promote experiment results to new strategy version.

    Finds best run from experiment, extracts parameters, and creates new strategy.

    Args:
        experiment_id: Experiment ID
        session: Database session
        new_strategy_name: Name for new strategy version
        base_strategy_path: Path to base strategy
        output_dir: Output directory for strategy
        log_path: Path to strategy log
        expected_impact: Expected performance impact

    Returns:
        Path to new strategy file

    Example:
        >>> from otonom_trader.data import get_session
        >>> with get_session() as session:
        ...     new_path = promote_experiment_to_strategy(
        ...         experiment_id=42,
        ...         session=session,
        ...         new_strategy_name="baseline_v2",
        ...         base_strategy_path="strategies/baseline_v1.yaml",
        ...         expected_impact="+15% CAGR, +0.3 Sharpe on test set"
        ...     )
    """
    from ..data import Experiment
    import json

    # Get experiment
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Find best run
    successful_runs = [
        r for r in experiment.runs
        if r.status == "done" and r.test_sharpe is not None
    ]

    if not successful_runs:
        raise ValueError(f"No successful runs in experiment {experiment_id}")

    best_run = max(successful_runs, key=lambda r: r.test_sharpe)

    # Extract parameter changes
    param_changes = json.loads(best_run.param_values_json)

    # Create reason
    reason = (
        f"Best run from experiment #{experiment_id}: "
        f"Test Sharpe={best_run.test_sharpe:.2f}, "
        f"CAGR={best_run.test_cagr:.1f}%, "
        f"MaxDD={best_run.test_max_dd:.1f}%"
    )

    # Create new strategy
    new_strategy_path = create_strategy_version(
        base_strategy_path=base_strategy_path,
        new_version_name=new_strategy_name,
        param_changes=param_changes,
        output_dir=output_dir,
        reason=reason,
    )

    # Update strategy log
    base_name = Path(base_strategy_path).stem
    append_to_strategy_log(
        log_path=log_path,
        version_name=new_strategy_name,
        parent_version=base_name,
        param_changes=param_changes,
        reason=reason,
        experiment_id=experiment_id,
        expected_impact=expected_impact,
    )

    logger.info(
        f"Promoted experiment {experiment_id} (run {best_run.run_index}) "
        f"to strategy: {new_strategy_path}"
    )

    return new_strategy_path
