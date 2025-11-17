"""
Experiment engine for systematic parameter optimization.

Provides grid search and random search capabilities for strategy optimization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..config import load_strategy, StrategyConfig
from ..data import Experiment, ExperimentRun
from .grid import ParamGrid
from .config_utils import apply_param_overrides as apply_config_overrides
from .runner import run_train_test_backtest

logger = logging.getLogger(__name__)


def load_param_grid(path: str | Path) -> ParamGrid:
    """
    Load parameter grid from YAML file.

    Args:
        path: Path to grid YAML file

    Returns:
        ParamGrid instance

    Example:
        >>> grid = load_param_grid("grids/baseline_grid.yaml")
        >>> print(grid.params)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parameter grid file not found: {path}")

    return ParamGrid.from_yaml(str(path))


def generate_param_combinations(
    param_grid: ParamGrid | Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate parameter combinations from grid.

    Args:
        param_grid: ParamGrid instance or dict with grid config

    Returns:
        List of parameter combination dictionaries

    Example:
        >>> grid = ParamGrid(params={"risk_pct": [0.5, 1.0]})
        >>> combos = generate_param_combinations(grid)
        >>> len(combos)
        2
    """
    # Handle both ParamGrid and dict inputs for backward compatibility
    if isinstance(param_grid, dict):
        # Legacy dict format - convert to ParamGrid
        logger.warning("Passing dict to generate_param_combinations is deprecated. Use ParamGrid instead.")

        parameters = param_grid.get("parameters", {})
        if not parameters:
            logger.warning("No parameters defined in grid")
            return [{}]

        # Extract values
        params = {}
        for param_name, param_config in parameters.items():
            if "values" in param_config:
                params[param_name] = param_config["values"]

        grid = ParamGrid(
            params=params,
            search_method=param_grid.get("search_method", "grid"),
            random_samples=param_grid.get("random_samples", 50),
        )
    else:
        grid = param_grid

    # Use ParamGrid's iter_combinations
    combinations = grid.iter_combinations()
    logger.info(f"Generated {len(combinations)} parameter combinations ({grid.search_method} search)")

    return combinations


def apply_param_overrides(
    strategy_config: StrategyConfig,
    param_overrides: Dict[str, Any],
) -> StrategyConfig:
    """
    Apply parameter overrides to strategy config.

    Args:
        strategy_config: Base strategy configuration
        param_overrides: Parameter overrides (dot-notation keys)

    Returns:
        Modified strategy configuration

    Example:
        >>> overrides = {"risk_management.stop_loss.percentage": 5.0}
        >>> new_config = apply_param_overrides(base_config, overrides)
    """
    # Use config_utils.apply_param_overrides for the actual work
    modified_config_dict = apply_config_overrides(
        strategy_config.raw_config,
        param_overrides,
    )

    # Create new StrategyConfig with modified dict
    return StrategyConfig(
        name=strategy_config.name,
        description=strategy_config.description,
        version=strategy_config.version,
        raw_config=modified_config_dict,
    )


def run_single_experiment_run(
    session: Session,
    experiment_id: int,
    run_index: int,
    strategy_config: StrategyConfig,
    param_overrides: Dict[str, Any],
    train_start: str,
    train_end: str,
    test_start: str | None,
    test_end: str | None,
    symbols: List[str],
) -> ExperimentRun:
    """
    Run single experiment with given parameters.

    Args:
        session: Database session
        experiment_id: Parent experiment ID
        run_index: Run index
        strategy_config: Base strategy config
        param_overrides: Parameter overrides
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date (optional)
        test_end: Test end date (optional)
        symbols: Symbols to test

    Returns:
        ExperimentRun with results
    """
    logger.info(f"Running experiment {experiment_id} run {run_index}")
    logger.debug(f"Parameters: {param_overrides}")

    # Apply overrides
    modified_config = apply_param_overrides(strategy_config, param_overrides)

    # Create experiment run
    exp_run = ExperimentRun(
        experiment_id=experiment_id,
        run_index=run_index,
        param_values_json=json.dumps(param_overrides),
        train_start=datetime.strptime(train_start, "%Y-%m-%d").date(),
        train_end=datetime.strptime(train_end, "%Y-%m-%d").date(),
        test_start=datetime.strptime(test_start, "%Y-%m-%d").date() if test_start else None,
        test_end=datetime.strptime(test_end, "%Y-%m-%d").date() if test_end else None,
        status="running",
    )

    session.add(exp_run)
    session.commit()

    try:
        # Use first symbol (or could aggregate across all symbols)
        symbol = symbols[0] if symbols else modified_config.get_symbols()[0]

        # Parse dates
        from datetime import datetime as dt
        train_start_date = dt.strptime(train_start, "%Y-%m-%d").date()
        train_end_date = dt.strptime(train_end, "%Y-%m-%d").date()
        test_start_date = dt.strptime(test_start, "%Y-%m-%d").date() if test_start else None
        test_end_date = dt.strptime(test_end, "%Y-%m-%d").date() if test_end else None

        # Run train/test backtest using runner
        metrics = run_train_test_backtest(
            session=session,
            symbol=symbol,
            strategy_cfg=modified_config.raw_config,
            train_start=train_start_date,
            train_end=train_end_date,
            test_start=test_start_date,
            test_end=test_end_date,
        )

        # Extract train metrics
        train_metrics = metrics["train"]
        exp_run.train_cagr = train_metrics.cagr
        exp_run.train_sharpe = train_metrics.sharpe
        exp_run.train_max_dd = train_metrics.max_dd
        exp_run.train_win_rate = train_metrics.win_rate
        exp_run.train_total_trades = train_metrics.total_trades

        # Extract test metrics
        test_metrics = metrics["test"]
        exp_run.test_cagr = test_metrics.cagr
        exp_run.test_sharpe = test_metrics.sharpe
        exp_run.test_max_dd = test_metrics.max_dd
        exp_run.test_win_rate = test_metrics.win_rate
        exp_run.test_total_trades = test_metrics.total_trades

        # Mark as done
        exp_run.status = "done"
        logger.info(
            f"Run {run_index} complete: "
            f"train_sharpe={exp_run.train_sharpe:.2f}, "
            f"test_sharpe={exp_run.test_sharpe:.2f if exp_run.test_sharpe else 'N/A'}"
        )

    except Exception as e:
        logger.error(f"Run {run_index} failed: {e}", exc_info=True)
        exp_run.status = "failed"
        exp_run.error_message = str(e)

    session.commit()
    return exp_run


def run_grid_search(
    session: Session,
    experiment_name: str,
    strategy_path: str | Path,
    grid_path: str | Path,
    train_start_override: Optional[date] = None,
    train_end_override: Optional[date] = None,
    test_start_override: Optional[date] = None,
    test_end_override: Optional[date] = None,
) -> Experiment:
    """
    Run grid search experiment.

    Args:
        session: Database session
        experiment_name: Name for this experiment
        strategy_path: Path to base strategy YAML
        grid_path: Path to parameter grid YAML
        train_start_override: Optional train start date (overrides grid YAML)
        train_end_override: Optional train end date (overrides grid YAML)
        test_start_override: Optional test start date (overrides grid YAML)
        test_end_override: Optional test end date (overrides grid YAML)

    Returns:
        Experiment with all runs

    Example:
        >>> with get_session() as session:
        ...     exp = run_grid_search(
        ...         session,
        ...         "baseline_v1_grid_search",
        ...         "strategies/baseline_v1.yaml",
        ...         "grids/baseline_grid.yaml",
        ...         train_start_override=date(2018, 1, 1),
        ...         train_end_override=date(2022, 12, 31),
        ...     )
        ...     print(f"Completed {len(exp.runs)} runs")
    """
    logger.info(f"Starting grid search experiment: {experiment_name}")

    # Load strategy and grid
    strategy_config = load_strategy(strategy_path)
    param_grid = load_param_grid(grid_path)

    # Also load raw YAML for metadata (split, symbols, etc.)
    import yaml
    with open(grid_path, "r") as f:
        grid_metadata = yaml.safe_load(f)

    logger.info(f"Base strategy: {strategy_config.name} v{strategy_config.version}")
    logger.info(f"Parameter grid: {grid_metadata.get('name', 'unnamed')}")

    # Create experiment
    experiment = Experiment(
        name=experiment_name,
        description=grid_metadata.get("description", "Grid search experiment"),
        base_strategy_name=strategy_config.name,
        param_grid_path=str(grid_path),
    )

    session.add(experiment)
    session.commit()

    # Get split dates from grid YAML
    split = grid_metadata.get("split", {})
    train_start = split.get("train_start", "2017-01-01")
    train_end = split.get("train_end", "2022-12-31")
    test_start = split.get("test_start")
    test_end = split.get("test_end")

    # Apply date overrides if provided
    if train_start_override:
        train_start = train_start_override.strftime("%Y-%m-%d")
        logger.info(f"Overriding train_start: {train_start}")
    if train_end_override:
        train_end = train_end_override.strftime("%Y-%m-%d")
        logger.info(f"Overriding train_end: {train_end}")
    if test_start_override:
        test_start = test_start_override.strftime("%Y-%m-%d")
        logger.info(f"Overriding test_start: {test_start}")
    if test_end_override:
        test_end = test_end_override.strftime("%Y-%m-%d")
        logger.info(f"Overriding test_end: {test_end}")

    # Get symbols
    symbols = grid_metadata.get("symbols", [])
    if not symbols:
        symbols = strategy_config.get_symbols()

    # Generate combinations using ParamGrid
    combinations = generate_param_combinations(param_grid)

    logger.info(f"Running {len(combinations)} parameter combinations")

    # Run each combination
    for run_index, param_overrides in enumerate(combinations, start=1):
        logger.info(f"Run {run_index}/{len(combinations)}")

        exp_run = run_single_experiment_run(
            session=session,
            experiment_id=experiment.id,
            run_index=run_index,
            strategy_config=strategy_config,
            param_overrides=param_overrides,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            symbols=symbols,
        )

    logger.info(f"Grid search complete: {experiment_name}")

    # Refresh to get all runs
    session.refresh(experiment)
    return experiment


def run_random_search(
    session: Session,
    experiment_name: str,
    strategy_path: str | Path,
    grid_path: str | Path,
    train_start_override: Optional[date] = None,
    train_end_override: Optional[date] = None,
    test_start_override: Optional[date] = None,
    test_end_override: Optional[date] = None,
) -> Experiment:
    """
    Run random search experiment.

    Args:
        session: Database session
        experiment_name: Name for this experiment
        strategy_path: Path to base strategy YAML
        grid_path: Path to parameter grid YAML
        train_start_override: Optional train start date (overrides grid YAML)
        train_end_override: Optional train end date (overrides grid YAML)
        test_start_override: Optional test start date (overrides grid YAML)
        test_end_override: Optional test end date (overrides grid YAML)

    Returns:
        Experiment with all runs
    """
    # Random search is same as grid search with search_method="random"
    return run_grid_search(
        session,
        experiment_name,
        strategy_path,
        grid_path,
        train_start_override=train_start_override,
        train_end_override=train_end_override,
        test_start_override=test_start_override,
        test_end_override=test_end_override,
    )
