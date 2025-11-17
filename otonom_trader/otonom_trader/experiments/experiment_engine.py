"""
Experiment engine for systematic parameter optimization.

Provides grid search and random search capabilities for strategy optimization.
"""

from __future__ import annotations

import json
import logging
import random
import yaml
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from ..config import load_strategy, StrategyConfig
from ..data import Experiment, ExperimentRun
from ..eval.portfolio_backtest import run_backtest
from ..eval.performance_report import calculate_metrics

logger = logging.getLogger(__name__)


def load_param_grid(path: str | Path) -> Dict[str, Any]:
    """
    Load parameter grid from YAML file.

    Args:
        path: Path to grid YAML file

    Returns:
        Dictionary with grid configuration

    Example:
        >>> grid = load_param_grid("grids/baseline_grid.yaml")
        >>> print(grid["parameters"])
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parameter grid file not found: {path}")

    with open(path, "r") as f:
        grid = yaml.safe_load(f)

    if not isinstance(grid, dict):
        raise ValueError(f"Invalid grid file: expected dict, got {type(grid)}")

    return grid


def generate_param_combinations(
    param_grid: Dict[str, Any],
    search_method: str = "grid",
) -> List[Dict[str, Any]]:
    """
    Generate parameter combinations from grid.

    Args:
        param_grid: Parameter grid dictionary
        search_method: "grid" (exhaustive) or "random" (sample)

    Returns:
        List of parameter combination dictionaries

    Example:
        >>> grid = {"risk_pct": {"values": [0.5, 1.0]}, "sl": {"values": [3, 5]}}
        >>> combos = generate_param_combinations({"parameters": grid})
        >>> len(combos)  # 2 * 2 = 4 combinations
        4
    """
    parameters = param_grid.get("parameters", {})

    if not parameters:
        logger.warning("No parameters defined in grid")
        return [{}]

    # Extract parameter names and values
    param_names = []
    param_values = []

    for param_name, param_config in parameters.items():
        if "values" in param_config:
            param_names.append(param_name)
            param_values.append(param_config["values"])

    if not param_names:
        return [{}]

    # Generate combinations
    if search_method == "grid":
        # Exhaustive grid search
        combinations = []
        for combo_values in product(*param_values):
            combo = dict(zip(param_names, combo_values))
            combinations.append(combo)

        logger.info(f"Generated {len(combinations)} parameter combinations (grid search)")

    elif search_method == "random":
        # Random search
        n_samples = param_grid.get("random_samples", 50)
        combinations = []

        for _ in range(n_samples):
            combo = {}
            for param_name, values in zip(param_names, param_values):
                combo[param_name] = random.choice(values)
            combinations.append(combo)

        logger.info(f"Generated {n_samples} parameter combinations (random search)")

    else:
        raise ValueError(f"Unknown search method: {search_method}")

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
    # Deep copy to avoid modifying original
    config_dict = deepcopy(strategy_config.raw_config)

    # Apply overrides
    for param_path, value in param_overrides.items():
        keys = param_path.split(".")
        current = config_dict

        # Navigate to target location
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set value
        current[keys[-1]] = value

    # Create new config
    return StrategyConfig(
        name=strategy_config.name,
        description=strategy_config.description,
        version=strategy_config.version,
        raw_config=config_dict,
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

        # Run training backtest
        logger.info(f"Training backtest: {symbol} {train_start} to {train_end}")
        train_result = run_backtest(
            session=session,
            symbol=symbol,
            start_date=train_start,
            end_date=train_end,
            initial_cash=modified_config.get_initial_capital(),
            risk_per_trade=modified_config.get_risk_per_trade_pct() / 100,
            use_ensemble=modified_config.get("ensemble.enabled", True),
        )

        if train_result and train_result.get("equity_curve"):
            train_metrics = calculate_metrics(
                train_result["equity_curve"],
                train_result.get("trades", []),
                modified_config.get_initial_capital(),
            )

            exp_run.train_cagr = train_metrics["cagr"]
            exp_run.train_sharpe = train_metrics["sharpe_ratio"]
            exp_run.train_max_dd = train_metrics["max_drawdown"]
            exp_run.train_win_rate = train_metrics["win_rate"]
            exp_run.train_total_trades = train_metrics["total_trades"]

        # Run test backtest (if specified)
        if test_start and test_end:
            logger.info(f"Test backtest: {symbol} {test_start} to {test_end}")
            test_result = run_backtest(
                session=session,
                symbol=symbol,
                start_date=test_start,
                end_date=test_end,
                initial_cash=modified_config.get_initial_capital(),
                risk_per_trade=modified_config.get_risk_per_trade_pct() / 100,
                use_ensemble=modified_config.get("ensemble.enabled", True),
            )

            if test_result and test_result.get("equity_curve"):
                test_metrics = calculate_metrics(
                    test_result["equity_curve"],
                    test_result.get("trades", []),
                    modified_config.get_initial_capital(),
                )

                exp_run.test_cagr = test_metrics["cagr"]
                exp_run.test_sharpe = test_metrics["sharpe_ratio"]
                exp_run.test_max_dd = test_metrics["max_drawdown"]
                exp_run.test_win_rate = test_metrics["win_rate"]
                exp_run.test_total_trades = test_metrics["total_trades"]

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
) -> Experiment:
    """
    Run grid search experiment.

    Args:
        session: Database session
        experiment_name: Name for this experiment
        strategy_path: Path to base strategy YAML
        grid_path: Path to parameter grid YAML

    Returns:
        Experiment with all runs

    Example:
        >>> with get_session() as session:
        ...     exp = run_grid_search(
        ...         session,
        ...         "baseline_v1_grid_search",
        ...         "strategies/baseline_v1.yaml",
        ...         "grids/baseline_grid.yaml"
        ...     )
        ...     print(f"Completed {len(exp.runs)} runs")
    """
    logger.info(f"Starting grid search experiment: {experiment_name}")

    # Load strategy and grid
    strategy_config = load_strategy(strategy_path)
    param_grid = load_param_grid(grid_path)

    logger.info(f"Base strategy: {strategy_config.name} v{strategy_config.version}")
    logger.info(f"Parameter grid: {param_grid.get('name', 'unnamed')}")

    # Create experiment
    experiment = Experiment(
        name=experiment_name,
        description=param_grid.get("description", "Grid search experiment"),
        base_strategy_name=strategy_config.name,
        param_grid_path=str(grid_path),
    )

    session.add(experiment)
    session.commit()

    # Get split dates
    split = param_grid.get("split", {})
    train_start = split.get("train_start", "2017-01-01")
    train_end = split.get("train_end", "2022-12-31")
    test_start = split.get("test_start")
    test_end = split.get("test_end")

    # Get symbols
    symbols = param_grid.get("symbols", [])
    if not symbols:
        symbols = strategy_config.get_symbols()

    # Generate combinations
    search_method = param_grid.get("search_method", "grid")
    combinations = generate_param_combinations(param_grid, search_method)

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
) -> Experiment:
    """
    Run random search experiment.

    Args:
        session: Database session
        experiment_name: Name for this experiment
        strategy_path: Path to base strategy YAML
        grid_path: Path to parameter grid YAML

    Returns:
        Experiment with all runs
    """
    # Random search is same as grid search with search_method="random"
    return run_grid_search(session, experiment_name, strategy_path, grid_path)
