#!/usr/bin/env python
"""
Grid search experiment runner.

Runs parameter grid search experiments and saves results to database.

Usage:
    python scripts/run_grid_search.py \
        --config experiments/param_sweep_baseline.yaml \
        --strategy strategies/baseline_v1.0.yaml

Example YAML config:
    name: "baseline_v1_param_sweep"
    description: "Parameter sweep for baseline v1"
    base_strategy: "strategies/baseline_v1.0.yaml"

    search_method: "grid"  # or "random"
    random_samples: 100  # if search_method == "random"

    split:
        train_start: "2018-01-01"
        train_end: "2021-12-31"
        test_start: "2022-01-01"
        test_end: "2024-12-31"

    parameters:
        risk.risk_pct:
            values: [0.5, 1.0, 1.5]
        risk.stop_loss_pct:
            values: [3.0, 5.0, 8.0]
        ensemble.analyst_weights.tech:
            values: [0.8, 1.0, 1.2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from itertools import product
from pathlib import Path
from random import sample
from typing import Any, Dict, Iterator, List

import yaml

# Add otonom_trader to path
sys.path.insert(0, str(Path(__file__).parent.parent / "otonom_trader"))

from otonom_trader.data import get_session
from otonom_trader.data.schema_experiments import Experiment, ExperimentRun
from otonom_trader.research.backtest_runner import run_backtest_for_strategy
from otonom_trader.strategy.config import load_strategy_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_experiment_yaml(path: str | Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _iter_grid(params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Generate all parameter combinations from grid.

    Args:
        params: Dict of parameter paths to value lists
            Example: {"risk.risk_pct": [0.5, 1.0], "risk.stop_loss_pct": [3, 5]}

    Yields:
        Dict of parameter combinations
            Example: {"risk.risk_pct": 0.5, "risk.stop_loss_pct": 3}
    """
    # Extract parameter names and values
    param_names = []
    param_values = []

    for param_path, param_spec in params.items():
        param_names.append(param_path)

        # Handle both formats:
        # 1. {"values": [0.5, 1.0, 1.5]}
        # 2. [0.5, 1.0, 1.5]
        if isinstance(param_spec, dict) and "values" in param_spec:
            values = param_spec["values"]
        elif isinstance(param_spec, list):
            values = param_spec
        else:
            raise ValueError(f"Invalid param spec for {param_path}: {param_spec}")

        param_values.append(values)

    # Generate Cartesian product
    for combination in product(*param_values):
        yield dict(zip(param_names, combination))


def _apply_param_overrides(base_config: Any, param_values: Dict[str, Any]) -> Any:
    """
    Apply parameter overrides to base strategy config.

    Args:
        base_config: Base StrategyConfig object
        param_values: Parameter overrides (dot notation)

    Returns:
        New StrategyConfig with overrides applied
    """
    # Convert to dict, apply overrides, reload
    config_dict = base_config.to_dict() if hasattr(base_config, "to_dict") else {}

    for param_path, value in param_values.items():
        keys = param_path.split(".")
        target = config_dict

        # Navigate to parent
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Set value
        target[keys[-1]] = value

    # Reload config from dict
    # For now, just update the base_config attributes
    # In production, you'd want to reload from YAML
    for param_path, value in param_values.items():
        keys = param_path.split(".")
        target = base_config

        for key in keys[:-1]:
            if hasattr(target, key):
                target = getattr(target, key)
            else:
                break

        # Set attribute
        if hasattr(target, keys[-1]):
            setattr(target, keys[-1], value)

    return base_config


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run grid search experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        help="Override base strategy path",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        help="Limit number of runs (for testing)",
    )

    args = parser.parse_args()

    # Load experiment config
    logger.info(f"Loading experiment config: {args.config}")
    exp_cfg = _load_experiment_yaml(args.config)

    # Extract config
    exp_name = exp_cfg["name"]
    exp_desc = exp_cfg.get("description", "")
    base_strategy_path = args.strategy or exp_cfg["base_strategy"]
    search_method = exp_cfg.get("search_method", "grid")
    random_samples = exp_cfg.get("random_samples", 100)

    # Train/test split
    split = exp_cfg["split"]
    train_start = date.fromisoformat(split["train_start"])
    train_end = date.fromisoformat(split["train_end"])
    test_start = date.fromisoformat(split["test_start"])
    test_end = date.fromisoformat(split["test_end"])

    # Parameters
    params = exp_cfg["parameters"]

    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Description: {exp_desc}")
    logger.info(f"Base strategy: {base_strategy_path}")
    logger.info(f"Search method: {search_method}")
    logger.info(f"Train: {train_start} to {train_end}")
    logger.info(f"Test: {test_start} to {test_end}")
    logger.info("=" * 60)

    # Load base strategy
    try:
        base_cfg = load_strategy_config(base_strategy_path)
    except Exception as e:
        logger.error(f"Failed to load base strategy: {e}")
        return 1

    # Generate parameter grid
    param_grid = list(_iter_grid(params))
    total_runs = len(param_grid)

    logger.info(f"Total parameter combinations: {total_runs}")

    # Apply random sampling if needed
    if search_method == "random" and total_runs > random_samples:
        logger.info(f"Using random sampling: {random_samples} runs")
        param_grid = sample(param_grid, random_samples)
        total_runs = len(param_grid)

    # Apply max-runs limit if specified
    if args.max_runs and total_runs > args.max_runs:
        logger.info(f"Limiting to {args.max_runs} runs (--max-runs)")
        param_grid = param_grid[: args.max_runs]
        total_runs = len(param_grid)

    if args.dry_run:
        logger.info("\nDRY RUN - Would execute the following:")
        logger.info(f"  Experiment: {exp_name}")
        logger.info(f"  Total runs: {total_runs}")
        logger.info("\nFirst 5 parameter combinations:")
        for i, params_combo in enumerate(param_grid[:5]):
            logger.info(f"  {i+1}. {params_combo}")
        return 0

    # Create experiment in database
    with get_session() as session:
        # Check if experiment already exists
        existing = session.query(Experiment).filter_by(name=exp_name).first()
        if existing:
            logger.warning(f"Experiment '{exp_name}' already exists (ID={existing.id})")
            logger.info("Appending new runs to existing experiment")
            experiment = existing
            start_run_index = len(existing.runs)
        else:
            # Create new experiment
            experiment = Experiment(
                name=exp_name,
                description=exp_desc,
                base_strategy_name=base_cfg.name,
                param_grid_path=str(args.config),
            )
            session.add(experiment)
            session.commit()
            start_run_index = 0
            logger.info(f"Created experiment: ID={experiment.id}")

        # Run grid search
        logger.info(f"\nStarting grid search: {total_runs} runs")
        logger.info("=" * 60)

        for idx, param_values in enumerate(param_grid):
            run_index = start_run_index + idx

            logger.info(f"\nRun {run_index + 1}/{start_run_index + total_runs}")
            logger.info(f"Parameters: {param_values}")

            # Create experiment run record
            run = ExperimentRun(
                experiment_id=experiment.id,
                run_index=run_index,
                param_values_json=json.dumps(param_values),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                status="running",
            )
            session.add(run)
            session.commit()

            try:
                # Apply parameter overrides to base config
                train_cfg = _apply_param_overrides(base_cfg, param_values)

                # Run training backtest
                logger.info("  Running training backtest...")
                train_report = run_backtest_for_strategy(
                    strategy_cfg=train_cfg,
                    start_date=train_start,
                    end_date=train_end,
                )

                # Update run with training metrics
                run.train_cagr = train_report.metrics.cagr
                run.train_sharpe = train_report.metrics.sharpe
                run.train_max_dd = train_report.metrics.max_drawdown
                run.train_win_rate = train_report.metrics.win_rate
                run.train_total_trades = train_report.metrics.total_trades

                logger.info(
                    f"  Train results: Sharpe={train_report.metrics.sharpe:.2f}, "
                    f"CAGR={train_report.metrics.cagr:.2f}%, "
                    f"Trades={train_report.metrics.total_trades}"
                )

                # Run test backtest
                logger.info("  Running test backtest...")
                test_cfg = _apply_param_overrides(base_cfg, param_values)

                test_report = run_backtest_for_strategy(
                    strategy_cfg=test_cfg,
                    start_date=test_start,
                    end_date=test_end,
                )

                # Update run with test metrics
                run.test_cagr = test_report.metrics.cagr
                run.test_sharpe = test_report.metrics.sharpe
                run.test_max_dd = test_report.metrics.max_drawdown
                run.test_win_rate = test_report.metrics.win_rate
                run.test_total_trades = test_report.metrics.total_trades

                logger.info(
                    f"  Test results: Sharpe={test_report.metrics.sharpe:.2f}, "
                    f"CAGR={test_report.metrics.cagr:.2f}%, "
                    f"Trades={test_report.metrics.total_trades}"
                )

                # Mark as done
                run.status = "done"

            except Exception as e:
                logger.error(f"  Run failed: {e}")
                run.status = "failed"
                run.error_message = str(e)

            finally:
                session.commit()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("GRID SEARCH COMPLETE!")
        logger.info("=" * 60)

        # Query results
        successful_runs = (
            session.query(ExperimentRun)
            .filter_by(experiment_id=experiment.id, status="done")
            .all()
        )

        failed_runs = (
            session.query(ExperimentRun)
            .filter_by(experiment_id=experiment.id, status="failed")
            .all()
        )

        logger.info(f"\nExperiment ID: {experiment.id}")
        logger.info(f"Successful runs: {len(successful_runs)}")
        logger.info(f"Failed runs: {len(failed_runs)}")

        if successful_runs:
            # Sort by test Sharpe
            sorted_runs = sorted(
                successful_runs,
                key=lambda r: r.test_sharpe or 0.0,
                reverse=True,
            )

            logger.info("\nTop 5 runs by test Sharpe:")
            for i, run in enumerate(sorted_runs[:5]):
                logger.info(
                    f"{i+1}. Run #{run.run_index}: "
                    f"Test Sharpe={run.test_sharpe:.2f}, "
                    f"CAGR={run.test_cagr:.2f}%, "
                    f"MaxDD={run.test_max_dd:.2f}%"
                )
                params = json.loads(run.param_values_json)
                logger.info(f"   Params: {params}")

        logger.info(f"\nNext steps:")
        logger.info(f"1. Analyze results:")
        logger.info(f"   SELECT * FROM experiment_runs WHERE experiment_id = {experiment.id}")
        logger.info(f"2. Promote best run to new strategy:")
        logger.info(f"   python scripts/promote_experiment_to_strategy.py \\")
        logger.info(f"     --experiment-id {experiment.id} \\")
        logger.info(f"     --output-path strategies/{base_cfg.name}_v1.1.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
