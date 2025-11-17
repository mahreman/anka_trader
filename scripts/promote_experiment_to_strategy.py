#!/usr/bin/env python
"""
Promote best experiment run to new strategy version.

This script automates the promotion ritual:
1. Select best run from experiment (by test Sharpe + CAGR)
2. Apply parameter overrides to base strategy
3. Bump version (major or minor)
4. Save new strategy YAML
5. Document in STRATEGY_LOG.md

Usage:
    # Auto-select best run and promote to v1.1
    python scripts/promote_experiment_to_strategy.py \
        --experiment-id 3 \
        --output-path strategies/baseline_v1.1.yaml \
        --new-version 1.1.0

    # Manually specify run index
    python scripts/promote_experiment_to_strategy.py \
        --experiment-id 3 \
        --run-index 42 \
        --output-path strategies/baseline_v1.1.yaml \
        --new-version 1.1.0

    # Auto-detect version bump (minor for param changes, major for structural)
    python scripts/promote_experiment_to_strategy.py \
        --experiment-id 3 \
        --auto-version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add otonom_trader to path
sys.path.insert(0, str(Path(__file__).parent.parent / "otonom_trader"))

from otonom_trader.data import get_session
from otonom_trader.data.schema_experiments import Experiment, ExperimentRun
from otonom_trader.strategy.config import load_strategy_config
from otonom_trader.strategy.versioning import (
    StrategyVersion,
    PromotionRecord,
    append_to_strategy_log,
    compare_strategy_configs,
    detect_promotion_type,
)


def select_best_run(
    experiment: Experiment,
    run_index: int | None = None,
) -> ExperimentRun:
    """
    Select best run from experiment.

    Args:
        experiment: Experiment object
        run_index: Specific run index (optional)

    Returns:
        Best ExperimentRun object
    """
    if run_index is not None:
        # Find specific run
        for run in experiment.runs:
            if run.run_index == run_index:
                return run
        raise ValueError(f"Run index {run_index} not found in experiment {experiment.id}")

    # Find best run by test_sharpe
    successful_runs = [r for r in experiment.runs if r.status == "done"]
    if not successful_runs:
        raise ValueError(f"No successful runs in experiment {experiment.id}")

    sorted_runs = sorted(
        successful_runs,
        key=lambda r: (r.test_sharpe or 0.0, r.test_cagr or 0.0),
        reverse=True,
    )

    return sorted_runs[0]


def apply_param_overrides(base_config: dict, params: dict) -> dict:
    """
    Apply parameter overrides to base config.

    Args:
        base_config: Base strategy config dict
        params: Parameter overrides (dot notation keys)

    Returns:
        New config dict with overrides applied
    """
    new_config = yaml.safe_load(yaml.dump(base_config))  # Deep copy

    for param_path, value in params.items():
        keys = param_path.split(".")
        target = new_config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Set value
        target[keys[-1]] = value

    return new_config


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Promote best experiment run to new strategy version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="Experiment ID to promote from",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for new strategy (e.g., strategies/baseline_v1.1.yaml)",
    )

    # Optional arguments
    parser.add_argument(
        "--run-index",
        type=int,
        help="Specific run index to promote (default: auto-select best)",
    )

    parser.add_argument(
        "--new-version",
        type=str,
        help="New version string (e.g., '1.1.0', '2.0.0')",
    )

    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="Auto-detect version bump (minor for params, major for structure)",
    )

    parser.add_argument(
        "--promotion-type",
        choices=["major", "minor"],
        help="Promotion type (major or minor)",
    )

    parser.add_argument(
        "--rationale",
        type=str,
        help="Rationale for promotion (default: auto-generated)",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="STRATEGY_LOG.md",
        help="Path to strategy log (default: STRATEGY_LOG.md)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.output_path and not args.auto_version:
        parser.error("Either --output-path or --auto-version must be specified")

    if args.new_version and args.auto_version:
        parser.error("Cannot use both --new-version and --auto-version")

    # Get experiment from database
    with get_session() as session:
        experiment = session.query(Experiment).filter_by(id=args.experiment_id).first()
        if not experiment:
            print(f"✗ Experiment {args.experiment_id} not found")
            return 1

        print(f"Experiment: {experiment.name}")
        print(f"Base strategy: {experiment.base_strategy_name}")
        print()

        # Select best run
        try:
            best_run = select_best_run(experiment, args.run_index)
        except ValueError as e:
            print(f"✗ {e}")
            return 1

        print(f"Selected run: #{best_run.run_index}")
        print(f"  Test Sharpe: {best_run.test_sharpe:.2f}")
        print(f"  Test CAGR: {best_run.test_cagr:.2f}%")
        print(f"  Test Max DD: {best_run.test_max_dd:.2f}%")
        print(f"  Test Win Rate: {best_run.test_win_rate:.1%}")
        print(f"  Test Trades: {best_run.test_total_trades}")
        print()

        # Load base strategy
        try:
            base_strategy_path = Path(experiment.base_strategy_name)
            base_config = load_strategy_config(base_strategy_path)
        except Exception as e:
            print(f"✗ Failed to load base strategy: {e}")
            return 1

        # Parse current version
        current_version = StrategyVersion.from_path(base_strategy_path)
        if not current_version:
            print(f"⚠ Could not parse version from {base_strategy_path}, assuming v1.0")
            current_version = StrategyVersion(name=base_config.name, major=1, minor=0)

        print(f"Current version: {current_version}")

        # Load base config as dict
        with open(base_strategy_path, "r") as f:
            old_config_dict = yaml.safe_load(f)

        # Apply parameter overrides
        params = json.loads(best_run.param_values_json)
        new_config_dict = apply_param_overrides(old_config_dict, params)

        # Detect promotion type
        if args.promotion_type:
            promotion_type = args.promotion_type
        else:
            promotion_type = detect_promotion_type(old_config_dict, new_config_dict)

        print(f"Promotion type: {promotion_type}")

        # Determine new version
        if args.new_version:
            new_version_str = args.new_version
            # Parse to get major/minor
            parts = new_version_str.split(".")
            new_version = StrategyVersion(
                name=current_version.name,
                major=int(parts[0]),
                minor=int(parts[1]) if len(parts) > 1 else 0,
            )
        else:
            if promotion_type == "major":
                new_version = current_version.bump_major()
            else:
                new_version = current_version.bump_minor()

        print(f"New version: {new_version}")

        # Update version in config
        new_config_dict["version"] = new_version.full_version

        # Determine output path
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = Path("strategies") / new_version.file_name

        print(f"Output path: {output_path}")
        print()

        # Compare configs
        changes = compare_strategy_configs(old_config_dict, new_config_dict)
        print("Changes:")
        print(changes)
        print()

        # Generate rationale
        if args.rationale:
            rationale = args.rationale
        else:
            rationale = (
                f"Promoted from experiment #{args.experiment_id} ({experiment.name}). "
                f"Best run #{best_run.run_index} achieved test Sharpe={best_run.test_sharpe:.2f}, "
                f"CAGR={best_run.test_cagr:.2f}%, MaxDD={best_run.test_max_dd:.2f}%."
            )

        # Create promotion record
        record = PromotionRecord(
            from_version=current_version,
            to_version=new_version,
            promotion_type=promotion_type,
            experiment_id=args.experiment_id,
            test_sharpe=best_run.test_sharpe or 0.0,
            test_cagr=best_run.test_cagr or 0.0,
            test_max_dd=best_run.test_max_dd or 0.0,
            changes=changes,
            rationale=rationale,
        )

        # Dry run or execute
        if args.dry_run:
            print("DRY RUN - Would perform:")
            print(f"  1. Save new strategy to: {output_path}")
            print(f"  2. Append to strategy log: {args.log_path}")
            print()
            print("Promotion record:")
            print(record.to_log_entry())
            return 0

        # Save new strategy
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                yaml.dump(new_config_dict, f, default_flow_style=False, sort_keys=False)
            print(f"✓ Saved new strategy: {output_path}")
        except Exception as e:
            print(f"✗ Failed to save strategy: {e}")
            return 1

        # Append to strategy log
        try:
            append_to_strategy_log(record, log_path=Path(args.log_path))
            print(f"✓ Updated strategy log: {args.log_path}")
        except Exception as e:
            print(f"✗ Failed to update log: {e}")
            return 1

        print()
        print("=" * 60)
        print("PROMOTION COMPLETE!")
        print("=" * 60)
        print()
        print("Next steps:")
        print(f"1. Review new strategy: {output_path}")
        print(f"2. Run full backtest:")
        print(f"   python -m otonom_trader.cli backtest run \\")
        print(f"     --strategy {output_path} \\")
        print(f"     --start 2018-01-01 --end 2024-12-31")
        print()
        print(f"3. Deploy to paper daemon (champion/challenger):")
        print(f"   python -m otonom_trader.cli daemon-once \\")
        print(f"     --strategy {output_path}")
        print()

        return 0


if __name__ == "__main__":
    sys.exit(main())
