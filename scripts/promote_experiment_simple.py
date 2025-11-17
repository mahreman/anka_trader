#!/usr/bin/env python
"""
Simple script to promote experiment to strategy (low-level version).

This is a simpler, more transparent version that manually selects the best run
and applies parameter overrides. Good for learning and customization.

Example:
    python scripts/promote_experiment_simple.py --experiment-id 3 \\
        --output-path strategies/baseline_v2.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.data.schema_experiments import Experiment, ExperimentRun
from otonom_trader.experiments.config_utils import (
    load_strategy_config,
    apply_param_overrides,
    save_strategy_config,
)


def select_best_run(session: Session, experiment_id: int) -> ExperimentRun:
    """
    Select best run from experiment.

    Sorts by test_sharpe first, then test_cagr.

    Args:
        session: Database session
        experiment_id: Experiment ID

    Returns:
        Best ExperimentRun

    Raises:
        SystemExit: If no successful runs found
    """
    runs = (
        session.query(ExperimentRun)
        .filter(ExperimentRun.experiment_id == experiment_id)
        .filter(ExperimentRun.status == "done")
        .all()
    )

    if not runs:
        raise SystemExit(f"❌ No 'done' runs for experiment {experiment_id}")

    # Sort by test_sharpe, then test_cagr
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            r.test_sharpe if r.test_sharpe is not None else -999.0,
            r.test_cagr if r.test_cagr is not None else -999.0,
        ),
        reverse=True,
    )

    return runs_sorted[0]


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple experiment promotion (low-level)",
    )

    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="Experiment ID",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="strategies/baseline_v2.yaml",
        help="Output strategy path",
    )

    args = parser.parse_args()

    # Initialize database
    engine = get_engine()
    init_db(engine)

    with get_session() as session:  # type: Session
        # Get experiment
        exp = session.query(Experiment).get(args.experiment_id)
        if exp is None:
            raise SystemExit(f"❌ Experiment {args.experiment_id} not found")

        print(f"📊 Experiment: {exp.experiment_name}")
        print(f"   Base strategy: {exp.base_strategy_name}")
        print(f"   Total runs: {len(exp.runs)}")

        # Select best run
        best_run = select_best_run(session, exp.id)

        print(f"\n✅ Best run selected:")
        print(f"   Run index: {best_run.run_index}")
        print(f"   Test Sharpe: {best_run.test_sharpe:.3f}")
        print(f"   Test CAGR: {best_run.test_cagr:.1f}%")
        print(f"   Test MaxDD: {best_run.test_max_dd:.1f}%")
        print(f"   Test Win Rate: {best_run.test_win_rate:.1f}%")

        # Extract parameters
        params = json.loads(best_run.param_values_json)

        print(f"\n🔧 Parameter overrides:")
        for key, value in params.items():
            print(f"   {key}: {value}")

        # Load base config
        base_cfg = load_strategy_config(exp.base_strategy_name)

        # Apply overrides
        new_cfg = apply_param_overrides(base_cfg, params)

        # Update strategy name
        output_path = Path(args.output_path)
        new_name = output_path.stem
        new_cfg["name"] = new_name

        # Add metadata
        if "metadata" not in new_cfg:
            new_cfg["metadata"] = {}

        new_cfg["metadata"]["promoted_from_experiment"] = exp.id
        new_cfg["metadata"]["promoted_from_run"] = best_run.run_index
        new_cfg["metadata"]["test_sharpe"] = float(best_run.test_sharpe)
        new_cfg["metadata"]["test_cagr"] = float(best_run.test_cagr)
        new_cfg["metadata"]["param_overrides"] = params

        # Save new strategy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_strategy_config(new_cfg, str(output_path))

        print(f"\n📄 Created new strategy: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Review: cat {output_path}")
        print(f"  2. Backtest: python -m otonom_trader.cli backtest ...")
        print(f"  3. Deploy to production if results are good")


if __name__ == "__main__":
    main()
