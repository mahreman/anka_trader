#!/usr/bin/env python
"""
Promote the best experiment run to a new strategy configuration.

This script:
1. Finds the best run from an experiment (by test Sharpe + CAGR)
2. Extracts the parameter values from that run
3. Applies those parameters to the base strategy config
4. Saves the result as a new strategy YAML file (e.g., baseline_v2.yaml)

Usage:
    # After running a grid search experiment
    python scripts/promote_experiment_to_strategy.py \
        --experiment-id 3 \
        --output-path strategies/baseline_v2.yaml

    # With custom ranking metric
    python scripts/promote_experiment_to_strategy.py \
        --experiment-id 3 \
        --output-path strategies/baseline_v2.yaml \
        --rank-by sharpe  # or "cagr", "sharpe_cagr" (default)

Example workflow:
    1. Run grid search:
       python -m otonom_trader.cli experiments grid-search \\
           --experiment-name baseline_v1_grid \\
           --strategy-path strategies/baseline_v1.yaml \\
           --grid-path grids/baseline_grid.yaml

    2. Promote best run:
       python scripts/promote_experiment_to_strategy.py \\
           --experiment-id 1 \\
           --output-path strategies/baseline_v2.yaml

    3. Use new strategy:
       python -m otonom_trader.cli backtest \\
           --strategy strategies/baseline_v2.yaml
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.data.schema_experiments import Experiment, ExperimentRun
from otonom_trader.experiments.config_utils import (
    load_strategy_config,
    apply_param_overrides,
    save_strategy_config,
)


def select_best_run(
    session: Session,
    experiment_id: int,
    rank_by: str = "sharpe_cagr",
) -> ExperimentRun:
    """
    Select the best run from an experiment.

    Args:
        session: Database session
        experiment_id: Experiment ID
        rank_by: Ranking metric ("sharpe", "cagr", "sharpe_cagr")

    Returns:
        Best ExperimentRun

    Raises:
        SystemExit: If no completed runs found
    """
    runs = (
        session.query(ExperimentRun)
        .filter(ExperimentRun.experiment_id == experiment_id)
        .filter(ExperimentRun.status == "done")
        .all()
    )

    if not runs:
        raise SystemExit(
            f"❌ No completed runs found for experiment {experiment_id}.\n"
            f"   Check experiment status with:\n"
            f"   python -m otonom_trader.cli experiments show --experiment-id {experiment_id}"
        )

    print(f"Found {len(runs)} completed runs for experiment {experiment_id}")

    # Rank runs by metric
    if rank_by == "sharpe":
        # Sort by test Sharpe only
        runs_sorted = sorted(
            runs,
            key=lambda r: (r.test_sharpe or -999.0),
            reverse=True,
        )
    elif rank_by == "cagr":
        # Sort by test CAGR only
        runs_sorted = sorted(
            runs,
            key=lambda r: (r.test_cagr or -999.0),
            reverse=True,
        )
    else:  # sharpe_cagr (default)
        # Sort by both (Sharpe primary, CAGR secondary)
        runs_sorted = sorted(
            runs,
            key=lambda r: (
                (r.test_sharpe or -999.0),
                (r.test_cagr or -999.0),
            ),
            reverse=True,
        )

    return runs_sorted[0]


def promote_to_strategy(
    experiment_id: int,
    output_path: str,
    rank_by: str = "sharpe_cagr",
) -> None:
    """
    Promote best experiment run to a new strategy configuration.

    Args:
        experiment_id: Experiment ID
        output_path: Output path for new strategy YAML
        rank_by: Ranking metric for selecting best run
    """
    engine = get_engine()
    init_db(engine)

    with next(get_session()) as session:
        # Get experiment
        exp = session.query(Experiment).get(experiment_id)
        if exp is None:
            raise SystemExit(f"❌ Experiment {experiment_id} not found in database.")

        print(f"\n{'=' * 80}")
        print(f"PROMOTING EXPERIMENT TO STRATEGY")
        print(f"{'=' * 80}")
        print(f"Experiment: {exp.name} (ID: {exp.id})")
        print(f"Base Strategy: {exp.base_strategy_name}")
        print(f"Description: {exp.description or 'N/A'}")
        print(f"{'=' * 80}\n")

        # Select best run
        best_run = select_best_run(session, exp.id, rank_by)

        print(f"✅ Best run selected:")
        print(f"   Run ID: {best_run.id}")
        print(f"   Run Index: {best_run.run_index}")
        print(f"   Status: {best_run.status}")
        print()

        # Display metrics
        print(f"📊 Training Metrics:")
        print(f"   Sharpe: {best_run.train_sharpe:.3f}" if best_run.train_sharpe else "   Sharpe: N/A")
        print(f"   CAGR: {best_run.train_cagr:.2%}" if best_run.train_cagr else "   CAGR: N/A")
        print(f"   Max DD: {best_run.train_max_dd:.2%}" if best_run.train_max_dd else "   Max DD: N/A")
        print(f"   Win Rate: {best_run.train_win_rate:.2%}" if best_run.train_win_rate else "   Win Rate: N/A")
        print(f"   Total Trades: {best_run.train_total_trades}" if best_run.train_total_trades else "   Total Trades: N/A")
        print()

        print(f"📊 Test Metrics:")
        print(f"   Sharpe: {best_run.test_sharpe:.3f}" if best_run.test_sharpe else "   Sharpe: N/A")
        print(f"   CAGR: {best_run.test_cagr:.2%}" if best_run.test_cagr else "   CAGR: N/A")
        print(f"   Max DD: {best_run.test_max_dd:.2%}" if best_run.test_max_dd else "   Max DD: N/A")
        print(f"   Win Rate: {best_run.test_win_rate:.2%}" if best_run.test_win_rate else "   Win Rate: N/A")
        print(f"   Total Trades: {best_run.test_total_trades}" if best_run.test_total_trades else "   Total Trades: N/A")
        print()

        # Extract parameters
        params = json.loads(best_run.param_values_json)
        print(f"🔧 Parameter Values:")
        for key, val in params.items():
            print(f"   {key}: {val}")
        print()

        # Load base strategy
        base_strategy_path = f"strategies/{exp.base_strategy_name}.yaml"
        print(f"📄 Loading base strategy: {base_strategy_path}")

        try:
            base_cfg = load_strategy_config(base_strategy_path)
        except FileNotFoundError:
            raise SystemExit(
                f"❌ Base strategy not found: {base_strategy_path}\n"
                f"   Ensure the strategy file exists."
            )

        # Apply parameter overrides
        print(f"⚙️  Applying parameter overrides...")
        new_cfg = apply_param_overrides(base_cfg, params)

        # Update strategy metadata
        output_name = Path(output_path).stem  # e.g., "baseline_v2"
        new_cfg["name"] = output_name
        new_cfg["description"] = (
            f"Optimized version of {exp.base_strategy_name} "
            f"from experiment '{exp.name}' (run {best_run.run_index})"
        )

        # Save new strategy
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_strategy_config(new_cfg, str(out_path))

        print(f"\n{'=' * 80}")
        print(f"✅ SUCCESS: New strategy created!")
        print(f"{'=' * 80}")
        print(f"Output: {out_path}")
        print(f"Strategy Name: {output_name}")
        print()
        print(f"Next steps:")
        print(f"1. Review the strategy: cat {out_path}")
        print(f"2. Backtest it: python -m otonom_trader.cli backtest --strategy {out_path}")
        print(f"3. Compare to baseline: Compare metrics in backtest reports")
        print(f"{'=' * 80}\n")


def main() -> None:
    """
    Main entry point for promote script.
    """
    parser = argparse.ArgumentParser(
        description="Promote best experiment run to a new strategy configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Promote experiment 3 to baseline_v2.yaml
  python scripts/promote_experiment_to_strategy.py \\
      --experiment-id 3 \\
      --output-path strategies/baseline_v2.yaml

  # Rank by CAGR instead of Sharpe
  python scripts/promote_experiment_to_strategy.py \\
      --experiment-id 3 \\
      --output-path strategies/baseline_v2.yaml \\
      --rank-by cagr
        """,
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="Experiment ID to promote from",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="strategies/baseline_v2.yaml",
        help="Output path for new strategy YAML (default: strategies/baseline_v2.yaml)",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["sharpe", "cagr", "sharpe_cagr"],
        default="sharpe_cagr",
        help="Metric to rank runs by (default: sharpe_cagr)",
    )

    args = parser.parse_args()

    try:
        promote_to_strategy(
            experiment_id=args.experiment_id,
            output_path=args.output_path,
            rank_by=args.rank_by,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
