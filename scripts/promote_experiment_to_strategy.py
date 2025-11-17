#!/usr/bin/env python
"""
Promote best run from experiment to new strategy version.

Selects the best run from an experiment (by test Sharpe, then test CAGR),
extracts parameter values, and creates a new strategy version.

Example:
    # After running grid search (experiment_id=3):
    python scripts/promote_experiment_to_strategy.py --experiment-id 3 \\
        --output-name baseline_v2 \\
        --base-strategy strategies/baseline_v1.yaml

    # With expected impact estimate:
    python scripts/promote_experiment_to_strategy.py --experiment-id 3 \\
        --output-name baseline_v2 \\
        --base-strategy strategies/baseline_v1.yaml \\
        --expected-impact "+15% CAGR, +0.3 Sharpe on test set"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.experiments.strategy_versioning import promote_experiment_to_strategy


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Promote best experiment run to new strategy version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="Experiment ID to promote",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name for new strategy version (e.g., 'baseline_v2')",
    )

    parser.add_argument(
        "--base-strategy",
        type=str,
        required=True,
        help="Path to base strategy YAML (e.g., 'strategies/baseline_v1.yaml')",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="strategies",
        help="Output directory for new strategy (default: strategies)",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="STRATEGY_LOG.md",
        help="Path to strategy evolution log (default: STRATEGY_LOG.md)",
    )

    parser.add_argument(
        "--expected-impact",
        type=str,
        default=None,
        help="Expected performance impact (e.g., '+15%% CAGR, +0.3 Sharpe')",
    )

    args = parser.parse_args()

    # Validate base strategy exists
    base_path = Path(args.base_strategy)
    if not base_path.exists():
        print(f"❌ Error: Base strategy not found: {args.base_strategy}")
        return 1

    # Initialize database
    engine = get_engine()
    init_db(engine)

    # Promote experiment
    try:
        with get_session() as session:
            new_strategy_path = promote_experiment_to_strategy(
                experiment_id=args.experiment_id,
                session=session,
                new_strategy_name=args.output_name,
                base_strategy_path=args.base_strategy,
                output_dir=args.output_dir,
                log_path=args.log_path,
                expected_impact=args.expected_impact,
            )

            print(f"✅ Successfully promoted experiment {args.experiment_id}")
            print(f"📄 New strategy: {new_strategy_path}")
            print(f"📝 Updated log: {args.log_path}")
            print()
            print("Next steps:")
            print(f"  1. Review new strategy: cat {new_strategy_path}")
            print(f"  2. Backtest new strategy: python -m otonom_trader.cli backtest ...")
            print(f"  3. Compare with baseline: python -m otonom_trader.cli compare ...")
            print(f"  4. Deploy to production if results are good")

            return 0

    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
