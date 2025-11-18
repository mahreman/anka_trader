"""
Autonomous trading orchestrator.

P3 feature: Daemon that runs the trading pipeline in a loop.

Pipeline:
1. Incremental data ingest
2. Anomaly detection
3. Patron decision generation
4. Paper trade execution

Usage:
    python -m otonom_trader.orchestrator
"""
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from .daemon import run_daemon_cycle, DaemonConfig, get_or_create_paper_trader
from .data import get_session, init_db, get_engine
from .config import DB_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.info(f"\n{signal_name} received. Shutting down gracefully...")
    _shutdown_requested = True


def run_orchestrator_loop(
    cycle_interval_seconds: float = 900,  # 15 minutes
    max_cycles: Optional[int] = None,
    config: Optional[DaemonConfig] = None,
) -> None:
    """
    Run the orchestrator in a continuous loop.

    Args:
        cycle_interval_seconds: Time to wait between cycles (default: 900s = 15 min)
        max_cycles: Maximum number of cycles to run (None = infinite)
        config: Daemon configuration (uses default if None)
    """
    global _shutdown_requested

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if config is None:
        config = DaemonConfig()

    cycle_count = 0

    logger.info("Starting orchestrator loop...")

    # Initialize database schema
    logger.info(f"Initializing database at: {DB_PATH}")
    engine = get_engine(DB_PATH)
    init_db(engine)
    logger.info("Database initialized successfully")

    try:
        while not _shutdown_requested:
            cycle_count += 1

            # Check if we've reached max cycles
            if max_cycles is not None and cycle_count > max_cycles:
                logger.info(f"Reached maximum cycles ({max_cycles}). Exiting.")
                break

            # Run a single cycle
            logger.info(
                f"===== Orchestrator cycle #{cycle_count} starting at {datetime.utcnow().isoformat()} ====="
            )
            cycle_start = time.time()

            try:
                # Create a new session for this cycle
                with next(get_session()) as session:
                    # Get or create paper trader (restores state from DB)
                    paper_trader = get_or_create_paper_trader(session, config.initial_cash)

                    # Run the daemon cycle
                    run = run_daemon_cycle(
                        session=session,
                        config=config,
                        paper_trader=paper_trader,
                    )

                    logger.info(f"Cycle #{cycle_count} completed with status: {run.status}")

            except Exception as e:
                logger.error(f"Orchestrator cycle #{cycle_count} failed: {e}", exc_info=True)

            cycle_duration = time.time() - cycle_start
            logger.info(
                f"Cycle #{cycle_count} finished in {cycle_duration:.1f}s. "
                f"Sleeping {cycle_interval_seconds:.1f}s before next cycle..."
            )

            # Sleep until next cycle (with periodic checks for shutdown)
            sleep_start = time.time()
            while not _shutdown_requested:
                elapsed = time.time() - sleep_start
                if elapsed >= cycle_interval_seconds:
                    break

                # Sleep in small increments to check for shutdown signal
                time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in orchestrator loop: {e}", exc_info=True)
        raise
    finally:
        logger.info(f"Orchestrator stopped after {cycle_count} cycles.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous trading orchestrator")
    parser.add_argument(
        "--interval",
        type=float,
        default=900,
        help="Cycle interval in seconds (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of cycles to run (default: infinite)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash for paper trader (default: 100000)",
    )
    parser.add_argument(
        "--ingest-days-back",
        type=int,
        default=7,
        help="Days to look back for incremental ingest (default: 7)",
    )
    parser.add_argument(
        "--anomaly-lookback-days",
        type=int,
        default=30,
        help="Days to look back for anomaly detection (default: 30)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.0,
        help="Risk percentage per trade (default: 1.0)",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use ensemble mode for Patron decisions",
    )

    args = parser.parse_args()

    # Create daemon config from args
    config = DaemonConfig(
        ingest_days_back=args.ingest_days_back,
        anomaly_lookback_days=args.anomaly_lookback_days,
        use_ensemble=args.use_ensemble,
        paper_trade_enabled=True,
        paper_trade_risk_pct=args.risk_pct,
        initial_cash=args.initial_cash,
    )

    # Run orchestrator
    run_orchestrator_loop(
        cycle_interval_seconds=args.interval,
        max_cycles=args.max_cycles,
        config=config,
    )


if __name__ == "__main__":
    main()
