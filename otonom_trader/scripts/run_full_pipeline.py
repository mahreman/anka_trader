#!/usr/bin/env python
"""
End-to-end smoke test pipeline for Otonom Trader.

This script runs the full P0+P1 pipeline to verify that all components work together:

Steps:
1) DB init
2) Data ingest (from 2018-01-01 onwards)
3) Anomaly detection
4) Patron decision engine (P0 rules)
5) Regime + DSI computation
6) Event-based backtest
7) Log summary metrics

Note:
- CLI commands are invoked via subprocess (init / ingest / anomalies / patron)
- Regime / DSI / backtest functions are called directly via Python API
"""

from __future__ import annotations

import subprocess
import sys
import logging
from typing import List

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.analytics.regime import compute_regimes_for_symbol, persist_regimes
from otonom_trader.analytics.dsi import compute_dsi_for_symbol, persist_dsi
from otonom_trader.eval.backtest import (
    BacktestConfig,
    create_or_get_hypothesis,
    run_event_backtest,
)
from otonom_trader.data.schema import Symbol, HypothesisResult


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def run_cli(args: List[str]) -> None:
    """
    Run CLI commands via subprocess.

    Args:
        args: CLI command arguments (e.g., ["init"], ["ingest-data", "--start", "2018-01-01"])

    Raises:
        SystemExit: If CLI command fails

    Example:
        >>> run_cli(["init"])
    """
    cmd = [sys.executable, "-m", "otonom_trader.cli"] + args
    logger.info("Running CLI: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"CLI command failed: {' '.join(cmd)} (rc={result.returncode})")


def step_init_db() -> None:
    """Initialize database schema."""
    logger.info("=== STEP 1: DB init ===")
    # Use direct API to avoid confirmation prompt
    engine = get_engine()
    init_db(engine)
    logger.info("Database initialized successfully")


def step_ingest_data() -> None:
    """Ingest historical data from 2018-01-01 to today."""
    logger.info("=== STEP 2: Data ingest (2018-01-01 -> today) ===")
    # CLI command: ingest-data --start 2018-01-01
    run_cli(["ingest-data", "--start", "2018-01-01"])


def step_detect_anomalies() -> None:
    """Detect anomalies (spikes/crashes) in all symbols."""
    logger.info("=== STEP 3: Anomaly detection ===")
    run_cli(["detect-anomalies"])


def step_run_patron() -> None:
    """Run Patron decision engine with P0/P1 rules."""
    logger.info("=== STEP 4: Patron (P0 rules) ===")
    # Use basic P0 rules (no ensemble for smoke test)
    run_cli(["run-patron"])


def step_compute_regimes_and_dsi() -> None:
    """Compute regime detection and DSI for all symbols."""
    logger.info("=== STEP 5: Regime + DSI computation ===")

    with get_session() as session:
        symbols = session.query(Symbol).all()

        for sym in symbols:
            logger.info("Processing %s...", sym.symbol)

            # Compute regimes
            logger.info("  Computing regimes for %s", sym.symbol)
            regimes = compute_regimes_for_symbol(session, sym.symbol)
            if regimes:
                persist_regimes(session, regimes)
                logger.info("    -> %d regime points persisted", len(regimes))
            else:
                logger.warning("    -> No regime data")

            # Compute DSI
            logger.info("  Computing DSI for %s", sym.symbol)
            dsi_points = compute_dsi_for_symbol(session, sym.symbol)
            if dsi_points:
                persist_dsi(session, dsi_points)
                logger.info("    -> %d DSI points persisted", len(dsi_points))
            else:
                logger.warning("    -> No DSI data")


def step_backtest() -> None:
    """Run event-based backtest on all decisions."""
    logger.info("=== STEP 6: Event-based backtest ===")

    engine = get_engine()
    init_db(engine)

    with get_session() as session:
        # Create hypothesis for smoke test
        hyp = create_or_get_hypothesis(
            session,
            name="P1_smoke",
            rule_signature="P0 basic rules; 5d hold",
            description="Smoke test hypothesis for full pipeline",
        )

        # Configure backtest
        config = BacktestConfig(
            holding_days=5,
            slippage_bps=5.0,
        )

        # Run backtest
        run_event_backtest(
            session,
            hypothesis=hyp,
            config=config,
            regime_resolver=None,  # P2: Add regime/DSI context
            dsi_resolver=None,
        )

        # Query results
        results = (
            session.query(HypothesisResult)
            .filter(HypothesisResult.hypothesis_id == hyp.id)
            .all()
        )

        n_trades = len(results)
        if n_trades == 0:
            logger.warning("Backtest produced 0 trades (no anomalies/decisions found)")
            return

        # Calculate metrics
        total_pnl = sum(r.pnl for r in results)
        win_trades = sum(1 for r in results if r.pnl > 0)
        win_rate = win_trades / n_trades if n_trades > 0 else 0.0

        logger.info("=== Smoke Backtest Summary ===")
        logger.info("Trades      : %d", n_trades)
        logger.info("Win rate    : %.2f%%", win_rate * 100)
        logger.info("Total PnL   : %.4f", total_pnl)


def main() -> None:
    """
    Run full end-to-end pipeline.

    Usage:
        python scripts/run_full_pipeline.py
    """
    logger.info("=" * 60)
    logger.info("Starting FULL PIPELINE smoke test")
    logger.info("=" * 60)

    try:
        step_init_db()
        step_ingest_data()
        step_detect_anomalies()
        step_run_patron()
        step_compute_regimes_and_dsi()
        step_backtest()

        logger.info("=" * 60)
        logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("✗ PIPELINE FAILED: %s", e)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
