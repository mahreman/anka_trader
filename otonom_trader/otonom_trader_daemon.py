#!/usr/bin/env python
"""
Autonomous trader daemon skeleton (P2.5/P3).

This daemon runs periodically to:
1. Ingest latest market data
2. Recompute regime detection and DSI
3. Detect new anomalies
4. Run Patron decision engine
5. Execute trading decisions (P3)

Usage:
    python otonom_trader_daemon.py

Scheduling:
    Use cron, systemd timer, or task scheduler to run this script periodically.
    Example cron (every 15 minutes):
        */15 * * * * /usr/bin/python /path/to/otonom_trader_daemon.py

Note:
    This is a "run_once" daemon. It performs one cycle and exits.
    For continuous operation, use an external scheduler.
"""

from __future__ import annotations

import logging

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.data.schema import Symbol
from otonom_trader.analytics.regime import compute_regimes_for_symbol, persist_regimes
from otonom_trader.analytics.dsi import compute_dsi_for_symbol, persist_dsi
from otonom_trader.utils import utc_now


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def run_once() -> None:
    """
    Run one daemon cycle.

    Steps:
    1. Initialize database
    2. Ingest new data (incremental)
    3. Recompute analytics (regime, DSI)
    4. Detect anomalies
    5. Run Patron decision engine
    6. Execute trades (P3)
    """
    logger.info("=" * 60)
    logger.info("Daemon cycle started at %s", utc_now().isoformat())
    logger.info("=" * 60)

    engine = get_engine()
    init_db(engine)

    # TODO: Ingest new data
    # Option 1: Call CLI via subprocess
    # subprocess.run([sys.executable, "-m", "otonom_trader.cli", "ingest-data", "--mode", "incremental"])
    #
    # Option 2: Import and call directly
    # from otonom_trader.data.ingest import ingest_incremental
    # ingest_incremental()

    logger.info("Step 1: Ingesting new data (TODO)")
    # For now, assume data is already ingested

    # Recompute analytics for all symbols
    logger.info("Step 2: Recomputing analytics (regime, DSI)")

    with get_session() as session:
        symbols = session.query(Symbol).all()

        for sym in symbols:
            logger.info("Processing symbol: %s", sym.symbol)

            # Recompute regimes
            try:
                regimes = compute_regimes_for_symbol(session, sym.symbol)
                if regimes:
                    persist_regimes(session, regimes)
                    logger.info("  -> %d regime points computed", len(regimes))
                else:
                    logger.warning("  -> No regime data for %s", sym.symbol)
            except Exception as e:
                logger.error("  -> Regime computation failed: %s", e)

            # Recompute DSI
            try:
                dsi_points = compute_dsi_for_symbol(session, sym.symbol)
                if dsi_points:
                    persist_dsi(session, dsi_points)
                    logger.info("  -> %d DSI points computed", len(dsi_points))
                else:
                    logger.warning("  -> No DSI data for %s", sym.symbol)
            except Exception as e:
                logger.error("  -> DSI computation failed: %s", e)

    # TODO: Detect new anomalies
    logger.info("Step 3: Detecting anomalies (TODO)")
    # from otonom_trader.analytics.anomaly import detect_anomalies_for_symbol
    # for sym in symbols:
    #     detect_anomalies_for_symbol(session, sym.symbol)

    # TODO: Run Patron
    logger.info("Step 4: Running Patron decision engine (TODO)")
    # from otonom_trader.patron.rules import run_daily_decision_pass
    # run_daily_decision_pass(session, days_back=1, use_ensemble=True)

    # TODO: Execute trades (P3)
    logger.info("Step 5: Executing trades (TODO - P3 feature)")
    # from otonom_trader.execution.broker import execute_pending_orders
    # execute_pending_orders()

    logger.info("=" * 60)
    logger.info("Daemon cycle completed at %s", utc_now().isoformat())
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        logger.exception("Daemon cycle failed: %s", e)
        raise
