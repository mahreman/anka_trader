"""
Daemon main loop with broker shadow mode integration.

Integrates patron decision-making with broker order execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional

from sqlalchemy.orm import Session

from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.data.schema import Symbol, Decision
from otonom_trader.brokers import create_broker
from otonom_trader.daemon.broker_integration import ShadowModeExecutor
from otonom_trader.alerts.engine import check_alerts

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


class TradingDaemon:
    """
    Main trading daemon with broker integration.

    Handles:
    - Data ingestion
    - Decision generation (patron)
    - Shadow/live order execution
    - Alert monitoring

    Example:
        >>> daemon = TradingDaemon()
        >>> daemon.run_once()  # Single tick
        >>> daemon.run_loop()  # Continuous loop
    """

    def __init__(
        self,
        db_path: str = "trader.db",
        broker_config_path: str = "config/broker.yaml",
    ):
        """
        Initialize trading daemon.

        Args:
            db_path: Database path
            broker_config_path: Broker configuration path
        """
        self.db_path = db_path
        self.broker_config_path = broker_config_path

        # Initialize database
        self.engine = get_engine(db_path)
        init_db(self.engine)

        # Create broker (wrapped with risk guardrails)
        self.broker = create_broker(config_path=broker_config_path)
        logger.info("Broker initialized")

        # Create shadow mode executor
        self.shadow_executor = ShadowModeExecutor(
            broker=self.broker,
            enable_broker_orders=True,  # Set False to disable broker calls
            log_to_database=True,
        )
        logger.info("Shadow mode executor initialized")

    def run_once(self, current_date: Optional[date] = None) -> None:
        """
        Run single daemon cycle.

        Args:
            current_date: Optional date (defaults to today)

        Steps:
        1. Ingest incremental data
        2. Run patron decision-making
        3. Execute decisions via broker (shadow/live)
        4. Check alerts
        """
        if current_date is None:
            current_date = date.today()

        logger.info(f"=== Daemon tick at {datetime.utcnow().isoformat()} ===")

        with get_session(self.db_path) as session:
            # Step 1: Ingest incremental data (TODO: implement)
            # self._ingest_incremental_data(session, current_date)

            # Step 2: Generate decisions (TODO: integrate patron)
            # decisions = self._generate_decisions(session, current_date)

            # Step 3: Execute decisions via broker
            self._execute_pending_decisions(session, current_date)

            # Step 4: Check alerts
            self._check_and_send_alerts(session)

        logger.info("=== Daemon cycle completed ===")

    def _execute_pending_decisions(
        self,
        session: Session,
        current_date: date,
    ) -> None:
        """
        Execute pending decisions via broker.

        Finds decisions from current day and sends orders to broker.

        Args:
            session: Database session
            current_date: Current date
        """
        from otonom_trader.data import DailyBar

        # Find decisions for today that haven't been executed
        decisions = (
            session.query(Decision)
            .filter(
                Decision.timestamp >= datetime.combine(current_date, datetime.min.time()),
                Decision.timestamp < datetime.combine(current_date, datetime.max.time()),
                # Decision.executed == False,  # Add this field if tracking execution
            )
            .all()
        )

        if not decisions:
            logger.info("No pending decisions to execute")
            return

        logger.info(f"Found {len(decisions)} decisions to execute")

        for decision in decisions:
            # Get current price
            latest_bar = (
                session.query(DailyBar)
                .filter(DailyBar.symbol == decision.symbol)
                .order_by(DailyBar.date.desc())
                .first()
            )

            if not latest_bar:
                logger.warning(f"No price data for {decision.symbol}, skipping")
                continue

            current_price = latest_bar.close

            # Get current equity (simplified - should track actual portfolio)
            current_equity = 100000.0  # TODO: Query from EquitySnapshot

            # Execute decision via shadow mode executor
            try:
                execution_log = self.shadow_executor.execute_decision(
                    session=session,
                    decision=decision,
                    current_price=current_price,
                    current_equity=current_equity,
                    current_positions=0,  # TODO: Track actual positions
                )

                # Log execution results
                logger.info(
                    f"Decision {decision.id} executed: "
                    f"paper=${execution_log.paper_fill_price:.2f}, "
                    f"broker_ok={execution_log.broker_ok}, "
                    f"latency={execution_log.latency_ms:.1f}ms"
                )

                if execution_log.broker_order_id:
                    logger.info(f"  Broker order ID: {execution_log.broker_order_id}")

                if execution_log.slippage_estimate:
                    logger.info(f"  Slippage estimate: {execution_log.slippage_estimate:.2f}%")

                # Mark decision as executed (if tracking)
                # decision.executed = True
                # decision.execution_time = datetime.utcnow()
                # decision.broker_order_id = execution_log.broker_order_id

            except Exception as e:
                logger.error(f"Failed to execute decision {decision.id}: {e}", exc_info=True)

        session.commit()

    def _check_and_send_alerts(self, session: Session) -> None:
        """
        Check system health and send alerts.

        Args:
            session: Database session
        """
        alerts = check_alerts(
            session=session,
            max_dd_threshold=-20.0,
            max_daemon_staleness_hours=2,
            max_days_without_trade=7,
            max_data_staleness_hours=24,
        )

        if alerts:
            logger.warning(f"Found {len(alerts)} alerts")
            for alert in alerts:
                logger.warning(f"  {alert.level}: {alert.message}")

            # TODO: Send to notification channels
            # send_alerts(alerts, channels=["email", "slack"])

    def run_loop(self, interval_seconds: int = 3600) -> None:
        """
        Run daemon in continuous loop.

        Args:
            interval_seconds: Sleep interval between cycles (default: 1 hour)

        Example:
            >>> daemon = TradingDaemon()
            >>> daemon.run_loop(interval_seconds=3600)  # Run every hour
        """
        import time

        logger.info(f"Starting daemon loop (interval: {interval_seconds}s)")

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}", exc_info=True)

            logger.info(f"Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)


def run_daemon_once() -> None:
    """
    Run daemon for single tick.

    Convenience function for CLI or cron jobs.

    Example:
        >>> run_daemon_once()
    """
    daemon = TradingDaemon()
    daemon.run_once()


def run_daemon_loop(interval_seconds: int = 3600) -> None:
    """
    Run daemon in continuous loop.

    Args:
        interval_seconds: Sleep interval between cycles

    Example:
        >>> run_daemon_loop(interval_seconds=3600)  # Every hour
    """
    daemon = TradingDaemon()
    daemon.run_loop(interval_seconds=interval_seconds)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        # Run continuous loop
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
        run_daemon_loop(interval_seconds=interval)
    else:
        # Run once
        run_daemon_once()
