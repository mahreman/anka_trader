"""
Example: Daemon with broker shadow mode integration.

Shows how to integrate broker execution into the daemon loop.

Usage:
    python examples/daemon_with_broker_example.py
"""

from __future__ import annotations

import logging
from datetime import datetime, date

# Our existing infrastructure
from otonom_trader.data import get_engine, init_db, get_session
from otonom_trader.data.schema import Decision
from otonom_trader.brokers import create_broker, OrderRequest
from otonom_trader.brokers.config import get_broker_config
from otonom_trader.alerts.engine import check_alerts

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


def execute_shadow_order(decision, current_price, broker):
    """
    Execute shadow order for a decision.

    This is the simplified version matching your template.

    Args:
        decision: Decision record
        current_price: Current market price
        broker: Broker instance (GuardedBroker)
    """
    # Determine order side
    side = "BUY" if decision.direction.upper() == "BUY" else "SELL"

    # Create order request
    # In real system, qty would come from position sizing
    order_request = OrderRequest(
        symbol=decision.symbol,
        side=side,
        qty=decision.strength,  # Use strength as quantity modifier
        price=current_price,  # Use current price (or None for market order)
        order_type="LIMIT",  # or "MARKET"
    )

    # Place order (broker handles shadow mode internally)
    result = broker.place_order(
        order_request,
        current_price=current_price,
        current_equity=100000.0,  # Should track actual equity
        current_positions=0,  # Should track actual positions
    )

    # Log result
    if result.ok:
        logger.info(
            f"✓ Shadow order executed: {side} {decision.symbol} "
            f"qty={order_request.qty:.4f} @ ${current_price:.2f} "
            f"order_id={result.order_id}"
        )
    else:
        logger.warning(
            f"✗ Shadow order rejected: {result.message}"
        )

    return result


def run_daemon_tick():
    """
    Single daemon cycle - simplified version.

    Steps:
    1. Load broker configuration
    2. Create broker instance
    3. Find pending decisions
    4. Execute shadow orders
    5. Check alerts
    """
    logger.info("=== Daemon Tick ===")

    # Step 1: Load broker config
    broker_config = get_broker_config("config/broker.yaml")
    logger.info(f"Broker: type={broker_config.broker_type}, shadow_mode={broker_config.shadow_mode}")

    # Step 2: Create broker (automatically wrapped with risk guardrails)
    broker = create_broker(config=broker_config)
    logger.info("Broker initialized")

    # Step 3: Initialize database
    engine = get_engine("trader.db")
    init_db(engine)

    with get_session("trader.db") as session:
        # Step 4: Find today's decisions
        today = date.today()
        decisions = (
            session.query(Decision)
            .filter(Decision.timestamp >= datetime.combine(today, datetime.min.time()))
            .all()
        )

        if not decisions:
            logger.info("No decisions to execute")
        else:
            logger.info(f"Found {len(decisions)} decisions to execute")

            # Step 5: Execute each decision
            from otonom_trader.data import DailyBar

            for decision in decisions:
                # Get current price
                latest_bar = (
                    session.query(DailyBar)
                    .filter(DailyBar.symbol == decision.symbol)
                    .order_by(DailyBar.date.desc())
                    .first()
                )

                if not latest_bar:
                    logger.warning(f"No price data for {decision.symbol}")
                    continue

                current_price = latest_bar.close

                # Execute shadow order
                execute_shadow_order(decision, current_price, broker)

        # Step 6: Check alerts
        alerts = check_alerts(session)
        if alerts:
            logger.warning(f"Found {len(alerts)} alerts:")
            for alert in alerts:
                logger.warning(f"  [{alert.level}] {alert.message}")

    logger.info("=== Daemon Tick Complete ===\n")


# Alternative: Using our advanced ShadowModeExecutor
def run_daemon_tick_advanced():
    """
    Advanced version using ShadowModeExecutor.

    This version provides more detailed logging and tracking.
    """
    from otonom_trader.daemon.broker_integration import create_shadow_executor

    logger.info("=== Daemon Tick (Advanced) ===")

    # Create shadow mode executor (wraps broker with execution tracking)
    executor = create_shadow_executor(
        broker_config_path="config/broker.yaml",
        enable_broker_orders=True,  # Set False to disable broker calls
    )

    engine = get_engine("trader.db")
    init_db(engine)

    with get_session("trader.db") as session:
        today = date.today()
        decisions = (
            session.query(Decision)
            .filter(Decision.timestamp >= datetime.combine(today, datetime.min.time()))
            .all()
        )

        if not decisions:
            logger.info("No decisions to execute")
            return

        logger.info(f"Found {len(decisions)} decisions to execute")

        from otonom_trader.data import DailyBar

        for decision in decisions:
            # Get current price
            latest_bar = (
                session.query(DailyBar)
                .filter(DailyBar.symbol == decision.symbol)
                .order_by(DailyBar.date.desc())
                .first()
            )

            if not latest_bar:
                logger.warning(f"No price data for {decision.symbol}")
                continue

            current_price = latest_bar.close

            # Execute via shadow executor (tracks paper + broker execution)
            execution_log = executor.execute_decision(
                session=session,
                decision=decision,
                current_price=current_price,
                current_equity=100000.0,
                current_positions=0,
            )

            # Detailed logging
            logger.info(
                f"Decision {decision.id} executed:\n"
                f"  Paper fill: ${execution_log.paper_fill_price:.2f}\n"
                f"  Broker OK: {execution_log.broker_ok}\n"
                f"  Order ID: {execution_log.broker_order_id}\n"
                f"  Latency: {execution_log.latency_ms:.1f}ms\n"
                f"  Slippage: {execution_log.slippage_estimate:.2f}%"
                if execution_log.slippage_estimate else ""
            )

    logger.info("=== Daemon Tick Complete ===\n")


def show_broker_config():
    """
    Display current broker configuration.

    Useful for debugging.
    """
    from otonom_trader.brokers.config import get_broker_config

    config = get_broker_config("config/broker.yaml")

    print("=" * 60)
    print("BROKER CONFIGURATION")
    print("=" * 60)
    print(f"Broker Type:  {config.broker_type}")
    print(f"Shadow Mode:  {config.shadow_mode}")
    print(f"Use Testnet:  {config.use_testnet}")
    print(f"Base URL:     {config.base_url}")
    print()

    guardrails = config.get_risk_guardrails()
    print("Risk Guardrails:")
    print(f"  Max notional per order:  ${guardrails.max_notional_per_order:,.2f}")
    print(f"  Max open risk:           {guardrails.max_open_risk_pct:.1f}%")
    print(f"  Max total positions:     {guardrails.max_total_positions}")
    print(f"  Max position size:       {guardrails.max_position_size_pct:.1f}%")
    print(f"  Symbol blacklist:        {', '.join(guardrails.symbol_blacklist) or 'None'}")
    print(f"  Kill switch:             {guardrails.kill_switch}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "config":
            # Show broker config
            show_broker_config()
        elif command == "simple":
            # Run simplified version
            run_daemon_tick()
        elif command == "advanced":
            # Run advanced version
            run_daemon_tick_advanced()
        elif command == "loop":
            # Run continuous loop
            import time

            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
            print(f"Running daemon loop (interval: {interval}s)")
            print("Press Ctrl+C to stop\n")

            try:
                while True:
                    run_daemon_tick_advanced()
                    print(f"Sleeping {interval}s...")
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\nDaemon stopped by user")
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python examples/daemon_with_broker_example.py config    # Show config")
            print("  python examples/daemon_with_broker_example.py simple    # Run simple version")
            print("  python examples/daemon_with_broker_example.py advanced  # Run advanced version")
            print("  python examples/daemon_with_broker_example.py loop [interval]  # Run continuous")
    else:
        # Default: show config
        show_broker_config()
