"""
Broker integration for daemon.

Handles shadow mode trading: executes both paper trades and real broker orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..brokers import Broker, OrderRequest, OrderResult, create_broker
from ..data import DecisionORM as Decision
from ..utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class TradeExecutionLog:
    """
    Trade execution log entry.

    Records details of both paper and broker execution for comparison.

    Attributes:
        decision_id: Decision ID that triggered trade
        symbol: Symbol traded
        side: Trade side (BUY/SELL)
        qty: Quantity
        paper_fill_price: Fill price in paper account
        broker_ack_time: Time broker acknowledged order
        broker_order_id: Broker order ID
        broker_ok: Whether broker order succeeded
        broker_message: Broker response message
        latency_ms: Time from decision to broker ack (ms)
        slippage_estimate: Estimated slippage (%)
    """

    decision_id: int
    symbol: str
    side: str
    qty: float
    paper_fill_price: float
    broker_ack_time: Optional[datetime] = None
    broker_order_id: Optional[str] = None
    broker_ok: bool = False
    broker_message: Optional[str] = None
    latency_ms: Optional[float] = None
    slippage_estimate: Optional[float] = None


class ShadowModeExecutor:
    """
    Shadow mode trade executor.

    Executes trades in both paper account and via broker for validation.

    Example:
        >>> executor = ShadowModeExecutor(broker=create_broker())
        >>> decision = Decision(...)  # From patron
        >>> log = executor.execute_decision(session, decision, current_price=50000)
        >>> print(f"Paper: ${log.paper_fill_price}, Broker: {log.broker_ok}")
    """

    def __init__(
        self,
        broker: Broker,
        enable_broker_orders: bool = True,
        log_to_database: bool = True,
    ):
        """
        Initialize shadow mode executor.

        Args:
            broker: Broker instance (created via create_broker())
            enable_broker_orders: If False, only paper trades (no broker calls)
            log_to_database: If True, log trades to database
        """
        self.broker = broker
        self.enable_broker_orders = enable_broker_orders
        self.log_to_database = log_to_database

        logger.info(
            f"ShadowModeExecutor initialized: "
            f"broker_orders={enable_broker_orders}, "
            f"db_logging={log_to_database}"
        )

    def execute_decision(
        self,
        session: Session,
        decision: Decision,
        current_price: float,
        current_equity: Optional[float] = None,
        current_positions: Optional[int] = None,
    ) -> TradeExecutionLog:
        """
        Execute decision in shadow mode.

        1. Execute paper trade (update paper portfolio)
        2. Send order to broker (if enabled)
        3. Log execution details for comparison

        Args:
            session: Database session
            decision: Decision to execute
            current_price: Current market price
            current_equity: Current account equity
            current_positions: Current number of positions

        Returns:
            TradeExecutionLog with execution details

        Example:
            >>> decision = Decision(symbol="BTC-USD", direction="BUY", strength=0.8)
            >>> log = executor.execute_decision(session, decision, current_price=50000)
        """
        start_time = utc_now()

        # Determine trade parameters
        symbol = decision.symbol
        side = "BUY" if decision.direction == "BUY" else "SELL"
        qty = self._calculate_position_size(
            decision=decision,
            current_price=current_price,
            current_equity=current_equity,
        )

        # Step 1: Execute paper trade
        paper_fill_price = current_price  # Simplification: assume fill at current price

        logger.info(
            f"[PAPER] Executed {side} {qty:.6f} {symbol} @ ${paper_fill_price:.2f}"
        )

        # Step 2: Send order to broker (if enabled)
        broker_result = None
        broker_ack_time = None
        latency_ms = None

        if self.enable_broker_orders:
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                qty=qty,
                price=None,  # Market order
                order_type="MARKET",
            )

            try:
                # Send order to broker
                broker_result = self.broker.place_order(
                    order_request,
                    current_price=current_price,
                    current_equity=current_equity,
                    current_positions=current_positions,
                )

                broker_ack_time = utc_now()
                latency_ms = (broker_ack_time - start_time).total_seconds() * 1000

                if broker_result.ok:
                    logger.info(
                        f"[BROKER] Order acked: {broker_result.order_id} "
                        f"(latency: {latency_ms:.1f}ms)"
                    )
                else:
                    logger.warning(
                        f"[BROKER] Order rejected: {broker_result.message} "
                        f"(latency: {latency_ms:.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"[BROKER] Order failed: {e}")
                broker_result = OrderResult(ok=False, message=str(e))

        # Calculate slippage estimate
        slippage_estimate = None
        if broker_result and broker_result.ok and broker_result.avg_price:
            slippage_estimate = (
                (broker_result.avg_price - paper_fill_price) / paper_fill_price * 100
            )

        # Create execution log
        log = TradeExecutionLog(
            decision_id=decision.id,
            symbol=symbol,
            side=side,
            qty=qty,
            paper_fill_price=paper_fill_price,
            broker_ack_time=broker_ack_time,
            broker_order_id=broker_result.order_id if broker_result else None,
            broker_ok=broker_result.ok if broker_result else False,
            broker_message=broker_result.message if broker_result else None,
            latency_ms=latency_ms,
            slippage_estimate=slippage_estimate,
        )

        # TODO: Log to database if enabled
        if self.log_to_database:
            self._log_to_database(session, log)

        return log

    def _calculate_position_size(
        self,
        decision: Decision,
        current_price: float,
        current_equity: Optional[float],
    ) -> float:
        """
        Calculate position size based on decision strength and risk parameters.

        Args:
            decision: Decision with direction and strength
            current_price: Current market price
            current_equity: Current account equity

        Returns:
            Position size (quantity)
        """
        # Default equity if not provided
        if current_equity is None:
            current_equity = 100000.0

        # Default: 1% risk per trade, scaled by strength
        risk_pct = 0.01 * decision.strength  # 0.8 strength = 0.8% risk
        risk_amount = current_equity * risk_pct

        # Calculate quantity
        qty = risk_amount / current_price

        return qty

    def _log_to_database(self, session: Session, log: TradeExecutionLog):
        """
        Log trade execution to database.

        Args:
            session: Database session
            log: Trade execution log

        TODO: Create TradeExecutionLog table in schema
        """
        # TODO: Implement database logging
        # from ..data import TradeExecutionLog as TradeExecutionLogORM
        # orm_log = TradeExecutionLogORM(
        #     decision_id=log.decision_id,
        #     symbol=log.symbol,
        #     side=log.side,
        #     qty=log.qty,
        #     paper_fill_price=log.paper_fill_price,
        #     broker_order_id=log.broker_order_id,
        #     broker_ok=log.broker_ok,
        #     broker_message=log.broker_message,
        #     latency_ms=log.latency_ms,
        #     slippage_estimate=log.slippage_estimate,
        # )
        # session.add(orm_log)
        # session.commit()

        logger.debug(f"Trade execution logged: {log.symbol} {log.side}")


def create_shadow_executor(
    broker_config_path: str = "config/broker.yaml",
    enable_broker_orders: bool = True,
) -> ShadowModeExecutor:
    """
    Create shadow mode executor with configured broker.

    Args:
        broker_config_path: Path to broker.yaml
        enable_broker_orders: Enable real broker orders

    Returns:
        ShadowModeExecutor instance

    Example:
        >>> executor = create_shadow_executor()
        >>> # Use in daemon...
    """
    broker = create_broker(config_path=broker_config_path)

    executor = ShadowModeExecutor(
        broker=broker,
        enable_broker_orders=enable_broker_orders,
    )

    return executor
