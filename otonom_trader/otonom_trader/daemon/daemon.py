"""
Autonomous trading daemon.
P3 preparation: Full pipeline automation.

Pipeline:
1. Incremental data ingest
2. Anomaly detection
3. Patron decision generation
4. Paper trade execution
"""
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..data import get_session
from ..data.ingest import ingest_incremental
from ..data.symbols import get_p0_assets
from ..data.schema import DaemonRun, Symbol, Anomaly as AnomalyORM
from ..analytics import detect_anomalies_all_assets
from ..patron import run_daily_decision_pass
from ..domain import Decision as DecisionDomain, AnomalyType
from ..config import (
    ANOMALY_ZSCORE_THRESHOLD,
    ANOMALY_VOLUME_QUANTILE,
    ANOMALY_ROLLING_WINDOW,
)
from .paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class DaemonConfig:
    """Configuration for daemon runs."""

    def __init__(
        self,
        ingest_days_back: int = 7,
        anomaly_lookback_days: int = 30,
        anomaly_k: float = ANOMALY_ZSCORE_THRESHOLD,
        anomaly_q: float = ANOMALY_VOLUME_QUANTILE,
        anomaly_window: int = ANOMALY_ROLLING_WINDOW,
        use_ensemble: bool = False,
        paper_trade_enabled: bool = True,
        paper_trade_risk_pct: float = 1.0,
        initial_cash: float = 100000.0,
    ):
        """
        Initialize daemon config.

        Args:
            ingest_days_back: Days to look back for incremental ingest (if no data)
            anomaly_lookback_days: Days to look back for anomaly detection
            anomaly_k: Z-score threshold for anomaly detection
            anomaly_q: Volume quantile threshold for anomaly detection
            anomaly_window: Rolling window size for anomaly detection
            use_ensemble: Use ensemble mode for Patron
            paper_trade_enabled: Enable paper trading
            paper_trade_risk_pct: Risk percentage per trade
            initial_cash: Initial cash for paper trader
        """
        self.ingest_days_back = ingest_days_back
        self.anomaly_lookback_days = anomaly_lookback_days
        self.anomaly_k = anomaly_k
        self.anomaly_q = anomaly_q
        self.anomaly_window = anomaly_window
        self.use_ensemble = use_ensemble
        self.paper_trade_enabled = paper_trade_enabled
        self.paper_trade_risk_pct = paper_trade_risk_pct
        self.initial_cash = initial_cash


def run_daemon_cycle(
    session: Session,
    config: Optional[DaemonConfig] = None,
    paper_trader: Optional[PaperTrader] = None,
) -> DaemonRun:
    """
    Run a single daemon cycle.

    Pipeline:
    1. Incremental data ingest
    2. Anomaly detection (on recent data)
    3. Patron decision generation
    4. Paper trade execution (optional)

    Args:
        session: Database session
        config: Daemon configuration
        paper_trader: Paper trader instance (will create if None and enabled)

    Returns:
        DaemonRun record
    """
    if config is None:
        config = DaemonConfig()

    start_time = time.time()
    timestamp = datetime.utcnow()

    logger.info("=" * 60)
    logger.info(f"Starting daemon cycle at {timestamp}")
    logger.info("=" * 60)

    # Initialize run record
    run = DaemonRun(
        timestamp=timestamp,
        status="RUNNING",
    )
    session.add(run)
    session.commit()

    try:
        # Step 1: Incremental data ingest
        logger.info("\n[1/4] Incremental data ingest...")
        assets = get_p0_assets()

        ingest_results = ingest_incremental(
            session, days_back=config.ingest_days_back, assets=assets
        )

        bars_ingested = sum(ingest_results.values())
        run.bars_ingested = bars_ingested

        logger.info(f"  ✓ Ingested {bars_ingested} bars across {len(assets)} assets")

        # Step 2: Anomaly detection (on recent data only)
        logger.info("\n[2/4] Anomaly detection...")

        # Get date range for recent data
        end_date = date.today()
        start_date = end_date - timedelta(days=config.anomaly_lookback_days)

        # Delete old anomalies in this range to avoid duplicates
        for asset in assets:
            symbol_obj = session.query(Symbol).filter_by(symbol=asset.symbol).first()
            if symbol_obj:
                session.query(AnomalyORM).filter(
                    AnomalyORM.symbol_id == symbol_obj.id,
                    AnomalyORM.date >= start_date,
                    AnomalyORM.date <= end_date,
                ).delete()
        session.commit()

        anomaly_results = detect_anomalies_all_assets(
            session=session,
            assets=assets,
            k=config.anomaly_k,
            q=config.anomaly_q,
            window=config.anomaly_window,
        )

        anomalies_detected = sum(len(a) for a in anomaly_results.values())
        run.anomalies_detected = anomalies_detected

        logger.info(f"  ✓ Detected {anomalies_detected} anomalies")

        # Step 3: Patron decision generation
        logger.info("\n[3/4] Patron decision generation...")

        decision_results = run_daily_decision_pass(
            session,
            days_back=config.anomaly_lookback_days,
            use_ensemble=config.use_ensemble,
        )

        # Flatten decisions
        all_decisions = []
        for decisions in decision_results.values():
            all_decisions.extend(decisions)

        decisions_made = len(all_decisions)
        run.decisions_made = decisions_made

        logger.info(f"  ✓ Generated {decisions_made} decisions")

        # Step 4: Paper trade execution
        trades_executed = 0
        if config.paper_trade_enabled and all_decisions:
            logger.info("\n[4/4] Paper trade execution...")

            # Initialize paper trader if not provided
            if paper_trader is None:
                paper_trader = PaperTrader(session, initial_cash=config.initial_cash)

            # Update portfolio prices before trading
            paper_trader.update_portfolio_prices()

            # Execute decisions
            for decision in all_decisions:
                trade = paper_trader.execute_decision(
                    decision, risk_pct=config.paper_trade_risk_pct
                )
                if trade:
                    trades_executed += 1

            run.trades_executed = trades_executed

            # Create portfolio snapshot
            snapshot = paper_trader.create_portfolio_snapshot()

            # Update daemon run with portfolio metrics
            run.portfolio_value = snapshot.equity
            run.cash = snapshot.cash

            logger.info(f"  ✓ Executed {trades_executed} trades")
            logger.info(f"  Portfolio value: ${snapshot.equity:,.2f}")
            logger.info(f"  Cash: ${snapshot.cash:,.2f}")
            logger.info(f"  Positions: {snapshot.num_positions}")
            if snapshot.max_drawdown is not None:
                logger.info(f"  Max drawdown: {snapshot.max_drawdown:.2%}")

        else:
            logger.info("\n[4/4] Paper trade execution... SKIPPED")

        # Success
        duration = time.time() - start_time
        run.status = "SUCCESS"
        run.duration_seconds = duration
        session.commit()

        logger.info("\n" + "=" * 60)
        logger.info(f"Daemon cycle completed in {duration:.2f}s")
        logger.info(f"  Bars ingested:      {bars_ingested}")
        logger.info(f"  Anomalies detected: {anomalies_detected}")
        logger.info(f"  Decisions made:     {decisions_made}")
        logger.info(f"  Trades executed:    {trades_executed}")
        logger.info("=" * 60)

        return run

    except Exception as e:
        # Log error
        duration = time.time() - start_time
        run.status = "FAILED"
        run.error_message = str(e)
        run.duration_seconds = duration
        session.commit()

        logger.error(f"\n✗ Daemon cycle failed: {e}", exc_info=True)
        raise


def get_or_create_paper_trader(session: Session, initial_cash: float = 100000.0) -> PaperTrader:
    """
    Get or create a paper trader instance.

    Loads existing portfolio state from last paper trade if available.

    Args:
        session: Database session
        initial_cash: Initial cash if creating new trader

    Returns:
        PaperTrader instance
    """
    from ..data.schema import PaperTrade

    # Get last paper trade to restore portfolio state
    last_trade = (
        session.query(PaperTrade).order_by(PaperTrade.timestamp.desc()).first()
    )

    if last_trade:
        # Restore from last state
        trader = PaperTrader(session, initial_cash=last_trade.cash)

        # Reconstruct positions from trade history
        # (This is a simplified approach - in production you'd want a separate
        # positions table or more sophisticated state management)
        logger.info(f"Restored paper trader from last trade (cash: ${last_trade.cash:,.2f})")

    else:
        # Create new trader
        trader = PaperTrader(session, initial_cash=initial_cash)
        logger.info(f"Created new paper trader with ${initial_cash:,.2f}")

    return trader
