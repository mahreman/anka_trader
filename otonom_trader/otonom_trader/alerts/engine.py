"""
Alert engine for monitoring and notifying about critical events.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .notifier import Notifier
from ..data import get_session
from ..data.schema import DaemonRun, PortfolioSnapshot

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Monitors system health and sends alerts for critical events.
    
    Monitors:
    - Broker errors
    - Kill-switch triggers
    - Daemon health (heartbeat)
    - Portfolio performance
    
    Example:
        >>> alerts = AlertEngine()
        >>> alerts.notify_broker_error("Connection timeout")
        >>> alerts.notify_kill_switch("Max daily loss exceeded")
        >>> alerts.check_and_notify(datetime.utcnow())
    """

    def __init__(self, alerts_config_path: str = "config/alerts.yaml"):
        """
        Initialize alert engine.
        
        Args:
            alerts_config_path: Path to alerts configuration YAML
        """
        self.notifier = Notifier(alerts_config_path)
        logger.info("Initialized AlertEngine")

    def notify_broker_error(self, msg: str) -> None:
        """
        Send alert for broker error.
        
        Args:
            msg: Error message
        """
        self.notifier.notify(
            subject="🚨 [Otonom Trader] Broker Error",
            body=f"Broker operation failed:\n\n{msg}\n\nPlease check broker connectivity and API keys.",
        )

    def notify_kill_switch(self, reason: str) -> None:
        """
        Send alert for kill-switch trigger.
        
        Args:
            reason: Reason for kill-switch activation
        """
        self.notifier.notify(
            subject="⛔ [Otonom Trader] KILL-SWITCH TRIGGERED",
            body=(
                f"Trading has been halted due to risk guardrails.\n\n"
                f"Reason: {reason}\n\n"
                f"No new orders will be placed until guardrails are cleared.\n"
                f"Please review the situation and take appropriate action."
            ),
        )

    def notify_daemon_stall(self, last_run: datetime, delta: timedelta) -> None:
        """
        Send alert for daemon stall/delay.
        
        Args:
            last_run: Last daemon run timestamp
            delta: Time since last run
        """
        self.notifier.notify(
            subject="⏰ [Otonom Trader] Daemon Stalled",
            body=(
                f"Daemon has not run recently.\n\n"
                f"Last run: {last_run.isoformat()}\n"
                f"Time since: {delta}\n\n"
                f"Please check daemon status and restart if necessary."
            ),
        )

    def notify_portfolio_alert(self, msg: str) -> None:
        """
        Send alert for portfolio event.
        
        Args:
            msg: Alert message
        """
        self.notifier.notify(
            subject="📊 [Otonom Trader] Portfolio Alert",
            body=msg,
        )

    def check_and_notify(self, now: datetime) -> None:
        """
        Check system health and send alerts if needed.
        
        Called at the end of each daemon cycle to monitor:
        - Daemon heartbeat (has it run recently?)
        - Portfolio performance (extreme drawdown?)
        
        Args:
            now: Current timestamp
        """
        self._check_daemon_heartbeat(now)
        self._check_portfolio_health(now)

    def _check_daemon_heartbeat(self, now: datetime) -> None:
        """
        Check if daemon has run recently.
        
        Args:
            now: Current timestamp
        """
        try:
            with get_session() as session:
                last_run = (
                    session.query(DaemonRun)
                    .order_by(DaemonRun.timestamp.desc())
                    .first()
                )

            if last_run is None:
                logger.warning("No daemon runs found in database")
                self.notifier.notify(
                    subject="⚠️ [Otonom Trader] No Daemon Runs",
                    body="No daemon run records found in database.",
                )
                return

            delta = now - last_run.timestamp
            
            # Alert if daemon hasn't run in 30 minutes
            if delta > timedelta(minutes=30):
                self.notify_daemon_stall(last_run.timestamp, delta)
                
        except Exception as e:
            logger.error(f"Error checking daemon heartbeat: {e}")

    def _check_portfolio_health(self, now: datetime) -> None:
        """
        Check portfolio health metrics.
        
        Args:
            now: Current timestamp
        """
        try:
            with get_session() as session:
                # Get recent snapshots
                recent_snaps = (
                    session.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.desc())
                    .limit(100)
                    .all()
                )

            if len(recent_snaps) < 2:
                return

            # Calculate current drawdown
            equity_series = [float(s.equity) for s in reversed(recent_snaps)]
            peak = max(equity_series)
            current = equity_series[-1]
            
            if peak == 0:
                return
            
            dd_pct = (current - peak) / peak * 100.0
            
            # Alert if extreme drawdown (>50%)
            if dd_pct <= -50.0:
                self.notify_portfolio_alert(
                    f"Extreme drawdown detected: {dd_pct:.2f}%\n"
                    f"Peak equity: ${peak:,.2f}\n"
                    f"Current equity: ${current:,.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error checking portfolio health: {e}")
