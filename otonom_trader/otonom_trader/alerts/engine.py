"""
Alert engine for portfolio monitoring and health checks.

Monitors:
- Equity drawdown
- Daemon failures
- No-trade periods
- Data staleness
- Risk threshold breaches
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARN = "WARN"
    CRIT = "CRIT"


@dataclass
class Alert:
    """
    Alert notification.

    Attributes:
        level: Alert severity (INFO, WARN, CRIT)
        message: Human-readable alert message
        timestamp: When alert was generated
        category: Alert category (e.g., "drawdown", "daemon", "data")
        details: Optional additional details

    Example:
        >>> alert = Alert(
        ...     level=AlertLevel.WARN,
        ...     message="Equity drawdown exceeds 10%",
        ...     category="drawdown",
        ...     details={"current_dd": -12.5}
        ... )
    """

    level: AlertLevel
    message: str
    timestamp: datetime = None
    category: str = "general"
    details: Optional[dict] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def __repr__(self) -> str:
        return f"[{self.level.value}] {self.category}: {self.message}"


def check_equity_drawdown(
    session: Session,
    max_dd_threshold: float = -20.0,
) -> List[Alert]:
    """
    Check for excessive equity drawdown.

    Args:
        session: Database session
        max_dd_threshold: Maximum acceptable drawdown (%)

    Returns:
        List of alerts (empty if no issues)

    Example:
        >>> alerts = check_equity_drawdown(session, max_dd_threshold=-15.0)
        >>> if alerts:
        ...     print(f"Drawdown alert: {alerts[0].message}")
    """
    alerts = []

    # TODO: Query EquitySnapshot table and calculate current drawdown
    # from ..daemon.portfolio_state import calculate_current_drawdown
    # current_dd = calculate_current_drawdown(session)
    #
    # if current_dd < max_dd_threshold:
    #     alerts.append(Alert(
    #         level=AlertLevel.CRIT,
    #         message=f"Equity drawdown {current_dd:.1f}% exceeds threshold {max_dd_threshold:.1f}%",
    #         category="drawdown",
    #         details={"current_dd": current_dd, "threshold": max_dd_threshold}
    #     ))

    return alerts


def check_daemon_health(
    session: Session,
    max_staleness_hours: int = 2,
) -> List[Alert]:
    """
    Check daemon liveness and health.

    Args:
        session: Database session
        max_staleness_hours: Maximum hours since last daemon activity

    Returns:
        List of alerts (empty if healthy)
    """
    alerts = []

    # TODO: Query DaemonState or EquitySnapshot for last update
    # from ..daemon.state import get_last_daemon_heartbeat
    # last_heartbeat = get_last_daemon_heartbeat(session)
    #
    # if last_heartbeat:
    #     staleness_hours = (datetime.utcnow() - last_heartbeat).total_seconds() / 3600
    #     if staleness_hours > max_staleness_hours:
    #         alerts.append(Alert(
    #             level=AlertLevel.CRIT,
    #             message=f"Daemon inactive for {staleness_hours:.1f} hours",
    #             category="daemon",
    #             details={"last_heartbeat": last_heartbeat, "staleness_hours": staleness_hours}
    #         ))

    return alerts


def check_no_trade_period(
    session: Session,
    max_days_without_trade: int = 7,
) -> List[Alert]:
    """
    Check for prolonged periods without trading activity.

    Args:
        session: Database session
        max_days_without_trade: Maximum days without a trade

    Returns:
        List of alerts (empty if recent trades)
    """
    alerts = []

    # TODO: Query Decision or Trade table for last trade
    # from ..data import Decision
    # last_trade = session.query(Decision).order_by(Decision.timestamp.desc()).first()
    #
    # if last_trade:
    #     days_since_trade = (datetime.utcnow() - last_trade.timestamp).days
    #     if days_since_trade > max_days_without_trade:
    #         alerts.append(Alert(
    #             level=AlertLevel.WARN,
    #             message=f"No trades for {days_since_trade} days",
    #             category="activity",
    #             details={"days_since_trade": days_since_trade, "last_trade": last_trade.timestamp}
    #         ))

    return alerts


def check_data_staleness(
    session: Session,
    max_staleness_hours: int = 24,
) -> List[Alert]:
    """
    Check for stale market data.

    Args:
        session: Database session
        max_staleness_hours: Maximum hours since last data update

    Returns:
        List of alerts (empty if data is fresh)
    """
    alerts = []

    # TODO: Query DailyBar table for most recent data
    # from ..data import DailyBar
    # latest_bar = session.query(DailyBar).order_by(DailyBar.date.desc()).first()
    #
    # if latest_bar:
    #     staleness_hours = (datetime.utcnow() - latest_bar.date).total_seconds() / 3600
    #     if staleness_hours > max_staleness_hours:
    #         alerts.append(Alert(
    #             level=AlertLevel.WARN,
    #             message=f"Market data is {staleness_hours:.1f} hours stale",
    #             category="data",
    #             details={"last_update": latest_bar.date, "staleness_hours": staleness_hours}
    #         ))

    return alerts


def check_alerts(
    session: Session,
    max_dd_threshold: float = -20.0,
    max_daemon_staleness_hours: int = 2,
    max_days_without_trade: int = 7,
    max_data_staleness_hours: int = 24,
) -> List[Alert]:
    """
    Run all health checks and return alerts.

    Args:
        session: Database session
        max_dd_threshold: Maximum acceptable drawdown (%)
        max_daemon_staleness_hours: Maximum hours since daemon heartbeat
        max_days_without_trade: Maximum days without trading
        max_data_staleness_hours: Maximum hours since data update

    Returns:
        List of all alerts found

    Example:
        >>> with get_session() as session:
        ...     alerts = check_alerts(session)
        ...     for alert in alerts:
        ...         print(f"{alert.level}: {alert.message}")
    """
    all_alerts = []

    # Run all checks
    all_alerts.extend(check_equity_drawdown(session, max_dd_threshold))
    all_alerts.extend(check_daemon_health(session, max_daemon_staleness_hours))
    all_alerts.extend(check_no_trade_period(session, max_days_without_trade))
    all_alerts.extend(check_data_staleness(session, max_data_staleness_hours))

    # Log summary
    if all_alerts:
        critical_count = sum(1 for a in all_alerts if a.level == AlertLevel.CRIT)
        warning_count = sum(1 for a in all_alerts if a.level == AlertLevel.WARN)
        info_count = sum(1 for a in all_alerts if a.level == AlertLevel.INFO)

        logger.warning(
            f"Health check complete: {len(all_alerts)} alerts "
            f"(CRIT={critical_count}, WARN={warning_count}, INFO={info_count})"
        )
    else:
        logger.info("Health check complete: All systems nominal")

    return all_alerts


def send_alert(alert: Alert, notification_channels: Optional[List[str]] = None) -> bool:
    """
    Send alert notification via configured channels.

    Args:
        alert: Alert to send
        notification_channels: List of channels (e.g., ["email", "slack", "telegram"])

    Returns:
        True if sent successfully, False otherwise

    Example:
        >>> alert = Alert(level=AlertLevel.CRIT, message="Critical drawdown")
        >>> send_alert(alert, channels=["email", "slack"])
    """
    if notification_channels is None:
        notification_channels = ["log"]  # Default to logging only

    success = True

    for channel in notification_channels:
        if channel == "log":
            # Always log alerts
            log_func = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARN: logger.warning,
                AlertLevel.CRIT: logger.critical,
            }.get(alert.level, logger.info)

            log_func(f"ALERT [{alert.category}]: {alert.message}")

        elif channel == "email":
            # TODO: Implement email notification
            logger.debug(f"Email alert not implemented: {alert.message}")
            success = False

        elif channel == "slack":
            # TODO: Implement Slack webhook
            logger.debug(f"Slack alert not implemented: {alert.message}")
            success = False

        elif channel == "telegram":
            # TODO: Implement Telegram bot
            logger.debug(f"Telegram alert not implemented: {alert.message}")
            success = False

        else:
            logger.warning(f"Unknown notification channel: {channel}")
            success = False

    return success
