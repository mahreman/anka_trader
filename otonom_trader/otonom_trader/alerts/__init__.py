"""
Alert engine for sending notifications via email and Telegram.

Supports:
- Email notifications via SMTP
- Telegram notifications via Bot API
- Broker errors
- Kill-switch triggers
- Daemon health checks
"""

from .notifier import Notifier
from .engine import AlertEngine

__all__ = ["Notifier", "AlertEngine"]
