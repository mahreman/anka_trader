"""
Notification service for sending alerts via email and Telegram.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

logger = logging.getLogger(__name__)


class Notifier:
    """
    Sends notifications via email and Telegram.
    
    Configuration is loaded from a YAML file with email and telegram sections.
    
    Example config (config/alerts.yaml):
        email:
          enabled: true
          smtp_host: "smtp.gmail.com"
          smtp_port: 587
          username: "your@email"
          password: "app_password"
          from_addr: "your@email"
          to_addrs:
            - "you@domain.com"
        
        telegram:
          enabled: true
          bot_token: "YOUR_BOT_TOKEN"
          chat_id: "123456789"
    """

    def __init__(self, config_path: str | Path = "config/alerts.yaml"):
        """
        Initialize notifier.
        
        Args:
            config_path: Path to alerts configuration YAML
        """
        self.config = self._load_config(config_path)
        logger.info(f"Initialized Notifier from {config_path}")

    def _load_config(self, path: str | Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"Alert config not found: {path}, notifications disabled")
            return {}
        
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ==================== Email ====================

    def send_email(self, subject: str, body: str) -> None:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Email body (plain text)
        """
        cfg = self.config.get("email") or {}
        if not cfg.get("enabled", False):
            logger.debug("Email notifications disabled, skipping")
            return

        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = cfg["from_addr"]
            msg["To"] = ", ".join(cfg["to_addrs"])

            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["username"], cfg["password"])
                server.sendmail(cfg["from_addr"], cfg["to_addrs"], msg.as_string())

            logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    # ==================== Telegram ====================

    def send_telegram(self, text: str) -> None:
        """
        Send Telegram notification.
        
        Args:
            text: Message text
        """
        cfg = self.config.get("telegram") or {}
        if not cfg.get("enabled", False):
            logger.debug("Telegram notifications disabled, skipping")
            return

        try:
            token = cfg["bot_token"]
            chat_id = cfg["chat_id"]
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",  # Support HTML formatting
            }
            
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            
            logger.info("Telegram message sent")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    # ==================== High-level Interface ====================

    def notify(self, subject: str, body: str) -> None:
        """
        Send notification via all enabled channels.
        
        Args:
            subject: Notification subject/title
            body: Notification body/message
        """
        logger.info(f"Sending notification: {subject}")
        
        # Send email
        self.send_email(subject, body)
        
        # Send Telegram (combine subject and body)
        telegram_text = f"<b>{subject}</b>\n\n{body}"
        self.send_telegram(telegram_text)
