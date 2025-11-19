"""Datetime helpers to enforce timezone-aware UTC usage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)


def ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
    """Ensure that the provided datetime has UTC timezone information."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
