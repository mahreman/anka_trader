"""
Economic calendar client for Analist-2 (P2).

This module provides macroeconomic event data for fundamental analysis.
Initial version uses simple CSV/JSON log format.
Can be upgraded to real calendar APIs (e.g., Trading Economics, FRED) later.

P2 Features:
- Load macro events from CSV/JSON files
- Filter events by date and region
- Impact scoring (LOW/MEDIUM/HIGH)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EventImpact(str, Enum):
    """Impact level of a macroeconomic event."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class MacroEvent:
    """
    Represents a macroeconomic event (e.g., CPI, NFP, Fed decision).

    Attributes:
        date: Event date
        event_name: Name of the event (e.g., "US CPI YoY")
        region: Region/country (e.g., "US", "EU", "GLOBAL")
        impact: Expected impact level (LOW/MEDIUM/HIGH)
        actual: Actual value (if released)
        forecast: Forecasted value
        previous: Previous value
        direction: Direction relative to forecast (+1=beat, 0=met, -1=miss)
        notes: Additional notes
    """
    date: date
    event_name: str
    region: str = "GLOBAL"
    impact: EventImpact = EventImpact.MEDIUM
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    direction: int = 0  # +1 beat, 0 met, -1 miss
    notes: str = ""

    def __post_init__(self):
        """Auto-compute direction if actual and forecast are available."""
        if self.actual is not None and self.forecast is not None:
            if self.actual > self.forecast:
                self.direction = 1
            elif self.actual < self.forecast:
                self.direction = -1
            else:
                self.direction = 0


def load_calendar_from_csv(
    csv_path: Path | str,
    region: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_impact: EventImpact = EventImpact.LOW,
) -> List[MacroEvent]:
    """
    Load macro events from CSV file.

    CSV Format:
        date,event_name,region,impact,actual,forecast,previous,notes
        2025-01-15,US CPI YoY,US,HIGH,3.2,3.0,2.9,"Inflation beat forecast"

    Args:
        csv_path: Path to CSV file
        region: Filter by region (optional, e.g., "US", "EU")
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)
        min_impact: Minimum impact level to include

    Returns:
        List of MacroEvent objects

    Example:
        >>> events = load_calendar_from_csv("data/calendar.csv", region="US")
        >>> print(f"Loaded {len(events)} US events")
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.warning(f"Calendar CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Apply filters
        if region:
            df = df[(df["region"] == region) | (df["region"] == "GLOBAL")]

        if start_date:
            df = df[df["date"] >= start_date]

        if end_date:
            df = df[df["date"] <= end_date]

        # Convert to MacroEvent objects
        items = []
        impact_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_impact_level = impact_order.get(min_impact.value, 0)

        for _, row in df.iterrows():
            impact_str = row.get("impact", "MEDIUM").upper()
            impact = EventImpact(impact_str) if impact_str in ["LOW", "MEDIUM", "HIGH"] else EventImpact.MEDIUM

            # Skip if impact too low
            if impact_order.get(impact.value, 0) < min_impact_level:
                continue

            items.append(
                MacroEvent(
                    date=row["date"],
                    event_name=row["event_name"],
                    region=row.get("region", "GLOBAL"),
                    impact=impact,
                    actual=row.get("actual", None) if not pd.isna(row.get("actual", None)) else None,
                    forecast=row.get("forecast", None) if not pd.isna(row.get("forecast", None)) else None,
                    previous=row.get("previous", None) if not pd.isna(row.get("previous", None)) else None,
                    notes=row.get("notes", ""),
                )
            )

        logger.info(f"Loaded {len(items)} macro events from {csv_path}")
        return items

    except Exception as e:
        logger.error(f"Failed to load calendar CSV: {e}")
        return []


def load_calendar_from_json(
    json_path: Path | str,
    region: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_impact: EventImpact = EventImpact.LOW,
) -> List[MacroEvent]:
    """
    Load macro events from JSON file.

    JSON Format:
        [
            {
                "date": "2025-01-15",
                "event_name": "US CPI YoY",
                "region": "US",
                "impact": "HIGH",
                "actual": 3.2,
                "forecast": 3.0,
                "previous": 2.9,
                "notes": "Inflation beat forecast"
            },
            ...
        ]

    Args:
        json_path: Path to JSON file
        region: Filter by region (optional)
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)
        min_impact: Minimum impact level to include

    Returns:
        List of MacroEvent objects
    """
    json_path = Path(json_path)

    if not json_path.exists():
        logger.warning(f"Calendar JSON not found: {json_path}")
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        items = []
        impact_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_impact_level = impact_order.get(min_impact.value, 0)

        for item in data:
            item_date = pd.to_datetime(item["date"]).date()
            item_region = item.get("region", "GLOBAL")

            # Apply filters
            if region and item_region != region and item_region != "GLOBAL":
                continue

            if start_date and item_date < start_date:
                continue

            if end_date and item_date > end_date:
                continue

            impact_str = item.get("impact", "MEDIUM").upper()
            impact = EventImpact(impact_str) if impact_str in ["LOW", "MEDIUM", "HIGH"] else EventImpact.MEDIUM

            # Skip if impact too low
            if impact_order.get(impact.value, 0) < min_impact_level:
                continue

            items.append(
                MacroEvent(
                    date=item_date,
                    event_name=item["event_name"],
                    region=item_region,
                    impact=impact,
                    actual=item.get("actual", None),
                    forecast=item.get("forecast", None),
                    previous=item.get("previous", None),
                    notes=item.get("notes", ""),
                )
            )

        logger.info(f"Loaded {len(items)} macro events from {json_path}")
        return items

    except Exception as e:
        logger.error(f"Failed to load calendar JSON: {e}")
        return []


def get_upcoming_events(
    days_ahead: int = 7,
    region: Optional[str] = None,
    min_impact: EventImpact = EventImpact.MEDIUM,
    data_dir: Path | str = "data/calendar",
) -> List[MacroEvent]:
    """
    Get upcoming macro events from all available sources.

    Args:
        days_ahead: Number of days to look ahead
        region: Filter by region (optional)
        min_impact: Minimum impact level to include
        data_dir: Directory containing calendar files

    Returns:
        List of MacroEvent objects sorted by date

    Example:
        >>> events = get_upcoming_events(days_ahead=7, region="US", min_impact=EventImpact.HIGH)
        >>> if events:
        ...     print(f"Next HIGH event: {events[0].event_name} on {events[0].date}")
    """
    start_date = date.today()
    end_date = start_date + timedelta(days=days_ahead)

    data_dir = Path(data_dir)
    all_events = []

    # Try to load from CSV
    csv_path = data_dir / "calendar.csv"
    if csv_path.exists():
        all_events.extend(
            load_calendar_from_csv(
                csv_path, region=region, start_date=start_date, end_date=end_date, min_impact=min_impact
            )
        )

    # Try to load from JSON
    json_path = data_dir / "calendar.json"
    if json_path.exists():
        all_events.extend(
            load_calendar_from_json(
                json_path, region=region, start_date=start_date, end_date=end_date, min_impact=min_impact
            )
        )

    # Sort by date
    all_events.sort(key=lambda e: e.date)

    return all_events


def get_recent_events(
    days_back: int = 7,
    region: Optional[str] = None,
    min_impact: EventImpact = EventImpact.MEDIUM,
    data_dir: Path | str = "data/calendar",
) -> List[MacroEvent]:
    """
    Get recent macro events (already released).

    Args:
        days_back: Number of days to look back
        region: Filter by region (optional)
        min_impact: Minimum impact level to include
        data_dir: Directory containing calendar files

    Returns:
        List of MacroEvent objects sorted by date (newest first)

    Example:
        >>> events = get_recent_events(days_back=7, region="US")
        >>> if events:
        ...     print(f"Latest: {events[0].event_name} on {events[0].date}")
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    data_dir = Path(data_dir)
    all_events = []

    # Try to load from CSV
    csv_path = data_dir / "calendar.csv"
    if csv_path.exists():
        all_events.extend(
            load_calendar_from_csv(
                csv_path, region=region, start_date=start_date, end_date=end_date, min_impact=min_impact
            )
        )

    # Try to load from JSON
    json_path = data_dir / "calendar.json"
    if json_path.exists():
        all_events.extend(
            load_calendar_from_json(
                json_path, region=region, start_date=start_date, end_date=end_date, min_impact=min_impact
            )
        )

    # Sort by date (newest first)
    all_events.sort(key=lambda e: e.date, reverse=True)

    return all_events


def compute_macro_bias(
    events: List[MacroEvent],
    weights: Optional[dict] = None,
) -> float:
    """
    Compute macro bias from recent events.

    Bias is computed based on:
    - Event direction (+1 beat, -1 miss)
    - Event impact (HIGH=3, MEDIUM=2, LOW=1)

    Args:
        events: List of MacroEvent objects
        weights: Optional custom weights by impact level

    Returns:
        Macro bias in [-1, 1] where +1=bullish, -1=bearish

    Example:
        >>> events = get_recent_events(days_back=7, region="US")
        >>> bias = compute_macro_bias(events)
        >>> print(f"7-day macro bias: {bias:.2f}")
    """
    if not events:
        return 0.0

    if weights is None:
        # Default weights by impact
        weights = {
            EventImpact.HIGH: 3.0,
            EventImpact.MEDIUM: 2.0,
            EventImpact.LOW: 1.0,
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for event in events:
        if event.direction == 0:
            continue  # Met expectations, neutral

        weight = weights.get(event.impact, 1.0)
        weighted_sum += event.direction * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Normalize to [-1, 1]
    bias = weighted_sum / total_weight
    return max(-1.0, min(1.0, bias))
