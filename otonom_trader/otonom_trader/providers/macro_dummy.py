"""
Dummy macro provider for testing.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Dict, Any
import random

from .base import MacroProvider, MacroIndicator

logger = logging.getLogger(__name__)


class DummyMacroProvider(MacroProvider):
    """Dummy macro provider that returns synthetic economic data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("DummyMacroProvider initialized (returns synthetic data)")

    def fetch_indicator(
        self,
        indicator_code: str,
        start_date: date,
        end_date: date,
    ) -> List[MacroIndicator]:
        """Generate synthetic indicator time series."""
        logger.warning(f"DummyMacroProvider returning synthetic data for {indicator_code}")

        indicators = []
        current_date = start_date
        value = random.uniform(1.0, 5.0)  # Starting value

        # Generate monthly data
        while current_date <= end_date:
            # Random walk
            value = max(0.1, value + random.uniform(-0.2, 0.2))

            indicator = MacroIndicator(
                indicator_code=indicator_code,
                name=f"Dummy {indicator_code}",
                date=current_date,
                value=value,
                unit="Percent",
                frequency="Monthly",
            )

            indicators.append(indicator)

            # Move to next month
            if current_date.month == 12:
                current_date = date(current_date.year + 1, 1, 1)
            else:
                current_date = date(current_date.year, current_date.month + 1, 1)

        return indicators

    def get_available_indicators(self) -> List[Dict[str, str]]:
        """Return dummy indicator list."""
        return [
            {"code": "DUMMY_GDP", "name": "Dummy GDP", "frequency": "Quarterly"},
            {"code": "DUMMY_UNRATE", "name": "Dummy Unemployment", "frequency": "Monthly"},
        ]

    def fetch_latest_value(self, indicator_code: str) -> MacroIndicator:
        """Generate synthetic latest value."""
        return MacroIndicator(
            indicator_code=indicator_code,
            name=f"Dummy {indicator_code}",
            date=date.today(),
            value=random.uniform(1.0, 5.0),
            unit="Percent",
            frequency="Monthly",
        )
