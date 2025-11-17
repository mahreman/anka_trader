"""
Base interface for data providers.
"""
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for data providers.

    Each provider must implement:
    - fetch_data(): Fetch data for a given date range
    - get_name(): Return provider name
    """

    def __init__(self, config: dict):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config

    @abstractmethod
    def fetch_data(
        self,
        start: date,
        end: Optional[date] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data for given date range.

        Args:
            start: Start date
            end: End date (if None, defaults to today)
            **kwargs: Additional provider-specific parameters

        Returns:
            DataFrame with fetched data
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return provider name.

        Returns:
            Provider name string
        """
        pass
