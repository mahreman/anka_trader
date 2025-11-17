"""
FRED (Federal Reserve Economic Data) provider for macroeconomic indicators.
"""
import logging
from datetime import date
from typing import Optional

import pandas as pd
import requests

from .base import DataProvider

logger = logging.getLogger(__name__)


class FREDProvider(DataProvider):
    """
    FRED data provider for macroeconomic indicators.

    Uses FRED API to fetch economic time series data.
    API docs: https://fred.stlouisfed.org/docs/api/fred/
    """

    def __init__(self, config: dict):
        """
        Initialize FRED provider.

        Args:
            config: Configuration dictionary with:
                - api_key: FRED API key (required, get from https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("FRED requires an api_key in config")

        self.base_url = "https://api.stlouisfed.org/fred"

    def get_name(self) -> str:
        return "fred"

    def fetch_data(
        self,
        start: date,
        end: Optional[date] = None,
        series_id: str = "DGS10",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch economic time series from FRED.

        Args:
            start: Start date
            end: End date (default: today)
            series_id: FRED series ID (e.g., "DGS10" for 10-Year Treasury, "CPIAUCSL" for CPI)
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns: [date, value, series_id, series_name, units]
        """
        if end is None:
            end = date.today()

        logger.info(f"Fetching FRED data for series={series_id} from {start} to {end}")

        # First, get series info (name, units, etc.)
        series_info = self._get_series_info(series_id)

        # Then fetch the actual observations
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])

            if not observations:
                logger.warning(f"No observations returned for series={series_id}")
                return pd.DataFrame()

            # Parse observations
            records = []
            for obs in observations:
                value_str = obs.get("value", ".")
                # FRED uses "." for missing values
                if value_str == ".":
                    continue

                try:
                    value = float(value_str)
                except ValueError:
                    logger.warning(f"Invalid value '{value_str}' for series={series_id}, skipping")
                    continue

                records.append({
                    "date": pd.to_datetime(obs["date"]).date(),
                    "value": value,
                    "series_id": series_id,
                    "series_name": series_info.get("title", series_id),
                    "units": series_info.get("units", ""),
                })

            if not records:
                logger.warning(f"No valid observations for series={series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(records)

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} observations for series={series_id}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API error for series={series_id}: {e}")
            raise

    def _get_series_info(self, series_id: str) -> dict:
        """
        Get metadata for a FRED series.

        Args:
            series_id: FRED series ID

        Returns:
            Dictionary with series metadata (title, units, etc.)
        """
        url = f"{self.base_url}/series"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            series_list = data.get("seriess", [])
            if not series_list:
                return {}

            return series_list[0]

        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch series info for {series_id}: {e}")
            return {}
