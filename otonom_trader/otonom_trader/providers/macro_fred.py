"""
FRED (Federal Reserve Economic Data) provider.

Fetches macroeconomic indicators from FRED API.
API Docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Dict, Any

import requests

from .base import MacroProvider, MacroIndicator, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


class FREDProvider(MacroProvider):
    """
    FRED (Federal Reserve Economic Data) provider.

    Free API key available from: https://fred.stlouisfed.org/docs/api/api_key.html

    Supports thousands of economic indicators:
    - Interest rates (DFF, GS10, T10Y2Y)
    - Inflation (CPIAUCSL, PCEPI)
    - Employment (UNRATE, PAYEMS)
    - GDP (GDP, GDPC1)
    - And many more
    """

    # Common FRED indicators
    COMMON_INDICATORS = {
        "DFF": "Federal Funds Effective Rate",
        "GS10": "10-Year Treasury Constant Maturity Rate",
        "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
        "GDP": "Gross Domestic Product",
        "PAYEMS": "Total Nonfarm Payrolls",
        "INDPRO": "Industrial Production Index",
        "HOUST": "Housing Starts",
        "RSXFS": "Retail Sales",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            logger.warning("FRED: No API key configured (get free key from fred.stlouisfed.org)")

        self.base_url = config.get("base_url", "https://api.stlouisfed.org/fred")

        logger.info("FREDProvider initialized")

    def _request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make FRED API request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response

        Raises:
            ProviderError: If request fails
            RateLimitError: If rate limited
        """
        url = f"{self.base_url}/{endpoint}"

        # Add API key
        params["api_key"] = self.api_key
        params["file_type"] = "json"

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.config.get("timeout_seconds", 30),
            )

            # Check rate limiting (FRED: 120 req/min)
            if response.status_code == 429:
                raise RateLimitError("FRED rate limit exceeded")

            # Check authentication
            if response.status_code == 400:
                error = response.json().get("error_message", "Unknown error")
                if "api_key" in error.lower():
                    raise ProviderError(f"FRED authentication failed: {error}")

            # Check for errors
            if response.status_code != 200:
                raise ProviderError(f"FRED API error: HTTP {response.status_code}")

            return response.json()

        except requests.RequestException as e:
            raise ProviderError(f"FRED request failed: {e}")

    def fetch_indicator(
        self,
        indicator_code: str,
        start_date: date,
        end_date: date,
    ) -> List[MacroIndicator]:
        """
        Fetch economic indicator time series from FRED.

        Args:
            indicator_code: FRED series ID (e.g., "GDP", "UNRATE", "DFF")
            start_date: Start date
            end_date: End date

        Returns:
            List of indicator observations

        Example:
            >>> provider = FREDProvider({"api_key": "..."})
            >>> indicators = provider.fetch_indicator("UNRATE", date(2023, 1, 1), date(2024, 1, 1))
            >>> print(f"Fetched {len(indicators)} unemployment rate observations")
        """
        logger.info(f"Fetching FRED indicator: {indicator_code} from {start_date} to {end_date}")

        # First, get series info to get metadata
        series_params = {"series_id": indicator_code}

        try:
            series_data = self._request("series", series_params)
        except ProviderError as e:
            logger.error(f"Failed to fetch FRED series info: {e}")
            raise

        series_info = series_data.get("seriess", [{}])[0]
        indicator_name = series_info.get("title", indicator_code)
        frequency = series_info.get("frequency_short", "Unknown")
        units = series_info.get("units", "Unknown")

        # Fetch observations
        obs_params = {
            "series_id": indicator_code,
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "sort_order": "asc",
        }

        try:
            obs_data = self._request("series/observations", obs_params)
        except ProviderError as e:
            logger.error(f"Failed to fetch FRED observations: {e}")
            raise

        # Parse observations
        indicators = []
        for obs in obs_data.get("observations", []):
            obs_date_str = obs.get("date")
            value_str = obs.get("value")

            if not obs_date_str or not value_str or value_str == ".":
                continue  # Skip missing values

            # Parse date
            obs_date = datetime.strptime(obs_date_str, "%Y-%m-%d").date()

            # Parse value
            try:
                value = float(value_str)
            except ValueError:
                continue

            # Create indicator
            indicator = MacroIndicator(
                indicator_code=indicator_code,
                name=indicator_name,
                date=obs_date,
                value=value,
                unit=units,
                frequency=frequency,
            )

            indicators.append(indicator)

        logger.info(f"Fetched {len(indicators)} FRED observations for {indicator_code}")

        return indicators

    def get_available_indicators(self) -> List[Dict[str, str]]:
        """
        Get list of common FRED indicators.

        Returns:
            List of dicts with 'code', 'name', 'frequency'

        Example:
            >>> indicators = provider.get_available_indicators()
            >>> for ind in indicators:
            ...     print(f"{ind['code']}: {ind['name']}")
        """
        # Return common indicators
        indicators = []

        for code, name in self.COMMON_INDICATORS.items():
            indicators.append({
                "code": code,
                "name": name,
                "frequency": "Various",  # Would need separate API call per indicator
            })

        logger.info(f"Returning {len(indicators)} common FRED indicators")

        return indicators

    def fetch_latest_value(self, indicator_code: str) -> MacroIndicator:
        """
        Fetch latest value for a FRED indicator.

        Args:
            indicator_code: FRED series ID

        Returns:
            Latest indicator observation

        Example:
            >>> latest = provider.fetch_latest_value("UNRATE")
            >>> print(f"Current unemployment rate: {latest.value}%")
        """
        logger.info(f"Fetching latest FRED value: {indicator_code}")

        # Get series info
        series_params = {"series_id": indicator_code}

        try:
            series_data = self._request("series", series_params)
        except ProviderError as e:
            logger.error(f"Failed to fetch FRED series info: {e}")
            raise

        series_info = series_data.get("seriess", [{}])[0]
        indicator_name = series_info.get("title", indicator_code)
        frequency = series_info.get("frequency_short", "Unknown")
        units = series_info.get("units", "Unknown")

        # Fetch latest observation
        obs_params = {
            "series_id": indicator_code,
            "sort_order": "desc",
            "limit": 1,
        }

        try:
            obs_data = self._request("series/observations", obs_params)
        except ProviderError as e:
            logger.error(f"Failed to fetch FRED latest observation: {e}")
            raise

        # Parse latest observation
        observations = obs_data.get("observations", [])
        if not observations:
            raise ProviderError(f"No observations found for {indicator_code}")

        obs = observations[0]
        obs_date_str = obs.get("date")
        value_str = obs.get("value")

        if not obs_date_str or not value_str or value_str == ".":
            raise ProviderError(f"Invalid observation data for {indicator_code}")

        # Parse
        obs_date = datetime.strptime(obs_date_str, "%Y-%m-%d").date()
        value = float(value_str)

        # Create indicator
        indicator = MacroIndicator(
            indicator_code=indicator_code,
            name=indicator_name,
            date=obs_date,
            value=value,
            unit=units,
            frequency=frequency,
        )

        logger.info(f"Latest {indicator_code}: {value} {units} as of {obs_date}")

        return indicator

    def fetch_multiple_indicators(
        self,
        indicator_codes: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, List[MacroIndicator]]:
        """
        Fetch multiple indicators efficiently.

        Args:
            indicator_codes: List of FRED series IDs
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping indicator code to list of observations

        Example:
            >>> codes = ["DFF", "UNRATE", "GDP"]
            >>> data = provider.fetch_multiple_indicators(codes, start, end)
            >>> print(f"Federal Funds Rate: {len(data['DFF'])} observations")
        """
        results = {}

        for code in indicator_codes:
            try:
                indicators = self.fetch_indicator(code, start_date, end_date)
                results[code] = indicators
            except ProviderError as e:
                logger.warning(f"Failed to fetch {code}: {e}")
                results[code] = []

        return results
