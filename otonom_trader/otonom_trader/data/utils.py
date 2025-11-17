"""
Utility functions for data operations.
"""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


def parse_date_string(date_str: Optional[str]) -> Optional[date]:
    """
    Parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date string or None

    Returns:
        date object or None
    """
    if date_str is None:
        return None

    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"Invalid date format '{date_str}': {e}")
        raise ValueError(f"Date must be in YYYY-MM-DD format, got: {date_str}")


def get_date_range(
    start: Optional[str] = None, end: Optional[str] = None
) -> Tuple[date, date]:
    """
    Get date range with defaults.

    Args:
        start: Start date string (YYYY-MM-DD) or None for default
        end: End date string (YYYY-MM-DD) or None for today

    Returns:
        Tuple of (start_date, end_date)
    """
    from ..config import DEFAULT_START_DATE

    start_date = parse_date_string(start) if start else parse_date_string(DEFAULT_START_DATE)
    end_date = parse_date_string(end) if end else date.today()

    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")

    return start_date, end_date


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplication factor for delay
        exceptions: Tuple of exceptions to catch

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                logger.error(f"All {max_retries} retry attempts failed")
                raise

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= backoff_factor


def calculate_days_ago(days: int) -> date:
    """
    Calculate date N days ago from today.

    Args:
        days: Number of days to go back

    Returns:
        Date object
    """
    return date.today() - timedelta(days=days)
