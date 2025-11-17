"""
Return calculations and statistical metrics.
"""
import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute daily log returns and add to DataFrame.

    Args:
        df: DataFrame with OHLCV data
        price_col: Column name for price (default: 'close')

    Returns:
        DataFrame with added 'ret' column (log returns)
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Sort by date to ensure correct order
    if "date" in df.columns:
        df = df.sort_values("date")

    # Calculate log returns
    df["ret"] = np.log(df[price_col] / df[price_col].shift(1))

    # First row will be NaN, which is expected
    return df


def compute_rolling_stats(
    df: pd.DataFrame, window: int = 60, ret_col: str = "ret"
) -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation for returns.

    Args:
        df: DataFrame with returns
        window: Rolling window size (days)
        ret_col: Column name for returns

    Returns:
        DataFrame with added rolling statistics columns
    """
    if ret_col not in df.columns:
        raise ValueError(f"Column '{ret_col}' not found in DataFrame")

    df = df.copy()

    # Calculate rolling statistics
    df["rolling_mean"] = df[ret_col].rolling(window=window, min_periods=window).mean()
    df["rolling_std"] = df[ret_col].rolling(window=window, min_periods=window).std()

    # Calculate z-score
    df["ret_zscore"] = (df[ret_col] - df["rolling_mean"]) / df["rolling_std"]

    return df


def compute_volume_quantile(
    df: pd.DataFrame, window: int = 60, volume_col: str = "volume"
) -> pd.DataFrame:
    """
    Compute rolling volume quantile (percentile rank).

    Args:
        df: DataFrame with volume data
        window: Rolling window size (days)
        volume_col: Column name for volume

    Returns:
        DataFrame with added 'volume_quantile' column (0-1)
    """
    if volume_col not in df.columns:
        raise ValueError(f"Column '{volume_col}' not found in DataFrame")

    df = df.copy()

    # Calculate rolling quantile using rank
    def rolling_quantile(series, window):
        """Calculate percentile rank within rolling window."""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window - 1:
                result.iloc[i] = np.nan
            else:
                window_data = series.iloc[i - window + 1 : i + 1]
                current_value = series.iloc[i]
                # Percentile rank: fraction of values <= current value
                rank = (window_data <= current_value).sum() / len(window_data)
                result.iloc[i] = rank
        return result

    df["volume_quantile"] = rolling_quantile(df[volume_col], window)

    return df
