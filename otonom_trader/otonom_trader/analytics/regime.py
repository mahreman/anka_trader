"""
Regime detection and structural break detection for Analist-1.

This module extends the P0 technical analytics layer with:
- Regime detection based on rolling volatility and trend.
- Simple CUSUM-based structural break detection.

It is intentionally simple and dependency-light so it can run
on the existing P0 data model (DailyBar + returns).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .returns import compute_returns
from ..data.schema import DailyBar, Symbol

logger = logging.getLogger(__name__)


@dataclass
class RegimePoint:
    """
    Represents a regime classification for a specific date.

    Attributes:
        symbol: Asset symbol
        trade_date: Date of the regime point
        regime_id: Integer regime identifier (0, 1, 2 for low/normal/high vol)
        volatility: Rolling volatility value
        trend: Rolling trend value
        is_structural_break: Whether a structural break was detected
    """
    symbol: str
    trade_date: date
    regime_id: int
    volatility: float
    trend: float
    is_structural_break: bool


def _load_price_series(
    session: Session, symbol: str
) -> pd.DataFrame:
    """
    Load OHLCV data for a given symbol into a DataFrame.

    This uses the P0 schema:
    - DailyBar with columns: date, open, high, low, close, volume
    - Symbol with symbol field that identifies the asset.

    Args:
        session: Database session
        symbol: Asset symbol (e.g., 'BTC-USD')

    Returns:
        DataFrame with OHLCV data indexed by date
    """
    logger.info(f"Loading price series for symbol={symbol}")

    q = (
        session.query(
            DailyBar.date,
            DailyBar.open,
            DailyBar.high,
            DailyBar.low,
            DailyBar.close,
            DailyBar.volume
        )
        .join(Symbol, Symbol.id == DailyBar.symbol_id)
        .filter(Symbol.symbol == symbol)
        .order_by(DailyBar.date.asc())
    )

    rows = q.all()
    if not rows:
        raise ValueError(f"No DailyBar data found for symbol={symbol}")

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df.set_index("date", inplace=True)
    return df


def _build_regime_features(
    df: pd.DataFrame,
    price_col: str = "close",
    vol_window: int = 20,
    trend_window: int = 20,
) -> pd.DataFrame:
    """
    Build regime features: rolling volatility and trend.

    Args:
        df: DataFrame with OHLCV data
        price_col: Column to use for price (default: 'close')
        vol_window: Window for volatility calculation
        trend_window: Window for trend calculation

    Returns:
        DataFrame with 'volatility' and 'trend' columns

    Features:
        - volatility: rolling std of log returns over vol_window
        - trend: rolling mean of returns over trend_window
    """
    # Reset index to have 'date' as column for compute_returns compatibility
    df_reset = df.reset_index()
    df_ret = compute_returns(df_reset, price_col=price_col)

    # Calculate rolling volatility (std of returns)
    rolling_vol = (
        df_ret["ret"]
        .rolling(window=vol_window, min_periods=vol_window)
        .std()
    )

    # Calculate rolling trend (mean of returns)
    rolling_trend = (
        df_ret["ret"]
        .rolling(window=trend_window, min_periods=trend_window)
        .mean()
    )

    features = pd.DataFrame(index=df_ret["date"])
    features["volatility"] = rolling_vol.values
    features["trend"] = rolling_trend.values

    # Drop initial NaNs where windows are not full
    features = features.dropna()
    return features


def _kmeans_1d(x: np.ndarray, k: int, n_iter: int = 50) -> np.ndarray:
    """
    Ultra-simple 1D k-means used only for volatility clustering.

    This is intentionally naive to avoid extra dependencies.

    Args:
        x: 1D array of values to cluster
        k: Number of clusters
        n_iter: Number of iterations

    Returns:
        Array of cluster labels (0 to k-1)
    """
    x = x.reshape(-1, 1)
    # Initialize centers as quantiles
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(x, qs).reshape(-1, 1)

    for _ in range(n_iter):
        # Assign
        distances = np.abs(x - centers.T)
        labels = np.argmin(distances, axis=1)

        # Update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean()

    return labels


def _detect_cusum(
    series: pd.Series,
    threshold: float,
    drift: float = 0.0,
) -> pd.Series:
    """
    Simple CUSUM detector for structural breaks on a 1D signal.

    Args:
        series: Time series to analyze
        threshold: Detection threshold
        drift: Drift parameter (default: 0)

    Returns:
        Boolean Series: True where a structural break is detected.
    """
    pos = 0.0
    neg = 0.0
    flags = np.zeros(len(series), dtype=bool)

    for i, x in enumerate(series.values):
        pos = max(0.0, pos + x - drift)
        neg = min(0.0, neg + x + drift)

        if pos > threshold or neg < -threshold:
            flags[i] = True
            # reset after break
            pos = 0.0
            neg = 0.0

    return pd.Series(flags, index=series.index)


def compute_regimes_for_symbol(
    session: Session,
    symbol: str,
    vol_window: int = 20,
    trend_window: int = 20,
    k_regimes: int = 3,
    cusum_threshold: float = 3.0,
) -> List[RegimePoint]:
    """
    Main entrypoint: compute regimes + structural breaks for a symbol.

    Steps:
    1. Load price series from DB.
    2. Compute regime features (volatility, trend).
    3. Cluster volatility into k_regimes using simple k-means.
    4. Run CUSUM on trend to flag structural breaks.

    Args:
        session: Database session
        symbol: Asset symbol
        vol_window: Window for volatility calculation (default: 20)
        trend_window: Window for trend calculation (default: 20)
        k_regimes: Number of regime clusters (default: 3 for low/normal/high)
        cusum_threshold: CUSUM threshold for structural breaks (default: 3.0)

    Returns:
        List of RegimePoint objects

    Example:
        >>> from otonom_trader.data import get_session
        >>> with next(get_session()) as session:
        ...     regimes = compute_regimes_for_symbol(session, "BTC-USD")
        ...     print(f"Detected {len(regimes)} regime points")
    """
    df = _load_price_series(session, symbol)
    feats = _build_regime_features(
        df,
        price_col="close",
        vol_window=vol_window,
        trend_window=trend_window,
    )

    vol = feats["volatility"].values
    regimes = _kmeans_1d(vol, k=k_regimes)

    # CUSUM on trend (differences of trend)
    trend_diff = feats["trend"].diff().fillna(0.0)
    breaks = _detect_cusum(trend_diff, threshold=cusum_threshold)

    out: List[RegimePoint] = []
    for (d, row), reg, brk in zip(feats.iterrows(), regimes, breaks.values):
        out.append(
            RegimePoint(
                symbol=symbol,
                trade_date=d,
                volatility=float(row["volatility"]),
                trend=float(row["trend"]),
                regime_id=int(reg),
                is_structural_break=bool(brk),
            )
        )

    logger.info(f"Computed {len(out)} regime points for {symbol}")
    return out


def regimes_to_dataframe(regimes: List[RegimePoint]) -> pd.DataFrame:
    """
    Helper to convert regime list into a DataFrame for analysis/backtest.

    Args:
        regimes: List of RegimePoint objects

    Returns:
        DataFrame indexed by trade_date with regime information
    """
    if not regimes:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "regime_id",
                "volatility",
                "trend",
                "is_structural_break",
            ]
        )

    data = [
        {
            "symbol": r.symbol,
            "trade_date": r.trade_date,
            "regime_id": r.regime_id,
            "volatility": r.volatility,
            "trend": r.trend,
            "is_structural_break": r.is_structural_break,
        }
        for r in regimes
    ]
    df = pd.DataFrame(data)
    df.set_index("trade_date", inplace=True)
    return df


def compute_regimes_all_symbols(
    session: Session,
    symbols: List[str] = None,
    **kwargs
) -> dict[str, List[RegimePoint]]:
    """
    Compute regimes for multiple symbols.

    Args:
        session: Database session
        symbols: List of symbols. If None, uses all symbols in DB.
        **kwargs: Additional parameters passed to compute_regimes_for_symbol

    Returns:
        Dictionary mapping symbol to list of RegimePoint objects
    """
    if symbols is None:
        # Get all symbols from database
        all_symbols = session.query(Symbol.symbol).all()
        symbols = [s[0] for s in all_symbols]

    results = {}
    for symbol in symbols:
        try:
            regimes = compute_regimes_for_symbol(session, symbol, **kwargs)
            results[symbol] = regimes
        except Exception as e:
            logger.error(f"Failed to compute regimes for {symbol}: {e}")
            results[symbol] = []

    return results


def persist_regimes(
    session: Session,
    regimes: List[RegimePoint],
    upsert: bool = True,
) -> int:
    """
    Persist regime points to database.

    Args:
        session: Database session
        regimes: List of RegimePoint objects
        upsert: If True, update existing records; if False, skip duplicates

    Returns:
        Number of records inserted/updated
    """
    from ..data.schema import Regime as RegimeORM

    if not regimes:
        return 0

    # Get symbol_id
    symbol = regimes[0].symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
    if symbol_obj is None:
        raise ValueError(f"Symbol {symbol} not found in database")

    count = 0
    for r in regimes:
        # Check if exists
        existing = (
            session.query(RegimeORM)
            .filter_by(symbol_id=symbol_obj.id, date=r.trade_date)
            .first()
        )

        if existing and upsert:
            # Update
            existing.regime_id = r.regime_id
            existing.volatility = r.volatility
            existing.trend = r.trend
            existing.is_structural_break = int(r.is_structural_break)
            count += 1
        elif not existing:
            # Insert
            regime_orm = RegimeORM(
                symbol_id=symbol_obj.id,
                date=r.trade_date,
                regime_id=r.regime_id,
                volatility=r.volatility,
                trend=r.trend,
                is_structural_break=int(r.is_structural_break),
            )
            session.add(regime_orm)
            count += 1

    session.commit()
    logger.info(f"Persisted {count} regime records for {symbol}")
    return count
