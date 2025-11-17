"""
Data Health Index (DSI) for assets.

This module computes a simple DSI score per (symbol, date) window
based on missing data and extreme outliers. It is deliberately
simple and can be refined later (source trust, on-chain, etc.).

DSI Components:
- missing_ratio: Fraction of days with missing OHLCV data
- outlier_ratio: Fraction of days with extreme return z-scores (|z| > threshold)
- volume_jump_ratio: Fraction of days with extreme volume z-scores

DSI Score: Combined metric in [0, 1] where 1 = perfect data quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..data.schema import DailyBar, Symbol
from .returns import compute_returns

logger = logging.getLogger(__name__)


@dataclass
class DsiPoint:
    """
    Data health index point for a specific date.

    Attributes:
        symbol: Asset symbol
        trade_date: Date of the DSI measurement
        dsi: Overall data health score (0-1, higher is better)
        missing_ratio: Fraction of missing data points in window
        outlier_ratio: Fraction of extreme return outliers
        volume_jump_ratio: Fraction of extreme volume jumps
    """
    symbol: str
    trade_date: date
    dsi: float
    missing_ratio: float
    outlier_ratio: float
    volume_jump_ratio: float


def _load_bars(
    session: Session,
    symbol: str,
) -> pd.DataFrame:
    """
    Load OHLCV bars for a symbol.

    Args:
        session: Database session
        symbol: Asset symbol

    Returns:
        DataFrame with OHLCV data indexed by date
    """
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
        raise ValueError(f"No data for symbol={symbol}")

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    ).set_index("date")
    return df


def compute_dsi_for_symbol(
    session: Session,
    symbol: str,
    window: int = 60,
    outlier_z: float = 4.0,
) -> List[DsiPoint]:
    """
    Compute rolling DSI for a symbol.

    For each window of length `window`, we compute:
    - missing_ratio: fraction of days with missing OHLCV
    - outlier_ratio: fraction of days with |return_z| > outlier_z
    - volume_jump_ratio: fraction of days with |volume_z| > outlier_z

    Then combine into DSI in [0, 1], where 1 = perfect data.

    Args:
        session: Database session
        symbol: Asset symbol
        window: Rolling window size (default: 60)
        outlier_z: Z-score threshold for outliers (default: 4.0)

    Returns:
        List of DsiPoint objects

    Example:
        >>> from otonom_trader.data import get_session
        >>> with next(get_session()) as session:
        ...     dsi_points = compute_dsi_for_symbol(session, "BTC-USD")
        ...     avg_dsi = sum(p.dsi for p in dsi_points) / len(dsi_points)
        ...     print(f"Average DSI: {avg_dsi:.2f}")
    """
    logger.info(f"Computing DSI for {symbol} (window={window}, outlier_z={outlier_z})")

    df = _load_bars(session, symbol)

    # Compute return stats
    df_reset = df.reset_index()
    df_ret = compute_returns(df_reset, price_col="close")

    # Calculate rolling mean and std for returns
    ret = df_ret["ret"]
    ret_mean = ret.rolling(window=window, min_periods=window).mean()
    ret_std = ret.rolling(window=window, min_periods=window).std()
    z_ret = (ret - ret_mean) / ret_std
    z_ret = z_ret.replace([np.inf, -np.inf], np.nan)

    # Volume z-score
    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(window=window, min_periods=window).mean()
    vol_std = vol.rolling(window=window, min_periods=window).std()
    z_vol = (vol - vol_mean) / vol_std
    z_vol = z_vol.replace([np.inf, -np.inf], np.nan)

    points: List[DsiPoint] = []

    for i in range(len(df)):
        if i + 1 < window:
            continue  # need full window

        idx_window = df.index[i + 1 - window : i + 1]

        # Missing ratio: any NaN in close or volume
        sub = df.loc[idx_window]
        missing_mask = sub["close"].isna() | sub["volume"].isna()
        missing_ratio = float(missing_mask.sum()) / float(window)

        # Outlier ratio on returns
        z_sub = z_ret.iloc[i + 1 - window : i + 1]
        outlier_mask = z_sub.abs() > outlier_z
        outlier_ratio = float(outlier_mask.sum()) / float(window)

        # Volume jump ratio
        zv_sub = z_vol.iloc[i + 1 - window : i + 1]
        vol_jump_mask = zv_sub.abs() > outlier_z
        volume_jump_ratio = float(vol_jump_mask.sum()) / float(window)

        # Combine into simple DSI
        # Penalty = weighted sum of ratios, then DSI = max(0, 1 - penalty)
        penalty = (
            1.5 * missing_ratio +  # missing is worst
            1.0 * outlier_ratio +
            0.5 * volume_jump_ratio
        )
        dsi = float(max(0.0, min(1.0, 1.0 - penalty)))

        d = df.index[i]
        if isinstance(d, pd.Timestamp):
            d = d.date()

        points.append(
            DsiPoint(
                symbol=symbol,
                trade_date=d,
                dsi=dsi,
                missing_ratio=missing_ratio,
                outlier_ratio=outlier_ratio,
                volume_jump_ratio=volume_jump_ratio,
            )
        )

    logger.info(f"Computed {len(points)} DSI points for {symbol}")
    return points


def dsi_to_dataframe(points: List[DsiPoint]) -> pd.DataFrame:
    """
    Convert DSI points to DataFrame.

    Args:
        points: List of DsiPoint objects

    Returns:
        DataFrame indexed by trade_date with DSI metrics
    """
    if not points:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "dsi",
                "missing_ratio",
                "outlier_ratio",
                "volume_jump_ratio",
            ]
        )

    data = [
        {
            "symbol": p.symbol,
            "trade_date": p.trade_date,
            "dsi": p.dsi,
            "missing_ratio": p.missing_ratio,
            "outlier_ratio": p.outlier_ratio,
            "volume_jump_ratio": p.volume_jump_ratio,
        }
        for p in points
    ]
    df = pd.DataFrame(data)
    df.set_index("trade_date", inplace=True)
    return df


def compute_dsi_all_symbols(
    session: Session,
    symbols: List[str] = None,
    **kwargs
) -> dict[str, List[DsiPoint]]:
    """
    Compute DSI for multiple symbols.

    Args:
        session: Database session
        symbols: List of symbols. If None, uses all symbols in DB.
        **kwargs: Additional parameters passed to compute_dsi_for_symbol

    Returns:
        Dictionary mapping symbol to list of DsiPoint objects
    """
    if symbols is None:
        # Get all symbols from database
        all_symbols = session.query(Symbol.symbol).all()
        symbols = [s[0] for s in all_symbols]

    results = {}
    for symbol in symbols:
        try:
            dsi_points = compute_dsi_for_symbol(session, symbol, **kwargs)
            results[symbol] = dsi_points
        except Exception as e:
            logger.error(f"Failed to compute DSI for {symbol}: {e}")
            results[symbol] = []

    return results


def persist_dsi(
    session: Session,
    dsi_points: List[DsiPoint],
    upsert: bool = True,
) -> int:
    """
    Persist DSI points to database.

    Args:
        session: Database session
        dsi_points: List of DsiPoint objects
        upsert: If True, update existing records; if False, skip duplicates

    Returns:
        Number of records inserted/updated
    """
    from ..data.schema import DataHealthIndex as DsiORM

    if not dsi_points:
        return 0

    # Get symbol_id
    symbol = dsi_points[0].symbol
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
    if symbol_obj is None:
        raise ValueError(f"Symbol {symbol} not found in database")

    count = 0
    for p in dsi_points:
        # Check if exists
        existing = (
            session.query(DsiORM)
            .filter_by(symbol_id=symbol_obj.id, date=p.trade_date)
            .first()
        )

        if existing and upsert:
            # Update
            existing.dsi = p.dsi
            existing.missing_ratio = p.missing_ratio
            existing.outlier_ratio = p.outlier_ratio
            existing.volume_jump_ratio = p.volume_jump_ratio
            count += 1
        elif not existing:
            # Insert
            dsi_orm = DsiORM(
                symbol_id=symbol_obj.id,
                date=p.trade_date,
                dsi=p.dsi,
                missing_ratio=p.missing_ratio,
                outlier_ratio=p.outlier_ratio,
                volume_jump_ratio=p.volume_jump_ratio,
            )
            session.add(dsi_orm)
            count += 1

    session.commit()
    logger.info(f"Persisted {count} DSI records for {symbol}")
    return count
