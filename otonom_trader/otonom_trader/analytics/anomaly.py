"""
Anomaly detection - Identify spikes and crashes in price movements.
"""
import logging
from datetime import date
from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from ..domain import (
    Asset,
    AssetClass,
    Anomaly as AnomalyDomain,
    AnomalyType,
)
from ..data.schema import DailyBar, Symbol, Anomaly as AnomalyORM
from .returns import compute_returns, compute_rolling_stats, compute_volume_quantile
from .labeling import classify_anomaly, is_anomaly

logger = logging.getLogger(__name__)


def detect_anomalies_for_asset(
    session: Session,
    asset: Asset,
    k: float = 2.5,
    q: float = 0.8,
    window: int = 60,
    persist: bool = True,
) -> List[AnomalyDomain]:
    """
    Detect anomalies for a specific asset.

    Args:
        session: Database session
        asset: Asset to analyze
        k: Z-score threshold for anomaly detection
        q: Volume quantile threshold (0-1)
        window: Rolling window size in days
        persist: Whether to save anomalies to database

    Returns:
        List of Anomaly domain objects

    Process:
        1. Fetch all daily bars for the asset
        2. Calculate returns and rolling statistics
        3. Identify spikes/crashes based on thresholds
        4. Optionally persist to database
    """
    logger.info(f"Detecting anomalies for {asset.symbol}")

    # Get symbol from database
    symbol_obj = session.query(Symbol).filter_by(symbol=asset.symbol).first()
    if symbol_obj is None:
        logger.warning(f"Symbol {asset.symbol} not found in database")
        return []

    # Fetch all daily bars
    bars = (
        session.query(DailyBar)
        .filter_by(symbol_id=symbol_obj.id)
        .order_by(DailyBar.date)
        .all()
    )

    if len(bars) < window + 10:
        logger.warning(
            f"Insufficient data for {asset.symbol}: {len(bars)} bars "
            f"(need at least {window + 10})"
        )
        return []

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )

    # Calculate returns and statistics
    df = compute_returns(df)
    df = compute_rolling_stats(df, window=window)
    df = compute_volume_quantile(df, window=window)

    # Identify anomalies
    anomalies = []

    for _, row in df.iterrows():
        # Skip rows with insufficient data
        if pd.isna(row["ret_zscore"]) or pd.isna(row["volume_quantile"]):
            continue

        # Classify anomaly
        anomaly_type = classify_anomaly(
            zscore=row["ret_zscore"],
            volume_quantile=row["volume_quantile"],
            k_threshold=k,
            q_threshold=q,
        )

        # Only record actual anomalies
        if is_anomaly(anomaly_type):
            anomaly = AnomalyDomain(
                asset_symbol=asset.symbol,
                date=row["date"],
                anomaly_type=anomaly_type,
                abs_return=abs(row["ret"]),
                zscore=row["ret_zscore"],
                volume_rank=row["volume_quantile"],
            )
            anomalies.append(anomaly)

            # Persist to database if requested
            if persist:
                # Check if anomaly already exists
                existing = (
                    session.query(AnomalyORM)
                    .filter_by(symbol_id=symbol_obj.id, date=row["date"])
                    .first()
                )

                if existing:
                    # Update existing
                    existing.anomaly_type = str(anomaly_type)
                    existing.abs_return = float(abs(row["ret"]))
                    existing.zscore = float(row["ret_zscore"])
                    existing.volume_rank = float(row["volume_quantile"])
                else:
                    # Create new
                    anomaly_orm = AnomalyORM(
                        symbol_id=symbol_obj.id,
                        date=row["date"],
                        anomaly_type=str(anomaly_type),
                        abs_return=float(abs(row["ret"])),
                        zscore=float(row["ret_zscore"]),
                        volume_rank=float(row["volume_quantile"]),
                    )
                    session.add(anomaly_orm)

    if persist:
        session.commit()

    logger.info(f"Detected {len(anomalies)} anomalies for {asset.symbol}")
    return anomalies


def detect_anomalies_all_assets(
    session: Session,
    assets: List[Asset],
    k: float = 2.5,
    q: float = 0.8,
    window: int = 60,
) -> dict[str, List[AnomalyDomain]]:
    """
    Detect anomalies for all assets.

    Args:
        session: Database session
        assets: List of assets to analyze
        k: Z-score threshold
        q: Volume quantile threshold
        window: Rolling window size

    Returns:
        Dictionary mapping symbol to list of anomalies
    """
    results = {}

    for asset in assets:
        try:
            anomalies = detect_anomalies_for_asset(
                session=session,
                asset=asset,
                k=k,
                q=q,
                window=window,
                persist=True,
            )
            results[asset.symbol] = anomalies
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {asset.symbol}: {e}")
            results[asset.symbol] = []

    return results


def detect_anomalies_for_universe(
    session: Session,
    symbols: List[str],
    lookback_days: int = 60,
    k: float = 2.5,
    q: float = 0.8,
) -> List[AnomalyDomain]:
    """Convenience helper to detect anomalies for a list of symbol strings.

    Args:
        session: Active SQLAlchemy session.
        symbols: Ordered list of ticker symbols to scan.
        lookback_days: Rolling window (in days) passed to the per-asset detector.
        k: Z-score threshold.
        q: Volume quantile threshold.

    Returns:
        List of anomaly domain objects aggregated across all requested symbols.
    """

    if not symbols:
        return []

    # Fetch symbol metadata once and keep caller order when iterating.
    symbol_rows = (
        session.query(Symbol)
        .filter(Symbol.symbol.in_(symbols))
        .all()
    )
    row_by_symbol = {row.symbol: row for row in symbol_rows}

    def _to_asset(row: Symbol) -> Asset:
        asset_class_value = (row.asset_class or "OTHER").upper()
        try:
            asset_class = AssetClass(asset_class_value)
        except ValueError:
            asset_class = AssetClass.OTHER
        return Asset(
            symbol=row.symbol,
            name=row.name,
            asset_class=asset_class,
        )

    anomalies: List[AnomalyDomain] = []
    for symbol in symbols:
        row = row_by_symbol.get(symbol)
        if row is None:
            logger.warning("Symbol %s not found in database for anomaly scan", symbol)
            continue

        asset = _to_asset(row)
        anomalies.extend(
            detect_anomalies_for_asset(
                session=session,
                asset=asset,
                k=k,
                q=q,
                window=lookback_days,
                persist=True,
            )
        )

    return anomalies
