"""
Anomaly detection - Identify spikes and crashes in price movements.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List
from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from ..domain import (
    Asset,
    AssetClass,
    Anomaly as AnomalyDomain,
    AnomalyType,
)
from ..data.schema import IntradayBar, Symbol, Anomaly as AnomalyORM
from .returns import compute_returns, compute_rolling_stats, compute_volume_quantile
from .labeling import classify_anomaly, is_anomaly
from ..utils import utc_now

logger = logging.getLogger(__name__)


def detect_anomalies_for_asset(
    session: Session,
    asset: Asset,
    k: float = 2.5,
    q: float = 0.8,
    window: int = 60,
    persist: bool = True,
    interval: str = "15m",
    lookback_days: int = 60,
) -> List[AnomalyDomain]:
    """
    Detect anomalies for a specific asset using intraday (15m) bars.

    Args:
        session: Database session
        asset: Asset to analyze
        k: Z-score threshold for anomaly detection
        q: Volume quantile threshold (0-1)
        window: Rolling window size in bars
        persist: Whether to save anomalies to database
        interval: Intraday interval to analyze (default 15m)
        lookback_days: How many days of history to fetch

    Returns:
        List of Anomaly domain objects
    """
    logger.info(f"Detecting anomalies for {asset.symbol}")

    symbol_obj = session.query(Symbol).filter_by(symbol=asset.symbol).first()
    if symbol_obj is None:
        logger.warning(f"Symbol {asset.symbol} not found in database")
        return []

    cutoff = utc_now() - timedelta(days=max(lookback_days, 1))
    bars = (
        session.query(IntradayBar)
        .filter_by(symbol_id=symbol_obj.id, interval=interval)
        .filter(IntradayBar.ts >= cutoff)
        .order_by(IntradayBar.ts.asc())
        .all()
    )

    if len(bars) < window + 10:
        logger.warning(
            f"Insufficient data for {asset.symbol}: {len(bars)} bars "
            f"(need at least {window + 10})"
        )
        return []

    df = pd.DataFrame(
        [
            {
                "timestamp": bar.ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )

    df = df.sort_values("timestamp")
    df = compute_returns(df)
    df = compute_rolling_stats(df, window=window)
    df = compute_volume_quantile(df, window=window)

    anomalies: List[AnomalyDomain] = []
    existing_cache: Dict[date, AnomalyORM] = {}

    for _, row in df.iterrows():
        if pd.isna(row["ret_zscore"]) or pd.isna(row["volume_quantile"]):
            continue

        anomaly_type = classify_anomaly(
            zscore=row["ret_zscore"],
            volume_quantile=row["volume_quantile"],
            k_threshold=k,
            q_threshold=q,
        )

        if is_anomaly(anomaly_type):
            ts_value = row.get("timestamp")
            if isinstance(ts_value, pd.Timestamp):
                ts_dt = ts_value.to_pydatetime()
            else:
                ts_dt = ts_value if isinstance(ts_value, datetime) else None
            if ts_dt is None:
                continue
            anomaly_date = ts_dt.date()

            anomaly = AnomalyDomain(
                asset_symbol=asset.symbol,
                date=anomaly_date,
                anomaly_type=anomaly_type,
                abs_return=abs(row["ret"]),
                zscore=row["ret_zscore"],
                volume_rank=row["volume_quantile"],
            )
            anomalies.append(anomaly)

            if persist:
                cache_key = anomaly_date
                existing = existing_cache.get(cache_key)
                if existing is None:
                    existing = (
                        session.query(AnomalyORM)
                        .filter_by(symbol_id=symbol_obj.id, date=anomaly_date)
                        .first()
                    )
                    if existing:
                        existing_cache[cache_key] = existing
                existing = (
                    session.query(AnomalyORM)
                    .filter_by(symbol_id=symbol_obj.id, date=anomaly_date)
                    .first()
                )

                if existing:
                    existing.anomaly_type = str(anomaly_type)
                    existing.abs_return = float(abs(row["ret"]))
                    existing.zscore = float(row["ret_zscore"])
                    existing.volume_rank = float(row["volume_quantile"])
                else:
                    anomaly_orm = AnomalyORM(
                        symbol_id=symbol_obj.id,
                        date=anomaly_date,
                        anomaly_type=str(anomaly_type),
                        abs_return=float(abs(row["ret"])),
                        zscore=float(row["ret_zscore"]),
                        volume_rank=float(row["volume_quantile"]),
                    )
                    session.add(anomaly_orm)
                    existing_cache[cache_key] = anomaly_orm

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
    interval: str = "15m",
    lookback_days: int = 60,
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
                interval=interval,
                lookback_days=lookback_days,
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
    interval: str = "15m",
) -> List[AnomalyDomain]:
    """Convenience helper to detect anomalies for a list of symbol strings.

    Args:
        session: Active SQLAlchemy session.
        symbols: Ordered list of ticker symbols to scan.
        lookback_days: Rolling window (in days) passed to the per-asset detector.
        k: Z-score threshold.
        q: Volume quantile threshold.
        interval: Intraday interval to analyze (default 15m).

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
                interval=interval,
                lookback_days=lookback_days,
            )
        )

    return anomalies
