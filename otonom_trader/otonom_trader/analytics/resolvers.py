"""
Resolver utilities for Analist-3 (P2).

These functions provide quick lookups for regime and DSI data
by (date, symbol), with caching for performance.

P2 Features:
- Regime lookup by (date, symbol)
- DSI lookup by (date, symbol)
- In-memory cache for hot data
- Fallback to DB query
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Dict, Tuple

from sqlalchemy.orm import Session

from ..data.schema import Symbol, Regime, DataHealthIndex

logger = logging.getLogger(__name__)


# Simple in-memory cache
# Key: (symbol, date_str)
# Value: (regime_id, volatility, trend, is_structural_break)
_regime_cache: Dict[Tuple[str, str], Tuple[int, float, float, bool]] = {}

# Key: (symbol, date_str)
# Value: (dsi, missing_ratio, outlier_ratio, volume_jump_ratio)
_dsi_cache: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}


def clear_cache():
    """Clear all resolver caches."""
    global _regime_cache, _dsi_cache
    _regime_cache.clear()
    _dsi_cache.clear()
    logger.info("Resolver caches cleared")


def resolve_regime(
    session: Session,
    symbol: str,
    query_date: date,
    use_cache: bool = True,
) -> Optional[Tuple[int, float, float, bool]]:
    """
    Resolve regime data for (symbol, date).

    Args:
        session: Database session
        symbol: Asset symbol
        query_date: Query date
        use_cache: Use in-memory cache (default: True)

    Returns:
        Tuple of (regime_id, volatility, trend, is_structural_break), or None if not found

    Example:
        >>> from otonom_trader.data import get_session
        >>> with next(get_session()) as session:
        ...     regime = resolve_regime(session, "BTC-USD", date(2025, 1, 15))
        ...     if regime:
        ...         regime_id, vol, trend, is_break = regime
        ...         print(f"Regime: {regime_id}, Vol: {vol:.4f}")
    """
    cache_key = (symbol, query_date.isoformat())

    # Check cache
    if use_cache and cache_key in _regime_cache:
        logger.debug(f"Regime cache hit: {cache_key}")
        return _regime_cache[cache_key]

    # Query DB
    try:
        symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            logger.warning(f"Symbol not found: {symbol}")
            return None

        regime = (
            session.query(Regime)
            .filter_by(symbol_id=symbol_obj.id, date=query_date)
            .first()
        )

        if not regime:
            logger.debug(f"Regime not found for {symbol} on {query_date}")
            return None

        result = (
            regime.regime_id,
            regime.volatility,
            regime.trend,
            bool(regime.is_structural_break),
        )

        # Update cache
        if use_cache:
            _regime_cache[cache_key] = result

        return result

    except Exception as e:
        logger.error(f"Error resolving regime for {symbol} on {query_date}: {e}")
        return None


def resolve_dsi(
    session: Session,
    symbol: str,
    query_date: date,
    use_cache: bool = True,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Resolve DSI data for (symbol, date).

    Args:
        session: Database session
        symbol: Asset symbol
        query_date: Query date
        use_cache: Use in-memory cache (default: True)

    Returns:
        Tuple of (dsi, missing_ratio, outlier_ratio, volume_jump_ratio), or None if not found

    Example:
        >>> from otonom_trader.data import get_session
        >>> with next(get_session()) as session:
        ...     dsi = resolve_dsi(session, "BTC-USD", date(2025, 1, 15))
        ...     if dsi:
        ...         dsi_score, missing, outlier, volume_jump = dsi
        ...         print(f"DSI: {dsi_score:.2f}")
    """
    cache_key = (symbol, query_date.isoformat())

    # Check cache
    if use_cache and cache_key in _dsi_cache:
        logger.debug(f"DSI cache hit: {cache_key}")
        return _dsi_cache[cache_key]

    # Query DB
    try:
        symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            logger.warning(f"Symbol not found: {symbol}")
            return None

        dsi_record = (
            session.query(DataHealthIndex)
            .filter_by(symbol_id=symbol_obj.id, date=query_date)
            .first()
        )

        if not dsi_record:
            logger.debug(f"DSI not found for {symbol} on {query_date}")
            return None

        result = (
            dsi_record.dsi,
            dsi_record.missing_ratio,
            dsi_record.outlier_ratio,
            dsi_record.volume_jump_ratio,
        )

        # Update cache
        if use_cache:
            _dsi_cache[cache_key] = result

        return result

    except Exception as e:
        logger.error(f"Error resolving DSI for {symbol} on {query_date}: {e}")
        return None


def preload_regime_cache(
    session: Session,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> int:
    """
    Preload regime cache for a date range.

    Useful for batch processing to avoid repeated DB queries.

    Args:
        session: Database session
        symbols: List of symbols to preload
        start_date: Start date
        end_date: End date

    Returns:
        Number of records preloaded

    Example:
        >>> from datetime import date, timedelta
        >>> with next(get_session()) as session:
        ...     count = preload_regime_cache(
        ...         session,
        ...         ["BTC-USD", "ETH-USD"],
        ...         date.today() - timedelta(days=30),
        ...         date.today(),
        ...     )
        ...     print(f"Preloaded {count} regime records")
    """
    count = 0

    for symbol in symbols:
        symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            continue

        regimes = (
            session.query(Regime)
            .filter(
                Regime.symbol_id == symbol_obj.id,
                Regime.date >= start_date,
                Regime.date <= end_date,
            )
            .all()
        )

        for regime in regimes:
            cache_key = (symbol, regime.date.isoformat())
            _regime_cache[cache_key] = (
                regime.regime_id,
                regime.volatility,
                regime.trend,
                bool(regime.is_structural_break),
            )
            count += 1

    logger.info(f"Preloaded {count} regime records into cache")
    return count


def preload_dsi_cache(
    session: Session,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> int:
    """
    Preload DSI cache for a date range.

    Args:
        session: Database session
        symbols: List of symbols to preload
        start_date: Start date
        end_date: End date

    Returns:
        Number of records preloaded
    """
    count = 0

    for symbol in symbols:
        symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            continue

        dsi_records = (
            session.query(DataHealthIndex)
            .filter(
                DataHealthIndex.symbol_id == symbol_obj.id,
                DataHealthIndex.date >= start_date,
                DataHealthIndex.date <= end_date,
            )
            .all()
        )

        for dsi_record in dsi_records:
            cache_key = (symbol, dsi_record.date.isoformat())
            _dsi_cache[cache_key] = (
                dsi_record.dsi,
                dsi_record.missing_ratio,
                dsi_record.outlier_ratio,
                dsi_record.volume_jump_ratio,
            )
            count += 1

    logger.info(f"Preloaded {count} DSI records into cache")
    return count
