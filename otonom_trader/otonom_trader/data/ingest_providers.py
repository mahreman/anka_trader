"""
Enhanced data ingestion using real providers.

Integrates price, news, and macro data from multiple sources into database.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List, Dict

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..providers import (
    get_primary_price_provider,
    get_primary_macro_provider,
    ProviderError,
)
from ..providers.base import (
    OHLCVBar,
    NewsArticle as ProviderNewsArticle,
    MacroIndicator as ProviderMacroIndicator,
    PriceProvider,
)
from ..providers.config import ProviderConfig, get_provider_config
from ..providers.factory import create_price_provider, create_news_provider
from .schema import Symbol, DailyBar, IntradayBar, NewsArticle, MacroIndicator
from .symbols import ensure_p0_symbols
from ..utils import ensure_aware

logger = logging.getLogger(__name__)


def ingest_price_data(
    session: Session,
    symbol: str,
    start_date: date,
    end_date: date,
    provider_config_path: str = "config/providers.yaml",
) -> int:
    """
    Ingest price data using configured provider.

    Args:
        session: Database session
        symbol: Symbol to ingest
        start_date: Start date
        end_date: End date
        provider_config_path: Path to provider config

    Returns:
        Number of bars ingested

    Example:
        >>> with get_session() as session:
        ...     count = ingest_price_data(session, "BTC-USD", date(2024, 1, 1), date(2024, 1, 31))
        ...     print(f"Ingested {count} bars")
    """
    logger.info(f"Ingesting price data for {symbol}: {start_date} to {end_date}")

    # Get primary price provider
    provider = get_primary_price_provider(provider_config_path)

    if provider is None:
        logger.error("No enabled price provider found")
        return 0

    try:
        # Fetch OHLCV bars
        bars = provider.fetch_ohlcv(symbol, start_date, end_date)

        if not bars:
            logger.warning(f"No bars returned for {symbol}")
            return 0

        # Get or create symbol
        symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
        if symbol_obj is None:
            symbol_obj = Symbol(
                symbol=symbol,
                name=symbol,  # TODO: Get proper name from provider
                asset_class="CRYPTO",  # TODO: Detect asset class
            )
            session.add(symbol_obj)
            session.flush()

        # Upsert bars
        count = 0
        for bar in bars:
            bar_date = bar.date.date() if isinstance(bar.date, datetime) else bar.date
            # Check if bar exists
            existing = (
                session.query(DailyBar)
                .filter_by(symbol_id=symbol_obj.id, date=bar_date)
                .first()
            )

            if existing:
                # Update existing
                existing.open = bar.open
                existing.high = bar.high
                existing.low = bar.low
                existing.close = bar.close
                existing.volume = bar.volume
                existing.adj_close = bar.adj_close
            else:
                # Insert new
                new_bar = DailyBar(
                    symbol_id=symbol_obj.id,
                    date=bar_date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    adj_close=bar.adj_close,
                )
                session.add(new_bar)

            count += 1

        session.commit()
        logger.info(f"Ingested {count} bars for {symbol}")

        return count

    except ProviderError as e:
        logger.error(f"Failed to ingest price data for {symbol}: {e}")
        session.rollback()
        return 0


def _interval_to_timedelta(interval: str) -> timedelta:
    interval = (interval or "15m").lower()

    try:
        qty = int(interval[:-1])
        unit = interval[-1]
    except ValueError:
        return timedelta(minutes=15)

    if unit == "m":
        return timedelta(minutes=qty)
    if unit == "h":
        return timedelta(hours=qty)
    if unit == "d":
        return timedelta(days=qty)
    if unit == "w":
        return timedelta(weeks=qty)
    return timedelta(minutes=15)


def _normalize_timestamp(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.combine(value, datetime.min.time())

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_asset_class(value: Optional[str]) -> str:
    if not value:
        return ""
    upper = value.upper()
    if upper.startswith("ASSETCLASS."):
        return upper.split(".", 1)[1]
    return upper


def _get_provider_for_type(
    provider_type: str,
    provider_cache: Dict[str, Optional[PriceProvider]],
    provider_config: ProviderConfig,
):
    key = (provider_type or "").lower()
    if key in provider_cache:
        return provider_cache[key]

    for p_cfg in provider_config.get_enabled_price_providers():
        if p_cfg.provider_type.lower() == key:
            provider = create_price_provider(p_cfg)
            provider_cache[key] = provider
            return provider

    logger.warning("Price provider '%s' requested but not enabled", provider_type)
    provider_cache[key] = None
    return None


def _resolve_intraday_provider(
    symbol_obj: Symbol,
    provider_cache: Dict[str, Optional[PriceProvider]],
    provider_config: ProviderConfig,
) -> Optional[PriceProvider]:
    asset_class = _normalize_asset_class(symbol_obj.asset_class)
    provider_type = "binance" if asset_class == "CRYPTO" else "yfinance"
    provider = _get_provider_for_type(provider_type, provider_cache, provider_config)
    if provider is None and provider_type != "yfinance":
        # Fallback to yfinance for non-crypto assets when Binance is unavailable
        provider = _get_provider_for_type("yfinance", provider_cache, provider_config)
    return provider


def ingest_intraday_bars_for_symbol(
    session: Session,
    symbol_obj: Symbol,
    interval: str = "15m",
    lookback_days: int = 7,
    provider: Optional[PriceProvider] = None,
    provider_config_path: str = "config/providers.yaml",
) -> int:
    """Ingest intraday bars for a single symbol."""

    local_provider = provider or get_primary_price_provider(provider_config_path)
    if local_provider is None:
        logger.warning("No price provider available for intraday ingest")
        return 0

    now_utc = datetime.now(timezone.utc)
    delta = _interval_to_timedelta(interval)

    last_bar = (
        session.query(IntradayBar)
        .filter_by(symbol_id=symbol_obj.id, interval=interval)
        .order_by(IntradayBar.ts.desc())
        .first()
    )

    if last_bar:
        last_ts = last_bar.ts
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        start_ts = last_ts + delta
    else:
        start_ts = now_utc - timedelta(days=lookback_days)

    if start_ts >= now_utc:
        return 0

    logger.info(
        "Fetching %s bars for %s from %s to %s",
        interval,
        symbol_obj.symbol,
        start_ts,
        now_utc,
    )

    try:
        bars = local_provider.fetch_ohlcv(
            symbol_obj.symbol,
            start_ts,
            now_utc,
            interval=interval,
        )
    except ProviderError as exc:
        logger.error("Failed to fetch intraday data for %s: %s", symbol_obj.symbol, exc)
        return 0

    if not bars:
        return 0

    count = 0
    for bar in bars:
        bar_ts = _normalize_timestamp(bar.date)
        if bar_ts < start_ts:
            continue

        ts_value = bar_ts.replace(tzinfo=None)

        existing = (
            session.query(IntradayBar)
            .filter_by(symbol_id=symbol_obj.id, ts=ts_value, interval=interval)
            .first()
        )

        if existing:
            existing.open = bar.open
            existing.high = bar.high
            existing.low = bar.low
            existing.close = bar.close
            existing.volume = bar.volume
            existing.adj_close = bar.adj_close
        else:
            session.add(
                IntradayBar(
                    symbol_id=symbol_obj.id,
                    ts=ts_value,
                    interval=interval,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    adj_close=bar.adj_close,
                )
            )

        count += 1

    return count


def ingest_intraday_bars_all(
    session: Session,
    interval: str = "15m",
    lookback_days: int = 7,
    provider_config_path: str = "config/providers.yaml",
) -> int:
    """Ingest intraday bars for all symbols in the database."""

    seeded = ensure_p0_symbols(session)
    if seeded:
        logger.info("Seeded %s default symbols for intraday ingest", seeded)

    symbols = session.query(Symbol).order_by(Symbol.symbol.asc()).all()
    if not symbols:
        logger.info("No symbols available for intraday ingest")
        return 0

    provider_cache: Dict[str, Optional[PriceProvider]] = {}
    provider_config = get_provider_config(provider_config_path)

    total = 0
    processed = 0
    for symbol_obj in symbols:
        provider = _resolve_intraday_provider(symbol_obj, provider_cache, provider_config)
        if provider is None:
            logger.warning(
                "Skipping %s; no provider configured for asset class '%s'",
                symbol_obj.symbol,
                symbol_obj.asset_class,
            )
            continue

        count = ingest_intraday_bars_for_symbol(
            session,
            symbol_obj,
            interval=interval,
            lookback_days=lookback_days,
            provider=provider,
            provider_config_path=provider_config_path,
        )
        processed += 1
        if count:
            session.commit()
        total += count

    logger.info("Ingested %s intraday bars across %s symbols", total, processed)
    return total


def ingest_news_data(
    session: Session,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    provider_config_path: str = "config/providers.yaml",
) -> int:
    """
    Ingest news data using configured provider.

    Args:
        session: Database session
        symbol: Filter by symbol (optional)
        start_date: Start datetime (optional)
        end_date: End datetime (optional)
        limit: Max articles
        provider_config_path: Path to provider config

    Returns:
        Number of articles ingested

    Example:
        >>> with get_session() as session:
        ...     count = ingest_news_data(session, symbol="BTC-USD", limit=50)
        ...     print(f"Ingested {count} news articles")
    """
    logger.info(f"Ingesting news data (symbol={symbol}, limit={limit})")

    normalized_symbol = symbol.strip() if symbol and symbol.strip() else None
    symbol_obj: Optional[Symbol] = None
    asset_class = ""

    if normalized_symbol:
        symbol_obj = (
            session.query(Symbol)
            .filter(func.lower(Symbol.symbol) == normalized_symbol.lower())
            .first()
        )
        if symbol_obj:
            asset_class = _normalize_asset_class(symbol_obj.asset_class)
            if asset_class == "CRYPTO":
                logger.info(
                    "Skipping news ingestion for crypto symbol %s", normalized_symbol
                )
                return 0
        else:
            logger.info(
                "Symbol %s not found in DB; proceeding with news ingest without asset class gating",
                normalized_symbol,
            )

    provider_config = get_provider_config(provider_config_path)
    provider_cfg = None

    if normalized_symbol and asset_class != "CRYPTO":
        provider_cfg = next(
            (
                cfg
                for cfg in provider_config.get_enabled_news_providers()
                if cfg.provider_type.lower() in {"yfinance", "yfinance_news"}
            ),
            None,
        )
        if provider_cfg is None:
            logger.debug(
                "YFinance news provider not enabled; falling back to primary for %s",
                normalized_symbol,
            )

    if provider_cfg is None:
        provider_cfg = provider_config.get_primary_news_provider()

    if provider_cfg is None:
        logger.warning("No enabled news provider found")
        return 0

    provider = create_news_provider(provider_cfg)

    try:
        # Fetch news articles
        articles = provider.fetch_news(
            symbol=normalized_symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        if not articles:
            logger.warning("No news articles returned")
            return 0

        # Upsert articles
        count = 0
        normalized_upper = normalized_symbol.upper() if normalized_symbol else None

        for article in articles:
            # Check if article exists (by URL)
            existing = session.query(NewsArticle).filter_by(url=article.url).first()

            if existing:
                # Update existing (sentiment may change)
                existing.sentiment = article.sentiment
                existing.sentiment_source = "provider"
            else:
                # Insert new
                symbols_list = [s for s in (article.symbols or []) if s]
                if normalized_symbol:
                    if not any(
                        (sym or "").upper() == normalized_upper for sym in symbols_list
                    ):
                        symbols_list.append(normalized_symbol)
                symbols_str = ",".join(symbols_list) if symbols_list else None

                new_article = NewsArticle(
                    source=article.source,
                    title=article.title,
                    description=article.description,
                    url=article.url,
                    published_at=ensure_aware(article.published_at),
                    author=article.author,
                    sentiment=article.sentiment,
                    sentiment_source="provider" if article.sentiment else None,
                    symbols=symbols_str,
                )
                session.add(new_article)

            count += 1

        session.commit()
        logger.info(f"Ingested {count} news articles")

        return count

    except ProviderError as e:
        logger.error(f"Failed to ingest news data: {e}")
        session.rollback()
        return 0


def ingest_macro_data(
    session: Session,
    indicator_code: str,
    start_date: date,
    end_date: date,
    provider_config_path: str = "config/providers.yaml",
) -> int:
    """
    Ingest macroeconomic indicator data using configured provider.

    Args:
        session: Database session
        indicator_code: Indicator code (e.g., "GDP", "UNRATE", "DFF")
        start_date: Start date
        end_date: End date
        provider_config_path: Path to provider config

    Returns:
        Number of indicator observations ingested

    Example:
        >>> with get_session() as session:
        ...     count = ingest_macro_data(session, "UNRATE", date(2023, 1, 1), date(2024, 1, 1))
        ...     print(f"Ingested {count} unemployment rate observations")
    """
    logger.info(f"Ingesting macro data: {indicator_code} from {start_date} to {end_date}")

    # Get primary macro provider
    provider = get_primary_macro_provider(provider_config_path)

    if provider is None:
        logger.warning("No enabled macro provider found")
        return 0

    try:
        # Fetch indicator observations
        indicators = provider.fetch_indicator(indicator_code, start_date, end_date)

        if not indicators:
            logger.warning(f"No indicators returned for {indicator_code}")
            return 0

        # Upsert indicators
        count = 0
        for ind in indicators:
            # Check if observation exists
            existing = (
                session.query(MacroIndicator)
                .filter_by(indicator_code=ind.indicator_code, date=ind.date)
                .first()
            )

            if existing:
                # Update existing
                existing.value = ind.value
                existing.indicator_name = ind.name
                existing.unit = ind.unit
                existing.frequency = ind.frequency
                existing.provider = provider.name
            else:
                # Insert new
                new_ind = MacroIndicator(
                    indicator_code=ind.indicator_code,
                    indicator_name=ind.name,
                    date=ind.date,
                    value=ind.value,
                    unit=ind.unit,
                    frequency=ind.frequency,
                    provider=provider.name,
                )
                session.add(new_ind)

            count += 1

        session.commit()
        logger.info(f"Ingested {count} observations for {indicator_code}")

        return count

    except ProviderError as e:
        logger.error(f"Failed to ingest macro data for {indicator_code}: {e}")
        session.rollback()
        return 0


def ingest_all_data_types(
    session: Session,
    symbol: str,
    start_date: date,
    end_date: date,
    macro_indicators: Optional[List[str]] = None,
    provider_config_path: str = "config/providers.yaml",
) -> Dict[str, int]:
    """
    Ingest all data types (price, news, macro) for a symbol.

    Args:
        session: Database session
        symbol: Symbol to ingest
        start_date: Start date
        end_date: End date
        macro_indicators: List of macro indicator codes (optional)
        provider_config_path: Path to provider config

    Returns:
        Dict with counts for each data type

    Example:
        >>> with get_session() as session:
        ...     counts = ingest_all_data_types(
        ...         session, "BTC-USD",
        ...         date(2024, 1, 1), date(2024, 1, 31),
        ...         macro_indicators=["DFF", "UNRATE"]
        ...     )
        ...     print(counts)
        {'price': 31, 'news': 42, 'macro': 2}
    """
    logger.info(f"Ingesting all data types for {symbol}")

    results = {}

    # Ingest price data
    price_count = ingest_price_data(
        session, symbol, start_date, end_date, provider_config_path
    )
    results["price"] = price_count

    # Ingest news data
    news_start = datetime.combine(start_date, datetime.min.time())
    news_end = datetime.combine(end_date, datetime.max.time())
    news_count = ingest_news_data(
        session, symbol, news_start, news_end, limit=200, provider_config_path=provider_config_path
    )
    results["news"] = news_count

    # Ingest macro data
    if macro_indicators is None:
        macro_indicators = ["DFF", "UNRATE", "CPIAUCSL"]  # Default indicators

    macro_count = 0
    for indicator_code in macro_indicators:
        count = ingest_macro_data(
            session, indicator_code, start_date, end_date, provider_config_path
        )
        macro_count += count

    results["macro"] = macro_count

    logger.info(f"Ingestion complete: {results}")

    return results


def ingest_incremental_all(
    session: Session,
    symbols: List[str],
    days_back: int = 7,
    macro_indicators: Optional[List[str]] = None,
    provider_config_path: str = "config/providers.yaml",
) -> Dict[str, Dict[str, int]]:
    """
    Incremental ingestion for all data types and symbols.

    Args:
        session: Database session
        symbols: List of symbols to ingest
        days_back: Number of days to look back
        macro_indicators: List of macro indicators
        provider_config_path: Path to provider config

    Returns:
        Dict mapping symbol to ingest counts

    Example:
        >>> with get_session() as session:
        ...     results = ingest_incremental_all(
        ...         session,
        ...         ["BTC-USD", "ETH-USD"],
        ...         days_back=7
        ...     )
    """
    logger.info(f"Running incremental ingestion for {len(symbols)} symbols")

    today = date.today()
    start_date = today - timedelta(days=days_back)

    results = {}

    for symbol in symbols:
        counts = ingest_all_data_types(
            session,
            symbol,
            start_date,
            today,
            macro_indicators=macro_indicators,
            provider_config_path=provider_config_path,
        )
        results[symbol] = counts

    logger.info(f"Incremental ingestion complete for {len(symbols)} symbols")

    return results
