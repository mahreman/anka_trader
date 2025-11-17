"""
Orchestration module for ingesting data from multiple providers.
"""
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..data import get_engine, get_session, init_db, Symbol, NewsArticle, MacroIndicator
from ..data.ingest import upsert_daily_bars
from ..domain import Asset, AssetClass
from .config import (
    load_provider_config,
    get_price_config,
    get_news_config,
    get_macro_config,
    get_provider_credentials,
)
from .binance_provider import BinanceProvider
from .yfinance_provider import YFinanceProvider
from .newsapi_provider import NewsAPIProvider
from .fred_provider import FREDProvider

logger = logging.getLogger(__name__)


def ingest_price_data(
    session: Session,
    config: dict,
    start: date,
    end: Optional[date] = None,
    universe_name: str = "crypto",
) -> dict[str, int]:
    """
    Ingest price data using configured providers.

    Args:
        session: Database session
        config: Provider configuration
        start: Start date
        end: End date (default: today)
        universe_name: Universe to ingest (e.g., "crypto", "fx", "equity", "all")

    Returns:
        Dictionary mapping symbol to number of bars ingested
    """
    if end is None:
        end = date.today()

    price_config = get_price_config(config)
    primary_provider_name = price_config.get("primary", "yfinance")
    universes = price_config.get("universes", {})

    # Get symbols for universe
    if universe_name == "all":
        symbols = []
        for universe_symbols in universes.values():
            symbols.extend(universe_symbols.get("symbols", []))
    else:
        universe = universes.get(universe_name, {})
        symbols = universe.get("symbols", [])

    if not symbols:
        logger.warning(f"No symbols found for universe: {universe_name}")
        return {}

    # Initialize provider
    if primary_provider_name == "binance":
        provider_creds = get_provider_credentials(config, "binance")
        provider = BinanceProvider(provider_creds)
    elif primary_provider_name == "yfinance":
        provider_creds = get_provider_credentials(config, "yfinance")
        provider = YFinanceProvider(provider_creds)
    else:
        raise ValueError(f"Unknown price provider: {primary_provider_name}")

    logger.info(
        f"Ingesting price data using {primary_provider_name} "
        f"for {len(symbols)} symbols from {start} to {end}"
    )

    results = {}

    for symbol in symbols:
        try:
            # Fetch data
            df = provider.fetch_data(start=start, end=end, symbol=symbol)

            if df.empty:
                logger.warning(f"No data fetched for {symbol}")
                results[symbol] = 0
                continue

            # Determine asset class based on symbol
            if symbol.endswith("USDT"):
                asset_class = AssetClass.CRYPTO
            elif "=" in symbol or symbol.endswith("=X"):
                asset_class = AssetClass.FX
            else:
                asset_class = AssetClass.EQUITY

            # Create Asset object
            asset = Asset(
                symbol=symbol,
                name=symbol,  # We don't have name info, use symbol
                asset_class=asset_class,
            )

            # Upsert to database
            count = upsert_daily_bars(df, asset, session)
            results[symbol] = count

        except Exception as e:
            logger.error(f"Failed to ingest {symbol}: {e}")
            results[symbol] = 0

    return results


def ingest_news_data(
    session: Session,
    config: dict,
    start: date,
    end: Optional[date] = None,
) -> int:
    """
    Ingest news data using configured providers.

    Args:
        session: Database session
        config: Provider configuration
        start: Start date
        end: End date (default: today)

    Returns:
        Number of articles ingested
    """
    if end is None:
        end = date.today()

    news_config = get_news_config(config)
    provider_name = news_config.get("provider", "newsapi")

    if provider_name == "newsapi":
        provider_creds = get_provider_credentials(config, "newsapi")
        newsapi_config = news_config.get("newsapi", {})

        # Merge credentials with config
        full_config = {**provider_creds, **newsapi_config}
        provider = NewsAPIProvider(full_config)

        tickers = newsapi_config.get("tickers", ["BTC", "ETH"])
        max_per_symbol = newsapi_config.get("max_per_symbol", 50)
    else:
        raise ValueError(f"Unknown news provider: {provider_name}")

    logger.info(
        f"Ingesting news data using {provider_name} "
        f"for {len(tickers)} tickers from {start} to {end}"
    )

    total_count = 0

    for ticker in tickers:
        try:
            # Fetch news
            df = provider.fetch_data(start=start, end=end, query=ticker)

            if df.empty:
                logger.warning(f"No news fetched for ticker: {ticker}")
                continue

            # Limit articles per symbol
            if len(df) > max_per_symbol:
                df = df.head(max_per_symbol)

            # Upsert to database
            count = 0
            for _, row in df.iterrows():
                # Check if article already exists (by URL)
                existing = session.query(NewsArticle).filter_by(url=row["url"]).first()

                if existing:
                    # Skip duplicate
                    continue

                # Insert new article
                article = NewsArticle(
                    provider=provider_name,
                    title=row["title"],
                    description=row.get("description"),
                    url=row["url"],
                    published_at=row["published_at"],
                    source_name=row.get("source_name"),
                    symbols=ticker,  # Tag with ticker
                )
                session.add(article)
                count += 1

            session.commit()
            logger.info(f"Ingested {count} articles for ticker: {ticker}")
            total_count += count

        except Exception as e:
            logger.error(f"Failed to ingest news for {ticker}: {e}")
            session.rollback()

    return total_count


def ingest_macro_data(
    session: Session,
    config: dict,
    start: date,
    end: Optional[date] = None,
) -> int:
    """
    Ingest macroeconomic data using configured providers.

    Args:
        session: Database session
        config: Provider configuration
        start: Start date
        end: End date (default: today)

    Returns:
        Number of observations ingested
    """
    if end is None:
        end = date.today()

    macro_config = get_macro_config(config)
    provider_name = macro_config.get("provider", "fred")

    if provider_name == "fred":
        provider_creds = get_provider_credentials(config, "fred")
        fred_config = macro_config.get("fred", {})

        # Merge credentials with config
        full_config = {**provider_creds, **fred_config}
        provider = FREDProvider(full_config)

        series_list = fred_config.get("series", ["DGS10", "CPIAUCSL"])
    else:
        raise ValueError(f"Unknown macro provider: {provider_name}")

    logger.info(
        f"Ingesting macro data using {provider_name} "
        f"for {len(series_list)} series from {start} to {end}"
    )

    total_count = 0

    for series_id in series_list:
        try:
            # Fetch data
            df = provider.fetch_data(start=start, end=end, series_id=series_id)

            if df.empty:
                logger.warning(f"No data fetched for series: {series_id}")
                continue

            # Upsert to database
            count = 0
            for _, row in df.iterrows():
                # Check if observation already exists
                existing = (
                    session.query(MacroIndicator)
                    .filter_by(series_id=row["series_id"], date=row["date"])
                    .first()
                )

                if existing:
                    # Update existing
                    existing.value = float(row["value"])
                    existing.series_name = row.get("series_name")
                    existing.units = row.get("units")
                else:
                    # Insert new
                    indicator = MacroIndicator(
                        provider=provider_name,
                        series_id=row["series_id"],
                        series_name=row.get("series_name"),
                        date=row["date"],
                        value=float(row["value"]),
                        units=row.get("units"),
                    )
                    session.add(indicator)

                count += 1

            session.commit()
            logger.info(f"Ingested {count} observations for series: {series_id}")
            total_count += count

        except Exception as e:
            logger.error(f"Failed to ingest macro data for {series_id}: {e}")
            session.rollback()

    return total_count


def ingest_all_data_types(
    start_date: str,
    end_date: Optional[str] = None,
    providers_config_path: str = "config/providers.yaml",
    universe_name: str = "crypto",
) -> dict:
    """
    Ingest all data types (price, news, macro) from configured providers.

    This is the main orchestration function for full data sync.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format, defaults to today)
        providers_config_path: Path to providers.yaml config file
        universe_name: Universe to ingest (e.g., "crypto", "fx", "equity", "all")

    Returns:
        Dictionary with ingestion results
    """
    # Parse dates
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today()

    # Load configuration
    config = load_provider_config(providers_config_path)

    # Initialize database
    engine = get_engine()
    init_db(engine)

    results = {}

    # Ingest price data
    logger.info("=" * 60)
    logger.info("Starting price data ingestion")
    logger.info("=" * 60)

    with next(get_session()) as session:
        price_results = ingest_price_data(
            session, config, start, end, universe_name
        )
        results["price"] = price_results

    # Ingest news data
    logger.info("=" * 60)
    logger.info("Starting news data ingestion")
    logger.info("=" * 60)

    with next(get_session()) as session:
        news_count = ingest_news_data(session, config, start, end)
        results["news"] = news_count

    # Ingest macro data
    logger.info("=" * 60)
    logger.info("Starting macro data ingestion")
    logger.info("=" * 60)

    with next(get_session()) as session:
        macro_count = ingest_macro_data(session, config, start, end)
        results["macro"] = macro_count

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"Results: {results}")
    logger.info("=" * 60)

    return results


def ingest_incremental_all(
    providers_config_path: str = "config/providers.yaml",
    universe_name: str = "crypto",
    days_back: int = 7,
) -> dict:
    """
    Incremental ingestion: fetch only recent data for all data types.

    This is the main orchestration function for incremental updates.

    Args:
        providers_config_path: Path to providers.yaml config file
        universe_name: Universe to ingest (e.g., "crypto", "fx", "equity", "all")
        days_back: Number of days to look back (default: 7)

    Returns:
        Dictionary with ingestion results
    """
    # Calculate date range
    end = date.today()
    start = end - timedelta(days=days_back)

    logger.info(f"Running incremental ingestion from {start} to {end}")

    # Use full sync function with incremental date range
    return ingest_all_data_types(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        providers_config_path=providers_config_path,
        universe_name=universe_name,
    )
