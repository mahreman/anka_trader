"""
Daemon loop for the autonomous trading system.

This module coordinates:
1. Incremental data ingest (prices, news, macro)
2. Anomaly detection
3. Patron decision generation
4. Paper trade execution

The daemon is intended to be called periodically by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import logging

from sqlalchemy.orm import Session

from otonom_trader.domain import Asset, AssetClass
from otonom_trader.analytics.anomaly import detect_anomalies_for_universe
from otonom_trader.data.ingest import ingest_incremental
from otonom_trader.data.ingest_providers import (
    ingest_intraday_bars_all,
    ingest_news_for_universe,
    ingest_macro_for_universe,
)
from otonom_trader.data.schema import DaemonRun
from otonom_trader.data.symbols import get_tracked_assets, get_p0_assets
from otonom_trader.patron.rules import run_patron_decisions
from .paper_trader import PaperTrader

log = logging.getLogger(__name__)


# Paper trader cache keyed by DB URL so repeated orchestrator runs reuse
# the same simulated portfolio for a database/file.
_PAPER_TRADER_CACHE: Dict[str, PaperTrader] = {}


def _normalize_symbol_list(symbols: Optional[Sequence[str]]) -> List[str]:
    """Return a cleaned list of unique ticker strings preserving order."""

    if not symbols:
        return []

    normalized: List[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = (raw or "").strip()
        if not sym:
            continue
        key = sym.upper()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(sym)
    return normalized


def _resolve_universe_symbols(session: Session, config: "DaemonConfig") -> List[str]:
    """Determine which tickers the daemon should operate on."""

    cleaned = _normalize_symbol_list(config.universe)
    if cleaned:
        return cleaned

    tracked_assets = get_tracked_assets(session, include_p0_fallback=True)
    if tracked_assets:
        return [asset.symbol for asset in tracked_assets]

    # Absolute fallback to the static P0 list.
    return [asset.symbol for asset in get_p0_assets()]


def _assets_for_ingest(session: Session, symbols: Sequence[str]) -> List[Asset]:
    """Build Asset objects for ingestion routines."""

    available = {
        asset.symbol: asset
        for asset in get_tracked_assets(session, include_p0_fallback=True)
    }

    assets: List[Asset] = []
    for symbol in symbols:
        asset = available.get(symbol)
        if asset is None:
            asset = Asset(
                symbol=symbol,
                name=symbol,
                asset_class=AssetClass.OTHER,
                base_currency="USD",
            )
        assets.append(asset)
    return assets


@dataclass
class DaemonConfig:
    """
    Configuration for a single daemon run.

    Attributes
    ----------
    universe : List[str]
        Optional list of symbols to process. If empty, fall back to all tracked
        symbols (or the static P0 list if the database has none).
    ingest_days_back : int
        How many days back to ingest data for incremental daily ingest.
    anomaly_lookback_days : int
        How many days back anomaly detection should consider.
    price_interval : str
        Price interval for intraday ingest (e.g. "15m", "1h", "1d").
    ingest_news : bool
        Whether to ingest news.
    ingest_macro : bool
        Whether to ingest macro indicators.
    use_ensemble : bool
        Whether Patron should use ensemble mode.
    paper_trade_enabled : bool
        Whether paper trades should be executed.
    paper_trade_risk_pct : float
        Percent of equity risked per trade.
    initial_cash : float
        Initial virtual cash balance for paper trading.
    """

    universe: Optional[List[str]] = field(default_factory=list)
    ingest_days_back: int = 7
    anomaly_lookback_days: int = 30
    price_interval: str = "15m"
    ingest_news: bool = True
    ingest_macro: bool = True
    use_ensemble: bool = True
    paper_trade_enabled: bool = True
    paper_trade_risk_pct: float = 1.0
    initial_cash: float = 100_000.0


def run_daemon_cycle(
    session: Session,
    config: DaemonConfig,
    paper_trader,
) -> DaemonRun:
    """
    Run a full daemon cycle and persist a DaemonRun row.

    Returns
    -------
    DaemonRun
        The persisted run object.
    """
    start_time = datetime.now(timezone.utc)

    run = DaemonRun(
        timestamp=start_time,
        status="RUNNING",
        bars_ingested=0,
        anomalies_detected=0,
        decisions_made=0,
        trades_executed=0,
    )
    session.add(run)
    session.commit()

    try:
        log.info("=" * 60)
        log.info("Starting daemon cycle at %s", start_time)
        log.info("=" * 60)

        # ------------------------------------------------------------------
        # 1) Incremental data ingest
        # ------------------------------------------------------------------
        log.info("[1/4] Incremental data ingest...")

        symbols = _resolve_universe_symbols(session, config)
        assets = _assets_for_ingest(session, symbols)

        # Her durumda günlük barları ingest et ki anomaly engine çalışabilsin.
        ingest_results = ingest_incremental(
            session, days_back=config.ingest_days_back, assets=assets
        )
        daily_bars_ingested = sum(ingest_results.values())

        # Interval 1d'den farklıysa ek olarak intraday barları da çek.
        intraday_bars_ingested = 0
        if config.price_interval.lower() != "1d":
            intraday_bars_ingested = ingest_intraday_bars_all(
                session,
                interval=config.price_interval,
                lookback_days=config.ingest_days_back,
            )

        bars_ingested = daily_bars_ingested + intraday_bars_ingested
        run.bars_ingested = bars_ingested

        log.info("  ✓ Ingested %d bars across %d assets", bars_ingested, len(symbols))

        # Haber ingest
        if config.ingest_news:
            log.info("  Ingesting news data...")
            news_count = ingest_news_for_universe(session, symbols, limit=5)
            log.info("  ✓ Ingested %d news articles", news_count)

        # Makro ingest
        if config.ingest_macro:
            log.info("  Ingesting macro indicators...")
            macro_count = ingest_macro_for_universe(session)
            log.info("  ✓ Ingested %d macro indicator rows", macro_count)

        session.commit()

        # ------------------------------------------------------------------
        # 2) Anomaly detection
        # ------------------------------------------------------------------
        log.info("[2/4] Anomaly detection...")
        anomalies = detect_anomalies_for_universe(
            session,
            symbols,
            lookback_days=config.anomaly_lookback_days,
        )
        run.anomalies_detected = len(anomalies)
        log.info("  ✓ Detected %d anomalies", len(anomalies))
        session.commit()

        # ------------------------------------------------------------------
        # 3) Patron decision generation
        # ------------------------------------------------------------------
        log.info("[3/4] Patron decision generation...")
        decisions = run_patron_decisions(
            session,
            anomalies,
            use_ensemble=config.use_ensemble,
        )
        run.decisions_made = len(decisions)
        log.info("  ✓ Generated %d decisions", len(decisions))
        session.commit()

        # ------------------------------------------------------------------
        # 4) Paper trade execution
        # ------------------------------------------------------------------
        if config.paper_trade_enabled:
            log.info("[4/4] Paper trade execution...")
            trades = paper_trader.execute_decisions(
                decisions, risk_pct=config.paper_trade_risk_pct
            )
            run.trades_executed = len(trades)
            log.info("  ✓ Executed %d paper trades", len(trades))
        else:
            log.info("[4/4] Paper trade execution... SKIPPED")

        # ------------------------------------------------------------------
        # Finalize
        # ------------------------------------------------------------------
        end_time = datetime.now(timezone.utc)
        run.duration_seconds = (end_time - start_time).total_seconds()
        run.status = "SUCCESS"
        session.commit()

        log.info("=" * 60)
        log.info(
            "Daemon cycle completed in %.2fs", run.duration_seconds or 0.0
        )
        log.info(
            "  Bars ingested:      %d\n"
            "  Anomalies detected: %d\n"
            "  Decisions made:     %d\n"
            "  Trades executed:    %d",
            run.bars_ingested or 0,
            run.anomalies_detected or 0,
            run.decisions_made or 0,
            run.trades_executed or 0,
        )
        log.info("=" * 60)

        return run

    except Exception as exc:
        log.exception("Daemon cycle failed: %s", exc)
        run.status = "FAILED"
        run.error_message = str(exc)
        session.commit()
        raise


def get_or_create_paper_trader(
    session: Session, initial_cash: float = 100_000.0
) -> PaperTrader:
    """Return a cached :class:`PaperTrader` tied to the current DB.

    The orchestrator may call this helper on every cycle with a new
    ``Session`` object; we cache per-database URL so that paper-trading
    state persists across runs that share the same SQLite file. When a
    cached trader is reused we simply update its session handle so it can
    continue persisting trades in the active transaction context.
    """

    bind = session.get_bind()
    cache_key = str(bind.url) if bind is not None else f"session:{id(session)}"

    trader = _PAPER_TRADER_CACHE.get(cache_key)
    if trader is None:
        trader = PaperTrader(session, initial_cash=initial_cash)
        _PAPER_TRADER_CACHE[cache_key] = trader
    else:
        trader.session = session

    return trader
