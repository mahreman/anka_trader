"""
Command-line interface for Otonom Trader.
"""
import logging
import sys
from typing import Optional

import typer
from sqlalchemy.orm import Session

from . import __version__
from .config import (
    ANOMALY_ZSCORE_THRESHOLD,
    ANOMALY_VOLUME_QUANTILE,
    ANOMALY_ROLLING_WINDOW,
    LOG_LEVEL,
)
from .data import get_engine, get_session, init_db
from .data.symbols import get_p0_assets, get_asset_by_symbol
from .data.ingest import ingest_all_assets
from .data.utils import get_date_range
from .data.schema import Anomaly as AnomalyORM, Decision as DecisionORM, Symbol
from .analytics import detect_anomalies_all_assets
from .patron import run_daily_decision_pass, format_decision_summary
from .patron.reporter import format_anomaly_list
from .domain import Anomaly as AnomalyDomain, Decision as DecisionDomain, AnomalyType

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="otonom-trader",
    help="Otonom Trader - Autonomous trading system with anomaly detection",
    add_completion=False,
)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"Otonom Trader v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Otonom Trader - P0 Core Version"""
    pass


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Force reinitialize database"),
):
    """
    Initialize the database schema.
    """
    try:
        if force:
            typer.echo("⚠️  Force mode: This will drop all existing tables!")
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                typer.echo("Cancelled.")
                raise typer.Exit()

            # Drop all tables
            from .data.schema import Base

            engine = get_engine()
            Base.metadata.drop_all(bind=engine)
            typer.echo("Dropped all tables")

        init_db()
        typer.echo("✓ Database initialized successfully")

    except Exception as e:
        typer.echo(f"✗ Error initializing database: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest_data(
    start: Optional[str] = typer.Option(
        None, "--start", help="Start date (YYYY-MM-DD)"
    ),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Specific symbol to ingest (default: all P0 assets)"
    ),
):
    """
    Fetch and store historical OHLCV data.

    Examples:
        otonom-trader ingest-data --start 2013-01-01
        otonom-trader ingest-data --symbol BTC-USD --start 2020-01-01
    """
    try:
        # Parse date range
        start_date, end_date = get_date_range(start, end)
        typer.echo(f"Ingesting data from {start_date} to {end_date}")

        # Get assets
        if symbol:
            assets = [get_asset_by_symbol(symbol)]
            typer.echo(f"Asset: {symbol}")
        else:
            assets = get_p0_assets()
            typer.echo(f"Assets: {len(assets)} P0 assets")

        # Ingest data
        with next(get_session()) as session:
            results = ingest_all_assets(session, start_date, end_date, assets)

        # Display results
        typer.echo("\nResults:")
        for sym, count in results.items():
            status = "✓" if count > 0 else "✗"
            typer.echo(f"  {status} {sym:>10}: {count:>5} bars")

        total = sum(results.values())
        typer.echo(f"\nTotal: {total} bars ingested")

    except Exception as e:
        typer.echo(f"✗ Error ingesting data: {e}", err=True)
        logger.exception("Ingest failed")
        raise typer.Exit(code=1)


@app.command()
def detect_anomalies(
    k: float = typer.Option(
        ANOMALY_ZSCORE_THRESHOLD, "--k", help="Z-score threshold"
    ),
    q: float = typer.Option(
        ANOMALY_VOLUME_QUANTILE, "--q", help="Volume quantile threshold (0-1)"
    ),
    window: int = typer.Option(
        ANOMALY_ROLLING_WINDOW, "--window", help="Rolling window size (days)"
    ),
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Specific symbol (default: all P0 assets)"
    ),
):
    """
    Detect anomalies (spikes/crashes) in price data.

    Examples:
        otonom-trader detect-anomalies
        otonom-trader detect-anomalies --k 3.0 --q 0.9
        otonom-trader detect-anomalies --symbol BTC-USD
    """
    try:
        typer.echo(f"Detecting anomalies (k={k}, q={q}, window={window})")

        # Get assets
        if symbol:
            assets = [get_asset_by_symbol(symbol)]
        else:
            assets = get_p0_assets()

        # Detect anomalies
        with next(get_session()) as session:
            results = detect_anomalies_all_assets(
                session=session, assets=assets, k=k, q=q, window=window
            )

        # Display results
        typer.echo("\nResults:")
        total = 0
        for sym, anomalies in results.items():
            count = len(anomalies)
            total += count
            spike_up = sum(1 for a in anomalies if a.anomaly_type == AnomalyType.SPIKE_UP)
            spike_down = sum(
                1 for a in anomalies if a.anomaly_type == AnomalyType.SPIKE_DOWN
            )
            typer.echo(
                f"  {sym:>10}: {count:>3} anomalies "
                f"(↑{spike_up} ↓{spike_down})"
            )

        typer.echo(f"\nTotal: {total} anomalies detected")

    except Exception as e:
        typer.echo(f"✗ Error detecting anomalies: {e}", err=True)
        logger.exception("Anomaly detection failed")
        raise typer.Exit(code=1)


@app.command()
def list_anomalies(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
):
    """
    List detected anomalies.

    Examples:
        otonom-trader list-anomalies
        otonom-trader list-anomalies --symbol BTC-USD --limit 10
    """
    try:
        with next(get_session()) as session:
            query = session.query(AnomalyORM, Symbol.symbol).join(Symbol)

            if symbol:
                query = query.filter(Symbol.symbol == symbol)

            query = query.order_by(AnomalyORM.date.desc()).limit(limit)
            results = query.all()

        if not results:
            typer.echo("No anomalies found")
            return

        # Convert to domain objects
        anomalies = [
            AnomalyDomain(
                asset_symbol=sym,
                date=a.date,
                anomaly_type=AnomalyType(a.anomaly_type),
                abs_return=a.abs_return,
                zscore=a.zscore,
                volume_rank=a.volume_rank,
                comment=a.comment,
            )
            for a, sym in results
        ]

        # Display
        typer.echo(format_anomaly_list(anomalies))

    except Exception as e:
        typer.echo(f"✗ Error listing anomalies: {e}", err=True)
        logger.exception("List anomalies failed")
        raise typer.Exit(code=1)


@app.command()
def run_patron(
    days: int = typer.Option(30, "--days", help="Look back N days for anomalies"),
):
    """
    Run Patron decision engine on recent anomalies.

    Generates BUY/SELL/HOLD signals based on anomaly + trend analysis.

    Examples:
        otonom-trader run-patron
        otonom-trader run-patron --days 60
    """
    try:
        typer.echo(f"Running Patron on anomalies from last {days} days...")

        with next(get_session()) as session:
            results = run_daily_decision_pass(session, days_back=days)

        if not results:
            typer.echo("No decisions generated (no recent anomalies found)")
            return

        # Flatten decisions for summary
        all_decisions = []
        for decisions in results.values():
            all_decisions.extend(decisions)

        # Display summary
        typer.echo(format_decision_summary(all_decisions))

        # Display individual decisions
        typer.echo("\nRecent Decisions:")
        typer.echo("-" * 80)

        # Sort by date descending
        all_decisions.sort(key=lambda d: d.date, reverse=True)

        for decision in all_decisions[:20]:  # Show top 20
            from .patron.reporter import format_decision

            typer.echo(format_decision(decision))

        if len(all_decisions) > 20:
            typer.echo(f"\n... and {len(all_decisions) - 20} more")

        typer.echo(f"\n✓ Generated {len(all_decisions)} decisions")

    except Exception as e:
        typer.echo(f"✗ Error running Patron: {e}", err=True)
        logger.exception("Patron execution failed")
        raise typer.Exit(code=1)


@app.command()
def show_decisions(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
    signal: Optional[str] = typer.Option(
        None, "--signal", help="Filter by signal (BUY/SELL/HOLD)"
    ),
):
    """
    Display trading decisions.

    Examples:
        otonom-trader show-decisions
        otonom-trader show-decisions --symbol SUGAR --limit 10
        otonom-trader show-decisions --signal BUY
    """
    try:
        with next(get_session()) as session:
            query = session.query(DecisionORM, Symbol.symbol).join(Symbol)

            if symbol:
                query = query.filter(Symbol.symbol == symbol)

            if signal:
                query = query.filter(DecisionORM.signal == signal.upper())

            query = query.order_by(DecisionORM.date.desc()).limit(limit)
            results = query.all()

        if not results:
            typer.echo("No decisions found")
            return

        # Display
        typer.echo("Date       | Symbol     | Signal | Confidence | Reason")
        typer.echo("-" * 80)

        for d, sym in results:
            # Truncate reason
            reason = d.reason if len(d.reason) <= 50 else d.reason[:47] + "..."
            typer.echo(
                f"{d.date} | {sym:>10} | {d.signal:>4} | "
                f"{d.confidence:>4.2f}       | {reason}"
            )

        typer.echo(f"\nShowing {len(results)} decisions")

    except Exception as e:
        typer.echo(f"✗ Error showing decisions: {e}", err=True)
        logger.exception("Show decisions failed")
        raise typer.Exit(code=1)


@app.command()
def detect_regimes(
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Specific symbol (default: all P0 assets)"
    ),
    vol_window: int = typer.Option(20, "--vol-window", help="Volatility window (days)"),
    trend_window: int = typer.Option(20, "--trend-window", help="Trend window (days)"),
    k_regimes: int = typer.Option(3, "--k-regimes", help="Number of regime clusters"),
    cusum_threshold: float = typer.Option(
        3.0, "--cusum-threshold", help="CUSUM threshold for structural breaks"
    ),
):
    """
    Detect market regimes and structural breaks (P1 feature).

    Uses K-means clustering on rolling volatility to classify regimes
    and CUSUM to detect structural breaks.

    Examples:
        otonom-trader detect-regimes
        otonom-trader detect-regimes --symbol BTC-USD
        otonom-trader detect-regimes --k-regimes 4 --cusum-threshold 4.0
    """
    try:
        from .analytics.regime import (
            compute_regimes_for_symbol,
            compute_regimes_all_symbols,
            persist_regimes,
        )

        typer.echo(
            f"Detecting regimes (vol_window={vol_window}, trend_window={trend_window}, "
            f"k={k_regimes}, cusum_threshold={cusum_threshold})"
        )

        with next(get_session()) as session:
            if symbol:
                # Single symbol
                regimes = compute_regimes_for_symbol(
                    session,
                    symbol=symbol,
                    vol_window=vol_window,
                    trend_window=trend_window,
                    k_regimes=k_regimes,
                    cusum_threshold=cusum_threshold,
                )
                persist_regimes(session, regimes)

                # Count structural breaks
                breaks = sum(1 for r in regimes if r.is_structural_break)
                typer.echo(f"\n{symbol}:")
                typer.echo(f"  Total regime points: {len(regimes)}")
                typer.echo(f"  Structural breaks:   {breaks}")

            else:
                # All symbols
                assets = get_p0_assets()
                symbols = [a.symbol for a in assets]

                results = compute_regimes_all_symbols(
                    session,
                    symbols=symbols,
                    vol_window=vol_window,
                    trend_window=trend_window,
                    k_regimes=k_regimes,
                    cusum_threshold=cusum_threshold,
                )

                # Persist all
                total_points = 0
                total_breaks = 0

                typer.echo("\nResults:")
                for sym, regimes in results.items():
                    if regimes:
                        persist_regimes(session, regimes)
                        breaks = sum(1 for r in regimes if r.is_structural_break)
                        total_points += len(regimes)
                        total_breaks += breaks
                        typer.echo(
                            f"  {sym:>10}: {len(regimes):>4} points, {breaks:>3} breaks"
                        )
                    else:
                        typer.echo(f"  {sym:>10}: No data")

                typer.echo(f"\nTotal: {total_points} regime points, {total_breaks} structural breaks")

    except Exception as e:
        typer.echo(f"✗ Error detecting regimes: {e}", err=True)
        logger.exception("Regime detection failed")
        raise typer.Exit(code=1)


@app.command()
def list_regimes(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
    breaks_only: bool = typer.Option(
        False, "--breaks-only", help="Show only structural breaks"
    ),
):
    """
    List detected regimes (P1 feature).

    Examples:
        otonom-trader list-regimes
        otonom-trader list-regimes --symbol BTC-USD --limit 10
        otonom-trader list-regimes --breaks-only
    """
    try:
        from .data.schema import Regime as RegimeORM

        with next(get_session()) as session:
            query = session.query(RegimeORM, Symbol.symbol).join(Symbol)

            if symbol:
                query = query.filter(Symbol.symbol == symbol)

            if breaks_only:
                query = query.filter(RegimeORM.is_structural_break == 1)

            query = query.order_by(RegimeORM.date.desc()).limit(limit)
            results = query.all()

        if not results:
            typer.echo("No regimes found")
            return

        # Display
        typer.echo("Date       | Symbol     | Regime | Volatility | Trend    | Break")
        typer.echo("-" * 75)

        for r, sym in results:
            break_marker = "⚠" if r.is_structural_break else " "
            typer.echo(
                f"{r.date} | {sym:>10} | {r.regime_id:>6} | "
                f"{r.volatility:>10.6f} | {r.trend:>+8.6f} | {break_marker}"
            )

        typer.echo(f"\nShowing {len(results)} regime points")

    except Exception as e:
        typer.echo(f"✗ Error listing regimes: {e}", err=True)
        logger.exception("List regimes failed")
        raise typer.Exit(code=1)


@app.command()
def status():
    """
    Show database status and statistics.
    """
    try:
        from .data.schema import Regime as RegimeORM

        with next(get_session()) as session:
            # Count symbols
            symbol_count = session.query(Symbol).count()

            # Count bars
            from .data.schema import DailyBar

            bar_count = session.query(DailyBar).count()

            # Count anomalies
            anomaly_count = session.query(AnomalyORM).count()

            # Count decisions
            decision_count = session.query(DecisionORM).count()

            # Count regimes (P1)
            regime_count = session.query(RegimeORM).count()

            typer.echo("Database Status")
            typer.echo("=" * 40)
            typer.echo(f"Symbols:   {symbol_count:>6}")
            typer.echo(f"Bars:      {bar_count:>6}")
            typer.echo(f"Anomalies: {anomaly_count:>6}")
            typer.echo(f"Decisions: {decision_count:>6}")
            typer.echo(f"Regimes:   {regime_count:>6}  [P1]")

    except Exception as e:
        typer.echo(f"✗ Error getting status: {e}", err=True)
        logger.exception("Status check failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
