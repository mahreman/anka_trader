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
from .data.ingest import ingest_all_assets, ingest_incremental
from .data.utils import get_date_range
from .data.schema import Anomaly as AnomalyORM, Decision as DecisionORM, Symbol
from .analytics import detect_anomalies_all_assets
from .patron import run_daily_decision_pass, format_decision_summary
from .patron.reporter import format_anomaly_list
from .domain import Anomaly as AnomalyDomain, Decision as DecisionDomain, AnomalyType
from .utils import utc_now

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

# Add experiments sub-command
from .cli_experiments import app as experiments_app
app.add_typer(experiments_app, name="experiments")

# Add RL sub-command
from .cli_rl import app as rl_app
app.add_typer(rl_app, name="rl")

# Add research sub-command
from .cli_research import app as research_app
app.add_typer(research_app, name="research")

# Add strategy sub-command
from .cli_strategy import app as strategy_app
app.add_typer(strategy_app, name="strategy")

# Add backtest sub-command
from .cli_backtest import app as backtest_app
app.add_typer(backtest_app, name="backtest")


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
    ensemble: bool = typer.Option(True, "--ensemble/--no-ensemble", help="Use ensemble mode (P2 default)"),
):
    """
    Run Patron decision engine on recent anomalies.

    Generates BUY/SELL/HOLD signals based on anomaly + trend analysis.

    P2 Default: Ensemble mode with 3 analysts (Technical, News/Macro/LLM, Regime/DSI).
    Use --no-ensemble to revert to P0 simple rules.

    Examples:
        otonom-trader run-patron              # P2: ensemble mode (default)
        otonom-trader run-patron --days 60    # P2: ensemble mode, 60 days
        otonom-trader run-patron --no-ensemble  # P0: simple rules only
    """
    try:
        if ensemble:
            typer.echo(f"Running Patron (ENSEMBLE MODE) on anomalies from last {days} days...")
        else:
            typer.echo(f"Running Patron on anomalies from last {days} days...")

        with next(get_session()) as session:
            results = run_daily_decision_pass(session, days_back=days, use_ensemble=ensemble)

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
def compute_dsi(
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Specific symbol (default: all P0 assets)"
    ),
    window: int = typer.Option(60, "--window", help="Rolling window size (days)"),
    outlier_z: float = typer.Option(4.0, "--outlier-z", help="Z-score threshold for outliers"),
):
    """
    Compute Data Health Index (DSI) for assets (P1 feature).

    DSI measures data quality based on:
    - Missing data ratio
    - Extreme outlier ratio
    - Volume jump ratio

    Examples:
        otonom-trader compute-dsi
        otonom-trader compute-dsi --symbol BTC-USD
        otonom-trader compute-dsi --window 90 --outlier-z 5.0
    """
    try:
        from .analytics.dsi import (
            compute_dsi_for_symbol,
            compute_dsi_all_symbols,
            persist_dsi,
        )

        typer.echo(f"Computing DSI (window={window}, outlier_z={outlier_z})")

        with next(get_session()) as session:
            if symbol:
                # Single symbol
                dsi_points = compute_dsi_for_symbol(
                    session,
                    symbol=symbol,
                    window=window,
                    outlier_z=outlier_z,
                )
                persist_dsi(session, dsi_points)

                # Calculate statistics
                if dsi_points:
                    avg_dsi = sum(p.dsi for p in dsi_points) / len(dsi_points)
                    min_dsi = min(p.dsi for p in dsi_points)
                    max_dsi = max(p.dsi for p in dsi_points)

                    typer.echo(f"\n{symbol}:")
                    typer.echo(f"  Total points:  {len(dsi_points)}")
                    typer.echo(f"  Average DSI:   {avg_dsi:.3f}")
                    typer.echo(f"  Min DSI:       {min_dsi:.3f}")
                    typer.echo(f"  Max DSI:       {max_dsi:.3f}")

            else:
                # All symbols
                assets = get_p0_assets()
                symbols = [a.symbol for a in assets]

                results = compute_dsi_all_symbols(
                    session,
                    symbols=symbols,
                    window=window,
                    outlier_z=outlier_z,
                )

                # Persist all
                total_points = 0
                typer.echo("\nResults:")

                for sym, dsi_points in results.items():
                    if dsi_points:
                        persist_dsi(session, dsi_points)
                        avg_dsi = sum(p.dsi for p in dsi_points) / len(dsi_points)
                        total_points += len(dsi_points)
                        typer.echo(
                            f"  {sym:>10}: {len(dsi_points):>4} points, "
                            f"avg DSI={avg_dsi:.3f}"
                        )
                    else:
                        typer.echo(f"  {sym:>10}: No data")

                typer.echo(f"\nTotal: {total_points} DSI points computed")

    except Exception as e:
        typer.echo(f"✗ Error computing DSI: {e}", err=True)
        logger.exception("DSI computation failed")
        raise typer.Exit(code=1)


@app.command()
def list_dsi(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
    min_dsi: Optional[float] = typer.Option(
        None, "--min-dsi", help="Show only records with DSI >= this value"
    ),
):
    """
    List computed DSI values (P1 feature).

    Examples:
        otonom-trader list-dsi
        otonom-trader list-dsi --symbol BTC-USD --limit 10
        otonom-trader list-dsi --min-dsi 0.8
    """
    try:
        from .data.schema import DataHealthIndex as DsiORM

        with next(get_session()) as session:
            query = session.query(DsiORM, Symbol.symbol).join(Symbol)

            if symbol:
                query = query.filter(Symbol.symbol == symbol)

            if min_dsi is not None:
                query = query.filter(DsiORM.dsi >= min_dsi)

            query = query.order_by(DsiORM.date.desc()).limit(limit)
            results = query.all()

        if not results:
            typer.echo("No DSI records found")
            return

        # Display
        typer.echo("Date       | Symbol     | DSI   | Missing | Outlier | Vol Jump")
        typer.echo("-" * 75)

        for d, sym in results:
            typer.echo(
                f"{d.date} | {sym:>10} | {d.dsi:.3f} | "
                f"{d.missing_ratio:.3f}   | {d.outlier_ratio:.3f}   | "
                f"{d.volume_jump_ratio:.3f}"
            )

        typer.echo(f"\nShowing {len(results)} DSI records")

    except Exception as e:
        typer.echo(f"✗ Error listing DSI: {e}", err=True)
        logger.exception("List DSI failed")
        raise typer.Exit(code=1)


@app.command()
def run_backtest(
    name: str = typer.Option(..., "--name", help="Hypothesis name"),
    rule: str = typer.Option(..., "--rule", help="Rule signature"),
    holding_days: int = typer.Option(5, "--holding-days", help="Holding period (days)"),
    slippage_bps: float = typer.Option(5.0, "--slippage-bps", help="Slippage (basis points)"),
    description: Optional[str] = typer.Option(None, "--description", help="Hypothesis description"),
):
    """
    Run event-based backtest on existing decisions (P1 feature).

    Creates or uses existing hypothesis and backtests all historical decisions.
    Requires anomalies and decisions to be already computed.

    Examples:
        otonom-trader run-backtest --name "P0_basic_v1" \\
            --rule "SPIKE_DOWN+Uptrend→BUY; SPIKE_UP+Downtrend→SELL; 5d hold"

        otonom-trader run-backtest --name "test_10d" --rule "Basic rules" \\
            --holding-days 10 --slippage-bps 10.0
    """
    try:
        from .eval.backtest import (
            BacktestConfig,
            create_or_get_hypothesis,
            run_event_backtest,
            get_backtest_summary,
        )

        typer.echo(f"Running backtest: {name}")
        typer.echo(f"Rule: {rule}")
        typer.echo(f"Config: holding_days={holding_days}, slippage_bps={slippage_bps}")

        with next(get_session()) as session:
            # Create or get hypothesis
            hypothesis = create_or_get_hypothesis(
                session,
                name=name,
                rule_signature=rule,
                description=description,
            )

            # Configure backtest
            config = BacktestConfig(
                holding_days=holding_days,
                slippage_bps=slippage_bps,
            )

            # Run backtest
            count = run_event_backtest(session, hypothesis, config)

            typer.echo(f"\n✓ Backtest completed: {count} trades simulated")

            # Show summary
            summary = get_backtest_summary(session, hypothesis.id)

            typer.echo("\nBacktest Summary:")
            typer.echo("=" * 40)
            typer.echo(f"Total trades:  {summary['total_trades']}")
            typer.echo(f"Win rate:      {summary['win_rate']*100:.2f}%")
            typer.echo(f"Avg PnL:       {summary['avg_pnl_pct']*100:+.2f}%")
            typer.echo(f"Total PnL:     {summary['total_pnl']:+.2f}")
            if summary['avg_win_pct']:
                typer.echo(f"Avg win:       {summary['avg_win_pct']*100:+.2f}%")
            if summary['avg_loss_pct']:
                typer.echo(f"Avg loss:      {summary['avg_loss_pct']*100:+.2f}%")

    except Exception as e:
        typer.echo(f"✗ Error running backtest: {e}", err=True)
        logger.exception("Backtest failed")
        raise typer.Exit(code=1)


@app.command()
def list_hypotheses(
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
):
    """
    List all hypotheses (P1 feature).

    Examples:
        otonom-trader list-hypotheses
        otonom-trader list-hypotheses --limit 10
    """
    try:
        from .data.schema import Hypothesis

        with next(get_session()) as session:
            hypotheses = (
                session.query(Hypothesis)
                .order_by(Hypothesis.created_at.desc())
                .limit(limit)
                .all()
            )

        if not hypotheses:
            typer.echo("No hypotheses found")
            return

        typer.echo("ID   | Name                    | Rule")
        typer.echo("-" * 70)

        for h in hypotheses:
            rule = h.rule_signature if len(h.rule_signature) <= 40 else h.rule_signature[:37] + "..."
            typer.echo(f"{h.id:>4} | {h.name:23} | {rule}")

        typer.echo(f"\nShowing {len(hypotheses)} hypotheses")

    except Exception as e:
        typer.echo(f"✗ Error listing hypotheses: {e}", err=True)
        logger.exception("List hypotheses failed")
        raise typer.Exit(code=1)


@app.command()
def show_backtest(
    hypothesis_id: Optional[int] = typer.Option(None, "--id", help="Hypothesis ID"),
    name: Optional[str] = typer.Option(None, "--name", help="Hypothesis name"),
    limit: int = typer.Option(20, "--limit", help="Max results to show"),
):
    """
    Show backtest results for a hypothesis (P1 feature).

    Examples:
        otonom-trader show-backtest --id 1
        otonom-trader show-backtest --name "P0_basic_v1"
        otonom-trader show-backtest --id 1 --limit 50
    """
    try:
        from .data.schema import Hypothesis, HypothesisResult
        from .eval.backtest import get_backtest_summary

        if not hypothesis_id and not name:
            typer.echo("✗ Must specify either --id or --name")
            raise typer.Exit(code=1)

        with next(get_session()) as session:
            # Find hypothesis
            if hypothesis_id:
                hypothesis = session.query(Hypothesis).get(hypothesis_id)
            else:
                hypothesis = session.query(Hypothesis).filter_by(name=name).first()

            if not hypothesis:
                typer.echo("✗ Hypothesis not found")
                raise typer.Exit(code=1)

            typer.echo(f"Hypothesis: {hypothesis.name}")
            typer.echo(f"Rule: {hypothesis.rule_signature}")
            typer.echo()

            # Get summary
            summary = get_backtest_summary(session, hypothesis.id)

            typer.echo("Summary:")
            typer.echo("=" * 40)
            typer.echo(f"Total trades:  {summary['total_trades']}")
            typer.echo(f"Win rate:      {summary['win_rate']*100:.2f}%")
            typer.echo(f"Avg PnL:       {summary['avg_pnl_pct']*100:+.2f}%")
            typer.echo(f"Total PnL:     {summary['total_pnl']:+.2f}")
            if summary['avg_win_pct']:
                typer.echo(f"Avg win:       {summary['avg_win_pct']*100:+.2f}%")
            if summary['avg_loss_pct']:
                typer.echo(f"Avg loss:      {summary['avg_loss_pct']*100:+.2f}%")
            typer.echo()

            # Show individual results
            results = (
                session.query(HypothesisResult, Symbol.symbol)
                .join(Symbol, HypothesisResult.symbol_id == Symbol.id)
                .filter(HypothesisResult.hypothesis_id == hypothesis.id)
                .order_by(HypothesisResult.entry_date.desc())
                .limit(limit)
                .all()
            )

            if results:
                typer.echo(f"Recent results (top {limit}):")
                typer.echo("Entry Date | Symbol     | PnL %    | Regime | DSI")
                typer.echo("-" * 60)

                for r, sym in results:
                    regime_str = f"{r.regime_id}" if r.regime_id is not None else "-"
                    dsi_str = f"{r.dsi:.2f}" if r.dsi is not None else "-"
                    typer.echo(
                        f"{r.entry_date} | {sym:>10} | {r.pnl_pct*100:>+7.2f}% | "
                        f"{regime_str:>6} | {dsi_str:>4}"
                    )

    except Exception as e:
        typer.echo(f"✗ Error showing backtest: {e}", err=True)
        logger.exception("Show backtest failed")
        raise typer.Exit(code=1)


@app.command()
def backtest_portfolio(
    hypothesis_id: Optional[int] = typer.Option(None, "--id", help="Hypothesis ID"),
    name: Optional[str] = typer.Option(None, "--name", help="Hypothesis name"),
    initial_cash: float = typer.Option(100000.0, "--initial-cash", help="Initial cash balance"),
    risk_pct: float = typer.Option(1.0, "--risk-pct", help="Risk per trade (% of equity)"),
):
    """
    Run portfolio-based backtest on hypothesis results (P2.5 feature).

    Replays event-based backtest trades with position sizing and portfolio tracking.

    Examples:
        otonom-trader backtest-portfolio --id 1
        otonom-trader backtest-portfolio --name "P1_smoke"
        otonom-trader backtest-portfolio --id 1 --initial-cash 50000 --risk-pct 2.0
    """
    try:
        from .eval.backtest_portfolio import (
            PortfolioConfig,
            run_portfolio_backtest,
            get_portfolio_summary,
        )
        from .data.schema import Hypothesis

        # Find hypothesis
        with next(get_session()) as session:
            if hypothesis_id is None and name is None:
                typer.echo("✗ Error: Must specify either --id or --name", err=True)
                raise typer.Exit(code=1)

            if hypothesis_id:
                hypothesis = (
                    session.query(Hypothesis)
                    .filter(Hypothesis.id == hypothesis_id)
                    .one_or_none()
                )
            else:
                hypothesis = (
                    session.query(Hypothesis)
                    .filter(Hypothesis.name == name)
                    .one_or_none()
                )

            if hypothesis is None:
                if hypothesis_id:
                    typer.echo(f"✗ Error: Hypothesis ID {hypothesis_id} not found", err=True)
                else:
                    typer.echo(f"✗ Error: Hypothesis '{name}' not found", err=True)
                raise typer.Exit(code=1)

            typer.echo(f"[P2.5] Running portfolio backtest for: {hypothesis.name}")
            typer.echo(f"Initial cash: ${initial_cash:,.2f}")
            typer.echo(f"Risk per trade: {risk_pct}% of equity")
            typer.echo("")

            # Configure portfolio
            config = PortfolioConfig(
                initial_cash=initial_cash,
                risk_per_trade_pct=risk_pct,
            )

            # Run backtest
            metrics = run_portfolio_backtest(
                session,
                hypothesis_id=hypothesis.id,
                config=config,
            )

            # Display results
            typer.echo("\n" + "=" * 60)
            typer.echo("PORTFOLIO BACKTEST RESULTS")
            typer.echo("=" * 60)
            typer.echo(f"Total trades:     {metrics.total_trades}")
            typer.echo(f"Win rate:         {metrics.win_rate * 100:.2f}%")
            typer.echo("")
            typer.echo(f"Initial equity:   ${config.initial_cash:,.2f}")
            typer.echo(f"Final equity:     ${metrics.final_equity:,.2f}")
            typer.echo(f"Total PnL:        ${metrics.total_pnl:+,.2f}")
            typer.echo(
                f"Return:           {((metrics.final_equity - config.initial_cash) / config.initial_cash * 100):+.2f}%"
            )
            typer.echo("")
            typer.echo(f"Max drawdown:     {metrics.max_drawdown * 100:.2f}%")
            typer.echo("")

            if metrics.total_trades > 0:
                typer.echo(f"Average win:      ${metrics.avg_win:,.2f}")
                typer.echo(f"Average loss:     ${metrics.avg_loss:,.2f}")
                if metrics.avg_loss != 0:
                    profit_factor = abs(metrics.avg_win / metrics.avg_loss)
                    typer.echo(f"Profit factor:    {profit_factor:.2f}")

            typer.echo("=" * 60)

            # Commit snapshots
            session.commit()
            typer.echo("\n✓ Portfolio backtest completed")

    except Exception as e:
        typer.echo(f"✗ Error running portfolio backtest: {e}", err=True)
        logger.exception("Portfolio backtest failed")
        raise typer.Exit(code=1)


@app.command()
def ingest_incremental(
    days_back: int = typer.Option(7, "--days-back", help="Days to look back if no data"),
):
    """
    Incremental data ingest - fetch only recent data (P3 feature).

    Examples:
        otonom-trader ingest-incremental
        otonom-trader ingest-incremental --days-back 30
    """
    try:
        typer.echo(f"Running incremental ingest (days_back={days_back})")

        with next(get_session()) as session:
            results = ingest_incremental(session, days_back=days_back)

        # Display results
        typer.echo("\nResults:")
        for sym, count in results.items():
            status = "✓" if count > 0 else "○"
            typer.echo(f"  {status} {sym:>10}: {count:>5} bars")

        total = sum(results.values())
        typer.echo(f"\nTotal: {total} bars ingested")

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        logger.exception("Incremental ingest failed")
        raise typer.Exit(code=1)


@app.command()
def daemon_once(
    initial_cash: float = typer.Option(100000.0, "--initial-cash", help="Initial cash balance"),
    risk_pct: float = typer.Option(1.0, "--risk-pct", help="Risk per trade (%)"),
    ensemble: bool = typer.Option(True, "--ensemble/--no-ensemble", help="Use ensemble mode (default: True)"),
    no_trade: bool = typer.Option(False, "--no-trade", help="Disable paper trading"),
):
    """
    Run a single daemon cycle (P3 feature).

    Full pipeline: ingest → anomaly → patron → paper trade → snapshot

    Perfect for cron jobs: runs once and exits.

    Examples:
        otonom-trader daemon-once                        # Default: ensemble + paper trading
        otonom-trader daemon-once --risk-pct 2.0         # Higher risk
        otonom-trader daemon-once --no-trade             # Analysis only
        otonom-trader daemon-once --no-ensemble          # Disable ensemble

    Cron example (every 15 minutes):
        */15 * * * * cd /path/to/project && otonom-trader daemon-once
    """
    try:
        from .daemon import run_daemon_cycle, DaemonConfig, get_or_create_paper_trader

        typer.echo("Starting daemon cycle...")
        typer.echo(f"Initial cash: ${initial_cash:,.2f}")
        typer.echo(f"Risk per trade: {risk_pct}%")
        typer.echo(f"Ensemble mode: {'ON' if ensemble else 'OFF'}")
        typer.echo(f"Paper trading: {'OFF' if no_trade else 'ON'}")
        typer.echo("")

        # Configure daemon
        config = DaemonConfig(
            use_ensemble=ensemble,
            paper_trade_enabled=not no_trade,
            paper_trade_risk_pct=risk_pct,
            initial_cash=initial_cash,
        )

        with next(get_session()) as session:
            # Get or create paper trader
            paper_trader = None
            if not no_trade:
                paper_trader = get_or_create_paper_trader(
                    session,
                    initial_cash,
                    config.price_interval,
                )

            # Run daemon cycle
            run = run_daemon_cycle(session, config, paper_trader)

            if run.status == "SUCCESS":
                typer.echo("\n✓ Daemon cycle completed successfully")
            else:
                typer.echo(f"\n✗ Daemon cycle failed: {run.error_message}", err=True)
                raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        logger.exception("Daemon failed")
        raise typer.Exit(code=1)


@app.command()
def daemon_loop(
    interval: int = typer.Option(900, "--interval", help="Interval between cycles (seconds)"),
    initial_cash: float = typer.Option(100000.0, "--initial-cash", help="Initial cash balance"),
    risk_pct: float = typer.Option(1.0, "--risk-pct", help="Risk per trade (%)"),
    ensemble: bool = typer.Option(True, "--ensemble/--no-ensemble", help="Use ensemble mode"),
    no_trade: bool = typer.Option(False, "--no-trade", help="Disable paper trading"),
):
    """
    Run daemon in continuous loop mode (P3 feature).

    Runs indefinitely with specified interval between cycles.
    Use Ctrl+C to stop.

    Examples:
        otonom-trader daemon-loop                        # Every 15 minutes (900s)
        otonom-trader daemon-loop --interval 3600        # Every hour
        otonom-trader daemon-loop --interval 300         # Every 5 minutes
        otonom-trader daemon-loop --no-trade             # Analysis only

    Note: For production, prefer cron + daemon-once for better reliability.
    """
    import time as time_module

    try:
        from .daemon import run_daemon_cycle, DaemonConfig, get_or_create_paper_trader

        typer.echo("=" * 60)
        typer.echo("Starting daemon in LOOP mode")
        typer.echo(f"Interval: {interval}s ({interval//60} minutes)")
        typer.echo(f"Initial cash: ${initial_cash:,.2f}")
        typer.echo(f"Risk per trade: {risk_pct}%")
        typer.echo(f"Ensemble mode: {'ON' if ensemble else 'OFF'}")
        typer.echo(f"Paper trading: {'OFF' if no_trade else 'ON'}")
        typer.echo("\nPress Ctrl+C to stop")
        typer.echo("=" * 60)
        typer.echo("")

        # Configure daemon
        config = DaemonConfig(
            use_ensemble=ensemble,
            paper_trade_enabled=not no_trade,
            paper_trade_risk_pct=risk_pct,
            initial_cash=initial_cash,
        )

        cycle_count = 0

        while True:
            cycle_count += 1
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Cycle #{cycle_count} starting...")
            typer.echo(f"{'='*60}")

            try:
                with next(get_session()) as session:
                    # Get or create paper trader
                    paper_trader = None
                    if not no_trade:
                        paper_trader = get_or_create_paper_trader(
                            session,
                            initial_cash,
                            config.price_interval,
                        )

                    # Run cycle
                    run = run_daemon_cycle(session, config, paper_trader)

                    if run.status == "SUCCESS":
                        typer.echo(f"\n✓ Cycle #{cycle_count} completed successfully")
                    else:
                        typer.echo(f"\n✗ Cycle #{cycle_count} failed: {run.error_message}", err=True)

            except KeyboardInterrupt:
                typer.echo("\n\nReceived interrupt signal, stopping daemon...")
                break
            except Exception as e:
                typer.echo(f"\n✗ Cycle #{cycle_count} error: {e}", err=True)
                logger.exception(f"Cycle #{cycle_count} failed")

            # Wait for next cycle
            typer.echo(f"\nWaiting {interval}s until next cycle...")
            try:
                time_module.sleep(interval)
            except KeyboardInterrupt:
                typer.echo("\n\nReceived interrupt signal, stopping daemon...")
                break

        typer.echo(f"\nDaemon stopped after {cycle_count} cycles")

    except KeyboardInterrupt:
        typer.echo("\n\nDaemon stopped by user")
    except Exception as e:
        typer.echo(f"✗ Fatal error: {e}", err=True)
        logger.exception("Daemon loop failed")
        raise typer.Exit(code=1)


@app.command()
def show_paper_trades(
    limit: int = typer.Option(20, "--limit", help="Maximum number to display"),
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
):
    """
    Show paper trading execution log (P3 feature).

    Examples:
        otonom-trader show-paper-trades
        otonom-trader show-paper-trades --limit 50
        otonom-trader show-paper-trades --symbol BTC-USD
    """
    try:
        from .data.schema import PaperTrade

        with next(get_session()) as session:
            query = session.query(PaperTrade, Symbol.symbol).join(Symbol)

            if symbol:
                query = query.filter(Symbol.symbol == symbol)

            query = query.order_by(PaperTrade.timestamp.desc()).limit(limit)
            results = query.all()

        if not results:
            typer.echo("No paper trades found")
            return

        # Display
        typer.echo("Timestamp           | Symbol     | Action | Quantity  | Price     | Value      | Portfolio")
        typer.echo("-" * 100)

        for t, sym in results:
            typer.echo(
                f"{t.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {sym:>10} | {t.action:>6} | "
                f"{t.quantity:>9.4f} | ${t.price:>8.2f} | ${t.value:>9.2f} | ${t.portfolio_value:>12,.2f}"
            )

        typer.echo(f"\nShowing {len(results)} paper trades")

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        logger.exception("Show paper trades failed")
        raise typer.Exit(code=1)


@app.command()
def daemon_status(
    limit: int = typer.Option(10, "--limit", help="Maximum number to display"),
):
    """
    Show daemon execution history (P3 feature).

    Examples:
        otonom-trader daemon-status
        otonom-trader daemon-status --limit 20
    """
    try:
        from .data.schema import DaemonRun

        with next(get_session()) as session:
            runs = (
                session.query(DaemonRun)
                .order_by(DaemonRun.timestamp.desc())
                .limit(limit)
                .all()
            )

        if not runs:
            typer.echo("No daemon runs found")
            return

        # Display
        typer.echo("Timestamp           | Status  | Bars | Anomalies | Decisions | Trades | Duration | Portfolio")
        typer.echo("-" * 110)

        for r in runs:
            portfolio_str = f"${r.portfolio_value:>10,.2f}" if r.portfolio_value else "N/A".ljust(13)
            duration_str = f"{r.duration_seconds:.1f}s" if r.duration_seconds else "N/A"

            typer.echo(
                f"{r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {r.status:>7} | "
                f"{r.bars_ingested:>4} | {r.anomalies_detected:>9} | "
                f"{r.decisions_made:>9} | {r.trades_executed:>6} | "
                f"{duration_str:>8} | {portfolio_str}"
            )

        typer.echo(f"\nShowing {len(runs)} daemon runs")

    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        logger.exception("Daemon status failed")
        raise typer.Exit(code=1)


@app.command()
def status():
    """
    Show database status and health check (P3 enhanced).

    Displays:
    - Database statistics (total counts)
    - Recent activity (last 24 hours)
    - Portfolio performance
    - Daemon health check
    """
    try:
        from datetime import timedelta
        from .data.schema import (
            Regime as RegimeORM,
            DataHealthIndex as DsiORM,
            Hypothesis,
            HypothesisResult,
            PaperTrade,
            DaemonRun,
            PortfolioSnapshot,
            DailyBar,
        )

        with next(get_session()) as session:
            # ===== DATABASE STATISTICS =====
            typer.echo("=" * 60)
            typer.echo("DATABASE STATUS")
            typer.echo("=" * 60)

            symbol_count = session.query(Symbol).count()
            bar_count = session.query(DailyBar).count()
            anomaly_count = session.query(AnomalyORM).count()
            decision_count = session.query(DecisionORM).count()
            regime_count = session.query(RegimeORM).count()
            dsi_count = session.query(DsiORM).count()
            hypothesis_count = session.query(Hypothesis).count()
            backtest_count = session.query(HypothesisResult).count()
            paper_trades_count = session.query(PaperTrade).count()
            daemon_runs_count = session.query(DaemonRun).count()
            snapshots_count = session.query(PortfolioSnapshot).count()

            typer.echo(f"Symbols:            {symbol_count:>8}")
            typer.echo(f"Bars:               {bar_count:>8}")
            typer.echo(f"Anomalies:          {anomaly_count:>8}")
            typer.echo(f"Decisions:          {decision_count:>8}")
            typer.echo(f"Regimes:            {regime_count:>8}  [P1]")
            typer.echo(f"DSI:                {dsi_count:>8}  [P1]")
            typer.echo(f"Hypotheses:         {hypothesis_count:>8}  [P1]")
            typer.echo(f"Backtests:          {backtest_count:>8}  [P1]")
            typer.echo(f"Paper Trades:       {paper_trades_count:>8}  [P3]")
            typer.echo(f"Daemon Runs:        {daemon_runs_count:>8}  [P3]")
            typer.echo(f"Portfolio Snapshots:{snapshots_count:>8}  [P3]")

            # ===== 24-HOUR ACTIVITY =====
            typer.echo("\n" + "=" * 60)
            typer.echo("LAST 24 HOURS")
            typer.echo("=" * 60)

            now = utc_now()
            yesterday = now - timedelta(hours=24)

            # Bars ingested
            recent_bars = (
                session.query(DailyBar)
                .filter(DailyBar.date >= yesterday.date())
                .count()
            )

            # Anomalies detected
            recent_anomalies = (
                session.query(AnomalyORM)
                .filter(AnomalyORM.created_at >= yesterday)
                .count()
            )

            # Decisions made
            recent_decisions = (
                session.query(DecisionORM)
                .filter(DecisionORM.created_at >= yesterday)
                .count()
            )

            # Daemon runs
            recent_runs = (
                session.query(DaemonRun)
                .filter(DaemonRun.timestamp >= yesterday)
                .count()
            )

            # Successful runs
            successful_runs = (
                session.query(DaemonRun)
                .filter(DaemonRun.timestamp >= yesterday, DaemonRun.status == "SUCCESS")
                .count()
            )

            # Paper trades
            recent_trades = (
                session.query(PaperTrade)
                .filter(PaperTrade.timestamp >= yesterday)
                .count()
            )

            health_status = "✓ HEALTHY" if successful_runs >= recent_runs * 0.9 else "⚠ WARNING"

            typer.echo(f"Bars ingested:      {recent_bars:>8}")
            typer.echo(f"Anomalies detected: {recent_anomalies:>8}")
            typer.echo(f"Decisions made:     {recent_decisions:>8}")
            typer.echo(f"Daemon runs:        {recent_runs:>8}  ({successful_runs} successful)")
            typer.echo(f"Paper trades:       {recent_trades:>8}")
            typer.echo(f"\nHealth status:      {health_status}")

            # ===== PORTFOLIO PERFORMANCE =====
            if snapshots_count > 0:
                typer.echo("\n" + "=" * 60)
                typer.echo("PORTFOLIO PERFORMANCE")
                typer.echo("=" * 60)

                # Latest snapshot
                latest_snapshot = (
                    session.query(PortfolioSnapshot)
                    .order_by(PortfolioSnapshot.timestamp.desc())
                    .first()
                )

                if latest_snapshot:
                    typer.echo(f"Latest update:      {latest_snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    typer.echo(f"Total equity:       ${latest_snapshot.equity:>12,.2f}")
                    typer.echo(f"Cash:               ${latest_snapshot.cash:>12,.2f}")
                    typer.echo(f"Positions value:    ${latest_snapshot.positions_value:>12,.2f}")
                    typer.echo(f"Number of positions:{latest_snapshot.num_positions:>8}")

                    if latest_snapshot.max_drawdown is not None:
                        dd_str = f"{latest_snapshot.max_drawdown:.2%}"
                        dd_status = "✓" if latest_snapshot.max_drawdown < 0.1 else "⚠"
                        typer.echo(f"Max drawdown:       {dd_status} {dd_str:>10}")

                    # Calculate performance over last N snapshots
                    if snapshots_count >= 2:
                        first_snapshot = (
                            session.query(PortfolioSnapshot)
                            .order_by(PortfolioSnapshot.timestamp.asc())
                            .first()
                        )

                        total_return = (
                            (latest_snapshot.equity - first_snapshot.equity)
                            / first_snapshot.equity
                        )
                        total_return_pct = total_return * 100

                        typer.echo(f"Total return:       {total_return_pct:>+12.2f}%")

                        # Days tracked
                        days_tracked = (latest_snapshot.timestamp - first_snapshot.timestamp).days
                        if days_tracked > 0:
                            typer.echo(f"Days tracked:       {days_tracked:>8}")

    except Exception as e:
        typer.echo(f"✗ Error getting status: {e}", err=True)
        logger.exception("Status check failed")
        raise typer.Exit(code=1)



if __name__ == "__main__":
    app()
