"""
Research Engine CLI commands.

Provides command-line interface for running comprehensive backtests
with different test types (smoke/full/scenario).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .data import get_session
from .eval.research_engine import (
    BacktestType,
    BacktestPeriod,
    ResearchEngine,
    run_research_backtest,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="research",
    help="Run comprehensive research backtests (smoke/full/scenario)",
)


def _parse_date(s: str | None) -> date | None:
    """Parse date string to date object."""
    if s is None:
        return None
    return date.fromisoformat(s)


@app.command("smoke")
def smoke_test(
    strategy: str = typer.Argument(..., help="Strategy name or path (e.g., baseline_v1 or strategies/baseline_v1.yaml)"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated symbols (overrides config)"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    output: str = typer.Option(None, "--output", help="Output directory for reports (default: reports/)"),
):
    """
    Run quick smoke test (30 days, 1-2 symbols).

    Smoke tests are used to verify strategy correctness without running
    a full historical backtest.

    Example:
        otonom-trader research smoke baseline_v1
        otonom-trader research smoke strategies/baseline_v1.yaml --symbols BTC-USD
    """
    # Determine strategy path
    if strategy.endswith(".yaml"):
        strategy_path = strategy
    else:
        strategy_path = f"strategies/{strategy}.yaml"

    console.print(Panel(
        f"[bold cyan]Smoke Test[/bold cyan]\n"
        f"Strategy: {strategy_path}\n"
        f"Duration: Last 30 days\n"
        f"Symbols: {'Auto (1-2)' if not symbols else symbols}",
        title="Research Engine",
        border_style="cyan"
    ))

    # Parse symbols
    symbol_list = symbols.split(",") if symbols else None

    # Run backtest
    with get_session(db) as session:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Running smoke test...", total=None)

                reports = run_research_backtest(
                    session=session,
                    strategy_path=strategy_path,
                    backtest_type=BacktestType.SMOKE,
                    symbols=symbol_list,
                )

            if not reports:
                console.print("[red]No backtest results generated.[/red]")
                raise typer.Exit(code=1)

            # Display results
            for report in reports:
                console.print("\n" + report.summary())

            # Save reports if output specified
            if output:
                _save_reports(reports, output, "smoke")

            console.print("\n[bold green]✓ Smoke test completed successfully![/bold green]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Smoke test failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("full")
def full_backtest(
    strategy: str = typer.Argument(..., help="Strategy name or path"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated symbols (overrides config)"),
    start_date: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD, overrides config)"),
    end_date: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD, overrides config)"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    output: str = typer.Option("reports", "--output", help="Output directory for reports"),
):
    """
    Run full historical backtest with train/val/test splits.

    Full backtests run across entire history with 60% train, 20% validation,
    20% test splits to evaluate strategy performance and overfitting.

    Example:
        otonom-trader research full baseline_v1
        otonom-trader research full baseline_v1 --symbols BTC-USD,ETH-USD --start 2018-01-01 --end 2024-12-31
    """
    # Determine strategy path
    if strategy.endswith(".yaml"):
        strategy_path = strategy
    else:
        strategy_path = f"strategies/{strategy}.yaml"

    console.print(Panel(
        f"[bold cyan]Full Historical Backtest[/bold cyan]\n"
        f"Strategy: {strategy_path}\n"
        f"Splits: Train (60%) / Validation (20%) / Test (20%)\n"
        f"Period: {'Custom' if start_date else 'From config'}\n"
        f"Symbols: {symbols if symbols else 'From config'}",
        title="Research Engine",
        border_style="cyan"
    ))

    # Parse symbols
    symbol_list = symbols.split(",") if symbols else None

    # Parse dates
    custom_period = None
    if start_date and end_date:
        custom_period = BacktestPeriod(
            name="Custom Period",
            start_date=_parse_date(start_date),
            end_date=_parse_date(end_date),
            description=f"Custom period from {start_date} to {end_date}"
        )

    # Run backtest
    with get_session(db) as session:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Running full backtest...", total=None)

                reports = run_research_backtest(
                    session=session,
                    strategy_path=strategy_path,
                    backtest_type=BacktestType.FULL,
                    symbols=symbol_list,
                    custom_period=custom_period,
                )

            if not reports:
                console.print("[red]No backtest results generated.[/red]")
                raise typer.Exit(code=1)

            # Display results
            for report in reports:
                console.print("\n" + report.summary())

            # Check for overfitting across splits
            if len(reports) == 3:  # Train, Val, Test
                _check_overfitting(reports)

            # Save reports
            _save_reports(reports, output, "full")

            console.print(f"\n[bold green]✓ Full backtest completed![/bold green]")
            console.print(f"Reports saved to: {output}/")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Full backtest failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("scenario")
def scenario_backtest(
    strategy: str = typer.Argument(..., help="Strategy name or path"),
    scenario: str = typer.Option(
        None,
        "--scenario",
        help="Scenario name (covid_crash, crypto_bear_2022, high_inflation, bull_run_2021)"
    ),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    start_date: str = typer.Option(None, "--start", help="Custom scenario start (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", help="Custom scenario end (YYYY-MM-DD)"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    output: str = typer.Option("reports", "--output", help="Output directory for reports"),
):
    """
    Run scenario-based backtest (specific market periods).

    Scenario backtests evaluate strategy performance during specific
    market conditions (crashes, bear markets, high volatility, etc.).

    Available scenarios:
      - covid_crash: Feb-Apr 2020 market crash
      - crypto_bear_2022: 2022 crypto bear market
      - high_inflation: 2021-2023 high inflation period
      - bull_run_2021: Q4 2020 - Q4 2021 bull market

    Example:
        otonom-trader research scenario baseline_v1 --scenario covid_crash
        otonom-trader research scenario baseline_v1 --start 2020-03-01 --end 2020-04-30
    """
    # Determine strategy path
    if strategy.endswith(".yaml"):
        strategy_path = strategy
    else:
        strategy_path = f"strategies/{strategy}.yaml"

    # Determine scenario
    scenario_desc = scenario if scenario else "All predefined scenarios"
    if start_date and end_date:
        scenario_desc = f"Custom ({start_date} to {end_date})"

    console.print(Panel(
        f"[bold cyan]Scenario Backtest[/bold cyan]\n"
        f"Strategy: {strategy_path}\n"
        f"Scenario: {scenario_desc}\n"
        f"Symbols: {symbols if symbols else 'From config'}",
        title="Research Engine",
        border_style="cyan"
    ))

    # Parse symbols
    symbol_list = symbols.split(",") if symbols else None

    # Parse custom period
    custom_period = None
    if start_date and end_date:
        custom_period = BacktestPeriod(
            name="Custom Scenario",
            start_date=_parse_date(start_date),
            end_date=_parse_date(end_date),
            description=f"Custom scenario from {start_date} to {end_date}"
        )

    # Run backtest
    with get_session(db) as session:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Running scenario backtest...", total=None)

                reports = run_research_backtest(
                    session=session,
                    strategy_path=strategy_path,
                    backtest_type=BacktestType.SCENARIO,
                    symbols=symbol_list,
                    custom_period=custom_period,
                    scenario_name=scenario,
                )

            if not reports:
                console.print("[red]No backtest results generated.[/red]")
                raise typer.Exit(code=1)

            # Display results
            for report in reports:
                console.print("\n" + report.summary())

            # Save reports
            _save_reports(reports, output, "scenario")

            console.print(f"\n[bold green]✓ Scenario backtest completed![/bold green]")
            console.print(f"Reports saved to: {output}/")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Scenario backtest failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("list-scenarios")
def list_scenarios():
    """
    List available predefined scenarios.

    Example:
        otonom-trader research list-scenarios
    """
    console.print("\n[bold]Available Scenarios:[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario Name", style="cyan")
    table.add_column("Period")
    table.add_column("Description")

    for name, period in ResearchEngine.SCENARIOS.items():
        table.add_row(
            name,
            f"{period.start_date} to {period.end_date}",
            period.description or "N/A"
        )

    console.print(table)
    console.print()


def _check_overfitting(reports: list):
    """Check for overfitting across train/val/test splits."""
    if len(reports) != 3:
        return

    train_report = next((r for r in reports if r.period.name == "Train"), None)
    val_report = next((r for r in reports if r.period.name == "Validation"), None)
    test_report = next((r for r in reports if r.period.name == "Test"), None)

    if not (train_report and val_report and test_report):
        return

    console.print("\n[bold]Overfitting Analysis:[/bold]")

    # Compare Sharpe ratios
    train_sharpe = train_report.metrics.get("sharpe_ratio", 0)
    val_sharpe = val_report.metrics.get("sharpe_ratio", 0)
    test_sharpe = test_report.metrics.get("sharpe_ratio", 0)

    if train_sharpe > 0 and test_sharpe > 0:
        ratio = train_sharpe / test_sharpe
        if ratio > 1.5:
            console.print(
                f"  [yellow]⚠ Train/Test Sharpe ratio = {ratio:.2f} "
                f"(potential overfitting)[/yellow]"
            )
        elif ratio > 1.2:
            console.print(
                f"  [dim yellow]⚠ Train/Test Sharpe ratio = {ratio:.2f} "
                f"(mild overfitting)[/dim yellow]"
            )
        else:
            console.print(
                f"  [green]✓ Train/Test Sharpe ratio = {ratio:.2f} "
                f"(good generalization)[/green]"
            )

    # Compare CAGR
    train_cagr = train_report.metrics.get("cagr", 0)
    test_cagr = test_report.metrics.get("cagr", 0)

    if train_cagr > 0 and test_cagr > 0:
        ratio = train_cagr / test_cagr
        if ratio > 2.0:
            console.print(
                f"  [yellow]⚠ Train/Test CAGR ratio = {ratio:.2f} "
                f"(significant performance degradation)[/yellow]"
            )


def _save_reports(reports: list, output_dir: str, backtest_type: str):
    """Save research reports to files."""
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for report in reports:
        # Save JSON
        period_name = report.period.name.lower().replace(" ", "_")
        json_path = output_path / f"{backtest_type}_{period_name}_{timestamp}.json"

        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved JSON report: {json_path}")

        # Save CSV (trades)
        if report.trades:
            csv_path = output_path / f"{backtest_type}_{period_name}_{timestamp}_trades.csv"

            import pandas as pd
            df = pd.DataFrame([t.to_dict() for t in report.trades])
            df.to_csv(csv_path, index=False)

            logger.info(f"Saved trades CSV: {csv_path}")

        # Save equity curve CSV
        if report.equity_curve and report.equity_dates:
            equity_csv_path = output_path / f"{backtest_type}_{period_name}_{timestamp}_equity.csv"

            import pandas as pd
            df = pd.DataFrame({
                "date": report.equity_dates,
                "equity": report.equity_curve,
            })
            df.to_csv(equity_csv_path, index=False)

            logger.info(f"Saved equity curve CSV: {equity_csv_path}")
