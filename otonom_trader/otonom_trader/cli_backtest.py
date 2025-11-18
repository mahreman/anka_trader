"""
CLI commands for backtest runner.

Provides simple, unified commands for running backtests:
- backtest run: Run a full backtest from strategy YAML
- backtest report: View existing backtest report

Usage:
    python -m otonom_trader.cli backtest run \
        --strategy strategies/baseline_v1.0.yaml \
        --start 2018-01-01 \
        --end 2024-01-01 \
        --output reports/baseline_2018_2024.json

    python -m otonom_trader.cli backtest report \
        --file reports/baseline_2018_2024.json
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .strategy.config import load_strategy_config
from .research.backtest_runner import run_backtest_for_strategy, BacktestReport

logger = logging.getLogger(__name__)
console = Console()

# Create Typer app
app = typer.Typer(
    name="backtest",
    help="Unified backtest runner with comprehensive reporting"
)


def _parse_date(date_str: str) -> date:
    """Parse date string (YYYY-MM-DD) to date object."""
    try:
        return date.fromisoformat(date_str)
    except ValueError as e:
        console.print(f"[red]Invalid date format: {date_str}. Expected YYYY-MM-DD[/red]")
        raise typer.Exit(1) from e


def _load_report(file_path: str) -> BacktestReport:
    """Load backtest report from JSON file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Report file not found: {file_path}[/red]")
        raise typer.Exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct BacktestReport from dict
    # For simplicity, we'll just use the dict directly for display
    return data


@app.command("run")
def run_backtest(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Path to strategy YAML file"),
    start: str = typer.Option(..., "--start", help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", help="Backtest end date (YYYY-MM-DD)"),
    output: str = typer.Option("reports/backtest_report.json", "--output", "-o", help="Output JSON file path"),
    universe: str = typer.Option(None, "--universe", "-u", help="Override universe (comma-separated symbols)"),
    skip_backtest: bool = typer.Option(False, "--skip-backtest", help="Skip running backtest, just collect existing results"),
):
    """
    Run a backtest for a strategy.

    This command:
    1. Loads strategy configuration from YAML
    2. Runs backtest for specified date range
    3. Generates comprehensive report
    4. Saves report to JSON file

    Example:
        python -m otonom_trader.cli backtest run \\
            --strategy strategies/baseline_v1.0.yaml \\
            --start 2018-01-01 \\
            --end 2024-01-01 \\
            --output reports/baseline_2018_2024.json
    """
    console.print(f"\n[bold cyan]Running Backtest[/bold cyan]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Output: {output}\n")

    # Load strategy
    try:
        cfg = load_strategy_config(strategy)
        console.print(f"✓ Loaded strategy: {cfg.name} v{cfg.version}")
    except Exception as e:
        console.print(f"[red]Failed to load strategy: {e}[/red]")
        raise typer.Exit(1) from e

    # Parse dates
    start_date = _parse_date(start)
    end_date = _parse_date(end)

    if start_date >= end_date:
        console.print("[red]Start date must be before end date[/red]")
        raise typer.Exit(1)

    # Parse universe override
    universe_override = None
    if universe:
        universe_override = [s.strip() for s in universe.split(",")]
        console.print(f"Universe override: {universe_override}")

    # Run backtest
    try:
        console.print("\n[yellow]Running backtest engine...[/yellow]")
        report = run_backtest_for_strategy(
            strategy_cfg=cfg,
            start_date=start_date,
            end_date=end_date,
            universe_override=universe_override,
            run_backtest=not skip_backtest,
        )
        console.print("[green]✓ Backtest complete![/green]")
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        logger.exception("Backtest error")
        raise typer.Exit(1) from e

    # Save report
    try:
        report.save_json(output)
        console.print(f"[green]✓ Report saved to: {output}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save report: {e}[/red]")
        raise typer.Exit(1) from e

    # Display summary
    _display_summary(report)


@app.command("report")
def view_report(
    file: str = typer.Argument(..., help="Path to backtest report JSON file"),
    trades: bool = typer.Option(False, "--trades", "-t", help="Show detailed trade list"),
    regime: bool = typer.Option(False, "--regime", "-r", help="Show regime breakdown"),
):
    """
    View a backtest report.

    This command loads and displays a saved backtest report.

    Example:
        python -m otonom_trader.cli backtest report reports/baseline_2018_2024.json

        python -m otonom_trader.cli backtest report reports/baseline_2018_2024.json --trades --regime
    """
    console.print(f"\n[bold cyan]Backtest Report[/bold cyan]")
    console.print(f"File: {file}\n")

    # Load report
    try:
        report_data = _load_report(file)
    except Exception as e:
        console.print(f"[red]Failed to load report: {e}[/red]")
        raise typer.Exit(1) from e

    # Display summary
    _display_summary_from_dict(report_data)

    # Display trades if requested
    if trades and report_data.get("trades"):
        _display_trades(report_data["trades"])

    # Display regime breakdown if requested
    if regime and report_data.get("regime_breakdown"):
        _display_regime_breakdown(report_data["regime_breakdown"])


def _display_summary(report: BacktestReport):
    """Display backtest summary."""
    console.print("\n[bold]Backtest Summary[/bold]")
    console.print("=" * 60)

    # Strategy info
    console.print(f"[cyan]Strategy:[/cyan] {report.strategy_name} v{report.strategy_version}")
    console.print(f"[cyan]Period:[/cyan] {report.start_date} to {report.end_date}")
    console.print(f"[cyan]Universe:[/cyan] {', '.join(report.universe)}")
    console.print(f"[cyan]Initial Capital:[/cyan] ${report.initial_capital:,.2f}")
    console.print(f"[cyan]Final Equity:[/cyan] ${report.final_equity:,.2f}")

    console.print("\n[bold]Performance Metrics[/bold]")
    console.print("-" * 60)

    metrics = report.metrics

    # Color-code metrics
    cagr_color = "green" if metrics.cagr > 0 else "red"
    sharpe_color = "green" if metrics.sharpe > 1.0 else "yellow" if metrics.sharpe > 0 else "red"

    console.print(f"[cyan]CAGR:[/cyan] [{cagr_color}]{metrics.cagr:+.2f}%[/{cagr_color}]")
    console.print(f"[cyan]Sharpe Ratio:[/cyan] [{sharpe_color}]{metrics.sharpe:.2f}[/{sharpe_color}]")
    console.print(f"[cyan]Sortino Ratio:[/cyan] {metrics.sortino:.2f}")
    console.print(f"[cyan]Max Drawdown:[/cyan] [red]{metrics.max_drawdown:.2f}%[/red]")
    console.print(f"[cyan]Win Rate:[/cyan] {metrics.win_rate:.1%}")
    console.print(f"[cyan]Avg R-Multiple:[/cyan] {metrics.avg_r_multiple:.2f}")
    console.print(f"[cyan]Profit Factor:[/cyan] {metrics.profit_factor:.2f}")
    console.print(f"[cyan]Total Trades:[/cyan] {metrics.total_trades}")
    console.print(f"[cyan]Total P&L:[/cyan] ${metrics.total_pnl:,.2f}")

    console.print("=" * 60)


def _display_summary_from_dict(report_data: dict):
    """Display backtest summary from dictionary."""
    console.print("\n[bold]Backtest Summary[/bold]")
    console.print("=" * 60)

    # Strategy info
    console.print(f"[cyan]Strategy:[/cyan] {report_data['strategy_name']} v{report_data['strategy_version']}")
    console.print(f"[cyan]Period:[/cyan] {report_data['start_date']} to {report_data['end_date']}")
    console.print(f"[cyan]Universe:[/cyan] {', '.join(report_data['universe'])}")
    console.print(f"[cyan]Initial Capital:[/cyan] ${report_data['initial_capital']:,.2f}")
    console.print(f"[cyan]Final Equity:[/cyan] ${report_data['final_equity']:,.2f}")

    console.print("\n[bold]Performance Metrics[/bold]")
    console.print("-" * 60)

    metrics = report_data["metrics"]

    # Color-code metrics
    cagr_color = "green" if metrics["cagr"] > 0 else "red"
    sharpe_color = "green" if metrics["sharpe"] > 1.0 else "yellow" if metrics["sharpe"] > 0 else "red"

    console.print(f"[cyan]CAGR:[/cyan] [{cagr_color}]{metrics['cagr']:+.2f}%[/{cagr_color}]")
    console.print(f"[cyan]Sharpe Ratio:[/cyan] [{sharpe_color}]{metrics['sharpe']:.2f}[/{sharpe_color}]")
    console.print(f"[cyan]Sortino Ratio:[/cyan] {metrics['sortino']:.2f}")
    console.print(f"[cyan]Max Drawdown:[/cyan] [red]{metrics['max_drawdown']:.2f}%[/red]")
    console.print(f"[cyan]Win Rate:[/cyan] {metrics['win_rate']:.1%}")
    console.print(f"[cyan]Avg R-Multiple:[/cyan] {metrics['avg_r_multiple']:.2f}")
    console.print(f"[cyan]Profit Factor:[/cyan] {metrics['profit_factor']:.2f}")
    console.print(f"[cyan]Total Trades:[/cyan] {metrics['total_trades']}")
    console.print(f"[cyan]Total P&L:[/cyan] ${metrics['total_pnl']:,.2f}")

    console.print("=" * 60)


def _display_trades(trades: list):
    """Display trade list in a table."""
    if not trades:
        console.print("\n[yellow]No trades to display[/yellow]")
        return

    console.print(f"\n[bold]Trade List[/bold] ({len(trades)} trades)")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Entry", style="dim")
    table.add_column("Exit", style="dim")
    table.add_column("Dir", justify="center")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("R-Mult", justify="right")
    table.add_column("Days", justify="right")

    for trade in trades[:50]:  # Limit to first 50 trades
        pnl_color = "green" if trade["pnl"] > 0 else "red"
        table.add_row(
            trade["symbol"],
            trade["entry_date"],
            trade["exit_date"],
            trade["direction"],
            f"[{pnl_color}]${trade['pnl']:,.2f}[/{pnl_color}]",
            f"[{pnl_color}]{trade['pnl_pct']:+.2f}%[/{pnl_color}]",
            f"{trade['r_multiple']:.2f}",
            str(trade["holding_days"]),
        )

    console.print(table)

    if len(trades) > 50:
        console.print(f"\n[dim]Showing first 50 of {len(trades)} trades[/dim]")


def _display_regime_breakdown(regime_breakdown: list):
    """Display regime breakdown in a table."""
    if not regime_breakdown:
        console.print("\n[yellow]No regime breakdown available[/yellow]")
        return

    console.print(f"\n[bold]Regime Breakdown[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Regime", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg R-Mult", justify="right")

    for regime in regime_breakdown:
        sharpe_color = "green" if regime["sharpe"] > 1.0 else "yellow" if regime["sharpe"] > 0 else "red"
        table.add_row(
            regime["regime_name"],
            str(regime["trades"]),
            f"[{sharpe_color}]{regime['sharpe']:.2f}[/{sharpe_color}]",
            f"{regime['win_rate']:.1%}",
            f"{regime['avg_r_multiple']:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
