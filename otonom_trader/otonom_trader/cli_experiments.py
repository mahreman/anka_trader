"""
Experiment Engine CLI commands.

Provides command-line interface for running parameter optimization experiments.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .data import get_session
from .experiments import run_grid_search, run_random_search

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="experiments",
    help="Run parameter optimization experiments (grid search, random search)",
)


def _parse_date(s: str | None) -> date | None:
    """Parse date string to date object."""
    if s is None:
        return None
    return date.fromisoformat(s)


@app.command("grid-search")
def grid_search_cli(
    experiment_name: str = typer.Option(..., "--name", help="Experiment name"),
    strategy_path: str = typer.Option(
        "strategies/baseline_v1.yaml",
        "--strategy-path",
        help="Base strategy YAML path",
    ),
    grid_path: str = typer.Option(
        "grids/baseline_grid.yaml",
        "--grid-path",
        help="Parameter grid YAML path",
    ),
    train_start: str = typer.Option(
        None,
        "--train-start",
        help="Train start date (YYYY-MM-DD), overrides grid YAML",
    ),
    train_end: str = typer.Option(
        None,
        "--train-end",
        help="Train end date (YYYY-MM-DD), overrides grid YAML",
    ),
    test_start: str = typer.Option(
        None,
        "--test-start",
        help="Test start date (YYYY-MM-DD), overrides grid YAML",
    ),
    test_end: str = typer.Option(
        None,
        "--test-end",
        help="Test end date (YYYY-MM-DD), overrides grid YAML",
    ),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Run grid search parameter optimization.

    Example:
        otonom-trader experiments grid-search \\
            --name "baseline_v1_optimization" \\
            --strategy-path strategies/baseline_v1.yaml \\
            --grid-path grids/baseline_grid.yaml \\
            --train-start 2018-01-01 --train-end 2022-12-31 \\
            --test-start 2023-01-01 --test-end 2024-12-31
    """
    console.print(f"[bold cyan]Starting Grid Search:[/bold cyan] {experiment_name}")
    console.print(f"  Strategy: {strategy_path}")
    console.print(f"  Grid: {grid_path}")

    # Parse dates
    train_start_date = _parse_date(train_start)
    train_end_date = _parse_date(train_end)
    test_start_date = _parse_date(test_start)
    test_end_date = _parse_date(test_end)

    if train_start_date and train_end_date:
        console.print(f"  Train period: {train_start_date} to {train_end_date}")
    else:
        console.print("  Train period: [from grid YAML]")

    if test_start_date and test_end_date:
        console.print(f"  Test period: {test_start_date} to {test_end_date}")
    else:
        console.print("  Test period: [from grid YAML]")

    console.print()

    # Run grid search
    with get_session(db) as session:
        try:
            experiment = run_grid_search(
                session=session,
                experiment_name=experiment_name,
                strategy_path=Path(strategy_path),
                grid_path=Path(grid_path),
                train_start_override=train_start_date,
                train_end_override=train_end_date,
                test_start_override=test_start_date,
                test_end_override=test_end_date,
            )

            console.print(
                f"[bold green]✓ Grid search completed![/bold green] Experiment ID: {experiment.id}"
            )
            console.print(f"  Total runs: {len(experiment.runs)}")

            # Show top results
            if experiment.runs:
                successful_runs = [r for r in experiment.runs if r.status == "done"]
                if successful_runs:
                    console.print(f"  Successful runs: {len(successful_runs)}")

                    # Sort by test_sharpe (or train_sharpe if no test)
                    sorted_runs = sorted(
                        successful_runs,
                        key=lambda r: r.test_sharpe or r.train_sharpe or 0,
                        reverse=True,
                    )

                    console.print("\n[bold]Top 5 Results:[/bold]")
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Run", justify="right")
                    table.add_column("Train Sharpe", justify="right")
                    table.add_column("Test Sharpe", justify="right")
                    table.add_column("Test CAGR", justify="right")
                    table.add_column("Test MaxDD", justify="right")

                    for run in sorted_runs[:5]:
                        table.add_row(
                            str(run.run_index),
                            f"{run.train_sharpe:.2f}" if run.train_sharpe else "N/A",
                            f"{run.test_sharpe:.2f}" if run.test_sharpe else "N/A",
                            f"{run.test_cagr:.1f}%" if run.test_cagr else "N/A",
                            f"{run.test_max_dd:.1f}%" if run.test_max_dd else "N/A",
                        )

                    console.print(table)

                    # Overfitting warning
                    console.print("\n[bold]Overfitting Check:[/bold]")
                    for run in sorted_runs[:5]:
                        if run.test_sharpe and run.train_sharpe:
                            ratio = run.train_sharpe / run.test_sharpe
                            if ratio > 1.5:
                                console.print(
                                    f"  [yellow]⚠ Run {run.run_index}: "
                                    f"Train/Test Sharpe ratio = {ratio:.2f} "
                                    f"(potential overfitting)[/yellow]"
                                )
                else:
                    console.print("[red]  No successful runs![/red]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Grid search failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("random-search")
def random_search_cli(
    experiment_name: str = typer.Option(..., "--name", help="Experiment name"),
    strategy_path: str = typer.Option(
        "strategies/baseline_v1.yaml",
        "--strategy-path",
        help="Base strategy YAML path",
    ),
    grid_path: str = typer.Option(
        "grids/baseline_grid.yaml",
        "--grid-path",
        help="Parameter grid YAML path",
    ),
    train_start: str = typer.Option(
        None,
        "--train-start",
        help="Train start date (YYYY-MM-DD), overrides grid YAML",
    ),
    train_end: str = typer.Option(
        None,
        "--train-end",
        help="Train end date (YYYY-MM-DD), overrides grid YAML",
    ),
    test_start: str = typer.Option(
        None,
        "--test-start",
        help="Test start date (YYYY-MM-DD), overrides grid YAML",
    ),
    test_end: str = typer.Option(
        None,
        "--test-end",
        help="Test end date (YYYY-MM-DD), overrides grid YAML",
    ),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Run random search parameter optimization.

    Uses random sampling from parameter grid (see grid YAML for sample count).

    Example:
        otonom-trader experiments random-search \\
            --name "baseline_v1_random" \\
            --grid-path grids/baseline_grid.yaml \\
            --train-start 2018-01-01 --train-end 2022-12-31
    """
    console.print(f"[bold cyan]Starting Random Search:[/bold cyan] {experiment_name}")
    console.print(f"  Strategy: {strategy_path}")
    console.print(f"  Grid: {grid_path}")

    # Parse dates
    train_start_date = _parse_date(train_start)
    train_end_date = _parse_date(train_end)
    test_start_date = _parse_date(test_start)
    test_end_date = _parse_date(test_end)

    console.print()

    # Run random search
    with get_session(db) as session:
        try:
            experiment = run_random_search(
                session=session,
                experiment_name=experiment_name,
                strategy_path=Path(strategy_path),
                grid_path=Path(grid_path),
                train_start_override=train_start_date,
                train_end_override=train_end_date,
                test_start_override=test_start_date,
                test_end_override=test_end_date,
            )

            console.print(
                f"[bold green]✓ Random search completed![/bold green] Experiment ID: {experiment.id}"
            )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Random search failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("list")
def list_experiments(
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    limit: int = typer.Option(20, "--limit", help="Number of experiments to show"),
):
    """
    List recent experiments.

    Example:
        otonom-trader experiments list --limit 10
    """
    from .data import Experiment

    with get_session(db) as session:
        experiments = (
            session.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .all()
        )

        if not experiments:
            console.print("[yellow]No experiments found.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", justify="right")
        table.add_column("Name")
        table.add_column("Strategy")
        table.add_column("Runs", justify="right")
        table.add_column("Created")

        for exp in experiments:
            table.add_row(
                str(exp.id),
                exp.name,
                exp.base_strategy_name,
                str(len(exp.runs)),
                exp.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)


@app.command("show")
def show_experiment(
    experiment_id: int = typer.Argument(..., help="Experiment ID"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    top: int = typer.Option(10, "--top", help="Number of top results to show"),
    export: str = typer.Option(None, "--export", help="Export report to file (e.g., reports/exp_1.html)"),
):
    """
    Show detailed experiment results.

    Example:
        otonom-trader experiments show 1 --top 5
    """
    from .data import Experiment

    with get_session(db) as session:
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()

        if not experiment:
            console.print(f"[red]Experiment {experiment_id} not found.[/red]")
            raise typer.Exit(code=1)

        console.print(f"[bold]Experiment {experiment.id}:[/bold] {experiment.name}")
        console.print(f"  Description: {experiment.description or 'N/A'}")
        console.print(f"  Strategy: {experiment.base_strategy_name}")
        console.print(f"  Grid: {experiment.param_grid_path}")
        console.print(f"  Created: {experiment.created_at}")
        console.print(f"  Total runs: {len(experiment.runs)}")

        # Show run statistics
        successful_runs = [r for r in experiment.runs if r.status == "done"]
        failed_runs = [r for r in experiment.runs if r.status == "failed"]

        console.print(f"  Successful: {len(successful_runs)}")
        console.print(f"  Failed: {len(failed_runs)}")

        if not successful_runs:
            console.print("\n[yellow]No successful runs to display.[/yellow]")
            return

        # Sort by test_sharpe (or train_sharpe if no test)
        sorted_runs = sorted(
            successful_runs,
            key=lambda r: r.test_sharpe or r.train_sharpe or 0,
            reverse=True,
        )

        console.print(f"\n[bold]Top {top} Results:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Run", justify="right")
        table.add_column("Train Sharpe", justify="right")
        table.add_column("Test Sharpe", justify="right")
        table.add_column("Test CAGR", justify="right")
        table.add_column("Test MaxDD", justify="right")
        table.add_column("Test Trades", justify="right")

        for run in sorted_runs[:top]:
            table.add_row(
                str(run.run_index),
                f"{run.train_sharpe:.2f}" if run.train_sharpe else "N/A",
                f"{run.test_sharpe:.2f}" if run.test_sharpe else "N/A",
                f"{run.test_cagr:.1f}%" if run.test_cagr else "N/A",
                f"{run.test_max_dd:.1f}%" if run.test_max_dd else "N/A",
                str(run.test_total_trades) if run.test_total_trades else "N/A",
            )

        console.print(table)

        # Show best parameters
        best_run = sorted_runs[0]
        console.print(f"\n[bold]Best Run ({best_run.run_index}) Parameters:[/bold]")
        import json

        params = json.loads(best_run.param_values_json)
        for key, value in params.items():
            console.print(f"  {key}: {value}")

        # Export report if requested
        if export:
            from ..experiments import generate_experiment_report

            # Determine format from file extension
            if export.endswith(".html"):
                format = "html"
            elif export.endswith(".md"):
                format = "markdown"
            else:
                # Default to HTML
                format = "html"
                if not export.endswith(".html"):
                    export += ".html"

            console.print(f"\n[cyan]Generating {format.upper()} report...[/cyan]")

            try:
                report_path = generate_experiment_report(
                    session=session,
                    experiment_id=experiment_id,
                    output_path=export,
                    format=format,
                )
                console.print(f"[bold green]✓ Report saved to:[/bold green] {report_path}")
            except Exception as e:
                console.print(f"[red]Failed to generate report: {e}[/red]")
                logger.error(f"Report generation failed: {e}", exc_info=True)


@app.command("ablation")
def analyze_ablation_experiment(
    experiment_id: int = typer.Argument(..., help="Ablation experiment ID"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    export: str = typer.Option(None, "--export", help="Export to markdown/html file"),
):
    """
    Analyze ablation experiment results.

    Shows contribution of each analyst component.

    Example:
        otonom-trader experiments ablation 5
        otonom-trader experiments ablation 5 --export reports/ablation.md
    """
    from .experiments.analysis import analyze_ablation, generate_comparison_table

    with get_session(db) as session:
        try:
            console.print(f"[bold cyan]Analyzing ablation experiment {experiment_id}...[/bold cyan]\n")

            results = analyze_ablation(session, experiment_id)

            if not results:
                console.print("[yellow]No ablation results found.[/yellow]")
                return

            # Print results
            console.print("[bold]Ablation Analysis Results:[/bold]\n")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Configuration")
            table.add_column("Sharpe", justify="right")
            table.add_column("CAGR", justify="right")
            table.add_column("Max DD", justify="right")
            table.add_column("Win Rate", justify="right")
            table.add_column("Trades", justify="right")
            table.add_column("Contribution", justify="right")

            for r in results:
                contrib_str = f"{r.contribution:+.2f}" if r.contribution else "baseline"
                contrib_style = "green" if r.contribution and r.contribution > 0 else ("red" if r.contribution and r.contribution < 0 else "")

                table.add_row(
                    r.config_name,
                    f"{r.test_sharpe:.2f}",
                    f"{r.test_cagr:.1f}%",
                    f"{r.test_max_dd:.1f}%",
                    f"{r.test_win_rate:.1f}%",
                    str(r.total_trades),
                    f"[{contrib_style}]{contrib_str}[/{contrib_style}]" if contrib_style else contrib_str,
                )

            console.print(table)

            # Export if requested
            if export:
                output_format = "markdown" if export.endswith(".md") else "html"
                table_output = generate_comparison_table(results, output_format)

                with open(export, "w") as f:
                    f.write(table_output)

                console.print(f"\n[bold green]✓ Report exported to:[/bold green] {export}")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Ablation analysis failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("robustness")
def analyze_robustness_experiment(
    experiment_id: int = typer.Argument(..., help="Robustness experiment ID"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
    baseline_risk: float = typer.Option(1.0, "--baseline-risk", help="Baseline risk per trade %"),
    baseline_sl: float = typer.Option(5.0, "--baseline-sl", help="Baseline stop-loss %"),
    baseline_tp: float = typer.Option(10.0, "--baseline-tp", help="Baseline take-profit %"),
    export: str = typer.Option(None, "--export", help="Export to markdown/html file"),
):
    """
    Analyze robustness/sensitivity of parameters.

    Shows how sensitive performance is to parameter changes.

    Example:
        otonom-trader experiments robustness 7 --baseline-risk 1.0 --baseline-sl 5.0
        otonom-trader experiments robustness 7 --export reports/robustness.md
    """
    from .experiments.analysis import analyze_robustness, generate_comparison_table

    with get_session(db) as session:
        try:
            console.print(f"[bold cyan]Analyzing robustness experiment {experiment_id}...[/bold cyan]\n")

            baseline_params = {
                "risk_management.position_sizing.risk_per_trade_pct": baseline_risk,
                "risk_management.stop_loss.percentage": baseline_sl,
                "risk_management.take_profit.percentage": baseline_tp,
            }

            results = analyze_robustness(session, experiment_id, baseline_params)

            if not results:
                console.print("[yellow]No robustness results found.[/yellow]")
                return

            # Print results
            console.print("[bold]Robustness Analysis Results:[/bold]\n")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Parameter")
            table.add_column("Baseline", justify="right")
            table.add_column("Sharpe Range", justify="right")
            table.add_column("Std Dev", justify="right")
            table.add_column("Sensitivity", justify="right")
            table.add_column("Status")

            for r in results:
                status = "✓ Robust" if r.is_robust() else "⚠ Fragile"
                status_style = "green" if r.is_robust() else "yellow"

                table.add_row(
                    r.param_name.split(".")[-1],  # Show short name
                    f"{r.baseline_value:.2f}",
                    f"[{r.min_sharpe:.2f}, {r.max_sharpe:.2f}]",
                    f"{r.std_sharpe:.2f}",
                    f"{r.sensitivity:.2f}",
                    f"[{status_style}]{status}[/{status_style}]",
                )

            console.print(table)

            # Print summary
            robust_count = sum(1 for r in results if r.is_robust())
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Robust parameters: {robust_count}/{len(results)}")
            console.print(f"  Fragile parameters: {len(results) - robust_count}/{len(results)}")

            if robust_count == len(results):
                console.print(f"  [green]✓ Strategy is robust across all tested parameters[/green]")
            elif robust_count < len(results) / 2:
                console.print(f"  [red]⚠ Strategy is fragile - high sensitivity to parameter changes[/red]")

            # Export if requested
            if export:
                output_format = "markdown" if export.endswith(".md") else "html"
                table_output = generate_comparison_table(results, output_format)

                with open(export, "w") as f:
                    f.write(table_output)

                console.print(f"\n[bold green]✓ Report exported to:[/bold green] {export}")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Robustness analysis failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("compare")
def compare_experiments_cli(
    experiment_ids: str = typer.Argument(..., help="Comma-separated experiment IDs (e.g., 1,2,3)"),
    metric: str = typer.Option("test_sharpe", "--metric", help="Metric to compare"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Compare multiple experiments.

    Example:
        otonom-trader experiments compare 1,2,3
        otonom-trader experiments compare 1,2,3 --metric test_cagr
    """
    from .experiments.analysis import compare_experiments

    # Parse experiment IDs
    try:
        exp_ids = [int(x.strip()) for x in experiment_ids.split(",")]
    except ValueError:
        console.print("[red]Invalid experiment IDs. Use comma-separated numbers (e.g., 1,2,3)[/red]")
        raise typer.Exit(code=1)

    with get_session(db) as session:
        try:
            console.print(f"[bold cyan]Comparing experiments: {exp_ids}[/bold cyan]\n")

            df = compare_experiments(session, exp_ids, metric=metric)

            if df.empty:
                console.print("[yellow]No results to compare.[/yellow]")
                return

            # Print comparison table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Exp ID", justify="right")
            table.add_column("Name")
            table.add_column(f"Best {metric}", justify="right")
            table.add_column(f"Avg {metric}", justify="right")
            table.add_column(f"Worst {metric}", justify="right")
            table.add_column(f"Std {metric}", justify="right")
            table.add_column("Runs", justify="right")

            for _, row in df.iterrows():
                table.add_row(
                    str(row["experiment_id"]),
                    row["experiment_name"],
                    f"{row[f'best_{metric}']:.2f}",
                    f"{row[f'avg_{metric}']:.2f}",
                    f"{row[f'worst_{metric}']:.2f}",
                    f"{row[f'std_{metric}']:.2f}",
                    str(row["total_runs"]),
                )

            console.print(table)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Comparison failed: {e}", exc_info=True)
            raise typer.Exit(code=1)
