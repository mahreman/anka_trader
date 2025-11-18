"""
Strategy management CLI commands.

Provides commands for strategy versioning, promotion, and comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .data import get_session
from .strategy.versioning import (
    StrategyVersion,
    PromotionCriteria,
    find_latest_version,
    create_strategy_log_template,
)
from .strategy.promotion import (
    extract_promotion_candidates,
    validate_promotion,
    promote_strategy,
    run_promotion_workflow,
    compare_champion_challenger,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="strategy",
    help="Strategy versioning and promotion management",
)


@app.command("init-log")
def init_strategy_log(
    output: str = typer.Option("STRATEGY_LOG.md", "--output", help="Output path"),
):
    """
    Initialize strategy promotion log.

    Creates STRATEGY_LOG.md with promotion criteria and workflow template.

    Example:
        otonom-trader strategy init-log
        otonom-trader strategy init-log --output docs/STRATEGY_LOG.md
    """
    output_path = Path(output)

    try:
        create_strategy_log_template(output_path)
        console.print(f"[bold green]✓ Created strategy log:[/bold green] {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("latest")
def show_latest_version(
    strategy_name: str = typer.Argument(..., help="Strategy base name (e.g., 'baseline')"),
    strategies_dir: str = typer.Option("strategies", "--dir", help="Strategies directory"),
):
    """
    Show latest version of a strategy.

    Example:
        otonom-trader strategy latest baseline
    """
    version = find_latest_version(strategy_name, Path(strategies_dir))

    if not version:
        console.print(f"[yellow]No versions found for strategy: {strategy_name}[/yellow]")
        return

    console.print(f"[bold cyan]Latest version:[/bold cyan] {version}")
    console.print(f"  File: {version.file_name}")


@app.command("candidates")
def show_promotion_candidates(
    experiment_id: int = typer.Argument(..., help="Experiment ID"),
    top_n: int = typer.Option(5, "--top", help="Number of candidates to show"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Show promotion candidates from experiment.

    Example:
        otonom-trader strategy candidates 5 --top 3
    """
    with get_session(db) as session:
        try:
            console.print(f"[bold cyan]Extracting promotion candidates from experiment {experiment_id}...[/bold cyan]\n")

            candidates = extract_promotion_candidates(session, experiment_id, top_n=top_n)

            if not candidates:
                console.print("[yellow]No valid candidates found.[/yellow]")
                return

            # Display candidates
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", justify="right")
            table.add_column("Run", justify="right")
            table.add_column("Test Sharpe", justify="right")
            table.add_column("Test CAGR", justify="right")
            table.add_column("Max DD", justify="right")
            table.add_column("Win Rate", justify="right")
            table.add_column("Trades", justify="right")
            table.add_column("Overfit", justify="right")

            for i, candidate in enumerate(candidates, 1):
                overfit_color = "green" if candidate.overfitting_ratio < 1.3 else "yellow" if candidate.overfitting_ratio < 1.5 else "red"

                table.add_row(
                    str(i),
                    str(candidate.run_index),
                    f"{candidate.test_sharpe:.2f}",
                    f"{candidate.test_cagr:.1f}%",
                    f"{candidate.test_max_dd:.1f}%",
                    f"{candidate.test_win_rate:.1f}%",
                    str(candidate.test_trades),
                    f"[{overfit_color}]{candidate.overfitting_ratio:.2f}[/{overfit_color}]",
                )

            console.print(table)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Failed to extract candidates: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("promote")
def promote_strategy_cli(
    experiment_id: int = typer.Argument(..., help="Experiment ID"),
    base_strategy: str = typer.Argument(..., help="Base strategy path (e.g., strategies/baseline_v1.0.yaml)"),
    run_index: int = typer.Option(None, "--run", help="Specific run to promote (auto-select if not provided)"),
    promotion_type: str = typer.Option(None, "--type", help="'major' or 'minor' (auto-detected if not provided)"),
    changes: str = typer.Option(None, "--changes", help="Description of changes"),
    rationale: str = typer.Option(None, "--rationale", help="Why this promotion"),
    output_dir: str = typer.Option("strategies", "--output", help="Output directory"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Promote strategy to new version from experiment results.

    Example:
        # Auto-select best run
        otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml

        # Promote specific run
        otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --run 3

        # Force major version bump
        otonom-trader strategy promote 5 strategies/baseline_v1.0.yaml --type major
    """
    base_path = Path(base_strategy)

    if not base_path.exists():
        console.print(f"[red]Base strategy not found: {base_path}[/red]")
        raise typer.Exit(code=1)

    with get_session(db) as session:
        try:
            console.print(Panel(
                f"[bold cyan]Strategy Promotion Workflow[/bold cyan]\n"
                f"Experiment: #{experiment_id}\n"
                f"Base: {base_path}\n"
                f"Output: {output_dir}/",
                title="Promotion",
                border_style="cyan"
            ))

            # Extract candidates
            console.print("\n[bold]Step 1: Extracting candidates...[/bold]")
            candidates = extract_promotion_candidates(session, experiment_id, top_n=5)

            if not candidates:
                console.print("[red]No valid candidates found.[/red]")
                raise typer.Exit(code=1)

            console.print(f"  Found {len(candidates)} candidates")

            # Select candidate
            if run_index is not None:
                # Find specific run
                selected = next((c for c in candidates if c.run_index == run_index), None)
                if not selected:
                    console.print(f"[red]Run {run_index} not found in candidates.[/red]")
                    raise typer.Exit(code=1)
                console.print(f"\n[bold]Step 2: Selected run {run_index}[/bold]")
            else:
                # Auto-select best
                selected = candidates[0]
                console.print(f"\n[bold]Step 2: Auto-selected best run {selected.run_index}[/bold]")

            console.print(f"  {selected}")

            # Validate
            console.print("\n[bold]Step 3: Validating promotion criteria...[/bold]")
            criteria = PromotionCriteria()
            passes, reason = validate_promotion(selected, criteria)

            if not passes:
                console.print(f"  [red]✗ Validation failed: {reason}[/red]")
                raise typer.Exit(code=1)

            console.print(f"  [green]✓ {reason}[/green]")

            # Promote
            console.print("\n[bold]Step 4: Creating new strategy version...[/bold]")
            new_path, record = promote_strategy(
                base_strategy_path=base_path,
                candidate=selected,
                promotion_type=promotion_type,
                experiment_id=experiment_id,
                changes_description=changes,
                rationale=rationale,
                output_dir=Path(output_dir),
            )

            console.print(f"  [green]✓ Created: {new_path}[/green]")

            # Document
            console.print("\n[bold]Step 5: Documenting promotion...[/bold]")
            from .strategy.versioning import append_to_strategy_log
            append_to_strategy_log(record)
            console.print(f"  [green]✓ Updated: STRATEGY_LOG.md[/green]")

            # Summary
            console.print("\n" + "="*60)
            console.print(f"[bold green]✓ Promotion complete![/bold green]")
            console.print("="*60)
            console.print(f"New version: [bold]{record.to_version}[/bold]")
            console.print(f"Type: {record.promotion_type.capitalize()}")
            console.print(f"Performance: Sharpe={record.test_sharpe:.2f}, CAGR={record.test_cagr:.1f}%, MaxDD={record.test_max_dd:.1f}%")
            console.print(f"\n[bold]Next steps:[/bold]")
            console.print(f"  1. Run full backtest: otonom-trader research full {new_path.stem}")
            console.print(f"  2. Compare with previous: otonom-trader strategy compare [old] [new]")
            console.print(f"  3. Deploy to paper daemon for champion/challenger test")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Promotion failed: {e}", exc_info=True)
            raise typer.Exit(code=1)


@app.command("compare")
def compare_strategies(
    champion: str = typer.Argument(..., help="Champion strategy (current)"),
    challenger: str = typer.Argument(..., help="Challenger strategy (new)"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Compare champion vs challenger strategies.

    Example:
        otonom-trader strategy compare baseline_v1.0 baseline_v1.1
    """
    # TODO: Load metrics from research reports or database
    # For now, show structure

    console.print(Panel(
        f"[bold cyan]Champion vs Challenger Comparison[/bold cyan]\n"
        f"Champion: {champion}\n"
        f"Challenger: {challenger}",
        title="Strategy Comparison",
        border_style="cyan"
    ))

    # Placeholder metrics
    champion_metrics = {
        "sharpe": 1.5,
        "cagr": 25.0,
        "max_dd": -15.0,
        "win_rate": 55.0,
    }

    challenger_metrics = {
        "sharpe": 1.6,
        "cagr": 27.0,
        "max_dd": -14.0,
        "win_rate": 58.0,
    }

    comparison = compare_champion_challenger(champion_metrics, challenger_metrics)

    # Display comparison
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Champion", justify="right")
    table.add_column("Challenger", justify="right")
    table.add_column("Improvement", justify="right")

    table.add_row(
        "Sharpe Ratio",
        f"{champion_metrics['sharpe']:.2f}",
        f"{challenger_metrics['sharpe']:.2f}",
        f"[green]+{comparison['sharpe_improvement_pct']:.1f}%[/green]" if comparison['sharpe_improvement_pct'] > 0 else f"[red]{comparison['sharpe_improvement_pct']:.1f}%[/red]"
    )

    table.add_row(
        "CAGR",
        f"{champion_metrics['cagr']:.1f}%",
        f"{challenger_metrics['cagr']:.1f}%",
        f"[green]+{comparison['cagr_improvement_pct']:.1f}%[/green]" if comparison['cagr_improvement_pct'] > 0 else f"[red]{comparison['cagr_improvement_pct']:.1f}%[/red]"
    )

    table.add_row(
        "Max Drawdown",
        f"{champion_metrics['max_dd']:.1f}%",
        f"{challenger_metrics['max_dd']:.1f}%",
        f"[green]{comparison['dd_improvement']:.1f}% better[/green]" if comparison['dd_improvement'] <= 0 else f"[red]{comparison['dd_improvement']:.1f}% worse[/red]"
    )

    console.print(table)

    # Recommendation
    console.print(f"\n[bold]Recommendation:[/bold]")
    console.print(f"  {comparison['recommendation']}")

    console.print(f"\n[yellow]Note: This is a placeholder. Implement actual metric loading from research reports.[/yellow]")


@app.command("workflow")
def run_full_promotion_workflow(
    experiment_id: int = typer.Argument(..., help="Experiment ID"),
    base_strategy: str = typer.Argument(..., help="Base strategy path"),
    auto_select: bool = typer.Option(False, "--auto", help="Auto-select best candidate"),
    db: str = typer.Option("trader.db", "--db", help="Database path"),
):
    """
    Run full promotion workflow (6 steps).

    Steps:
    1. Extract candidates
    2. Validate criteria
    3. Select best (auto or manual)
    4. Promote to new version
    5. Document in log
    6. Ready for validation

    Example:
        # Manual selection
        otonom-trader strategy workflow 5 strategies/baseline_v1.0.yaml

        # Auto-select best
        otonom-trader strategy workflow 5 strategies/baseline_v1.0.yaml --auto
    """
    base_path = Path(base_strategy)

    with get_session(db) as session:
        try:
            new_path, record = run_promotion_workflow(
                session=session,
                experiment_id=experiment_id,
                base_strategy_path=base_path,
                auto_select=auto_select,
            )

            if not new_path:
                console.print("[yellow]Workflow incomplete (manual selection required)[/yellow]")
                return

            console.print(f"\n[bold green]✓ Workflow complete![/bold green]")
            console.print(f"New strategy: {new_path}")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.error(f"Workflow failed: {e}", exc_info=True)
            raise typer.Exit(code=1)
