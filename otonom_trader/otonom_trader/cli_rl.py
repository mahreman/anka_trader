"""
RL / Behavior Cloning CLI commands.

Commands for training and managing RL agents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .research.rl_training import (
    train_behavior_cloning,
    TrainingConfig,
)
from .research.rl_inference import load_rl_policy

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="RL / Behavior Cloning commands")


@app.command("train-bc")
def train_bc(
    dataset_path: str = typer.Option(
        "data/rl/offline_dataset.npz",
        "--dataset-path",
        "-d",
        help="Path to offline dataset (.npz file)",
    ),
    output_dir: str = typer.Option(
        "models/rl_bc",
        "--output-dir",
        "-o",
        help="Output directory for models and metrics",
    ),
    batch_size: int = typer.Option(
        256,
        "--batch-size",
        help="Training batch size",
    ),
    lr: float = typer.Option(
        1e-3,
        "--lr",
        help="Learning rate",
    ),
    max_epochs: int = typer.Option(
        20,
        "--max-epochs",
        help="Maximum training epochs",
    ),
    hidden_dim: int = typer.Option(
        128,
        "--hidden-dim",
        help="Hidden layer dimension",
    ),
    num_layers: int = typer.Option(
        2,
        "--num-layers",
        help="Number of hidden layers",
    ),
    dropout: float = typer.Option(
        0.1,
        "--dropout",
        help="Dropout probability",
    ),
    val_ratio: float = typer.Option(
        0.1,
        "--val-ratio",
        help="Validation set ratio",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device to use (cuda/cpu)",
    ),
):
    """
    Train RL agent using behavior cloning.

    Example:
        otonom-trader rl train-bc \\
            --dataset-path data/rl/offline_dataset.npz \\
            --output-dir models/rl_bc_v1 \\
            --max-epochs 30
    """
    console.print("\n[bold cyan]RL Behavior Cloning Training[/bold cyan]\n")

    # Validate dataset exists
    dataset = Path(dataset_path)
    if not dataset.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset not found: {dataset_path}")
        raise typer.Exit(1)

    console.print(f"Dataset: [cyan]{dataset_path}[/cyan]")
    console.print(f"Output: [cyan]{output_dir}[/cyan]")
    console.print(f"Epochs: [cyan]{max_epochs}[/cyan]")
    console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"Learning rate: [cyan]{lr}[/cyan]")
    console.print(f"Hidden dim: [cyan]{hidden_dim}[/cyan]")
    console.print(f"Num layers: [cyan]{num_layers}[/cyan]\n")

    # Create config
    cfg = TrainingConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        val_ratio=val_ratio,
        device=device,
    )

    # Train
    try:
        metrics = train_behavior_cloning(cfg.__dict__)

        # Display results
        console.print("\n[bold green]✓ Training Complete![/bold green]\n")

        results_table = Table(title="Training Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Best Val Loss", f"{metrics.best_val_loss:.4f}")
        results_table.add_row("Best Epoch", str(metrics.best_epoch))
        results_table.add_row("Final Train Loss", f"{metrics.final_train_loss:.4f}")
        results_table.add_row("Final Val Loss", f"{metrics.final_val_loss:.4f}")
        results_table.add_row(
            "Train Action Accuracy", f"{metrics.train_action_acc:.3f}"
        )
        results_table.add_row("Val Action Accuracy", f"{metrics.val_action_acc:.3f}")

        console.print(results_table)

        console.print(
            f"\n[bold]Model saved to:[/bold] [cyan]{output_dir}/policy_best.pt[/cyan]"
        )
        console.print(
            f"[bold]Metrics saved to:[/bold] [cyan]{output_dir}/training_metrics.json[/cyan]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error during training:[/bold red] {e}")
        logger.exception("Training failed")
        raise typer.Exit(1)


@app.command("info")
def info(
    model_path: str = typer.Argument(
        ...,
        help="Path to trained model (.pt file)",
    ),
):
    """
    Display information about trained RL model.

    Example:
        otonom-trader rl info models/rl_bc/policy_best.pt
    """
    console.print(f"\n[bold cyan]RL Model Information[/bold cyan]\n")

    # Validate model exists
    model = Path(model_path)
    if not model.exists():
        console.print(f"[bold red]Error:[/bold red] Model not found: {model_path}")
        raise typer.Exit(1)

    try:
        # Load model
        policy = load_rl_policy(model_path, device="cpu")
        metadata = policy.get_metadata()

        # Display info
        info_table = Table(title=f"Model: {model_path}")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("State Dimension", str(metadata["state_dim"]))
        info_table.add_row("Device", metadata["device"])

        if metadata["val_loss"] is not None:
            info_table.add_row(
                "Validation Loss", f"{metadata['val_loss']:.4f}"
            )

        if metadata["val_action_acc"] is not None:
            info_table.add_row(
                "Val Action Accuracy", f"{metadata['val_action_acc']:.3f}"
            )

        console.print(info_table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error loading model:[/bold red] {e}")
        logger.exception("Model loading failed")
        raise typer.Exit(1)


@app.command("predict")
def predict(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to trained model (.pt file)",
    ),
    state_dim: int = typer.Option(
        34,
        "--state-dim",
        help="State dimension (for random test)",
    ),
):
    """
    Test prediction with random state (for debugging).

    Example:
        otonom-trader rl predict --model-path models/rl_bc/policy_best.pt
    """
    import numpy as np
    from .research.rl_agent import RlState

    console.print(f"\n[bold cyan]RL Model Prediction Test[/bold cyan]\n")

    # Validate model exists
    model = Path(model_path)
    if not model.exists():
        console.print(f"[bold red]Error:[/bold red] Model not found: {model_path}")
        raise typer.Exit(1)

    try:
        # Load model
        policy = load_rl_policy(model_path, device="cpu")

        # Create random state for testing
        console.print("[yellow]Creating random test state...[/yellow]")

        # Random features (normalized)
        returns_history = np.random.randn(20) * 0.02  # 2% daily volatility
        technical_indicators = {
            "rsi": np.random.rand() * 100,
            "macd": np.random.randn() * 0.01,
            "bollinger_width": np.random.rand() * 0.1,
            "volume_ratio": np.random.rand() * 2.0,
        }

        state = RlState(
            returns_history=returns_history,
            technical_indicators=technical_indicators,
            portfolio_position=0.0,
            portfolio_leverage=0.0,
            portfolio_cash_pct=1.0,
            days_since_trade=0,
            regime=1,
            dsi=0.9,
            news_sentiment=0.0,
            macro_sentiment=0.0,
        )

        # Predict
        action, strength, probs = policy.predict(state, return_probs=True)

        # Display results
        console.print("\n[bold green]Prediction Results:[/bold green]\n")

        results_table = Table()
        results_table.add_column("Action", style="cyan")
        results_table.add_column("Probability", style="green")

        results_table.add_row("HOLD", f"{probs[0]:.3f}")
        results_table.add_row("BUY", f"{probs[1]:.3f}")
        results_table.add_row("SELL", f"{probs[2]:.3f}")

        console.print(results_table)

        console.print(
            f"\n[bold]Predicted Action:[/bold] [cyan]{action.name}[/cyan]"
        )
        console.print(
            f"[bold]Position Strength:[/bold] [cyan]{strength:.2f}[/cyan]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error during prediction:[/bold red] {e}")
        logger.exception("Prediction failed")
        raise typer.Exit(1)
