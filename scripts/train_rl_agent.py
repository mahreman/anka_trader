#!/usr/bin/env python
"""
Train RL agent using behavior cloning.

Example:
    # Basic training
    python scripts/train_rl_agent.py \\
        --dataset data/offline_dataset.npz \\
        --output models/rl_bc

    # Advanced training with custom parameters
    python scripts/train_rl_agent.py \\
        --dataset data/offline_dataset.npz \\
        --output models/rl_bc \\
        --epochs 50 \\
        --batch-size 512 \\
        --lr 0.001 \\
        --hidden-dim 256 \\
        --num-layers 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from otonom_trader.research.rl_training import (
    TrainingConfig,
    BehaviorCloningTrainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train RL agent using behavior cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to offline dataset (.npz file)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_bc",
        help="Output directory for models and metrics (default: models/rl_bc)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs (default: 20)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization weight (default: 1e-5)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1)",
    )

    # Loss weights
    parser.add_argument(
        "--action-weight",
        type=float,
        default=1.0,
        help="Weight for action classification loss (default: 1.0)",
    )

    parser.add_argument(
        "--strength-weight",
        type=float,
        default=1.0,
        help="Weight for strength regression loss (default: 1.0)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    # Validate dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    # Create config
    cfg = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        val_ratio=args.val_ratio,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        action_weight=args.action_weight,
        strength_weight=args.strength_weight,
        seed=args.seed,
        device=args.device,
    )

    logger.info("=" * 60)
    logger.info("RL Behavior Cloning Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Num layers: {args.num_layers}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Train
    try:
        trainer = BehaviorCloningTrainer(cfg)
        metrics = trainer.train()

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
        logger.info(f"Best epoch: {metrics.best_epoch}")
        logger.info(f"Final train loss: {metrics.final_train_loss:.4f}")
        logger.info(f"Final val loss: {metrics.final_val_loss:.4f}")
        logger.info(f"Train action accuracy: {metrics.train_action_acc:.3f}")
        logger.info(f"Val action accuracy: {metrics.val_action_acc:.3f}")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.output}/policy_best.pt")
        logger.info(f"Metrics saved to: {args.output}/training_metrics.json")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
