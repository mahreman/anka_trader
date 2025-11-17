"""
Experiment tracking database models.

Tracks parameter optimization experiments and their results.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    Float,
    ForeignKey,
    func,
)
from sqlalchemy.orm import relationship

from .schema import Base


class Experiment(Base):
    """
    Experiment metadata.

    An experiment represents a parameter search session over a strategy.

    Attributes:
        id: Unique experiment ID
        name: Experiment name (unique)
        description: Experiment description
        base_strategy_name: Base strategy (e.g., "baseline_v1")
        param_grid_path: Path to parameter grid YAML
        created_at: Creation timestamp
        runs: Related experiment runs
    """

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    base_strategy_name = Column(String(128), nullable=False)
    param_grid_path = Column(String(256), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', strategy='{self.base_strategy_name}')>"


class ExperimentRun(Base):
    """
    Individual experiment run with specific parameter values.

    Each run represents one parameter combination tested on train/test splits.

    Attributes:
        id: Unique run ID
        experiment_id: Parent experiment ID
        run_index: Sequential run number within experiment
        param_values_json: JSON string of parameter values
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date (optional)
        test_end: Test period end date (optional)
        train_cagr: Training CAGR
        train_sharpe: Training Sharpe ratio
        train_max_dd: Training maximum drawdown
        test_cagr: Test CAGR
        test_sharpe: Test Sharpe ratio
        test_max_dd: Test maximum drawdown
        status: Run status (pending/running/done/failed)
        error_message: Error message if failed
        created_at: Creation timestamp
        experiment: Parent experiment
    """

    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)

    run_index = Column(Integer, nullable=False)

    # JSON string: {"risk_pct": 1.0, "stop_loss_pct": 5, ...}
    param_values_json = Column(Text, nullable=False)

    # Date ranges
    train_start = Column(Date, nullable=False)
    train_end = Column(Date, nullable=False)
    test_start = Column(Date, nullable=True)
    test_end = Column(Date, nullable=True)

    # Training metrics
    train_cagr = Column(Float, nullable=True)
    train_sharpe = Column(Float, nullable=True)
    train_max_dd = Column(Float, nullable=True)
    train_win_rate = Column(Float, nullable=True)
    train_total_trades = Column(Integer, nullable=True)

    # Test metrics
    test_cagr = Column(Float, nullable=True)
    test_sharpe = Column(Float, nullable=True)
    test_max_dd = Column(Float, nullable=True)
    test_win_rate = Column(Float, nullable=True)
    test_total_trades = Column(Integer, nullable=True)

    # Status tracking
    status = Column(String(32), nullable=False, default="pending")  # pending/running/done/failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    experiment = relationship("Experiment", back_populates="runs")

    def __repr__(self) -> str:
        return (
            f"<ExperimentRun(id={self.id}, experiment_id={self.experiment_id}, "
            f"run_index={self.run_index}, status='{self.status}')>"
        )
