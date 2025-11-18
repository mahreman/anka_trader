"""
Strategy promotion workflow manager.

Implements the 6-step promotion ritual:
1. Run experiment (grid/ablation/robustness)
2. Select best runs (top 1-3 by Sharpe + DD + robustness)
3. Promote to new version (generate new YAML)
4. Document changes (STRATEGY_LOG.md)
5. Validate with full backtest
6. Deploy to paper daemon (champion/challenger)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sqlalchemy.orm import Session

from .config import load_strategy_config, StrategyConfig
from ..data import Experiment, ExperimentRun
from .versioning import (
    StrategyVersion,
    PromotionCriteria,
    PromotionRecord,
    find_latest_version,
    append_to_strategy_log,
    compare_strategy_configs,
    detect_promotion_type,
)

logger = logging.getLogger(__name__)


@dataclass
class PromotionCandidate:
    """
    Candidate for promotion from experiment results.

    Attributes:
        run_index: Experiment run index
        params: Parameter overrides
        test_sharpe: Test Sharpe ratio
        test_cagr: Test CAGR
        test_max_dd: Test max drawdown
        test_win_rate: Test win rate
        test_trades: Number of test trades
        train_sharpe: Train Sharpe ratio
        overfitting_ratio: Train/Test Sharpe ratio
    """
    run_index: int
    params: Dict[str, Any]
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    test_win_rate: float
    test_trades: int
    train_sharpe: float

    @property
    def overfitting_ratio(self) -> float:
        """Calculate overfitting ratio (train/test Sharpe)."""
        if self.test_sharpe == 0:
            return float('inf')
        return self.train_sharpe / self.test_sharpe

    def __repr__(self) -> str:
        return (
            f"Run {self.run_index}: "
            f"Sharpe={self.test_sharpe:.2f}, "
            f"CAGR={self.test_cagr:.1f}%, "
            f"MaxDD={self.test_max_dd:.1f}%, "
            f"Overfit={self.overfitting_ratio:.2f}"
        )


def extract_promotion_candidates(
    session: Session,
    experiment_id: int,
    top_n: int = 3,
    max_overfitting_ratio: float = 1.5,
) -> List[PromotionCandidate]:
    """
    Extract top promotion candidates from experiment.

    Args:
        session: Database session
        experiment_id: Experiment ID
        top_n: Number of top candidates to return
        max_overfitting_ratio: Maximum acceptable train/test ratio

    Returns:
        List of PromotionCandidate objects

    Example:
        >>> candidates = extract_promotion_candidates(session, 1, top_n=3)
        >>> for c in candidates:
        ...     print(c)
    """
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Get successful runs
    successful_runs = [r for r in experiment.runs if r.status == "done"]

    if not successful_runs:
        logger.warning(f"No successful runs in experiment {experiment_id}")
        return []

    # Sort by test Sharpe
    sorted_runs = sorted(
        successful_runs,
        key=lambda r: r.test_sharpe or 0,
        reverse=True,
    )

    # Extract candidates
    candidates = []

    for run in sorted_runs[:top_n * 2]:  # Get more to filter
        if not run.test_sharpe:
            continue

        # Parse params
        params = json.loads(run.param_values_json)

        # Create candidate
        candidate = PromotionCandidate(
            run_index=run.run_index,
            params=params,
            test_sharpe=run.test_sharpe or 0.0,
            test_cagr=run.test_cagr or 0.0,
            test_max_dd=run.test_max_dd or 0.0,
            test_win_rate=run.test_win_rate or 0.0,
            test_trades=run.test_total_trades or 0,
            train_sharpe=run.train_sharpe or 0.0,
        )

        # Filter by overfitting
        if candidate.overfitting_ratio > max_overfitting_ratio:
            logger.info(f"Skipping run {run.run_index}: overfitting ratio {candidate.overfitting_ratio:.2f} > {max_overfitting_ratio}")
            continue

        candidates.append(candidate)

        if len(candidates) >= top_n:
            break

    return candidates


def validate_promotion(
    candidate: PromotionCandidate,
    criteria: PromotionCriteria,
    previous_version: Optional[StrategyVersion] = None,
    previous_sharpe: Optional[float] = None,
    previous_max_dd: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate if candidate meets promotion criteria.

    Args:
        candidate: Promotion candidate
        criteria: Promotion criteria
        previous_version: Previous strategy version (for comparison)
        previous_sharpe: Previous version test Sharpe
        previous_max_dd: Previous version test max DD

    Returns:
        Tuple of (passes, reason)

    Example:
        >>> criteria = PromotionCriteria()
        >>> passes, reason = validate_promotion(candidate, criteria)
        >>> if passes:
        ...     print("Promotion approved!")
    """
    # Validate absolute criteria
    absolute_pass, absolute_reason = criteria.validate_absolute(
        test_sharpe=candidate.test_sharpe,
        test_max_dd=candidate.test_max_dd,
        test_trades=candidate.test_trades,
    )

    if not absolute_pass:
        return False, f"Absolute criteria failed: {absolute_reason}"

    # Validate improvement (if previous version exists)
    if previous_version and previous_sharpe and previous_max_dd:
        improvement_pass, improvement_reason = criteria.validate_improvement(
            new_test_sharpe=candidate.test_sharpe,
            new_test_max_dd=candidate.test_max_dd,
            old_test_sharpe=previous_sharpe,
            old_test_max_dd=previous_max_dd,
        )

        if not improvement_pass:
            return False, f"Improvement criteria failed: {improvement_reason}"

        return True, f"All criteria passed. {improvement_reason}"

    # No previous version, just validate absolute
    return True, f"All absolute criteria passed (no previous version to compare)"


def promote_strategy(
    base_strategy_path: Path,
    candidate: PromotionCandidate,
    promotion_type: Optional[str] = None,
    experiment_id: Optional[int] = None,
    changes_description: Optional[str] = None,
    rationale: Optional[str] = None,
    output_dir: Path = Path("strategies"),
) -> Tuple[Path, PromotionRecord]:
    """
    Promote strategy to new version.

    Args:
        base_strategy_path: Path to base strategy YAML
        candidate: Promotion candidate with params
        promotion_type: "major" or "minor" (auto-detected if None)
        experiment_id: Experiment that produced this candidate
        changes_description: Description of changes
        rationale: Why this promotion happened
        output_dir: Where to save new strategy file

    Returns:
        Tuple of (new_strategy_path, promotion_record)

    Example:
        >>> new_path, record = promote_strategy(
        ...     Path("strategies/baseline_v1.0.yaml"),
        ...     candidate,
        ...     promotion_type="minor",
        ...     changes_description="Increased risk from 1.0% to 1.5%",
        ...     rationale="Grid search showed better risk-adjusted returns",
        ... )
    """
    # Load base strategy
    base_strategy = load_strategy_config(base_strategy_path)

    # Parse current version
    current_version = StrategyVersion.from_path(base_strategy_path)

    if not current_version:
        # Assume initial version if parsing fails
        logger.warning(f"Could not parse version from {base_strategy_path}, assuming v1.0")
        current_version = StrategyVersion(name="baseline", major=1, minor=0)

    # Load config as dict for comparison
    with open(base_strategy_path, "r") as f:
        old_config = yaml.safe_load(f)

    # Apply parameter overrides
    new_config = old_config.copy()

    for param_name, param_value in candidate.params.items():
        # Navigate nested dict and set value
        keys = param_name.split(".")
        target = new_config

        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        target[keys[-1]] = param_value

    # Detect promotion type if not specified
    if promotion_type is None:
        promotion_type = detect_promotion_type(old_config, new_config)

    # Bump version
    if promotion_type == "major":
        new_version = current_version.bump_major()
    else:
        new_version = current_version.bump_minor()

    # Update version in config
    new_config["version"] = new_version.full_version

    # Generate new file path
    new_strategy_path = output_dir / new_version.file_name

    # Save new strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(new_strategy_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created new strategy: {new_strategy_path}")

    # Generate changes description if not provided
    if changes_description is None:
        changes_description = compare_strategy_configs(old_config, new_config)

    # Generate rationale if not provided
    if rationale is None:
        rationale = f"Experiment #{experiment_id} showed improved performance" if experiment_id else "Manual promotion"

    # Create promotion record
    record = PromotionRecord(
        from_version=current_version,
        to_version=new_version,
        promotion_type=promotion_type,
        experiment_id=experiment_id,
        test_sharpe=candidate.test_sharpe,
        test_cagr=candidate.test_cagr,
        test_max_dd=candidate.test_max_dd,
        changes=changes_description,
        rationale=rationale,
    )

    return new_strategy_path, record


def run_promotion_workflow(
    session: Session,
    experiment_id: int,
    base_strategy_path: Path,
    criteria: Optional[PromotionCriteria] = None,
    top_n: int = 3,
    auto_select: bool = False,
    output_dir: Path = Path("strategies"),
    log_path: Path = Path("STRATEGY_LOG.md"),
) -> Tuple[Optional[Path], Optional[PromotionRecord]]:
    """
    Run complete promotion workflow.

    Workflow:
    1. Extract top candidates from experiment
    2. Validate candidates against criteria
    3. Select best candidate (manual or auto)
    4. Promote to new version
    5. Document in strategy log
    6. Return new strategy for validation

    Args:
        session: Database session
        experiment_id: Experiment ID
        base_strategy_path: Path to base strategy
        criteria: Promotion criteria (uses defaults if None)
        top_n: Number of candidates to extract
        auto_select: Auto-select best candidate (vs manual)
        output_dir: Output directory for new strategy
        log_path: Path to strategy log file

    Returns:
        Tuple of (new_strategy_path, promotion_record) or (None, None) if no valid candidates

    Example:
        >>> new_path, record = run_promotion_workflow(
        ...     session=session,
        ...     experiment_id=5,
        ...     base_strategy_path=Path("strategies/baseline_v1.0.yaml"),
        ...     auto_select=True,
        ... )
        >>> if new_path:
        ...     print(f"Promoted to: {new_path}")
    """
    if criteria is None:
        criteria = PromotionCriteria()

    # Step 1: Extract candidates
    logger.info(f"Step 1: Extracting top {top_n} candidates from experiment {experiment_id}")
    candidates = extract_promotion_candidates(session, experiment_id, top_n=top_n)

    if not candidates:
        logger.warning("No valid candidates found")
        return None, None

    logger.info(f"Found {len(candidates)} candidates:")
    for c in candidates:
        logger.info(f"  {c}")

    # Get previous version info (if exists)
    current_version = StrategyVersion.from_path(base_strategy_path)
    previous_sharpe = None
    previous_max_dd = None

    # TODO: Load previous version metrics from log or database
    # For now, we don't have previous metrics

    # Step 2: Validate candidates
    logger.info("Step 2: Validating candidates against criteria")
    valid_candidates = []

    for candidate in candidates:
        passes, reason = validate_promotion(
            candidate=candidate,
            criteria=criteria,
            previous_version=current_version,
            previous_sharpe=previous_sharpe,
            previous_max_dd=previous_max_dd,
        )

        if passes:
            logger.info(f"  ✓ Run {candidate.run_index}: {reason}")
            valid_candidates.append(candidate)
        else:
            logger.warning(f"  ✗ Run {candidate.run_index}: {reason}")

    if not valid_candidates:
        logger.warning("No candidates passed validation criteria")
        return None, None

    # Step 3: Select best candidate
    if auto_select:
        logger.info("Step 3: Auto-selecting best candidate")
        selected = valid_candidates[0]  # Already sorted by Sharpe
        logger.info(f"  Selected: {selected}")
    else:
        logger.info("Step 3: Manual selection required")
        logger.info("Valid candidates:")
        for i, c in enumerate(valid_candidates):
            logger.info(f"  [{i}] {c}")
        # Return candidates for manual selection
        # CLI will handle this
        return None, None

    # Step 4: Promote
    logger.info("Step 4: Promoting strategy")
    new_strategy_path, record = promote_strategy(
        base_strategy_path=base_strategy_path,
        candidate=selected,
        experiment_id=experiment_id,
        output_dir=output_dir,
    )

    logger.info(f"  Created: {new_strategy_path}")

    # Step 5: Document
    logger.info("Step 5: Documenting promotion")
    append_to_strategy_log(record, log_path=log_path)
    logger.info(f"  Updated: {log_path}")

    # Step 6: Ready for validation
    logger.info("Step 6: New strategy ready for validation")
    logger.info(f"  Next: Run full backtest on {new_strategy_path}")
    logger.info(f"  Next: Deploy to paper daemon for champion/challenger test")

    return new_strategy_path, record


def compare_champion_challenger(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare champion (current) vs challenger (new) strategy.

    Args:
        champion_metrics: Champion strategy metrics
        challenger_metrics: Challenger strategy metrics

    Returns:
        Comparison results with recommendations

    Example:
        >>> comparison = compare_champion_challenger(
        ...     champion_metrics={"sharpe": 1.5, "max_dd": -15.0},
        ...     challenger_metrics={"sharpe": 1.6, "max_dd": -14.0},
        ... )
        >>> print(comparison["recommendation"])
    """
    # Calculate improvements
    sharpe_improvement = (
        (challenger_metrics["sharpe"] - champion_metrics["sharpe"])
        / champion_metrics["sharpe"] * 100
    )

    dd_improvement = challenger_metrics["max_dd"] - champion_metrics["max_dd"]  # Negative is better

    cagr_improvement = (
        (challenger_metrics.get("cagr", 0) - champion_metrics.get("cagr", 0))
        / champion_metrics.get("cagr", 1) * 100
    )

    # Determine recommendation
    if sharpe_improvement > 5 and dd_improvement <= 0:
        recommendation = "REPLACE: Challenger beats champion on all metrics"
    elif sharpe_improvement > 10:
        recommendation = "REPLACE: Significant Sharpe improvement"
    elif sharpe_improvement > 0 and dd_improvement <= 2:
        recommendation = "REPLACE: Modest improvement with acceptable DD"
    elif sharpe_improvement < -5:
        recommendation = "KEEP CHAMPION: Challenger underperforms"
    else:
        recommendation = "CONTINUE TESTING: Results inconclusive"

    return {
        "sharpe_improvement_pct": sharpe_improvement,
        "dd_improvement": dd_improvement,
        "cagr_improvement_pct": cagr_improvement,
        "recommendation": recommendation,
        "champion": champion_metrics,
        "challenger": challenger_metrics,
    }
