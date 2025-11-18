"""
Statistical significance testing for experiment comparison.

Provides tools to compare experiment runs and determine if performance
differences are statistically significant.

Usage:
    from otonom_trader.eval.significance import compare_top_two, summarize_runs
    from otonom_trader.data import get_session
    from otonom_trader.data.schema_experiments import Experiment

    with get_session() as session:
        experiment = session.query(Experiment).filter_by(id=1).first()
        runs = [r for r in experiment.runs if r.status == "done"]

        # Compare top 2 runs
        result = compare_top_two(runs, n_obs=252*3)  # 3 years of daily data
        print(f"Winner: Run #{result.run_a.run_index}")
        print(f"Sharpe difference: {result.sharpe_diff:.2f}")
        print(f"Z-score: {result.z_score:.2f}")
        print(f"Is significant? {result.is_significant}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from ..data.schema_experiments import ExperimentRun


@dataclass
class RunSummary:
    """
    Summary statistics for a single experiment run.

    Attributes:
        run_index: Run index within experiment
        test_sharpe: Test Sharpe ratio
        test_cagr: Test CAGR (%)
        test_max_dd: Test maximum drawdown (%)
        test_win_rate: Test win rate
        test_total_trades: Test total trades
        train_sharpe: Train Sharpe ratio
        train_cagr: Train CAGR (%)
        param_values: Parameter values (dict)
    """

    run_index: int
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    test_win_rate: float
    test_total_trades: int
    train_sharpe: float
    train_cagr: float
    param_values: dict


@dataclass
class ComparisonResult:
    """
    Result of comparing two experiment runs.

    Attributes:
        run_a: Summary of first run (higher Sharpe)
        run_b: Summary of second run (lower Sharpe)
        sharpe_diff: Difference in test Sharpe (A - B)
        z_score: Approximate z-score for Sharpe difference
        is_significant: True if difference is significant (|z| > 1.96)
        confidence_level: Confidence level (0.95 for z > 1.96)
    """

    run_a: RunSummary
    run_b: RunSummary
    sharpe_diff: float
    z_score: float
    is_significant: bool
    confidence_level: float


def approximate_sharpe_z_score(
    sharpe_a: float,
    sharpe_b: float,
    n_obs: int,
) -> float:
    """
    Calculate approximate z-score for difference between two Sharpe ratios.

    Uses simplified assumption that standard error is approximately sqrt(2/n).
    This is a quick first approximation - for rigorous testing, use
    more sophisticated methods like Jobson-Korkie test.

    Args:
        sharpe_a: First Sharpe ratio
        sharpe_b: Second Sharpe ratio
        n_obs: Number of observations (e.g., 252*3 for 3 years of daily data)

    Returns:
        Z-score for difference (sharpe_a - sharpe_b)

    Example:
        >>> # Compare two strategies over 3 years
        >>> z = approximate_sharpe_z_score(1.5, 1.2, n_obs=252*3)
        >>> print(f"Z-score: {z:.2f}")
        >>> if abs(z) > 1.96:
        ...     print("Difference is significant at 95% confidence")
    """
    if n_obs <= 0:
        return 0.0

    # Difference in Sharpe ratios
    diff = sharpe_a - sharpe_b

    # Approximate standard error (simplified)
    # For independent Sharpe ratios: SE ≈ sqrt(2/n)
    se = math.sqrt(2.0 / n_obs)

    # Z-score
    if se > 0:
        z = diff / se
    else:
        z = 0.0

    return z


def summarize_runs(runs: List[ExperimentRun]) -> List[RunSummary]:
    """
    Summarize experiment runs for comparison.

    Args:
        runs: List of ExperimentRun objects

    Returns:
        List of RunSummary objects, sorted by test_sharpe descending

    Example:
        >>> summaries = summarize_runs(experiment.runs)
        >>> for s in summaries[:5]:
        ...     print(f"Run {s.run_index}: Sharpe={s.test_sharpe:.2f}")
    """
    import json

    summaries = []

    for run in runs:
        if run.status != "done":
            continue

        # Parse parameter values
        try:
            params = json.loads(run.param_values_json)
        except Exception:
            params = {}

        summary = RunSummary(
            run_index=run.run_index,
            test_sharpe=run.test_sharpe or 0.0,
            test_cagr=run.test_cagr or 0.0,
            test_max_dd=run.test_max_dd or 0.0,
            test_win_rate=run.test_win_rate or 0.0,
            test_total_trades=run.test_total_trades or 0,
            train_sharpe=run.train_sharpe or 0.0,
            train_cagr=run.train_cagr or 0.0,
            param_values=params,
        )
        summaries.append(summary)

    # Sort by test Sharpe descending
    summaries.sort(key=lambda s: s.test_sharpe, reverse=True)

    return summaries


def compare_top_two(
    runs: List[ExperimentRun],
    n_obs: int = 252 * 3,
) -> Optional[ComparisonResult]:
    """
    Compare top two runs by test Sharpe ratio.

    Args:
        runs: List of ExperimentRun objects
        n_obs: Number of observations (default: 3 years of daily data = 252*3)

    Returns:
        ComparisonResult if at least 2 successful runs exist, else None

    Example:
        >>> result = compare_top_two(experiment.runs, n_obs=252*3)
        >>> if result:
        ...     print(f"Best: Run #{result.run_a.run_index}, Sharpe={result.run_a.test_sharpe:.2f}")
        ...     print(f"2nd:  Run #{result.run_b.run_index}, Sharpe={result.run_b.test_sharpe:.2f}")
        ...     print(f"Difference: {result.sharpe_diff:.2f} (z={result.z_score:.2f})")
        ...     if result.is_significant:
        ...         print("✓ Significant at 95% confidence")
        ...     else:
        ...         print("✗ Not significant")
    """
    # Summarize and filter successful runs
    summaries = summarize_runs(runs)

    if len(summaries) < 2:
        return None

    # Top 2 runs
    run_a = summaries[0]  # Best
    run_b = summaries[1]  # Second best

    # Calculate z-score
    z = approximate_sharpe_z_score(
        sharpe_a=run_a.test_sharpe,
        sharpe_b=run_b.test_sharpe,
        n_obs=n_obs,
    )

    # Check significance at 95% confidence (z > 1.96)
    is_significant = abs(z) > 1.96
    confidence_level = 0.95 if is_significant else 0.0

    return ComparisonResult(
        run_a=run_a,
        run_b=run_b,
        sharpe_diff=run_a.test_sharpe - run_b.test_sharpe,
        z_score=z,
        is_significant=is_significant,
        confidence_level=confidence_level,
    )


def compare_all_pairs(
    runs: List[ExperimentRun],
    n_obs: int = 252 * 3,
) -> List[ComparisonResult]:
    """
    Compare all pairs of successful runs.

    Args:
        runs: List of ExperimentRun objects
        n_obs: Number of observations

    Returns:
        List of ComparisonResult objects for all pairs

    Example:
        >>> results = compare_all_pairs(experiment.runs)
        >>> significant = [r for r in results if r.is_significant]
        >>> print(f"Found {len(significant)} significant differences out of {len(results)} pairs")
    """
    summaries = summarize_runs(runs)
    results = []

    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            run_a = summaries[i]
            run_b = summaries[j]

            z = approximate_sharpe_z_score(
                sharpe_a=run_a.test_sharpe,
                sharpe_b=run_b.test_sharpe,
                n_obs=n_obs,
            )

            is_significant = abs(z) > 1.96
            confidence_level = 0.95 if is_significant else 0.0

            result = ComparisonResult(
                run_a=run_a,
                run_b=run_b,
                sharpe_diff=run_a.test_sharpe - run_b.test_sharpe,
                z_score=z,
                is_significant=is_significant,
                confidence_level=confidence_level,
            )
            results.append(result)

    return results


def find_pareto_optimal(
    runs: List[ExperimentRun],
    metric_x: str = "test_sharpe",
    metric_y: str = "test_max_dd",
) -> List[RunSummary]:
    """
    Find Pareto-optimal runs (non-dominated solutions).

    A run is Pareto-optimal if no other run is better on all metrics.

    Args:
        runs: List of ExperimentRun objects
        metric_x: First metric to consider (default: test_sharpe, higher is better)
        metric_y: Second metric to consider (default: test_max_dd, higher is better for DD)

    Returns:
        List of Pareto-optimal RunSummary objects

    Example:
        >>> # Find runs with best Sharpe vs Max DD tradeoff
        >>> pareto = find_pareto_optimal(experiment.runs, "test_sharpe", "test_max_dd")
        >>> print(f"Found {len(pareto)} Pareto-optimal runs")
        >>> for run in pareto:
        ...     print(f"Run {run.run_index}: Sharpe={run.test_sharpe:.2f}, MaxDD={run.test_max_dd:.2f}%")
    """
    summaries = summarize_runs(runs)

    if not summaries:
        return []

    pareto_optimal = []

    for candidate in summaries:
        is_dominated = False

        # Get candidate metrics
        candidate_x = getattr(candidate, metric_x, 0.0)
        candidate_y = getattr(candidate, metric_y, 0.0)

        # Check if any other run dominates this candidate
        for other in summaries:
            if other.run_index == candidate.run_index:
                continue

            other_x = getattr(other, metric_x, 0.0)
            other_y = getattr(other, metric_y, 0.0)

            # For test_max_dd, less negative is better (so negate comparison)
            if metric_y == "test_max_dd":
                # Other dominates if it has higher Sharpe AND less negative DD
                if other_x >= candidate_x and other_y >= candidate_y:
                    if other_x > candidate_x or other_y > candidate_y:
                        is_dominated = True
                        break
            else:
                # Standard: other dominates if it's >= on both and > on at least one
                if other_x >= candidate_x and other_y >= candidate_y:
                    if other_x > candidate_x or other_y > candidate_y:
                        is_dominated = True
                        break

        if not is_dominated:
            pareto_optimal.append(candidate)

    return pareto_optimal


def check_overfitting(
    run: ExperimentRun,
    threshold: float = 1.5,
) -> tuple[bool, float]:
    """
    Check if a run shows signs of overfitting.

    Args:
        run: ExperimentRun object
        threshold: Train/test Sharpe ratio threshold (default: 1.5)

    Returns:
        Tuple of (is_overfit, train_test_ratio)

    Example:
        >>> is_overfit, ratio = check_overfitting(run)
        >>> if is_overfit:
        ...     print(f"⚠ Overfitting detected: train/test Sharpe = {ratio:.2f}")
    """
    if not run.train_sharpe or not run.test_sharpe:
        return False, 0.0

    if run.test_sharpe <= 0:
        return True, float("inf")

    ratio = run.train_sharpe / run.test_sharpe

    is_overfit = ratio > threshold

    return is_overfit, ratio
