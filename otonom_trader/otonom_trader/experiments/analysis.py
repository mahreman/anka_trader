"""
Experiment analysis and comparison utilities.

Provides tools for analyzing experiment results:
- Ablation analysis (component contribution)
- Robustness analysis (parameter sensitivity)
- Regime-specific performance comparison
- Top-N result extraction and ranking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..data import Experiment, ExperimentRun

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """
    Results from ablation analysis.

    Attributes:
        config_name: Configuration name (e.g., "Tech+News", "Tech only")
        test_sharpe: Test Sharpe ratio
        test_cagr: Test CAGR
        test_max_dd: Test maximum drawdown
        test_win_rate: Test win rate
        total_trades: Total number of trades
        contribution: Marginal contribution (vs baseline)
    """
    config_name: str
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    test_win_rate: float
    total_trades: int
    contribution: Optional[float] = None  # Marginal contribution vs baseline

    def __repr__(self) -> str:
        contrib_str = f" ({self.contribution:+.2f})" if self.contribution else ""
        return (
            f"{self.config_name}: "
            f"Sharpe={self.test_sharpe:.2f}{contrib_str}, "
            f"CAGR={self.test_cagr:.1f}%, "
            f"MaxDD={self.test_max_dd:.1f}%"
        )


@dataclass
class RobustnessResult:
    """
    Results from robustness analysis.

    Attributes:
        param_name: Parameter name
        baseline_value: Baseline parameter value
        baseline_sharpe: Sharpe at baseline
        min_sharpe: Minimum Sharpe across perturbations
        max_sharpe: Maximum Sharpe across perturbations
        std_sharpe: Standard deviation of Sharpe
        sensitivity: Sensitivity score (higher = more sensitive)
    """
    param_name: str
    baseline_value: float
    baseline_sharpe: float
    min_sharpe: float
    max_sharpe: float
    std_sharpe: float
    sensitivity: float  # (max - min) / baseline

    def is_robust(self, threshold: float = 0.3) -> bool:
        """Check if parameter is robust (sensitivity below threshold)."""
        return self.sensitivity < threshold

    def __repr__(self) -> str:
        status = "ROBUST" if self.is_robust() else "FRAGILE"
        return (
            f"{self.param_name}: "
            f"Baseline={self.baseline_value}, "
            f"Sharpe range=[{self.min_sharpe:.2f}, {self.max_sharpe:.2f}], "
            f"Sensitivity={self.sensitivity:.2f} ({status})"
        )


@dataclass
class RegimePerformance:
    """
    Performance in a specific market regime.

    Attributes:
        regime_name: Regime/scenario name
        period: Date range
        sharpe: Sharpe ratio
        cagr: CAGR
        max_dd: Maximum drawdown
        win_rate: Win rate
        total_trades: Total trades
        notes: Additional notes
    """
    regime_name: str
    period: str
    sharpe: float
    cagr: float
    max_dd: float
    win_rate: float
    total_trades: int
    notes: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"{self.regime_name} ({self.period}): "
            f"Sharpe={self.sharpe:.2f}, "
            f"CAGR={self.cagr:.1f}%, "
            f"MaxDD={self.max_dd:.1f}%"
        )


def extract_top_runs(
    experiment: Experiment,
    metric: str = "test_sharpe",
    top_n: int = 5,
    min_metric: Optional[float] = None,
) -> List[ExperimentRun]:
    """
    Extract top N runs from an experiment.

    Args:
        experiment: Experiment object
        metric: Metric to rank by (e.g., "test_sharpe", "test_cagr")
        top_n: Number of top runs to return
        min_metric: Minimum metric value (filter out poor runs)

    Returns:
        List of top ExperimentRun objects

    Example:
        >>> top_runs = extract_top_runs(experiment, metric="test_sharpe", top_n=5)
        >>> for run in top_runs:
        ...     print(f"Run {run.run_index}: Sharpe={run.test_sharpe:.2f}")
    """
    # Filter successful runs
    successful_runs = [r for r in experiment.runs if r.status == "done"]

    if not successful_runs:
        logger.warning(f"No successful runs in experiment {experiment.id}")
        return []

    # Filter by minimum metric
    if min_metric is not None:
        successful_runs = [
            r for r in successful_runs
            if getattr(r, metric, None) and getattr(r, metric) >= min_metric
        ]

    # Sort by metric
    sorted_runs = sorted(
        successful_runs,
        key=lambda r: getattr(r, metric, float('-inf')),
        reverse=True
    )

    return sorted_runs[:top_n]


def analyze_ablation(
    session: Session,
    experiment_id: int,
    baseline_config: Optional[Dict[str, float]] = None,
) -> List[AblationResult]:
    """
    Analyze ablation experiment results.

    Compares different analyst combinations to understand contribution.

    Args:
        session: Database session
        experiment_id: Ablation experiment ID
        baseline_config: Baseline configuration (all analysts ON)
                        Default: {"analist_1.weight": 1.0, "analist_2.weight": 1.0, "analist_3.weight": 1.0}

    Returns:
        List of AblationResult objects

    Example:
        >>> results = analyze_ablation(session, experiment_id=5)
        >>> for result in results:
        ...     print(result)
        All Analysts: Sharpe=1.50, CAGR=25.0%, MaxDD=-12.0%
        Tech only: Sharpe=1.20 (-0.30), CAGR=18.0%, MaxDD=-15.0%
    """
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Default baseline: all analysts ON
    if baseline_config is None:
        baseline_config = {
            "analist_1.weight": 1.0,
            "analist_2.weight": 1.0,
            "analist_3.weight": 1.0,
        }

    # Extract all runs
    successful_runs = [r for r in experiment.runs if r.status == "done"]

    if not successful_runs:
        logger.warning(f"No successful runs in experiment {experiment_id}")
        return []

    # Group runs by analyst configuration
    configs = {}

    for run in successful_runs:
        import json
        params = json.loads(run.param_values_json)

        # Create config name based on which analysts are enabled
        tech_on = params.get("analist_1.weight", 0.0) > 0.0
        news_on = params.get("analist_2.weight", 0.0) > 0.0
        risk_on = params.get("analist_3.weight", 0.0) > 0.0

        config_name = _get_config_name(tech_on, news_on, risk_on)

        # Store run under config name
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(run)

    # Calculate results for each configuration
    results = []
    baseline_sharpe = None

    # First pass: find baseline
    for config_name, runs in configs.items():
        avg_sharpe = np.mean([r.test_sharpe for r in runs if r.test_sharpe])

        if config_name == "All Analysts":
            baseline_sharpe = avg_sharpe

    # Second pass: calculate results with contribution
    for config_name, runs in configs.items():
        avg_sharpe = np.mean([r.test_sharpe for r in runs if r.test_sharpe])
        avg_cagr = np.mean([r.test_cagr for r in runs if r.test_cagr])
        avg_max_dd = np.mean([r.test_max_dd for r in runs if r.test_max_dd])
        avg_win_rate = np.mean([r.test_win_rate for r in runs if r.test_win_rate]) if any(r.test_win_rate for r in runs) else 0.0
        total_trades = int(np.mean([r.test_total_trades for r in runs if r.test_total_trades])) if any(r.test_total_trades for r in runs) else 0

        # Calculate contribution (vs baseline)
        contribution = None
        if baseline_sharpe is not None and config_name != "All Analysts":
            contribution = avg_sharpe - baseline_sharpe

        results.append(AblationResult(
            config_name=config_name,
            test_sharpe=avg_sharpe,
            test_cagr=avg_cagr,
            test_max_dd=avg_max_dd,
            test_win_rate=avg_win_rate,
            total_trades=total_trades,
            contribution=contribution,
        ))

    # Sort: baseline first, then by Sharpe descending
    results.sort(key=lambda r: (r.config_name != "All Analysts", -r.test_sharpe))

    return results


def analyze_robustness(
    session: Session,
    experiment_id: int,
    baseline_params: Dict[str, float],
) -> List[RobustnessResult]:
    """
    Analyze robustness/sensitivity of parameters.

    Measures how sensitive performance is to parameter perturbations.

    Args:
        session: Database session
        experiment_id: Robustness experiment ID
        baseline_params: Baseline parameter values
                        Example: {"risk_management.position_sizing.risk_per_trade_pct": 1.0,
                                 "risk_management.stop_loss.percentage": 5.0}

    Returns:
        List of RobustnessResult objects

    Example:
        >>> baseline = {
        ...     "risk_management.position_sizing.risk_per_trade_pct": 1.0,
        ...     "risk_management.stop_loss.percentage": 5.0,
        ... }
        >>> results = analyze_robustness(session, experiment_id=7, baseline_params=baseline)
        >>> for result in results:
        ...     print(result)
        risk_per_trade_pct: Baseline=1.0, Sharpe range=[1.2, 1.8], Sensitivity=0.4 (FRAGILE)
    """
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Extract successful runs
    successful_runs = [r for r in experiment.runs if r.status == "done"]

    if not successful_runs:
        logger.warning(f"No successful runs in experiment {experiment_id}")
        return []

    # Analyze each parameter
    results = []

    for param_name, baseline_value in baseline_params.items():
        import json

        # Find runs with this parameter
        param_sharpes = []
        baseline_sharpe = None

        for run in successful_runs:
            params = json.loads(run.param_values_json)

            if param_name in params and run.test_sharpe:
                param_sharpes.append((params[param_name], run.test_sharpe))

                # Check if this is baseline
                if abs(params[param_name] - baseline_value) < 0.01:
                    baseline_sharpe = run.test_sharpe

        if not param_sharpes:
            continue

        # Calculate statistics
        sharpes = [s for _, s in param_sharpes]
        min_sharpe = min(sharpes)
        max_sharpe = max(sharpes)
        std_sharpe = np.std(sharpes)

        # Use average if no exact baseline match
        if baseline_sharpe is None:
            baseline_sharpe = np.mean(sharpes)

        # Sensitivity: relative range
        sensitivity = (max_sharpe - min_sharpe) / baseline_sharpe if baseline_sharpe != 0 else 0.0

        results.append(RobustnessResult(
            param_name=param_name,
            baseline_value=baseline_value,
            baseline_sharpe=baseline_sharpe,
            min_sharpe=min_sharpe,
            max_sharpe=max_sharpe,
            std_sharpe=std_sharpe,
            sensitivity=sensitivity,
        ))

    # Sort by sensitivity (most sensitive first)
    results.sort(key=lambda r: r.sensitivity, reverse=True)

    return results


def compare_experiments(
    session: Session,
    experiment_ids: List[int],
    metric: str = "test_sharpe",
) -> pd.DataFrame:
    """
    Compare multiple experiments.

    Args:
        session: Database session
        experiment_ids: List of experiment IDs to compare
        metric: Metric to compare (e.g., "test_sharpe")

    Returns:
        DataFrame with comparison

    Example:
        >>> df = compare_experiments(session, [1, 2, 3], metric="test_sharpe")
        >>> print(df)
          experiment_id  experiment_name  best_sharpe  avg_sharpe  worst_sharpe
        0             1  param_sweep_v1         2.1         1.5           0.8
        1             2  ablation_v1            1.8         1.3           0.5
        2             3  robustness_v1          1.9         1.6           1.2
    """
    comparison_data = []

    for exp_id in experiment_ids:
        experiment = session.query(Experiment).filter_by(id=exp_id).first()

        if not experiment:
            logger.warning(f"Experiment {exp_id} not found")
            continue

        successful_runs = [r for r in experiment.runs if r.status == "done"]

        if not successful_runs:
            continue

        metric_values = [getattr(r, metric, None) for r in successful_runs if getattr(r, metric, None)]

        if not metric_values:
            continue

        comparison_data.append({
            "experiment_id": exp_id,
            "experiment_name": experiment.name,
            f"best_{metric}": max(metric_values),
            f"avg_{metric}": np.mean(metric_values),
            f"worst_{metric}": min(metric_values),
            f"std_{metric}": np.std(metric_values),
            "total_runs": len(successful_runs),
        })

    return pd.DataFrame(comparison_data)


def _get_config_name(tech_on: bool, news_on: bool, risk_on: bool) -> str:
    """Get human-readable config name from analyst flags."""
    if tech_on and news_on and risk_on:
        return "All Analysts"
    elif tech_on and news_on:
        return "Tech + News"
    elif tech_on and risk_on:
        return "Tech + Risk"
    elif news_on and risk_on:
        return "News + Risk"
    elif tech_on:
        return "Tech only"
    elif news_on:
        return "News only"
    elif risk_on:
        return "Risk only"
    else:
        return "None (baseline)"


def generate_comparison_table(
    results: List[AblationResult] | List[RobustnessResult] | List[RegimePerformance],
    output_format: str = "markdown",
) -> str:
    """
    Generate comparison table in Markdown or HTML format.

    Args:
        results: List of result objects
        output_format: "markdown" or "html"

    Returns:
        Formatted table as string

    Example:
        >>> ablation_results = analyze_ablation(session, 5)
        >>> table = generate_comparison_table(ablation_results, output_format="markdown")
        >>> print(table)
    """
    if not results:
        return "No results to display."

    # Determine result type
    if isinstance(results[0], AblationResult):
        return _generate_ablation_table(results, output_format)
    elif isinstance(results[0], RobustnessResult):
        return _generate_robustness_table(results, output_format)
    elif isinstance(results[0], RegimePerformance):
        return _generate_regime_table(results, output_format)
    else:
        raise ValueError(f"Unknown result type: {type(results[0])}")


def _generate_ablation_table(results: List[AblationResult], output_format: str) -> str:
    """Generate ablation comparison table."""
    if output_format == "markdown":
        lines = [
            "# Ablation Analysis Results",
            "",
            "| Configuration | Sharpe | CAGR | Max DD | Win Rate | Trades | Contribution |",
            "|---------------|--------|------|--------|----------|--------|--------------|",
        ]

        for r in results:
            contrib_str = f"{r.contribution:+.2f}" if r.contribution else "baseline"
            lines.append(
                f"| {r.config_name:<13} | "
                f"{r.test_sharpe:>6.2f} | "
                f"{r.test_cagr:>4.1f}% | "
                f"{r.test_max_dd:>6.1f}% | "
                f"{r.test_win_rate:>7.1f}% | "
                f"{r.total_trades:>6d} | "
                f"{contrib_str:>12} |"
            )

        return "\n".join(lines)

    else:  # HTML
        html = """
<table>
<thead>
    <tr>
        <th>Configuration</th>
        <th>Sharpe</th>
        <th>CAGR</th>
        <th>Max DD</th>
        <th>Win Rate</th>
        <th>Trades</th>
        <th>Contribution</th>
    </tr>
</thead>
<tbody>
"""
        for r in results:
            contrib_str = f"{r.contribution:+.2f}" if r.contribution else "baseline"
            color = "green" if r.contribution and r.contribution > 0 else ("red" if r.contribution and r.contribution < 0 else "black")

            html += f"""
    <tr>
        <td>{r.config_name}</td>
        <td>{r.test_sharpe:.2f}</td>
        <td>{r.test_cagr:.1f}%</td>
        <td>{r.test_max_dd:.1f}%</td>
        <td>{r.test_win_rate:.1f}%</td>
        <td>{r.total_trades}</td>
        <td style="color: {color};">{contrib_str}</td>
    </tr>
"""

        html += """
</tbody>
</table>
"""
        return html


def _generate_robustness_table(results: List[RobustnessResult], output_format: str) -> str:
    """Generate robustness analysis table."""
    if output_format == "markdown":
        lines = [
            "# Robustness Analysis Results",
            "",
            "| Parameter | Baseline | Sharpe Range | Std Dev | Sensitivity | Status |",
            "|-----------|----------|--------------|---------|-------------|--------|",
        ]

        for r in results:
            status = "✓ Robust" if r.is_robust() else "⚠ Fragile"
            lines.append(
                f"| {r.param_name:<25} | "
                f"{r.baseline_value:>8.2f} | "
                f"[{r.min_sharpe:.2f}, {r.max_sharpe:.2f}] | "
                f"{r.std_sharpe:>7.2f} | "
                f"{r.sensitivity:>11.2f} | "
                f"{status} |"
            )

        return "\n".join(lines)

    else:  # HTML
        html = """
<table>
<thead>
    <tr>
        <th>Parameter</th>
        <th>Baseline</th>
        <th>Sharpe Range</th>
        <th>Std Dev</th>
        <th>Sensitivity</th>
        <th>Status</th>
    </tr>
</thead>
<tbody>
"""
        for r in results:
            status = "✓ Robust" if r.is_robust() else "⚠ Fragile"
            status_color = "green" if r.is_robust() else "orange"

            html += f"""
    <tr>
        <td>{r.param_name}</td>
        <td>{r.baseline_value:.2f}</td>
        <td>[{r.min_sharpe:.2f}, {r.max_sharpe:.2f}]</td>
        <td>{r.std_sharpe:.2f}</td>
        <td>{r.sensitivity:.2f}</td>
        <td style="color: {status_color};">{status}</td>
    </tr>
"""

        html += """
</tbody>
</table>
"""
        return html


def _generate_regime_table(results: List[RegimePerformance], output_format: str) -> str:
    """Generate regime performance table."""
    if output_format == "markdown":
        lines = [
            "# Regime Performance Analysis",
            "",
            "| Regime | Period | Sharpe | CAGR | Max DD | Win Rate | Trades | Notes |",
            "|--------|--------|--------|------|--------|----------|--------|-------|",
        ]

        for r in results:
            notes = r.notes or ""
            lines.append(
                f"| {r.regime_name:<20} | "
                f"{r.period:<15} | "
                f"{r.sharpe:>6.2f} | "
                f"{r.cagr:>5.1f}% | "
                f"{r.max_dd:>6.1f}% | "
                f"{r.win_rate:>7.1f}% | "
                f"{r.total_trades:>6d} | "
                f"{notes} |"
            )

        return "\n".join(lines)

    else:  # HTML
        html = """
<table>
<thead>
    <tr>
        <th>Regime</th>
        <th>Period</th>
        <th>Sharpe</th>
        <th>CAGR</th>
        <th>Max DD</th>
        <th>Win Rate</th>
        <th>Trades</th>
        <th>Notes</th>
    </tr>
</thead>
<tbody>
"""
        for r in results:
            notes = r.notes or ""
            sharpe_color = "green" if r.sharpe > 1.0 else ("red" if r.sharpe < 0 else "black")

            html += f"""
    <tr>
        <td>{r.regime_name}</td>
        <td>{r.period}</td>
        <td style="color: {sharpe_color};">{r.sharpe:.2f}</td>
        <td>{r.cagr:.1f}%</td>
        <td>{r.max_dd:.1f}%</td>
        <td>{r.win_rate:.1f}%</td>
        <td>{r.total_trades}</td>
        <td>{notes}</td>
    </tr>
"""

        html += """
</tbody>
</table>
"""
        return html
