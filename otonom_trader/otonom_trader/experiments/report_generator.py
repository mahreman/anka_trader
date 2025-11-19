"""
Experiment report generator.

Creates HTML and Markdown reports from experiment results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from ..data import Experiment, ExperimentRun
from ..utils import utc_now

logger = logging.getLogger(__name__)


class ExperimentReport:
    """
    Generate experiment analysis reports.

    Creates comprehensive HTML/Markdown reports with:
    - Experiment metadata
    - Top-N results table
    - Overfitting analysis
    - Parameter analysis
    - Success criteria evaluation

    Example:
        >>> with get_session() as session:
        ...     experiment = session.query(Experiment).get(1)
        ...     report = ExperimentReport(experiment)
        ...     report.generate_html("reports/exp_1.html")
    """

    def __init__(self, experiment: Experiment):
        """
        Initialize report generator.

        Args:
            experiment: Experiment instance with runs
        """
        self.experiment = experiment
        self.successful_runs = [
            r for r in experiment.runs
            if r.status == "done" and r.test_sharpe is not None
        ]
        self.failed_runs = [
            r for r in experiment.runs
            if r.status == "failed"
        ]

        # Sort by test Sharpe ratio
        self.successful_runs.sort(key=lambda r: r.test_sharpe or 0, reverse=True)

    def generate_markdown(self, output_path: Optional[str] = None) -> str:
        """
        Generate Markdown report.

        Args:
            output_path: Optional path to save report

        Returns:
            Markdown content

        Example:
            >>> report = ExperimentReport(experiment)
            >>> md = report.generate_markdown("reports/exp_1.md")
        """
        lines = []

        # Header
        lines.append(f"# Experiment Report: {self.experiment.name}")
        lines.append("")
        lines.append(f"**Generated**: {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # Metadata
        lines.append("## Experiment Metadata")
        lines.append("")
        lines.append(f"- **Experiment ID**: {self.experiment.id}")
        lines.append(f"- **Name**: {self.experiment.name}")
        lines.append(f"- **Description**: {self.experiment.description or 'N/A'}")
        lines.append(f"- **Base Strategy**: {self.experiment.base_strategy_name}")
        lines.append(f"- **Parameter Grid**: {self.experiment.param_grid_path}")
        lines.append(f"- **Created**: {self.experiment.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Total Runs**: {len(self.experiment.runs)}")
        lines.append(f"- **Successful Runs**: {len(self.successful_runs)}")
        lines.append(f"- **Failed Runs**: {len(self.failed_runs)}")
        lines.append("")

        # Summary statistics
        if self.successful_runs:
            lines.append("## Summary Statistics")
            lines.append("")
            lines.extend(self._generate_summary_stats())
            lines.append("")

        # Top results
        if self.successful_runs:
            lines.append("## Top 10 Results")
            lines.append("")
            lines.extend(self._generate_top_results_table(top_n=10))
            lines.append("")

        # Overfitting analysis
        if self.successful_runs:
            lines.append("## Overfitting Analysis")
            lines.append("")
            lines.extend(self._generate_overfitting_analysis())
            lines.append("")

        # Parameter analysis
        if self.successful_runs:
            lines.append("## Parameter Analysis")
            lines.append("")
            lines.extend(self._generate_parameter_analysis())
            lines.append("")

        # Best run details
        if self.successful_runs:
            lines.append("## Best Run Details")
            lines.append("")
            lines.extend(self._generate_best_run_details())
            lines.append("")

        # Failed runs
        if self.failed_runs:
            lines.append("## Failed Runs")
            lines.append("")
            lines.append(f"Total failed: {len(self.failed_runs)}")
            lines.append("")

        markdown = "\n".join(lines)

        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(markdown)
            logger.info(f"Markdown report saved to: {output_path}")

        return markdown

    def generate_html(self, output_path: Optional[str] = None) -> str:
        """
        Generate HTML report.

        Args:
            output_path: Optional path to save report

        Returns:
            HTML content

        Example:
            >>> report = ExperimentReport(experiment)
            >>> html = report.generate_html("reports/exp_1.html")
        """
        # Convert markdown to HTML (simplified)
        markdown = self.generate_markdown()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Report: {self.experiment.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: 600;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
"""

        # Add content sections as HTML
        html += f"<h1>Experiment Report: {self.experiment.name}</h1>\n"
        html += f"<p><strong>Generated</strong>: {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>\n"

        # Metadata
        html += "<h2>Experiment Metadata</h2>\n"
        html += "<ul>\n"
        html += f"<li><strong>Experiment ID</strong>: {self.experiment.id}</li>\n"
        html += f"<li><strong>Name</strong>: {self.experiment.name}</li>\n"
        html += f"<li><strong>Description</strong>: {self.experiment.description or 'N/A'}</li>\n"
        html += f"<li><strong>Base Strategy</strong>: {self.experiment.base_strategy_name}</li>\n"
        html += f"<li><strong>Parameter Grid</strong>: {self.experiment.param_grid_path}</li>\n"
        html += f"<li><strong>Created</strong>: {self.experiment.created_at.strftime('%Y-%m-%d %H:%M:%S')}</li>\n"
        html += f"<li><strong>Total Runs</strong>: {len(self.experiment.runs)}</li>\n"
        html += f"<li><strong>Successful Runs</strong>: {len(self.successful_runs)}</li>\n"
        html += f"<li><strong>Failed Runs</strong>: {len(self.failed_runs)}</li>\n"
        html += "</ul>\n"

        # Summary statistics
        if self.successful_runs:
            html += "<h2>Summary Statistics</h2>\n"
            html += self._generate_summary_stats_html()

        # Top results table
        if self.successful_runs:
            html += "<h2>Top 10 Results</h2>\n"
            html += self._generate_top_results_html(top_n=10)

        # Overfitting analysis
        if self.successful_runs:
            html += "<h2>Overfitting Analysis</h2>\n"
            html += self._generate_overfitting_analysis_html()

        # Best run details
        if self.successful_runs:
            html += "<h2>Best Run Details</h2>\n"
            html += self._generate_best_run_html()

        html += """
    </div>
</body>
</html>
"""

        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report saved to: {output_path}")

        return html

    def _generate_summary_stats(self) -> List[str]:
        """Generate summary statistics for Markdown."""
        lines = []

        if not self.successful_runs:
            return ["No successful runs to analyze."]

        test_sharpes = [r.test_sharpe for r in self.successful_runs]
        test_cagrs = [r.test_cagr for r in self.successful_runs if r.test_cagr]
        test_max_dds = [r.test_max_dd for r in self.successful_runs if r.test_max_dd]

        lines.append(f"- **Best Test Sharpe**: {max(test_sharpes):.2f}")
        lines.append(f"- **Average Test Sharpe**: {sum(test_sharpes)/len(test_sharpes):.2f}")
        if test_cagrs:
            lines.append(f"- **Best Test CAGR**: {max(test_cagrs):.1f}%")
        if test_max_dds:
            lines.append(f"- **Best Test MaxDD**: {min(test_max_dds):.1f}%")

        return lines

    def _generate_summary_stats_html(self) -> str:
        """Generate summary statistics as HTML."""
        if not self.successful_runs:
            return "<p>No successful runs to analyze.</p>"

        test_sharpes = [r.test_sharpe for r in self.successful_runs]
        test_cagrs = [r.test_cagr for r in self.successful_runs if r.test_cagr]
        test_max_dds = [r.test_max_dd for r in self.successful_runs if r.test_max_dd]

        html = "<div>\n"
        html += f'<div class="metric"><span class="metric-label">Best Test Sharpe:</span> <span class="metric-value">{max(test_sharpes):.2f}</span></div>\n'
        html += f'<div class="metric"><span class="metric-label">Average Test Sharpe:</span> <span class="metric-value">{sum(test_sharpes)/len(test_sharpes):.2f}</span></div>\n'

        if test_cagrs:
            html += f'<div class="metric"><span class="metric-label">Best Test CAGR:</span> <span class="metric-value">{max(test_cagrs):.1f}%</span></div>\n'
        if test_max_dds:
            html += f'<div class="metric"><span class="metric-label">Best Test MaxDD:</span> <span class="metric-value">{min(test_max_dds):.1f}%</span></div>\n'

        html += "</div>\n"
        return html

    def _generate_top_results_table(self, top_n: int = 10) -> List[str]:
        """Generate top results table for Markdown."""
        lines = []

        lines.append("| Run | Train Sharpe | Test Sharpe | Test CAGR | Test MaxDD | Test Trades |")
        lines.append("|-----|--------------|-------------|-----------|------------|-------------|")

        for run in self.successful_runs[:top_n]:
            lines.append(
                f"| {run.run_index} | "
                f"{run.train_sharpe:.2f} | "
                f"{run.test_sharpe:.2f} | "
                f"{run.test_cagr:.1f}% | "
                f"{run.test_max_dd:.1f}% | "
                f"{run.test_total_trades} |"
            )

        return lines

    def _generate_top_results_html(self, top_n: int = 10) -> str:
        """Generate top results table as HTML."""
        html = "<table>\n"
        html += "<tr><th>Run</th><th>Train Sharpe</th><th>Test Sharpe</th><th>Test CAGR</th><th>Test MaxDD</th><th>Test Trades</th></tr>\n"

        for run in self.successful_runs[:top_n]:
            html += f"<tr>"
            html += f"<td>{run.run_index}</td>"
            html += f"<td>{run.train_sharpe:.2f}</td>"
            html += f"<td>{run.test_sharpe:.2f}</td>"
            html += f"<td>{run.test_cagr:.1f}%</td>"
            html += f"<td>{run.test_max_dd:.1f}%</td>"
            html += f"<td>{run.test_total_trades}</td>"
            html += f"</tr>\n"

        html += "</table>\n"
        return html

    def _generate_overfitting_analysis(self) -> List[str]:
        """Generate overfitting analysis for Markdown."""
        lines = []

        overfit_runs = [
            r for r in self.successful_runs
            if r.train_sharpe and r.test_sharpe and (r.train_sharpe / r.test_sharpe) > 1.5
        ]

        if overfit_runs:
            lines.append(f"⚠️ **{len(overfit_runs)} runs show potential overfitting** (train/test Sharpe ratio > 1.5)")
            lines.append("")
            for run in overfit_runs[:5]:
                ratio = run.train_sharpe / run.test_sharpe
                lines.append(f"- Run {run.run_index}: Train Sharpe = {run.train_sharpe:.2f}, Test Sharpe = {run.test_sharpe:.2f}, Ratio = {ratio:.2f}")
        else:
            lines.append("✅ No significant overfitting detected (all runs have train/test Sharpe ratio < 1.5)")

        return lines

    def _generate_overfitting_analysis_html(self) -> str:
        """Generate overfitting analysis as HTML."""
        overfit_runs = [
            r for r in self.successful_runs
            if r.train_sharpe and r.test_sharpe and (r.train_sharpe / r.test_sharpe) > 1.5
        ]

        if overfit_runs:
            html = f'<div class="warning">\n'
            html += f"<strong>⚠️ {len(overfit_runs)} runs show potential overfitting</strong> (train/test Sharpe ratio &gt; 1.5)<br><br>\n"
            html += "<ul>\n"
            for run in overfit_runs[:5]:
                ratio = run.train_sharpe / run.test_sharpe
                html += f"<li>Run {run.run_index}: Train Sharpe = {run.train_sharpe:.2f}, Test Sharpe = {run.test_sharpe:.2f}, Ratio = {ratio:.2f}</li>\n"
            html += "</ul>\n"
            html += "</div>\n"
        else:
            html = '<div class="success">✅ No significant overfitting detected (all runs have train/test Sharpe ratio &lt; 1.5)</div>\n'

        return html

    def _generate_parameter_analysis(self) -> List[str]:
        """Generate parameter analysis for Markdown."""
        lines = []

        # Get best run parameters
        if self.successful_runs:
            best_run = self.successful_runs[0]
            params = json.loads(best_run.param_values_json)

            lines.append("Best performing parameter values:")
            lines.append("")
            for key, value in sorted(params.items()):
                lines.append(f"- `{key}`: {value}")

        return lines

    def _generate_best_run_details(self) -> List[str]:
        """Generate best run details for Markdown."""
        lines = []

        if self.successful_runs:
            best = self.successful_runs[0]

            lines.append(f"**Run Index**: {best.run_index}")
            lines.append("")
            lines.append("### Performance Metrics")
            lines.append(f"- **Train Sharpe**: {best.train_sharpe:.2f}")
            lines.append(f"- **Train CAGR**: {best.train_cagr:.1f}%" if best.train_cagr else "")
            lines.append(f"- **Test Sharpe**: {best.test_sharpe:.2f}")
            lines.append(f"- **Test CAGR**: {best.test_cagr:.1f}%" if best.test_cagr else "")
            lines.append(f"- **Test Max Drawdown**: {best.test_max_dd:.1f}%" if best.test_max_dd else "")
            lines.append(f"- **Test Total Trades**: {best.test_total_trades}" if best.test_total_trades else "")
            lines.append("")

            lines.append("### Parameters")
            lines.append("```json")
            lines.append(json.dumps(json.loads(best.param_values_json), indent=2))
            lines.append("```")

        return lines

    def _generate_best_run_html(self) -> str:
        """Generate best run details as HTML."""
        if not self.successful_runs:
            return "<p>No successful runs.</p>"

        best = self.successful_runs[0]

        html = f"<h3>Run Index: {best.run_index}</h3>\n"

        html += "<h4>Performance Metrics</h4>\n"
        html += "<ul>\n"
        html += f"<li><strong>Train Sharpe</strong>: {best.train_sharpe:.2f}</li>\n"
        if best.train_cagr:
            html += f"<li><strong>Train CAGR</strong>: {best.train_cagr:.1f}%</li>\n"
        html += f"<li><strong>Test Sharpe</strong>: {best.test_sharpe:.2f}</li>\n"
        if best.test_cagr:
            html += f"<li><strong>Test CAGR</strong>: {best.test_cagr:.1f}%</li>\n"
        if best.test_max_dd:
            html += f"<li><strong>Test Max Drawdown</strong>: {best.test_max_dd:.1f}%</li>\n"
        if best.test_total_trades:
            html += f"<li><strong>Test Total Trades</strong>: {best.test_total_trades}</li>\n"
        html += "</ul>\n"

        html += "<h4>Parameters</h4>\n"
        html += f"<pre><code>{json.dumps(json.loads(best.param_values_json), indent=2)}</code></pre>\n"

        return html


def generate_experiment_report(
    session: Session,
    experiment_id: int,
    output_path: str,
    format: str = "html",
) -> str:
    """
    Generate experiment report.

    Args:
        session: Database session
        experiment_id: Experiment ID
        output_path: Output file path
        format: Report format ("html" or "markdown")

    Returns:
        Path to generated report

    Example:
        >>> with get_session() as session:
        ...     report_path = generate_experiment_report(
        ...         session, experiment_id=1, output_path="reports/exp_1.html"
        ...     )
        ...     print(f"Report saved to: {report_path}")
    """
    from ..data import Experiment

    experiment = session.query(Experiment).filter_by(id=experiment_id).first()

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    report = ExperimentReport(experiment)

    if format == "html":
        report.generate_html(output_path)
    elif format == "markdown":
        report.generate_markdown(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path
