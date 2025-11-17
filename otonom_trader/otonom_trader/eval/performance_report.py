"""
Performance metrics calculation and report generation.

Calculates standard trading performance metrics:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Average Win/Loss Ratio
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years

    Returns:
        CAGR as percentage

    Example:
        >>> calculate_cagr(100000, 150000, 2.0)
        22.47  # 22.47% CAGR
    """
    if initial_value <= 0 or years <= 0:
        return 0.0

    cagr = (pow(final_value / initial_value, 1 / years) - 1) * 100
    return cagr


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)

    Returns:
        Sharpe ratio (annualized)

    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015])
        >>> calculate_sharpe_ratio(returns)
        1.23
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(252)  # Annualize
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation).

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)
    mean_excess = excess_returns.mean()

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = (mean_excess / downside_std) * np.sqrt(252)
    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as percentage (negative value)

    Example:
        >>> equity = pd.Series([100, 110, 105, 95, 100])
        >>> calculate_max_drawdown(equity)
        -13.64  # Max DD from peak 110 to trough 95
    """
    if len(equity_curve) == 0:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max * 100

    max_dd = drawdown.min()
    return max_dd


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate.

    Args:
        trades: List of trade dictionaries with 'pnl' key

    Returns:
        Win rate as percentage

    Example:
        >>> trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 75}, {"pnl": -25}]
        >>> calculate_win_rate(trades)
        50.0  # 2 wins out of 4 trades
    """
    if not trades:
        return 0.0

    winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
    win_rate = (winning_trades / len(trades)) * 100

    return win_rate


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade dictionaries with 'pnl' key

    Returns:
        Profit factor

    Example:
        >>> trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 75}]
        >>> calculate_profit_factor(trades)
        3.5  # (100 + 75) / 50
    """
    if not trades:
        return 0.0

    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return 0.0

    return gross_profit / gross_loss


def calculate_avg_win_loss_ratio(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate average win / average loss ratio.

    Args:
        trades: List of trade dictionaries with 'pnl' key

    Returns:
        Average win/loss ratio
    """
    if not trades:
        return 0.0

    wins = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
    losses = [abs(t.get("pnl", 0)) for t in trades if t.get("pnl", 0) < 0]

    if not wins or not losses:
        return 0.0

    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 0.0

    return avg_win / avg_loss


def calculate_metrics(
    equity_curve: pd.Series | List[float],
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        equity_curve: Portfolio equity over time
        trades: List of executed trades
        initial_capital: Starting capital

    Returns:
        Dictionary of metrics

    Example:
        >>> equity = pd.Series([100000, 105000, 110000, 108000])
        >>> trades = [{"pnl": 5000}, {"pnl": 5000}, {"pnl": -2000}]
        >>> metrics = calculate_metrics(equity, trades, 100000)
        >>> print(metrics["cagr"])
    """
    # Convert to pandas Series if needed
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)

    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Time period (assuming daily data)
    days = len(equity_curve)
    years = days / 252.0 if days > 0 else 0

    final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital

    # Calculate metrics
    metrics = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": ((final_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0,
        "cagr": calculate_cagr(initial_capital, final_value, years) if years > 0 else 0,
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "sortino_ratio": calculate_sortino_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "total_trades": len(trades),
        "win_rate": calculate_win_rate(trades),
        "profit_factor": calculate_profit_factor(trades),
        "avg_win_loss_ratio": calculate_avg_win_loss_ratio(trades),
        "days": days,
        "years": years,
    }

    return metrics


def save_results_csv(results: List[Dict[str, Any]], output_path: str | Path):
    """
    Save backtest results to CSV.

    Args:
        results: List of backtest results
        output_path: Output CSV path
    """
    rows = []

    for res in results:
        metrics = res["metrics"]
        row = {
            "Symbol": res["symbol"],
            "Start Date": res["start_date"],
            "End Date": res["end_date"],
            "Initial Capital": f"${metrics['initial_capital']:,.2f}",
            "Final Value": f"${metrics['final_value']:,.2f}",
            "Total Return": f"{metrics['total_return']:.2f}%",
            "CAGR": f"{metrics['cagr']:.2f}%",
            "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
            "Sortino Ratio": f"{metrics['sortino_ratio']:.2f}",
            "Max Drawdown": f"{metrics['max_drawdown']:.2f}%",
            "Total Trades": metrics['total_trades'],
            "Win Rate": f"{metrics['win_rate']:.1f}%",
            "Profit Factor": f"{metrics['profit_factor']:.2f}",
            "Avg Win/Loss": f"{metrics['avg_win_loss_ratio']:.2f}",
            "Days": metrics['days'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved CSV report: {output_path}")


def generate_html_report(
    results: List[Dict[str, Any]],
    strategy_config: Any,
    output_path: str | Path,
):
    """
    Generate HTML performance report.

    Args:
        results: List of backtest results
        strategy_config: Strategy configuration
        output_path: Output HTML path
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Backtest Report - {strategy_config.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .metric-negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .info-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Performance Report</h1>

        <div class="info-box">
            <p><strong>Strategy:</strong> {strategy_config.name} v{strategy_config.version}</p>
            <p><strong>Description:</strong> {strategy_config.description}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>

        <h2>Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Period</th>
                    <th>CAGR</th>
                    <th>Sharpe</th>
                    <th>Sortino</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add rows
    for res in results:
        metrics = res["metrics"]
        cagr = metrics["cagr"]
        max_dd = metrics["max_drawdown"]

        cagr_class = "metric-positive" if cagr > 0 else "metric-negative"
        dd_class = "metric-negative"

        html += f"""
                <tr>
                    <td><strong>{res["symbol"]}</strong></td>
                    <td>{res["start_date"]} to {res["end_date"]}</td>
                    <td class="{cagr_class}">{cagr:.2f}%</td>
                    <td>{metrics["sharpe_ratio"]:.2f}</td>
                    <td>{metrics["sortino_ratio"]:.2f}</td>
                    <td class="{dd_class}">{max_dd:.2f}%</td>
                    <td>{metrics["win_rate"]:.1f}%</td>
                    <td>{metrics["total_trades"]}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by Otonom Trader Research Backtest Runner</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Saved HTML report: {output_path}")
