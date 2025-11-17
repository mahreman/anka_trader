#!/usr/bin/env python
"""
Research Backtest Runner

Runs comprehensive backtests across all symbols and generates performance reports.

Usage:
    python scripts/run_research_backtests.py --strategy baseline_v1
    python scripts/run_research_backtests.py --strategy baseline_v1 --symbols BTC-USD ETH-USD
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from otonom_trader.config import load_strategy
from otonom_trader.data import get_session
from otonom_trader.eval.portfolio_backtest import run_backtest
from otonom_trader.eval.performance_report import (
    calculate_metrics,
    generate_html_report,
    save_results_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_single_backtest(
    session,
    symbol: str,
    strategy_config,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Run backtest for a single symbol.

    Args:
        session: Database session
        symbol: Asset symbol
        strategy_config: Strategy configuration
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary with backtest results and metrics
    """
    logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")

    try:
        # Run backtest
        result = run_backtest(
            session=session,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_cash=strategy_config.get_initial_capital(),
            risk_per_trade=strategy_config.get_risk_per_trade_pct() / 100,
            use_ensemble=strategy_config.get("ensemble.enabled", True),
        )

        if not result or not result.get("equity_curve"):
            logger.warning(f"No results for {symbol}")
            return None

        # Calculate metrics
        metrics = calculate_metrics(
            equity_curve=result["equity_curve"],
            trades=result.get("trades", []),
            initial_capital=strategy_config.get_initial_capital(),
        )

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "result": result,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run research backtests")
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline_v1",
        help="Strategy name (YAML file in strategies/)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Symbols to backtest (default: all from strategy config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Load strategy
    strategy_path = Path(f"strategies/{args.strategy}.yaml")
    if not strategy_path.exists():
        logger.error(f"Strategy file not found: {strategy_path}")
        return 1

    logger.info(f"Loading strategy: {strategy_path}")
    strategy_config = load_strategy(strategy_path)

    logger.info(f"Strategy: {strategy_config.name} v{strategy_config.version}")
    logger.info(f"Description: {strategy_config.description}")

    # Get symbols
    symbols = args.symbols if args.symbols else strategy_config.get_symbols()

    if not symbols:
        logger.error("No symbols specified")
        return 1

    logger.info(f"Backtesting {len(symbols)} symbols: {symbols}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run backtests
    all_results = []

    with get_session() as session:
        for symbol in symbols:
            # Determine asset class (simple heuristic)
            if "BTC" in symbol or "ETH" in symbol or "BNB" in symbol:
                asset_class = "crypto"
            else:
                asset_class = "traditional"

            start_date = strategy_config.get_backtest_start_date(asset_class)
            end_date = strategy_config.get_backtest_end_date(asset_class)

            result = run_single_backtest(
                session=session,
                symbol=symbol,
                strategy_config=strategy_config,
                start_date=start_date,
                end_date=end_date,
            )

            if result:
                all_results.append(result)

    if not all_results:
        logger.error("No successful backtests")
        return 1

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV report
    csv_path = output_dir / f"{strategy_config.name}_{timestamp}.csv"
    save_results_csv(all_results, csv_path)
    logger.info(f"Saved CSV report: {csv_path}")

    # Generate HTML report
    html_path = output_dir / f"{strategy_config.name}_{timestamp}.html"
    generate_html_report(
        results=all_results,
        strategy_config=strategy_config,
        output_path=html_path,
    )
    logger.info(f"Saved HTML report: {html_path}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"BACKTEST SUMMARY - {strategy_config.name} v{strategy_config.version}")
    print("=" * 80)
    print(f"{'Symbol':<12} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<10} {'Trades':<10}")
    print("-" * 80)

    for res in all_results:
        metrics = res["metrics"]
        print(
            f"{res['symbol']:<12} "
            f"{metrics.get('cagr', 0):>9.2f}% "
            f"{metrics.get('sharpe_ratio', 0):>9.2f} "
            f"{metrics.get('max_drawdown', 0):>9.2f}% "
            f"{metrics.get('win_rate', 0):>9.1f}% "
            f"{metrics.get('total_trades', 0):>9d}"
        )

    print("=" * 80)

    logger.info("Backtest complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
