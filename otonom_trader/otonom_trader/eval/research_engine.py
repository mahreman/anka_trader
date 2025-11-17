"""
Research Engine for comprehensive backtesting.

Transforms the backtest runner into a research engine that:
1. Supports different backtest types (Smoke/Full/Scenario)
2. Generates comprehensive reports with metrics, equity curves, trade lists
3. Provides regime-based performance breakdown
4. Calculates R-multiples and advanced risk metrics

Usage:
    from otonom_trader.eval.research_engine import (
        ResearchEngine,
        BacktestType,
        run_research_backtest
    )

    # Run a smoke test
    report = run_research_backtest(
        session=session,
        strategy_path="strategies/baseline_v1.yaml",
        backtest_type=BacktestType.SMOKE,
    )

    # Run a full historical backtest
    report = run_research_backtest(
        session=session,
        strategy_path="strategies/baseline_v1.yaml",
        backtest_type=BacktestType.FULL,
        symbols=["BTC-USD", "ETH-USD"],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..config import load_strategy, StrategyConfig
from ..data.schema import Regime as RegimeORM
from .portfolio_backtest import run_backtest
from .performance_report import calculate_metrics

logger = logging.getLogger(__name__)


class BacktestType(str, Enum):
    """
    Types of backtests supported by the research engine.

    Attributes:
        SMOKE: Quick smoke test (small date range, 1-2 symbols)
        FULL: Full historical backtest (train/val/test splits)
        SCENARIO: Scenario-based backtest (specific market periods)
    """
    SMOKE = "smoke"
    FULL = "full"
    SCENARIO = "scenario"


@dataclass
class BacktestPeriod:
    """
    Defines a backtest period.

    Attributes:
        name: Period name (e.g., "Train", "Test", "Covid Crash")
        start_date: Start date
        end_date: End date
        description: Optional description
    """
    name: str
    start_date: date
    end_date: date
    description: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.name} ({self.start_date} to {self.end_date})"


@dataclass
class TradeRecord:
    """
    Detailed trade record with risk metrics.

    Attributes:
        symbol: Asset symbol
        entry_date: Entry date
        exit_date: Exit date
        direction: Trade direction (BUY/SELL)
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position quantity
        pnl: Profit/loss in currency
        pnl_pct: Profit/loss percentage
        holding_days: Holding period in days
        r_multiple: R-multiple (reward-to-risk ratio)
        regime_id: Regime at entry (if available)
    """
    symbol: str
    entry_date: date
    exit_date: date
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_days: int
    r_multiple: Optional[float] = None
    regime_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
            "r_multiple": self.r_multiple,
            "regime_id": self.regime_id,
        }


@dataclass
class RegimePerformance:
    """
    Performance metrics broken down by regime.

    Attributes:
        regime_id: Regime identifier
        total_trades: Number of trades
        win_rate: Win rate percentage
        avg_r_multiple: Average R-multiple
        total_pnl: Total PnL
        avg_pnl_pct: Average PnL percentage
    """
    regime_id: int
    total_trades: int
    win_rate: float
    avg_r_multiple: float
    total_pnl: float
    avg_pnl_pct: float

    def __repr__(self) -> str:
        return (
            f"Regime {self.regime_id}: "
            f"{self.total_trades} trades, "
            f"{self.win_rate:.1f}% WR, "
            f"{self.avg_r_multiple:.2f}R avg"
        )


@dataclass
class ResearchReport:
    """
    Comprehensive backtest report.

    Attributes:
        strategy_name: Strategy name
        backtest_type: Type of backtest
        period: Backtest period
        symbols: List of symbols tested
        metrics: Performance metrics
        equity_curve: Daily equity values
        equity_dates: Dates for equity curve
        trades: Detailed trade records
        regime_breakdown: Performance by regime
        metadata: Additional metadata
    """
    strategy_name: str
    backtest_type: BacktestType
    period: BacktestPeriod
    symbols: List[str]
    metrics: Dict[str, float]
    equity_curve: List[float]
    equity_dates: List[date]
    trades: List[TradeRecord]
    regime_breakdown: Dict[int, RegimePerformance] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            "=" * 80,
            f"RESEARCH BACKTEST REPORT - {self.strategy_name}",
            "=" * 80,
            f"Type: {self.backtest_type.value.upper()}",
            f"Period: {self.period}",
            f"Symbols: {', '.join(self.symbols)}",
            "",
            "PERFORMANCE METRICS:",
            "-" * 80,
            f"  CAGR:           {self.metrics.get('cagr', 0):>10.2f}%",
            f"  Sharpe Ratio:   {self.metrics.get('sharpe_ratio', 0):>10.2f}",
            f"  Sortino Ratio:  {self.metrics.get('sortino_ratio', 0):>10.2f}",
            f"  Max Drawdown:   {self.metrics.get('max_drawdown', 0):>10.2f}%",
            f"  Win Rate:       {self.metrics.get('win_rate', 0):>10.1f}%",
            f"  Profit Factor:  {self.metrics.get('profit_factor', 0):>10.2f}",
            f"  Total Trades:   {self.metrics.get('total_trades', 0):>10d}",
            "",
        ]

        # Add regime breakdown if available
        if self.regime_breakdown:
            lines.extend([
                "REGIME BREAKDOWN:",
                "-" * 80,
            ])
            for regime_id, perf in sorted(self.regime_breakdown.items()):
                lines.append(
                    f"  Regime {regime_id}: "
                    f"{perf.total_trades:>4d} trades, "
                    f"{perf.win_rate:>5.1f}% WR, "
                    f"{perf.avg_r_multiple:>5.2f}R avg, "
                    f"${perf.total_pnl:>10,.2f} PnL"
                )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "backtest_type": self.backtest_type.value,
            "period": {
                "name": self.period.name,
                "start_date": str(self.period.start_date),
                "end_date": str(self.period.end_date),
                "description": self.period.description,
            },
            "symbols": self.symbols,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "equity_dates": [str(d) for d in self.equity_dates],
            "trades": [t.to_dict() for t in self.trades],
            "regime_breakdown": {
                str(regime_id): {
                    "total_trades": perf.total_trades,
                    "win_rate": perf.win_rate,
                    "avg_r_multiple": perf.avg_r_multiple,
                    "total_pnl": perf.total_pnl,
                    "avg_pnl_pct": perf.avg_pnl_pct,
                }
                for regime_id, perf in self.regime_breakdown.items()
            },
            "metadata": self.metadata,
        }


class ResearchEngine:
    """
    Research engine for running comprehensive backtests.

    Provides different backtest types and generates detailed reports.
    """

    # Predefined scenario periods
    SCENARIOS = {
        "covid_crash": BacktestPeriod(
            name="Covid Crash",
            start_date=date(2020, 2, 1),
            end_date=date(2020, 5, 1),
            description="February-April 2020 market crash"
        ),
        "crypto_bear_2022": BacktestPeriod(
            name="Crypto Bear 2022",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 12, 31),
            description="2022 crypto bear market"
        ),
        "high_inflation": BacktestPeriod(
            name="High Inflation Period",
            start_date=date(2021, 1, 1),
            end_date=date(2023, 12, 31),
            description="2021-2023 high inflation period"
        ),
        "bull_run_2021": BacktestPeriod(
            name="Bull Run 2021",
            start_date=date(2020, 10, 1),
            end_date=date(2021, 11, 30),
            description="Q4 2020 - Q4 2021 bull market"
        ),
    }

    def __init__(self, session: Session):
        """
        Initialize research engine.

        Args:
            session: Database session
        """
        self.session = session

    def get_backtest_config(
        self,
        backtest_type: BacktestType,
        strategy_config: StrategyConfig,
        symbols: Optional[List[str]] = None,
        custom_period: Optional[BacktestPeriod] = None,
        scenario_name: Optional[str] = None,
    ) -> Tuple[List[str], List[BacktestPeriod]]:
        """
        Get backtest configuration based on type.

        Args:
            backtest_type: Type of backtest
            strategy_config: Strategy configuration
            symbols: Optional symbol list (overrides config)
            custom_period: Custom backtest period
            scenario_name: Scenario name for scenario backtests

        Returns:
            Tuple of (symbols, periods)
        """
        # Get symbols
        if symbols:
            symbol_list = symbols
        elif backtest_type == BacktestType.SMOKE:
            # For smoke test, use only first 1-2 symbols
            all_symbols = strategy_config.get_symbols()
            symbol_list = all_symbols[:2] if all_symbols else []
        else:
            symbol_list = strategy_config.get_symbols()

        # Get periods
        if backtest_type == BacktestType.SMOKE:
            # Smoke test: last 30 days
            end = date.today()
            start = end - timedelta(days=30)
            periods = [BacktestPeriod(
                name="Smoke Test",
                start_date=start,
                end_date=end,
                description="30-day smoke test"
            )]

        elif backtest_type == BacktestType.FULL:
            # Full backtest: train/val/test splits
            if custom_period:
                total_start = custom_period.start_date
                total_end = custom_period.end_date
            else:
                # Use default from config
                total_start = datetime.strptime(
                    strategy_config.get_backtest_start_date("crypto"),
                    "%Y-%m-%d"
                ).date()
                total_end = datetime.strptime(
                    strategy_config.get_backtest_end_date("crypto"),
                    "%Y-%m-%d"
                ).date()

            # Split: 60% train, 20% validation, 20% test
            total_days = (total_end - total_start).days
            train_days = int(total_days * 0.6)
            val_days = int(total_days * 0.2)

            train_end = total_start + timedelta(days=train_days)
            val_end = train_end + timedelta(days=val_days)

            periods = [
                BacktestPeriod(
                    name="Train",
                    start_date=total_start,
                    end_date=train_end,
                    description="Training period (60%)"
                ),
                BacktestPeriod(
                    name="Validation",
                    start_date=train_end + timedelta(days=1),
                    end_date=val_end,
                    description="Validation period (20%)"
                ),
                BacktestPeriod(
                    name="Test",
                    start_date=val_end + timedelta(days=1),
                    end_date=total_end,
                    description="Test period (20%)"
                ),
            ]

        elif backtest_type == BacktestType.SCENARIO:
            # Scenario backtest
            if custom_period:
                periods = [custom_period]
            elif scenario_name and scenario_name in self.SCENARIOS:
                periods = [self.SCENARIOS[scenario_name]]
            else:
                # Use all predefined scenarios
                periods = list(self.SCENARIOS.values())

        else:
            raise ValueError(f"Unknown backtest type: {backtest_type}")

        return symbol_list, periods

    def calculate_r_multiple(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        initial_stop_loss_pct: float = 0.02,
    ) -> float:
        """
        Calculate R-multiple for a trade.

        R-multiple is the profit/loss divided by the initial risk.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction (BUY/SELL)
            initial_stop_loss_pct: Initial stop loss percentage

        Returns:
            R-multiple
        """
        # Calculate PnL percentage
        if direction == "BUY":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - exit_price) / entry_price

        # R-multiple = PnL / Initial Risk
        r_multiple = pnl_pct / initial_stop_loss_pct

        return r_multiple

    def get_regime_for_date(
        self,
        symbol: str,
        trade_date: date,
    ) -> Optional[int]:
        """
        Get regime ID for a symbol on a specific date.

        Args:
            symbol: Asset symbol
            trade_date: Date

        Returns:
            Regime ID or None
        """
        from ..data.schema import Symbol

        symbol_obj = self.session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            return None

        regime = (
            self.session.query(RegimeORM)
            .filter(
                RegimeORM.symbol_id == symbol_obj.id,
                RegimeORM.date == trade_date
            )
            .first()
        )

        return regime.regime_id if regime else None

    def enhance_trade_records(
        self,
        trades: List[Dict[str, Any]],
        symbol: str,
        initial_stop_loss_pct: float = 0.02,
    ) -> List[TradeRecord]:
        """
        Enhance trade records with additional metrics.

        Args:
            trades: List of basic trade dictionaries
            symbol: Asset symbol
            initial_stop_loss_pct: Initial stop loss percentage

        Returns:
            List of enhanced TradeRecord objects
        """
        enhanced_trades = []

        for trade in trades:
            # Calculate holding days
            entry_date = trade["entry_date"]
            exit_date = trade["exit_date"]
            holding_days = (exit_date - entry_date).days

            # Calculate R-multiple
            r_multiple = self.calculate_r_multiple(
                entry_price=trade["entry_price"],
                exit_price=trade["exit_price"],
                direction=trade["direction"],
                initial_stop_loss_pct=initial_stop_loss_pct,
            )

            # Get regime
            regime_id = self.get_regime_for_date(symbol, entry_date)

            enhanced_trades.append(TradeRecord(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=exit_date,
                direction=trade["direction"],
                entry_price=trade["entry_price"],
                exit_price=trade["exit_price"],
                quantity=trade["quantity"],
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                holding_days=holding_days,
                r_multiple=r_multiple,
                regime_id=regime_id,
            ))

        return enhanced_trades

    def calculate_regime_breakdown(
        self,
        trades: List[TradeRecord],
    ) -> Dict[int, RegimePerformance]:
        """
        Calculate performance breakdown by regime.

        Args:
            trades: List of trade records

        Returns:
            Dictionary mapping regime_id to RegimePerformance
        """
        # Group trades by regime
        regime_trades = {}
        for trade in trades:
            if trade.regime_id is not None:
                if trade.regime_id not in regime_trades:
                    regime_trades[trade.regime_id] = []
                regime_trades[trade.regime_id].append(trade)

        # Calculate metrics for each regime
        regime_breakdown = {}

        for regime_id, regime_trade_list in regime_trades.items():
            total_trades = len(regime_trade_list)
            winning_trades = sum(1 for t in regime_trade_list if t.pnl > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            r_multiples = [t.r_multiple for t in regime_trade_list if t.r_multiple is not None]
            avg_r_multiple = np.mean(r_multiples) if r_multiples else 0.0

            total_pnl = sum(t.pnl for t in regime_trade_list)
            avg_pnl_pct = np.mean([t.pnl_pct for t in regime_trade_list]) if regime_trade_list else 0.0

            regime_breakdown[regime_id] = RegimePerformance(
                regime_id=regime_id,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_r_multiple=avg_r_multiple,
                total_pnl=total_pnl,
                avg_pnl_pct=avg_pnl_pct,
            )

        return regime_breakdown

    def run_backtest(
        self,
        strategy_config: StrategyConfig,
        symbol: str,
        period: BacktestPeriod,
    ) -> Optional[Dict[str, Any]]:
        """
        Run backtest for a single symbol and period.

        Args:
            strategy_config: Strategy configuration
            symbol: Asset symbol
            period: Backtest period

        Returns:
            Backtest results or None if failed
        """
        logger.info(f"Running backtest: {symbol} - {period}")

        try:
            # Run backtest
            result = run_backtest(
                session=self.session,
                symbol=symbol,
                start_date=period.start_date.strftime("%Y-%m-%d"),
                end_date=period.end_date.strftime("%Y-%m-%d"),
                initial_cash=strategy_config.get_initial_capital(),
                risk_per_trade=strategy_config.get_risk_per_trade_pct() / 100,
                use_ensemble=strategy_config.get("ensemble.enabled", True),
            )

            if not result or not result.get("equity_curve"):
                logger.warning(f"No results for {symbol} - {period}")
                return None

            return result

        except Exception as e:
            logger.error(f"Backtest failed for {symbol} - {period}: {e}", exc_info=True)
            return None

    def generate_report(
        self,
        strategy_config: StrategyConfig,
        backtest_type: BacktestType,
        symbols: List[str],
        periods: List[BacktestPeriod],
    ) -> List[ResearchReport]:
        """
        Generate comprehensive backtest reports.

        Args:
            strategy_config: Strategy configuration
            backtest_type: Type of backtest
            symbols: List of symbols
            periods: List of backtest periods

        Returns:
            List of ResearchReport objects (one per period)
        """
        reports = []

        for period in periods:
            logger.info(f"Generating report for period: {period}")

            # Run backtests for all symbols
            all_trades = []
            all_equity_curves = []
            all_dates = []

            for symbol in symbols:
                result = self.run_backtest(strategy_config, symbol, period)

                if result:
                    # Enhance trade records
                    enhanced_trades = self.enhance_trade_records(
                        trades=result.get("trades", []),
                        symbol=symbol,
                        initial_stop_loss_pct=strategy_config.get_risk_per_trade_pct() / 100,
                    )
                    all_trades.extend(enhanced_trades)

                    # Store equity curve
                    all_equity_curves.append(result["equity_curve"])
                    if not all_dates:
                        all_dates = result["dates"]

            if not all_trades:
                logger.warning(f"No trades for period: {period}")
                continue

            # Aggregate equity curves (simple average for multi-symbol)
            if len(all_equity_curves) > 1:
                equity_curve = np.mean(all_equity_curves, axis=0).tolist()
            elif all_equity_curves:
                equity_curve = all_equity_curves[0]
            else:
                equity_curve = []

            # Calculate metrics
            metrics = calculate_metrics(
                equity_curve=equity_curve,
                trades=[t.to_dict() for t in all_trades],
                initial_capital=strategy_config.get_initial_capital(),
            )

            # Calculate regime breakdown
            regime_breakdown = self.calculate_regime_breakdown(all_trades)

            # Create report
            report = ResearchReport(
                strategy_name=strategy_config.name,
                backtest_type=backtest_type,
                period=period,
                symbols=symbols,
                metrics=metrics,
                equity_curve=equity_curve,
                equity_dates=all_dates,
                trades=all_trades,
                regime_breakdown=regime_breakdown,
                metadata={
                    "strategy_version": strategy_config.version,
                    "strategy_description": strategy_config.description,
                    "initial_capital": strategy_config.get_initial_capital(),
                    "risk_per_trade_pct": strategy_config.get_risk_per_trade_pct(),
                    "generated_at": datetime.now().isoformat(),
                }
            )

            reports.append(report)

        return reports


def run_research_backtest(
    session: Session,
    strategy_path: str | Path,
    backtest_type: BacktestType = BacktestType.FULL,
    symbols: Optional[List[str]] = None,
    custom_period: Optional[BacktestPeriod] = None,
    scenario_name: Optional[str] = None,
) -> List[ResearchReport]:
    """
    Run research backtest with comprehensive reporting.

    This is the main entry point for running backtests.

    Args:
        session: Database session
        strategy_path: Path to strategy YAML file
        backtest_type: Type of backtest (SMOKE/FULL/SCENARIO)
        symbols: Optional list of symbols (overrides config)
        custom_period: Optional custom backtest period
        scenario_name: Optional scenario name for scenario backtests

    Returns:
        List of ResearchReport objects

    Example:
        >>> # Run smoke test
        >>> reports = run_research_backtest(
        ...     session=session,
        ...     strategy_path="strategies/baseline_v1.yaml",
        ...     backtest_type=BacktestType.SMOKE,
        ... )
        >>> print(reports[0].summary())

        >>> # Run full backtest
        >>> reports = run_research_backtest(
        ...     session=session,
        ...     strategy_path="strategies/baseline_v1.yaml",
        ...     backtest_type=BacktestType.FULL,
        ...     symbols=["BTC-USD", "ETH-USD"],
        ... )
        >>> for report in reports:
        ...     print(report.summary())

        >>> # Run scenario backtest
        >>> reports = run_research_backtest(
        ...     session=session,
        ...     strategy_path="strategies/baseline_v1.yaml",
        ...     backtest_type=BacktestType.SCENARIO,
        ...     scenario_name="covid_crash",
        ... )
    """
    # Load strategy
    strategy_config = load_strategy(Path(strategy_path))
    logger.info(f"Loaded strategy: {strategy_config.name} v{strategy_config.version}")

    # Initialize research engine
    engine = ResearchEngine(session)

    # Get backtest configuration
    symbol_list, periods = engine.get_backtest_config(
        backtest_type=backtest_type,
        strategy_config=strategy_config,
        symbols=symbols,
        custom_period=custom_period,
        scenario_name=scenario_name,
    )

    logger.info(f"Running {backtest_type.value} backtest:")
    logger.info(f"  Symbols: {symbol_list}")
    logger.info(f"  Periods: {[str(p) for p in periods]}")

    # Generate reports
    reports = engine.generate_report(
        strategy_config=strategy_config,
        backtest_type=backtest_type,
        symbols=symbol_list,
        periods=periods,
    )

    return reports
