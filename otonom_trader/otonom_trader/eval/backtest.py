"""
Event-based backtesting and hypothesis backlog.

This module assumes P0 schema with:
- DailyBar
- Anomaly
- Decision

It adds:
- Hypothesis and HypothesisResult ORM models (in schema.py).
- A simple event-based backtest engine that:
    * replays decisions on historical anomalies
    * computes PnL under simple assumptions
    * logs results into HypothesisResult with regime context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Callable, Dict

from sqlalchemy.orm import Session

from ..data.schema import (
    DailyBar,
    Anomaly as AnomalyORM,
    Decision as DecisionORM,
    Symbol,
    Hypothesis,
    HypothesisResult,
    Regime as RegimeORM,
    DataHealthIndex as DsiORM,
)
from ..domain import SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Simple event-based backtest configuration.

    Attributes:
        holding_days: Fixed holding period after signal (default: 5)
        slippage_bps: Round-trip slippage in basis points (default: 5.0)
    """
    holding_days: int = 5
    slippage_bps: float = 5.0


def _get_price_on(
    session: Session,
    symbol_id: int,
    d: date,
) -> Optional[float]:
    """
    Get closing price for a symbol on a specific date.

    Args:
        session: Database session
        symbol_id: Symbol ID
        d: Date

    Returns:
        Closing price or None if no data
    """
    bar = (
        session.query(DailyBar)
        .filter(
            DailyBar.symbol_id == symbol_id,
            DailyBar.date == d,
        )
        .first()
    )
    if bar is None:
        return None
    return float(bar.close)


def _apply_slippage(price: float, slippage_bps: float, direction: int) -> float:
    """
    Apply slippage to a price.

    Args:
        price: Base price
        slippage_bps: Slippage in basis points
        direction: +1 for buying (increases price), -1 for selling (decreases price)

    Returns:
        Price with slippage applied
    """
    return price * (1.0 + direction * slippage_bps / 10000.0)


def create_or_get_hypothesis(
    session: Session,
    name: str,
    rule_signature: str,
    description: Optional[str] = None,
    config_json: Optional[str] = None,
) -> Hypothesis:
    """
    Create or retrieve a hypothesis by name.

    Args:
        session: Database session
        name: Unique hypothesis name
        rule_signature: Rule description (e.g., "SPIKE_DOWN + Uptrend → BUY")
        description: Optional long description
        config_json: Optional JSON config string

    Returns:
        Hypothesis object
    """
    obj = (
        session.query(Hypothesis)
        .filter(Hypothesis.name == name)
        .first()
    )
    if obj:
        return obj

    obj = Hypothesis(
        name=name,
        rule_signature=rule_signature,
        description=description,
        config_json=config_json,
    )
    session.add(obj)
    session.flush()
    logger.info(f"Created new hypothesis: {name}")
    return obj


def _get_regime_for_date(
    session: Session,
    symbol_id: int,
    d: date,
) -> Optional[int]:
    """Get regime_id for a symbol on a specific date."""
    regime = (
        session.query(RegimeORM)
        .filter(RegimeORM.symbol_id == symbol_id, RegimeORM.date == d)
        .first()
    )
    return regime.regime_id if regime else None


def _get_dsi_for_date(
    session: Session,
    symbol_id: int,
    d: date,
) -> Optional[float]:
    """Get DSI for a symbol on a specific date."""
    dsi_record = (
        session.query(DsiORM)
        .filter(DsiORM.symbol_id == symbol_id, DsiORM.date == d)
        .first()
    )
    return dsi_record.dsi if dsi_record else None


def run_event_backtest(
    session: Session,
    hypothesis: Hypothesis,
    config: BacktestConfig,
    regime_resolver: Optional[Callable[[date, int], Optional[int]]] = None,
    dsi_resolver: Optional[Callable[[date, int], Optional[float]]] = None,
) -> int:
    """
    Replay decisions for all anomalies and record results.

    Args:
        session: Database session
        hypothesis: Hypothesis object to test
        config: Backtest configuration
        regime_resolver: Optional function(date, symbol_id) -> regime_id
        dsi_resolver: Optional function(date, symbol_id) -> dsi

    Returns:
        Number of backtest results created

    Notes:
        - If regime_resolver/dsi_resolver are None, uses default DB lookup
        - Only trades BUY/SELL signals (ignores HOLD)
        - Uses fixed holding period from config
    """
    # Use default resolvers if not provided
    if regime_resolver is None:
        regime_resolver = lambda d, sid: _get_regime_for_date(session, sid, d)
    if dsi_resolver is None:
        dsi_resolver = lambda d, sid: _get_dsi_for_date(session, sid, d)

    # Get all anomalies with their decisions
    anomalies = session.query(AnomalyORM).order_by(AnomalyORM.date.asc()).all()
    logger.info(f"Running backtest on {len(anomalies)} anomalies for hypothesis: {hypothesis.name}")

    count = 0

    for anom in anomalies:
        symbol_id = anom.symbol_id

        # Find decision for this anomaly (match by symbol_id and date)
        decision = (
            session.query(DecisionORM)
            .filter(
                DecisionORM.symbol_id == symbol_id,
                DecisionORM.date == anom.date
            )
            .first()
        )

        if decision is None:
            continue

        # Only trade BUY/SELL, ignore HOLD
        if decision.signal not in [SignalType.BUY.value, SignalType.SELL.value]:
            continue

        entry_date = anom.date
        exit_date = entry_date + timedelta(days=config.holding_days)

        entry_price = _get_price_on(session, symbol_id, entry_date)
        exit_price = _get_price_on(session, symbol_id, exit_date)

        if entry_price is None or exit_price is None:
            continue

        # Apply slippage and calculate PnL
        if decision.signal == SignalType.BUY.value:
            # Buy: entry with +slippage, exit with -slippage
            entry_price = _apply_slippage(entry_price, config.slippage_bps, +1)
            exit_price = _apply_slippage(exit_price, config.slippage_bps, -1)
            pnl = exit_price - entry_price
        elif decision.signal == SignalType.SELL.value:
            # Short sell: entry with -slippage, exit with +slippage
            entry_price = _apply_slippage(entry_price, config.slippage_bps, -1)
            exit_price = _apply_slippage(exit_price, config.slippage_bps, +1)
            pnl = entry_price - exit_price
        else:
            continue

        pnl_pct = pnl / entry_price if entry_price != 0 else 0.0

        # Get P1 context
        regime_id = regime_resolver(entry_date, symbol_id)
        dsi = dsi_resolver(entry_date, symbol_id)

        # Create result record
        result = HypothesisResult(
            hypothesis_id=hypothesis.id,
            symbol_id=symbol_id,
            anomaly_id=anom.id,
            decision_id=decision.id,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            regime_id=regime_id,
            dsi=dsi,
        )
        session.add(result)
        count += 1

    session.commit()
    logger.info(f"Created {count} backtest results for hypothesis: {hypothesis.name}")
    return count


def get_backtest_summary(
    session: Session,
    hypothesis_id: int,
) -> Dict[str, float]:
    """
    Get summary statistics for a hypothesis backtest.

    Args:
        session: Database session
        hypothesis_id: Hypothesis ID

    Returns:
        Dictionary with summary statistics:
        - total_trades: Number of trades
        - win_rate: Percentage of profitable trades
        - avg_pnl_pct: Average PnL percentage
        - total_pnl: Sum of all PnL
        - avg_win_pct: Average winning trade percentage
        - avg_loss_pct: Average losing trade percentage
    """
    results = (
        session.query(HypothesisResult)
        .filter(HypothesisResult.hypothesis_id == hypothesis_id)
        .all()
    )

    if not results:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }

    total_trades = len(results)
    wins = [r for r in results if r.pnl > 0]
    losses = [r for r in results if r.pnl <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_pnl_pct = sum(r.pnl_pct for r in results) / total_trades if total_trades > 0 else 0.0
    total_pnl = sum(r.pnl for r in results)
    avg_win_pct = sum(r.pnl_pct for r in wins) / len(wins) if wins else 0.0
    avg_loss_pct = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct,
        "total_pnl": total_pnl,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }
