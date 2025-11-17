"""
Simplified portfolio backtest for research.

Provides a simple run_backtest() function for backtesting strategies.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..data.schema import DailyBar, Symbol, Decision, Anomaly
from ..domain.enums import SignalType

logger = logging.getLogger(__name__)


def run_backtest(
    session: Session,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_cash: float = 100000.0,
    risk_per_trade: float = 0.01,
    use_ensemble: bool = True,
) -> Dict[str, Any]:
    """
    Run a simple backtest for a symbol.

    This is a simplified backtest that:
    1. Loads decisions from the database
    2. Simulates trades based on decisions
    3. Tracks equity curve
    4. Returns results

    Args:
        session: Database session
        symbol: Asset symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Starting capital
        risk_per_trade: Risk per trade (fraction, e.g., 0.01 = 1%)
        use_ensemble: Whether ensemble decisions were used

    Returns:
        Dictionary with:
            - equity_curve: List of equity values
            - trades: List of trade dictionaries
            - dates: List of dates

    Example:
        >>> result = run_backtest(session, "BTC-USD", "2020-01-01", "2021-01-01")
        >>> print(f"Final equity: ${result['equity_curve'][-1]:,.2f}")
    """
    logger.info(f"Running backtest: {symbol} from {start_date} to {end_date}")

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Get symbol object
    symbol_obj = session.query(Symbol).filter_by(symbol=symbol).first()
    if not symbol_obj:
        logger.warning(f"Symbol not found: {symbol}")
        return {}

    # Get decisions in date range
    decisions = (
        session.query(Decision)
        .filter(
            Decision.symbol_id == symbol_obj.id,
            Decision.date >= start,
            Decision.date <= end,
        )
        .order_by(Decision.date)
        .all()
    )

    if not decisions:
        logger.warning(f"No decisions found for {symbol} in date range")
        return {}

    # Get price data
    bars = (
        session.query(DailyBar)
        .filter(
            DailyBar.symbol_id == symbol_obj.id,
            DailyBar.date >= start,
            DailyBar.date <= end,
        )
        .order_by(DailyBar.date)
        .all()
    )

    if not bars:
        logger.warning(f"No price data for {symbol}")
        return {}

    # Create price lookup
    price_map = {bar.date: bar.close for bar in bars}

    # Initialize portfolio
    cash = initial_cash
    position = None  # Current position: {"entry_date", "entry_price", "quantity", "direction"}
    equity_curve = []
    trades = []
    dates = []

    # Simulate trading
    for bar in bars:
        current_date = bar.date
        current_price = bar.close

        # Check if we have a decision for this date
        decision = next((d for d in decisions if d.date == current_date), None)

        # Update equity
        if position:
            position_value = position["quantity"] * current_price
            total_equity = cash + position_value
        else:
            total_equity = cash

        equity_curve.append(total_equity)
        dates.append(current_date)

        # Process decision
        if decision:
            signal = decision.signal

            # Entry logic
            if position is None and signal in ("BUY", "SELL"):
                # Calculate position size based on risk
                position_value = total_equity * risk_per_trade * 10  # Simplified sizing
                quantity = position_value / current_price

                # Open position
                position = {
                    "entry_date": current_date,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "direction": signal,
                }

                # Deduct cash for BUY
                if signal == "BUY":
                    cash -= position_value

                logger.debug(
                    f"{current_date}: {signal} {quantity:.4f} @ ${current_price:.2f}"
                )

            # Exit logic
            elif position and signal == "HOLD":
                # Close position
                exit_value = position["quantity"] * current_price

                # Calculate PnL
                if position["direction"] == "BUY":
                    pnl = exit_value - (position["quantity"] * position["entry_price"])
                    cash += exit_value
                else:  # SELL (short)
                    pnl = (position["quantity"] * position["entry_price"]) - exit_value
                    cash += pnl

                pnl_pct = pnl / (position["quantity"] * position["entry_price"]) * 100

                # Record trade
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })

                logger.debug(
                    f"{current_date}: Close {position['direction']} - "
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                )

                position = None

    # Close any remaining position at end
    if position:
        current_price = bars[-1].close
        exit_value = position["quantity"] * current_price

        if position["direction"] == "BUY":
            pnl = exit_value - (position["quantity"] * position["entry_price"])
            cash += exit_value
        else:
            pnl = (position["quantity"] * position["entry_price"]) - exit_value
            cash += pnl

        pnl_pct = pnl / (position["quantity"] * position["entry_price"]) * 100

        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": bars[-1].date,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "quantity": position["quantity"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    logger.info(
        f"Backtest complete: {len(trades)} trades, "
        f"Final equity: ${equity_curve[-1]:,.2f}"
    )

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "dates": dates,
    }
