"""
RL state builder - constructs state from market and portfolio data.

Converts raw market data into RL-compatible state representation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from .rl_agent import RlState

logger = logging.getLogger(__name__)


class RlStateBuilder:
    """
    Build RL state from market and portfolio data.

    Aggregates all relevant features into RlState for RL agent input.

    Example:
        >>> builder = RlStateBuilder(lookback_window=20)
        >>> state = builder.build_state(
        ...     session=session,
        ...     symbol="BTC-USD",
        ...     current_date=date(2024, 1, 15),
        ...     portfolio_position=0.5,
        ...     portfolio_cash=50000,
        ...     portfolio_equity=100000
        ... )
    """

    def __init__(
        self,
        lookback_window: int = 20,
        normalize_returns: bool = True,
    ):
        """
        Initialize state builder.

        Args:
            lookback_window: Number of historical bars to include
            normalize_returns: Whether to normalize returns
        """
        self.lookback_window = lookback_window
        self.normalize_returns = normalize_returns

    def build_state(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
        portfolio_position: float = 0.0,
        portfolio_cash: float = 100000.0,
        portfolio_equity: float = 100000.0,
        last_trade_date: Optional[datetime] = None,
    ) -> RlState:
        """
        Build RL state from current market and portfolio data.

        Args:
            session: Database session
            symbol: Symbol to build state for
            current_date: Current date
            portfolio_position: Current position size (-1 to 1, where 0=neutral)
            portfolio_cash: Current cash
            portfolio_equity: Current total equity
            last_trade_date: Date of last trade (for time-since-trade feature)

        Returns:
            RlState instance

        Example:
            >>> state = builder.build_state(
            ...     session, "BTC-USD", date(2024, 1, 15),
            ...     portfolio_position=0.5, portfolio_equity=100000
            ... )
        """
        # Get price history
        price_history = self._get_price_history(session, symbol, current_date)

        # Calculate returns
        returns = self._calculate_returns(price_history)

        # Get technical indicators
        technical_indicators = self._get_technical_indicators(
            session, symbol, current_date
        )

        # Get regime
        regime = self._get_regime(session, symbol, current_date)

        # Get DSI
        dsi = self._get_dsi(session, symbol, current_date)

        # Get sentiment scores
        news_sentiment, macro_sentiment = self._get_sentiment_scores(
            session, symbol, current_date
        )

        # Calculate portfolio features
        portfolio_features = self._get_portfolio_features(
            portfolio_position=portfolio_position,
            portfolio_cash=portfolio_cash,
            portfolio_equity=portfolio_equity,
            last_trade_date=last_trade_date,
            current_date=current_date,
        )

        # Build state
        state = RlState(
            returns_history=returns,
            technical_indicators=technical_indicators,
            portfolio_position=portfolio_position,
            portfolio_leverage=portfolio_features["leverage"],
            portfolio_cash_pct=portfolio_features["cash_pct"],
            days_since_trade=portfolio_features["days_since_trade"],
            regime=regime,
            dsi=dsi,
            news_sentiment=news_sentiment,
            macro_sentiment=macro_sentiment,
        )

        return state

    def _get_price_history(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
    ) -> np.ndarray:
        """Get recent price history."""
        from ..data import DailyBar

        # Get last N bars
        bars = (
            session.query(DailyBar)
            .filter(
                DailyBar.symbol == symbol,
                DailyBar.date <= current_date,
            )
            .order_by(DailyBar.date.desc())
            .limit(self.lookback_window + 1)  # +1 for return calculation
            .all()
        )

        if len(bars) < 2:
            # Not enough data
            return np.zeros(self.lookback_window)

        # Extract close prices (reverse to chronological order)
        prices = np.array([bar.close for bar in reversed(bars)])

        return prices

    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate returns from prices."""
        if len(prices) < 2:
            return np.zeros(self.lookback_window)

        # Calculate log returns
        returns = np.diff(np.log(prices))

        # Pad if needed
        if len(returns) < self.lookback_window:
            padding = np.zeros(self.lookback_window - len(returns))
            returns = np.concatenate([padding, returns])
        else:
            returns = returns[-self.lookback_window :]

        # Normalize if requested
        if self.normalize_returns and returns.std() > 0:
            returns = (returns - returns.mean()) / returns.std()

        return returns

    def _get_technical_indicators(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Get technical indicator values."""
        # TODO: Implement actual technical indicator calculation
        # For now, return placeholder values

        return {
            "rsi": 50.0,
            "macd": 0.0,
            "bb_width": 0.05,
            "volume_ratio": 1.0,
        }

    def _get_regime(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
    ) -> int:
        """Get current market regime."""
        from ..data import Regime

        regime_record = (
            session.query(Regime)
            .filter(
                Regime.symbol == symbol,
                Regime.date <= current_date,
            )
            .order_by(Regime.date.desc())
            .first()
        )

        if regime_record:
            return regime_record.regime_id
        else:
            return 0  # Default to low volatility

    def _get_dsi(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
    ) -> float:
        """Get DSI (Data Health Score)."""
        from ..data import DSI

        dsi_record = (
            session.query(DSI)
            .filter(
                DSI.symbol == symbol,
                DSI.date <= current_date,
            )
            .order_by(DSI.date.desc())
            .first()
        )

        if dsi_record:
            return dsi_record.score
        else:
            return 0.5  # Default neutral DSI

    def _get_sentiment_scores(
        self,
        session: Session,
        symbol: str,
        current_date: datetime,
    ) -> tuple[float, float]:
        """Get news and macro sentiment scores."""
        # TODO: Implement sentiment extraction from news/macro data
        # For now, return neutral scores

        news_sentiment = 0.0  # -1 (bearish) to +1 (bullish)
        macro_sentiment = 0.0

        return news_sentiment, macro_sentiment

    def _get_portfolio_features(
        self,
        portfolio_position: float,
        portfolio_cash: float,
        portfolio_equity: float,
        last_trade_date: Optional[datetime],
        current_date: datetime,
    ) -> Dict[str, float]:
        """Calculate portfolio-derived features."""
        # Calculate leverage
        if portfolio_equity > 0:
            leverage = abs(portfolio_position) / portfolio_equity
        else:
            leverage = 0.0

        # Calculate cash percentage
        if portfolio_equity > 0:
            cash_pct = portfolio_cash / portfolio_equity
        else:
            cash_pct = 1.0

        # Calculate days since last trade
        if last_trade_date:
            days_since_trade = (current_date - last_trade_date).days
        else:
            days_since_trade = 999  # Large number if no trades yet

        return {
            "leverage": leverage,
            "cash_pct": cash_pct,
            "days_since_trade": days_since_trade,
        }


def build_rl_state(
    session: Session,
    symbol: str,
    current_date: datetime,
    portfolio_position: float = 0.0,
    portfolio_equity: float = 100000.0,
    lookback_window: int = 20,
) -> RlState:
    """
    Convenience function to build RL state.

    Args:
        session: Database session
        symbol: Symbol
        current_date: Current date
        portfolio_position: Current position
        portfolio_equity: Current equity
        lookback_window: Lookback window

    Returns:
        RlState instance

    Example:
        >>> state = build_rl_state(session, "BTC-USD", date(2024, 1, 15))
    """
    builder = RlStateBuilder(lookback_window=lookback_window)

    return builder.build_state(
        session=session,
        symbol=symbol,
        current_date=current_date,
        portfolio_position=portfolio_position,
        portfolio_equity=portfolio_equity,
    )
