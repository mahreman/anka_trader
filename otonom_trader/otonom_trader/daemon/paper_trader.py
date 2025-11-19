"""
Paper trading engine.
P3 preparation: Simulated trade execution and portfolio tracking.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from sqlalchemy.orm import Session

from ..data.schema import Symbol, PaperTrade, DailyBar, IntradayBar
from ..domain import Decision, SignalType
from ..utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in an asset."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class PortfolioState:
    """Current state of the paper trading portfolio."""

    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value

    def add_position(self, symbol: str, quantity: float, price: float):
        """Add or update a position."""
        if symbol in self.positions:
            # Average in
            pos = self.positions[symbol]
            total_cost = pos.cost_basis + (quantity * price)
            total_qty = pos.quantity + quantity
            pos.quantity = total_qty
            pos.avg_price = total_cost / total_qty if total_qty > 0 else price
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
            )

    def reduce_position(self, symbol: str, quantity: float) -> Optional[float]:
        """
        Reduce or close a position.

        Returns:
            Average price of the sold position, or None if position doesn't exist
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        avg_price = pos.avg_price

        pos.quantity -= quantity

        # Remove position if closed
        if pos.quantity <= 0:
            del self.positions[symbol]

        return avg_price

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price


class PaperTrader:
    """
    Paper trading engine.
    Executes simulated trades based on decisions.
    """

    def __init__(
        self,
        session: Session,
        initial_cash: float = 100000.0,
        price_interval: str = "15m",
    ):
        """
        Initialize paper trader.

        Args:
            session: Database session
            initial_cash: Initial cash balance
            price_interval: Intraday interval used when looking up prices
        """
        self.session = session
        self.portfolio = PortfolioState(cash=initial_cash)
        self.trade_history: List[PaperTrade] = []
        self.price_interval = price_interval
        logger.info(
            f"Initialized paper trader with ${initial_cash:,.2f} (interval={self.price_interval})"
        )

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current (latest) price for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Latest close price or None if not found
        """
        symbol_obj = self.session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            return None

        latest_bar = (
            self.session.query(DailyBar)
            .filter_by(symbol_id=symbol_obj.id)
            .order_by(DailyBar.date.desc())
            .first()
        )

        if latest_bar:
            return latest_bar.close

        latest_intraday = (
            self.session.query(IntradayBar)
            .filter_by(symbol_id=symbol_obj.id, interval=self.price_interval)
            .order_by(IntradayBar.ts.desc())
            .first()
        )

        return latest_intraday.close if latest_intraday else None

    def calculate_position_size(
        self, symbol: str, price: float, risk_pct: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk percentage.

        Args:
            symbol: Asset symbol
            price: Current price
            risk_pct: Risk percentage of total equity (e.g., 1.0 for 1%)

        Returns:
            Number of shares/units to buy
        """
        risk_amount = self.portfolio.total_value * (risk_pct / 100.0)
        quantity = risk_amount / price
        return quantity

    def execute_buy(
        self, symbol: str, price: float, quantity: float, decision_id: Optional[int] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a BUY order.

        Args:
            symbol: Asset symbol
            price: Execution price
            quantity: Number of shares/units
            decision_id: Associated decision ID

        Returns:
            PaperTrade record or None if failed
        """
        value = price * quantity

        # Check if we have enough cash
        if value > self.portfolio.cash:
            logger.warning(
                f"Insufficient cash for BUY {symbol}: need ${value:,.2f}, have ${self.portfolio.cash:,.2f}"
            )
            return None

        # Execute trade
        self.portfolio.cash -= value
        self.portfolio.add_position(symbol, quantity, price)

        # Log trade
        trade = self._log_trade(
            symbol=symbol,
            action="BUY",
            price=price,
            quantity=quantity,
            value=value,
            decision_id=decision_id,
        )

        logger.info(
            f"BUY {quantity:.4f} {symbol} @ ${price:.2f} | Total: ${value:,.2f} | Cash: ${self.portfolio.cash:,.2f}"
        )

        return trade

    def execute_sell(
        self, symbol: str, price: float, quantity: float, decision_id: Optional[int] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a SELL order.

        Args:
            symbol: Asset symbol
            price: Execution price
            quantity: Number of shares/units
            decision_id: Associated decision ID

        Returns:
            PaperTrade record or None if failed
        """
        # Check if we have the position
        if symbol not in self.portfolio.positions:
            logger.warning(f"Cannot SELL {symbol}: No position")
            return None

        pos = self.portfolio.positions[symbol]
        if quantity > pos.quantity:
            logger.warning(
                f"Cannot SELL {quantity:.4f} {symbol}: Only have {pos.quantity:.4f}"
            )
            return None

        # Execute trade
        value = price * quantity
        avg_cost = self.portfolio.reduce_position(symbol, quantity)

        self.portfolio.cash += value

        # Log trade
        trade = self._log_trade(
            symbol=symbol,
            action="SELL",
            price=price,
            quantity=quantity,
            value=value,
            decision_id=decision_id,
        )

        # Calculate realized PnL
        if avg_cost:
            realized_pnl = (price - avg_cost) * quantity
            logger.info(
                f"SELL {quantity:.4f} {symbol} @ ${price:.2f} | "
                f"Total: ${value:,.2f} | PnL: ${realized_pnl:+,.2f} | Cash: ${self.portfolio.cash:,.2f}"
            )
        else:
            logger.info(
                f"SELL {quantity:.4f} {symbol} @ ${price:.2f} | Total: ${value:,.2f}"
            )

        return trade

    def execute_decision(
        self, decision: Decision, risk_pct: float = 1.0
    ) -> Optional[PaperTrade]:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision
            risk_pct: Risk percentage for position sizing

        Returns:
            PaperTrade record or None if no action taken
        """
        symbol = decision.asset_symbol
        signal = decision.signal

        # Get current price
        price = self.get_current_price(symbol)
        if price is None:
            logger.warning(f"Cannot execute decision for {symbol}: No price data")
            return None

        # Get decision ID from DB
        from ..data.schema import Decision as DecisionORM

        decision_orm = (
            self.session.query(DecisionORM)
            .join(Symbol)
            .filter(Symbol.symbol == symbol, DecisionORM.date == decision.date)
            .first()
        )
        decision_id = decision_orm.id if decision_orm else None

        # Execute based on signal
        if signal == SignalType.BUY:
            # Calculate position size
            quantity = self.calculate_position_size(symbol, price, risk_pct)
            return self.execute_buy(symbol, price, quantity, decision_id)

        elif signal == SignalType.SELL:
            # Sell entire position if we have one
            if symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[symbol].quantity
                return self.execute_sell(symbol, price, quantity, decision_id)
            else:
                logger.info(f"SELL signal for {symbol} but no position to sell")
                return None

        elif signal == SignalType.HOLD:
            # No action for HOLD
            logger.debug(f"HOLD signal for {symbol}")
            return None

        return None

    def execute_decisions(
        self, decisions: List[Decision], risk_pct: float = 1.0
    ) -> List[PaperTrade]:
        """Execute multiple trading decisions sequentially."""

        executed_trades: List[PaperTrade] = []
        for decision in decisions:
            trade = self.execute_decision(decision, risk_pct=risk_pct)
            if trade is not None:
                executed_trades.append(trade)

        return executed_trades

    def update_portfolio_prices(self):
        """Update current prices for all positions."""
        prices = {}
        for symbol in self.portfolio.positions.keys():
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price

        self.portfolio.update_prices(prices)

    def _log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        value: float,
        decision_id: Optional[int] = None,
    ) -> PaperTrade:
        """
        Log a paper trade to the database.

        Args:
            symbol: Asset symbol
            action: BUY, SELL, HOLD
            price: Execution price
            quantity: Number of shares/units
            value: Total value
            decision_id: Associated decision ID

        Returns:
            PaperTrade record
        """
        # Get symbol ID
        symbol_obj = self.session.query(Symbol).filter_by(symbol=symbol).first()
        if not symbol_obj:
            raise ValueError(f"Symbol not found: {symbol}")

        # Create trade record
        trade = PaperTrade(
            timestamp=utc_now(),
            symbol_id=symbol_obj.id,
            decision_id=decision_id,
            action=action,
            price=price,
            quantity=quantity,
            value=value,
            portfolio_value=self.portfolio.total_value,
            cash=self.portfolio.cash,
        )

        self.session.add(trade)
        self.session.commit()

        return trade

    def get_portfolio_summary(self) -> dict:
        """
        Get current portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        self.update_portfolio_prices()

        return {
            "cash": self.portfolio.cash,
            "positions_value": self.portfolio.positions_value,
            "total_value": self.portfolio.total_value,
            "num_positions": len(self.portfolio.positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct * 100,
                }
                for p in self.portfolio.positions.values()
            ],
        }

    def create_portfolio_snapshot(self) -> "PortfolioSnapshot":
        """
        Create a portfolio snapshot and persist to database.

        Returns:
            PortfolioSnapshot record

        Example:
            >>> trader = PaperTrader(session)
            >>> snapshot = trader.create_portfolio_snapshot()
            >>> print(f"Equity: ${snapshot.equity:.2f}")
        """
        from ..data.schema import PortfolioSnapshot

        # Update prices first
        self.update_portfolio_prices()

        # Get current metrics
        equity = self.portfolio.total_value
        cash = self.portfolio.cash
        positions_value = self.portfolio.positions_value
        num_positions = len(self.portfolio.positions)

        # Calculate max drawdown
        # Get all previous snapshots to find peak equity
        previous_snapshots = (
            self.session.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .all()
        )

        max_equity = equity
        if previous_snapshots:
            historical_max = max(s.equity for s in previous_snapshots)
            max_equity = max(historical_max, equity)

        # Calculate drawdown
        if max_equity > 0:
            drawdown = (max_equity - equity) / max_equity
        else:
            drawdown = 0.0

        # Create snapshot
        snapshot = PortfolioSnapshot(
            timestamp=utc_now(),
            equity=equity,
            cash=cash,
            positions_value=positions_value,
            num_positions=num_positions,
            max_drawdown=drawdown,
            max_equity=max_equity,
        )

        self.session.add(snapshot)
        self.session.commit()

        logger.info(
            f"Portfolio snapshot created: equity=${equity:,.2f}, "
            f"cash=${cash:,.2f}, positions=${positions_value:,.2f}, "
            f"drawdown={drawdown:.2%}"
        )

        return snapshot
