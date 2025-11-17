"""
Intraday bar and portfolio models for ORM schema (P2.5/P3).

This is a separate file to avoid disrupting the existing schema.
You can merge it into the main schema.py later if desired.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from .schema import Base, Symbol  # Reuse existing Base and Symbol


class IntradayBar(Base):
    """
    Intraday OHLCV bars for higher-frequency trading.

    Supports multiple timeframes (M15, H1, H4, etc.).
    """

    __tablename__ = "intraday_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    ts = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(8), nullable=False)  # "M15", "H1", "H4", "D1"

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    # Relationships
    symbol_obj = relationship("Symbol")

    def __repr__(self) -> str:
        return f"<IntradayBar(symbol_id={self.symbol_id}, ts={self.ts}, tf={self.timeframe})>"


class PortfolioPosition(Base):
    """
    Open position in the portfolio.

    Tracks entry price, quantity, and timestamp.
    """

    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    opened_at = Column(DateTime, nullable=False)
    qty = Column(Float, nullable=False)  # Positive for long, negative for short
    entry_price = Column(Float, nullable=False)

    # Relationships
    symbol_obj = relationship("Symbol")

    def __repr__(self) -> str:
        return f"<PortfolioPosition(symbol_id={self.symbol_id}, qty={self.qty}, entry={self.entry_price})>"


class PortfolioSnapshot(Base):
    """
    Portfolio state snapshot at a given timestamp.

    Tracks equity, cash, and drawdown for performance monitoring.
    """

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, index=True)
    equity = Column(Float, nullable=False)  # Total portfolio value
    cash = Column(Float, nullable=False)  # Available cash
    max_drawdown = Column(Float, nullable=True)  # Max drawdown from peak

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot(ts={self.ts}, equity={self.equity:.2f}, dd={self.max_drawdown:.2%})>"
