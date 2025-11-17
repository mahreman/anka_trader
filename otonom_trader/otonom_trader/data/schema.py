"""
SQLAlchemy ORM models for database tables.
"""
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Symbol(Base):
    """
    Asset symbol metadata table.
    """

    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    asset_class = Column(String(20), nullable=False)

    # Relationships
    daily_bars = relationship("DailyBar", back_populates="symbol_obj", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="symbol_obj", cascade="all, delete-orphan")
    decisions = relationship("Decision", back_populates="symbol_obj", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Symbol(id={self.id}, symbol='{self.symbol}', name='{self.name}')>"


class DailyBar(Base):
    """
    Daily OHLCV bars for each asset.
    """

    __tablename__ = "daily_bars"
    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_symbol_date"),
        Index("ix_daily_bars_symbol_date", "symbol_id", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=True)  # Adjusted close (optional)

    # Relationships
    symbol_obj = relationship("Symbol", back_populates="daily_bars")

    def __repr__(self) -> str:
        return f"<DailyBar(symbol_id={self.symbol_id}, date={self.date}, close={self.close})>"


class Anomaly(Base):
    """
    Detected price anomalies (spikes/crashes).
    """

    __tablename__ = "anomalies"
    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_anomaly_symbol_date"),
        Index("ix_anomalies_symbol_date", "symbol_id", "date"),
        Index("ix_anomalies_type", "anomaly_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    anomaly_type = Column(String(20), nullable=False)  # SPIKE_UP, SPIKE_DOWN
    abs_return = Column(Float, nullable=False)  # Absolute return value
    zscore = Column(Float, nullable=False)  # Z-score of return
    volume_rank = Column(Float, nullable=False)  # Volume percentile (0-1)
    comment = Column(Text, nullable=True)  # Manual comment/note
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol_obj = relationship("Symbol", back_populates="anomalies")

    def __repr__(self) -> str:
        return f"<Anomaly(symbol_id={self.symbol_id}, date={self.date}, type={self.anomaly_type})>"


class Decision(Base):
    """
    Trading decisions made by the Patron.
    """

    __tablename__ = "decisions"
    __table_args__ = (
        Index("ix_decisions_symbol_date", "symbol_id", "date"),
        Index("ix_decisions_signal", "signal"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    signal = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)  # 0-1
    reason = Column(Text, nullable=False)  # Explanation
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol_obj = relationship("Symbol", back_populates="decisions")

    def __repr__(self) -> str:
        return f"<Decision(symbol_id={self.symbol_id}, date={self.date}, signal={self.signal})>"
