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
    regimes = relationship("Regime", back_populates="symbol_obj", cascade="all, delete-orphan")
    dsi_records = relationship("DataHealthIndex", back_populates="symbol_obj", cascade="all, delete-orphan")

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


class Regime(Base):
    """
    Market regime classifications for each symbol and date.
    P1 extension: Analist-1 regime detection.
    """

    __tablename__ = "regimes"
    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_regime_symbol_date"),
        Index("ix_regimes_symbol_date", "symbol_id", "date"),
        Index("ix_regimes_regime_id", "regime_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    regime_id = Column(Integer, nullable=False)  # 0, 1, 2 for low/normal/high vol
    volatility = Column(Float, nullable=False)  # Rolling volatility
    trend = Column(Float, nullable=False)  # Rolling trend
    is_structural_break = Column(Integer, nullable=False, default=0)  # Boolean as int
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol_obj = relationship("Symbol", back_populates="regimes")

    def __repr__(self) -> str:
        return f"<Regime(symbol_id={self.symbol_id}, date={self.date}, regime_id={self.regime_id})>"


class DataHealthIndex(Base):
    """
    Data Health Index (DSI) for each symbol and date.
    P1 extension: Data quality monitoring.
    """

    __tablename__ = "data_health_index"
    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_dsi_symbol_date"),
        Index("ix_dsi_symbol_date", "symbol_id", "date"),
        Index("ix_dsi_score", "dsi"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    dsi = Column(Float, nullable=False)  # Overall data health score (0-1)
    missing_ratio = Column(Float, nullable=False)  # Fraction of missing data
    outlier_ratio = Column(Float, nullable=False)  # Fraction of extreme outliers
    volume_jump_ratio = Column(Float, nullable=False)  # Fraction of volume jumps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol_obj = relationship("Symbol", back_populates="dsi_records")

    def __repr__(self) -> str:
        return f"<DataHealthIndex(symbol_id={self.symbol_id}, date={self.date}, dsi={self.dsi:.2f})>"
