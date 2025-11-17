"""
Database connection and session management.
"""
import logging
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from ..config import DB_PATH

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Engine = None
_SessionLocal: sessionmaker = None


def get_engine(db_path: str = None) -> Engine:
    """
    Get or create SQLAlchemy engine.

    Args:
        db_path: Path to SQLite database file. If None, uses config default.

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine

    if _engine is None:
        path = db_path or DB_PATH
        logger.info(f"Creating database engine for: {path}")
        _engine = create_engine(
            f"sqlite:///{path}",
            echo=False,  # Set to True for SQL query logging
            connect_args={"check_same_thread": False},  # Needed for SQLite
        )

    return _engine


def get_session_factory(engine: Engine = None) -> sessionmaker:
    """
    Get or create session factory.

    Args:
        engine: SQLAlchemy engine. If None, uses default engine.

    Returns:
        Session factory
    """
    global _SessionLocal

    if _SessionLocal is None:
        eng = engine or get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)

    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session (context manager style).

    Yields:
        SQLAlchemy Session

    Example:
        with get_session() as session:
            # use session
            pass
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Engine = None) -> None:
    """
    Initialize the database schema.
    Creates all tables defined in schema.py.

    Args:
        engine: SQLAlchemy engine. If None, uses default engine.
    """
    from .schema import Base

    eng = engine or get_engine()
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=eng)
    logger.info("Database schema initialized successfully")
