"""
Database connection and session management.
"""
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from ..config import DB_PATH

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_engine_path: Optional[str] = None
_SessionLocal: Optional[sessionmaker] = None


def _create_engine(path: str) -> Engine:
    """Create a new SQLAlchemy engine for the given SQLite path."""
    logger.info(f"Creating database engine for: {path}")
    return create_engine(
        f"sqlite:///{path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )


def get_engine(db_path: Optional[str] = None) -> Engine:
    """
    Get or create SQLAlchemy engine.

    Args:
        db_path: Path to SQLite database file. If None, uses config default.

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine, _engine_path, _SessionLocal

    requested_path = db_path or DB_PATH

    if _engine is None:
        _engine = _create_engine(requested_path)
        _engine_path = requested_path
        return _engine

    if requested_path != _engine_path:
        logger.info("Switching database engine to: %s", requested_path)
        _engine.dispose()
        _engine = _create_engine(requested_path)
        _engine_path = requested_path
        _SessionLocal = None  # reset session factory so future sessions use new engine

    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
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


@contextmanager
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

    Gracefully handles existing tables and indexes.

    Args:
        engine: SQLAlchemy engine. If None, uses default engine.
    """
    from sqlalchemy import inspect
    from sqlalchemy.exc import OperationalError
    from .schema import Base

    eng = engine or get_engine()
    logger.info("Initializing database schema...")

    try:
        # Create all tables with checkfirst=True (default behavior)
        Base.metadata.create_all(bind=eng, checkfirst=True)
        logger.info("Database schema initialized successfully")
    except OperationalError as e:
        error_msg = str(e)

        # If it's an "already exists" error, that's fine - tables/indexes are there
        if "already exists" in error_msg.lower():
            logger.info("Database schema already exists (some tables/indexes present)")

            # Try to create missing tables individually
            inspector = inspect(eng)
            existing_tables = inspector.get_table_names()

            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    try:
                        table.create(bind=eng, checkfirst=True)
                        logger.info(f"Created missing table: {table.name}")
                    except OperationalError as table_error:
                        if "already exists" not in str(table_error).lower():
                            logger.warning(f"Could not create table {table.name}: {table_error}")

            logger.info("Database schema verification complete")
        else:
            # Some other error - re-raise it
            logger.error(f"Database initialization failed: {e}")
            raise
