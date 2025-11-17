#!/usr/bin/env python3
"""
P2 Migration: Add ensemble fields to Decision table.

This script adds three new fields to the decisions table:
1. p_up: Ensemble probability of upward price movement (0-1)
2. disagreement: Analyst disagreement metric (0-1)
3. analyst_signals: JSON string of individual analyst signals

Run this BEFORE using the P2 ensemble features.

Usage:
    python scripts/migrate_p2_ensemble.py
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from otonom_trader.data import get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_p2_ensemble():
    """Add P2 ensemble fields to Decision table."""

    with next(get_session()) as session:
        try:
            # Check if columns already exist
            result = session.execute(text("PRAGMA table_info(decisions)"))
            columns = [row[1] for row in result]

            if 'p_up' in columns:
                logger.info("Migration already applied - p_up column exists")
                return

            # Add new columns
            logger.info("Adding p_up column...")
            session.execute(text("ALTER TABLE decisions ADD COLUMN p_up REAL DEFAULT NULL"))

            logger.info("Adding disagreement column...")
            session.execute(text("ALTER TABLE decisions ADD COLUMN disagreement REAL DEFAULT NULL"))

            logger.info("Adding analyst_signals column...")
            session.execute(text("ALTER TABLE decisions ADD COLUMN analyst_signals TEXT DEFAULT NULL"))

            session.commit()

            # Verify
            result = session.execute(text("PRAGMA table_info(decisions)"))
            columns = [row[1] for row in result]

            if all(col in columns for col in ['p_up', 'disagreement', 'analyst_signals']):
                logger.info("✓ Migration successful!")
                logger.info("  Added columns: p_up, disagreement, analyst_signals")
            else:
                logger.error("✗ Migration verification failed")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()
            sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting P2 ensemble migration...")
    migrate_p2_ensemble()
    logger.info("Done!")
