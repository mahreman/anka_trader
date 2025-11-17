#!/usr/bin/env python
"""
Incremental data ingestion script.

This script performs incremental updates by fetching only recent data:
- Price data (last N days)
- News articles (last N days)
- Macroeconomic indicators (last N days)

Schedule this script to run periodically (e.g., every 15 minutes, hourly, or daily)
using cron or a task scheduler.

Usage:
    python scripts/run_incremental_ingest.py

Cron example (run every 15 minutes):
    */15 * * * * cd /path/to/anka_trader && /path/to/.venv/bin/python scripts/run_incremental_ingest.py >> logs/incremental_ingest.log 2>&1

Configuration:
    Edit config/providers.yaml to set your API keys and data sources.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from otonom_trader.providers.ingest_providers import ingest_incremental_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/incremental_ingest.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """
    Run incremental data ingestion.
    """
    parser = argparse.ArgumentParser(
        description="Incremental data ingestion for Anka Trader"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to look back for incremental data. Default: 7",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="crypto",
        choices=["crypto", "fx", "equity", "all"],
        help="Asset universe to ingest. Default: crypto",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/providers.yaml",
        help="Path to providers config file. Default: config/providers.yaml",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ANKA TRADER - INCREMENTAL DATA INGESTION")
    logger.info("=" * 80)
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    try:
        # Run incremental ingestion
        results = ingest_incremental_all(
            providers_config_path=args.config,
            universe_name=args.universe,
            days_back=args.days_back,
        )

        logger.info("=" * 80)
        logger.info("INCREMENTAL INGESTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Price data: {results.get('price', {})}")
        logger.info(f"News articles: {results.get('news', 0)}")
        logger.info(f"Macro indicators: {results.get('macro', 0)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Incremental ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
