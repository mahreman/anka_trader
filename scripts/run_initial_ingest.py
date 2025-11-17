#!/usr/bin/env python
"""
Initial full data ingestion script.

This script performs a one-time full sync of:
- Price data (OHLCV from Binance/YFinance)
- News articles (from NewsAPI)
- Macroeconomic indicators (from FRED)

Run this once to bootstrap your database with historical data.

Usage:
    python scripts/run_initial_ingest.py

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

from otonom_trader.providers.ingest_providers import ingest_all_data_types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/initial_ingest.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """
    Run initial full data ingestion.
    """
    parser = argparse.ArgumentParser(
        description="Initial full data ingestion for Anka Trader"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for data ingestion (YYYY-MM-DD). Default: 2020-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data ingestion (YYYY-MM-DD). Default: today",
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
    logger.info("ANKA TRADER - INITIAL DATA INGESTION")
    logger.info("=" * 80)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date or 'today'}")
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    try:
        # Run ingestion
        results = ingest_all_data_types(
            start_date=args.start_date,
            end_date=args.end_date,
            providers_config_path=args.config,
            universe_name=args.universe,
        )

        logger.info("=" * 80)
        logger.info("INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Price data: {results.get('price', {})}")
        logger.info(f"News articles: {results.get('news', 0)}")
        logger.info(f"Macro indicators: {results.get('macro', 0)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
