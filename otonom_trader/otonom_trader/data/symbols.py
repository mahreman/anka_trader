"""
Asset symbols and metadata for P0 version.
"""
from typing import List

from ..domain import Asset, AssetClass


def get_p0_assets() -> List[Asset]:
    """
    Returns the list of assets to track in P0 version.

    P0 Assets:
    - SUGAR: Commodity (using sugar futures ticker)
    - WTI: Oil commodity (West Texas Intermediate)
    - GOLD: Precious metal commodity
    - SP500: Stock index
    - BTC: Bitcoin cryptocurrency
    - ETH: Ethereum cryptocurrency

    Returns:
        List of Asset objects with metadata
    """
    return [
        Asset(
            symbol="SB=F",  # Sugar #11 Futures
            name="Sugar",
            asset_class=AssetClass.COMMODITY,
            base_currency="USD",
        ),
        Asset(
            symbol="CL=F",  # Crude Oil WTI Futures
            name="WTI Crude Oil",
            asset_class=AssetClass.COMMODITY,
            base_currency="USD",
        ),
        Asset(
            symbol="GC=F",  # Gold Futures
            name="Gold",
            asset_class=AssetClass.COMMODITY,
            base_currency="USD",
        ),
        Asset(
            symbol="^GSPC",  # S&P 500 Index
            name="S&P 500",
            asset_class=AssetClass.INDEX,
            base_currency="USD",
        ),
        Asset(
            symbol="BTC-USD",  # Bitcoin
            name="Bitcoin",
            asset_class=AssetClass.CRYPTO,
            base_currency="USD",
        ),
        Asset(
            symbol="ETH-USD",  # Ethereum
            name="Ethereum",
            asset_class=AssetClass.CRYPTO,
            base_currency="USD",
        ),
    ]


def get_asset_by_symbol(symbol: str) -> Asset:
    """
    Get a specific asset by its symbol.

    Args:
        symbol: The trading symbol

    Returns:
        Asset object

    Raises:
        ValueError: If symbol not found
    """
    assets = get_p0_assets()
    for asset in assets:
        if asset.symbol == symbol:
            return asset
    raise ValueError(f"Asset with symbol '{symbol}' not found in P0 asset list")
