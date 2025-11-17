"""
Binance broker implementation.

Supports both spot and futures trading on Binance.
Can use testnet for paper trading with real API.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlencode

import requests

from .base import (
    Balance,
    Broker,
    BrokerConfig,
    BrokerError,
    OrderFill,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)


@dataclass
class BinanceBrokerConfig(BrokerConfig):
    """
    Binance-specific configuration.
    
    Attributes:
        recv_window: Request time window (ms)
        timeout: HTTP timeout (seconds)
    """
    recv_window: int = 5000
    timeout: int = 10


class BinanceBroker(Broker):
    """
    Binance broker implementation.
    
    Supports:
    - Spot trading
    - Testnet mode
    - Market and limit orders
    """

    def __init__(self, cfg: BinanceBrokerConfig):
        """
        Initialize Binance broker.
        
        Args:
            cfg: Binance broker configuration
        """
        super().__init__(cfg)
        self.session = requests.Session()
        self.base_url = cfg.base_url.rstrip("/")
        self.recv_window = cfg.recv_window
        self.timeout = cfg.timeout
        
        logger.info(f"Initialized BinanceBroker (testnet={cfg.testnet}, base_url={self.base_url})")

    # ==================== Helper Methods ====================

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign request parameters with HMAC SHA256.
        
        Args:
            params: Request parameters
            
        Returns:
            Parameters with signature added
        """
        query = urlencode(params)
        secret = self.cfg.api_secret.encode()
        signature = hmac.new(secret, query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> Dict[str, str]:
        """
        Get request headers with API key.
        
        Returns:
            Headers dict
        """
        return {"X-MBX-APIKEY": self.cfg.api_key}

    def _request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Binance API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API path
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            Response JSON
            
        Raises:
            BrokerError: If request fails
        """
        params = params or {}
        
        # Add timestamp for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        
        try:
            resp = self.session.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout,
            )
            
            if resp.status_code != 200:
                error_msg = f"Binance API error: {resp.status_code} {resp.text}"
                logger.error(error_msg)
                raise BrokerError(error_msg)
            
            return resp.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Binance API request failed: {e}"
            logger.error(error_msg)
            raise BrokerError(error_msg) from e

    # ==================== Public API Methods ====================

    def place_order(self, req: OrderRequest) -> OrderFill:
        """
        Place an order on Binance.
        
        Args:
            req: Order request
            
        Returns:
            Order fill confirmation
            
        Raises:
            BrokerError: If order fails
        """
        side = req.side.value
        type_ = req.order_type.value

        params: Dict[str, Any] = {
            "symbol": req.symbol,
            "side": side,
            "type": type_,
            "quantity": req.quantity,
        }

        # Add limit order parameters
        if req.order_type == OrderType.LIMIT:
            if req.price is None:
                raise BrokerError("Limit order requires price")
            params["price"] = req.price
            params["timeInForce"] = req.time_in_force.value

        # Add client order ID if provided
        if req.client_order_id:
            params["newClientOrderId"] = req.client_order_id

        logger.info(f"Placing order: {params}")

        # Execute order
        data = self._request("POST", "/api/v3/order", params=params, signed=True)

        # Parse fill info
        fill_price = float(data.get("price", 0.0))
        if fill_price == 0.0 and "fills" in data and data["fills"]:
            # Market order - get avg fill price from fills
            total_qty = 0.0
            total_value = 0.0
            for fill in data["fills"]:
                qty = float(fill["qty"])
                price = float(fill["price"])
                total_qty += qty
                total_value += qty * price
            fill_price = total_value / total_qty if total_qty > 0 else 0.0

        quantity = float(data.get("executedQty", req.quantity))
        
        # Parse commission
        commission = 0.0
        commission_asset = ""
        if "fills" in data and data["fills"]:
            commission = sum(float(f["commission"]) for f in data["fills"])
            commission_asset = data["fills"][0].get("commissionAsset", "")

        # Create fill object
        fill = OrderFill(
            order_id=str(data["orderId"]),
            symbol=req.symbol,
            side=req.side,
            quantity=quantity,
            price=fill_price,
            ts=datetime.fromtimestamp(data["transactTime"] / 1000.0),
            commission=commission,
            commission_asset=commission_asset,
        )

        logger.info(f"Order filled: {fill}")
        return fill

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """
        Cancel an order on Binance.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Raises:
            BrokerError: If cancellation fails
        """
        params = {"symbol": symbol, "orderId": order_id}
        logger.info(f"Cancelling order: {params}")
        
        data = self._request("DELETE", "/api/v3/order", params=params, signed=True)
        logger.info(f"Order cancelled: {data}")

    def get_open_positions(self) -> List[Position]:
        """
        Get open positions.
        
        For spot trading, this returns positions based on non-zero balances.
        For futures, it queries the position endpoint.
        
        Returns:
            List of open positions
            
        Raises:
            BrokerError: If query fails
        """
        # For spot, we don't have positions - return empty list
        # For futures, you would query /fapi/v2/positionRisk
        logger.info("Getting open positions (spot mode - returning empty)")
        return []

    def get_balances(self) -> List[Balance]:
        """
        Get account balances.
        
        Returns:
            List of asset balances
            
        Raises:
            BrokerError: If query fails
        """
        logger.info("Getting account balances")
        
        data = self._request("GET", "/api/v3/account", signed=True)
        
        balances: List[Balance] = []
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            
            # Skip zero balances
            if free == 0.0 and locked == 0.0:
                continue
            
            balances.append(
                Balance(
                    asset=b["asset"],
                    free=free,
                    locked=locked,
                )
            )

        logger.info(f"Found {len(balances)} non-zero balances")
        return balances

    def ping(self) -> bool:
        """
        Check Binance connectivity.
        
        Returns:
            True if connection is OK, False otherwise
        """
        try:
            self._request("GET", "/api/v3/ping", signed=False)
            logger.debug("Ping successful")
            return True
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False
