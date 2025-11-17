"""
Binance Spot broker adapter.

Supports both testnet and live trading via Binance REST API.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

from .base import Broker, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class BinanceBroker(Broker):
    """
    Binance Spot broker implementation.

    Supports:
    - Spot trading (MARKET and LIMIT orders)
    - Testnet and live environments
    - Order placement, cancellation, and status queries
    - Position and balance queries

    Example:
        >>> broker = BinanceBroker(
        ...     api_key="your_api_key",
        ...     api_secret="your_api_secret",
        ...     use_testnet=True
        ... )
        >>> req = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.001)
        >>> result = broker.place_order(req)
        >>> print(result.order_id)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        use_testnet: bool = True,
    ):
        """
        Initialize Binance broker.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            base_url: Optional custom base URL
            use_testnet: Use testnet (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet

        # Set base URL
        if base_url:
            self.base_url = base_url
        elif use_testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        logger.info(
            f"BinanceBroker initialized: base_url={self.base_url}, "
            f"testnet={use_testnet}"
        )

    def _sign_request(self, params: Dict) -> Dict:
        """
        Sign request with HMAC SHA256.

        Args:
            params: Request parameters

        Returns:
            Signed parameters with signature
        """
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Dict:
        """
        Make API request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign request

        Returns:
            Response JSON

        Raises:
            Exception: If request fails
        """
        if params is None:
            params = {}

        if signed:
            params = self._sign_request(params)

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(url, params=params, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request failed: {e}")
            raise

    def place_order(self, req: OrderRequest) -> OrderResult:
        """
        Place order on Binance.

        Args:
            req: Order request

        Returns:
            OrderResult with execution status

        Example:
            >>> req = OrderRequest(
            ...     symbol="BTCUSDT",
            ...     side="BUY",
            ...     qty=0.001,
            ...     price=50000.0,
            ...     order_type="LIMIT"
            ... )
            >>> result = broker.place_order(req)
        """
        # Convert symbol format (BTC-USD -> BTCUSDT)
        symbol = req.symbol.replace("-", "").upper()

        # Build order parameters
        params = {
            "symbol": symbol,
            "side": req.side.upper(),
            "type": req.order_type.upper() if req.order_type else "MARKET",
            "quantity": req.qty,
        }

        # Add price for LIMIT orders
        if params["type"] == "LIMIT":
            if req.price is None:
                return OrderResult(
                    ok=False,
                    message="Price required for LIMIT orders",
                )
            params["price"] = req.price
            params["timeInForce"] = "GTC"  # Good Till Cancel

        try:
            logger.info(f"Placing Binance order: {params}")
            response = self._request("POST", "/api/v3/order", params, signed=True)

            return OrderResult(
                ok=True,
                order_id=str(response["orderId"]),
                message=f"Order placed: {response['status']}",
                filled_qty=float(response.get("executedQty", 0)),
                avg_price=float(response.get("price", 0)) if response.get("price") else None,
            )

        except Exception as e:
            logger.error(f"Failed to place Binance order: {e}")
            return OrderResult(
                ok=False,
                message=f"Order failed: {str(e)}",
            )

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> OrderResult:
        """
        Cancel order on Binance.

        Args:
            order_id: Binance order ID
            symbol: Symbol (required for Binance)

        Returns:
            OrderResult with cancellation status

        Example:
            >>> result = broker.cancel_order("12345", symbol="BTCUSDT")
        """
        if not symbol:
            return OrderResult(
                ok=False,
                order_id=order_id,
                message="Symbol required for Binance order cancellation",
            )

        # Convert symbol format
        symbol = symbol.replace("-", "").upper()

        params = {
            "symbol": symbol,
            "orderId": order_id,
        }

        try:
            logger.info(f"Canceling Binance order: {order_id} ({symbol})")
            response = self._request("DELETE", "/api/v3/order", params, signed=True)

            return OrderResult(
                ok=True,
                order_id=str(response["orderId"]),
                message=f"Order canceled: {response['status']}",
            )

        except Exception as e:
            logger.error(f"Failed to cancel Binance order: {e}")
            return OrderResult(
                ok=False,
                order_id=order_id,
                message=f"Cancel failed: {str(e)}",
            )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Get open orders from Binance.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders

        Example:
            >>> orders = broker.get_open_orders(symbol="BTCUSDT")
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "").upper()

        try:
            response = self._request("GET", "/api/v3/openOrders", params, signed=True)

            orders = []
            for order in response:
                orders.append(
                    OrderResult(
                        ok=True,
                        order_id=str(order["orderId"]),
                        message=order["status"],
                        filled_qty=float(order.get("executedQty", 0)),
                        avg_price=float(order.get("price", 0)) if order.get("price") else None,
                    )
                )

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(self) -> List[dict]:
        """
        Get current positions (account balances) from Binance.

        Returns:
            List of position dictionaries

        Example:
            >>> positions = broker.get_positions()
            >>> for pos in positions:
            ...     print(f"{pos['symbol']}: {pos['qty']}")
        """
        try:
            response = self._request("GET", "/api/v3/account", signed=True)

            positions = []
            for balance in response.get("balances", []):
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked

                if total > 0:  # Only include non-zero balances
                    positions.append({
                        "symbol": balance["asset"],
                        "qty": total,
                        "free": free,
                        "locked": locked,
                        "avg_entry_price": None,  # Not available from Binance API
                        "market_value": None,
                    })

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_balance(self) -> dict:
        """
        Get account balance from Binance.

        Returns:
            Dictionary with balances

        Example:
            >>> balance = broker.get_account_balance()
            >>> print(f"USDT: {balance['USDT']}")
        """
        try:
            response = self._request("GET", "/api/v3/account", signed=True)

            balances = {}
            for balance in response.get("balances", []):
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked

                if total > 0:
                    balances[asset] = {
                        "free": free,
                        "locked": locked,
                        "total": total,
                    }

            # Calculate total equity in USDT (simplified)
            usdt_balance = balances.get("USDT", {}).get("total", 0.0)

            return {
                "cash": usdt_balance,
                "equity": usdt_balance,  # Simplified: not counting other assets
                "buying_power": balances.get("USDT", {}).get("free", 0.0),
                "balances": balances,
            }

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {
                "cash": 0.0,
                "equity": 0.0,
                "buying_power": 0.0,
                "balances": {},
            }

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Current price or None if not available

        Example:
            >>> price = broker.get_ticker_price("BTCUSDT")
            >>> print(f"BTC price: ${price}")
        """
        symbol = symbol.replace("-", "").upper()

        try:
            response = self._request(
                "GET",
                "/api/v3/ticker/price",
                params={"symbol": symbol},
                signed=False,
            )
            return float(response["price"])

        except Exception as e:
            logger.error(f"Failed to get ticker price for {symbol}: {e}")
            return None
