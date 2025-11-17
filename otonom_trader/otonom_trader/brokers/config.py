"""
Broker configuration loader.

Loads broker settings from config/broker.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """
    Broker configuration.

    Attributes:
        broker_type: Broker type (dummy, binance, alpaca, ibkr)
        shadow_mode: If true, log but don't execute real orders
        api_key: Broker API key
        api_secret: Broker API secret
        base_url: Broker API base URL
        use_testnet: Use testnet/paper account
        config: Raw config dictionary

    Example:
        >>> config = BrokerConfig.from_yaml("config/broker.yaml")
        >>> print(config.broker_type)
        'dummy'
    """

    broker_type: str
    shadow_mode: bool
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    use_testnet: bool = True
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, path: str | Path = "config/broker.yaml") -> "BrokerConfig":
        """
        Load broker config from YAML file.

        Args:
            path: Path to broker.yaml

        Returns:
            BrokerConfig instance

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid

        Example:
            >>> config = BrokerConfig.from_yaml()
            >>> print(f"Using {config.broker_type} broker")
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Broker config not found: {path}, using defaults")
            return cls(
                broker_type="dummy",
                shadow_mode=True,
                config={},
            )

        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        broker_section = raw_config.get("broker", {})
        broker_type = broker_section.get("type", "dummy")
        shadow_mode = broker_section.get("shadow_mode", True)

        # Load broker-specific config
        broker_specific = broker_section.get(broker_type, {})

        api_key = broker_specific.get("api_key")
        api_secret = broker_specific.get("api_secret")
        base_url = broker_specific.get("base_url")
        use_testnet = broker_specific.get("use_testnet", True)

        # Validate API credentials if not in shadow mode
        if not shadow_mode and broker_type != "dummy":
            if not api_key or not api_secret:
                raise ValueError(
                    f"API credentials required for {broker_type} broker in live mode. "
                    f"Set api_key and api_secret in {path}"
                )

        logger.info(
            f"Loaded broker config: type={broker_type}, shadow_mode={shadow_mode}, "
            f"testnet={use_testnet}"
        )

        return cls(
            broker_type=broker_type,
            shadow_mode=shadow_mode,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            use_testnet=use_testnet,
            config=raw_config,
        )

    def get_risk_guardrails(self) -> "RiskGuardrails":
        """
        Get risk guardrails configuration.

        Returns:
            RiskGuardrails instance
        """
        risk_config = self.config.get("risk_guardrails", {})
        return RiskGuardrails.from_dict(risk_config)


@dataclass
class RiskGuardrails:
    """
    Risk guardrails configuration.

    Attributes:
        max_notional_per_order: Maximum order size (USD)
        max_open_risk_pct: Maximum open risk (% of equity)
        max_total_positions: Maximum number of positions
        max_position_size_pct: Maximum position size per symbol (% of equity)
        symbol_blacklist: Symbols that cannot be traded
        require_confirmation_above: Require manual confirmation above this amount
        kill_switch: If true, halt all trading

    Example:
        >>> guardrails = RiskGuardrails(max_notional_per_order=10000)
        >>> guardrails.check_order_notional(5000)  # OK
        >>> guardrails.check_order_notional(15000)  # Raises ValueError
    """

    max_notional_per_order: float = 10000.0
    max_open_risk_pct: float = 25.0
    max_total_positions: int = 10
    max_position_size_pct: float = 10.0
    symbol_blacklist: List[str] = None
    require_confirmation_above: float = 50000.0
    kill_switch: bool = False

    def __post_init__(self):
        """Initialize default blacklist."""
        if self.symbol_blacklist is None:
            self.symbol_blacklist = []

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RiskGuardrails":
        """
        Create RiskGuardrails from config dictionary.

        Args:
            config: Risk guardrails config

        Returns:
            RiskGuardrails instance
        """
        return cls(
            max_notional_per_order=config.get("max_notional_per_order", 10000.0),
            max_open_risk_pct=config.get("max_open_risk_pct", 25.0),
            max_total_positions=config.get("max_total_positions", 10),
            max_position_size_pct=config.get("max_position_size_pct", 10.0),
            symbol_blacklist=config.get("symbol_blacklist", []),
            require_confirmation_above=config.get("require_confirmation_above", 50000.0),
            kill_switch=config.get("kill_switch", False),
        )

    def check_order(
        self,
        symbol: str,
        notional: float,
        current_equity: float,
        current_positions: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if order passes risk guardrails.

        Args:
            symbol: Symbol to trade
            notional: Order notional value (USD)
            current_equity: Current account equity
            current_positions: Current number of open positions

        Returns:
            (is_ok, error_message) tuple

        Example:
            >>> guardrails = RiskGuardrails(max_notional_per_order=10000)
            >>> ok, msg = guardrails.check_order("BTC-USD", 5000, 100000, 3)
            >>> if not ok:
            ...     print(f"Order rejected: {msg}")
        """
        # Kill switch
        if self.kill_switch:
            return False, "KILL SWITCH ACTIVE - all trading halted"

        # Symbol blacklist
        if symbol in self.symbol_blacklist:
            return False, f"Symbol {symbol} is blacklisted"

        # Max notional per order
        if notional > self.max_notional_per_order:
            return (
                False,
                f"Order notional ${notional:.2f} exceeds max ${self.max_notional_per_order:.2f}",
            )

        # Max position size as % of equity
        position_pct = (notional / current_equity) * 100 if current_equity > 0 else 0
        if position_pct > self.max_position_size_pct:
            return (
                False,
                f"Position size {position_pct:.1f}% exceeds max {self.max_position_size_pct:.1f}%",
            )

        # Max total positions
        if current_positions >= self.max_total_positions:
            return (
                False,
                f"Total positions {current_positions} exceeds max {self.max_total_positions}",
            )

        # Require confirmation for large orders
        if notional > self.require_confirmation_above:
            return (
                False,
                f"Order ${notional:.2f} requires manual confirmation (threshold: ${self.require_confirmation_above:.2f})",
            )

        return True, None


def get_broker_config(path: str | Path = "config/broker.yaml") -> BrokerConfig:
    """
    Load broker configuration.

    Args:
        path: Path to broker.yaml

    Returns:
        BrokerConfig instance

    Example:
        >>> config = get_broker_config()
        >>> print(f"Broker: {config.broker_type}")
    """
    return BrokerConfig.from_yaml(path)
