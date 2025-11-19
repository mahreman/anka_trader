"""
High-level orchestrator for Otonom Trader.

- Reads config/orchestrator.yaml before each cycle
- Runs the ingestion/anomaly/decision pipeline via DaemonConfig
- Depending on execution mode:
    - paper: only paper trading
    - shadow: paper + real broker execution
    - live: only real broker execution
- Uses AlertEngine for health checks
- Runs every 15 minutes (900s) by default

Usage:
    python -m otonom_trader.orchestrator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Set

import yaml

from .data import get_session, get_engine, init_db
from .daemon.daemon import DaemonConfig, run_daemon_cycle, get_or_create_paper_trader
from .daemon.trading_daemon import TradingDaemon, ExecutionMode
from .alerts.engine import AlertEngine
from .utils import utc_now


logger = logging.getLogger(__name__)


# =========================
# Orchestrator Config
# =========================


@dataclass
class OrchestratorConfig:
    # global
    enabled: bool = True
    interval_seconds: int = 900  # 15 minutes

    # execution mode: paper / shadow / live
    mode: str = "paper"

    # paths
    db_path: str = "data/trader.db"
    strategy_path: str = "strategies/baseline_v1.0.yaml"
    broker_config_path: str = "config/broker.yaml"

    # daemon (ingestion + anomaly + ensemble + paper trader)
    use_ensemble: bool = True
    paper_trade_enabled: bool = True
    paper_trade_risk_pct: float = 1.0
    initial_cash: float = 100_000.0
    ingest_days_back: int = 7
    anomaly_lookback_days: int = 30
    price_interval: str = "15m"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            interval_seconds=int(data.get("interval_seconds", 900)),
            mode=str(data.get("mode", "paper")).lower(),
            db_path=str(data.get("db_path", "data/trader.db")),
            strategy_path=str(data.get("strategy_path", "strategies/baseline_v1.0.yaml")),
            broker_config_path=str(data.get("broker_config_path", "config/broker.yaml")),
            use_ensemble=bool(data.get("use_ensemble", True)),
            paper_trade_enabled=bool(data.get("paper_trade_enabled", True)),
            paper_trade_risk_pct=float(data.get("paper_trade_risk_pct", 1.0)),
            initial_cash=float(data.get("initial_cash", 100_000.0)),
            ingest_days_back=int(data.get("ingest_days_back", 7)),
            anomaly_lookback_days=int(data.get("anomaly_lookback_days", 30)),
            price_interval=str(data.get("price_interval", "15m")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG_DIR = Path("config")
ORCH_CONFIG_PATH = CONFIG_DIR / "orchestrator.yaml"


def load_orchestrator_config() -> OrchestratorConfig:
    """Load orchestrator config from YAML (defaults if file missing)."""
    if not ORCH_CONFIG_PATH.exists():
        logger.warning("No orchestrator config found, using defaults.")
        return OrchestratorConfig()

    try:
        with ORCH_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        orch_data = data.get("orchestrator", data)
        cfg = OrchestratorConfig.from_dict(orch_data)
        return cfg
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load orchestrator config: %s", exc, exc_info=True)
        return OrchestratorConfig()


def ensure_default_config_exists() -> None:
    """Create default orchestrator config if missing."""
    if ORCH_CONFIG_PATH.exists():
        return

    cfg = OrchestratorConfig()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with ORCH_CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"orchestrator": cfg.to_dict()}, f, sort_keys=False, allow_unicode=True)
    logger.info("Created default orchestrator config at %s", ORCH_CONFIG_PATH)


def save_orchestrator_config(cfg: OrchestratorConfig) -> None:
    """Persist orchestrator config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with ORCH_CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"orchestrator": cfg.to_dict()}, f, sort_keys=False, allow_unicode=True)


# =========================
# Mode helpers
# =========================


def _normalize_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m not in {"paper", "shadow", "live"}:
        return "paper"
    return m


def _build_execution_mode_for_trading_daemon(mode: str) -> ExecutionMode:
    """Map orchestrator mode to TradingDaemon ExecutionMode."""
    m = _normalize_mode(mode)
    if m == "shadow":
        return ExecutionMode.SHADOW
    if m == "live":
        return ExecutionMode.LIVE
    return ExecutionMode.PAPER


# =========================
# Orchestrator Loop
# =========================


def run_orchestrator_loop() -> None:
    """Main orchestrator loop."""
    ensure_default_config_exists()

    alerts = AlertEngine(alerts_config_path="config/alerts.yaml")
    initialized_db_paths: Set[str] = set()
    cycle = 0

    logger.info("Starting orchestrator loop...")

    while True:
        cfg = load_orchestrator_config()
        mode = _normalize_mode(cfg.mode)

        if not cfg.enabled:
            sleep_for = max(cfg.interval_seconds, 60)
            logger.info(
                "Orchestrator disabled in config. Sleeping for %ds before re-checking...",
                sleep_for,
            )
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                logger.info("Orchestrator stopped by user.")
                break
            continue

        cycle += 1
        start_time = utc_now()
        logger.info(
            "===== Orchestrator cycle #%d starting at %s (mode=%s) =====",
            cycle,
            start_time.isoformat(),
            mode,
        )

        try:
            engine = get_engine(cfg.db_path)
            if cfg.db_path not in initialized_db_paths:
                init_db(engine)
                initialized_db_paths.add(cfg.db_path)

            # 1) Ingestion + anomaly + ensemble + (optional) paper trader
            with get_session() as session:
                # mode determines paper trade availability
                if mode == "paper":
                    paper_enabled = True
                elif mode == "shadow":
                    paper_enabled = True
                else:  # live
                    paper_enabled = False

                # Build the daemon configuration once per cycle so we pass a
                # clean, explicit snapshot of the current orchestrator
                # settings into the daemon layer.
                daemon_cfg = DaemonConfig(
                    ingest_days_back=cfg.ingest_days_back,
                    anomaly_lookback_days=cfg.anomaly_lookback_days,
                    use_ensemble=cfg.use_ensemble,
                    paper_trade_enabled=paper_enabled,
                    paper_trade_risk_pct=cfg.paper_trade_risk_pct,
                    initial_cash=cfg.initial_cash,
                    price_interval=cfg.price_interval,
                )

                paper_trader = None
                if paper_enabled:
                    paper_trader = get_or_create_paper_trader(
                        session=session,
                        initial_cash=cfg.initial_cash,
                        price_interval=cfg.price_interval,
                    )

                run = run_daemon_cycle(
                    session=session,
                    config=daemon_cfg,
                    paper_trader=paper_trader,
                )
                logger.info(
                    "Daemon cycle completed: bars=%s anomalies=%s decisions=%s trades=%s status=%s",
                    run.bars_ingested,
                    run.anomalies_detected,
                    run.decisions_made,
                    run.trades_executed,
                    run.status,
                )

            # 2) If mode is shadow/live trigger TradingDaemon for broker execution
            if mode in {"shadow", "live"}:
                exec_mode = _build_execution_mode_for_trading_daemon(mode)
                td = TradingDaemon(
                    db_path=cfg.db_path,
                    strategy_path=cfg.strategy_path,
                    broker_config_path=cfg.broker_config_path,
                    mode=exec_mode,
                )
                td.run_once()
                logger.info("TradingDaemon run_once() completed (mode=%s).", mode)

            # 3) Alert engine health checks
            alerts.check_and_notify()

        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user.")
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Orchestrator cycle #%d failed: %s", cycle, exc, exc_info=True)

        end_time = utc_now()
        elapsed = (end_time - start_time).total_seconds()
        target_interval = max(cfg.interval_seconds, 60)
        sleep_for = max(target_interval - elapsed, 0)

        logger.info(
            "Cycle #%d finished in %.1fs. Sleeping %.1fs before next cycle...",
            cycle,
            elapsed,
            sleep_for,
        )

        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user.")
            break

    logger.info("Orchestrator loop exited.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    run_orchestrator_loop()


if __name__ == "__main__":
    main()
