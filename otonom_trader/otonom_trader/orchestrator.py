"""
High-level orchestrator for Otonom Trader.

- Her döngüde config/orchestrator.yaml dosyasını okur
- DaemonConfig oluşturur ve run_daemon_cycle() çalıştırır
- AlertEngine ile sistem sağlığını kontrol eder
- Varsayılan olarak 15 dakikada bir (900s) çalışır

Kullanım:
    python -m otonom_trader.orchestrator

veya
    python otonom_trader/otonom_trader/orchestrator.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from .data import get_session
from .daemon.daemon import DaemonConfig, run_daemon_cycle, get_or_create_paper_trader
from .alerts.engine import AlertEngine

logger = logging.getLogger(__name__)


# =========================
# Orchestrator Config
# =========================

@dataclass
class OrchestratorConfig:
    enabled: bool = True
    interval_seconds: int = 900  # 15 dakika
    use_ensemble: bool = True
    paper_trade_enabled: bool = True
    paper_trade_risk_pct: float = 1.0
    initial_cash: float = 100_000.0
    ingest_days_back: int = 7
    anomaly_lookback_days: int = 30

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            interval_seconds=int(data.get("interval_seconds", 900)),
            use_ensemble=bool(data.get("use_ensemble", True)),
            paper_trade_enabled=bool(data.get("paper_trade_enabled", True)),
            paper_trade_risk_pct=float(data.get("paper_trade_risk_pct", 1.0)),
            initial_cash=float(data.get("initial_cash", 100_000.0)),
            ingest_days_back=int(data.get("ingest_days_back", 7)),
            anomaly_lookback_days=int(data.get("anomaly_lookback_days", 30)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONFIG_DIR = Path("config")
ORCH_CONFIG_PATH = CONFIG_DIR / "orchestrator.yaml"


def load_orchestrator_config() -> OrchestratorConfig:
    """
    YAML'dan orchestrator config'i yükler.
    Dosya yoksa varsayılanları döner.
    """
    if not ORCH_CONFIG_PATH.exists():
        logger.warning("No orchestrator config found, using defaults.")
        return OrchestratorConfig()

    try:
        with ORCH_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        orch_data = data.get("orchestrator", data)
        cfg = OrchestratorConfig.from_dict(orch_data)
        return cfg
    except Exception as e:
        logger.error(f"Failed to load orchestrator config: {e}", exc_info=True)
        return OrchestratorConfig()


def ensure_default_config_exists() -> None:
    """
    Eğer config/orchestrator.yaml yoksa varsayılan bir tane yazar.
    Streamlit tarafında da aynı format kullanılıyor.
    """
    if ORCH_CONFIG_PATH.exists():
        return

    cfg = OrchestratorConfig()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with ORCH_CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"orchestrator": cfg.to_dict()}, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Created default orchestrator config at {ORCH_CONFIG_PATH}")


# =========================
# Orchestrator Loop
# =========================

def run_orchestrator_loop() -> None:
    """
    Ana orkestratör döngüsü.

    - Her turda config/orchestrator.yaml okunur (Streamlit'ten gelen değişiklikler alınır)
    - OrchestratorConfig -> DaemonConfig map edilir
    - run_daemon_cycle() çalıştırılır
    - AlertEngine.check_and_notify() ile sağlık kontrolü yapılır
    """
    ensure_default_config_exists()

    alerts = AlertEngine(alerts_config_path="config/alerts.yaml")
    cycle = 0

    logger.info("Starting orchestrator loop...")

    while True:
        cfg = load_orchestrator_config()

        if not cfg.enabled:
            # Orkestratör devre dışı: sadece uyku
            sleep_for = max(cfg.interval_seconds, 60)
            logger.info(
                "Orchestrator disabled in config. Sleeping for %ds before re-checking...",
                sleep_for,
            )
            # Sleep in 1-second intervals to allow Ctrl+C to work
            remaining = sleep_for
            try:
                while remaining > 0:
                    sleep_chunk = min(remaining, 1.0)
                    time.sleep(sleep_chunk)
                    remaining -= sleep_chunk
            except KeyboardInterrupt:
                logger.info("Orchestrator stopped by user.")
                break
            continue

        cycle += 1
        start_time = datetime.now()
        logger.info(
            "===== Orchestrator cycle #%d starting at %s =====",
            cycle,
            start_time.isoformat(),
        )

        try:
            session = next(get_session())
            try:
                # DaemonConfig oluştur
                daemon_cfg = DaemonConfig(
                    ingest_days_back=cfg.ingest_days_back,
                    anomaly_lookback_days=cfg.anomaly_lookback_days,
                    use_ensemble=cfg.use_ensemble,
                    paper_trade_enabled=cfg.paper_trade_enabled,
                    paper_trade_risk_pct=cfg.paper_trade_risk_pct,
                    initial_cash=cfg.initial_cash,
                )

                paper_trader = None
                if cfg.paper_trade_enabled:
                    paper_trader = get_or_create_paper_trader(session, cfg.initial_cash)

                # Tek cycle çalıştır
                run = run_daemon_cycle(session, daemon_cfg, paper_trader)
                logger.info(
                    "Daemon cycle completed: bars=%s anomalies=%s decisions=%s trades=%s status=%s",
                    run.bars_ingested,
                    run.anomalies_detected,
                    run.decisions_made,
                    run.trades_executed,
                    run.status,
                )

                # Alert engine ile sağlık kontrolü
                alerts.check_and_notify(datetime.now())

            finally:
                session.close()

        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user.")
            break
        except Exception as e:
            logger.error(f"Orchestrator cycle #{cycle} failed: {e}", exc_info=True)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        target_interval = max(cfg.interval_seconds, 60)
        sleep_for = max(target_interval - elapsed, 0)

        logger.info(
            "Cycle #%d finished in %.1fs. Sleeping %.1fs before next cycle...",
            cycle,
            elapsed,
            sleep_for,
        )

        # Sleep in 1-second intervals to allow Ctrl+C to work
        remaining = sleep_for
        try:
            while remaining > 0:
                sleep_chunk = min(remaining, 1.0)
                time.sleep(sleep_chunk)
                remaining -= sleep_chunk
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
