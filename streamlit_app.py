"""
Otonom Trader - Portfolio Monitoring Dashboard

A read-only Streamlit dashboard for monitoring:
- Recent anomalies
- Trading decisions with multi-analyst explanations
- Portfolio equity curve and drawdown
- Risk metrics and constraints

Usage:
    streamlit run streamlit_app.py
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from sqlalchemy import func
from sqlalchemy.orm import Session

from otonom_trader.data import get_session, get_engine
from otonom_trader.data.schema import (
    Symbol,
    Anomaly,
    Decision,
    PortfolioSnapshot,
    PaperTrade,
    DaemonRun,
    DailyBar,
    NewsArticle,
)

PROVIDERS_PATH = Path("config/providers.yaml")
ORCH_CONFIG_PATH = Path("config/orchestrator.yaml")
BROKER_CONFIG_PATH = Path("config/broker.yaml")


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


DEFAULT_ORCH_CONFIG: dict = {
    "enabled": True,
    "interval_seconds": 900,
    "mode": "paper",
    "db_path": "data/trader.db",
    "strategy_path": "strategies/baseline_v1.0.yaml",
    "broker_config_path": "config/broker.yaml",
    "use_ensemble": True,
    "paper_trade_enabled": True,
    "paper_trade_risk_pct": 1.0,
    "initial_cash": 100_000.0,
    "ingest_days_back": 7,
    "anomaly_lookback_days": 30,
    "price_interval": "15m",
}


def load_orchestrator_config() -> dict:
    """
    Load orchestrator.yaml and merge with defaults.
    """
    cfg = DEFAULT_ORCH_CONFIG.copy()
    cfg.update(load_yaml(ORCH_CONFIG_PATH))
    return cfg


def get_app_session() -> Session:
    """Dashboard’ın, orchestrator ile aynı veritabanına bağlanması için
    kullanılan session helper.

    - orchestrator.yaml içinden db_path okunur
    - get_engine(db_path) ile global engine aynı DB’ye ayarlanır
    - Son olarak get_session() ile o engine’e bağlı Session döner
    """
    cfg = load_orchestrator_config()
    # Orchestrator'ın kullandığı DB'yi zorla aktif et
    get_engine(cfg.get("db_path", "data/trader.db"))
    return get_session()


def save_orchestrator_config(cfg: dict) -> None:
    save_yaml(ORCH_CONFIG_PATH, cfg)


def load_providers_config() -> dict:
    return load_yaml(PROVIDERS_PATH)


def load_broker_config() -> dict:
    return load_yaml(BROKER_CONFIG_PATH)


# -----------------------------------------------------------------------------
# DB query helpers
# -----------------------------------------------------------------------------


def get_universe(session: Session) -> List[Symbol]:
    return (
        session.query(Symbol)
        .filter(Symbol.is_active.is_(True))
        .order_by(Symbol.symbol.asc())
        .all()
    )


def get_latest_daemon_run(session: Session) -> Optional[DaemonRun]:
    return (
        session.query(DaemonRun)
        .order_by(DaemonRun.timestamp.desc())
        .limit(1)
        .one_or_none()
    )


def get_daemon_stats(session: Session) -> Tuple[int, int, float]:
    """
    Returns: total_runs, success_runs, avg_duration
    """
    total_runs = session.query(func.count(DaemonRun.id)).scalar() or 0
    success_runs = (
        session.query(func.count(DaemonRun.id))
        .filter(DaemonRun.status == "SUCCESS")
        .scalar()
        or 0
    )
    avg_duration = (
        session.query(func.avg(DaemonRun.duration_seconds)).scalar() or 0.0
    )
    return total_runs, success_runs, avg_duration


def get_recent_anomalies(
    session: Session,
    days_back: int = 7,
    limit: int = 100,
    symbol_filter: Optional[str] = None,
) -> List[Anomaly]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    q = (
        session.query(Anomaly)
        .filter(Anomaly.timestamp >= cutoff)
        .order_by(Anomaly.timestamp.desc())
    )
    if symbol_filter:
        q = q.join(Symbol).filter(Symbol.symbol == symbol_filter)
    return q.limit(limit).all()


def get_recent_decisions(
    session: Session,
    days_back: int = 7,
    limit: int = 100,
    symbol_filter: Optional[str] = None,
) -> List[Decision]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    q = (
        session.query(Decision)
        .filter(Decision.timestamp >= cutoff)
        .order_by(Decision.timestamp.desc())
    )
    if symbol_filter:
        q = q.join(Symbol).filter(Symbol.symbol == symbol_filter)
    return q.limit(limit).all()


def get_portfolio_history(
    session: Session, days_back: int = 30
) -> List[PortfolioSnapshot]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return (
        session.query(PortfolioSnapshot)
        .filter(PortfolioSnapshot.timestamp >= cutoff)
        .order_by(PortfolioSnapshot.timestamp.asc())
        .all()
    )


def get_recent_trades(
    session: Session, days_back: int = 30, limit: int = 200
) -> List[PaperTrade]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return (
        session.query(PaperTrade)
        .filter(PaperTrade.opened_at >= cutoff)
        .order_by(PaperTrade.opened_at.desc())
        .limit(limit)
        .all()
    )


def get_recent_news(
    session: Session, symbol: Optional[str] = None, days_back: int = 7
) -> List[NewsArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    q = (
        session.query(NewsArticle)
        .filter(NewsArticle.published_at >= cutoff)
        .order_by(NewsArticle.published_at.desc())
    )
    if symbol:
        q = q.filter(NewsArticle.symbol == symbol)
    return q.limit(100).all()


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def format_timedelta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def decision_color(decision: str) -> str:
    decision = (decision or "").upper()
    if decision in ("BUY", "LONG"):
        return "🟢 BUY"
    if decision in ("SELL", "SHORT"):
        return "🔴 SELL"
    if decision == "HOLD":
        return "⚪ HOLD"
    return decision or "N/A"


# -----------------------------------------------------------------------------
# Streamlit UI sections
# -----------------------------------------------------------------------------


def render_overview(session: Session, orch_cfg: dict) -> None:
    st.header("Overview")

    latest_run = get_latest_daemon_run(session)
    universe = get_universe(session)

    if latest_run is None:
        st.info("No daemon runs found. Run the orchestrator first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mode", orch_cfg.get("mode", "paper"))
        st.metric("Universe Size", len(universe))
    with col2:
        st.metric(
            "Last Run",
            latest_run.timestamp.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        )
        st.metric("Bars Ingested (last run)", latest_run.bars_ingested or 0)
    with col3:
        st.metric("Anomalies (last run)", latest_run.anomalies_detected or 0)
        st.metric("Decisions (last run)", latest_run.decisions_made or 0)


def render_daemon_health(session: Session) -> None:
    st.header("Daemon Health")

    total_runs, success_runs, avg_duration = get_daemon_stats(session)
    success_rate = (success_runs / total_runs * 100.0) if total_runs else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        st.metric("Total Trades", session.query(func.count(PaperTrade.id)).scalar() or 0)
    with col4:
        st.metric("Avg Duration", format_timedelta(avg_duration))

    latest_run = get_latest_daemon_run(session)
    latest_run_caption = (
        f"Latest Run: {latest_run.timestamp if latest_run else 'N/A'} | "
        f"Status: {latest_run.status if latest_run else 'N/A'} | "
        f"Bars: {latest_run.bars_ingested if latest_run else 0} | "
        f"Anomalies: {latest_run.anomalies_detected if latest_run else 0} | "
        f"Decisions: {latest_run.decisions_made if latest_run else 0} | "
        f"Trades: {latest_run.trades_executed if latest_run else 0}"
    )
    st.caption(latest_run_caption)


def render_portfolio_performance(session: Session) -> None:
    st.header("Portfolio Performance")

    snapshots = get_portfolio_history(session, days_back=60)
    if not snapshots:
        st.info("No portfolio history available yet.")
        return

    df = pd.DataFrame(
        [
            {
                "timestamp": s.timestamp,
                "equity": s.equity,
                "cash": s.cash,
                "exposure": s.exposure,
                "drawdown": s.max_drawdown,
            }
            for s in snapshots
        ]
    )

    df.set_index("timestamp", inplace=True)

    st.line_chart(df[["equity", "cash"]])
    st.area_chart(df[["drawdown"]])


def render_recent_activity(session: Session, symbol_filter: Optional[str]) -> None:
    st.header("Recent Activity")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomalies")
        anomalies = get_recent_anomalies(
            session, days_back=7, limit=100, symbol_filter=symbol_filter
        )
        if not anomalies:
            st.write("No recent anomalies.")
        else:
            rows = []
            for a in anomalies:
                rows.append(
                    {
                        "Time": a.timestamp,
                        "Symbol": a.symbol,
                        "Score": f"{a.score:.2f}",
                        "Direction": a.direction,
                        "Summary": a.summary or "",
                    }
                )
            st.dataframe(pd.DataFrame(rows))

    with col2:
        st.subheader("Decisions")
        decisions = get_recent_decisions(
            session, days_back=7, limit=100, symbol_filter=symbol_filter
        )
        if not decisions:
            st.write("No recent decisions.")
        else:
            rows = []
            for d in decisions:
                rows.append(
                    {
                        "Time": d.timestamp,
                        "Symbol": d.symbol,
                        "Decision": decision_color(d.action),
                        "Confidence": f"{(d.confidence or 0.0) * 100:.1f}%",
                        "Reason": (d.rationale or "")[:200],
                    }
                )
            st.dataframe(pd.DataFrame(rows))


def render_trades(session: Session, symbol_filter: Optional[str]) -> None:
    st.header("Recent Paper Trades")

    trades = get_recent_trades(session, days_back=30, limit=200)
    if symbol_filter:
        trades = [t for t in trades if t.symbol == symbol_filter]

    if not trades:
        st.info("No trades executed yet.")
        return

    rows = []
    for t in trades:
        rows.append(
            {
                "Opened": t.opened_at,
                "Closed": t.closed_at,
                "Symbol": t.symbol,
                "Side": t.side,
                "Qty": t.quantity,
                "Entry": t.entry_price,
                "Exit": t.exit_price,
                "PnL": t.realized_pnl,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df)


def render_news(session: Session, symbol_filter: Optional[str]) -> None:
    st.header("News Headlines")

    news = get_recent_news(session, symbol_filter, days_back=7)
    if not news:
        st.info("Bu sembol için haber bulunamadı ya da ingest edilmemiş.")
        return

    rows = []
    for n in news:
        rows.append(
            {
                "Time": n.published_at,
                "Source": n.source,
                "Title": n.title,
                "Symbol": n.symbol,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df)


def render_raw_prices(session: Session, symbol_filter: Optional[str]) -> None:
    st.header("Raw Daily Prices")

    if not symbol_filter:
        st.info("Grafik için bir sembol seçin.")
        return

    bars = (
        session.query(DailyBar)
        .filter(DailyBar.symbol == symbol_filter)
        .order_by(DailyBar.date.asc())
        .all()
    )
    if not bars:
        st.info("Seçilen sembol için günlük veri yok.")
        return

    df = pd.DataFrame(
        [
            {
                "Date": b.date,
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Adj Close": b.adj_close,
                "Volume": b.volume,
            }
            for b in bars
        ]
    )
    st.dataframe(df.set_index("Date"))


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Otonom Trader - Portfolio Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Otonom Trader - Portfolio Dashboard")
    st.markdown("Real-time monitoring of autonomous trading system")

    orch_cfg = load_orchestrator_config()

    with get_app_session() as sym_session:
        universe = get_universe(sym_session)

    symbol_names = ["(All)"] + [s.symbol for s in universe]
    symbol_choice = st.sidebar.selectbox("Symbol filter", symbol_names)
    symbol_filter = None if symbol_choice == "(All)" else symbol_choice

    st.sidebar.header("Orchestrator Config (read-only)")
    st.sidebar.json(orch_cfg)

    with get_app_session() as session:
        render_overview(session, orch_cfg)
        render_daemon_health(session)
        render_portfolio_performance(session)
        render_recent_activity(session, symbol_filter)
        render_trades(session, symbol_filter)
        render_news(session, symbol_filter)
        render_raw_prices(session, symbol_filter)


if __name__ == "__main__":
    main()
