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
from datetime import datetime, timedelta, date, timezone

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy.orm import Session
import yaml
import requests

from otonom_trader.data import get_session
from otonom_trader.data.schema import (
    Symbol,
    Anomaly,
    Decision,
    PortfolioSnapshot,
    PaperTrade,
    DaemonRun,
    DailyBar,
)

PROVIDERS_PATH = Path("config/providers.yaml")
ORCH_CONFIG_PATH = Path("config/orchestrator.yaml")

# Page configuration
st.set_page_config(
    page_title="Otonom Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .neutral {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)


def _load_yaml(path: Path) -> dict:
    """Load YAML file if it exists."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: dict) -> None:
    """Persist YAML content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


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
}


def ensure_default_config_exists() -> None:
    """Ensure the orchestrator config file exists with defaults."""
    if ORCH_CONFIG_PATH.exists():
        return
    _save_yaml(ORCH_CONFIG_PATH, {"orchestrator": DEFAULT_ORCH_CONFIG.copy()})


def load_orchestrator_config() -> dict:
    """Load orchestrator configuration as a dictionary."""
    data = _load_yaml(ORCH_CONFIG_PATH)
    config = DEFAULT_ORCH_CONFIG.copy()
    config.update(data.get("orchestrator", {}))
    return config


def save_orchestrator_config(cfg: dict) -> None:
    """Save orchestrator configuration back to YAML."""
    data = DEFAULT_ORCH_CONFIG.copy()
    data.update(cfg)
    _save_yaml(ORCH_CONFIG_PATH, {"orchestrator": data})


def load_providers_config() -> dict:
    """Load providers configuration from YAML."""
    if not PROVIDERS_PATH.exists():
        return {}
    with PROVIDERS_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_providers_config(config: dict) -> None:
    """Persist providers configuration to disk."""
    PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROVIDERS_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)


def fetch_ollama_models(base_url: str) -> list[str]:
    """Fetch available Ollama models from the local server."""
    if not base_url:
        return []
    try:
        url = base_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]
    except Exception:
        return []


def get_strategy_files(strategy_dir: Path = Path("strategies")) -> list[Path]:
    """Return available strategy YAML files for orchestrator selection."""
    if not strategy_dir.exists():
        return []
    return sorted(p for p in strategy_dir.glob("*.yaml") if p.is_file())


def get_recent_anomalies(session: Session, days: int = 7) -> pd.DataFrame:
    """Get recent anomalies."""
    cutoff = date.today() - timedelta(days=days)

    anomalies = (
        session.query(Anomaly, Symbol.symbol)
        .join(Symbol)
        .filter(Anomaly.date >= cutoff)
        .order_by(Anomaly.date.desc())
        .limit(50)
        .all()
    )

    if not anomalies:
        return pd.DataFrame()

    data = []
    for anom, symbol in anomalies:
        data.append({
            "Date": anom.date,
            "Symbol": symbol,
            "Type": anom.anomaly_type,
            "Z-Score": round(anom.zscore, 2),
            "Return": f"{anom.abs_return:+.2%}",
            "Volume Rank": f"{anom.volume_rank:.2%}",
        })

    return pd.DataFrame(data)


def get_recent_decisions(session: Session, days: int = 7) -> pd.DataFrame:
    """Get recent trading decisions."""
    cutoff = date.today() - timedelta(days=days)

    decisions = (
        session.query(Decision, Symbol.symbol)
        .join(Symbol)
        .filter(Decision.date >= cutoff)
        .order_by(Decision.date.desc())
        .limit(50)
        .all()
    )

    if not decisions:
        return pd.DataFrame()

    data = []
    for dec, symbol in decisions:
        data.append({
            "Date": dec.date,
            "Symbol": symbol,
            "Signal": dec.signal,
            "Confidence": f"{dec.confidence:.2%}",
            "P(Up)": f"{dec.p_up:.2%}" if dec.p_up else "N/A",
            "Disagreement": f"{dec.disagreement:.2%}" if dec.disagreement else "N/A",
            "Uncertainty": f"{dec.uncertainty:.2%}" if dec.uncertainty else "N/A",
            "Reason": dec.reason[:100] + "..." if len(dec.reason) > 100 else dec.reason,
        })

    return pd.DataFrame(data)


def get_portfolio_history(session: Session) -> pd.DataFrame:
    """Get portfolio snapshot history."""
    snapshots = (
        session.query(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.timestamp.asc())
        .all()
    )

    if not snapshots:
        return pd.DataFrame()

    data = []
    for snap in snapshots:
        data.append({
            "Timestamp": snap.timestamp,
            "Equity": snap.equity,
            "Cash": snap.cash,
            "Positions Value": snap.positions_value,
            "Drawdown": snap.max_drawdown if snap.max_drawdown else 0.0,
        })

    return pd.DataFrame(data)


def get_recent_trades(session: Session, days: int = 7) -> pd.DataFrame:
    """Get recent paper trades."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    trades = (
        session.query(PaperTrade, Symbol.symbol)
        .join(Symbol)
        .filter(PaperTrade.timestamp >= cutoff)
        .order_by(PaperTrade.timestamp.desc())
        .limit(50)
        .all()
    )

    if not trades:
        return pd.DataFrame()

    data = []
    for trade, symbol in trades:
        data.append({
            "Timestamp": trade.timestamp,
            "Symbol": symbol,
            "Action": trade.action,
            "Price": f"${trade.price:,.2f}",
            "Quantity": f"{trade.quantity:.4f}",
            "Value": f"${trade.value:,.2f}",
            "Portfolio Value": f"${trade.portfolio_value:,.2f}",
        })

    return pd.DataFrame(data)


def get_daemon_stats(session: Session, days: int = 7) -> dict:
    """Get daemon run statistics."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    runs = (
        session.query(DaemonRun)
        .filter(DaemonRun.timestamp >= cutoff)
        .order_by(DaemonRun.timestamp.desc())
        .all()
    )

    if not runs:
        return {}

    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.status == "SUCCESS")
    failed_runs = total_runs - successful_runs

    total_bars = sum(r.bars_ingested or 0 for r in runs)
    total_anomalies = sum(r.anomalies_detected or 0 for r in runs)
    total_decisions = sum(r.decisions_made or 0 for r in runs)
    total_trades = sum(r.trades_executed or 0 for r in runs)

    avg_duration = sum(r.duration_seconds or 0 for r in runs) / total_runs if total_runs > 0 else 0

    latest_run = runs[0] if runs else None

    return {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
        "total_bars": total_bars,
        "total_anomalies": total_anomalies,
        "total_decisions": total_decisions,
        "total_trades": total_trades,
        "avg_duration": avg_duration,
        "latest_run": latest_run,
    }


def plot_equity_curve(df: pd.DataFrame):
    """Plot portfolio equity curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#00ff00", width=2),
    ))

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown(df: pd.DataFrame):
    """Plot portfolio drawdown."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Drawdown"] * 100,  # Convert to percentage
        mode="lines",
        name="Drawdown",
        line=dict(color="#ff0000", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.2)",
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def get_symbols_df(session: Session) -> pd.DataFrame:
    """Return registered symbols as a dataframe."""
    symbols = session.query(Symbol).order_by(Symbol.symbol.asc()).all()
    if not symbols:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "ID": sym.id,
                "Symbol": sym.symbol,
                "Name": sym.name,
                "Asset Class": sym.asset_class,
            }
            for sym in symbols
        ]
    )


def get_symbol_bars_df(session: Session, symbol: str, days: int = 30) -> pd.DataFrame:
    """Return historical OHLCV data for a symbol using DailyBar records."""
    if not symbol:
        return pd.DataFrame()

    cutoff = date.today() - timedelta(days=days)
    sym_obj = session.query(Symbol).filter_by(symbol=symbol).first()
    if not sym_obj:
        return pd.DataFrame()

    bars = (
        session.query(DailyBar)
        .filter(DailyBar.symbol_id == sym_obj.id)
        .filter(DailyBar.date >= cutoff)
        .order_by(DailyBar.date.asc())
        .all()
    )
    if not bars:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Date": bar.date,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Adj Close": bar.adj_close,
                "Volume": bar.volume,
            }
            for bar in bars
        ]
    )


def main():
    """Main dashboard."""
    st.title("📈 Otonom Trader - Portfolio Dashboard")
    st.markdown("Real-time monitoring of autonomous trading system")

    ensure_default_config_exists()
    orchestrator_cfg = load_orchestrator_config()
    strategy_dir = Path("strategies")
    strategy_files = get_strategy_files(strategy_dir)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        lookback_days = st.slider("Lookback Days", 1, 30, 7)
        auto_refresh = st.checkbox("Auto Refresh (60s)", value=False)

        st.markdown("---")
        st.subheader("🔐 Providers & API Keys")
        providers_cfg = load_providers_config()
        price_providers = list(providers_cfg.get("price_providers", []))
        binance_cfg = next((p for p in price_providers if p.get("type") == "binance"), None)
        if binance_cfg is None:
            binance_cfg = {
                "type": "binance",
                "enabled": True,
                "api_key": "",
                "api_secret": "",
                "base_url": "https://api.binance.com",
                "extra": {"default_interval": "1d"},
            }
            price_providers.append(binance_cfg)
        providers_cfg["price_providers"] = price_providers

        binance_enabled = st.checkbox(
            "Binance Enabled",
            value=binance_cfg.get("enabled", True),
        )
        binance_api_key = st.text_input(
            "Binance API Key",
            value=binance_cfg.get("api_key", ""),
            type="password",
        )
        binance_api_secret = st.text_input(
            "Binance API Secret",
            value=binance_cfg.get("api_secret", ""),
            type="password",
        )
        binance_base_url = st.text_input(
            "Binance Base URL",
            value=binance_cfg.get("base_url", "https://api.binance.com"),
        )
        interval_options = ["1m", "5m", "15m", "1h", "4h", "1d"]
        current_interval = str(binance_cfg.get("extra", {}).get("default_interval", "1d"))
        if current_interval not in interval_options:
            current_interval = "1d"
        binance_default_interval = st.selectbox(
            "Binance Default Interval",
            interval_options,
            index=interval_options.index(current_interval),
            help="15 dakikalık veri istersen burada '15m' seç.",
        )

        llm_section = list(providers_cfg.get("llm_providers", []))
        ollama_cfg = next((p for p in llm_section if p.get("type") == "ollama"), None)
        ollama_exists = ollama_cfg is not None
        if not ollama_exists:
            ollama_cfg = {
                "type": "ollama",
                "enabled": False,
                "base_url": "http://localhost:11434",
                "default_model": "",
            }

        ollama_enabled = st.checkbox(
            "Ollama Enabled",
            value=ollama_cfg.get("enabled", True),
        )
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=ollama_cfg.get("base_url", "http://localhost:11434"),
        )

        model_options = fetch_ollama_models(ollama_base_url) if ollama_enabled else []
        default_model_value = ollama_cfg.get("default_model", "") or (model_options[0] if model_options else "unknown")
        if model_options and default_model_value in model_options:
            default_index = model_options.index(default_model_value)
        else:
            model_options = model_options or [default_model_value]
            default_index = 0

        selected_model = st.selectbox(
            "Ollama Default Model",
            options=model_options,
            index=default_index,
        )

        if st.button("Save Provider Settings"):
            binance_cfg["enabled"] = binance_enabled
            binance_cfg["api_key"] = binance_api_key
            binance_cfg["api_secret"] = binance_api_secret
            binance_cfg["base_url"] = binance_base_url
            extra = binance_cfg.get("extra", {})
            extra["default_interval"] = binance_default_interval
            binance_cfg["extra"] = extra

            if not ollama_exists:
                llm_section.append(ollama_cfg)
                ollama_exists = True

            if ollama_exists:
                ollama_cfg["enabled"] = ollama_enabled
                ollama_cfg["base_url"] = ollama_base_url
                ollama_cfg["default_model"] = selected_model

            providers_cfg["llm_providers"] = llm_section
            save_providers_config(providers_cfg)
            st.success("Provider / API ayarları kaydedildi.")

        st.markdown("---")
        st.subheader("🛠️ Orchestrator Control")
        mode_options = ["paper", "shadow", "live"]
        current_mode = orchestrator_cfg.get("mode", "paper")
        if current_mode not in mode_options:
            current_mode = "paper"
        mode_index = mode_options.index(current_mode)
        selected_strategy_name = None

        with st.form("orchestrator_settings"):
            selected_mode = st.selectbox(
                "Execution Mode",
                mode_options,
                index=mode_index,
                key="orchestrator_mode",
            )

            if strategy_files:
                strategy_names = [p.name for p in strategy_files]
                cfg_strategy_name = Path(orchestrator_cfg.get("strategy_path", "")).name if orchestrator_cfg.get("strategy_path") else ""
                strategy_index = strategy_names.index(cfg_strategy_name) if cfg_strategy_name in strategy_names else 0
                selected_strategy_name = st.selectbox(
                    "Strategy",
                    strategy_names,
                    index=strategy_index,
                    key="orchestrator_strategy",
                )
            else:
                st.info("No strategy YAML files found in the strategies/ directory.")

            submitted = st.form_submit_button("Save Orchestrator Settings")

        if submitted:
            updated = False
            if selected_mode != orchestrator_cfg.get("mode"):
                orchestrator_cfg["mode"] = selected_mode
                updated = True

            if selected_strategy_name:
                new_strategy_path = str(strategy_dir / selected_strategy_name)
                if orchestrator_cfg.get("strategy_path") != new_strategy_path:
                    orchestrator_cfg["strategy_path"] = new_strategy_path
                    updated = True

            if updated:
                save_orchestrator_config(orchestrator_cfg)
                st.success("Orchestrator configuration updated.")
            else:
                st.info("No changes to orchestrator configuration.")

        st.caption(f"Config file: {ORCH_CONFIG_PATH}")

        st.markdown("---")
        st.subheader("📇 Symbol Management")
        with st.form("add_symbol_form"):
            new_sym = st.text_input("Symbol (örn: BTCUSDT / THYAO.IS)")
            new_name = st.text_input("Name", "")
            asset_options = ["CRYPTO", "EQUITY", "FX", "COMMODITY", "INDEX"]
            new_class = st.selectbox("Asset Class", asset_options, index=0)
            submitted_symbol = st.form_submit_button("Add Symbol")

        if submitted_symbol and new_sym:
            with get_session() as sym_session:
                existing = sym_session.query(Symbol).filter_by(symbol=new_sym).first()
                if existing:
                    st.warning(f"{new_sym} zaten kayıtlı.")
                else:
                    record = Symbol(symbol=new_sym, name=new_name or new_sym, asset_class=new_class)
                    sym_session.add(record)
                    sym_session.commit()
                    st.success(f"{new_sym} eklendi.")

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This dashboard provides read-only monitoring of the "
            "Otonom Trader system. No trading actions can be performed from here."
        )

    # Auto-refresh
    if auto_refresh:
        st.empty()
        import time
        time.sleep(60)
        st.rerun()

    # Get database session
    with get_session() as session:
        # ============================================================
        # SECTION 1: Overview Metrics
        # ============================================================
        st.header("📊 Overview")

        col1, col2, col3, col4 = st.columns(4)

        # Get latest portfolio snapshot
        latest_snapshot = (
            session.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .first()
        )

        if latest_snapshot:
            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${latest_snapshot.equity:,.2f}",
                    delta=None,
                )
            with col2:
                st.metric(
                    "Cash",
                    f"${latest_snapshot.cash:,.2f}",
                )
            with col3:
                st.metric(
                    "Positions Value",
                    f"${latest_snapshot.positions_value:,.2f}",
                )
            with col4:
                drawdown_pct = latest_snapshot.max_drawdown * 100 if latest_snapshot.max_drawdown else 0
                st.metric(
                    "Max Drawdown",
                    f"{drawdown_pct:.2f}%",
                    delta=None,
                    delta_color="inverse",
                )
        else:
            st.warning("No portfolio data available yet. Run the daemon first.")

        # ============================================================
        # SECTION 2: Daemon Health
        # ============================================================
        st.header("🏥 Daemon Health")

        daemon_stats = get_daemon_stats(session, lookback_days)

        if daemon_stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Runs", daemon_stats["total_runs"])
            with col2:
                success_rate = daemon_stats["success_rate"] * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Total Trades", daemon_stats["total_trades"])
            with col4:
                st.metric("Avg Duration", f"{daemon_stats['avg_duration']:.1f}s")

            # Latest run status
            if daemon_stats["latest_run"]:
                latest = daemon_stats["latest_run"]
                status_color = "green" if latest.status == "SUCCESS" else "red"
                st.markdown(
                    f"**Latest Run:** {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| Status: :{status_color}[{latest.status}] "
                    f"| Bars: {latest.bars_ingested or 0} "
                    f"| Anomalies: {latest.anomalies_detected or 0} "
                    f"| Decisions: {latest.decisions_made or 0} "
                    f"| Trades: {latest.trades_executed or 0}"
                )
        else:
            st.info("No daemon runs in the selected period.")

        # ============================================================
        # SECTION 3: Portfolio Performance
        # ============================================================
        st.header("💰 Portfolio Performance")

        portfolio_df = get_portfolio_history(session)

        if not portfolio_df.empty:
            # Equity curve
            plot_equity_curve(portfolio_df)

            # Drawdown chart
            plot_drawdown(portfolio_df)

            # Performance metrics
            if len(portfolio_df) > 1:
                initial_equity = portfolio_df.iloc[0]["Equity"]
                final_equity = portfolio_df.iloc[-1]["Equity"]
                total_return = (final_equity - initial_equity) / initial_equity * 100
                max_dd = portfolio_df["Drawdown"].max() * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Return", f"{total_return:+.2f}%")
                with col2:
                    st.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
        else:
            st.info("No portfolio history available yet.")

        # ============================================================
        # SECTION 4: Recent Activity
        # ============================================================
        st.header("🔄 Recent Activity")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Anomalies", "Decisions", "Trades", "Details", "Symbols"])

        with tab1:
            st.subheader("Recent Anomalies")
            anomalies_df = get_recent_anomalies(session, lookback_days)
            if not anomalies_df.empty:
                st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies detected in the selected period.")

        with tab2:
            st.subheader("Recent Decisions")
            decisions_df = get_recent_decisions(session, lookback_days)
            if not decisions_df.empty:
                st.dataframe(decisions_df, use_container_width=True, hide_index=True)

                # Show full reasoning for selected decision
                if not decisions_df.empty:
                    with st.expander("View Full Decision Reasoning"):
                        selected_idx = st.selectbox(
                            "Select Decision",
                            range(len(decisions_df)),
                            format_func=lambda i: f"{decisions_df.iloc[i]['Date']} - {decisions_df.iloc[i]['Symbol']} - {decisions_df.iloc[i]['Signal']}",
                        )
                        full_reason = (
                            session.query(Decision, Symbol.symbol)
                            .join(Symbol)
                            .filter(Decision.date >= date.today() - timedelta(days=lookback_days))
                            .order_by(Decision.date.desc())
                            .limit(50)
                            .all()
                        )[selected_idx][0].reason

                        st.code(full_reason, language="text")
            else:
                st.info("No decisions made in the selected period.")

        with tab3:
            st.subheader("Recent Trades")
            trades_df = get_recent_trades(session, lookback_days)
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trades executed in the selected period.")

        with tab4:
            st.subheader("System Details")

            # Database stats
            st.markdown("**Database Statistics**")
            total_symbols = session.query(Symbol).count()
            total_anomalies = session.query(Anomaly).count()
            total_decisions = session.query(Decision).count()
            total_trades = session.query(PaperTrade).count()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Symbols", total_symbols)
            col2.metric("Anomalies", total_anomalies)
            col3.metric("Decisions", total_decisions)
            col4.metric("Trades", total_trades)

            st.markdown("---")
            st.markdown("### Orchestrator Configuration")

            orch_cfg = _load_yaml(ORCH_CONFIG_PATH)
            orch_section = orch_cfg.get("orchestrator", {})

            enabled = st.checkbox(
                "Enable Orchestrator Loop",
                value=orch_section.get("enabled", True),
                help="If disabled, the orchestrator will not run new daemon cycles.",
            )

            mode_options = ["paper", "shadow", "live"]
            mode_value = str(orch_section.get("mode", "paper")).lower()
            if mode_value not in mode_options:
                mode_value = "paper"
            mode = st.selectbox(
                "Execution Mode",
                options=mode_options,
                index=mode_options.index(mode_value),
                help="paper: only paper trading, shadow: paper + real broker, live: only real broker.",
            )

            interval_seconds = st.number_input(
                "Loop Interval (seconds)",
                min_value=60,
                max_value=3600,
                value=int(orch_section.get("interval_seconds", 900)),
                help="Default is 900 seconds (15 minutes).",
            )

            db_path = st.text_input(
                "DB Path",
                value=str(orch_section.get("db_path", "data/trader.db")),
            )
            strategy_path = st.text_input(
                "Strategy YAML Path",
                value=str(orch_section.get("strategy_path", "strategies/baseline_v1.0.yaml")),
            )
            broker_cfg_path = st.text_input(
                "Broker Config Path",
                value=str(orch_section.get("broker_config_path", "config/broker.yaml")),
            )

            use_ensemble = st.checkbox(
                "Use Ensemble Mode",
                value=orch_section.get("use_ensemble", True),
            )
            paper_trade_enabled = st.checkbox(
                "Paper Trading Enabled (base)",
                value=orch_section.get("paper_trade_enabled", True),
                help="Note: actual behavior is overridden by mode.",
            )
            paper_trade_risk_pct = st.number_input(
                "Risk per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(orch_section.get("paper_trade_risk_pct", 1.0)),
                step=0.1,
            )
            initial_cash = st.number_input(
                "Initial Cash",
                min_value=1000.0,
                max_value=10_000_000.0,
                value=float(orch_section.get("initial_cash", 100_000.0)),
                step=1000.0,
            )
            ingest_days_back = st.number_input(
                "Ingest Days Back",
                min_value=1,
                max_value=365,
                value=int(orch_section.get("ingest_days_back", 7)),
            )
            anomaly_lookback_days = st.number_input(
                "Anomaly Lookback Days",
                min_value=1,
                max_value=365,
                value=int(orch_section.get("anomaly_lookback_days", 30)),
            )

            if st.button("💾 Save Configuration"):
                new_orch = {
                    "enabled": enabled,
                    "mode": mode,
                    "interval_seconds": int(interval_seconds),
                    "db_path": db_path,
                    "strategy_path": strategy_path,
                    "broker_config_path": broker_cfg_path,
                    "use_ensemble": bool(use_ensemble),
                    "paper_trade_enabled": bool(paper_trade_enabled),
                    "paper_trade_risk_pct": float(paper_trade_risk_pct),
                    "initial_cash": float(initial_cash),
                    "ingest_days_back": int(ingest_days_back),
                    "anomaly_lookback_days": int(anomaly_lookback_days),
                }
                _save_yaml(ORCH_CONFIG_PATH, {"orchestrator": new_orch})
                st.success("Orchestrator configuration saved.")

        with tab5:
            st.subheader("Symbols & Price Data")
            symbols_df = get_symbols_df(session)
            if symbols_df.empty:
                st.info("Henüz sembol yok. Soldaki 'Symbol Management' bölümünden ekleyebilirsin.")
            else:
                st.dataframe(symbols_df, use_container_width=True, hide_index=True)
                symbol_choices = symbols_df["Symbol"].tolist()
                selected_symbol = st.selectbox("Symbol", symbol_choices)
                history_days = st.slider("History Days", min_value=7, max_value=365, value=90, key="symbol_history_days")
                bars_df = get_symbol_bars_df(session, selected_symbol, days=history_days)
                if bars_df.empty:
                    st.warning("Bu sembol için veri yok (henüz ingest edilmemiş olabilir).")
                else:
                    st.line_chart(bars_df.set_index("Date")["Close"], use_container_width=True)
                    st.bar_chart(bars_df.set_index("Date")["Volume"], use_container_width=True)
                    st.dataframe(bars_df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
