# Daemon Paper Trader - Kullanım Kılavuzu

## Genel Bakış

Daemon sistemi, Otonom Trader'ı tam otonom bir paper trading bot'una dönüştürür. Her çalıştırıldığında:

1. **Incremental Ingest**: Son verileriverileri çeker (sadece eksik günler)
2. **Anomaly Detection**: Yeni verilerde anomali taraması yapar
3. **Patron Decision**: Anomalilere dayalı trading sinyalleri üretir
4. **Paper Trading**: Kararları portföy simülasyonunda execute eder

## Yeni Özellikler (P3 Hazırlığı)

### 1. Execution Log Tabloları

#### `paper_trades` Tablosu
Her paper trade kaydını tutar:
- `timestamp`: İşlem zamanı
- `symbol_id`, `decision_id`: İlişkili sembol ve karar
- `action`: BUY, SELL, HOLD
- `price`, `quantity`, `value`: İşlem detayları
- `portfolio_value`, `cash`: İşlem sonrası portföy durumu

#### `daemon_runs` Tablosu
Her daemon cycle'ının logunu tutar:
- `timestamp`: Çalışma zamanı
- `bars_ingested`, `anomalies_detected`, `decisions_made`, `trades_executed`: Pipeline istatistikleri
- `portfolio_value`, `cash`: Portföy snapshot'ı
- `status`: SUCCESS, PARTIAL, FAILED
- `duration_seconds`: Süre

### 2. Incremental Ingest

`ingest_incremental()` fonksiyonu:
- Her sembol için son veri tarihini kontrol eder
- Sadece eksik günleri çeker (API usage optimizasyonu)
- Yeni veri yoksa atlar

```python
from otonom_trader.data.ingest import ingest_incremental

# Son 7 günü çek (eğer veri yoksa)
results = ingest_incremental(session, days_back=7)
```

### 3. Paper Trader Engine

`PaperTrader` sınıfı tam bir simülasyon portföyü yönetir:

```python
from otonom_trader.daemon import PaperTrader

# Initialize
trader = PaperTrader(session, initial_cash=100000.0)

# Execute decision
from otonom_trader.domain import Decision, SignalType
decision = Decision(
    asset_symbol="BTC-USD",
    date=date.today(),
    signal=SignalType.BUY,
    confidence=0.8,
    reason="SPIKE_DOWN + Uptrend"
)

trade = trader.execute_decision(decision, risk_pct=1.0)

# Portfolio summary
summary = trader.get_portfolio_summary()
print(f"Total value: ${summary['total_value']:,.2f}")
print(f"Positions: {summary['num_positions']}")
```

**Position Sizing**: Risk percentage bazlı
- `risk_pct=1.0`: Her trade portföy değerinin %1'i kadar risk

**Portfolio State**: Gerçek zamanlı tracking
- Cash balance
- Positions (quantity, avg_price, current_price, P&L)
- Total portfolio value

### 4. Daemon Pipeline

`run_daemon_cycle()` tam pipeline'ı çalıştırır:

```python
from otonom_trader.daemon import run_daemon_cycle, DaemonConfig

config = DaemonConfig(
    ingest_days_back=7,           # Incremental ingest gün sayısı
    anomaly_lookback_days=30,     # Anomaly detection penceresi
    use_ensemble=False,           # Patron ensemble modu
    paper_trade_enabled=True,     # Paper trading aç/kapa
    paper_trade_risk_pct=1.0,     # Trade risk yüzdesi
    initial_cash=100000.0,        # Başlangıç cash'i
)

# Run single cycle
run = run_daemon_cycle(session, config)

print(f"Status: {run.status}")
print(f"Bars ingested: {run.bars_ingested}")
print(f"Anomalies: {run.anomalies_detected}")
print(f"Decisions: {run.decisions_made}")
print(f"Trades: {run.trades_executed}")
print(f"Portfolio: ${run.portfolio_value:,.2f}")
```

## CLI Komutları

### Yeni Komutlar

#### 1. `ingest-incremental`
Incremental veri ingest:

```bash
otonom-trader ingest-incremental
otonom-trader ingest-incremental --days-back 30
```

#### 2. `run-daemon`
Tek daemon cycle çalıştır (tam pipeline):

```bash
# Basic run
otonom-trader run-daemon

# Custom config
otonom-trader run-daemon --initial-cash 50000 --risk-pct 2.0

# Ensemble mode
otonom-trader run-daemon --ensemble

# Analysis only (no trading)
otonom-trader run-daemon --no-trade
```

**Parameters**:
- `--initial-cash`: Başlangıç cash (default: $100,000)
- `--risk-pct`: Trade başına risk % (default: 1.0%)
- `--ensemble`: Patron ensemble mode (default: False)
- `--no-trade`: Paper trading'i kapat (sadece analiz)

#### 3. `show-paper-trades`
Paper trade geçmişini göster:

```bash
otonom-trader show-paper-trades
otonom-trader show-paper-trades --limit 50
otonom-trader show-paper-trades --symbol BTC-USD
```

**Output örneği**:
```
Timestamp           | Symbol     | Action | Quantity  | Price     | Value      | Portfolio
------------------------------------------------------------------------------------------------
2025-01-15 14:32:10 |    BTC-USD |    BUY |   0.2500 |  $42000.00 |  $10500.00 |  $100,000.00
2025-01-15 14:32:15 |     SUGAR  |    BUY | 100.0000 |     $18.50 |   $1850.00 |   $98,150.00
```

#### 4. `daemon-status`
Daemon çalışma geçmişini göster:

```bash
otonom-trader daemon-status
otonom-trader daemon-status --limit 20
```

**Output örneği**:
```
Timestamp           | Status  | Bars | Anomalies | Decisions | Trades | Duration | Portfolio
-------------------------------------------------------------------------------------------------------
2025-01-15 14:30:00 | SUCCESS |   12 |         3 |         2 |      2 |     5.2s |  $100,000.00
2025-01-14 14:30:00 | SUCCESS |    8 |         1 |         1 |      1 |     4.8s |   $99,500.00
```

#### 5. `status` (Güncellendi)
Database durumu - artık paper trading metrics de gösterir:

```bash
otonom-trader status
```

**Output**:
```
Database Status
========================================
Symbols:          8
Bars:       25000
Anomalies:    150
Decisions:     80
Regimes:     5000  [P1]
DSI:         4800  [P1]
Hypotheses:     2  [P1]
Backtests:    120  [P1]
Paper Trades:  45  [P3]  ← YENİ
Daemon Runs:   10  [P3]  ← YENİ
```

## Cron ile Otomasyonu

Daemon'u günlük otomatik çalıştırmak için crontab ekleyin:

```bash
# Her gün 09:00'da çalıştır
0 9 * * * cd /path/to/otonom_trader && otonom-trader run-daemon >> /var/log/otonom-trader.log 2>&1

# Her 6 saatte bir çalıştır
0 */6 * * * cd /path/to/otonom_trader && otonom-trader run-daemon --risk-pct 0.5
```

**Production Setup Önerisi**:
```bash
# 1. Virtual environment aktif et
# 2. Tam path kullan
# 3. Log dosyası yaz
# 4. Error handling ekle

0 9 * * * /path/to/venv/bin/otonom-trader run-daemon \
    --initial-cash 100000 \
    --risk-pct 1.0 \
    --ensemble \
    >> /var/log/otonom-trader.log 2>&1 || echo "Daemon failed" | mail -s "Otonom Trader Error" you@example.com
```

## Örnek Workflow

### İlk Kurulum

```bash
# 1. Database initialize
otonom-trader init

# 2. İlk veri yükle (historical)
otonom-trader ingest-data --start 2013-01-01

# 3. Anomaly detection
otonom-trader detect-anomalies

# 4. İlk decisions
otonom-trader run-patron

# 5. İlk daemon run (paper trading)
otonom-trader run-daemon --initial-cash 100000
```

### Günlük Kullanım

Cron ile otomatik çalışacak, ama manuel de çalıştırabilirsiniz:

```bash
# Single cycle
otonom-trader run-daemon

# Sonuçları kontrol et
otonom-trader daemon-status
otonom-trader show-paper-trades --limit 10

# Portfolio performans
otonom-trader status
```

### Monitoring

```bash
# Son daemon runs
otonom-trader daemon-status --limit 5

# Son trades
otonom-trader show-paper-trades --limit 20

# Specific symbol
otonom-trader show-paper-trades --symbol BTC-USD

# Recent decisions
otonom-trader show-decisions --limit 10
```

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DAEMON CYCLE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INCREMENTAL INGEST                                      │
│     ├─ Check latest bar date per symbol                    │
│     ├─ Fetch only missing days                             │
│     └─ Upsert to daily_bars table                          │
│                                                             │
│  2. ANOMALY DETECTION                                       │
│     ├─ Scan recent data (last N days)                      │
│     ├─ Z-score + volume analysis                           │
│     └─ Store in anomalies table                            │
│                                                             │
│  3. PATRON DECISION ENGINE                                  │
│     ├─ Load recent anomalies                               │
│     ├─ Trend analysis (20-day SMA)                         │
│     ├─ Generate BUY/SELL/HOLD signals                      │
│     └─ Store in decisions table                            │
│                                                             │
│  4. PAPER TRADING EXECUTION                                 │
│     ├─ Load new decisions                                  │
│     ├─ Calculate position sizes (risk %)                   │
│     ├─ Execute simulated trades                            │
│     ├─ Update portfolio state                              │
│     └─ Log to paper_trades table                           │
│                                                             │
│  5. LOGGING                                                 │
│     └─ Store cycle metrics in daemon_runs table            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema (Yeni Tablolar)

**paper_trades**:
```sql
CREATE TABLE paper_trades (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    symbol_id INTEGER REFERENCES symbols(id),
    decision_id INTEGER REFERENCES decisions(id),
    action VARCHAR(10),  -- BUY, SELL, HOLD
    price FLOAT,
    quantity FLOAT,
    value FLOAT,
    portfolio_value FLOAT,
    cash FLOAT,
    notes TEXT,
    created_at DATETIME
);
```

**daemon_runs**:
```sql
CREATE TABLE daemon_runs (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    bars_ingested INTEGER,
    anomalies_detected INTEGER,
    decisions_made INTEGER,
    trades_executed INTEGER,
    portfolio_value FLOAT,
    cash FLOAT,
    status VARCHAR(20),  -- SUCCESS, PARTIAL, FAILED
    error_message TEXT,
    duration_seconds FLOAT,
    created_at DATETIME
);
```

## Performans ve Optimizasyon

### Incremental Ingest Avantajları

**Önceki Sistem** (full historical):
- Her run: 8 sembol × 3000+ gün = 24,000+ API call
- Süre: ~5-10 dakika
- Rate limiting riski

**Yeni Sistem** (incremental):
- Her run: 8 sembol × 1-7 gün = 8-56 API call
- Süre: ~10-30 saniye
- API efficient

### Memory Management

PaperTrader sadece aktif state tutar:
- Cash balance (float)
- Positions dict (symbol → Position)
- Trade history DB'de

### Scalability

- **Sembol sayısı**: Sınır yok, P0 assets ile test edildi (8 sembol)
- **Trade frequency**: Anomaly bazlı (günlük 0-5 trade tipik)
- **Historical data**: SQLite efficient (100K+ bars tested)

## Troubleshooting

### Daemon Run Başarısız

```bash
# Daemon status kontrol et
otonom-trader daemon-status --limit 1

# Error message göster
sqlite3 data/otonom_trader.db \
  "SELECT error_message FROM daemon_runs WHERE status='FAILED' ORDER BY timestamp DESC LIMIT 1;"
```

Common issues:
- **Network error**: yfinance API timeout → Retry
- **No data**: Symbol data yok → Önceden ingest et
- **Insufficient cash**: Trade için yeterli cash yok → initial_cash artır

### Paper Trading Issues

**"Insufficient cash" hatası**:
```bash
# Portfolio state kontrol et
otonom-trader show-paper-trades --limit 1
```

Solution: `--initial-cash` parametresini artır

**"No position to sell" hatası**:
- Normal: SELL signal var ama position yok
- Action: HOLD olarak loglanır

### Database Migration

Yeni tablolar eklendiğinde:

```bash
# Mevcut DB'yi backup al
cp data/otonom_trader.db data/otonom_trader.db.backup

# Re-initialize (dikkat: veri kaybı!)
otonom-trader init --force

# Ya da manuel migration:
# schema.py'deki yeni tabloları ekle
```

## İleri Konular

### Custom Risk Management

```python
from otonom_trader.daemon import PaperTrader

class CustomPaperTrader(PaperTrader):
    def calculate_position_size(self, symbol, price, risk_pct):
        # Custom logic
        # Örnek: BTC için daha fazla risk
        if "BTC" in symbol:
            risk_pct *= 1.5

        return super().calculate_position_size(symbol, price, risk_pct)
```

### Portfolio Analytics

```python
from otonom_trader.data.schema import PaperTrade

# Calculate returns
trades = session.query(PaperTrade).order_by(PaperTrade.timestamp).all()

initial_value = trades[0].portfolio_value if trades else 100000
final_value = trades[-1].portfolio_value if trades else 100000

total_return = (final_value - initial_value) / initial_value * 100
print(f"Total return: {total_return:+.2f}%")
```

### Multi-Strategy Support

Future enhancement: Farklı hipotezler için ayrı paper traders:

```python
# Strategy 1: Conservative (low risk)
trader_conservative = PaperTrader(session, initial_cash=50000)

# Strategy 2: Aggressive (high risk)
trader_aggressive = PaperTrader(session, initial_cash=50000)

# Execute different decisions with different traders
```

## Sonraki Adımlar (P3 → Production)

1. **Real Broker Integration**: Paper trading → Real trading
2. **WebSocket Data**: Real-time price feeds
3. **Order Management**: Limit orders, stop losses
4. **Risk Management**: Position limits, max drawdown controls
5. **Monitoring Dashboard**: Web UI for portfolio tracking
6. **Alerts**: Telegram/email notifications
7. **Multi-timeframe**: Intraday trading support

---

**Hazırlayan**: Claude Code
**Tarih**: 2025-11-17
**Version**: P3 Preparation (Daemon Paper Trader)
