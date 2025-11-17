# Strategy YAML Ekosistemi 📋

Bu dizin tüm trading stratejilerini içerir. Her strateji **standart YAML şeması** kullanır.

## 🎯 Amaç

"Strateji" denen şey **net bir konfig nesnesi** olsun; backtest, daemon, experiment, RL hepsi aynı yapıyı kullansın. "Bu parametre nereden geldi?" derdi bitsin.

## 📁 Dosyalar

- **`baseline_v1.yaml`** - İlk baseline strateji (eski format, hala çalışıyor)
- **`baseline_v2.yaml`** - Yeni standart şema kullanan baseline strateji
- **`TEMPLATE.yaml`** - Yeni strateji oluşturmak için template

## 🏗️ Standart Şema

Her strategy YAML **5 ana bölüm** içerir:

### 1. **universe** - Hangi sembolleri trade edeceğiz?

```yaml
universe:
  symbols:
    - "BTC-USD"
    - "ETH-USD"
  universe_tags: ["crypto", "fx", "equity"]  # Opsiyonel
```

### 2. **risk** - Risk yönetimi parametreleri

```yaml
risk:
  risk_pct: 1.0              # Equity'nin %1'i her trade'de (0-10 arası)
  stop_loss_pct: 5.0         # %5 stop-loss (0-50 arası)
  take_profit_pct: 10.0      # %10 take-profit
  max_drawdown_pct: 20.0     # Opsiyonel alarm seviyesi
```

**Validation:**
- `risk_pct`: 0 < value ≤ 10
- `stop_loss_pct`: 0 < value ≤ 50
- `take_profit_pct`: value > 0

### 3. **filters** - Signal filtreleme kriterleri

```yaml
filters:
  dsi_threshold: 0.3         # DSI < 0.3 = extreme fear
  regime_vol_min: 0.01       # Min volatility
  regime_vol_max: 0.05       # Max volatility
  min_volume: 1000000        # Min volume
  min_price: 1.0             # Min price
```

Tüm alanlar **opsiyoneldir** (null = filter yok).

### 4. **ensemble** - Analist weight'leri

```yaml
ensemble:
  analyst_weights:
    tech: 1.0                # Analist-1: Technical anomaly
    news: 1.2                # Analist-2: News/Macro/LLM
    risk: 0.8                # Analist-3: Regime/DSI
    rl: 0.0                  # RL agent (disabled)

  disagreement_threshold: 0.5  # Çok kavga varsa HOLD
```

**Validation:** Total weight > 0

### 5. **execution** - Execution parametreleri

```yaml
execution:
  bar_type: "D1"             # Daily bars (D1, M15, H1, etc.)
  slippage_pct: 0.1          # %0.1 slippage
  max_trades_per_day: 10     # Max 10 trade/day
```

## 💻 Kullanım

### Python'da Strateji Yükleme

```python
from otonom_trader.config import load_strategy

# Strateji yükle (otomatik validation)
config = load_strategy("strategies/baseline_v2.yaml")

# Yeni standart alanlara erişim
print(config.risk.risk_pct)           # 1.0
print(config.universe.symbols)        # ['BTC-USD', 'ETH-USD', ...]
print(config.ensemble.tech_weight)    # 1.0
print(config.execution.bar_type)      # 'D1'

# Eski helper methodlar da çalışıyor (backward compatibility)
print(config.get_symbols())           # Aynı şey
print(config.get_risk_per_trade_pct())  # Aynı şey
```

### Validation Devre Dışı Bırakma

```python
# Hatalı config test etmek için
config = load_strategy("strategies/test.yaml", validate=False)
```

### Manuel Validation

```python
from otonom_trader.config import validate_strategy_config

try:
    validate_strategy_config(config)
    print("✓ Config geçerli")
except ValueError as e:
    print(f"✗ Hata: {e}")
```

## 🆕 Yeni Strateji Oluşturma

### Adım 1: Template'i Kopyala

```bash
cp strategies/TEMPLATE.yaml strategies/my_new_strategy.yaml
```

### Adım 2: Parametreleri Düzenle

```yaml
name: "my_new_strategy"
version: "1.0.0"

universe:
  symbols: ["BTC-USD", "ETH-USD"]

risk:
  risk_pct: 2.0              # Daha agresif
  stop_loss_pct: 3.0         # Daha dar stop
  take_profit_pct: 15.0      # Daha yüksek target

ensemble:
  analyst_weights:
    tech: 1.5                # Tech'e daha çok ağırlık
    news: 0.5
    risk: 1.0
```

### Adım 3: Test Et

```bash
PYTHONPATH=otonom_trader:$PYTHONPATH python -c "
from otonom_trader.config import load_strategy
config = load_strategy('strategies/my_new_strategy.yaml')
print('✓ Config geçerli!')
print(f'  Risk: {config.risk.risk_pct}%')
print(f'  Symbols: {config.universe.symbols}')
"
```

## 🔄 Eski Format ile Uyumluluk

Eski `baseline_v1.yaml` formatı **hala çalışır**! Loader otomatik olarak:
- `data_sources.price_data.symbols` → `universe.symbols`
- `risk_management.position_sizing.risk_per_trade_pct` → `risk.risk_pct`
- `analist_1.weight` → `ensemble.tech_weight`

gibi mapping'leri yapar.

## 📊 Dataclass Yapısı

```python
@dataclass
class StrategyConfig:
    name: str
    description: str
    version: str
    universe: UniverseConfig      # ← Yeni
    risk: RiskConfig              # ← Yeni
    filters: FiltersConfig        # ← Yeni
    ensemble: EnsembleConfig      # ← Yeni
    execution: ExecutionConfig    # ← Yeni
    raw_config: Dict[str, Any]    # Backward compatibility
```

Tüm alt-config'ler de dataclass:
- `UniverseConfig`
- `RiskConfig`
- `FiltersConfig`
- `EnsembleConfig`
- `ExecutionConfig`

## ✅ Validation Kuralları

| Alan | Kural |
|------|-------|
| `risk.risk_pct` | 0 < value ≤ 10 |
| `risk.stop_loss_pct` | 0 < value ≤ 50 |
| `risk.take_profit_pct` | value > 0 |
| `universe.symbols` | Boş olmamalı |
| `ensemble` total weights | > 0 |
| `execution.max_trades_per_day` | > 0 |
| `execution.slippage_pct` | ≥ 0 |

## 🚀 Örnek Kullanımlar

### Backtest

```python
from otonom_trader.config import load_strategy
from scripts.run_research_backtests import run_backtest

config = load_strategy("strategies/baseline_v2.yaml")
results = run_backtest(config)
```

### Experiment

```python
from otonom_trader.experiments.experiment_engine import run_experiment

config = load_strategy("strategies/baseline_v2.yaml")
grid = load_param_grid("grids/baseline_grid.yaml")

# Config'i grid ile combine et
run_experiment(strategy_config=config, param_grid=grid)
```

### Daemon

```python
from otonom_trader.daemon import TradingDaemon

config = load_strategy("strategies/baseline_v2.yaml")
daemon = TradingDaemon(config)
daemon.run()
```

## 📝 Notlar

- **Backward compatibility** korunuyor - mevcut kodlar çalışmaya devam eder
- **Validation** default olarak açık, istenirse kapatılabilir
- **Nested config** detaylar için `raw_config` kullanılabilir
- **Helper methodlar** hem eski hem yeni alanlarla çalışır

## 🔗 İlgili Dosyalar

- **Loader:** `otonom_trader/otonom_trader/config/strategy_loader.py`
- **Tests:** `test_strategy_config.py`
- **Grids:** `grids/baseline_grid.yaml`
- **Experiments:** `experiments/`

---

**Hazırlayan:** Strategy YAML Ecosystem v1.0
**Son Güncelleme:** 2025-01-17
