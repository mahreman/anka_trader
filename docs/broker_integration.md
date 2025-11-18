# Broker Integration Guide

Complete guide to broker abstraction, shadow mode, kill-switch guardrails, and alerts.

## Overview

The broker abstraction layer provides:
1. **Unified Interface**: Single API for all brokers (Binance, Bybit, etc.)
2. **Shadow Mode**: Run paper and real broker in parallel for testing
3. **Kill-Switch**: Automatic trading halt when risk limits exceeded
4. **Alerts**: Email and Telegram notifications for critical events

## Architecture

```
TradingDaemon
    ├── PaperTrader (always runs for logging)
    ├── Broker (optional, based on mode)
    │   └── GuardedBroker (wraps broker with kill-switch)
    │       └── BinanceBroker (actual implementation)
    └── AlertEngine (monitors and notifies)
```

## 1. Broker Abstraction Layer

### Interface

All brokers implement the `Broker` abstract class:

```python
from otonom_trader.broker import (
    Broker,
    OrderRequest,
    OrderSide,
    OrderType,
)

class Broker(ABC):
    def place_order(self, req: OrderRequest) -> OrderFill: ...
    def cancel_order(self, symbol: str, order_id: str) -> None: ...
    def get_open_positions(self) -> List[Position]: ...
    def get_balances(self) -> List[Balance]: ...
    def ping(self) -> bool: ...
```

### Binance Implementation

**Setup**:
```bash
# Copy config template
cp config/broker.yaml.example config/broker.yaml

# Edit with your credentials
# For testnet: https://testnet.binance.vision/
nano config/broker.yaml
```

**Configuration**:
```yaml
kind: binance
api_key: "YOUR_API_KEY"
api_secret: "YOUR_API_SECRET"
base_url: "https://testnet.binance.vision"
testnet: true
```

**Usage**:
```python
from otonom_trader.broker import build_broker

# Build from config
broker = build_broker("config/broker.yaml")

# Check connectivity
if broker.ping():
    print("Connected to Binance!")

# Place market order
from otonom_trader.broker import OrderRequest, OrderSide, OrderType

req = OrderRequest(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    quantity=0.001,
    order_type=OrderType.MARKET,
)

fill = broker.place_order(req)
print(f"Order filled at {fill.price}")
```

## 2. Shadow Mode

Shadow mode runs paper trader and real broker in parallel for validation.

### Execution Modes

**PAPER** (Default):
- Only paper trader runs
- No real orders
- Safe for testing

**SHADOW**:
- Paper trader + real broker
- Both execute same orders
- Compare results for validation

**LIVE**:
- Only real broker
- Paper trader optional (for logging)
- Production mode

### Usage

**Python API**:
```python
from otonom_trader.daemon import TradingDaemon, ExecutionMode

daemon = TradingDaemon(
    db_path="trader.db",
    strategy_path="strategies/baseline_v1.0.yaml",
    broker_config_path="config/broker.yaml",
    mode=ExecutionMode.SHADOW,  # PAPER, SHADOW, or LIVE
)

daemon.run_once()
```

**CLI**:
```bash
# Paper mode (default)
python -m otonom_trader.cli daemon-once \
  --strategy strategies/baseline_v1.0.yaml \
  --mode paper

# Shadow mode (parallel validation)
python -m otonom_trader.cli daemon-once \
  --strategy strategies/baseline_v1.0.yaml \
  --broker-config config/broker.yaml \
  --mode shadow

# Live mode (real trading)
python -m otonom_trader.cli daemon-once \
  --strategy strategies/baseline_v1.0.yaml \
  --broker-config config/broker.yaml \
  --mode live
```

## 3. Kill-Switch and Guardrails

Automatic trading halt when risk limits are exceeded.

### Guardrail Types

**Daily Loss Limit**:
- Triggers if daily loss exceeds X% of starting equity
- Default: 5%

**Maximum Drawdown**:
- Triggers if drawdown from peak exceeds X%
- Default: 40%

**Consecutive Losses**:
- Triggers after N consecutive losing trades
- Default: 5

### Configuration

```python
from otonom_trader.broker import GuardedBroker, GuardrailConfig

# Create guardrail config
guard_cfg = GuardrailConfig(
    max_daily_loss_pct=5.0,      # Max 5% daily loss
    max_drawdown_pct=40.0,        # Max 40% drawdown
    max_consecutive_losses=5,     # Max 5 losses in a row
    enabled=True,                 # Enable guardrails
)

# Wrap broker with guardrails
raw_broker = build_broker("config/broker.yaml")
guarded_broker = GuardedBroker(raw_broker, guard_cfg)

# Use guarded broker
try:
    fill = guarded_broker.place_order(req)
except BrokerError as e:
    if "KILL-SWITCH" in str(e):
        print("Trading halted by kill-switch!")
```

### Kill-Switch Behavior

**When Triggered**:
1. All new orders are rejected with `BrokerError`
2. Alerts are sent via email/Telegram
3. Order cancellations still work (can close positions)
4. System logs the trigger reason

**Reset**:
- Kill-switch automatically resets when all guardrails clear
- For example, if triggered by daily loss, it resets next day

### Checking Status

```python
# Get kill-switch status
active, reason = guarded_broker.get_kill_switch_status()

if active:
    print(f"Kill-switch active: {reason}")
else:
    print("All guardrails clear")
```

## 4. Alert Engine

Send notifications via email and Telegram for critical events.

### Setup

**Configure Alerts**:
```bash
cp config/alerts.yaml.example config/alerts.yaml
nano config/alerts.yaml
```

**Email (Gmail example)**:
```yaml
email:
  enabled: true
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  username: "your@gmail.com"
  password: "your_app_password"  # Use App Password
  from_addr: "your@gmail.com"
  to_addrs:
    - "your@email.com"
```

**Telegram**:
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"  # From @BotFather
  chat_id: "YOUR_CHAT_ID"      # Your Telegram user ID
```

### Usage

```python
from otonom_trader.alerts import AlertEngine

alerts = AlertEngine("config/alerts.yaml")

# Broker error
alerts.notify_broker_error("Connection timeout")

# Kill-switch trigger
alerts.notify_kill_switch("Max daily loss 5% exceeded")

# Portfolio alert
alerts.notify_portfolio_alert("Drawdown > 50%")

# System health check
alerts.check_and_notify(datetime.utcnow())
```

### Alert Types

**Broker Errors**:
- API failures
- Connection timeouts
- Invalid orders

**Kill-Switch Triggers**:
- Daily loss limit exceeded
- Max drawdown exceeded
- Consecutive losses exceeded

**Daemon Health**:
- Daemon hasn't run in 30+ minutes
- System stalls or crashes

**Portfolio Alerts**:
- Extreme drawdown (>50%)
- Other performance issues

## 5. Complete Example

### Setup

```bash
# 1. Configure broker
cp config/broker.yaml.example config/broker.yaml
# Edit with Binance testnet credentials

# 2. Configure alerts
cp config/alerts.yaml.example config/alerts.yaml
# Edit with email/Telegram settings

# 3. Install package
pip install -e .
```

### Shadow Mode Testing

```python
from otonom_trader.daemon import TradingDaemon, ExecutionMode
from otonom_trader.broker import GuardrailConfig

# Create daemon in shadow mode
daemon = TradingDaemon(
    db_path="trader.db",
    strategy_path="strategies/baseline_v1.0.yaml",
    broker_config_path="config/broker.yaml",
    mode=ExecutionMode.SHADOW,
    guard_cfg=GuardrailConfig(
        max_daily_loss_pct=5.0,
        max_drawdown_pct=40.0,
        max_consecutive_losses=5,
    ),
)

# Run once
daemon.run_once()

# Check results
# - Paper trader logs to DB
# - Real broker executes on testnet
# - Alerts sent if issues occur
```

### Production Checklist

Before going live:

- [ ] Test extensively on testnet (shadow mode)
- [ ] Validate strategy performance
- [ ] Configure guardrails appropriately
- [ ] Set up alerts (email + Telegram)
- [ ] Test alert notifications
- [ ] Review API key permissions (no withdrawal rights)
- [ ] Start with small position sizes
- [ ] Monitor closely for first week
- [ ] Have rollback plan ready

## 6. Troubleshooting

### Broker Connection Issues

**Problem**: `BrokerError: Connection timeout`

**Solutions**:
- Check internet connection
- Verify API keys are correct
- Check if testnet is down (use production base_url temporarily)
- Increase timeout in config

### Kill-Switch False Triggers

**Problem**: Kill-switch triggers too early

**Solutions**:
- Adjust guardrail thresholds
- Check if portfolio snapshots are recording correctly
- Review trade history for accuracy
- Temporarily disable with `enabled: false` in GuardrailConfig

### Alerts Not Sending

**Problem**: No email/Telegram notifications

**Solutions**:
- Check `enabled: true` in config
- Test with `notifier.notify("Test", "Test message")`
- For Gmail, use App Password (not account password)
- For Telegram, verify bot token and chat ID
- Check logs for error messages

## 7. Best Practices

**Security**:
- Never commit API keys to git
- Use testnet first
- Restrict API key permissions (no withdrawal)
- Keep secrets in config files (gitignored)

**Risk Management**:
- Start with conservative guardrails (5% daily loss)
- Test kill-switch triggers manually
- Monitor first week closely
- Have emergency shutdown procedure

**Monitoring**:
- Enable all alerts
- Check email/Telegram regularly
- Review daemon logs daily
- Monitor broker connectivity

**Testing**:
- Always use shadow mode before live
- Run for at least 30 days on testnet
- Compare paper vs real execution
- Validate fill prices and slippage

## See Also

- [Strategy Configuration](strategy_config.md)
- [Promotion Workflow](promotion_workflow.md)
- [Backtest Runner](../otonom_trader/otonom_trader/research/backtest_runner.py)
