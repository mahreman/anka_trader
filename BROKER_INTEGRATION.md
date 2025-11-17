# Broker Integration Guide

Complete guide for broker shadow mode and live trading integration.

## Overview

The anka_trader system supports three broker modes:

1. **Off Mode**: No broker orders (paper trading only)
2. **Shadow Mode**: Broker receives orders but they're logged only (safe testing)
3. **Live Mode**: Real order execution (production)

## Architecture

```
Decision (Patron)
    ↓
ShadowModeExecutor
    ↓
    ├─→ Paper Trader (local simulation)
    ↓
    └─→ create_broker()
            ↓
        GuardedBroker (risk checks)
            ↓
        BinanceBroker / DummyBroker
            ↓
        Real Broker API (testnet or live)
```

## Configuration

### 1. Broker Configuration (`config/broker.yaml`)

```yaml
broker:
  # Broker type: dummy, binance, alpaca, ibkr
  type: binance

  # Shadow mode: log orders without execution
  shadow_mode: true

  # Binance settings
  binance:
    use_testnet: true  # Use testnet for safety
    api_key: "your_testnet_api_key"
    api_secret: "your_testnet_api_secret"
    base_url: "https://testnet.binance.vision"

# Risk Guardrails (enforced before all orders)
risk_guardrails:
  max_notional_per_order: 10000.0     # Max $10k per order
  max_open_risk_pct: 25.0              # Max 25% of equity at risk
  max_total_positions: 10              # Max 10 open positions
  max_position_size_pct: 10.0          # Max 10% per position
  symbol_blacklist:                    # Never trade these
    - "LUNA-USD"
    - "FTT-USD"
  require_confirmation_above: 50000.0  # Require manual confirm > $50k
  kill_switch: false                   # Emergency halt all trading
```

### 2. Mode Comparison

| Mode | Paper Trading | Broker API Calls | Risk Checks | Use Case |
|------|---------------|------------------|-------------|----------|
| **Off** | ✅ Yes | ❌ No | ❌ No | Pure simulation |
| **Shadow** | ✅ Yes | ✅ Yes (logged only) | ✅ Yes | Testnet validation |
| **Live** | ✅ Yes (parallel) | ✅ Yes (real execution) | ✅ Yes | Production |

## Components

### 1. Broker Factory (`brokers/factory.py`)

Creates broker instances from config:

```python
from otonom_trader.brokers import create_broker

# Load from config/broker.yaml
broker = create_broker()

# Or with custom config
broker = create_broker(config_path="custom_broker.yaml")
```

Returns a `GuardedBroker` that wraps the base broker with risk checks.

### 2. Base Broker Interface (`brokers/base.py`)

All brokers implement:

```python
class Broker:
    def place_order(self, req: OrderRequest) -> OrderResult:
        """Place order on broker."""

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel existing order."""

    def get_open_orders(self) -> List[OrderResult]:
        """Query open orders."""

    def get_positions(self) -> List[dict]:
        """Query current positions."""

    def get_account_balance(self) -> dict:
        """Query account balance."""
```

### 3. Binance Broker (`brokers/binance.py`)

Full Binance Spot API integration:

```python
from otonom_trader.brokers import BinanceBroker

broker = BinanceBroker(
    api_key="your_key",
    api_secret="your_secret",
    use_testnet=True  # Testnet by default
)

# Place market order
result = broker.place_order(
    OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.001)
)

# Place limit order
result = broker.place_order(
    OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.001,
        price=50000.0,
        order_type="LIMIT"
    )
)
```

**Features:**
- ✅ HMAC SHA256 authentication
- ✅ Testnet and live support
- ✅ Market and limit orders
- ✅ Order cancellation
- ✅ Position and balance queries
- ✅ Real-time price lookup

### 4. Risk Guardrails (`brokers/risk_guardrails.py`)

Pre-execution safety checks:

```python
from otonom_trader.brokers import GuardedBroker

# GuardedBroker wraps any broker
guarded = GuardedBroker(
    underlying_broker=binance_broker,
    guardrails=risk_guardrails
)

# Risk checks happen automatically
result = guarded.place_order(req, current_price, current_equity)

# If risk check fails:
# result.ok = False
# result.message = "RISK REJECTED: Order notional $15000 exceeds max $10000"
```

**Checks performed:**
- Kill switch (halts all trading)
- Symbol blacklist
- Max notional per order
- Max position size (% of equity)
- Max total positions
- Large order confirmation threshold

### 5. Shadow Mode Executor (`daemon/broker_integration.py`)

Manages parallel paper + broker execution:

```python
from otonom_trader.daemon.broker_integration import create_shadow_executor

# Create executor
executor = create_shadow_executor(
    broker_config_path="config/broker.yaml",
    enable_broker_orders=True  # Set False to disable broker calls
)

# Execute decision
execution_log = executor.execute_decision(
    session=session,
    decision=decision,
    current_price=50000,
    current_equity=100000
)

# Check results
print(f"Paper fill: ${execution_log.paper_fill_price:.2f}")
print(f"Broker OK: {execution_log.broker_ok}")
print(f"Order ID: {execution_log.broker_order_id}")
print(f"Latency: {execution_log.latency_ms:.1f}ms")
print(f"Slippage: {execution_log.slippage_estimate:.2f}%")
```

**TradeExecutionLog tracks:**
- Paper execution price
- Broker order ID
- Broker success/failure
- Latency (ms)
- Slippage estimate (%)

## Usage Examples

### Example 1: Show Broker Config

```python
from otonom_trader.brokers.config import get_broker_config

config = get_broker_config("config/broker.yaml")
print(f"Broker: {config.broker_type}")
print(f"Shadow mode: {config.shadow_mode}")
print(f"Testnet: {config.use_testnet}")

guardrails = config.get_risk_guardrails()
print(f"Max notional: ${guardrails.max_notional_per_order:,.2f}")
```

### Example 2: Simple Shadow Order

```python
from otonom_trader.brokers import create_broker, OrderRequest

# Create broker (with risk guardrails)
broker = create_broker()

# Place order
result = broker.place_order(
    OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.001),
    current_price=50000,
    current_equity=100000
)

if result.ok:
    print(f"Order placed: {result.order_id}")
else:
    print(f"Order rejected: {result.message}")
```

### Example 3: Daemon Integration (Simple)

```python
from otonom_trader.data import get_session
from otonom_trader.brokers import create_broker, OrderRequest

# Create broker
broker = create_broker()

with get_session() as session:
    # Get today's decisions
    decisions = session.query(Decision).filter(...).all()

    for decision in decisions:
        # Create order request
        order_req = OrderRequest(
            symbol=decision.symbol,
            side=decision.direction,
            qty=decision.strength,
            price=current_price
        )

        # Execute via broker
        result = broker.place_order(
            order_req,
            current_price=current_price,
            current_equity=100000
        )

        print(f"Decision {decision.id}: {result.ok}")
```

### Example 4: Daemon Integration (Advanced)

```python
from otonom_trader.daemon.broker_integration import create_shadow_executor

# Create shadow executor
executor = create_shadow_executor()

with get_session() as session:
    decisions = session.query(Decision).filter(...).all()

    for decision in decisions:
        # Execute with full tracking
        log = executor.execute_decision(
            session, decision, current_price, current_equity
        )

        # Log execution details
        print(f"Paper: ${log.paper_fill_price:.2f}")
        print(f"Broker: {log.broker_ok} ({log.latency_ms:.1f}ms)")
        print(f"Slippage: {log.slippage_estimate:.2f}%")
```

### Example 5: Run Daemon with Broker

See `examples/daemon_with_broker_example.py`:

```bash
# Show config
python examples/daemon_with_broker_example.py config

# Run once (simple)
python examples/daemon_with_broker_example.py simple

# Run once (advanced with tracking)
python examples/daemon_with_broker_example.py advanced

# Run continuous loop
python examples/daemon_with_broker_example.py loop 3600  # Every hour
```

## Deployment Workflow

### Phase 1: Testnet Validation (Shadow Mode)

1. **Configure testnet:**
   ```yaml
   broker:
     type: binance
     shadow_mode: true
     binance:
       use_testnet: true
       api_key: "testnet_key"
       api_secret: "testnet_secret"
   ```

2. **Run daemon with shadow mode:**
   ```bash
   python examples/daemon_with_broker_example.py loop 3600
   ```

3. **Monitor execution logs:**
   - Paper fill prices
   - Broker acknowledgments
   - Latency metrics
   - Slippage estimates
   - Risk rejections

4. **Validate for 7-30 days:**
   - No risk limit violations
   - Acceptable latency (< 500ms)
   - Low slippage (< 1%)
   - API stability

### Phase 2: Live Trading (with safeguards)

1. **Update to live mode:**
   ```yaml
   broker:
     type: binance
     shadow_mode: false  # Enable real execution
     binance:
       use_testnet: false  # Use live API
       api_key: "live_key"
       api_secret: "live_secret"

   risk_guardrails:
     max_notional_per_order: 1000.0  # Start conservative
     max_total_positions: 3
     # ... strict limits
   ```

2. **Start with minimal capital:**
   - Max $1k per order
   - Max 3 positions
   - Max 5% of equity at risk

3. **Monitor closely for first week:**
   - Check every order execution
   - Verify slippage is acceptable
   - Ensure no risk limit breaches
   - Monitor broker account balance

4. **Gradually increase limits:**
   - After 1 week: increase to $5k per order
   - After 1 month: increase to $10k per order
   - Only after proven stability

### Phase 3: Production Deployment

1. **Set production limits:**
   ```yaml
   risk_guardrails:
     max_notional_per_order: 10000.0
     max_open_risk_pct: 25.0
     max_total_positions: 10
   ```

2. **Enable kill switch monitoring:**
   ```python
   # In daemon loop
   if critical_error or max_drawdown_exceeded:
       set_kill_switch(True)  # Halt all trading
   ```

3. **Set up alerts:**
   - Order rejections
   - High slippage
   - API failures
   - Risk limit breaches

## Risk Management

### Built-in Safeguards

1. **Risk Guardrails** (automatic):
   - Max notional per order
   - Max position size
   - Symbol blacklist
   - Kill switch

2. **Shadow Mode** (validation):
   - Log orders without execution
   - Validate API connectivity
   - Test risk limits
   - Measure latency

3. **Paper Trading** (parallel):
   - Always runs alongside broker
   - Compare paper vs. broker fills
   - Detect slippage
   - Audit trail

### Emergency Procedures

**Kill Switch Activation:**
```yaml
# In config/broker.yaml
risk_guardrails:
  kill_switch: true  # Halt ALL trading immediately
```

**Broker Disconnection:**
```python
# In daemon code
try:
    result = broker.place_order(...)
except Exception as e:
    logger.critical(f"Broker error: {e}")
    # Switch to paper trading only
    # Send alert to team
    # Investigate before resuming
```

## Troubleshooting

### Issue: Orders Rejected by Risk Guardrails

**Symptom:**
```
Order rejected: RISK REJECTED: Order notional $15000 exceeds max $10000
```

**Solution:**
1. Check `config/broker.yaml` risk limits
2. Increase `max_notional_per_order` if appropriate
3. Or reduce position sizing in strategy

### Issue: Broker API Errors

**Symptom:**
```
Failed to place Binance order: HTTP 401 Unauthorized
```

**Solution:**
1. Verify API key/secret in config
2. Check API key permissions
3. Ensure testnet key for testnet mode
4. Check IP whitelist if configured

### Issue: High Slippage

**Symptom:**
```
Slippage estimate: 2.5% (warning threshold: 2.0%)
```

**Solution:**
1. Use limit orders instead of market orders
2. Reduce order size
3. Trade more liquid symbols
4. Adjust timing (avoid news events)

## Summary

**Our Implementation vs. User Template:**

| Feature | User Template | Our Implementation | Status |
|---------|---------------|-------------------|--------|
| Config file | `config/broker.yaml` | ✅ `config/broker.yaml` | ✅ Better |
| Factory | `create_broker_from_config()` | `create_broker()` | ✅ Cleaner |
| Binance adapter | `BinanceTestnetBroker` | `BinanceBroker(use_testnet=True)` | ✅ Unified |
| Risk checks | Inline in broker | Separate `GuardedBroker` | ✅ Cleaner |
| Shadow executor | Manual | `ShadowModeExecutor` | ✅ Advanced |
| Execution log | Not included | `TradeExecutionLog` | ✅ Bonus |

**Key Advantages:**
1. ✅ Separation of concerns (GuardedBroker wrapper)
2. ✅ Reusable risk checks across all brokers
3. ✅ Detailed execution logging
4. ✅ Latency and slippage tracking
5. ✅ Complete testnet → live workflow

**Ready for production** with comprehensive safety features! 🚀
