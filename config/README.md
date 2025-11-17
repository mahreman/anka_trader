# Configuration Files

This directory contains configuration files for the trading system.

## Files

### broker.yaml (Required for live trading)
Broker API credentials and settings.

**Setup**:
```bash
cp broker.yaml.example broker.yaml
# Edit broker.yaml with your API credentials
```

**Important**:
- Start with `testnet: true` for testing
- NEVER commit real API keys to git
- Keep API secrets secure

### alerts.yaml (Optional)
Email and Telegram notification settings.

**Setup**:
```bash
cp alerts.yaml.example alerts.yaml
# Edit alerts.yaml with your notification preferences
```

**Supported channels**:
- Email (via SMTP)
- Telegram (via Bot API)

## Security

All `*.yaml` files (except `*.example`) are gitignored to prevent accidental commit of sensitive credentials.

Always keep your API keys and secrets secure.
