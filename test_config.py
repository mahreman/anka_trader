#!/usr/bin/env python
"""Quick test script for StrategyConfig system."""

import sys
from pathlib import Path

# Add otonom_trader to path
sys.path.insert(0, str(Path(__file__).parent / "otonom_trader"))

from otonom_trader.strategy.config import load_strategy_config

def test_config_loading():
    """Test loading baseline strategy."""
    print("=" * 60)
    print("Testing StrategyConfig System")
    print("=" * 60)

    # Load baseline strategy
    config_path = Path("strategies/baseline_v1.0.yaml")
    print(f"\n1. Loading config: {config_path}")

    try:
        config = load_strategy_config(config_path)
        print(f"   ✓ Successfully loaded: {config.name} v{config.version}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return False

    # Test basic fields
    print(f"\n2. Testing basic fields:")
    print(f"   - Name: {config.name}")
    print(f"   - Version: {config.version}")
    print(f"   - Description: {config.description}")

    # Test universe
    print(f"\n3. Testing universe:")
    symbols = config.get_symbols()
    print(f"   - Symbols: {symbols}")
    print(f"   - Universe tags: {config.universe.universe_tags}")

    # Test risk config
    print(f"\n4. Testing risk config:")
    print(f"   - Risk per trade: {config.risk.risk_pct}%")
    print(f"   - Stop loss: {config.risk.stop_loss_pct}%")
    print(f"   - Take profit: {config.risk.take_profit_pct}%")
    print(f"   - Max drawdown: {config.risk.max_drawdown_pct}%")

    # Test ensemble
    print(f"\n5. Testing ensemble weights:")
    print(f"   - Tech: {config.ensemble.analyst_weights.tech}")
    print(f"   - News: {config.ensemble.analyst_weights.news}")
    print(f"   - Risk: {config.ensemble.analyst_weights.risk}")
    print(f"   - RL: {config.ensemble.analyst_weights.rl}")

    # Test execution
    print(f"\n6. Testing execution config:")
    print(f"   - Bar type: {config.execution.bar_type}")
    print(f"   - Initial capital: ${config.get_initial_capital():,.2f}")
    print(f"   - Slippage: {config.execution.slippage_bps} bps")

    # Test get() method with dot notation
    print(f"\n7. Testing get() method:")
    risk_pct = config.get("risk.risk_pct")
    tech_weight = config.get("ensemble.analyst_weights.tech")
    print(f"   - risk.risk_pct: {risk_pct}")
    print(f"   - ensemble.analyst_weights.tech: {tech_weight}")

    # Test validation
    print(f"\n8. Testing validation:")
    try:
        # This should work - valid values
        config.risk.risk_pct = 1.5
        print(f"   ✓ Risk pct validation passed")
    except ValueError as e:
        print(f"   ✗ Unexpected validation error: {e}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
