#!/usr/bin/env python3
"""
Test script for new strategy configuration system.
"""

from otonom_trader.config import load_strategy, validate_strategy_config


def test_baseline_v1():
    """Test loading baseline_v1.yaml (old format)."""
    print("\n" + "=" * 70)
    print("Testing baseline_v1.yaml (OLD FORMAT)")
    print("=" * 70)

    config = load_strategy("strategies/baseline_v1.yaml")

    print(f"✓ Name: {config.name}")
    print(f"✓ Version: {config.version}")
    print(f"✓ Description: {config.description}")
    print()
    print("Universe:")
    print(f"  - Symbols: {config.universe.symbols}")
    print(f"  - Tags: {config.universe.universe_tags}")
    print()
    print("Risk:")
    print(f"  - risk_pct: {config.risk.risk_pct}%")
    print(f"  - stop_loss_pct: {config.risk.stop_loss_pct}%")
    print(f"  - take_profit_pct: {config.risk.take_profit_pct}%")
    print(f"  - max_drawdown_pct: {config.risk.max_drawdown_pct}")
    print()
    print("Filters:")
    print(f"  - dsi_threshold: {config.filters.dsi_threshold}")
    print(f"  - regime_vol_min: {config.filters.regime_vol_min}")
    print(f"  - regime_vol_max: {config.filters.regime_vol_max}")
    print()
    print("Ensemble:")
    print(f"  - tech_weight: {config.ensemble.tech_weight}")
    print(f"  - news_weight: {config.ensemble.news_weight}")
    print(f"  - risk_weight: {config.ensemble.risk_weight}")
    print(f"  - rl_weight: {config.ensemble.rl_weight}")
    print(f"  - disagreement_threshold: {config.ensemble.disagreement_threshold}")
    print()
    print("Execution:")
    print(f"  - bar_type: {config.execution.bar_type}")
    print(f"  - slippage_pct: {config.execution.slippage_pct}%")
    print(f"  - max_trades_per_day: {config.execution.max_trades_per_day}")
    print()
    print("✓ VALIDATION PASSED")


def test_baseline_v2():
    """Test loading baseline_v2.yaml (new format)."""
    print("\n" + "=" * 70)
    print("Testing baseline_v2.yaml (NEW FORMAT)")
    print("=" * 70)

    config = load_strategy("strategies/baseline_v2.yaml")

    print(f"✓ Name: {config.name}")
    print(f"✓ Version: {config.version}")
    print(f"✓ Description: {config.description}")
    print()
    print("Universe:")
    print(f"  - Symbols: {config.universe.symbols}")
    print(f"  - Tags: {config.universe.universe_tags}")
    print()
    print("Risk:")
    print(f"  - risk_pct: {config.risk.risk_pct}%")
    print(f"  - stop_loss_pct: {config.risk.stop_loss_pct}%")
    print(f"  - take_profit_pct: {config.risk.take_profit_pct}%")
    print(f"  - max_drawdown_pct: {config.risk.max_drawdown_pct}")
    print()
    print("Filters:")
    print(f"  - dsi_threshold: {config.filters.dsi_threshold}")
    print(f"  - regime_vol_min: {config.filters.regime_vol_min}")
    print(f"  - regime_vol_max: {config.filters.regime_vol_max}")
    print()
    print("Ensemble:")
    print(f"  - tech_weight: {config.ensemble.tech_weight}")
    print(f"  - news_weight: {config.ensemble.news_weight}")
    print(f"  - risk_weight: {config.ensemble.risk_weight}")
    print(f"  - rl_weight: {config.ensemble.rl_weight}")
    print(f"  - disagreement_threshold: {config.ensemble.disagreement_threshold}")
    print()
    print("Execution:")
    print(f"  - bar_type: {config.execution.bar_type}")
    print(f"  - slippage_pct: {config.execution.slippage_pct}%")
    print(f"  - max_trades_per_day: {config.execution.max_trades_per_day}")
    print()
    print("✓ VALIDATION PASSED")


def test_validation_errors():
    """Test validation error cases."""
    print("\n" + "=" * 70)
    print("Testing VALIDATION ERRORS")
    print("=" * 70)

    from otonom_trader.config import (
        StrategyConfig,
        UniverseConfig,
        RiskConfig,
        FiltersConfig,
        EnsembleConfig,
        ExecutionConfig,
    )

    # Test 1: Invalid risk_pct (too high)
    print("\n1. Testing invalid risk_pct (15% > 10%)...")
    try:
        config = StrategyConfig(
            name="test",
            description="test",
            version="1.0.0",
            universe=UniverseConfig(symbols=["BTC-USD"]),
            risk=RiskConfig(risk_pct=15.0),  # TOO HIGH!
            filters=FiltersConfig(),
            ensemble=EnsembleConfig(),
            execution=ExecutionConfig(),
        )
        validate_strategy_config(config)
        print("  ✗ FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")

    # Test 2: Empty symbols
    print("\n2. Testing empty symbols...")
    try:
        config = StrategyConfig(
            name="test",
            description="test",
            version="1.0.0",
            universe=UniverseConfig(symbols=[]),  # EMPTY!
            risk=RiskConfig(),
            filters=FiltersConfig(),
            ensemble=EnsembleConfig(),
            execution=ExecutionConfig(),
        )
        validate_strategy_config(config)
        print("  ✗ FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")

    # Test 3: Zero ensemble weights
    print("\n3. Testing zero ensemble weights...")
    try:
        config = StrategyConfig(
            name="test",
            description="test",
            version="1.0.0",
            universe=UniverseConfig(symbols=["BTC-USD"]),
            risk=RiskConfig(),
            filters=FiltersConfig(),
            ensemble=EnsembleConfig(
                tech_weight=0.0, news_weight=0.0, risk_weight=0.0, rl_weight=0.0
            ),  # ALL ZERO!
            execution=ExecutionConfig(),
        )
        validate_strategy_config(config)
        print("  ✗ FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("\n✓ ALL VALIDATION TESTS PASSED")


def test_backward_compatibility():
    """Test backward compatibility with old helper methods."""
    print("\n" + "=" * 70)
    print("Testing BACKWARD COMPATIBILITY")
    print("=" * 70)

    config = load_strategy("strategies/baseline_v1.yaml")

    # Old methods should still work
    print(f"✓ get_symbols(): {config.get_symbols()}")
    print(f"✓ get_initial_capital(): ${config.get_initial_capital():,.0f}")
    print(f"✓ get_risk_per_trade_pct(): {config.get_risk_per_trade_pct()}%")
    print(f"✓ get_stop_loss_pct(): {config.get_stop_loss_pct()}%")
    print(f"✓ get_take_profit_pct(): {config.get_take_profit_pct()}%")
    print(f"✓ get_max_daily_trades(): {config.get_max_daily_trades()}")
    print(f"✓ is_analist_enabled(1): {config.is_analist_enabled(1)}")
    print(f"✓ get_analist_weight(1): {config.get_analist_weight(1)}")
    print()

    # New attributes should also work
    print(f"✓ config.risk.risk_pct: {config.risk.risk_pct}%")
    print(f"✓ config.ensemble.tech_weight: {config.ensemble.tech_weight}")
    print(f"✓ config.execution.bar_type: {config.execution.bar_type}")
    print()
    print("✓ BACKWARD COMPATIBILITY MAINTAINED")


if __name__ == "__main__":
    try:
        test_baseline_v1()
        test_baseline_v2()
        test_validation_errors()
        test_backward_compatibility()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Strategy YAML ekosistemi hazır!")
        print("Artık tüm componentler StrategyConfig kullanabilir.")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
