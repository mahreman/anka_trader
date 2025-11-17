"""
Unit tests for event-based backtest module.

Tests backtest configuration and utility functions
without requiring full database setup.
"""

import datetime as dt

from otonom_trader.eval.backtest import BacktestConfig, _apply_slippage


def test_apply_slippage_basic():
    """Test that slippage is applied correctly for buying."""
    price = 100.0
    # 10 bps = 0.1% = 0.001
    # direction=+1 (buying) increases price
    out = _apply_slippage(price, 10.0, direction=1)

    # 100 * (1 + 0.001) = 100.1
    assert abs(out - 100.1) < 1e-6


def test_apply_slippage_zero():
    """Test that zero slippage returns original price."""
    price = 50.0
    out = _apply_slippage(price, 0.0, direction=1)

    assert abs(out - 50.0) < 1e-9


def test_apply_slippage_sell_direction():
    """Test that selling direction decreases price."""
    price = 100.0
    # direction=-1 (selling) decreases price
    out = _apply_slippage(price, 10.0, direction=-1)

    # 100 * (1 - 0.001) = 99.9
    assert abs(out - 99.9) < 1e-6


def test_apply_slippage_large_bps():
    """Test large slippage (e.g., 100 bps = 1%)."""
    price = 1000.0
    out = _apply_slippage(price, 100.0, direction=1)

    # 1000 * (1 + 0.01) = 1010.0
    assert abs(out - 1010.0) < 1e-6


def test_backtest_config_defaults():
    """Test BacktestConfig default values."""
    cfg = BacktestConfig()

    assert cfg.holding_days == 5
    assert cfg.slippage_bps == 5.0


def test_backtest_config_custom_values():
    """Test BacktestConfig with custom values."""
    cfg = BacktestConfig(holding_days=10, slippage_bps=25.0)

    assert cfg.holding_days == 10
    assert cfg.slippage_bps == 25.0


def test_backtest_config_extreme_values():
    """Test BacktestConfig with extreme values."""
    # Very long holding period
    cfg1 = BacktestConfig(holding_days=252, slippage_bps=0.0)
    assert cfg1.holding_days == 252
    assert cfg1.slippage_bps == 0.0

    # Very high slippage
    cfg2 = BacktestConfig(holding_days=1, slippage_bps=500.0)
    assert cfg2.holding_days == 1
    assert cfg2.slippage_bps == 500.0


def test_slippage_calculation_accuracy():
    """Test slippage calculation with various basis point values (buying)."""
    test_cases = [
        (100.0, 1.0, 100.01),     # 1 bp
        (100.0, 5.0, 100.05),     # 5 bp
        (100.0, 10.0, 100.10),    # 10 bp
        (100.0, 50.0, 100.50),    # 50 bp
        (50.0, 20.0, 50.10),      # 20 bp on lower price
        (1000.0, 15.0, 1001.50),  # 15 bp on higher price
    ]

    for price, bps, expected in test_cases:
        result = _apply_slippage(price, bps, direction=1)
        assert abs(result - expected) < 1e-6, f"Failed for price={price}, bps={bps}"
