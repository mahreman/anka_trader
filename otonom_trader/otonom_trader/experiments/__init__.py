"""
Experiment tracking and parameter optimization.
"""
from .experiment_engine import (
    run_grid_search,
    run_random_search,
    load_param_grid,
    generate_param_combinations,
    apply_param_overrides,
)
from .grid import ParamGrid
from .config_utils import (
    load_strategy_config,
    save_strategy_config,
    get_nested,
    set_nested,
    apply_param_overrides as apply_config_overrides,
    merge_configs,
)
from .runner import (
    BacktestMetrics,
    run_backtest_for_strategy,
    run_train_test_backtest,
    run_multi_symbol_backtest,
    aggregate_backtest_metrics,
)

__all__ = [
    # Experiment runners
    "run_grid_search",
    "run_random_search",
    # Parameter grid
    "ParamGrid",
    "load_param_grid",
    "generate_param_combinations",
    # Config utilities
    "apply_param_overrides",
    "load_strategy_config",
    "save_strategy_config",
    "get_nested",
    "set_nested",
    "apply_config_overrides",
    "merge_configs",
    # Backtest runner
    "BacktestMetrics",
    "run_backtest_for_strategy",
    "run_train_test_backtest",
    "run_multi_symbol_backtest",
    "aggregate_backtest_metrics",
]
