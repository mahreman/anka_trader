"""
Strategy management package.

Provides utilities for:
- Strategy versioning (major.minor)
- Promotion workflows
- Criteria validation
- Champion/challenger comparison
"""

from .versioning import (
    StrategyVersion,
    PromotionCriteria,
    PromotionRecord,
    find_latest_version,
    create_strategy_log_template,
)

from .promotion import (
    PromotionCandidate,
    extract_promotion_candidates,
    validate_promotion,
    promote_strategy,
    run_promotion_workflow,
    compare_champion_challenger,
)

__all__ = [
    # Versioning
    "StrategyVersion",
    "PromotionCriteria",
    "PromotionRecord",
    "find_latest_version",
    "create_strategy_log_template",
    # Promotion
    "PromotionCandidate",
    "extract_promotion_candidates",
    "validate_promotion",
    "promote_strategy",
    "run_promotion_workflow",
    "compare_champion_challenger",
]
