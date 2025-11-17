"""
Strategy versioning and promotion utilities.

Implements semantic versioning (major.minor) for trading strategies:
- Major version: Behavioral changes (new analyst, risk model change)
- Minor version: Parameter tuning (risk%, weights, thresholds)

Example:
    baseline_v1.0.yaml → Initial working version
    baseline_v1.1.yaml → Parameter tuning from grid search
    baseline_v2.0.yaml → RL analyst added (behavioral change)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """
    Strategy version information.

    Attributes:
        name: Strategy base name (e.g., "baseline")
        major: Major version number
        minor: Minor version number
        full_version: Full version string (e.g., "1.2")
    """
    name: str
    major: int
    minor: int

    @property
    def full_version(self) -> str:
        """Get full version string (e.g., '1.2')."""
        return f"{self.major}.{self.minor}"

    @property
    def file_name(self) -> str:
        """Get strategy file name (e.g., 'baseline_v1.2.yaml')."""
        return f"{self.name}_v{self.full_version}.yaml"

    def __repr__(self) -> str:
        return f"{self.name} v{self.full_version}"

    def __lt__(self, other: "StrategyVersion") -> bool:
        """Compare versions for sorting."""
        if self.name != other.name:
            return self.name < other.name
        if self.major != other.major:
            return self.major < other.major
        return self.minor < other.minor

    def bump_major(self) -> "StrategyVersion":
        """
        Bump major version (behavioral change).

        Returns:
            New StrategyVersion with major+1, minor=0

        Example:
            >>> v = StrategyVersion("baseline", 1, 2)
            >>> v.bump_major()
            StrategyVersion(name='baseline', major=2, minor=0)
        """
        return StrategyVersion(
            name=self.name,
            major=self.major + 1,
            minor=0,
        )

    def bump_minor(self) -> "StrategyVersion":
        """
        Bump minor version (parameter tuning).

        Returns:
            New StrategyVersion with minor+1

        Example:
            >>> v = StrategyVersion("baseline", 1, 2)
            >>> v.bump_minor()
            StrategyVersion(name='baseline', major=1, minor=3)
        """
        return StrategyVersion(
            name=self.name,
            major=self.major,
            minor=self.minor + 1,
        )

    @classmethod
    def from_file_name(cls, file_name: str) -> Optional["StrategyVersion"]:
        """
        Parse version from strategy file name.

        Args:
            file_name: Strategy file name (e.g., "baseline_v1.2.yaml")

        Returns:
            StrategyVersion or None if parsing fails

        Example:
            >>> StrategyVersion.from_file_name("baseline_v1.2.yaml")
            StrategyVersion(name='baseline', major=1, minor=2)
        """
        # Pattern: name_vMAJOR.MINOR.yaml
        match = re.match(r"^(.+?)_v(\d+)\.(\d+)\.yaml$", file_name)
        if not match:
            logger.warning(f"Could not parse version from: {file_name}")
            return None

        name = match.group(1)
        major = int(match.group(2))
        minor = int(match.group(3))

        return cls(name=name, major=major, minor=minor)

    @classmethod
    def from_path(cls, path: Path) -> Optional["StrategyVersion"]:
        """Parse version from strategy file path."""
        return cls.from_file_name(path.name)


@dataclass
class PromotionCriteria:
    """
    Criteria for strategy promotion.

    Attributes:
        min_test_sharpe: Minimum test Sharpe ratio
        max_test_max_dd: Maximum acceptable drawdown (negative %)
        min_test_trades: Minimum number of test trades
        min_sharpe_improvement: Minimum Sharpe improvement vs previous version
        max_dd_degradation: Max acceptable DD degradation vs previous version
        min_robustness_score: Minimum robustness score (0-1)
    """
    min_test_sharpe: float = 1.2
    max_test_max_dd: float = -30.0
    min_test_trades: int = 50
    min_sharpe_improvement: float = 0.05  # 5% improvement
    max_dd_degradation: float = 5.0       # Max 5% worse DD
    min_robustness_score: float = 0.7     # 70% robust params

    def validate_absolute(
        self,
        test_sharpe: float,
        test_max_dd: float,
        test_trades: int,
    ) -> Tuple[bool, str]:
        """
        Validate absolute criteria (without comparison).

        Args:
            test_sharpe: Test Sharpe ratio
            test_max_dd: Test maximum drawdown
            test_trades: Number of test trades

        Returns:
            Tuple of (passes, reason)

        Example:
            >>> criteria = PromotionCriteria()
            >>> criteria.validate_absolute(1.5, -20.0, 100)
            (True, "All absolute criteria passed")
        """
        if test_sharpe < self.min_test_sharpe:
            return False, f"Test Sharpe {test_sharpe:.2f} < {self.min_test_sharpe}"

        if test_max_dd < self.max_test_max_dd:
            return False, f"Test MaxDD {test_max_dd:.1f}% < {self.max_test_max_dd:.1f}%"

        if test_trades < self.min_test_trades:
            return False, f"Test trades {test_trades} < {self.min_test_trades}"

        return True, "All absolute criteria passed"

    def validate_improvement(
        self,
        new_test_sharpe: float,
        new_test_max_dd: float,
        old_test_sharpe: float,
        old_test_max_dd: float,
    ) -> Tuple[bool, str]:
        """
        Validate improvement criteria vs previous version.

        Args:
            new_test_sharpe: New version test Sharpe
            new_test_max_dd: New version test max DD
            old_test_sharpe: Old version test Sharpe
            old_test_max_dd: Old version test max DD

        Returns:
            Tuple of (passes, reason)

        Example:
            >>> criteria = PromotionCriteria()
            >>> criteria.validate_improvement(1.6, -15.0, 1.5, -18.0)
            (True, "Sharpe improved by 6.67%, DD improved by 3.0%")
        """
        sharpe_improvement = (new_test_sharpe - old_test_sharpe) / old_test_sharpe

        if sharpe_improvement < self.min_sharpe_improvement:
            return False, f"Sharpe improvement {sharpe_improvement*100:.1f}% < {self.min_sharpe_improvement*100:.1f}%"

        # DD improvement (negative is better, so check if it got worse)
        dd_change = new_test_max_dd - old_test_max_dd  # Positive = worse

        if dd_change > self.max_dd_degradation:
            return False, f"DD degraded by {dd_change:.1f}%, max allowed: {self.max_dd_degradation:.1f}%"

        return True, f"Sharpe improved by {sharpe_improvement*100:.1f}%, DD changed by {dd_change:.1f}%"


@dataclass
class PromotionRecord:
    """
    Record of a strategy promotion.

    Attributes:
        from_version: Previous version
        to_version: New version
        promotion_type: "major" or "minor"
        experiment_id: Experiment that produced this version
        test_sharpe: Test Sharpe ratio
        test_cagr: Test CAGR
        test_max_dd: Test maximum drawdown
        changes: Description of changes
        rationale: Why this version was promoted
        timestamp: When promotion occurred
    """
    from_version: Optional[StrategyVersion]
    to_version: StrategyVersion
    promotion_type: str  # "major" or "minor"
    experiment_id: Optional[int]
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    changes: str
    rationale: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_log_entry(self) -> str:
        """
        Format promotion as log entry for STRATEGY_LOG.md.

        Returns:
            Markdown formatted log entry

        Example:
            >>> record.to_log_entry()
            ## v1.1 → v1.2 (Minor) - 2024-01-15

            **Experiment**: #5
            **Performance**: Sharpe=1.6, CAGR=25%, MaxDD=-15%

            **Changes**: Increased risk per trade from 1.0% to 1.5%

            **Rationale**: Grid search showed better risk-adjusted returns...
        """
        from_str = f"v{self.from_version.full_version}" if self.from_version else "Initial"
        to_str = f"v{self.to_version.full_version}"
        date_str = self.timestamp.strftime("%Y-%m-%d")

        lines = [
            f"## {from_str} → {to_str} ({self.promotion_type.capitalize()}) - {date_str}",
            "",
            f"**Experiment**: #{self.experiment_id}" if self.experiment_id else "**Experiment**: Manual",
            f"**Performance**: Sharpe={self.test_sharpe:.2f}, CAGR={self.test_cagr:.1f}%, MaxDD={self.test_max_dd:.1f}%",
            "",
            f"**Changes**: {self.changes}",
            "",
            f"**Rationale**: {self.rationale}",
            "",
            "---",
            "",
        ]

        return "\n".join(lines)


def find_latest_version(
    strategy_name: str,
    strategies_dir: Path = Path("strategies"),
) -> Optional[StrategyVersion]:
    """
    Find latest version of a strategy.

    Args:
        strategy_name: Strategy base name (e.g., "baseline")
        strategies_dir: Directory containing strategy files

    Returns:
        Latest StrategyVersion or None if not found

    Example:
        >>> find_latest_version("baseline")
        StrategyVersion(name='baseline', major=1, minor=2)
    """
    if not strategies_dir.exists():
        logger.warning(f"Strategies directory not found: {strategies_dir}")
        return None

    # Find all matching strategy files
    pattern = f"{strategy_name}_v*.yaml"
    matching_files = list(strategies_dir.glob(pattern))

    if not matching_files:
        logger.info(f"No existing versions found for: {strategy_name}")
        return None

    # Parse versions
    versions = []
    for file_path in matching_files:
        version = StrategyVersion.from_path(file_path)
        if version:
            versions.append(version)

    if not versions:
        return None

    # Return latest
    return max(versions)


def append_to_strategy_log(
    record: PromotionRecord,
    log_path: Path = Path("STRATEGY_LOG.md"),
):
    """
    Append promotion record to strategy log.

    Args:
        record: Promotion record to append
        log_path: Path to strategy log file

    Example:
        >>> record = PromotionRecord(...)
        >>> append_to_strategy_log(record)
    """
    # Create log file if doesn't exist
    if not log_path.exists():
        log_path.write_text(f"# Strategy Promotion Log\n\nLog of all strategy promotions and changes.\n\n---\n\n")

    # Append record
    with open(log_path, "a") as f:
        f.write(record.to_log_entry())

    logger.info(f"Appended promotion record to: {log_path}")


def create_strategy_log_template(
    output_path: Path = Path("STRATEGY_LOG.md"),
) -> None:
    """
    Create initial strategy log template.

    Args:
        output_path: Where to create the log file
    """
    if output_path.exists():
        logger.warning(f"Strategy log already exists: {output_path}")
        return

    template = """# Strategy Promotion Log

This document tracks all strategy versions and promotions.

## Promotion Criteria

A strategy must meet these criteria to be promoted:

### Absolute Requirements
- Test Sharpe ≥ 1.2
- Test Max Drawdown ≤ -30%
- Test Trades ≥ 50

### Improvement Requirements (vs previous version)
- Test Sharpe improvement ≥ 5%
- Max DD degradation ≤ 5%

### Regime Requirements
- Must not blow up in crisis periods (Sharpe > -1.0)
- Should work across multiple regimes

## Versioning Scheme

**Major version** (X.0): Behavioral changes
- New analyst added/removed
- Risk model changed
- Fundamental logic change

**Minor version** (X.Y): Parameter tuning
- Risk % adjusted
- Weights tuned
- Thresholds optimized

## Promotion Workflow

1. **Run Experiment**: Grid search, ablation, or robustness test
2. **Select Best**: Pick top 1-3 runs based on Sharpe + DD + robustness
3. **Promote**: Generate new strategy YAML with updated version
4. **Document**: Add entry to this log with changes and rationale
5. **Validate**: Run full backtest on train+test, check regression
6. **Paper Trade**: Deploy to paper daemon (champion/challenger)

---

## Promotion History

"""

    output_path.write_text(template)
    logger.info(f"Created strategy log template: {output_path}")


def compare_strategy_configs(
    old_config: dict,
    new_config: dict,
) -> str:
    """
    Compare two strategy configs and summarize changes.

    Args:
        old_config: Previous strategy config dict
        new_config: New strategy config dict

    Returns:
        Human-readable summary of changes

    Example:
        >>> changes = compare_strategy_configs(old_cfg, new_cfg)
        >>> print(changes)
        - risk_per_trade: 1.0% → 1.5%
        - stop_loss: 5.0% → 4.5%
    """
    def _flatten_dict(d: dict, parent_key: str = "") -> dict:
        """Flatten nested dict with dot notation."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    old_flat = _flatten_dict(old_config)
    new_flat = _flatten_dict(new_config)

    # Find changes
    changes = []

    # Check all keys in new config
    for key in new_flat:
        new_val = new_flat[key]
        old_val = old_flat.get(key)

        if old_val is None:
            changes.append(f"+ {key}: {new_val} (added)")
        elif old_val != new_val:
            changes.append(f"• {key}: {old_val} → {new_val}")

    # Check for removed keys
    for key in old_flat:
        if key not in new_flat:
            changes.append(f"- {key}: {old_flat[key]} (removed)")

    if not changes:
        return "No changes detected"

    return "\n".join(changes)


def detect_promotion_type(
    old_config: dict,
    new_config: dict,
) -> str:
    """
    Detect if promotion should be major or minor.

    Args:
        old_config: Previous strategy config
        new_config: New strategy config

    Returns:
        "major" or "minor"

    Logic:
        Major if:
        - Number of analysts changed
        - Risk model structure changed
        - Fundamental logic changed

        Minor if:
        - Only parameter values changed
    """
    # Check if analyst count changed
    old_analysts = len([k for k in old_config.keys() if k.startswith("analist_")])
    new_analysts = len([k for k in new_config.keys() if k.startswith("analist_")])

    if old_analysts != new_analysts:
        logger.info(f"Analyst count changed: {old_analysts} → {new_analysts} (major)")
        return "major"

    # Check if risk management structure changed
    old_risk_keys = set(old_config.get("risk_management", {}).keys())
    new_risk_keys = set(new_config.get("risk_management", {}).keys())

    if old_risk_keys != new_risk_keys:
        logger.info(f"Risk management structure changed (major)")
        return "major"

    # Check if ensemble settings changed
    old_ensemble = old_config.get("ensemble", {}).get("enabled", False)
    new_ensemble = new_config.get("ensemble", {}).get("enabled", False)

    if old_ensemble != new_ensemble:
        logger.info(f"Ensemble toggled: {old_ensemble} → {new_ensemble} (major)")
        return "major"

    # Otherwise, assume minor (parameter tuning)
    logger.info("Only parameter values changed (minor)")
    return "minor"
