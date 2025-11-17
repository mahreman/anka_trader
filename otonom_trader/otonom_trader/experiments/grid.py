"""
Parameter grid loader and combination generator.

Loads parameter grids from YAML files and generates all combinations
for grid search experiments.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class ParamGrid:
    """
    Parameter grid for systematic search.

    Attributes:
        params: Dictionary mapping parameter names to lists of values
        search_method: "grid" (exhaustive) or "random" (sampling)
        random_samples: Number of random samples (if search_method="random")

    Example:
        >>> grid = ParamGrid(
        ...     params={"risk_pct": [0.5, 1.0], "stop_loss": [3, 5]},
        ...     search_method="grid"
        ... )
        >>> combinations = list(grid.iter_combinations())
        >>> len(combinations)  # 2 * 2 = 4
        4
    """

    params: Dict[str, List[Any]]
    search_method: str = "grid"
    random_samples: int = 50

    @classmethod
    def from_yaml(cls, path: str) -> "ParamGrid":
        """
        Load parameter grid from YAML file.

        Expected YAML structure:
            search_method: "grid"  # or "random"
            random_samples: 50     # only for random search
            parameters:
              risk_pct:
                values: [0.5, 1.0, 2.0]
              stop_loss_pct:
                values: [3.0, 5.0, 8.0]

        Args:
            path: Path to YAML file

        Returns:
            ParamGrid instance

        Raises:
            ValueError: If YAML structure is invalid

        Example:
            >>> grid = ParamGrid.from_yaml("grids/my_grid.yaml")
            >>> print(grid.params)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Grid YAML must be a dict, got {type(data)}")

        # Support both "params" (simple) and "parameters" (structured) keys
        if "params" in data:
            # Simple format: params: {risk_pct: [0.5, 1.0]}
            params = data["params"]
            if not isinstance(params, dict):
                raise ValueError("'params' must be a dict of name -> list")

            for k, v in params.items():
                if not isinstance(v, list):
                    raise ValueError(f"Param '{k}' must have a list of values.")

        elif "parameters" in data:
            # Structured format: parameters: {risk_pct: {values: [0.5, 1.0]}}
            params_structured = data["parameters"]
            if not isinstance(params_structured, dict):
                raise ValueError("'parameters' must be a dict")

            params = {}
            for param_name, param_config in params_structured.items():
                if not isinstance(param_config, dict):
                    raise ValueError(f"Parameter '{param_name}' config must be a dict")

                if "values" in param_config:
                    values = param_config["values"]
                    if not isinstance(values, list):
                        raise ValueError(f"Parameter '{param_name}' values must be a list")
                    params[param_name] = values
                else:
                    raise ValueError(f"Parameter '{param_name}' must have 'values' key")

        else:
            raise ValueError("Grid YAML must have either 'params' or 'parameters' key")

        # Get search method
        search_method = data.get("search_method", "grid")
        random_samples = data.get("random_samples", 50)

        return cls(
            params=params,
            search_method=search_method,
            random_samples=random_samples,
        )

    def iter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations.

        For grid search, returns all combinations (Cartesian product).
        For random search, returns random samples.

        Returns:
            List of parameter combination dictionaries

        Example:
            >>> grid = ParamGrid(
            ...     params={"risk_pct": [0.5, 1.0], "stop_loss_pct": [3, 5]},
            ...     search_method="grid"
            ... )
            >>> list(grid.iter_combinations())
            [
                {"risk_pct": 0.5, "stop_loss_pct": 3},
                {"risk_pct": 0.5, "stop_loss_pct": 5},
                {"risk_pct": 1.0, "stop_loss_pct": 3},
                {"risk_pct": 1.0, "stop_loss_pct": 5},
            ]
        """
        if not self.params:
            return [{}]

        keys = list(self.params.keys())
        values_lists = [self.params[k] for k in keys]

        if self.search_method == "grid":
            # Exhaustive grid search
            combinations = []
            for combo in itertools.product(*values_lists):
                combinations.append(dict(zip(keys, combo)))
            return combinations

        elif self.search_method == "random":
            # Random search
            combinations = []
            for _ in range(self.random_samples):
                combo = {}
                for key, values in self.params.items():
                    combo[key] = random.choice(values)
                combinations.append(combo)
            return combinations

        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def __len__(self) -> int:
        """Return number of combinations."""
        if self.search_method == "grid":
            if not self.params:
                return 0
            count = 1
            for values in self.params.values():
                count *= len(values)
            return count
        else:  # random
            return self.random_samples

    def __repr__(self) -> str:
        return (
            f"ParamGrid(params={list(self.params.keys())}, "
            f"search_method={self.search_method}, "
            f"combinations={len(self)})"
        )
