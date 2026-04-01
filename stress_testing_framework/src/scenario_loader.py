"""
Scenario Loader — Reads and validates YAML stress scenario definitions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"scenario_id", "name", "severity", "shocks"}
VALID_SEVERITIES = {"Moderate", "Severe", "Extreme", "Regulatory"}

SHOCK_SCHEMA: dict[str, type] = {
    "rate_shift_bps": (int, float),
    "sofr_shock_bps": (int, float),
    "short_end_shift_bps": (int, float),
    "long_end_shift_bps": (int, float),
    "ig_spread_shock_bps": (int, float),
    "hy_spread_shock_bps": (int, float),
    "equity_shock_pct": float,
    "vol_shock_multiplier": float,
    "bid_ask_multiplier": float,
    "market_depth_reduction_pct": float,
    "demand_surge_factor": float,
    "remove_counterparty_ids": list,
    "recall_risk_spillover_factor": float,
    "replacement_cost_bps": (int, float),
    "haircut_floor_increase_pp": (int, float),
    "exclude_asset_classes": list,
}


class ScenarioLoader:
    """Loads and validates YAML stress scenario files."""

    def load(self, path: str) -> dict:
        """Load a single scenario YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Validated scenario dictionary.

        Raises:
            ValueError: If required fields are missing or values are invalid.
        """
        with open(path) as f:
            scenario = yaml.safe_load(f)

        self._validate(scenario, path)
        logger.info("Loaded scenario: %s (%s)", scenario["name"], scenario["scenario_id"])
        return scenario

    def load_all(self, directory: str) -> list[dict]:
        """Load all YAML scenario files from a directory.

        Returns:
            List of validated scenario dictionaries, sorted by scenario_id.
        """
        scenario_dir = Path(directory)
        yaml_files = sorted(scenario_dir.glob("*.yaml"))

        if not yaml_files:
            logger.warning("No YAML scenario files found in %s", directory)
            return []

        scenarios = []
        for f in yaml_files:
            try:
                scenarios.append(self.load(str(f)))
            except (ValueError, yaml.YAMLError) as exc:
                logger.error("Failed to load scenario %s: %s", f.name, exc)

        logger.info("Loaded %d scenarios from %s", len(scenarios), directory)
        return scenarios

    def _validate(self, scenario: dict, path: str) -> None:
        missing = REQUIRED_FIELDS - set(scenario.keys())
        if missing:
            raise ValueError(f"Scenario {path} missing required fields: {missing}")

        severity = scenario.get("severity", "")
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Scenario {path} has invalid severity '{severity}'. "
                f"Must be one of: {VALID_SEVERITIES}"
            )

        shocks = scenario.get("shocks", {})
        if not isinstance(shocks, dict):
            raise ValueError(f"Scenario {path}: 'shocks' must be a mapping")


def load_scenario(path: str) -> dict:
    """Convenience function to load a single scenario."""
    return ScenarioLoader().load(path)
