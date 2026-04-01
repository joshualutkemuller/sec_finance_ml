"""
Results Analyzer — Computes delta metrics between stressed and baseline optimizer outputs.

Produces the full set of stress test KPIs:
  - collateral_shortfall_usd
  - financing_cost_delta_bps
  - capital_charge_delta_usd
  - margin_call_estimate_usd
  - eligible_pool_shrinkage_pct
  - recall_risk_multiplier
  - pnl_impact_usd
  - hhi_delta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Complete stress test result for a single scenario."""

    scenario_id: str
    scenario_name: str
    severity: str

    # Core KPIs
    collateral_shortfall_usd: float
    financing_cost_delta_bps: float
    capital_charge_delta_usd: float
    margin_call_estimate_usd: float
    eligible_pool_shrinkage_pct: float
    recall_risk_multiplier: float
    pnl_impact_usd: float
    hhi_delta: float
    days_to_cover_delta: float

    # Breach flags
    any_breach: bool = False
    breach_details: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.breach_details is None:
            self.breach_details = {}

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "severity": self.severity,
            "collateral_shortfall_usd": self.collateral_shortfall_usd,
            "financing_cost_delta_bps": self.financing_cost_delta_bps,
            "capital_charge_delta_usd": self.capital_charge_delta_usd,
            "margin_call_estimate_usd": self.margin_call_estimate_usd,
            "eligible_pool_shrinkage_pct": self.eligible_pool_shrinkage_pct,
            "recall_risk_multiplier": self.recall_risk_multiplier,
            "pnl_impact_usd": self.pnl_impact_usd,
            "hhi_delta": self.hhi_delta,
            "days_to_cover_delta": self.days_to_cover_delta,
            "any_breach": self.any_breach,
            "breach_details": self.breach_details,
        }


class ResultsAnalyzer:
    """Computes stress KPIs by comparing stressed vs. baseline optimizer outputs."""

    def __init__(self, breach_thresholds: dict | None = None) -> None:
        self.thresholds = breach_thresholds or {
            "collateral_shortfall_usd": 10_000_000,
            "financing_cost_delta_bps": 50,
            "eligible_pool_shrinkage_pct": 0.15,
            "capital_charge_delta_usd": 5_000_000,
        }

    def analyze(
        self,
        scenario: dict,
        baseline_collateral: pd.DataFrame,
        stressed_collateral: pd.DataFrame,
        baseline_cost: float,
        stressed_cost: float,
        baseline_capital: float,
        stressed_capital: float,
        baseline_recall: float = 0.0,
        stressed_recall: float = 0.0,
    ) -> StressResult:
        """Compute all KPIs for a scenario.

        Args:
            scenario: Loaded scenario dict.
            baseline_collateral: Collateral pool at baseline values.
            stressed_collateral: Collateral pool after shocks applied.
            baseline_cost: Baseline total financing cost (decimal, annualized).
            stressed_cost: Stressed total financing cost.
            baseline_capital: Baseline capital charge (USD).
            stressed_capital: Stressed capital charge (USD).
            baseline_recall: Baseline weighted recall risk.
            stressed_recall: Stressed weighted recall risk.

        Returns:
            StressResult with all computed KPIs.
        """
        collateral_shortfall = self._collateral_shortfall(
            baseline_collateral, stressed_collateral
        )
        financing_delta_bps = (stressed_cost - baseline_cost) * 10_000
        capital_delta = stressed_capital - baseline_capital
        margin_call = self._margin_call_estimate(stressed_collateral, baseline_collateral)
        pool_shrinkage = self._eligible_pool_shrinkage(baseline_collateral, stressed_collateral)
        recall_mult = (stressed_recall / baseline_recall) if baseline_recall > 1e-9 else 1.0
        pnl_impact = self._pnl_impact(collateral_shortfall, financing_delta_bps, capital_delta)
        hhi_delta = self._hhi_delta(baseline_collateral, stressed_collateral)
        dtc_delta = self._days_to_cover_delta(baseline_collateral, stressed_collateral)

        breaches: dict[str, float] = {}
        if collateral_shortfall > self.thresholds.get("collateral_shortfall_usd", float("inf")):
            breaches["collateral_shortfall_usd"] = collateral_shortfall
        if financing_delta_bps > self.thresholds.get("financing_cost_delta_bps", float("inf")):
            breaches["financing_cost_delta_bps"] = financing_delta_bps
        if pool_shrinkage > self.thresholds.get("eligible_pool_shrinkage_pct", float("inf")):
            breaches["eligible_pool_shrinkage_pct"] = pool_shrinkage
        if capital_delta > self.thresholds.get("capital_charge_delta_usd", float("inf")):
            breaches["capital_charge_delta_usd"] = capital_delta

        result = StressResult(
            scenario_id=scenario["scenario_id"],
            scenario_name=scenario["name"],
            severity=scenario["severity"],
            collateral_shortfall_usd=collateral_shortfall,
            financing_cost_delta_bps=financing_delta_bps,
            capital_charge_delta_usd=capital_delta,
            margin_call_estimate_usd=margin_call,
            eligible_pool_shrinkage_pct=pool_shrinkage,
            recall_risk_multiplier=recall_mult,
            pnl_impact_usd=pnl_impact,
            hhi_delta=hhi_delta,
            days_to_cover_delta=dtc_delta,
            any_breach=bool(breaches),
            breach_details=breaches,
        )

        if breaches:
            logger.warning(
                "Scenario %s — %d threshold breaches: %s",
                scenario["scenario_id"],
                len(breaches),
                list(breaches.keys()),
            )

        return result

    # ── KPI calculation methods ───────────────────────────────────────────────

    def _collateral_shortfall(
        self, baseline: pd.DataFrame, stressed: pd.DataFrame
    ) -> float:
        """Max(0, required_coverage - stressed_collateral_value)."""
        if "collateral_value" not in stressed.columns:
            return 0.0
        stressed_total = stressed["collateral_value"].sum()
        required = baseline["collateral_value"].sum() if "collateral_value" in baseline.columns else stressed_total
        return float(max(0.0, required - stressed_total))

    def _margin_call_estimate(
        self, stressed: pd.DataFrame, baseline: pd.DataFrame
    ) -> float:
        """Estimated variation margin call (stressed - baseline collateral value delta)."""
        if "collateral_value" not in stressed.columns:
            return 0.0
        base_val = baseline["collateral_value"].sum() if "collateral_value" in baseline.columns else 0.0
        stressed_val = stressed["collateral_value"].sum()
        return float(max(0.0, base_val - stressed_val))

    def _eligible_pool_shrinkage(
        self, baseline: pd.DataFrame, stressed: pd.DataFrame
    ) -> float:
        """Fraction of eligible collateral pool that became ineligible under stress."""
        if "eligible" in stressed.columns:
            baseline_eligible = int(baseline.get("eligible", pd.Series([True] * len(baseline))).sum())
            stressed_eligible = int(stressed["eligible"].sum())
        elif "market_value" in stressed.columns and "market_value" in baseline.columns:
            baseline_eligible = len(baseline)
            stressed_eligible = int((stressed["market_value"] > 0).sum())
        else:
            return 0.0

        if baseline_eligible == 0:
            return 0.0
        return float((baseline_eligible - stressed_eligible) / baseline_eligible)

    def _pnl_impact(
        self, shortfall: float, cost_delta_bps: float, capital_delta: float
    ) -> float:
        """Rough P&L impact estimate combining key channels."""
        cost_pnl = cost_delta_bps / 10_000 * 1_000_000_000  # scale to 1bn notional
        return float(-(shortfall + cost_pnl + capital_delta * 0.1))

    def _hhi_delta(
        self, baseline: pd.DataFrame, stressed: pd.DataFrame
    ) -> float:
        """Change in HHI (Herfindahl-Hirschman Index) of collateral pool."""
        if "asset_class" not in baseline.columns or "market_value" not in baseline.columns:
            return 0.0

        def hhi(df: pd.DataFrame) -> float:
            total = df["market_value"].sum()
            if total <= 0:
                return 0.0
            shares = df.groupby("asset_class")["market_value"].sum() / total
            return float((shares ** 2).sum())

        return hhi(stressed) - hhi(baseline)

    def _days_to_cover_delta(
        self, baseline: pd.DataFrame, stressed: pd.DataFrame
    ) -> float:
        """Change in average days-to-cover across the portfolio."""
        if "days_to_cover" not in stressed.columns:
            return 0.0
        base_dtc = baseline["days_to_cover"].mean() if "days_to_cover" in baseline.columns else 0.0
        stressed_dtc = stressed["days_to_cover"].mean()
        return float(stressed_dtc - base_dtc)

    def build_summary_dataframe(self, results: list[StressResult]) -> pd.DataFrame:
        """Convert a list of StressResult objects to a summary DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results])
