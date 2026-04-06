"""
Attribution Report — CFO-ready financing cost decomposition narrative.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttributionReportGenerator:
    """Generates structured cost attribution reports for board/CFO consumption."""

    def __init__(self, config: dict) -> None:
        self.output_path = Path(config["output"].get("report_path", "outputs/attribution_report.txt"))

    def generate(
        self,
        period_label: str,
        total_cost_change_usd: float,
        ate_result,
        hte_by_asset_class: pd.DataFrame,
        counterfactual_results: dict[str, float],
        period_summary: dict,
    ) -> str:
        """
        Generate a CFO-ready attribution report.

        Args:
            period_label: e.g., "Q4 2025"
            total_cost_change_usd: Total financing cost change vs prior period ($)
            ate_result: ATEResult from CausalEstimator
            hte_by_asset_class: CATE by asset class from HeterogeneousEffectEstimator
            counterfactual_results: {scenario_name: counterfactual_saving_usd}
            period_summary: Dict of summary stats (n_hikes, total_bps, etc.)
        """
        lines = [
            f"FINANCING COST ATTRIBUTION REPORT — {period_label}",
            f"Author: Joshua Lutkemuller, Managing Director — Securities Finance & Agency Lending",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 70,
            "",
        ]

        # Total change
        pct = (total_cost_change_usd / max(abs(period_summary.get("prior_cost_usd", total_cost_change_usd)), 1)) * 100
        lines += [
            f"TOTAL FINANCING COST CHANGE VS PRIOR PERIOD: "
            f"${total_cost_change_usd:+,.0f} ({pct:+.1f}%)",
            "",
        ]

        # Causal decomposition
        fed_effect_usd = ate_result.ate * period_summary.get("total_fed_hike_bps", 0) * period_summary.get("avg_notional_bn", 1) * 1e4
        market_effect_usd = total_cost_change_usd - fed_effect_usd
        lines += [
            "CAUSAL DECOMPOSITION:",
            f"  Fed Policy Effect ({period_summary.get('n_hikes', 0)} hikes, "
            f"+{period_summary.get('total_fed_hike_bps', 0)}bps total): "
            f"  ${fed_effect_usd:+,.0f}  ({100*fed_effect_usd/max(abs(total_cost_change_usd),1):.1f}%)",
            f"    └─ IV estimate: ${ate_result.ate*1e4:,.0f} per 1bps hike (95% CI: "
            f"[${ate_result.ate_lower_ci*1e4:,.0f}, ${ate_result.ate_upper_ci*1e4:,.0f}])",
            f"  Market / Spread Effect (residual):  "
            f"${market_effect_usd:+,.0f}  ({100*market_effect_usd/max(abs(total_cost_change_usd),1):.1f}%)",
            "",
        ]

        # Counterfactuals
        if counterfactual_results:
            lines.append("COUNTERFACTUAL SCENARIOS:")
            for scenario, saving in counterfactual_results.items():
                lines.append(
                    f"  '{scenario}': estimated saving = ${saving:+,.0f}"
                )
            lines.append("")

        # HTE by asset class
        if not hte_by_asset_class.empty:
            lines.append("HETEROGENEOUS RATE SENSITIVITY (by asset class):")
            for _, row in hte_by_asset_class.head(8).iterrows():
                bar = "█" * max(1, int(abs(row["cate_mean"]) * 20))
                lines.append(
                    f"  {str(row['group_value']):<20s}  CATE={row['cate_mean']:+.4f}  {bar}"
                )
            lines.append("")

        # Interpretation
        lines += [
            "INTERPRETATION:",
            f"  {ate_result.interpretation}",
            f"  Estimation method: {ate_result.method}",
            f"  Observations: {ate_result.n_observations:,}",
            f"  p-value: {ate_result.p_value:.4f}" if not np.isnan(ate_result.p_value) else "  p-value: N/A",
            "",
            "─" * 70,
            "This report was produced using causal ML methods (DoWhy + EconML)",
            "and should be interpreted as causal estimates, not correlations.",
        ]

        report = "\n".join(lines)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report)
        logger.info("Attribution report saved to %s", self.output_path)
        return report
