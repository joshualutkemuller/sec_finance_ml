"""
Causal Estimator — Average Treatment Effect via Instrumental Variables.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Uses DoWhy for causal graph construction, identification, and ATE estimation.
FOMC meeting days serve as the instrumental variable — exogenous shocks to
Fed Funds Rate that are uncorrelated with simultaneous market conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ATEResult:
    """Average Treatment Effect estimation result."""
    ate: float                        # $/bps or bps/bps
    ate_lower_ci: float
    ate_upper_ci: float
    ate_std_error: float
    p_value: float
    method: str
    n_observations: int
    interpretation: str


class CausalEstimator:
    """
    Estimates the causal effect of Fed Funds Rate changes on financing costs
    using DoWhy causal graphs and instrumental variable estimation.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["causal"]
        self.ate_cfg = config["ate_estimator"]
        self.treatment = self.cfg["treatment"]
        self.outcome = self.cfg["outcome"]
        self.instrument = self.cfg["instrument"]
        self.confounders = self.cfg["confounders"]

    def estimate_ate(self, data: pd.DataFrame) -> ATEResult:
        """
        Estimate the Average Treatment Effect of a 1bps Fed Funds change
        on financing costs using IV estimation via DoWhy.
        """
        logger.info(
            "Estimating ATE: T=%s → Y=%s (IV=%s), n=%d",
            self.treatment, self.outcome, self.instrument, len(data),
        )
        method = self.ate_cfg.get("method", "iv")

        try:
            return self._estimate_dowhy(data, method)
        except Exception as exc:
            logger.warning("DoWhy estimation failed (%s) — falling back to 2SLS", exc)
            return self._estimate_2sls(data)

    def _estimate_dowhy(self, data: pd.DataFrame, method: str) -> ATEResult:
        """Use DoWhy causal model for identification and estimation."""
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError as exc:
            raise ImportError("Install dowhy: pip install dowhy") from exc

        graph_str = self._build_causal_graph()
        model = CausalModel(
            data=data.dropna(),
            treatment=self.treatment,
            outcome=self.outcome,
            graph=graph_str,
            instruments=[self.instrument] if self.instrument in data.columns else None,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        logger.info("Identified estimand: %s", identified_estimand.estimands)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.instrumental_variable",
            method_params={"iv_instrument_name": self.instrument}
            if self.instrument in data.columns
            else {},
            confidence_intervals=True,
            test_significance=True,
        )

        ate_val = float(estimate.value)
        ci = estimate.get_confidence_intervals()
        lower = float(ci[0]) if ci is not None else ate_val - 0.1 * abs(ate_val)
        upper = float(ci[1]) if ci is not None else ate_val + 0.1 * abs(ate_val)

        return ATEResult(
            ate=ate_val,
            ate_lower_ci=lower,
            ate_upper_ci=upper,
            ate_std_error=(upper - lower) / (2 * 1.96),
            p_value=float(estimate.test_stat_significance().get("p_value", np.nan)),
            method="dowhy_iv",
            n_observations=len(data.dropna()),
            interpretation=(
                f"A 1bps increase in Fed Funds Rate causes a "
                f"{ate_val:.4f} bps change in financing cost "
                f"(95% CI: [{lower:.4f}, {upper:.4f}])"
            ),
        )

    def _estimate_2sls(self, data: pd.DataFrame) -> ATEResult:
        """Two-Stage Least Squares fallback."""
        from statsmodels.sandbox.regression.gmm import IV2SLS
        import statsmodels.api as sm

        df = data.dropna(subset=[self.treatment, self.outcome, self.instrument]).copy()
        avail_confounders = [c for c in self.confounders if c in df.columns]

        X = sm.add_constant(df[[self.instrument] + avail_confounders])
        Z = sm.add_constant(df[[self.instrument] + avail_confounders])

        try:
            model = IV2SLS(df[self.outcome], X, Z).fit()
            treatment_coef = model.params.get(self.treatment, model.params.iloc[1])
            ci = model.conf_int()
            return ATEResult(
                ate=float(treatment_coef),
                ate_lower_ci=float(ci.loc[self.treatment, 0]) if self.treatment in ci.index else float(treatment_coef - 0.05),
                ate_upper_ci=float(ci.loc[self.treatment, 1]) if self.treatment in ci.index else float(treatment_coef + 0.05),
                ate_std_error=float(model.bse.get(self.treatment, 0.02)),
                p_value=float(model.pvalues.get(self.treatment, np.nan)),
                method="2sls_fallback",
                n_observations=len(df),
                interpretation=(
                    f"2SLS estimate: 1bps Fed Funds change → "
                    f"{treatment_coef:.4f} bps financing cost change"
                ),
            )
        except Exception as exc:
            logger.error("2SLS failed: %s", exc)
            # OLS last resort
            ols = sm.OLS(df[self.outcome], sm.add_constant(df[[self.treatment] + avail_confounders])).fit()
            coef = float(ols.params.get(self.treatment, 0.0))
            return ATEResult(
                ate=coef,
                ate_lower_ci=coef - 2 * float(ols.bse.get(self.treatment, 0.05)),
                ate_upper_ci=coef + 2 * float(ols.bse.get(self.treatment, 0.05)),
                ate_std_error=float(ols.bse.get(self.treatment, 0.05)),
                p_value=float(ols.pvalues.get(self.treatment, np.nan)),
                method="ols_fallback",
                n_observations=len(df),
                interpretation=f"OLS (non-causal fallback): coef={coef:.4f}",
            )

    def _build_causal_graph(self) -> str:
        """Build a GML causal graph string for DoWhy."""
        nodes = [self.treatment, self.outcome, self.instrument] + self.confounders
        edges = (
            [(self.instrument, self.treatment)]
            + [(self.treatment, self.outcome)]
            + [(c, self.outcome) for c in self.confounders]
            + [(c, self.treatment) for c in self.confounders]
        )
        node_strs = "\n".join(f'  node [ id "{n}" label "{n}" ]' for n in nodes)
        edge_strs = "\n".join(f'  edge [ source "{s}" target "{t}" ]' for s, t in edges)
        return f"graph [\n{node_strs}\n{edge_strs}\n]"
