"""
Heterogeneous Treatment Effects — CausalForest per counterparty and asset class.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Answers: "Which counterparties and asset classes are most causally sensitive
to Fed rate changes?"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HeterogeneousEffectEstimator:
    """
    Uses EconML CausalForestDML to estimate conditional average treatment
    effects (CATE) per subgroup (asset class, counterparty tier, tenor bucket).
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["causal_forest"]
        self.treatment = config["causal"]["treatment"]
        self.outcome = config["causal"]["outcome"]
        self.confounders = config["causal"]["confounders"]
        self.effect_modifiers = config["causal"].get("effect_modifiers", [])
        self._model = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit CausalForestDML on the training data."""
        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        except ImportError as exc:
            raise ImportError("Install econml: pip install econml") from exc

        df = data.dropna(subset=[self.treatment, self.outcome]).copy()
        avail_w = [c for c in self.confounders if c in df.columns]
        avail_x = [c for c in self.effect_modifiers if c in df.columns and df[c].dtype != object]

        Y = df[self.outcome].values
        T = df[self.treatment].values
        W = df[avail_w].fillna(0.0).values if avail_w else None
        X = df[avail_x].fillna(0.0).values if avail_x else np.ones((len(df), 1))

        logger.info(
            "Fitting CausalForestDML: n=%d, n_estimators=%d",
            len(df),
            self.cfg.get("n_estimators", 500),
        )

        self._model = CausalForestDML(
            n_estimators=self.cfg.get("n_estimators", 500),
            min_samples_leaf=self.cfg.get("min_samples_leaf", 20),
            random_state=self.cfg.get("random_state", 42),
            model_y=GradientBoostingRegressor(n_estimators=100, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, random_state=42),
        )
        self._model.fit(Y, T, X=X, W=W)
        logger.info("CausalForestDML fitted")

    def cate_by_subgroup(
        self, data: pd.DataFrame, group_col: str
    ) -> pd.DataFrame:
        """Return average CATE per subgroup value.

        Args:
            data: DataFrame with features and group_col.
            group_col: Column to group by (e.g., 'asset_class').

        Returns:
            DataFrame with columns: group_value, cate_mean, cate_std, n_obs.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first")

        avail_x = [c for c in self.effect_modifiers if c in data.columns and data[c].dtype != object]
        X = data[avail_x].fillna(0.0).values if avail_x else np.ones((len(data), 1))

        cates = self._model.effect(X)
        data = data.copy()
        data["_cate"] = cates

        if group_col not in data.columns:
            logger.warning("Group column '%s' not found", group_col)
            return pd.DataFrame()

        result = (
            data.groupby(group_col)["_cate"]
            .agg(cate_mean="mean", cate_std="std", n_obs="count")
            .reset_index()
            .rename(columns={group_col: "group_value"})
            .sort_values("cate_mean", ascending=False)
        )
        return result

    def cate_summary(self, data: pd.DataFrame) -> dict:
        """Return summary statistics of CATE distribution."""
        if self._model is None:
            raise RuntimeError("Call fit() first")
        avail_x = [c for c in self.effect_modifiers if c in data.columns and data[c].dtype != object]
        X = data[avail_x].fillna(0.0).values if avail_x else np.ones((len(data), 1))
        cates = self._model.effect(X)
        lb, ub = self._model.effect_interval(X, alpha=0.05)
        return {
            "cate_mean": float(cates.mean()),
            "cate_std": float(cates.std()),
            "cate_p10": float(np.percentile(cates, 10)),
            "cate_p90": float(np.percentile(cates, 90)),
            "fraction_positive": float((cates > 0).mean()),
            "ci_lb_mean": float(lb.mean()),
            "ci_ub_mean": float(ub.mean()),
        }
