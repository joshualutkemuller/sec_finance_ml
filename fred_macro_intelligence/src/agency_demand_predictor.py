"""
Agency/Prime Lending Demand Predictor — XGBoost + SHAP.

Predicts rolling changes in agency securities lending utilization as a function
of the macro environment. Key hypothesis: tight funding (low RRP, high SOFR-FF spread,
steep VIX) → higher demand for agency collateral lending.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class AgencyDemandPredictor:
    """XGBoost model for agency lending utilization forecasting with SHAP explainability."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["agency_demand_predictor"]
        self.shap_top_n = self.cfg.get("shap_top_n", 8)
        self._model: XGBRegressor | None = None
        self._explainer = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost on macro features.

        Args:
            X: Macro feature matrix (date index).
            y: Agency lending utilization rate or rolling change (date index).
        """
        self._feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[self._feature_cols].fillna(0.0)
        valid = y.notna() & X_num.notna().all(axis=1)

        logger.info(
            "Training AgencyDemandPredictor: %d samples, mean_y=%.4f",
            valid.sum(), y[valid].mean(),
        )

        self._model = XGBRegressor(
            n_estimators=self.cfg.get("n_estimators", 400),
            max_depth=self.cfg.get("max_depth", 6),
            learning_rate=self.cfg.get("learning_rate", 0.05),
            random_state=self.cfg.get("random_state", 42),
            n_jobs=-1,
            eval_metric="rmse",
            verbosity=0,
        )
        self._model.fit(X_num[valid], y[valid])

        # Initialize SHAP explainer
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP TreeExplainer initialized")
        except ImportError:
            logger.warning("shap not installed — SHAP explanations unavailable")

        logger.info("AgencyDemandPredictor trained")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict agency demand signal."""
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() or load()")
        X_num = X[self._feature_cols].fillna(0.0)
        return self._model.predict(X_num)

    def explain(self, X: pd.DataFrame, n_top: int | None = None) -> pd.DataFrame:
        """Return SHAP values for the provided feature matrix.

        Returns:
            DataFrame of SHAP values, shape (n_rows, n_features).
            Columns sorted by mean absolute SHAP value descending.
        """
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not available")
        X_num = X[self._feature_cols].fillna(0.0)
        shap_values = self._explainer.shap_values(X_num)
        shap_df = pd.DataFrame(shap_values, columns=self._feature_cols, index=X.index)

        top_n = n_top or self.shap_top_n
        top_cols = shap_df.abs().mean().sort_values(ascending=False).head(top_n).index
        return shap_df[top_cols]

    def top_drivers(self, X_row: pd.DataFrame) -> pd.Series:
        """For a single row, return top SHAP drivers as a named Series.

        Useful for PowerBI push and alert messages.
        """
        shap_row = self.explain(X_row, n_top=self.shap_top_n)
        return shap_row.iloc[0].sort_values(key=abs, ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "feature_cols": self._feature_cols,
        }, path)
        logger.info("AgencyDemandPredictor saved to %s", path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._feature_cols = bundle["feature_cols"]
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
        except ImportError:
            pass
        logger.info("AgencyDemandPredictor loaded from %s", path)
