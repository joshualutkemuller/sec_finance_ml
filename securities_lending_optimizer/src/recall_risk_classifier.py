"""
Recall Risk Classifier — XGBoost + isotonic calibration.

Estimates the probability that a counterparty will recall pledged collateral
within a configurable horizon (default 5 trading days).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class RecallRiskClassifier:
    """Calibrated XGBoost classifier for counterparty recall risk."""

    def __init__(self, config: dict) -> None:
        self.config = config["recall_classifier"]
        self.horizon_days: int = self.config.get("recall_threshold_days", 5)
        self._model: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost with isotonic calibration.

        Args:
            X: Feature matrix (counterparty-security pairs x features).
            y: Binary target — 1 if recall occurred within horizon, else 0.
        """
        logger.info(
            "Training recall classifier — %d samples, recall_rate=%.3f",
            len(y),
            y.mean(),
        )
        base = XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            random_state=self.config["random_state"],
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        # Isotonic calibration — better than Platt for tree-based models
        self._model = CalibratedClassifierCV(base, cv=5, method="isotonic")
        self._model.fit(X, y)
        logger.info("Recall classifier trained and calibrated")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(recall within horizon) for each row.

        Returns:
            1-D array of probabilities, shape (n_samples,).
        """
        if self._model is None:
            raise RuntimeError("RecallRiskClassifier not trained — call fit() or load()")
        proba = self._model.predict_proba(X)[:, 1]
        logger.debug(
            "Recall proba — mean=%.4f, p90=%.4f",
            proba.mean(),
            np.percentile(proba, 90),
        )
        return proba

    def classify(self, proba: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Convert probabilities to binary recall risk flag."""
        return (proba >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("RecallRiskClassifier saved to %s", path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)
        logger.info("RecallRiskClassifier loaded from %s", path)
