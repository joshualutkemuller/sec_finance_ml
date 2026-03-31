"""
model.py — Portfolio Concentration Risk Early Warning Monitor

Provides the :class:`RiskLevel` enumeration and the
:class:`RandomForestRiskScorer` which wraps a scikit-learn
``RandomForestClassifier`` with additional SHAP explainability, calibrated
probability scoring, and persistence helpers.

The scorer is trained in binary-classification mode:
    label 0 — LOW or MEDIUM risk (historical samples where no alert fired)
    label 1 — HIGH or CRITICAL risk (historical samples where an alert fired)

``score()`` returns the predicted probability of label 1 (i.e., P(HIGH or
CRITICAL)).  ``classify()`` maps those probabilities to the four-tier
:class:`RiskLevel` using the thresholds in config.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Four-tier risk classification for portfolio concentration risk.

    The ``str`` mixin allows instances to be used directly as strings,
    which simplifies JSON serialisation and DataFrame operations.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def is_actionable(self) -> bool:
        """Return True if this risk level should trigger an external alert.

        Only HIGH and CRITICAL risk levels result in email / Teams alerts;
        LOW and MEDIUM are logged only.

        Returns
        -------
        bool
        """
        return self in (RiskLevel.HIGH, RiskLevel.CRITICAL)


class RandomForestRiskScorer:
    """Random Forest ensemble for portfolio concentration risk scoring.

    Wraps a ``sklearn.ensemble.RandomForestClassifier`` with:
    - Configurable hyperparameters loaded from ``config.yaml``.
    - SHAP ``TreeExplainer`` for per-row feature attribution.
    - Four-tier risk level classification via configurable probability thresholds.
    - ``joblib`` persistence (``save`` / ``load``).

    Parameters
    ----------
    config : dict
        The ``model`` section of config.yaml.  Expected keys:
        ``n_estimators``, ``max_depth``, ``min_samples_leaf``,
        ``random_state``, ``risk_score_threshold_medium``,
        ``risk_score_threshold_high``, ``risk_score_threshold_critical``.
    shap_config : dict, optional
        The ``shap`` section of config.yaml.  Expected keys:
        ``enabled``, ``top_n_features``.
    """

    def __init__(
        self,
        config: dict,
        shap_config: Optional[dict] = None,
    ) -> None:
        self._config = config
        self._shap_config: dict = shap_config or {"enabled": True, "top_n_features": 5}

        self._threshold_medium: float = float(
            config.get("risk_score_threshold_medium", 0.35)
        )
        self._threshold_high: float = float(
            config.get("risk_score_threshold_high", 0.60)
        )
        self._threshold_critical: float = float(
            config.get("risk_score_threshold_critical", 0.85)
        )

        self._model = RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 200)),
            max_depth=config.get("max_depth", 8) or None,
            min_samples_leaf=int(config.get("min_samples_leaf", 5)),
            random_state=int(config.get("random_state", 42)),
            n_jobs=-1,
            class_weight="balanced",
        )
        self._feature_names: list[str] = []
        self._is_fitted: bool = False
        self._explainer = None  # lazy-loaded SHAP explainer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest on labelled historical concentration data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.  Column order must match
            :meth:`~src.feature_engineering.ConcentrationFeatureEngineer.feature_columns`.
        y : pd.Series
            Binary labels — 1 for HIGH/CRITICAL historical events, 0 otherwise.

        Returns
        -------
        None
        """
        if X.empty or y.empty:
            raise ValueError("Training data X and y must not be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length; got {len(X)} vs {len(y)}."
            )

        self._feature_names = list(X.columns)
        logger.info(
            "Fitting RandomForestRiskScorer on %d samples, %d features",
            len(X),
            len(self._feature_names),
        )
        self._model.fit(X.values, y.values)
        self._is_fitted = True
        self._explainer = None  # reset cached explainer after refit
        logger.info(
            "Model fitted. OOB score: %s",
            getattr(self._model, "oob_score_", "N/A (oob_score not enabled)"),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of HIGH or CRITICAL risk.

        The model is trained as a binary classifier (label 1 = HIGH/CRITICAL).
        This method returns ``predict_proba[:, 1]`` — the probability of the
        positive class.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns used during training.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_samples,)`` with values in [0, 1].

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before :meth:`fit` or :meth:`load`.
        """
        self._assert_fitted()
        X_aligned = self._align_features(X)
        probabilities: np.ndarray = self._model.predict_proba(X_aligned.values)
        # Class index 1 corresponds to the positive (HIGH/CRITICAL) class
        positive_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else -1
        if positive_idx == -1:
            # Edge case: model was trained only on negative samples
            logger.warning("Model has no positive class — returning zero scores")
            return np.zeros(len(X))
        return probabilities[:, positive_idx]

    def classify(self, scores: np.ndarray) -> list[RiskLevel]:
        """Map an array of risk scores to :class:`RiskLevel` categories.

        Thresholds are read from config:
            score < threshold_medium                 → LOW
            threshold_medium <= score < threshold_high → MEDIUM
            threshold_high   <= score < threshold_crit → HIGH
            score >= threshold_critical              → CRITICAL

        Parameters
        ----------
        scores : np.ndarray
            Array of probability scores in [0, 1] from :meth:`score`.

        Returns
        -------
        list[RiskLevel]
            Risk level for each score, same length as ``scores``.
        """
        result: list[RiskLevel] = []
        for s in scores:
            if s >= self._threshold_critical:
                result.append(RiskLevel.CRITICAL)
            elif s >= self._threshold_high:
                result.append(RiskLevel.HIGH)
            elif s >= self._threshold_medium:
                result.append(RiskLevel.MEDIUM)
            else:
                result.append(RiskLevel.LOW)
        return result

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values and return a DataFrame of top feature contributions.

        Uses ``shap.TreeExplainer`` which is efficient and exact for tree-based
        models.  Only computed if ``shap.enabled`` is True in config.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.  Uses the same aligned features as :meth:`score`.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index as X, columns named after the top-N
            features.  Each cell contains the raw SHAP value for that feature
            in that row.  If SHAP is disabled, returns an empty DataFrame.

        Raises
        ------
        ImportError
            If the ``shap`` package is not installed.
        """
        if not self._shap_config.get("enabled", True):
            logger.debug("SHAP disabled in config — skipping explain()")
            return pd.DataFrame(index=X.index)

        self._assert_fitted()

        try:
            import shap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'shap' package is required for explainability. "
                "Install it with: pip install shap"
            ) from exc

        X_aligned = self._align_features(X)
        top_n: int = int(self._shap_config.get("top_n_features", 5))

        if self._explainer is None:
            logger.debug("Initialising SHAP TreeExplainer")
            self._explainer = shap.TreeExplainer(self._model)

        # shap_values shape: (n_classes, n_samples, n_features) for RF
        shap_values = self._explainer.shap_values(X_aligned.values)

        # For binary classification scikit-learn RF returns a list of two arrays.
        # Index 1 = SHAP values for the positive class (HIGH/CRITICAL).
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv_positive = shap_values[1]  # shape (n_samples, n_features)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv_positive = shap_values[:, :, 1]
        else:
            sv_positive = np.array(shap_values)

        shap_df = pd.DataFrame(
            sv_positive,
            columns=X_aligned.columns,
            index=X.index,
        )

        # Select top-N features by mean absolute SHAP value
        mean_abs = shap_df.abs().mean().nlargest(top_n)
        top_features = mean_abs.index.tolist()
        logger.debug("Top SHAP features: %s", top_features)

        return shap_df[top_features]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted model (and feature names) to disk using joblib.

        Parameters
        ----------
        path : str
            File system path for the saved artefact (e.g.
            ``models/rf_risk_scorer.joblib``).

        Raises
        ------
        RuntimeError
            If called on an unfitted model.
        """
        self._assert_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "thresholds": {
                "medium": self._threshold_medium,
                "high": self._threshold_high,
                "critical": self._threshold_critical,
            },
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model artefact from disk.

        Parameters
        ----------
        path : str
            File system path to the saved joblib artefact.

        Raises
        ------
        FileNotFoundError
            If the artefact does not exist at the given path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artefact not found: {path}")
        payload: dict = joblib.load(path)
        self._model = payload["model"]
        self._feature_names = payload.get("feature_names", [])
        thresholds = payload.get("thresholds", {})
        self._threshold_medium = thresholds.get("medium", self._threshold_medium)
        self._threshold_high = thresholds.get("high", self._threshold_high)
        self._threshold_critical = thresholds.get("critical", self._threshold_critical)
        self._is_fitted = True
        self._explainer = None
        logger.info(
            "Model loaded from %s (%d features)",
            path,
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Feature importance (convenience)
    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.Series:
        """Return Gini impurity-based feature importances from the fitted model.

        Returns
        -------
        pd.Series
            Importances indexed by feature name, sorted descending.
        """
        self._assert_fitted()
        importances = self._model.feature_importances_
        return (
            pd.Series(importances, index=self._feature_names)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        """Raise if the model has not been fitted or loaded."""
        if not self._is_fitted:
            raise NotFittedError(
                "RandomForestRiskScorer has not been fitted. "
                "Call fit() or load() first."
            )

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has the same columns (and order) as the training data.

        Missing columns are filled with zeros; extra columns are dropped.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Re-aligned DataFrame.
        """
        if not self._feature_names:
            return X
        for col in self._feature_names:
            if col not in X.columns:
                logger.warning("Feature '%s' missing from input; filling with 0", col)
                X = X.copy()
                X[col] = 0.0
        return X[self._feature_names]
