"""
model.py — XGBoost-based short squeeze probability scorer and risk tier classifier.

The :class:`XGBoostSqueezeScorer` wraps an ``xgboost.XGBClassifier`` with
helpers for training, calibrated probability scoring, risk tier classification,
feature importance reporting, and model persistence.

The :class:`RiskTier` enum defines the four classification buckets:
    LOW       prob < elevated_threshold
    ELEVATED  elevated_threshold <= prob < high_threshold
    HIGH      high_threshold <= prob < critical_threshold
    CRITICAL  prob >= critical_threshold
"""

from __future__ import annotations

import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk tier enum
# ---------------------------------------------------------------------------


class RiskTier(str, Enum):
    """Four-level squeeze risk classification.

    Values are string-compatible so they serialise cleanly to JSON/CSV.
    """

    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class XGBoostSqueezeScorer:
    """Train, score, and classify securities for short squeeze probability.

    Wraps ``xgboost.XGBClassifier`` with isotonic calibration to produce
    reliable probability outputs.  Probability thresholds for risk tier
    assignment are read from the ``model`` block of ``config.yaml``.

    Args:
        config: The ``model`` block from ``config.yaml``.  Expected keys:
            ``n_estimators``, ``max_depth``, ``learning_rate``,
            ``subsample``, ``colsample_bytree``, ``random_state``,
            ``squeeze_prob_threshold_elevated``,
            ``squeeze_prob_threshold_high``,
            ``squeeze_prob_threshold_critical``.

    Example::

        scorer = XGBoostSqueezeScorer(config["model"])
        scorer.fit(X_train, y_train)
        probs = scorer.score(X_score)
        tiers = scorer.classify(probs)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._threshold_elevated: float = float(
            config.get("squeeze_prob_threshold_elevated", 0.35)
        )
        self._threshold_high: float = float(
            config.get("squeeze_prob_threshold_high", 0.60)
        )
        self._threshold_critical: float = float(
            config.get("squeeze_prob_threshold_critical", 0.80)
        )

        base_model = xgb.XGBClassifier(
            n_estimators=int(config.get("n_estimators", 300)),
            max_depth=int(config.get("max_depth", 5)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.8)),
            colsample_bytree=float(config.get("colsample_bytree", 0.8)),
            random_state=int(config.get("random_state", 42)),
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
        )

        # Wrap with isotonic calibration for reliable probability estimates
        self._model: CalibratedClassifierCV = CalibratedClassifierCV(
            estimator=base_model, method="isotonic", cv=3
        )
        self._feature_names: Optional[list[str]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost classifier with isotonic calibration.

        Args:
            X: Feature matrix. Rows are (ticker, date) observations;
                columns are engineered squeeze features.
            y: Binary target series (1 = squeeze event, 0 = no squeeze).
                Index must align with ``X``.

        Raises:
            ValueError: If ``X`` and ``y`` have different lengths or if
                ``y`` contains values outside {0, 1}.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: X={len(X)}, y={len(y)}"
            )
        unique_vals = set(y.unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target y must be binary (0/1). Found values: {unique_vals}"
            )

        self._feature_names = X.columns.tolist()
        logger.info(
            "Fitting XGBoostSqueezeScorer on %d samples, %d features.",
            len(X),
            X.shape[1],
        )

        self._model.fit(X.values, y.values)
        self._is_fitted = True
        logger.info("Model training complete.")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute calibrated squeeze probability for each row in X.

        Args:
            X: Feature matrix with the same columns used during :meth:`fit`.

        Returns:
            np.ndarray: 1-D float64 array of probabilities in [0, 1],
                one value per row in ``X``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If X is empty.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() before score()."
            )
        if X.empty:
            raise ValueError("Input DataFrame X is empty.")

        probs: np.ndarray = self._model.predict_proba(X.values)[:, 1]
        logger.debug(
            "Scored %d securities. Prob range: [%.4f, %.4f].",
            len(probs),
            probs.min(),
            probs.max(),
        )
        return probs

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, probs: np.ndarray) -> list[RiskTier]:
        """Map squeeze probability array to RiskTier labels.

        Thresholds (configurable in config.yaml):
            prob < elevated_threshold   -> LOW
            prob < high_threshold       -> ELEVATED
            prob < critical_threshold   -> HIGH
            prob >= critical_threshold  -> CRITICAL

        Args:
            probs: 1-D array of probabilities in [0, 1] as returned by
                :meth:`score`.

        Returns:
            list[RiskTier]: Risk tier label for each probability value.
        """
        tiers: list[RiskTier] = []
        for p in probs:
            if p >= self._threshold_critical:
                tiers.append(RiskTier.CRITICAL)
            elif p >= self._threshold_high:
                tiers.append(RiskTier.HIGH)
            elif p >= self._threshold_elevated:
                tiers.append(RiskTier.ELEVATED)
            else:
                tiers.append(RiskTier.LOW)
        return tiers

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances.

        Extracts the importance scores from the underlying XGBoost base
        estimator of the first fold's calibrated classifier.

        Returns:
            pd.DataFrame: DataFrame with columns ``feature`` and
                ``importance``, sorted descending by importance.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # CalibratedClassifierCV stores calibrated classifiers per fold;
        # extract the first estimator's underlying XGBClassifier.
        try:
            base_clf = self._model.calibrated_classifiers_[0].estimator
            importances = base_clf.feature_importances_
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(
                f"Could not extract feature importances: {exc}"
            ) from exc

        names = self._feature_names or [
            f"f{i}" for i in range(len(importances))
        ]
        imp_df = pd.DataFrame(
            {"feature": names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted model to disk using pickle.

        Args:
            path: File path to write. Parent directory is created if it
                does not exist.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save: model is not fitted.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(
                {
                    "model": self._model,
                    "feature_names": self._feature_names,
                    "thresholds": {
                        "elevated": self._threshold_elevated,
                        "high": self._threshold_high,
                        "critical": self._threshold_critical,
                    },
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Model saved to %s.", save_path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: File path to load from.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the file is missing expected keys.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        with open(load_path, "rb") as fh:
            payload = pickle.load(fh)
        self._model = payload["model"]
        self._feature_names = payload["feature_names"]
        thresholds = payload.get("thresholds", {})
        self._threshold_elevated = thresholds.get("elevated", self._threshold_elevated)
        self._threshold_high = thresholds.get("high", self._threshold_high)
        self._threshold_critical = thresholds.get("critical", self._threshold_critical)
        self._is_fitted = True
        logger.info("Model loaded from %s.", load_path)
