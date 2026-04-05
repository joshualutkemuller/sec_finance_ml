"""
Regime Detector — Labels each trading day with a macro rate regime.

Method A — Hidden Markov Model (hmmlearn):
  Latent states capture non-obvious regime transitions with probabilistic boundaries.
  Useful for generating regime transition probabilities.

Method B — KMeans on PCA factors (interpretable):
  Reduces dimensionality then clusters. Easier to label and explain to traders.

Ensemble: both methods vote; HMM tiebreaks on transition uncertainty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Regime names — update these after initial fit to match observed cluster profiles
DEFAULT_REGIME_LABELS = {
    0: "Easing / QE",
    1: "Neutral / Normal",
    2: "Hiking Cycle",
    3: "Stress / Crisis",
    4: "Late Cycle / Inversion",
}

REGIME_COLORS = {
    "Easing / QE": "#2ecc71",
    "Neutral / Normal": "#3498db",
    "Hiking Cycle": "#f39c12",
    "Stress / Crisis": "#e74c3c",
    "Late Cycle / Inversion": "#9b59b6",
}

# Features most informative for regime detection
REGIME_FEATURES = [
    "sofr", "curve_10y2y", "curve_10y3m",
    "hy_oas", "ig_oas", "vix",
    "rrp_balance_bn", "funding_stress_index",
    "fed_sofr_spread", "curve_butterfly",
]


class RegimeDetector:
    """Unsupervised macro regime detector using HMM and/or KMeans."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["regime_detector"]
        self.method = self.cfg.get("method", "hmm")
        self.n_regimes = int(self.cfg.get("n_regimes", 5))
        self.regime_labels: dict[int, str] = {
            int(k): v for k, v in self.cfg.get("regime_labels", DEFAULT_REGIME_LABELS).items()
        }
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=self.cfg.get("pca_components", 5))
        self._hmm_model = None
        self._kmeans_model = None

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """Fit regime detection models on the full feature history."""
        X = self._prepare(features)
        if self.method in ("hmm", "ensemble"):
            self._fit_hmm(X)
        if self.method in ("kmeans", "ensemble"):
            self._fit_kmeans(X)
        logger.info("RegimeDetector fitted (%s) on %d observations", self.method, len(X))
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict regime labels for each row.

        Returns:
            DataFrame with columns: regime_id, regime_label, regime_color,
            and (if HMM) transition_prob_* columns.
        """
        X = self._prepare(features)

        if self.method == "hmm":
            result = self._predict_hmm(X, features.index)
        elif self.method == "kmeans":
            result = self._predict_kmeans(X, features.index)
        else:  # ensemble
            result = self._predict_ensemble(X, features.index)

        result["regime_label"] = result["regime_id"].map(self.regime_labels)
        result["regime_color"] = result["regime_label"].map(REGIME_COLORS)
        return result

    def regime_transition_matrix(self, labels: pd.Series) -> pd.DataFrame:
        """Compute empirical regime transition probability matrix."""
        regimes = sorted(labels.unique())
        n = len(regimes)
        matrix = np.zeros((n, n))
        label_arr = labels.values
        for i in range(len(label_arr) - 1):
            from_r = label_arr[i]
            to_r = label_arr[i + 1]
            matrix[from_r, to_r] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob_matrix = matrix / row_sums
        labels_str = [self.regime_labels.get(r, str(r)) for r in regimes]
        return pd.DataFrame(prob_matrix, index=labels_str, columns=labels_str)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self._scaler,
            "pca": self._pca,
            "hmm": self._hmm_model,
            "kmeans": self._kmeans_model,
            "method": self.method,
        }, path)
        logger.info("RegimeDetector saved to %s", path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self._scaler = bundle["scaler"]
        self._pca = bundle["pca"]
        self._hmm_model = bundle.get("hmm")
        self._kmeans_model = bundle.get("kmeans")
        self.method = bundle.get("method", self.method)
        logger.info("RegimeDetector loaded from %s", path)

    # ── Private ───────────────────────────────────────────────────────────

    def _prepare(self, features: pd.DataFrame) -> np.ndarray:
        """Select, scale, and PCA-transform the regime feature set."""
        cols = [c for c in REGIME_FEATURES if c in features.columns]
        if not cols:
            cols = features.select_dtypes(include=[np.number]).columns.tolist()
        X = features[cols].fillna(method="ffill").fillna(0.0).to_numpy()
        X_scaled = self._scaler.fit_transform(X) if not hasattr(self._scaler, "mean_") \
            else self._scaler.transform(X)
        if hasattr(self._pca, "components_"):
            return self._pca.transform(X_scaled)
        return self._pca.fit_transform(X_scaled)

    def _fit_hmm(self, X: np.ndarray) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("Install hmmlearn: pip install hmmlearn") from exc
        self._hmm_model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.cfg.get("hmm_covariance_type", "full"),
            n_iter=self.cfg.get("hmm_n_iter", 200),
            random_state=42,
        )
        self._hmm_model.fit(X)
        logger.info("HMM fitted — log-likelihood: %.2f", self._hmm_model.score(X))

    def _fit_kmeans(self, X: np.ndarray) -> None:
        from sklearn.cluster import KMeans
        self._kmeans_model = KMeans(
            n_clusters=self.cfg.get("kmeans_n_clusters", self.n_regimes),
            random_state=self.cfg.get("kmeans_random_state", 42),
            n_init=20,
        )
        self._kmeans_model.fit(X)

    def _predict_hmm(self, X: np.ndarray, index) -> pd.DataFrame:
        labels = self._hmm_model.predict(X)
        proba = self._hmm_model.predict_proba(X)
        result = pd.DataFrame({"regime_id": labels}, index=index)
        for i in range(proba.shape[1]):
            result[f"regime_prob_{i}"] = proba[:, i]
        return result

    def _predict_kmeans(self, X: np.ndarray, index) -> pd.DataFrame:
        labels = self._kmeans_model.predict(X)
        return pd.DataFrame({"regime_id": labels}, index=index)

    def _predict_ensemble(self, X: np.ndarray, index) -> pd.DataFrame:
        hmm_result = self._predict_hmm(X, index)
        km_result = self._predict_kmeans(X, index)
        # Simple majority vote (both equal weight); HMM wins on tie
        result = hmm_result.copy()
        disagree = hmm_result["regime_id"] != km_result["regime_id"]
        # When they disagree, use HMM (probabilistic model preferred)
        result.loc[disagree, "ensemble_conflict"] = 1
        result.loc[~disagree, "ensemble_conflict"] = 0
        return result
