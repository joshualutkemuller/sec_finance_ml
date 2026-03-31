"""
model.py — CollateralQualityScorer and CollateralOptimizer

CollateralQualityScorer
    LightGBM-based regressor that predicts a quality score (0–1) for each
    collateral item based on engineered features.

CollateralOptimizer
    CVXPY-based convex optimiser that allocates collateral weights to
    minimise financing cost subject to eligibility, concentration, haircut,
    and quality-floor constraints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cvxpy as cp
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)


# ===========================================================================
# Quality Scorer
# ===========================================================================


class CollateralQualityScorer:
    """LightGBM regressor that scores collateral quality.

    The model is trained on historical labelled data where the target ``y``
    represents a quality label (e.g., 1 = performed well / not defaulted,
    0 = resulted in margin call or was rejected).  At inference time it
    predicts a continuous score in [0, 1] for each collateral item.

    Parameters
    ----------
    config : dict
        The ``model`` sub-section of ``config.yaml``.  Must contain:
        ``n_estimators``, ``max_depth``, ``learning_rate``,
        ``random_state``, ``quality_score_threshold``.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._model: Optional[LGBMRegressor] = None
        self._feature_cols: List[str] = config.get("feature_cols", FEATURE_COLS)
        self._threshold: float = config.get("quality_score_threshold", 0.55)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LightGBM quality scorer.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix; must contain all columns in
            ``self._feature_cols``.
        y : pd.Series
            Target quality labels in [0, 1].

        Raises
        ------
        ValueError
            If required feature columns are missing from ``X``.
        """
        missing = set(self._feature_cols) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing feature columns in training data: {sorted(missing)}"
            )

        X_train = X[self._feature_cols].copy()
        logger.info(
            "Training CollateralQualityScorer on %d samples, %d features",
            len(X_train),
            len(self._feature_cols),
        )

        self._model = LGBMRegressor(
            n_estimators=self._cfg.get("n_estimators", 300),
            max_depth=self._cfg.get("max_depth", 6),
            learning_rate=self._cfg.get("learning_rate", 0.05),
            random_state=self._cfg.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(X_train, y)
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict quality scores for each collateral item.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns used during training.

        Returns
        -------
        np.ndarray
            Quality scores clipped to [0, 1]; shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not trained. Call fit() or load() before score()."
            )

        X_infer = X[self._feature_cols].copy()
        raw_scores = self._model.predict(X_infer)
        scores = np.clip(raw_scores, 0.0, 1.0)
        logger.info(
            "Quality scoring complete; mean=%.4f, min=%.4f, max=%.4f",
            scores.mean(),
            scores.min(),
            scores.max(),
        )
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the trained model to disk with joblib.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"models/scorer.joblib"``).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Cannot save: model has not been trained.")

        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self._model, "feature_cols": self._feature_cols}, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously serialised model from disk.

        Parameters
        ----------
        path : str
            Source file path produced by :meth:`save`.
        """
        payload = joblib.load(path)
        self._model = payload["model"]
        self._feature_cols = payload.get("feature_cols", FEATURE_COLS)
        logger.info("Model loaded from %s", path)


# ===========================================================================
# CVXPY Optimizer
# ===========================================================================


class CollateralOptimizer:
    """CVXPY-based convex optimiser for collateral allocation.

    Formulates the following programme::

        minimise   cost_weight * (c @ x) - quality_weight * (q @ x)
        subject to
            sum(x) == 1                          (fully invested)
            x >= 0                               (long-only)
            x[ineligible] == 0                   (hard eligibility filter)
            x[asset_type==k].sum() <= max_conc   (concentration per type)
            H @ x <= max_haircut                 (blended haircut ceiling)
            q @ x >= min_quality_score           (quality floor)

    where:
        ``c``  = cost_of_carry vector (from feature engineering)
        ``q``  = quality_scores vector (from CollateralQualityScorer)
        ``H``  = haircut_pct vector
        ``x``  = allocation weight decision variable

    Parameters
    ----------
    config : dict
        The ``optimization`` sub-section of ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._max_conc: float = config.get("max_concentration_pct", 0.30)
        self._min_quality: float = config.get("min_quality_score", 0.50)
        self._max_haircut: float = config.get("max_haircut", 0.15)
        self._solver: str = config.get("solver", "ECOS")
        self._cost_weight: float = config.get("cost_weight", 1.0)
        self._quality_weight: float = config.get("quality_weight", 0.5)

    # ------------------------------------------------------------------
    # Core optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        collateral_df: pd.DataFrame,
        quality_scores: np.ndarray,
        required_value: float,
    ) -> pd.DataFrame:
        """Run the CVXPY optimisation and annotate ``collateral_df``.

        Parameters
        ----------
        collateral_df : pd.DataFrame
            Feature-engineered collateral data.  Must contain columns:
            ``eligible``, ``asset_type``, ``haircut_pct``,
            ``cost_of_carry``, ``market_value``.
        quality_scores : np.ndarray
            Quality scores from :class:`CollateralQualityScorer`,
            shape ``(n,)``.
        required_value : float
            Total notional value that must be allocated (used for
            logging; the constraint is normalised to weights).

        Returns
        -------
        pd.DataFrame
            ``collateral_df`` with three new columns appended:
            ``allocation_weight`` (float), ``recommended`` (bool),
            ``optimization_cost`` (float).

        Notes
        -----
        If the problem is infeasible or the solver raises an error the
        method logs a CRITICAL warning and returns the input DataFrame
        with all weights set to 0 and ``recommended=False``.
        """
        df = collateral_df.copy()
        n = len(df)

        # Build parameter vectors
        eligible_mask = df["eligible"].astype(bool).values
        haircut = df["haircut_pct"].fillna(0.0).values
        if haircut.max() > 1.0:
            haircut = haircut / 100.0

        cost_vec = df.get(
            "cost_of_carry", pd.Series(0.0, index=df.index)
        ).fillna(0.0).values

        q = quality_scores.copy()

        # ------------------------------------------------------------------
        # Decision variable
        # ------------------------------------------------------------------
        x = cp.Variable(n, nonneg=True)

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------
        objective = cp.Minimize(
            self._cost_weight * (cost_vec @ x) - self._quality_weight * (q @ x)
        )

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------
        constraints: List[Any] = []

        # 1. Fully invested
        constraints.append(cp.sum(x) == 1.0)

        # 2. Eligibility hard filter
        ineligible_idx = np.where(~eligible_mask)[0]
        if len(ineligible_idx):
            constraints.append(x[ineligible_idx] == 0.0)

        # 3. Concentration per asset type
        asset_types = df["asset_type"].unique()
        for at in asset_types:
            idx = np.where(df["asset_type"].values == at)[0]
            constraints.append(cp.sum(x[idx]) <= self._max_conc)

        # 4. Blended haircut ceiling
        constraints.append(haircut @ x <= self._max_haircut)

        # 5. Quality floor
        constraints.append(q @ x >= self._min_quality)

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        problem = cp.Problem(objective, constraints)
        solver_map = {"ECOS": cp.ECOS, "SCS": cp.SCS}
        solver = solver_map.get(self._solver.upper(), cp.ECOS)

        try:
            problem.solve(solver=solver, warm_start=True)
        except cp.SolverError as exc:
            logger.critical("CVXPY solver error: %s", exc)
            return self._infeasible_result(df)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.critical(
                "Optimisation infeasible or unbounded; status=%s", problem.status
            )
            return self._infeasible_result(df)

        # ------------------------------------------------------------------
        # Annotate results
        # ------------------------------------------------------------------
        weights = np.maximum(x.value, 0.0)  # clip tiny negatives
        df["allocation_weight"] = weights
        df["recommended"] = weights > 1e-4
        df["optimization_cost"] = cost_vec * weights

        logger.info(
            "Optimisation succeeded; status=%s, recommended=%d/%d, "
            "required_value=%.2f",
            problem.status,
            df["recommended"].sum(),
            n,
            required_value,
        )
        return df

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def generate_recommendations(
        self, optimized_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Build a list of recommendation dicts from the optimised DataFrame.

        For each row where ``recommended == True`` a dict is produced
        containing the fields required by the PowerBI push dataset schema
        plus ``cost_savings_bps`` (computed relative to a hypothetical
        equal-weight baseline).

        Parameters
        ----------
        optimized_df : pd.DataFrame
            Output of :meth:`optimize`; must contain ``recommended``,
            ``allocation_weight``, ``optimization_cost``, ``quality_score``
            (if present), ``haircut_pct``, ``asset_type``, ``collateral_id``.

        Returns
        -------
        list of dict
            Each dict has keys:
            ``collateral_id``, ``asset_type``, ``haircut``,
            ``quality_score``, ``allocation_weight``,
            ``optimization_cost``, ``cost_savings_bps``,
            ``recommended``, ``timestamp``.
        """
        now_utc = datetime.now(tz=timezone.utc).isoformat()
        n = len(optimized_df)
        baseline_weight = 1.0 / n if n > 0 else 0.0

        # Baseline cost (equal-weight allocation)
        cost_col = optimized_df.get(
            "cost_of_carry", pd.Series(0.0, index=optimized_df.index)
        ).fillna(0.0)
        baseline_cost = (cost_col * baseline_weight).sum()

        recommended_rows = optimized_df[optimized_df["recommended"]].copy()
        recommendations: List[Dict[str, Any]] = []

        for _, row in recommended_rows.iterrows():
            opt_cost = float(row.get("optimization_cost", 0.0))
            savings_bps = (baseline_cost - opt_cost) * 10_000.0

            rec: Dict[str, Any] = {
                "collateral_id": str(row.get("collateral_id", "UNKNOWN")),
                "asset_type": str(row.get("asset_type", "UNKNOWN")),
                "haircut": float(row.get("haircut_pct", 0.0)),
                "quality_score": float(row.get("quality_score", 0.0)),
                "allocation_weight": float(row.get("allocation_weight", 0.0)),
                "optimization_cost": opt_cost,
                "cost_savings_bps": round(savings_bps, 4),
                "recommended": True,
                "timestamp": now_utc,
            }
            recommendations.append(rec)

        logger.info(
            "Generated %d recommendations from %d eligible items",
            len(recommendations),
            n,
        )
        return recommendations

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _infeasible_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame annotated with zero weights (infeasible run)."""
        df = df.copy()
        df["allocation_weight"] = 0.0
        df["recommended"] = False
        df["optimization_cost"] = 0.0
        return df
