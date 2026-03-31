"""
model.py — Financing Cost Forecaster multi-horizon LightGBM model.

Provides MultiHorizonForecaster, which trains one LGBMRegressor per
forecast horizon (1-day, 3-day, 5-day) using forward-shifted targets,
generates predictions, evaluates accuracy, and serialises/deserialises
model state.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class MultiHorizonForecaster:
    """
    Train and serve independent LightGBM regressors for each forecast horizon.

    One LGBMRegressor model is trained per horizon in ``config['model']['horizons']``
    (default: [1, 3, 5]). Each model is trained on the same feature matrix X
    but a different target series whose values are shifted ``h`` periods forward,
    creating a proper multi-horizon forecast without leakage.

    Parameters
    ----------
    config : dict
        Full pipeline config dict. The ``model`` sub-key is used for
        LightGBM hyperparameters and horizon configuration.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg: dict = config.get("model", {})
        self.horizons: list[int] = self.model_cfg.get("horizons", [1, 3, 5])
        self.models: dict[int, LGBMRegressor] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lgbm_params(self, horizon: int) -> dict[str, Any]:
        """
        Build LightGBM keyword arguments from config, injecting a
        horizon-specific random seed to diversify model variance.

        Parameters
        ----------
        horizon : int
            Forecast horizon in days (used to offset random seed).

        Returns
        -------
        dict[str, Any]
            Keyword arguments for LGBMRegressor constructor.
        """
        base_seed: int = self.model_cfg.get("random_state", 42)
        return {
            "n_estimators": self.model_cfg.get("n_estimators", 400),
            "max_depth": self.model_cfg.get("max_depth", 6),
            "learning_rate": self.model_cfg.get("learning_rate", 0.05),
            "num_leaves": self.model_cfg.get("num_leaves", 31),
            "min_child_samples": self.model_cfg.get("min_child_samples", 20),
            "subsample": self.model_cfg.get("subsample", 0.8),
            "colsample_bytree": self.model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": self.model_cfg.get("reg_alpha", 0.1),
            "reg_lambda": self.model_cfg.get("reg_lambda", 0.1),
            "random_state": base_seed + horizon,
            "verbose": -1,
        }

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_horizon_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> pd.Series:
        """
        Construct the forward-shifted target series for a given horizon.

        The target for horizon ``h`` is the value of ``target_col`` ``h``
        rows ahead in the DataFrame. Rows at the tail of the series will
        have NaN targets (because future values are unknown); these must
        be excluded from training.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``target_col`` sorted by date (and
            already grouped by instrument_type when called per-group).
        target_col : str
            Column name of the quantity being forecast (e.g. ``rate_bps``).
        horizon : int
            Number of days (rows) to shift the target forward.

        Returns
        -------
        pd.Series
            Forward-shifted target series, same index as df, NaN at tail.
        """
        return df[target_col].shift(-horizon)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_dict: dict[int, pd.Series],
    ) -> None:
        """
        Train one LGBMRegressor per horizon.

        For each horizon h, rows where y_dict[h] is NaN (the last h rows
        whose forward target is unknown) are dropped before training.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix aligned with y_dict values.
        y_dict : dict[int, pd.Series]
            Mapping of {horizon: target_series}. Built externally via
            repeated calls to ``_build_horizon_target``.
        """
        for h in self.horizons:
            if h not in y_dict:
                logger.warning("Horizon %d not in y_dict — skipping.", h)
                continue

            y = y_dict[h]
            mask = y.notna()
            X_train = X.loc[mask]
            y_train = y.loc[mask]

            if len(X_train) < 10:
                logger.warning(
                    "Horizon %d: only %d training samples — skipping.", h, len(X_train)
                )
                continue

            logger.info(
                "Training horizon=%d  samples=%d  features=%d",
                h, len(X_train), X_train.shape[1],
            )
            model = LGBMRegressor(**self._lgbm_params(h))
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[],
            )
            self.models[h] = model
            logger.info("Horizon=%d  best_iteration=%s", h, getattr(model, "best_iteration_", "N/A"))

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        """
        Generate forecasts for all trained horizons.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same schema as used during fit).

        Returns
        -------
        dict[int, np.ndarray]
            Mapping of {horizon: predictions_array}, e.g.
            {1: array([...]), 3: array([...]), 5: array([...])}.

        Raises
        ------
        RuntimeError
            If called before ``fit`` has been run.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        predictions: dict[int, np.ndarray] = {}
        for h, model in self.models.items():
            preds = model.predict(X)
            predictions[h] = preds
            logger.debug("Horizon=%d  predicted %d rows", h, len(preds))
        return predictions

    # ------------------------------------------------------------------
    # Predict to DataFrame
    # ------------------------------------------------------------------

    def predict_df(
        self,
        X: pd.DataFrame,
        base_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a forecast output DataFrame with all horizon predictions
        alongside the base data columns required for downstream processing.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix aligned with ``base_df`` rows.
        base_df : pd.DataFrame
            Source DataFrame containing ``date``, ``instrument_type``,
            ``rate_bps`` (current rate), and optionally ``budget_rate_bps``.

        Returns
        -------
        pd.DataFrame
            One row per (date, instrument_type) with columns:
            date, instrument_type, current_rate_bps, forecast_1d_bps,
            forecast_3d_bps, forecast_5d_bps, cost_vs_budget_bps, alert_flag.

        Notes
        -----
        ``alert_flag`` is initialised to False here; it is set to True by
        the alert layer after threshold evaluation.
        """
        raw_preds = self.predict(X)

        out = pd.DataFrame(index=X.index)
        if "date" in base_df.columns:
            out["date"] = base_df["date"].values[: len(X)]
        if "instrument_type" in base_df.columns:
            out["instrument_type"] = base_df["instrument_type"].values[: len(X)]
        if "rate_bps" in base_df.columns:
            out["current_rate_bps"] = base_df["rate_bps"].values[: len(X)]

        for h in sorted(raw_preds.keys()):
            out[f"forecast_{h}d_bps"] = raw_preds[h]

        # Ensure standard horizon columns exist even if model not trained
        for h in [1, 3, 5]:
            col = f"forecast_{h}d_bps"
            if col not in out.columns:
                out[col] = np.nan

        # Cost vs budget (using 5-day horizon as primary budget comparison)
        if "budget_rate_bps" in base_df.columns:
            out["cost_vs_budget_bps"] = (
                out["forecast_5d_bps"] - base_df["budget_rate_bps"].values[: len(X)]
            )
        else:
            out["cost_vs_budget_bps"] = np.nan

        out["alert_flag"] = False

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        y_dict: dict[int, pd.Series],
    ) -> dict[int, dict[str, float]]:
        """
        Compute MAE and RMSE per horizon on a held-out evaluation set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for the evaluation period.
        y_dict : dict[int, pd.Series]
            True target values per horizon (aligned with X).

        Returns
        -------
        dict[int, dict[str, float]]
            Nested dict: {horizon: {"mae": float, "rmse": float,
            "n_samples": int}}.
        """
        raw_preds = self.predict(X)
        metrics: dict[int, dict[str, float]] = {}

        for h, preds in raw_preds.items():
            if h not in y_dict:
                continue
            y_true = y_dict[h]
            mask = y_true.notna()
            y_true_clean = y_true.loc[mask].values
            y_pred_clean = preds[mask.values]

            if len(y_true_clean) == 0:
                metrics[h] = {"mae": float("nan"), "rmse": float("nan"), "n_samples": 0}
                continue

            mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
            rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
            metrics[h] = {"mae": mae, "rmse": rmse, "n_samples": int(mask.sum())}
            logger.info("Horizon=%d  MAE=%.4f  RMSE=%.4f", h, mae, rmse)

        return metrics

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[int, pd.DataFrame]:
        """
        Return a sorted feature importance DataFrame per horizon.

        Returns
        -------
        dict[int, pd.DataFrame]
            {horizon: DataFrame with columns ['feature', 'importance']}
            sorted descending by importance.
        """
        result: dict[int, pd.DataFrame] = {}
        for h, model in self.models.items():
            fi = pd.DataFrame(
                {
                    "feature": model.feature_name_,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False).reset_index(drop=True)
            result[h] = fi
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise all trained models and config to a pickle file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``models/forecaster.pkl``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": self.models,
            "horizons": self.horizons,
            "config": self.config,
            "_is_fitted": self._is_fitted,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved MultiHorizonForecaster to %s", path)

    def load(self, path: str) -> None:
        """
        Deserialise a previously saved forecaster from a pickle file.

        Parameters
        ----------
        path : str
            Source file path produced by ``save()``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.models = payload["models"]
        self.horizons = payload["horizons"]
        self.config = payload["config"]
        self._is_fitted = payload["_is_fitted"]
        logger.info("Loaded MultiHorizonForecaster from %s", path)
