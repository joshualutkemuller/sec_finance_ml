"""
Demand Forecaster — Multi-horizon LightGBM model for securities borrow demand.

Predicts borrow demand (shares / notional) per security for horizons 1d, 3d, 5d.
Uses asymmetric quantile loss because over-forecasting demand causes excess
inventory reservation, which is more costly than under-forecasting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


class DemandForecaster:
    """Multi-horizon LightGBM demand forecaster with asymmetric quantile loss."""

    def __init__(self, config: dict) -> None:
        self.config = config["demand_forecaster"]
        self.horizons: list[int] = config["features"]["demand_horizons"]
        self._models: dict[int, LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y_dict: dict[int, pd.Series]) -> None:
        """Train one LGBMRegressor per horizon.

        Args:
            X: Feature matrix (securities x features), index aligned with y_dict.
            y_dict: {horizon_days: target_series} mapping.
        """
        alpha = self.config.get("quantile_alpha", 0.6)
        for h in self.horizons:
            if h not in y_dict:
                logger.warning("No target for horizon %dd — skipping", h)
                continue
            y = y_dict[h].loc[X.index]
            valid_mask = y.notna()
            logger.info("Training demand model h=%dd: %d samples", h, valid_mask.sum())

            model = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                learning_rate=self.config["learning_rate"],
                num_leaves=self.config["num_leaves"],
                random_state=self.config["random_state"],
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X[valid_mask], y[valid_mask])
            self._models[h] = model
            logger.info("Demand model h=%dd trained", h)

    def predict(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        """Predict borrow demand for all horizons.

        Returns:
            {horizon_days: predicted_demand_array} mapping.
        """
        if not self._models:
            raise RuntimeError("DemandForecaster not trained — call fit() or load() first")

        result: dict[int, np.ndarray] = {}
        for h, model in self._models.items():
            preds = model.predict(X)
            preds = np.maximum(preds, 0.0)  # demand cannot be negative
            result[h] = preds
            logger.debug("Demand forecast h=%dd: mean=%.2f, max=%.2f",
                         h, preds.mean(), preds.max())
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances averaged across all horizon models."""
        if not self._models:
            raise RuntimeError("No models trained yet")
        frames = []
        for h, model in self._models.items():
            fi = pd.Series(
                model.feature_importances_,
                index=model.feature_name_,
                name=f"h{h}d",
            )
            frames.append(fi)
        return pd.concat(frames, axis=1).mean(axis=1).sort_values(ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._models, path)
        logger.info("DemandForecaster saved to %s", path)

    def load(self, path: str) -> None:
        self._models = joblib.load(path)
        logger.info("DemandForecaster loaded from %s", path)
