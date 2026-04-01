"""
Spread & Rate Forecaster — Dual LightGBM + LSTM model.

LightGBM: fast, interpretable, low-latency daily forecasts (1d/3d/5d/10d)
LSTM:     trend-aware, provides uncertainty bands via MC Dropout

Targets: SOFR, HY OAS, IG OAS
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


# ── LightGBM forecaster ───────────────────────────────────────────────────────

class LGBMSpreadForecaster:
    """Multi-target, multi-horizon LightGBM forecaster."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["spread_forecaster"]["lgbm"]
        self.horizons: list[int] = config["spread_forecaster"]["horizons"]
        self.targets: list[str] = config["spread_forecaster"]["targets"]
        # models[target][horizon] = LGBMRegressor
        self._models: dict[str, dict[int, LGBMRegressor]] = {}

    def fit(self, X: pd.DataFrame, raw: pd.DataFrame) -> None:
        """Train one LGBMRegressor per (target, horizon) combination."""
        for target in self.targets:
            if target not in raw.columns:
                logger.warning("Target '%s' not in data — skipping", target)
                continue
            self._models[target] = {}
            for h in self.horizons:
                y = raw[target].shift(-h).loc[X.index]
                valid = y.notna() & X.notna().all(axis=1)
                logger.info("Training LGBM target=%s h=%dd: %d samples", target, h, valid.sum())
                model = LGBMRegressor(
                    objective="quantile",
                    alpha=self.cfg.get("quantile_alpha", 0.70),
                    n_estimators=self.cfg["n_estimators"],
                    max_depth=self.cfg["max_depth"],
                    learning_rate=self.cfg["learning_rate"],
                    num_leaves=self.cfg["num_leaves"],
                    random_state=self.cfg["random_state"],
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(X[valid], y[valid])
                self._models[target][h] = model

    def predict(self, X: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
        """Return forecasts for all targets and horizons.

        Returns:
            {target: {horizon: array_of_forecasts}}
        """
        results: dict[str, dict[int, np.ndarray]] = {}
        for target, h_models in self._models.items():
            results[target] = {}
            for h, model in h_models.items():
                results[target][h] = model.predict(X)
        return results

    def feature_importance(self, target: str | None = None) -> pd.DataFrame:
        """Return feature importances averaged across horizons (for a given target)."""
        targets = [target] if target else list(self._models.keys())
        frames = []
        for t in targets:
            for h, model in self._models.get(t, {}).items():
                fi = pd.Series(
                    model.feature_importances_,
                    index=model.feature_name_,
                    name=f"{t}_h{h}d",
                )
                frames.append(fi)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).mean(axis=1).sort_values(ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._models, path)
        logger.info("LGBMSpreadForecaster saved to %s", path)

    def load(self, path: str) -> None:
        self._models = joblib.load(path)
        logger.info("LGBMSpreadForecaster loaded from %s", path)


# ── LSTM forecaster ───────────────────────────────────────────────────────────

class LSTMSpreadForecaster:
    """Sequence-to-sequence LSTM with MC Dropout for uncertainty quantification."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["spread_forecaster"]["lstm"]
        self.seq_len = self.cfg.get("seq_len", 60)
        self.targets = config["spread_forecaster"]["targets"]
        self.horizons = config["spread_forecaster"]["horizons"]
        self._model = None
        self._scaler = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, raw: pd.DataFrame) -> None:
        """Train LSTM on sequence data."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("Install PyTorch: pip install torch") from exc

        from sklearn.preprocessing import StandardScaler

        self._feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[self._feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_num)

        # Build sequences: predict all targets at all horizons from one LSTM
        # Output: [n_targets * n_horizons] per sequence
        target_data = [raw[t].values for t in self.targets if t in raw.columns]
        n_targets = len(target_data)
        n_horizons = len(self.horizons)

        sequences, labels = [], []
        max_horizon = max(self.horizons)
        for i in range(self.seq_len, len(X_scaled) - max_horizon):
            sequences.append(X_scaled[i - self.seq_len:i])
            row_labels = []
            for t_arr in target_data:
                for h in self.horizons:
                    row_labels.append(t_arr[i + h] if (i + h) < len(t_arr) else np.nan)
            labels.append(row_labels)

        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.float32)

        # Remove rows with NaN labels
        valid = ~np.isnan(y_seq).any(axis=1)
        X_seq, y_seq = X_seq[valid], y_seq[valid]

        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq),
        )
        loader = DataLoader(dataset, batch_size=self.cfg.get("batch_size", 64), shuffle=True)

        input_size = X_seq.shape[2]
        output_size = n_targets * n_horizons

        self._model = _LSTMNet(
            input_size=input_size,
            hidden_size=self.cfg.get("hidden_size", 128),
            num_layers=self.cfg.get("num_layers", 2),
            output_size=output_size,
            dropout=self.cfg.get("dropout", 0.2),
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.cfg.get("learning_rate", 1e-3),
        )
        criterion = nn.MSELoss()

        epochs = self.cfg.get("epochs", 50)
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self._model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM epoch %d/%d — loss: %.5f", epoch + 1, epochs,
                            total_loss / len(loader))

        logger.info("LSTM training complete")

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Predict with MC Dropout uncertainty bands.

        Returns:
            {target: {horizon: (mean, lower_90, upper_90)}}
        """
        import torch

        if self._model is None:
            raise RuntimeError("LSTMSpreadForecaster not trained")

        X_num = X[self._feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        X_scaled = self._scaler.transform(X_num)

        # Use last seq_len rows for inference
        if len(X_scaled) < self.seq_len:
            logger.warning("Not enough rows for LSTM inference (%d < %d)", len(X_scaled), self.seq_len)
            return {}

        x_seq = torch.from_numpy(X_scaled[-self.seq_len:][np.newaxis, :, :])  # [1, seq, feat]

        n_passes = self.cfg.get("mc_dropout_passes", 100)
        self._model.train()  # keep dropout active for MC sampling
        samples = []
        with torch.no_grad():
            for _ in range(n_passes):
                out = self._model(x_seq).numpy()[0]  # [n_targets * n_horizons]
                samples.append(out)
        self._model.eval()

        samples_arr = np.array(samples)  # [n_passes, n_targets * n_horizons]
        mean_out = samples_arr.mean(axis=0)
        lower_out = np.percentile(samples_arr, 5, axis=0)
        upper_out = np.percentile(samples_arr, 95, axis=0)

        results: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        valid_targets = [t for t in self.targets if t in X.columns]
        idx = 0
        for t in valid_targets:
            results[t] = {}
            for h in self.horizons:
                results[t][h] = (
                    np.array([mean_out[idx]]),
                    np.array([lower_out[idx]]),
                    np.array([upper_out[idx]]),
                )
                idx += 1
        return results

    def save(self, path: str) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict() if self._model else None,
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
            "config": self.cfg,
        }, path)
        logger.info("LSTMSpreadForecaster saved to %s", path)

    def load(self, path: str) -> None:
        import torch
        bundle = torch.load(path, weights_only=False)
        self._scaler = bundle["scaler"]
        self._feature_cols = bundle["feature_cols"]
        input_size = len(self._feature_cols)
        n_out = len(self.targets) * len(self.horizons)
        self._model = _LSTMNet(
            input_size=input_size,
            hidden_size=bundle["config"].get("hidden_size", 128),
            num_layers=bundle["config"].get("num_layers", 2),
            output_size=n_out,
            dropout=bundle["config"].get("dropout", 0.2),
        )
        if bundle["model_state"]:
            self._model.load_state_dict(bundle["model_state"])
        self._model.eval()
        logger.info("LSTMSpreadForecaster loaded from %s", path)


class _LSTMNet:
    """Simple LSTM wrapper (PyTorch nn.Module)."""

    def __new__(cls, input_size, hidden_size, num_layers, output_size, dropout):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)

        return Net()
