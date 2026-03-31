"""
model.py
--------
Anomaly detection models for securities lending borrow rates.

Three detector classes are provided:

IsolationForestDetector
    Wraps ``sklearn.ensemble.IsolationForest``.  Scores are normalised to
    [0, 1] so that higher values always indicate greater anomalousness.

LSTMAutoencoder  (``torch.nn.Module``)
    Encoder-decoder architecture built from stacked LSTM layers.  Learns to
    reconstruct normal rate sequences; high reconstruction error signals
    an anomaly.

LSTMDetector
    Training / scoring wrapper around :class:`LSTMAutoencoder`.  Handles
    sequence construction, mini-batch training, threshold fitting and model
    persistence.

CombinedAnomalyDetector
    Composes :class:`IsolationForestDetector` and :class:`LSTMDetector` into
    a single interface.  Produces a weighted-average combined score and a
    boolean ``is_anomaly`` flag.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Isolation Forest Detector
# ============================================================================


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector.

    Wraps ``sklearn.ensemble.IsolationForest`` and normalises raw decision
    function scores to the [0, 1] range where *1.0* is maximally anomalous.

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration.  Uses the ``isolation_forest`` section:

        - ``contamination``    — expected outlier fraction
        - ``n_estimators``     — number of isolation trees
        - ``random_state``     — reproducibility seed
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from sklearn.ensemble import IsolationForest  # local import

        if_cfg = config.get("isolation_forest", {})
        self.contamination: float = float(if_cfg.get("contamination", 0.05))
        self.n_estimators: int = int(if_cfg.get("n_estimators", 200))
        self.random_state: int = int(if_cfg.get("random_state", 42))
        self.model_path: str = if_cfg.get("model_path", "models/isolation_forest.pkl")

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._is_fitted: bool = False
        self._score_min: float = 0.0
        self._score_max: float = 1.0

        logger.debug(
            "IsolationForestDetector created — contamination=%.3f, n_estimators=%d",
            self.contamination,
            self.n_estimators,
        )

    def fit(self, X: np.ndarray) -> None:
        """Fit the Isolation Forest on the training feature matrix.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.
        """
        logger.info(
            "Fitting IsolationForest on %d samples × %d features",
            X.shape[0],
            X.shape[1],
        )
        self._model.fit(X)
        self._is_fitted = True

        # Compute normalisation range from training data
        raw_scores = self._model.decision_function(X)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        logger.info(
            "IsolationForest fit complete — raw score range [%.4f, %.4f]",
            self._score_min,
            self._score_max,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return normalised anomaly scores in [0, 1].

        The raw ``decision_function`` values are inverted (lower raw =
        more anomalous in sklearn convention) and min-max scaled using
        the range observed during training.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            1-D float array of shape ``(n_samples,)`` with values in [0, 1].
            Higher values indicate greater anomalousness.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before scoring.")

        raw: np.ndarray = self._model.decision_function(X)

        # Invert so that high score = anomalous (sklearn: low raw = anomalous)
        inverted = -raw

        # Min-max normalise using training range (clipped to avoid OOD extremes)
        score_range = self._score_max - self._score_min
        if score_range == 0:
            return np.zeros(len(X), dtype=np.float64)

        # Use inverted training extremes for normalisation
        inv_min = -self._score_max
        inv_max = -self._score_min
        scores = (inverted - inv_min) / (inv_max - inv_min)
        scores = np.clip(scores, 0.0, 1.0)

        logger.debug(
            "IF scoring — %d samples, score stats: mean=%.4f, max=%.4f",
            len(X),
            scores.mean(),
            scores.max(),
        )
        return scores.astype(np.float64)

    def save(self, path: str) -> None:
        """Persist the fitted model to a pickle file.

        Parameters
        ----------
        path : str
            File path for the pickle file.  Parent directories are created
            if they do not exist.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model": self._model,
                    "is_fitted": self._is_fitted,
                    "score_min": self._score_min,
                    "score_max": self._score_max,
                },
                fh,
            )
        logger.info("IsolationForestDetector saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file produced by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"IsolationForest model file not found: {path}")

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        self._model = state["model"]
        self._is_fitted = state["is_fitted"]
        self._score_min = state["score_min"]
        self._score_max = state["score_max"]
        logger.info("IsolationForestDetector loaded from %s", path)


# ============================================================================
# LSTM Autoencoder (nn.Module)
# ============================================================================


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for temporal anomaly detection.

    Architecture
    ~~~~~~~~~~~~
    **Encoder**: stacked LSTM that compresses a sequence of feature vectors
    into a fixed-size context (hidden + cell state of the last encoder layer).

    **Decoder**: stacked LSTM that receives the encoder context repeated
    ``sequence_length`` times and reconstructs the input sequence.

    Reconstruction error (MSE) is used as the anomaly score.

    Parameters
    ----------
    input_size : int
        Number of features per timestep.
    hidden_size : int
        Dimensionality of LSTM hidden states.
    num_layers : int
        Number of stacked LSTM layers in both the encoder and decoder.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder: processes the input sequence
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Decoder: reconstructs the sequence from the encoder context
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Output projection: maps hidden_size back to input_size
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input sequence and reconstruct it.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of the same shape as ``x``.
        """
        batch_size, seq_len, _ = x.size()

        # Encode — we keep the final hidden/cell states
        _, (hidden, cell) = self.encoder(x)

        # Expand encoder context to feed into decoder at each timestep
        # hidden[-1] has shape (batch_size, hidden_size)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))

        # Project back to input feature space
        reconstruction = self.output_layer(decoder_out)
        return reconstruction


# ============================================================================
# LSTM Detector
# ============================================================================


class LSTMDetector:
    """Training / scoring wrapper around :class:`LSTMAutoencoder`.

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration.  Uses the ``lstm`` section:

        - ``hidden_size``           — LSTM hidden dimension
        - ``num_layers``            — stacked LSTM layers
        - ``epochs``                — training epochs
        - ``batch_size``            — mini-batch size
        - ``learning_rate``         — Adam learning rate
        - ``sequence_length``       — look-back window
        - ``threshold_percentile``  — percentile of training errors used as
                                      anomaly threshold
        - ``model_path``            — path to save/load model weights
        - ``threshold_path``        — path to save/load the scalar threshold
    """

    def __init__(self, config: dict[str, Any]) -> None:
        lstm_cfg = config.get("lstm", {})

        self.hidden_size: int = int(lstm_cfg.get("hidden_size", 64))
        self.num_layers: int = int(lstm_cfg.get("num_layers", 2))
        self.epochs: int = int(lstm_cfg.get("epochs", 50))
        self.batch_size: int = int(lstm_cfg.get("batch_size", 32))
        self.learning_rate: float = float(lstm_cfg.get("learning_rate", 0.001))
        self.sequence_length: int = int(lstm_cfg.get("sequence_length", 20))
        self.threshold_percentile: float = float(
            lstm_cfg.get("threshold_percentile", 95)
        )
        self.model_path: str = lstm_cfg.get("model_path", "models/lstm_autoencoder.pt")
        self.threshold_path: str = lstm_cfg.get(
            "threshold_path", "models/lstm_threshold.npy"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: LSTMAutoencoder | None = None
        self._threshold: float | None = None
        self._train_error_min: float = 0.0
        self._train_error_max: float = 1.0

        logger.debug(
            "LSTMDetector initialised — hidden_size=%d, seq_len=%d, device=%s",
            self.hidden_size,
            self.sequence_length,
            self.device,
        )

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Build sliding-window sequences from a feature matrix.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            3-D float array of shape
            ``(n_samples - sequence_length + 1, sequence_length, n_features)``.

        Raises
        ------
        ValueError
            If ``X`` has fewer rows than ``sequence_length``.
        """
        n_samples, n_features = X.shape
        if n_samples < self.sequence_length:
            raise ValueError(
                f"Input has only {n_samples} rows but sequence_length is "
                f"{self.sequence_length}.  Provide at least {self.sequence_length} samples."
            )

        n_sequences = n_samples - self.sequence_length + 1
        sequences = np.zeros(
            (n_sequences, self.sequence_length, n_features), dtype=np.float32
        )

        for i in range(n_sequences):
            sequences[i] = X[i : i + self.sequence_length]

        logger.debug(
            "Built %d sequences of shape (%d, %d)",
            n_sequences,
            self.sequence_length,
            n_features,
        )
        return sequences

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """Train the LSTM Autoencoder on the provided feature matrix.

        After training, the anomaly threshold is computed as the
        ``threshold_percentile``-th percentile of reconstruction errors on
        the training set.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.
        """
        sequences = self._build_sequences(X)
        n_features = X.shape[1]

        self._model = LSTMAutoencoder(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        tensor_seqs = torch.from_numpy(sequences)
        dataset = TensorDataset(tensor_seqs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(
            "Training LSTMAutoencoder — %d sequences, %d epochs, device=%s",
            len(sequences),
            self.epochs,
            self.device,
        )

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction = self._model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch [%d/%d] — avg MSE loss: %.6f", epoch, self.epochs, avg_loss)

        # Compute threshold on training data
        train_errors = self._compute_reconstruction_errors(sequences)
        self._threshold = float(np.percentile(train_errors, self.threshold_percentile))
        self._train_error_min = float(train_errors.min())
        self._train_error_max = float(train_errors.max())

        logger.info(
            "LSTM training complete — threshold (p%.0f) = %.6f  "
            "[min=%.6f, max=%.6f]",
            self.threshold_percentile,
            self._threshold,
            self._train_error_min,
            self._train_error_max,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute normalised reconstruction-error scores in [0, 1].

        Scores are normalised using the min and max reconstruction errors
        observed during training.  Observations with fewer preceding
        rows than ``sequence_length`` receive the score of the first
        valid sequence (padding by repetition of the earliest score).

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            1-D float array of shape ``(n_samples,)`` with values in [0, 1].
            Higher values indicate greater anomalousness.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called first.
        """
        if self._model is None or self._threshold is None:
            raise RuntimeError("LSTMDetector must be fitted before scoring.")

        sequences = self._build_sequences(X)
        errors = self._compute_reconstruction_errors(sequences)

        # Normalise to [0, 1] using training range
        error_range = self._train_error_max - self._train_error_min
        if error_range > 0:
            scores = (errors - self._train_error_min) / error_range
        else:
            scores = np.zeros_like(errors)

        scores = np.clip(scores, 0.0, 1.0)

        # Pad the beginning (warm-up) with the first valid score
        n_pad = len(X) - len(scores)
        if n_pad > 0:
            pad = np.full(n_pad, scores[0], dtype=np.float64)
            scores = np.concatenate([pad, scores])

        logger.debug(
            "LSTM scoring — %d samples, score stats: mean=%.4f, max=%.4f",
            len(X),
            scores.mean(),
            scores.max(),
        )
        return scores.astype(np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model state dict and metadata to disk.

        Parameters
        ----------
        path : str
            Path for the PyTorch state dict (``.pt`` file).  The threshold
            is saved to ``self.threshold_path``.
        """
        if self._model is None:
            raise RuntimeError("No model to save — call fit() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "state_dict": self._model.state_dict(),
            "input_size": self._model.input_size,
            "hidden_size": self._model.hidden_size,
            "num_layers": self._model.num_layers,
            "train_error_min": self._train_error_min,
            "train_error_max": self._train_error_max,
        }
        torch.save(checkpoint, path)
        logger.info("LSTMDetector model saved to %s", path)

        # Save threshold separately as a numpy scalar
        threshold_path = self.threshold_path
        os.makedirs(
            os.path.dirname(threshold_path) if os.path.dirname(threshold_path) else ".",
            exist_ok=True,
        )
        np.save(threshold_path, np.array([self._threshold]))
        logger.info("LSTM threshold saved to %s", threshold_path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str
            Path to the ``.pt`` checkpoint file produced by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If the checkpoint or threshold files are not found.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LSTM model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self._model = LSTMAutoencoder(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        ).to(self.device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

        self._train_error_min = checkpoint.get("train_error_min", 0.0)
        self._train_error_max = checkpoint.get("train_error_max", 1.0)

        threshold_path = self.threshold_path
        if os.path.isfile(threshold_path):
            self._threshold = float(np.load(threshold_path)[0])
            logger.info("LSTM threshold loaded: %.6f", self._threshold)
        else:
            logger.warning(
                "Threshold file not found at %s — default 0.5 used", threshold_path
            )
            self._threshold = 0.5

        logger.info("LSTMDetector loaded from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Compute per-sequence mean squared reconstruction error.

        Parameters
        ----------
        sequences : np.ndarray
            3-D array of shape ``(n_seqs, seq_len, n_features)``.

        Returns
        -------
        np.ndarray
            1-D float64 array of MSE values of shape ``(n_seqs,)``.
        """
        assert self._model is not None

        self._model.eval()
        tensor_seqs = torch.from_numpy(sequences).to(self.device)
        all_errors: list[float] = []

        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = tensor_seqs[i : i + self.batch_size]
                reconstruction = self._model(batch)
                # MSE per sequence: mean over time steps and features
                mse = ((batch - reconstruction) ** 2).mean(dim=(1, 2))
                all_errors.extend(mse.cpu().numpy().tolist())

        return np.array(all_errors, dtype=np.float64)


# ============================================================================
# Combined Anomaly Detector
# ============================================================================


class CombinedAnomalyDetector:
    """Ensemble anomaly detector combining Isolation Forest and LSTM scores.

    The combined score is a weighted average:

        combined = α × if_score + (1 − α) × lstm_score

    where α is controlled by ``config['isolation_forest']['if_weight']``
    (default 0.5).

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        if_cfg = config.get("isolation_forest", {})
        alerts_cfg = config.get("alerts", {})

        self.if_weight: float = float(if_cfg.get("if_weight", 0.5))
        self.anomaly_threshold: float = float(
            alerts_cfg.get("anomaly_score_threshold", 0.70)
        )

        self.if_detector = IsolationForestDetector(config)
        self.lstm_detector = LSTMDetector(config)

        logger.debug(
            "CombinedAnomalyDetector created — if_weight=%.2f, threshold=%.2f",
            self.if_weight,
            self.anomaly_threshold,
        )

    def fit(self, X: np.ndarray) -> None:
        """Fit both the Isolation Forest and LSTM Autoencoder on training data.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.
        """
        logger.info("Fitting CombinedAnomalyDetector on %d samples", len(X))
        self.if_detector.fit(X)
        self.lstm_detector.fit(X)
        logger.info("CombinedAnomalyDetector fit complete")

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Score observations and return a results DataFrame.

        Produces per-sample scores from both models and combines them into a
        final anomaly score and boolean flag.

        Parameters
        ----------
        X : np.ndarray
            2-D float array of shape ``(n_samples, n_features)``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - ``if_score``        — Isolation Forest anomaly score [0, 1]
            - ``lstm_score``      — LSTM reconstruction-error score [0, 1]
            - ``combined_score``  — Weighted average of the two scores [0, 1]
            - ``is_anomaly``      — True where ``combined_score ≥ threshold``
        """
        logger.info("Scoring %d observations with CombinedAnomalyDetector", len(X))

        if_scores = self.if_detector.score(X)
        lstm_scores = self.lstm_detector.score(X)

        # Weighted average
        combined_scores = (
            self.if_weight * if_scores
            + (1.0 - self.if_weight) * lstm_scores
        )

        is_anomaly = combined_scores >= self.anomaly_threshold

        results = pd.DataFrame(
            {
                "if_score": if_scores,
                "lstm_score": lstm_scores,
                "combined_score": combined_scores,
                "is_anomaly": is_anomaly,
            }
        )

        n_anomalies = int(is_anomaly.sum())
        logger.info(
            "Scoring complete — %d anomalies detected (%.1f%% of %d observations)",
            n_anomalies,
            100.0 * n_anomalies / max(len(X), 1),
            len(X),
        )
        return results

    def save_models(self) -> None:
        """Persist both sub-models to their configured paths."""
        self.if_detector.save(self.if_detector.model_path)
        self.lstm_detector.save(self.lstm_detector.model_path)

    def load_models(self) -> None:
        """Load both sub-models from their configured paths."""
        self.if_detector.load(self.if_detector.model_path)
        self.lstm_detector.load(self.lstm_detector.model_path)
