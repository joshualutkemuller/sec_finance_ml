"""
Audit Logger — Immutable prediction audit trail for SR 11-7 compliance.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Every model prediction is logged with:
  - run_id, model_id, model_version
  - timestamp (UTC)
  - feature hash (SHA-256 of feature vector for immutability)
  - prediction value(s)
  - input record identifier

Logs are append-only JSONL files, one per model per day.
Retention: 7 years (2,555 days) per SR 11-7.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AuditLogger:
    """Write immutable prediction audit entries for a single model."""

    def __init__(self, model_id: str, model_version: str, config: dict) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.log_dir = Path(config["audit"].get("log_dir", "logs/audit"))
        self.hash_features = config["audit"].get("hash_features", True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        run_id: str,
        record_id: str,
        features: pd.Series | dict,
        prediction: float | list,
        metadata: dict | None = None,
    ) -> None:
        """
        Write a single prediction audit entry.

        Args:
            run_id: Pipeline run UUID.
            record_id: Identifier for the input record (ticker, date, etc.).
            features: Feature vector for this prediction.
            prediction: Model output (float or list for multi-output).
            metadata: Optional extra fields (threshold used, alert fired, etc.).
        """
        entry = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "run_id": run_id,
            "record_id": str(record_id),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction if isinstance(prediction, (int, float)) else list(prediction),
            "feature_hash": self._hash_features(features) if self.hash_features else None,
        }
        if metadata:
            entry["metadata"] = metadata

        log_path = self._log_path()
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_batch(
        self,
        run_id: str,
        records: pd.DataFrame,
        predictions: np.ndarray,
        id_col: str = "ticker",
    ) -> None:
        """Log a full batch of predictions efficiently."""
        log_path = self._log_path()
        lines = []
        for i, (_, row) in enumerate(records.iterrows()):
            entry = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "run_id": run_id,
                "record_id": str(row.get(id_col, i)),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prediction": float(predictions[i]) if predictions.ndim == 1 else predictions[i].tolist(),
                "feature_hash": self._hash_features(row) if self.hash_features else None,
            }
            lines.append(json.dumps(entry))

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

        logger.debug("Audit: logged %d predictions for %s v%s",
                     len(records), self.model_id, self.model_version)

    def query(self, date_str: str) -> pd.DataFrame:
        """Load audit log for a specific date (YYYY-MM-DD)."""
        log_path = self.log_dir / self.model_id / f"{date_str}.jsonl"
        if not log_path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)

    def _log_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        model_dir = self.log_dir / self.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{date_str}.jsonl"

    @staticmethod
    def _hash_features(features: pd.Series | dict) -> str:
        """SHA-256 hash of feature values for immutable audit reference."""
        if isinstance(features, pd.Series):
            data = features.to_dict()
        else:
            data = dict(features)
        canonical = json.dumps(
            {k: round(float(v), 8) if isinstance(v, (int, float, np.floating)) else str(v)
             for k, v in sorted(data.items())},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
