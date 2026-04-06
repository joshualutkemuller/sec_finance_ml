"""
Online RRCF Anomaly Detector — Robust Random Cut Forest for streaming borrow rates.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

RRCF is an online anomaly detection algorithm suited for streaming financial data:
  - O(1) update per tick (no batch retraining window)
  - Handles concept drift — older observations age out of the forest
  - Collusive Displacement (CoDisp) score: higher = more anomalous
  - Memory bounded by fixed forest size
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class OnlineRRCFDetector:
    """
    Per-ticker online RRCF detector with shingle (sliding window) input.

    Each ticker maintains its own RRCF forest and running CoDisp statistics.
    All state is kept in-memory; the Redis sink in stream_job.py handles
    persistence across restarts.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["rrcf"]
        self.num_trees = self.cfg.get("num_trees", 200)
        self.shingle_size = self.cfg.get("shingle_size", 4)
        self.tree_size = self.cfg.get("tree_size", 256)
        self.threshold_sigma = self.cfg.get("anomaly_score_threshold_sigma", 3.0)
        self.ewm_alpha = self.cfg.get("ewm_alpha", 0.02)

        # Per-ticker state
        self._forests: dict[str, list] = {}
        self._shingles: dict[str, deque] = {}
        self._codisp_ewm_mean: dict[str, float] = {}
        self._codisp_ewm_var: dict[str, float] = {}
        self._n_points: dict[str, int] = {}

    def update(self, ticker: str, rate: float) -> dict:
        """
        Update the RRCF forest for a ticker with a new borrow rate observation.

        Returns:
            Dict with keys: ticker, rate, codisp_score, is_anomaly,
            ewm_mean, ewm_std, sigma_multiple
        """
        self._ensure_initialized(ticker)

        shingle = self._shingles[ticker]
        shingle.append(rate)
        if len(shingle) < self.shingle_size:
            return {"ticker": ticker, "rate": rate, "codisp_score": 0.0,
                    "is_anomaly": False, "ewm_mean": 0.0, "ewm_std": 0.01, "sigma_multiple": 0.0}

        point = np.array(list(shingle), dtype=np.float64)
        n = self._n_points[ticker]

        try:
            import rrcf
        except ImportError as exc:
            raise ImportError("Install rrcf: pip install rrcf") from exc

        avg_codisp = 0.0
        for tree in self._forests[ticker]:
            # Remove oldest point if tree is at capacity
            if len(tree.leaves) > self.tree_size:
                oldest_key = min(tree.leaves.keys())
                tree.forget_point(oldest_key)
            # Insert new point
            tree.insert_point(point, index=n)
            avg_codisp += tree.codisp(n) / self.num_trees

        self._n_points[ticker] = n + 1

        # Update EWM statistics for adaptive thresholding
        alpha = self.ewm_alpha
        ewm_mean = self._codisp_ewm_mean[ticker]
        ewm_var = self._codisp_ewm_var[ticker]

        # Welford-style EWM update
        diff = avg_codisp - ewm_mean
        new_mean = ewm_mean + alpha * diff
        new_var = (1 - alpha) * (ewm_var + alpha * diff ** 2)

        self._codisp_ewm_mean[ticker] = new_mean
        self._codisp_ewm_var[ticker] = new_var

        ewm_std = max(np.sqrt(new_var), 1e-6)
        sigma_multiple = (avg_codisp - new_mean) / ewm_std
        is_anomaly = sigma_multiple > self.threshold_sigma

        result = {
            "ticker": ticker,
            "rate": rate,
            "codisp_score": round(avg_codisp, 6),
            "is_anomaly": is_anomaly,
            "ewm_mean": round(new_mean, 6),
            "ewm_std": round(ewm_std, 6),
            "sigma_multiple": round(sigma_multiple, 3),
        }

        if is_anomaly:
            logger.warning(
                "ANOMALY | %s | rate=%.5f | codisp=%.3f | sigma=%.2f",
                ticker, rate, avg_codisp, sigma_multiple,
            )

        return result

    def _ensure_initialized(self, ticker: str) -> None:
        if ticker not in self._forests:
            import rrcf
            self._forests[ticker] = [rrcf.RCTree() for _ in range(self.num_trees)]
            self._shingles[ticker] = deque(maxlen=self.shingle_size)
            self._codisp_ewm_mean[ticker] = 0.0
            self._codisp_ewm_var[ticker] = 0.01
            self._n_points[ticker] = 0

    def state_summary(self) -> dict:
        return {
            "n_tickers_tracked": len(self._forests),
            "tickers": list(self._forests.keys()),
            "avg_codisp_per_ticker": {
                t: round(self._codisp_ewm_mean[t], 4)
                for t in self._codisp_ewm_mean
            },
        }
