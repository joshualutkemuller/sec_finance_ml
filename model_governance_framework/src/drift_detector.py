"""
Drift Detector — Data drift (PSI) and concept drift detection for all platform models.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Uses Evidently AI for feature distribution drift and custom rolling error
rate monitoring for concept drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection result for a single model snapshot."""
    model_id: str
    run_date: str
    n_features_tested: int
    n_features_drifted: int
    max_psi: float
    mean_psi: float
    drift_status: str          # "OK" | "WARNING" | "CRITICAL"
    drifted_features: list[str]
    concept_drift_detected: bool
    error_rate_delta: float    # relative change in error rate
    details: dict = field(default_factory=dict)


class DriftDetector:
    """
    Detects feature distribution drift (PSI) and concept drift (performance degradation)
    for all models in the FSIP platform.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["drift"]
        self.psi_warn = self.cfg.get("psi_warning_threshold", 0.10)
        self.psi_crit = self.cfg.get("psi_critical_threshold", 0.25)
        self.concept_window = self.cfg.get("concept_drift_window_days", 30)
        self.error_threshold = self.cfg.get("concept_drift_error_threshold", 0.15)
        self.min_samples = self.cfg.get("min_samples_for_drift_test", 100)

    def detect_data_drift(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DriftReport:
        """
        Compute PSI for each feature between reference (training) and current distributions.

        PSI interpretation:
          < 0.10  : No significant change (OK)
          0.10–0.25 : Some change, monitor closely (WARNING)
          > 0.25  : Major change, likely requires retraining (CRITICAL)
        """
        from datetime import date
        run_date = str(date.today())

        if len(current_data) < self.min_samples:
            logger.warning(
                "Insufficient current data for %s (%d < %d samples)",
                model_id, len(current_data), self.min_samples,
            )
            return DriftReport(
                model_id=model_id, run_date=run_date,
                n_features_tested=0, n_features_drifted=0,
                max_psi=0.0, mean_psi=0.0, drift_status="INSUFFICIENT_DATA",
                drifted_features=[], concept_drift_detected=False, error_rate_delta=0.0,
            )

        num_cols = reference_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in num_cols if c in current_data.columns]

        psi_scores: dict[str, float] = {}
        for col in common_cols:
            psi = self._compute_psi(
                reference_data[col].dropna().values,
                current_data[col].dropna().values,
            )
            psi_scores[col] = psi

        if not psi_scores:
            return DriftReport(
                model_id=model_id, run_date=run_date,
                n_features_tested=0, n_features_drifted=0,
                max_psi=0.0, mean_psi=0.0, drift_status="NO_FEATURES",
                drifted_features=[], concept_drift_detected=False, error_rate_delta=0.0,
            )

        max_psi = max(psi_scores.values())
        mean_psi = np.mean(list(psi_scores.values()))
        drifted = [col for col, psi in psi_scores.items() if psi > self.psi_warn]

        if max_psi >= self.psi_crit:
            status = "CRITICAL"
        elif max_psi >= self.psi_warn:
            status = "WARNING"
        else:
            status = "OK"

        report = DriftReport(
            model_id=model_id,
            run_date=run_date,
            n_features_tested=len(psi_scores),
            n_features_drifted=len(drifted),
            max_psi=round(max_psi, 4),
            mean_psi=round(mean_psi, 4),
            drift_status=status,
            drifted_features=sorted(drifted, key=lambda c: psi_scores[c], reverse=True)[:10],
            concept_drift_detected=False,
            error_rate_delta=0.0,
            details={col: round(psi, 4) for col, psi in
                     sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)[:20]},
        )

        if status != "OK":
            logger.warning(
                "DRIFT ALERT | %s | status=%s | max_psi=%.3f | drifted=%d/%d features",
                model_id, status, max_psi, len(drifted), len(psi_scores),
            )

        return report

    def detect_concept_drift(
        self,
        model_id: str,
        recent_errors: pd.Series,
        baseline_error: float,
    ) -> tuple[bool, float]:
        """
        Detect concept drift via rolling error rate increase.

        Args:
            model_id: Model identifier.
            recent_errors: Series of per-prediction absolute errors (recent window).
            baseline_error: Mean error from validation/training period.

        Returns:
            (is_drifted, relative_error_increase)
        """
        if len(recent_errors) < self.min_samples:
            return False, 0.0

        current_error = recent_errors.mean()
        if baseline_error <= 0:
            return False, 0.0

        delta = (current_error - baseline_error) / baseline_error
        is_drifted = delta > self.error_threshold

        if is_drifted:
            logger.warning(
                "CONCEPT DRIFT | %s | error increase: %.1f%% (threshold: %.1f%%)",
                model_id, delta * 100, self.error_threshold * 100,
            )
        return is_drifted, round(delta, 4)

    def try_evidently_report(
        self,
        model_id: str,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_dir: str = "outputs/drift_reports",
    ) -> str | None:
        """
        Generate a full Evidently HTML drift report if available.
        Returns the output file path or None if Evidently is not installed.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)

            out_path = Path(output_dir) / f"{model_id}_drift_report.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(out_path))
            logger.info("Evidently drift report saved: %s", out_path)
            return str(out_path)
        except ImportError:
            logger.info("evidently not installed — using PSI-only drift detection")
            return None

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        if max_val == min_val:
            return 0.0

        breakpoints = np.linspace(min_val, max_val, bins + 1)
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Add small epsilon to avoid log(0)
        ref_pct = (ref_counts + 0.5) / (len(reference) + 0.5 * bins)
        cur_pct = (cur_counts + 0.5) / (len(current) + 0.5 * bins)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(psi, 0.0))
