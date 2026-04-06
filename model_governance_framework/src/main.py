"""
Model Governance Framework — Main Entry Point

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Modes:
  validate   → run data validation on all model baselines
  drift      → run drift detection across all models
  report     → generate HTML governance report + PowerBI push
  full       → validate + drift + report
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# All 9 model IDs in the platform
PLATFORM_MODELS = [
    "01_lending_rate_anomaly",
    "02_collateral_optimizer",
    "03_portfolio_risk_monitor",
    "04_short_squeeze_scorer",
    "05_financing_cost_forecaster",
    "securities_lending_optimizer",
    "fred_macro_intelligence",
    "counterparty_graph_risk",
    "nlp_signal_engine",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Governance Framework — Joshua Lutkemuller, MD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["validate", "drift", "report", "full"],
        default="full",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_id = str(uuid.uuid4())
    logger = logging.getLogger("governance.main")
    logger.info("run_id=%s | mode=%s | Model Governance Framework", run_id, args.mode)

    from drift_detector import DriftDetector
    from governance_report import GovernanceReportGenerator
    from audit_logger import AuditLogger

    drift_detector = DriftDetector(config)
    baseline_dir = Path(config["models"]["baseline_data_dir"])
    predictions_dir = Path(config["models"]["predictions_log_dir"])

    drift_reports = []
    performance_records = []
    recent_alerts = []

    if args.mode in ("drift", "full"):
        for model_id in PLATFORM_MODELS:
            ref_path = baseline_dir / f"{model_id}_baseline.parquet"
            cur_path = predictions_dir / f"{model_id}_current.parquet"

            if not ref_path.exists() or not cur_path.exists():
                logger.info("Skipping %s — baseline or current data not found", model_id)
                continue

            reference = pd.read_parquet(ref_path)
            current = pd.read_parquet(cur_path)

            report = drift_detector.detect_data_drift(model_id, reference, current)
            drift_reports.append(report)

            # Try Evidently HTML report
            drift_detector.try_evidently_report(
                model_id, reference, current,
                output_dir=f"outputs/drift_reports",
            )

            if report.drift_status in ("WARNING", "CRITICAL"):
                recent_alerts.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_id": model_id,
                    "severity": report.drift_status,
                    "alert_type": "DATA_DRIFT",
                    "message": f"PSI max={report.max_psi:.3f}, {report.n_features_drifted} features drifted",
                })

        logger.info("Drift check complete: %d models checked, %d alerts",
                    len(drift_reports),
                    sum(1 for r in drift_reports if r.drift_status != "OK"))

    if args.mode in ("report", "full"):
        gen = GovernanceReportGenerator(config)
        output_path = gen.generate(drift_reports, performance_records, recent_alerts)
        logger.info("Governance report: %s", output_path)

    # Audit log
    log_path = Path(config.get("alerts", {}).get("log_path", "logs/governance_alerts.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": args.mode,
            "models_checked": len(drift_reports),
            "alerts_fired": len(recent_alerts),
        }) + "\n")

    logger.info("Governance run complete. run_id=%s", run_id)


if __name__ == "__main__":
    main()
