"""
main.py — Pipeline orchestration for Collateral Optimization & Substitution Recommender

Usage
-----
    python src/main.py
    python src/main.py --config config/config.yaml --mode full
    python src/main.py --mode score_only
    python src/main.py --mode optimize_only --scored-data /path/to/scored.csv

Modes
-----
    full            Ingest → Feature Engineer → Score → Optimise → Alert → Push
    score_only      Ingest → Feature Engineer → Score (writes scored CSV)
    optimize_only   Load pre-scored CSV → Optimise → Alert → Push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Add src/ to path so imports resolve when running as script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from alerts import CollateralAlertDispatcher, CollateralAlertFactory
from data_ingestion import CollateralDataLoader
from feature_engineering import CollateralFeatureEngineer, FEATURE_COLS
from model import CollateralOptimizer, CollateralQualityScorer
from powerbi_connector import PowerBIConnector

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("collateral_optimizer.main")


# ===========================================================================
# Config helpers
# ===========================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", path)
    return cfg


def _config_hash(config_path: str) -> str:
    """Return an MD5 hash of the config file for audit purposes."""
    with open(config_path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ===========================================================================
# Pipeline stages
# ===========================================================================


def ingest_and_engineer(config: Dict[str, Any]) -> pd.DataFrame:
    """Stage 1 & 2: load data and compute features.

    Parameters
    ----------
    config : dict
        Full pipeline config.

    Returns
    -------
    pd.DataFrame
        Feature-engineered collateral DataFrame.
    """
    loader = CollateralDataLoader(config["data"])
    merged_df = loader.merge_all()

    engineer = CollateralFeatureEngineer(config["optimization"])
    feature_df = engineer.build_features(merged_df)
    return feature_df


def score_collateral(
    feature_df: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    """Stage 3: run quality scorer.

    Attempts to load a pre-trained model from
    ``config.model.model_save_path``.  If no model exists, trains a
    synthetic one using random quality labels (development / demo mode).

    Parameters
    ----------
    feature_df : pd.DataFrame
    config : dict

    Returns
    -------
    np.ndarray
        Quality scores, shape ``(n,)``.
    """
    scorer = CollateralQualityScorer(config["model"])
    model_path: str = config["model"].get("model_save_path", "models/scorer.joblib")

    if Path(model_path).exists():
        scorer.load(model_path)
        logger.info("Pre-trained model loaded from %s", model_path)
    else:
        logger.warning(
            "No pre-trained model found at %s; training on synthetic labels (demo mode).",
            model_path,
        )
        rng = np.random.default_rng(config["model"].get("random_state", 42))
        y_synthetic = pd.Series(
            rng.uniform(0.0, 1.0, size=len(feature_df)), index=feature_df.index
        )
        scorer.fit(feature_df, y_synthetic)
        scorer.save(model_path)

    return scorer.score(feature_df)


def run_optimization(
    feature_df: pd.DataFrame,
    quality_scores: np.ndarray,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Stage 4 & 5: optimise allocation and generate recommendations df.

    Parameters
    ----------
    feature_df : pd.DataFrame
    quality_scores : np.ndarray
    config : dict

    Returns
    -------
    pd.DataFrame
        Optimised DataFrame with ``allocation_weight``, ``recommended``,
        ``optimization_cost`` columns and ``quality_score`` column added.
    """
    required_value = float(
        feature_df.get("required_value", pd.Series([1_000_000.0])).iloc[0]
        if "required_value" in feature_df.columns
        else 1_000_000.0
    )

    optimizer = CollateralOptimizer(config["optimization"])
    feature_df = feature_df.copy()
    feature_df["quality_score"] = quality_scores

    optimized_df = optimizer.optimize(feature_df, quality_scores, required_value)
    return optimized_df


def dispatch_alerts(
    recommendations: List[Dict[str, Any]],
    config: Dict[str, Any],
    optimization_failed: bool = False,
) -> int:
    """Stage 6 & 7: evaluate thresholds and dispatch alerts.

    Parameters
    ----------
    recommendations : list of dict
    config : dict
    optimization_failed : bool
        If ``True``, also fires an OPTIMISATION_FAILURE alert.

    Returns
    -------
    int
        Number of alerts dispatched.
    """
    dispatcher = CollateralAlertDispatcher(config["alerts"])
    alerts = CollateralAlertFactory.from_recommendations(recommendations, config)

    if optimization_failed:
        alerts.append(
            CollateralAlertFactory.optimisation_failure_alert("infeasible")
        )

    dispatcher.dispatch(alerts)
    return len(alerts)


def push_to_powerbi(
    recommendations: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> bool:
    """Stage 8: push results to PowerBI.

    Parameters
    ----------
    recommendations : list of dict
    config : dict

    Returns
    -------
    bool
        ``True`` if push succeeded or is disabled.
    """
    connector = PowerBIConnector(config["powerbi"])
    return connector.publish_recommendations(recommendations)


def write_audit_log(
    run_id: str,
    config_path: str,
    config: Dict[str, Any],
    n_evaluated: int,
    n_recommended: int,
    optimization_status: str,
    total_opt_cost: float,
    n_alerts: int,
    pbi_status: bool,
) -> None:
    """Stage 9: append an audit entry to the JSONL log.

    Parameters
    ----------
    run_id : str
    config_path : str
    config : dict
    n_evaluated : int
    n_recommended : int
    optimization_status : str
    total_opt_cost : float
    n_alerts : int
    pbi_status : bool
    """
    log_path = Path(config["alerts"].get("log_path", "logs/collateral_alerts.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "entry_type": "audit",
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "n_collateral_items_evaluated": n_evaluated,
        "n_recommended": n_recommended,
        "optimization_status": optimization_status,
        "total_optimization_cost": total_opt_cost,
        "n_alerts_dispatched": n_alerts,
        "powerbi_push_status": "success" if pbi_status else "failure",
        "config_snapshot": _config_hash(config_path),
    }

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    logger.info("Audit log entry written (run_id=%s)", run_id)


# ===========================================================================
# Top-level pipeline runners
# ===========================================================================


def run_pipeline(config_path: str, mode: str, scored_data_path: Optional[str]) -> None:
    """Execute the full (or partial) collateral optimization pipeline.

    Parameters
    ----------
    config_path : str
        Path to ``config.yaml``.
    mode : str
        One of ``"full"``, ``"score_only"``, ``"optimize_only"``.
    scored_data_path : str or None
        Required when ``mode == "optimize_only"``; path to a CSV
        containing pre-scored collateral data.
    """
    run_id = str(uuid.uuid4())
    logger.info("=== Collateral Optimizer run started [run_id=%s, mode=%s] ===", run_id, mode)

    config = load_config(config_path)
    load_dotenv()

    # ------------------------------------------------------------------
    # Stage 1 & 2: Ingest + Feature Engineering
    # ------------------------------------------------------------------
    if mode in ("full", "score_only"):
        feature_df = ingest_and_engineer(config)
    elif mode == "optimize_only":
        if not scored_data_path:
            raise ValueError("--scored-data is required for optimize_only mode.")
        feature_df = pd.read_csv(scored_data_path)
        logger.info("Pre-scored data loaded from %s (%d rows)", scored_data_path, len(feature_df))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from: full, score_only, optimize_only")

    n_evaluated = len(feature_df)

    # ------------------------------------------------------------------
    # Stage 3: Quality Scoring
    # ------------------------------------------------------------------
    if mode in ("full", "score_only"):
        quality_scores = score_collateral(feature_df, config)
        feature_df["quality_score"] = quality_scores
    else:
        # optimize_only: expect quality_score column in pre-scored data
        if "quality_score" not in feature_df.columns:
            raise ValueError(
                "Pre-scored data must contain a 'quality_score' column "
                "when running in optimize_only mode."
            )
        quality_scores = feature_df["quality_score"].values

    if mode == "score_only":
        out_path = Path("data/scored_collateral.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(out_path, index=False)
        logger.info("score_only mode complete; scored data written to %s", out_path)
        return

    # ------------------------------------------------------------------
    # Stage 4 & 5: Optimization + Recommendations
    # ------------------------------------------------------------------
    optimized_df = run_optimization(feature_df, quality_scores, config)

    n_recommended = int(optimized_df["recommended"].sum())
    optimization_failed = n_recommended == 0
    optimization_status = "infeasible" if optimization_failed else "optimal"
    total_opt_cost = float(optimized_df["optimization_cost"].sum())

    optimizer = CollateralOptimizer(config["optimization"])
    recommendations = optimizer.generate_recommendations(optimized_df)

    logger.info(
        "Optimization complete: %d / %d items recommended, total cost=%.6f",
        n_recommended,
        n_evaluated,
        total_opt_cost,
    )

    # ------------------------------------------------------------------
    # Stage 6 & 7: Alerts
    # ------------------------------------------------------------------
    n_alerts = dispatch_alerts(recommendations, config, optimization_failed)

    # ------------------------------------------------------------------
    # Stage 8: PowerBI Push
    # ------------------------------------------------------------------
    pbi_ok = push_to_powerbi(recommendations, config)

    # ------------------------------------------------------------------
    # Stage 9: Audit Log
    # ------------------------------------------------------------------
    write_audit_log(
        run_id=run_id,
        config_path=config_path,
        config=config,
        n_evaluated=n_evaluated,
        n_recommended=n_recommended,
        optimization_status=optimization_status,
        total_opt_cost=total_opt_cost,
        n_alerts=n_alerts,
        pbi_status=pbi_ok,
    )

    logger.info(
        "=== Run complete [run_id=%s]: %d recommended, %d alerts, PowerBI=%s ===",
        run_id,
        n_recommended,
        n_alerts,
        "OK" if pbi_ok else "SKIP/FAIL",
    )


# ===========================================================================
# CLI
# ===========================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="collateral_optimizer",
        description="Collateral Optimization & Substitution Recommender pipeline",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "score_only", "optimize_only"],
        default="full",
        help=(
            "Pipeline mode: 'full' runs all stages; "
            "'score_only' stops after quality scoring; "
            "'optimize_only' skips ingestion and scoring (requires --scored-data)."
        ),
    )
    parser.add_argument(
        "--scored-data",
        default=None,
        dest="scored_data",
        help="Path to pre-scored CSV (required for optimize_only mode).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        run_pipeline(
            config_path=args.config,
            mode=args.mode,
            scored_data_path=args.scored_data,
        )
    except Exception as exc:  # noqa: BLE001
        logger.critical("Pipeline failed with unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
