"""Entry point for the Short Squeeze Probability Scorer.

Usage:
    python src/main.py --config config/config.yaml --mode score
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from alerts import SqueezeAlertDispatcher, SqueezeAlertFactory
from data_ingestion import ShortInterestDataLoader, load_config
from feature_engineering import ShortSqueezeFeatureEngineer
from model import XGBoostSqueezeScorer
from powerbi_connector import PowerBIConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/squeeze_scorer.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short Squeeze Probability Scorer")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode", choices=["train", "score", "both"], default="score"
    )
    return parser.parse_args()


def run_pipeline(config: dict, mode: str) -> None:
    """Run the full short squeeze scoring pipeline.

    Args:
        config: Parsed config.yaml.
        mode:   train | score | both.
    """
    # 1. Ingest
    logger.info("Stage 1/5 — Data ingestion")
    loader = ShortInterestDataLoader(config)
    df = loader.merge_all()
    logger.info("Loaded %d rows across %d tickers.", len(df), df["ticker"].nunique())

    # 2. Feature engineering
    logger.info("Stage 2/5 — Feature engineering")
    cfg_feat = config.get("features", {})
    eng = ShortSqueezeFeatureEngineer(config)
    df = eng.build_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in {"ticker", "date", "close", "squeeze_event"}
    ]

    # 3. Train or load
    scorer = XGBoostSqueezeScorer(config["model"])
    if mode in ("train", "both"):
        logger.info("Stage 3/5 — Training")
        if "squeeze_event" not in df.columns:
            raise ValueError("Training requires 'squeeze_event' label column (0/1).")
        X = df[feature_cols].dropna()
        y = df.loc[X.index, "squeeze_event"].astype(int)
        scorer.fit(X, y)
        Path("models").mkdir(exist_ok=True)
        scorer.save(MODEL_PATH)
        logger.info("Model saved → %s", MODEL_PATH)
        if mode == "train":
            return
    else:
        scorer.load(MODEL_PATH)

    # 4. Score
    logger.info("Stage 3/5 — Scoring")
    X = df[feature_cols].dropna()
    probs = scorer.score(X)
    risk_tiers = scorer.classify(probs)

    result = df.loc[X.index, ["ticker", "date"]].copy()
    result["short_interest_ratio"] = df.loc[X.index, "short_interest_ratio"] \
        if "short_interest_ratio" in df.columns else float("nan")
    result["days_to_cover"] = df.loc[X.index, "days_to_cover"] \
        if "days_to_cover" in df.columns else float("nan")
    result["utilization_rate"] = df.loc[X.index, "utilization_rate"] \
        if "utilization_rate" in df.columns else float("nan")
    result["squeeze_probability"] = probs
    result["risk_tier"] = [r.value for r in risk_tiers]
    result["alert_flag"] = [r.value in ("HIGH", "CRITICAL") for r in risk_tiers]
    result.rename(columns={"date": "timestamp"}, inplace=True)

    # 5. Alerts
    logger.info("Stage 4/5 — Alert dispatch")
    factory = SqueezeAlertFactory()
    alerts = factory.from_scores(result, risk_tiers, config)
    if alerts:
        dispatcher = SqueezeAlertDispatcher(config["alerts"])
        dispatcher.dispatch(alerts)
        logger.info("Fired %d alert(s).", len(alerts))

    # 6. PowerBI
    logger.info("Stage 5/5 — PowerBI publish")
    connector = PowerBIConnector(config["powerbi"])
    connector.publish_scores(result)
    logger.info("Pipeline complete. %d rows published.", len(result))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, args.mode)
