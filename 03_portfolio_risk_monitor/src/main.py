"""Entry point for the Portfolio Concentration Risk Early Warning Monitor.

Usage:
    python src/main.py --config config/config.yaml --mode score
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from alerts import RiskAlertDispatcher, RiskAlertFactory
from data_ingestion import PortfolioDataLoader, load_config
from feature_engineering import ConcentrationFeatureEngineer
from model import RandomForestRiskScorer
from powerbi_connector import PowerBIConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/risk_scorer.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio Concentration Risk Early Warning Monitor"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode", choices=["train", "score", "both"], default="score"
    )
    return parser.parse_args()


def run_pipeline(config: dict, mode: str) -> None:
    """Orchestrate ingestion → features → model → alerts → PowerBI.

    Args:
        config: Parsed config.yaml.
        mode:   train | score | both.
    """
    # 1. Ingest
    logger.info("Stage 1/5 — Ingestion")
    loader = PortfolioDataLoader(config)
    df = loader.build_exposure_snapshot()
    logger.info("Exposure snapshot: %d rows", len(df))

    # 2. Feature engineering
    logger.info("Stage 2/5 — Feature engineering")
    eng = ConcentrationFeatureEngineer(config)
    df = eng.build_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in {"portfolio_id", "sector", "counterparty", "date", "is_high_risk"}
    ]

    # 3. Train or load
    scorer = RandomForestRiskScorer(config["model"])
    if mode in ("train", "both"):
        logger.info("Stage 3/5 — Training")
        if "is_high_risk" not in df.columns:
            raise ValueError("Training requires 'is_high_risk' label column.")
        X = df[feature_cols].dropna()
        y = df.loc[X.index, "is_high_risk"].astype(int)
        scorer.fit(X, y)
        Path("models").mkdir(exist_ok=True)
        scorer.save(MODEL_PATH)
        if mode == "train":
            logger.info("Training complete.")
            return
    else:
        scorer.load(MODEL_PATH)

    # 4. Score
    logger.info("Stage 3/5 — Scoring")
    X = df[feature_cols].dropna()
    scores = scorer.score(X)
    risk_levels = scorer.classify(scores)

    shap_df: pd.DataFrame | None = None
    if config.get("shap", {}).get("enabled", False):
        shap_df = scorer.explain(X)

    result = df.loc[X.index, ["portfolio_id", "sector", "counterparty"]].copy()
    result["concentration_score"] = scores
    result["risk_level"] = [r.value for r in risk_levels]
    result["alert_flag"] = [r.value in ("HIGH", "CRITICAL") for r in risk_levels]
    result["timestamp"] = pd.Timestamp.utcnow()

    # 5. Alerts
    logger.info("Stage 4/5 — Alerts")
    factory = RiskAlertFactory()
    alerts = factory.from_scores(result, risk_levels, shap_df, config)
    if alerts:
        dispatcher = RiskAlertDispatcher(config["alerts"])
        dispatcher.dispatch(alerts)
        logger.info("Fired %d alert(s).", len(alerts))

    # 6. PowerBI
    logger.info("Stage 5/5 — PowerBI publish")
    connector = PowerBIConnector(config["powerbi"])
    connector.publish_risk_scores(result)
    logger.info("Pipeline complete. %d portfolios scored.", len(result))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, args.mode)
