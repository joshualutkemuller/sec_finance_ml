"""Entry point for the Securities Lending Rate Anomaly Detector.

Run modes
---------
train  : fit Isolation Forest and LSTM Autoencoder on historical data.
score  : load saved models, score latest data, fire alerts, push to PowerBI.
both   : train then score in one run (useful for initial setup or retraining).

Usage
-----
    python src/main.py --config config/config.yaml --mode score
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Append project root so imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from alerts import AlertDispatcher, AlertFactory
from data_ingestion import LendingRateDataLoader, load_config
from feature_engineering import LendingRateFeatureEngineer
from model import CombinedAnomalyDetector
from powerbi_connector import PowerBIConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/combined_detector.pkl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Securities Lending Rate Anomaly Detector"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "score", "both"],
        default="score",
        help="Pipeline mode: train | score | both (default: score)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def ingest(config: dict) -> pd.DataFrame:
    logger.info("Stage 1/5 — Data ingestion")
    loader = LendingRateDataLoader(config)
    df = loader.load()
    logger.info("Loaded %d rows, %d columns.", len(df), df.shape[1])
    return df


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Stage 2/5 — Feature engineering")
    eng = LendingRateFeatureEngineer(config)
    df = eng.build_features(df)
    logger.info("Feature matrix shape: %s", df.shape)
    return df


def train(df: pd.DataFrame, config: dict) -> CombinedAnomalyDetector:
    logger.info("Stage 3/5 — Model training")
    detector = CombinedAnomalyDetector(config)
    feature_cols = [c for c in df.columns if c not in {"ticker", "date", "rate"}]
    X = df[feature_cols].dropna().values
    detector.fit(X)
    Path("models").mkdir(exist_ok=True)
    detector.isolation_forest.save(MODEL_PATH + ".if")
    detector.lstm_detector.save(MODEL_PATH + ".lstm")
    logger.info("Models saved to %s.*", MODEL_PATH)
    return detector


def score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Stage 3/5 — Loading models and scoring")
    detector = CombinedAnomalyDetector(config)
    detector.isolation_forest.load(MODEL_PATH + ".if")
    detector.lstm_detector.load(MODEL_PATH + ".lstm")

    feature_cols = [c for c in df.columns if c not in {"ticker", "date", "rate"}]
    X = df[feature_cols].dropna().values
    scores_df = detector.predict(X)

    # Align index back to original df
    idx = df[feature_cols].dropna().index
    result = df.loc[idx, ["ticker", "date", "rate"]].copy()
    result = result.join(scores_df.reset_index(drop=True).set_index(idx))
    result.rename(columns={"date": "timestamp"}, inplace=True)
    return result


def dispatch_alerts(result: pd.DataFrame, config: dict) -> None:
    logger.info("Stage 4/5 — Alert dispatch")
    factory = AlertFactory()
    alerts = factory.from_scores(result, config)
    if alerts:
        logger.info("Firing %d alert(s).", len(alerts))
        dispatcher = AlertDispatcher(config["alerts"])
        dispatcher.dispatch(alerts)
    else:
        logger.info("No alerts triggered.")


def publish(result: pd.DataFrame, config: dict) -> None:
    logger.info("Stage 5/5 — Publishing to PowerBI")
    connector = PowerBIConnector(config["powerbi"])
    connector.publish_alerts(result)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, mode: str) -> None:
    """Run the full anomaly detection pipeline.

    Args:
        config: Parsed config.yaml as a dict.
        mode:   One of "train", "score", "both".
    """
    df = ingest(config)
    df = engineer_features(df, config)

    if mode in ("train", "both"):
        detector = train(df, config)
        if mode == "train":
            logger.info("Training complete. Exiting.")
            return

    result = score(df, config)
    dispatch_alerts(result, config)
    publish(result, config)
    logger.info("Pipeline complete. %d securities scored.", len(result))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, args.mode)
