"""Entry point for the Repo & Financing Cost Forecaster.

Usage:
    python src/main.py --config config/config.yaml --mode forecast
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from alerts import ForecastAlertDispatcher, ForecastAlertFactory
from data_ingestion import FinancingDataLoader, load_config
from feature_engineering import FinancingCostFeatureEngineer
from model import MultiHorizonForecaster
from powerbi_connector import PowerBIConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/financing_forecaster.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo & Financing Cost Forecaster")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode", choices=["train", "forecast", "both"], default="forecast"
    )
    return parser.parse_args()


def run_pipeline(config: dict, mode: str) -> None:
    """Run the full financing cost forecasting pipeline.

    Args:
        config: Parsed config.yaml dict.
        mode:   train | forecast | both.
    """
    model_cfg = config["model"]
    horizons: list[int] = model_cfg.get("horizons", [1, 3, 5])

    # 1. Ingest
    logger.info("Stage 1/6 — Data ingestion")
    loader = FinancingDataLoader(config)
    df = loader.merge_all()
    budget_df = loader.load_budget()
    logger.info("Loaded %d rows.", len(df))

    # 2. Feature engineering
    logger.info("Stage 2/6 — Feature engineering")
    eng = FinancingCostFeatureEngineer(config)
    target_col = config["data"].get("rate_col", "rate_bps")
    df_feat, _ = eng.build_features(df, target_col)
    feature_cols = [
        c for c in df_feat.columns
        if c not in {"instrument_type", "date", target_col}
    ]

    # 3. Build multi-horizon targets
    forecaster = MultiHorizonForecaster(config["model"])
    y_dict: dict[int, pd.Series] = {}
    for h in horizons:
        y_dict[h] = forecaster._build_horizon_target(df_feat, target_col, h)

    # Align features and targets (drop NaN rows introduced by shifting)
    valid_idx = df_feat.index
    for h in horizons:
        valid_idx = valid_idx.intersection(y_dict[h].dropna().index)
    X = df_feat.loc[valid_idx, feature_cols]
    y_dict_aligned = {h: y_dict[h].loc[valid_idx] for h in horizons}

    # 4. Train or load
    if mode in ("train", "both"):
        logger.info("Stage 3/6 — Training %d horizon models", len(horizons))
        forecaster.fit(X, y_dict_aligned)
        Path("models").mkdir(exist_ok=True)
        forecaster.save(MODEL_PATH)
        logger.info("Model saved → %s", MODEL_PATH)
        if mode == "train":
            eval_metrics = forecaster.evaluate(X, y_dict_aligned)
            for h, metrics in eval_metrics.items():
                logger.info("Horizon %dd — MAE: %.2f bps, RMSE: %.2f bps",
                            h, metrics["mae"], metrics["rmse"])
            return
    else:
        forecaster.load(MODEL_PATH)

    # 5. Forecast
    logger.info("Stage 4/6 — Generating forecasts")
    # Use the most recent available rows for forward-looking forecast
    latest = df_feat.tail(1)
    X_latest = latest[feature_cols]
    forecasts = forecaster.predict(X_latest)

    result_rows = []
    for _, row in df_feat.loc[valid_idx].tail(
        df[config["data"].get("instrument_type_col", "instrument_type")].nunique()
        if "instrument_type" in df.columns else 1
    ).iterrows():
        instrument = row.get("instrument_type", "ALL")
        current_rate = float(row.get(target_col, 0.0))
        result_rows.append({
            "instrument_type": instrument,
            "current_rate_bps": current_rate,
            "forecast_1d_bps": float(forecasts.get(1, [current_rate])[0]),
            "forecast_3d_bps": float(forecasts.get(3, [current_rate])[0]),
            "forecast_5d_bps": float(forecasts.get(5, [current_rate])[0]),
            "timestamp": pd.Timestamp.utcnow(),
        })
    result = pd.DataFrame(result_rows)

    # Merge budget and compute cost_vs_budget
    result = result.merge(budget_df.rename(columns={"budget_rate_bps": "budget"}),
                          on="instrument_type", how="left")
    result["cost_vs_budget_bps"] = result["forecast_1d_bps"] - result.get("budget", float("nan"))
    result["alert_flag"] = result["cost_vs_budget_bps"].gt(
        config["alerts"].get("budget_breach_threshold_bps", 10)
    )

    # 6. Alerts
    logger.info("Stage 5/6 — Alert dispatch")
    factory = ForecastAlertFactory()
    alerts = factory.from_forecasts(result, budget_df, config)
    if alerts:
        dispatcher = ForecastAlertDispatcher(config["alerts"])
        dispatcher.dispatch(alerts)
        logger.info("Fired %d alert(s).", len(alerts))

    # 7. PowerBI
    logger.info("Stage 6/6 — PowerBI publish")
    connector = PowerBIConnector(config["powerbi"])
    connector.publish_forecasts(result)
    logger.info("Pipeline complete. %d instrument forecasts published.", len(result))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, args.mode)
