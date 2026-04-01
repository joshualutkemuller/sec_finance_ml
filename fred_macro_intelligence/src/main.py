"""
FRED Macro Intelligence — Main Entry Point

Modes:
  fetch  → pull latest FRED data into local cache
  train  → train regime detector, spread forecaster, agency demand predictor
  score  → run models, generate charts, push to PowerBI, dispatch alerts
  full   → fetch + train (if needed) + score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_run_id() -> str:
    return str(uuid.uuid4())


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def run_fetch(config: dict) -> None:
    from fred_loader import FREDLoader
    loader = FREDLoader(config)
    df = loader.fetch_all(force_refresh=True)
    logging.getLogger("main.fetch").info(
        "Data refreshed: %d series, %d rows", len(df.columns), len(df)
    )


def run_train(config: dict, run_id: str) -> None:
    logger = logging.getLogger("main.train")
    logger.info("run_id=%s | Starting training", run_id)

    from fred_loader import FREDLoader
    from feature_engineering import MacroFeatureEngineer
    from regime_detector import RegimeDetector
    from spread_forecaster import LGBMSpreadForecaster, LSTMSpreadForecaster
    from agency_demand_predictor import AgencyDemandPredictor

    raw = FREDLoader(config).fetch_all()
    engineer = MacroFeatureEngineer(config)
    features = engineer.build(raw)

    # 1. Regime detector
    detector = RegimeDetector(config)
    detector.fit(features)
    detector.save(config["regime_detector"]["model_path"])
    logger.info("Regime detector trained")

    # 2. LightGBM spread forecaster
    lgbm = LGBMSpreadForecaster(config)
    lgbm.fit(features, raw)
    lgbm.save(config["spread_forecaster"]["lgbm_model_path"])
    logger.info("LightGBM spread forecaster trained")

    # 3. LSTM spread forecaster
    lstm = LSTMSpreadForecaster(config)
    lstm.fit(features, raw)
    lstm.save(config["spread_forecaster"]["lstm_model_path"])
    logger.info("LSTM spread forecaster trained")

    # 4. Agency demand predictor
    # Target: rolling 5-day change in RRP balance as proxy for demand pressure
    if "rrp_balance" in raw.columns:
        y_demand = raw["rrp_balance"].pct_change(5).shift(-5)
        predictor = AgencyDemandPredictor(config)
        predictor.fit(features, y_demand)
        predictor.save(config["agency_demand_predictor"]["model_path"])
        logger.info("Agency demand predictor trained")

    logger.info("run_id=%s | Training complete", run_id)


def run_score(config: dict, run_id: str) -> None:
    logger = logging.getLogger("main.score")
    logger.info("run_id=%s | Starting scoring pipeline", run_id)

    from fred_loader import FREDLoader
    from feature_engineering import MacroFeatureEngineer
    from regime_detector import RegimeDetector
    from spread_forecaster import LGBMSpreadForecaster, LSTMSpreadForecaster
    from agency_demand_predictor import AgencyDemandPredictor
    import visualization as viz
    from alerts import MacroAlertDispatcher

    raw = FREDLoader(config).fetch_all()
    engineer = MacroFeatureEngineer(config)
    features = engineer.build(raw)

    # Regime detection
    detector = RegimeDetector(config)
    detector.load(config["regime_detector"]["model_path"])
    regime_df = detector.predict(features)
    current_regime = regime_df["regime_label"].iloc[-1]
    logger.info("Current regime: %s", current_regime)

    # Spread forecasting
    lgbm = LGBMSpreadForecaster(config)
    lgbm.load(config["spread_forecaster"]["lgbm_model_path"])
    lgbm_forecasts = lgbm.predict(features)

    lstm = LSTMSpreadForecaster(config)
    lstm.load(config["spread_forecaster"]["lstm_model_path"])
    try:
        lstm_forecasts = lstm.predict_with_uncertainty(features)
    except Exception as exc:
        logger.warning("LSTM inference failed: %s", exc)
        lstm_forecasts = None

    # Agency demand
    predictor = AgencyDemandPredictor(config)
    predictor.load(config["agency_demand_predictor"]["model_path"])
    demand_signal = predictor.predict(features)
    demand_drivers = predictor.top_drivers(features.tail(1))
    logger.info("Agency demand signal (latest): %.4f", demand_signal[-1])

    # Generate charts
    out_dir = config["visualization"].get("export_dir", "outputs/charts")
    fmt = config["visualization"].get("export_format", "png")

    charts = {
        "regime_timeline": viz.regime_timeline(raw, regime_df),
        "rolling_correlation": viz.rolling_correlation_matrix(features),
        "funding_stress": viz.funding_stress_dashboard(features),
        "regime_transition": viz.regime_transition_matrix(
            detector.regime_transition_matrix(regime_df["regime_id"])
        ),
    }
    for target in config["spread_forecaster"]["targets"]:
        if target in lgbm_forecasts:
            key = f"spread_forecast_{target}"
            lstm_t = lstm_forecasts.get(target) if lstm_forecasts else None
            charts[key] = viz.spread_forecast_bands(
                raw,
                forecasts_lgbm=lgbm_forecasts[target],
                forecasts_lstm=lstm_t,
                target_col=target,
            )

    for chart_name, fig in charts.items():
        export_path = f"{out_dir}/{chart_name}.{fmt}"
        viz.export_chart(fig, export_path, fmt=fmt)
        logger.info("Chart exported: %s", export_path)

    # Alerts
    dispatcher = MacroAlertDispatcher(config)
    dispatcher.dispatch_regime_change(current_regime, regime_df)
    dispatcher.dispatch_spread_spike(lgbm_forecasts, features)

    # Audit log
    _write_audit(config, run_id, current_regime, demand_signal[-1])
    logger.info("run_id=%s | Scoring complete", run_id)


def _write_audit(config: dict, run_id: str, regime: str, demand: float) -> None:
    log_path = Path(config["logging"]["audit_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "current_regime": regime,
        "agency_demand_signal": float(demand),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FRED Macro Intelligence Pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        choices=["fetch", "train", "score", "full"],
        default="full",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_id = _make_run_id()

    if args.mode == "fetch":
        run_fetch(config)
    elif args.mode == "train":
        run_train(config, run_id)
    elif args.mode == "score":
        run_score(config, run_id)
    elif args.mode == "full":
        run_fetch(config)
        run_train(config, run_id)
        run_score(config, run_id)


if __name__ == "__main__":
    main()
