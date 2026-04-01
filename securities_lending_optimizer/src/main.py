"""
Enhanced Securities Lending Optimizer — Main Entry Point

Orchestrates the full ML pipeline:
  train     → train demand forecaster + recall classifier + RL policy
  train_rl  → train RL policy only (assumes other models are already trained)
  score     → run demand forecast + recall scoring + multi-objective optimization
  both      → train then score
"""

import argparse
import logging
import uuid
import hashlib
import json
from pathlib import Path
from datetime import datetime

import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_run_id() -> str:
    return str(uuid.uuid4())


def _config_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:8]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def run_train(config: dict, run_id: str) -> None:
    """Train demand forecaster, recall classifier, and RL policy."""
    logger = logging.getLogger("slo.train")
    logger.info("run_id=%s | Starting training pipeline", run_id)

    from data_ingestion import LendingDataIngestion
    from feature_engineering import FeatureEngineer
    from demand_forecaster import DemandForecaster
    from recall_risk_classifier import RecallRiskClassifier

    loader = LendingDataIngestion(config)
    loan_book, collateral_pool, market_data, counterparties = loader.load_all()
    logger.info("Loaded data — loan_book=%d rows, collateral_pool=%d rows",
                len(loan_book), len(collateral_pool))

    engineer = FeatureEngineer(config)
    X_demand, y_dict = engineer.build_demand_features(loan_book, market_data)
    X_recall, y_recall = engineer.build_recall_features(loan_book, counterparties, market_data)

    # Demand forecaster
    forecaster = DemandForecaster(config)
    forecaster.fit(X_demand, y_dict)
    forecaster.save(config["demand_forecaster"]["model_path"])
    logger.info("Demand forecaster trained and saved")

    # Recall risk classifier
    classifier = RecallRiskClassifier(config)
    classifier.fit(X_recall, y_recall)
    classifier.save(config["recall_classifier"]["calibrated_model_path"])
    logger.info("Recall risk classifier trained and saved")

    # RL policy — expensive, log prominently
    logger.info("Starting RL policy training — this may take 30-60 minutes on CPU")
    _train_rl_policy(config, loan_book, collateral_pool, market_data, run_id)
    logger.info("run_id=%s | Training pipeline complete", run_id)


def _train_rl_policy(
    config: dict,
    loan_book,
    collateral_pool,
    market_data,
    run_id: str,
) -> None:
    """Train the PPO/SAC RL policy."""
    logger = logging.getLogger("slo.rl_train")
    from rl_policy import LendingEnv, RLPolicyAgent

    env = LendingEnv(loan_book, collateral_pool, market_data, config)
    agent = RLPolicyAgent(env, config)
    agent.train(total_timesteps=config["rl_policy"]["total_timesteps"])
    agent.save(config["rl_policy"]["model_path"])
    logger.info("RL policy saved to %s", config["rl_policy"]["model_path"])


def run_score(config: dict, run_id: str) -> None:
    """Load models, generate signals, run optimizer, dispatch alerts."""
    logger = logging.getLogger("slo.score")
    logger.info("run_id=%s | Starting scoring pipeline", run_id)

    from data_ingestion import LendingDataIngestion
    from feature_engineering import FeatureEngineer
    from demand_forecaster import DemandForecaster
    from recall_risk_classifier import RecallRiskClassifier
    from rl_policy import RLPolicyAgent, LendingEnv
    from multi_objective_optimizer import MultiObjectiveOptimizer
    from alerts import AlertDispatcher

    loader = LendingDataIngestion(config)
    loan_book, collateral_pool, market_data, counterparties = loader.load_all()

    engineer = FeatureEngineer(config)
    X_demand, _ = engineer.build_demand_features(loan_book, market_data)
    X_recall, _ = engineer.build_recall_features(loan_book, counterparties, market_data)
    X_collateral = engineer.build_collateral_features(collateral_pool, market_data)

    # Demand forecast
    forecaster = DemandForecaster(config)
    forecaster.load(config["demand_forecaster"]["model_path"])
    demand_forecast = forecaster.predict(X_demand)
    logger.info("Demand forecast generated for horizons %s",
                list(demand_forecast.keys()))

    # Recall risk
    classifier = RecallRiskClassifier(config)
    classifier.load(config["recall_classifier"]["calibrated_model_path"])
    recall_probs = classifier.predict_proba(X_recall)
    logger.info("Recall probabilities computed — mean=%.4f", recall_probs.mean())

    # RL policy → objective weights
    env = LendingEnv(loan_book, collateral_pool, market_data, config)
    agent = RLPolicyAgent(env, config)
    agent.load(config["rl_policy"]["model_path"])
    current_state = env.get_current_state()
    objective_weights = agent.recommend(current_state)
    logger.info("RL objective weights: %s", objective_weights.tolist())

    # Multi-objective optimization
    optimizer = MultiObjectiveOptimizer(config)
    result = optimizer.optimize(
        collateral_df=X_collateral,
        objective_weights=objective_weights,
        recall_probs=recall_probs,
        demand_forecast=demand_forecast,
    )
    logger.info("Optimization complete — status=%s, cost=%.4f",
                result.status, result.total_cost)

    # Alert dispatch
    dispatcher = AlertDispatcher(config)
    alerts = result.build_alerts(run_id)
    dispatcher.dispatch_all(alerts)
    logger.info("Dispatched %d alerts", len(alerts))

    # Audit log
    _write_audit(config, run_id, result, len(alerts))


def _write_audit(config: dict, run_id: str, result, alert_count: int) -> None:
    log_path = Path(config["logging"]["audit_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": result.status,
        "total_cost": result.total_cost,
        "alert_count": alert_count,
        "objective_weights": result.objective_weights.tolist()
        if hasattr(result.objective_weights, "tolist")
        else list(result.objective_weights),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Securities Lending Optimizer")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        choices=["train", "train_rl", "score", "both"],
        default="score",
        help="Run mode",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override RL training timesteps",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.timesteps:
        config["rl_policy"]["total_timesteps"] = args.timesteps

    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_id = _make_run_id()
    cfg_hash = _config_hash(config)
    logging.getLogger("slo.main").info(
        "run_id=%s | config_hash=%s | mode=%s", run_id, cfg_hash, args.mode
    )

    if args.mode == "train":
        run_train(config, run_id)
    elif args.mode == "train_rl":
        from data_ingestion import LendingDataIngestion
        loader = LendingDataIngestion(config)
        lb, cp, md, _ = loader.load_all()
        _train_rl_policy(config, lb, cp, md, run_id)
    elif args.mode == "score":
        run_score(config, run_id)
    elif args.mode == "both":
        run_train(config, run_id)
        run_score(config, run_id)


if __name__ == "__main__":
    main()
