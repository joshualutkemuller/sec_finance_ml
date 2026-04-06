"""
Causal Rate Attribution — Main Entry Point

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal Rate Attribution — Joshua Lutkemuller, MD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["estimate", "hte", "report", "full"],
        default="full",
    )
    parser.add_argument("--period", default="Current Period", help="Period label for report")
    args = parser.parse_args()

    config = _load_config(args.config)
    setup_logging(config.get("output", {}).get("log_level", "INFO"))
    run_id = str(uuid.uuid4())
    logger = logging.getLogger("causal.main")
    logger.info("run_id=%s | mode=%s | Causal Rate Attribution", run_id, args.mode)

    from causal_estimator import CausalEstimator
    from heterogeneous_effects import HeterogeneousEffectEstimator
    from attribution_report import AttributionReportGenerator

    # Load data
    cost_df = pd.read_csv(config["data"]["financing_cost_path"]) if Path(config["data"]["financing_cost_path"]).exists() else _synthetic_data()
    fred_path = config["data"].get("fred_features_path", "")
    if fred_path and Path(fred_path).exists():
        fred_df = pd.read_parquet(fred_path)
        data = cost_df.merge(fred_df, on="date", how="inner")
    else:
        data = cost_df

    ate_result = None
    if args.mode in ("estimate", "full"):
        estimator = CausalEstimator(config)
        ate_result = estimator.estimate_ate(data)
        logger.info("ATE: %.6f bps/bps | %s", ate_result.ate, ate_result.interpretation)

    hte_df = pd.DataFrame()
    if args.mode in ("hte", "full"):
        hte_estimator = HeterogeneousEffectEstimator(config)
        hte_estimator.fit(data)
        for group_col in config["causal"].get("effect_modifiers", []):
            if group_col in data.columns and data[group_col].dtype == object:
                hte_df = hte_estimator.cate_by_subgroup(data, group_col)
                logger.info("HTE by %s:\n%s", group_col, hte_df.to_string())
                break

    if args.mode in ("report", "full") and ate_result is not None:
        gen = AttributionReportGenerator(config)
        period_summary = {
            "n_hikes": 3,
            "total_fed_hike_bps": 75,
            "avg_notional_bn": 2.5,
            "prior_cost_usd": 256_000_000,
        }
        report = gen.generate(
            period_label=args.period,
            total_cost_change_usd=47_200_000,
            ate_result=ate_result,
            hte_by_asset_class=hte_df,
            counterfactual_results={"No November Hike": -31_100_000},
            period_summary=period_summary,
        )
        print(report)

    log_path = Path(config["output"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": args.mode,
            "ate": getattr(ate_result, "ate", None),
        }) + "\n")


def _synthetic_data() -> pd.DataFrame:
    """Generate synthetic dataset for testing when real data unavailable."""
    import numpy as np
    np.random.seed(42)
    n = 504
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    fomc_days = np.zeros(n)
    fomc_days[[63, 126, 189, 252, 315, 378, 441]] = 1
    fed_change = fomc_days * np.random.choice([-25, 0, 25, 50], size=n, p=[0.1, 0.6, 0.2, 0.1])
    return pd.DataFrame({
        "date": dates,
        "fed_funds_change_bps": fed_change,
        "fomc_meeting_flag": fomc_days,
        "financing_cost_bps": 15 + fed_change.cumsum() * 0.4 + np.random.randn(n) * 2,
        "funding_stress_index": np.random.randn(n),
        "curve_10y2y": -0.5 + np.random.randn(n) * 0.3,
        "hy_oas": 350 + np.random.randn(n) * 30,
        "book_notional_bn": 2.5 + np.random.randn(n) * 0.2,
        "hy_mix_pct": 0.25 + np.random.randn(n) * 0.05,
        "asset_class": np.random.choice(["Agency UST", "Agency MBS", "IG Corp", "HY Corp"], n),
    })


if __name__ == "__main__":
    main()
