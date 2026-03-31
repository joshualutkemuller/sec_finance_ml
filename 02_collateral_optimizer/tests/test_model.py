"""Unit tests for the Collateral Optimization & Substitution Recommender.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alerts import CollateralAlertFactory
from feature_engineering import CollateralFeatureEngineer
from model import CollateralOptimizer, CollateralQualityScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RATING_MAP = {"AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "BBB": 7}


@pytest.fixture()
def synthetic_collateral_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 30
    ratings = rng.choice(list(RATING_MAP.keys()), size=n)
    return pd.DataFrame(
        {
            "collateral_id": [f"BOND_{i:03d}" for i in range(n)],
            "asset_type": rng.choice(["Govt", "Corp", "Agency"], size=n),
            "haircut": rng.uniform(0.01, 0.15, size=n),
            "duration": rng.uniform(0.5, 10.0, size=n),
            "credit_rating": ratings,
            "market_value": rng.uniform(1e6, 50e6, size=n),
            "financing_cost_bps": rng.uniform(10, 120, size=n),
            "is_eligible": rng.choice([True, False], p=[0.8, 0.2], size=n),
        }
    )


@pytest.fixture()
def model_config() -> dict:
    return {
        "model": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "quality_score_threshold": 0.5,
        },
        "optimization": {
            "max_concentration_pct": 0.30,
            "min_quality_score": 0.4,
            "max_haircut": 0.12,
            "solver": "ECOS",
            "cost_weight": 1.0,
            "quality_weight": 0.5,
        },
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestCollateralFeatureEngineer:
    def test_credit_rating_numeric_added(
        self, synthetic_collateral_df: pd.DataFrame
    ) -> None:
        eng = CollateralFeatureEngineer({})
        result = eng.add_credit_rating_numeric(synthetic_collateral_df.copy())
        assert "credit_rating_numeric" in result.columns
        assert result["credit_rating_numeric"].between(1, 10).all()

    def test_build_features_returns_dataframe(
        self, synthetic_collateral_df: pd.DataFrame
    ) -> None:
        eng = CollateralFeatureEngineer({})
        result = eng.build_features(synthetic_collateral_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(synthetic_collateral_df)


# ---------------------------------------------------------------------------
# Quality scorer
# ---------------------------------------------------------------------------

class TestCollateralQualityScorer:
    def test_output_shape(
        self, synthetic_collateral_df: pd.DataFrame, model_config: dict
    ) -> None:
        scorer = CollateralQualityScorer(model_config["model"])
        feature_cols = ["haircut", "duration", "financing_cost_bps"]
        X = synthetic_collateral_df[feature_cols]
        y = synthetic_collateral_df["is_eligible"].astype(int)
        scorer.fit(X, y)
        scores = scorer.score(X)
        assert scores.shape == (len(X),)

    def test_scores_in_unit_interval(
        self, synthetic_collateral_df: pd.DataFrame, model_config: dict
    ) -> None:
        scorer = CollateralQualityScorer(model_config["model"])
        feature_cols = ["haircut", "duration", "financing_cost_bps"]
        X = synthetic_collateral_df[feature_cols]
        y = synthetic_collateral_df["is_eligible"].astype(int)
        scorer.fit(X, y)
        scores = scorer.score(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


# ---------------------------------------------------------------------------
# CVXPY optimizer
# ---------------------------------------------------------------------------

class TestCollateralOptimizer:
    def test_optimization_feasible(
        self, synthetic_collateral_df: pd.DataFrame, model_config: dict
    ) -> None:
        optimizer = CollateralOptimizer(model_config["optimization"])
        # Use eligible assets only to ensure feasibility
        eligible = synthetic_collateral_df[synthetic_collateral_df["is_eligible"]].copy()
        if len(eligible) < 3:
            pytest.skip("Not enough eligible assets for this random seed.")
        quality_scores = np.full(len(eligible), 0.7)
        result = optimizer.optimize(eligible, quality_scores, required_value=1e7)
        assert isinstance(result, pd.DataFrame)
        assert "allocation_weight" in result.columns

    def test_weights_sum_to_one(
        self, synthetic_collateral_df: pd.DataFrame, model_config: dict
    ) -> None:
        optimizer = CollateralOptimizer(model_config["optimization"])
        eligible = synthetic_collateral_df[synthetic_collateral_df["is_eligible"]].copy()
        if len(eligible) < 3:
            pytest.skip("Not enough eligible assets.")
        quality_scores = np.full(len(eligible), 0.75)
        result = optimizer.optimize(eligible, quality_scores, required_value=1e7)
        total_weight = result["allocation_weight"].sum()
        assert abs(total_weight - 1.0) < 1e-4

    def test_recommendations_generated(
        self, synthetic_collateral_df: pd.DataFrame, model_config: dict
    ) -> None:
        optimizer = CollateralOptimizer(model_config["optimization"])
        eligible = synthetic_collateral_df[synthetic_collateral_df["is_eligible"]].copy()
        if len(eligible) < 3:
            pytest.skip("Not enough eligible assets.")
        quality_scores = np.full(len(eligible), 0.7)
        optimized = optimizer.optimize(eligible, quality_scores, required_value=1e7)
        recs = optimizer.generate_recommendations(optimized)
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------

class TestCollateralAlertFactory:
    def test_alert_created_above_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "collateral_id": ["BOND_001", "BOND_002"],
                "asset_type": ["Govt", "Corp"],
                "quality_score": [0.3, 0.8],
                "cost_savings_bps": [25.0, 5.0],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        config = {"alerts": {"cost_savings_threshold_bps": 15}}
        factory = CollateralAlertFactory()
        alerts = factory.from_recommendations(df, config)
        # Only BOND_001 has savings above threshold
        assert len(alerts) == 1
        assert alerts[0].collateral_id == "BOND_001"

    def test_no_alerts_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "collateral_id": ["BOND_003"],
                "asset_type": ["Govt"],
                "quality_score": [0.9],
                "cost_savings_bps": [3.0],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        config = {"alerts": {"cost_savings_threshold_bps": 15}}
        factory = CollateralAlertFactory()
        alerts = factory.from_recommendations(df, config)
        assert len(alerts) == 0
