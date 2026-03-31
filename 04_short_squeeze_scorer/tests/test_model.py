"""Unit tests for the Short Squeeze Probability Scorer.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alerts import SqueezeAlertFactory
from feature_engineering import ShortSqueezeFeatureEngineer
from model import RiskTier, XGBoostSqueezeScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_si_df() -> pd.DataFrame:
    """Synthetic merged short-interest + price DataFrame."""
    rng = np.random.default_rng(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["GME"] * n,
            "date": dates,
            "shares_short": rng.integers(10_000_000, 80_000_000, size=n),
            "shares_float": np.full(n, 200_000_000),
            "short_interest_ratio": rng.uniform(0.05, 0.40, size=n),
            "available_shares": rng.integers(500_000, 5_000_000, size=n),
            "utilization_rate": rng.uniform(0.3, 0.99, size=n),
            "borrow_rate": rng.uniform(5, 200, size=n),  # bps
            "open": rng.uniform(10, 50, size=n),
            "high": rng.uniform(12, 55, size=n),
            "low": rng.uniform(8, 45, size=n),
            "close": rng.uniform(10, 50, size=n),
            "volume": rng.integers(5_000_000, 50_000_000, size=n),
        }
    )


@pytest.fixture()
def model_config() -> dict:
    return {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "squeeze_prob_threshold_elevated": 0.30,
        "squeeze_prob_threshold_high": 0.60,
        "squeeze_prob_threshold_critical": 0.85,
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestShortSqueezeFeatureEngineer:
    def test_days_to_cover_calculated(self, synthetic_si_df: pd.DataFrame) -> None:
        eng = ShortSqueezeFeatureEngineer({})
        result = eng.add_days_to_cover(synthetic_si_df.copy())
        assert "days_to_cover" in result.columns
        assert result["days_to_cover"].gt(0).all()

    def test_rsi_in_valid_range(self, synthetic_si_df: pd.DataFrame) -> None:
        eng = ShortSqueezeFeatureEngineer({})
        result = eng.add_rsi(synthetic_si_df.copy(), window=14)
        rsi_col = "rsi_14"
        if rsi_col in result.columns:
            valid = result[rsi_col].dropna()
            assert valid.between(0, 100).all()

    def test_volume_ratio_positive(self, synthetic_si_df: pd.DataFrame) -> None:
        eng = ShortSqueezeFeatureEngineer({})
        result = eng.add_volume_ratio(synthetic_si_df.copy(), window=20)
        col = "volume_ratio_20"
        if col in result.columns:
            assert result[col].dropna().gt(0).all()

    def test_build_features_returns_dataframe(
        self, synthetic_si_df: pd.DataFrame
    ) -> None:
        eng = ShortSqueezeFeatureEngineer({"features": {"si_ratio_windows": [5],
                                                         "momentum_windows": [5],
                                                         "volume_windows": [10]}})
        result = eng.build_features(synthetic_si_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# XGBoost scorer
# ---------------------------------------------------------------------------

class TestXGBoostSqueezeScorer:
    def _make_training_data(
        self, n: int = 200
    ) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            rng.normal(size=(n, 8)),
            columns=[f"feat_{i}" for i in range(8)],
        )
        y = pd.Series(rng.integers(0, 2, size=n))
        return X, y

    def test_output_shape(self, model_config: dict) -> None:
        scorer = XGBoostSqueezeScorer(model_config)
        X, y = self._make_training_data()
        scorer.fit(X, y)
        probs = scorer.score(X)
        assert probs.shape == (len(X),)

    def test_probabilities_in_unit_interval(self, model_config: dict) -> None:
        scorer = XGBoostSqueezeScorer(model_config)
        X, y = self._make_training_data()
        scorer.fit(X, y)
        probs = scorer.score(X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_feature_importance_returned(self, model_config: dict) -> None:
        scorer = XGBoostSqueezeScorer(model_config)
        X, y = self._make_training_data()
        scorer.fit(X, y)
        fi = scorer.feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "importance" in fi.columns or len(fi.columns) >= 1


# ---------------------------------------------------------------------------
# Risk tier classification
# ---------------------------------------------------------------------------

class TestRiskTierClassification:
    def test_classification_boundaries(self, model_config: dict) -> None:
        scorer = XGBoostSqueezeScorer(model_config)
        probs = np.array([0.05, 0.35, 0.65, 0.90])
        tiers = scorer.classify(probs)
        assert tiers[0] == RiskTier.LOW
        assert tiers[1] == RiskTier.ELEVATED
        assert tiers[2] == RiskTier.HIGH
        assert tiers[3] == RiskTier.CRITICAL

    def test_all_tiers_present_in_enum(self) -> None:
        expected = {"LOW", "ELEVATED", "HIGH", "CRITICAL"}
        actual = {t.value for t in RiskTier}
        assert expected == actual


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------

class TestSqueezeAlertFactory:
    def test_alerts_for_high_probability(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["GME", "AMC", "BB"],
                "short_interest_ratio": [0.35, 0.10, 0.55],
                "days_to_cover": [8.0, 2.0, 12.0],
                "utilization_rate": [0.95, 0.40, 0.98],
                "squeeze_probability": [0.88, 0.20, 0.72],
                "risk_tier": ["CRITICAL", "LOW", "HIGH"],
                "alert_flag": [True, False, True],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        risk_tiers = [RiskTier.CRITICAL, RiskTier.LOW, RiskTier.HIGH]
        config = {"alerts": {}}
        factory = SqueezeAlertFactory()
        alerts = factory.from_scores(df, risk_tiers, config)
        assert len(alerts) == 2  # GME and BB

    def test_no_alerts_for_low_risk(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "short_interest_ratio": [0.02],
                "days_to_cover": [1.0],
                "utilization_rate": [0.10],
                "squeeze_probability": [0.05],
                "risk_tier": ["LOW"],
                "alert_flag": [False],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        factory = SqueezeAlertFactory()
        alerts = factory.from_scores(df, [RiskTier.LOW], {})
        assert len(alerts) == 0
