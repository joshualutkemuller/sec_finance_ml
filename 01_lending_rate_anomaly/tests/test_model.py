"""Unit tests for the Securities Lending Rate Anomaly Detector.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alerts import AlertFactory
from feature_engineering import LendingRateFeatureEngineer
from model import CombinedAnomalyDetector, IsolationForestDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_rate_df() -> pd.DataFrame:
    """200-row lending rate DataFrame with required columns."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * 200,
            "date": dates,
            "rate": rng.normal(loc=150, scale=20, size=200).clip(10),  # bps
            "benchmark_rate": rng.normal(loc=120, scale=15, size=200).clip(5),
        }
    )


@pytest.fixture()
def model_config() -> dict:
    return {
        "isolation_forest": {
            "contamination": 0.05,
            "n_estimators": 50,
            "random_state": 42,
        },
        "lstm": {
            "hidden_size": 16,
            "num_layers": 1,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "sequence_length": 10,
            "threshold_percentile": 95,
        },
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_rolling_stats_added(self, synthetic_rate_df: pd.DataFrame) -> None:
        cfg = {"data": {"rate_col": "rate"}}
        eng = LendingRateFeatureEngineer(cfg)
        result = eng.add_rolling_stats(synthetic_rate_df.copy(), windows=[5, 10])
        expected_cols = {
            "rate_roll_mean_5", "rate_roll_std_5", "rate_zscore_5",
            "rate_roll_mean_10", "rate_roll_std_10", "rate_zscore_10",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_spread_features_added(self, synthetic_rate_df: pd.DataFrame) -> None:
        cfg = {"data": {"rate_col": "rate", "benchmark_col": "benchmark_rate"}}
        eng = LendingRateFeatureEngineer(cfg)
        result = eng.add_spread_features(synthetic_rate_df.copy())
        assert "spread_vs_benchmark" in result.columns

    def test_build_features_no_nans_after_fillna(
        self, synthetic_rate_df: pd.DataFrame
    ) -> None:
        cfg = {"data": {"rate_col": "rate", "benchmark_col": "benchmark_rate"}}
        eng = LendingRateFeatureEngineer(cfg)
        result = eng.build_features(synthetic_rate_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

class TestIsolationForestDetector:
    def test_scores_shape(self, model_config: dict) -> None:
        detector = IsolationForestDetector(model_config["isolation_forest"])
        X = np.random.default_rng(0).normal(size=(100, 5))
        detector.fit(X)
        scores = detector.score(X)
        assert scores.shape == (100,)

    def test_scores_in_range(self, model_config: dict) -> None:
        detector = IsolationForestDetector(model_config["isolation_forest"])
        X = np.random.default_rng(1).normal(size=(100, 5))
        detector.fit(X)
        scores = detector.score(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomaly_has_higher_score(self, model_config: dict) -> None:
        """Injected outliers should on average score higher than normal points."""
        rng = np.random.default_rng(7)
        X_normal = rng.normal(loc=0, scale=1, size=(200, 3))
        X_outlier = rng.normal(loc=10, scale=0.1, size=(5, 3))
        X_all = np.vstack([X_normal, X_outlier])

        detector = IsolationForestDetector(model_config["isolation_forest"])
        detector.fit(X_all)
        scores = detector.score(X_all)
        assert scores[-5:].mean() > scores[:200].mean()


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------

class TestCombinedAnomalyDetector:
    def test_predict_output_columns(self, model_config: dict) -> None:
        detector = CombinedAnomalyDetector(model_config)
        X = np.random.default_rng(2).normal(size=(80, 4))
        detector.fit(X)
        result = detector.predict(X)
        assert isinstance(result, pd.DataFrame)
        for col in ("if_score", "lstm_score", "combined_score", "is_anomaly"):
            assert col in result.columns, f"Missing column: {col}"

    def test_is_anomaly_is_boolean(self, model_config: dict) -> None:
        detector = CombinedAnomalyDetector(model_config)
        X = np.random.default_rng(3).normal(size=(80, 4))
        detector.fit(X)
        result = detector.predict(X)
        assert result["is_anomaly"].dtype == bool

    def test_row_count_matches_input(self, model_config: dict) -> None:
        detector = CombinedAnomalyDetector(model_config)
        X = np.random.default_rng(4).normal(size=(60, 4))
        detector.fit(X)
        result = detector.predict(X)
        assert len(result) == len(X)


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------

class TestAlertFactory:
    def test_alerts_created_for_anomalies(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "TSLA"],
                "rate": [200.0, 150.0, 500.0],
                "anomaly_score": [0.9, 0.3, 0.95],
                "is_anomaly": [True, False, True],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        config = {"alerts": {"anomaly_score_threshold": 0.7}}
        factory = AlertFactory()
        alerts = factory.from_scores(df, config)
        assert len(alerts) == 2  # AAPL and TSLA

    def test_no_alerts_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "rate": [150.0],
                "anomaly_score": [0.2],
                "is_anomaly": [False],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        config = {"alerts": {"anomaly_score_threshold": 0.7}}
        factory = AlertFactory()
        alerts = factory.from_scores(df, config)
        assert len(alerts) == 0
