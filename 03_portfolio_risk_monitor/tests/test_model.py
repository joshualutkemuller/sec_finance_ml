"""Unit tests for the Portfolio Concentration Risk Early Warning Monitor.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alerts import RiskAlertFactory
from feature_engineering import ConcentrationFeatureEngineer
from model import RandomForestRiskScorer, RiskLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def exposure_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "portfolio_id": [f"PORT_{i % 5:02d}" for i in range(n)],
            "sector": rng.choice(["Tech", "Financials", "Energy", "Healthcare"], size=n),
            "counterparty": rng.choice(["CP_A", "CP_B", "CP_C", "CP_D"], size=n),
            "market_value": rng.uniform(1e5, 5e7, size=n),
            "notional": rng.uniform(1e5, 5e7, size=n),
            "liquidity_score": rng.uniform(0.1, 1.0, size=n),
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


@pytest.fixture()
def model_config() -> dict:
    return {
        "n_estimators": 30,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "random_state": 42,
        "risk_score_threshold_medium": 0.3,
        "risk_score_threshold_high": 0.6,
        "risk_score_threshold_critical": 0.85,
    }


# ---------------------------------------------------------------------------
# HHI
# ---------------------------------------------------------------------------

class TestHHIComputation:
    def test_monopoly_hhi_equals_one(self, exposure_df: pd.DataFrame) -> None:
        eng = ConcentrationFeatureEngineer({})
        df_single = exposure_df.copy()
        df_single["market_value"] = 0.0
        df_single.loc[0, "market_value"] = 1.0
        hhi = eng.compute_hhi(df_single, "market_value")
        assert abs(hhi - 1.0) < 1e-6

    def test_uniform_hhi(self, exposure_df: pd.DataFrame) -> None:
        """Equal-weight portfolio → HHI = 1/n."""
        eng = ConcentrationFeatureEngineer({})
        df_uniform = pd.DataFrame({"market_value": [1.0] * 10})
        hhi = eng.compute_hhi(df_uniform, "market_value")
        assert abs(hhi - 0.1) < 1e-6

    def test_hhi_between_zero_and_one(self, exposure_df: pd.DataFrame) -> None:
        eng = ConcentrationFeatureEngineer({})
        hhi = eng.compute_hhi(exposure_df, "market_value")
        assert 0.0 < hhi <= 1.0


# ---------------------------------------------------------------------------
# Feature engineer
# ---------------------------------------------------------------------------

class TestConcentrationFeatureEngineer:
    def test_sector_concentration_added(self, exposure_df: pd.DataFrame) -> None:
        eng = ConcentrationFeatureEngineer({})
        result = eng.add_sector_concentration(exposure_df.copy())
        assert "sector_hhi" in result.columns or "sector_weight" in result.columns

    def test_build_features_returns_dataframe(self, exposure_df: pd.DataFrame) -> None:
        eng = ConcentrationFeatureEngineer({})
        result = eng.build_features(exposure_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Risk scorer
# ---------------------------------------------------------------------------

class TestRandomForestRiskScorer:
    def test_classify_returns_risk_levels(self, model_config: dict) -> None:
        scorer = RandomForestRiskScorer(model_config)
        scores = np.array([0.1, 0.4, 0.7, 0.9])
        levels = scorer.classify(scores)
        assert len(levels) == 4
        assert all(isinstance(l, RiskLevel) for l in levels)

    def test_classify_thresholds(self, model_config: dict) -> None:
        scorer = RandomForestRiskScorer(model_config)
        levels = scorer.classify(np.array([0.0, 0.5, 0.75, 0.95]))
        assert levels[0] == RiskLevel.LOW
        assert levels[1] == RiskLevel.MEDIUM
        assert levels[2] == RiskLevel.HIGH
        assert levels[3] == RiskLevel.CRITICAL

    def test_fit_and_score_shape(self, model_config: dict) -> None:
        rng = np.random.default_rng(99)
        X = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.integers(0, 2, size=120))
        scorer = RandomForestRiskScorer(model_config)
        scorer.fit(X, y)
        scores = scorer.score(X)
        assert scores.shape == (120,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------

class TestRiskAlertFactory:
    def test_alerts_created_for_high_risk(self) -> None:
        df = pd.DataFrame(
            {
                "portfolio_id": ["PORT_01", "PORT_02", "PORT_03"],
                "sector": ["Tech", "Energy", "Financials"],
                "counterparty": ["CP_A", "CP_B", "CP_C"],
                "concentration_score": [0.2, 0.72, 0.91],
                "risk_level": ["LOW", "HIGH", "CRITICAL"],
                "alert_flag": [False, True, True],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        risk_levels = [RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.CRITICAL]
        config = {"alerts": {}}
        factory = RiskAlertFactory()
        alerts = factory.from_scores(df, risk_levels, None, config)
        assert len(alerts) == 2

    def test_no_alerts_for_low_risk(self) -> None:
        df = pd.DataFrame(
            {
                "portfolio_id": ["PORT_01"],
                "sector": ["Tech"],
                "counterparty": ["CP_A"],
                "concentration_score": [0.1],
                "risk_level": ["LOW"],
                "alert_flag": [False],
                "timestamp": pd.Timestamp("2024-06-01"),
            }
        )
        factory = RiskAlertFactory()
        alerts = factory.from_scores(df, [RiskLevel.LOW], None, {})
        assert len(alerts) == 0
