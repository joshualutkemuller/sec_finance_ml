"""Unit tests for the Repo & Financing Cost Forecaster.

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alerts import ForecastAlert, ForecastAlertFactory, ForecastAlertType
from feature_engineering import FinancingCostFeatureEngineer
from model import MultiHorizonForecaster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_rate_df() -> pd.DataFrame:
    """Synthetic daily repo rate time series."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 530 + np.cumsum(rng.normal(0, 2, size=n))  # bps, drifting around SOFR + spread
    return pd.DataFrame(
        {
            "instrument_type": ["GC_REPO"] * n,
            "date": dates,
            "rate_bps": base,
            "sofr": base - rng.uniform(5, 20, size=n),
            "fed_funds": base - rng.uniform(8, 25, size=n),
            "notional": rng.uniform(100e6, 500e6, size=n),
        }
    )


@pytest.fixture()
def model_config() -> dict:
    return {
        "horizons": [1, 3, 5],
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "random_state": 42,
        "min_child_samples": 5,
    }


@pytest.fixture()
def feature_config() -> dict:
    return {
        "features": {
            "lag_windows": [1, 3, 5],
            "rolling_windows": [5, 10],
            "use_calendar_features": True,
            "use_macro_features": True,
        },
        "data": {"rate_col": "rate_bps"},
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFinancingCostFeatureEngineer:
    def test_lag_features_created(
        self, synthetic_rate_df: pd.DataFrame, feature_config: dict
    ) -> None:
        eng = FinancingCostFeatureEngineer(feature_config)
        result = eng.add_lag_features(synthetic_rate_df.copy(), "rate_bps", [1, 3, 5])
        for lag in [1, 3, 5]:
            assert f"rate_bps_lag_{lag}" in result.columns

    def test_calendar_features_created(
        self, synthetic_rate_df: pd.DataFrame, feature_config: dict
    ) -> None:
        eng = FinancingCostFeatureEngineer(feature_config)
        result = eng.add_calendar_features(synthetic_rate_df.copy())
        for col in ("day_of_week", "is_month_end", "is_quarter_end"):
            assert col in result.columns

    def test_is_month_end_boolean(
        self, synthetic_rate_df: pd.DataFrame, feature_config: dict
    ) -> None:
        eng = FinancingCostFeatureEngineer(feature_config)
        result = eng.add_calendar_features(synthetic_rate_df.copy())
        assert result["is_month_end"].dtype in (bool, np.dtype("bool"), int, np.dtype("int64"))

    def test_rolling_stats_created(
        self, synthetic_rate_df: pd.DataFrame, feature_config: dict
    ) -> None:
        eng = FinancingCostFeatureEngineer(feature_config)
        result = eng.add_rolling_stats(synthetic_rate_df.copy(), "rate_bps", [5, 10])
        assert "rate_bps_roll_mean_5" in result.columns
        assert "rate_bps_roll_std_10" in result.columns

    def test_build_features_returns_tuple(
        self, synthetic_rate_df: pd.DataFrame, feature_config: dict
    ) -> None:
        eng = FinancingCostFeatureEngineer(feature_config)
        X, y = eng.build_features(synthetic_rate_df.copy(), "rate_bps")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)


# ---------------------------------------------------------------------------
# Multi-horizon forecaster
# ---------------------------------------------------------------------------

class TestMultiHorizonForecaster:
    def _make_xy(
        self, n: int = 250
    ) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
        rng = np.random.default_rng(10)
        X = pd.DataFrame(
            rng.normal(size=(n, 10)),
            columns=[f"feat_{i}" for i in range(10)],
        )
        y_dict = {
            1: pd.Series(rng.normal(530, 5, size=n)),
            3: pd.Series(rng.normal(530, 7, size=n)),
            5: pd.Series(rng.normal(530, 9, size=n)),
        }
        return X, y_dict

    def test_fit_and_predict_output_keys(self, model_config: dict) -> None:
        forecaster = MultiHorizonForecaster(model_config)
        X, y_dict = self._make_xy()
        forecaster.fit(X, y_dict)
        preds = forecaster.predict(X.head(5))
        for h in [1, 3, 5]:
            assert h in preds
            assert len(preds[h]) == 5

    def test_predict_df_columns(self, model_config: dict) -> None:
        forecaster = MultiHorizonForecaster(model_config)
        X, y_dict = self._make_xy()
        forecaster.fit(X, y_dict)
        base_df = pd.DataFrame({"instrument_type": ["GC_REPO"] * len(X)})
        result = forecaster.predict_df(X, base_df)
        assert isinstance(result, pd.DataFrame)
        for col in ("forecast_1d", "forecast_3d", "forecast_5d"):
            assert col in result.columns

    def test_evaluate_returns_mae_rmse(self, model_config: dict) -> None:
        forecaster = MultiHorizonForecaster(model_config)
        X, y_dict = self._make_xy()
        forecaster.fit(X, y_dict)
        metrics = forecaster.evaluate(X, y_dict)
        for h in [1, 3, 5]:
            assert h in metrics
            assert "mae" in metrics[h]
            assert "rmse" in metrics[h]
            assert metrics[h]["mae"] >= 0
            assert metrics[h]["rmse"] >= 0

    def test_horizon_target_shift(self, model_config: dict) -> None:
        forecaster = MultiHorizonForecaster(model_config)
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"rate_bps": s})
        shifted = forecaster._build_horizon_target(df, "rate_bps", horizon=2)
        # Shifted by 2: index 0 → value 3, index 1 → value 4, last 2 are NaN
        assert shifted.iloc[0] == pytest.approx(3.0)
        assert pd.isna(shifted.iloc[-1])


# ---------------------------------------------------------------------------
# Alert factory
# ---------------------------------------------------------------------------

class TestForecastAlertFactory:
    def test_budget_breach_alert_created(self) -> None:
        forecast_df = pd.DataFrame(
            {
                "instrument_type": ["GC_REPO"],
                "current_rate_bps": [530.0],
                "forecast_1d_bps": [560.0],
                "forecast_3d_bps": [570.0],
                "forecast_5d_bps": [580.0],
                "timestamp": [pd.Timestamp("2024-06-01")],
            }
        )
        budget_df = pd.DataFrame(
            {"instrument_type": ["GC_REPO"], "budget_rate_bps": [540.0]}
        )
        config = {"alerts": {"spike_threshold_bps": 50, "budget_breach_threshold_bps": 10}}
        factory = ForecastAlertFactory()
        alerts = factory.from_forecasts(forecast_df, budget_df, config)
        breach_alerts = [a for a in alerts if a.alert_type == ForecastAlertType.BUDGET_BREACH]
        assert len(breach_alerts) > 0

    def test_spike_alert_created(self) -> None:
        forecast_df = pd.DataFrame(
            {
                "instrument_type": ["GC_REPO"],
                "current_rate_bps": [500.0],
                "forecast_1d_bps": [560.0],  # 60 bps spike
                "forecast_3d_bps": [555.0],
                "forecast_5d_bps": [550.0],
                "timestamp": [pd.Timestamp("2024-06-01")],
            }
        )
        budget_df = pd.DataFrame(
            {"instrument_type": ["GC_REPO"], "budget_rate_bps": [600.0]}
        )
        config = {"alerts": {"spike_threshold_bps": 50, "budget_breach_threshold_bps": 100}}
        factory = ForecastAlertFactory()
        alerts = factory.from_forecasts(forecast_df, budget_df, config)
        spike_alerts = [a for a in alerts if a.alert_type == ForecastAlertType.RATE_SPIKE]
        assert len(spike_alerts) >= 1

    def test_no_alerts_within_tolerance(self) -> None:
        forecast_df = pd.DataFrame(
            {
                "instrument_type": ["GC_REPO"],
                "current_rate_bps": [530.0],
                "forecast_1d_bps": [532.0],
                "forecast_3d_bps": [533.0],
                "forecast_5d_bps": [534.0],
                "timestamp": [pd.Timestamp("2024-06-01")],
            }
        )
        budget_df = pd.DataFrame(
            {"instrument_type": ["GC_REPO"], "budget_rate_bps": [550.0]}
        )
        config = {"alerts": {"spike_threshold_bps": 20, "budget_breach_threshold_bps": 10}}
        factory = ForecastAlertFactory()
        alerts = factory.from_forecasts(forecast_df, budget_df, config)
        assert len(alerts) == 0

    def test_forecast_alert_message_populated(self) -> None:
        alert = ForecastAlert(
            instrument_type="GC_REPO",
            current_rate_bps=530.0,
            forecast_bps=560.0,
            horizon_days=1,
            budget_bps=545.0,
            breach_bps=15.0,
            alert_type=ForecastAlertType.BUDGET_BREACH,
        )
        assert "GC_REPO" in alert.message
        assert "BUDGET_BREACH" in alert.message
