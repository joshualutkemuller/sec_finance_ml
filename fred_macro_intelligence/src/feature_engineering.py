"""
Feature Engineering — Constructs the full macro feature set from raw FRED series.

Output features are grouped into:
  1. Rate level features
  2. Yield curve shape features
  3. Spread features (z-scored)
  4. Liquidity features
  5. Calendar features
  6. Composite signal indices
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import zscore as sp_zscore

logger = logging.getLogger(__name__)


class MacroFeatureEngineer:
    """Build the full macro feature matrix from raw FRED DataFrame."""

    def __init__(self, config: dict) -> None:
        self.windows = config["features"].get("rolling_windows", [5, 10, 20, 60, 120])
        self.zscore_windows = config["features"].get("zscore_windows", [60, 120])
        self.stress_weights = config["features"].get("funding_stress_weights", {})
        self.quality_weights = config["features"].get("collateral_quality_weights", {})

    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature matrix from raw FRED series.

        Args:
            raw: DataFrame with one column per FRED series (friendly names as columns).

        Returns:
            Feature matrix, same date index, many derived columns.
        """
        df = raw.copy()
        df = self._rate_level_features(df)
        df = self._curve_shape_features(df)
        df = self._spread_features(df)
        df = self._liquidity_features(df)
        df = self._calendar_features(df)
        df = self._composite_signals(df)

        # Drop raw series that are now superseded by features
        feature_df = df.select_dtypes(include=[np.number])
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(how="all")

        logger.info("Feature matrix built: %d rows × %d features", *feature_df.shape)
        return feature_df

    # ── 1. Rate level features ─────────────────────────────────────────────

    def _rate_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rate_cols = ["sofr", "fed_funds", "prime_rate", "iorb"]
        for col in rate_cols:
            if col not in df.columns:
                continue
            for w in self.windows:
                df[f"{col}_ma{w}"] = df[col].rolling(w).mean()
                df[f"{col}_std{w}"] = df[col].rolling(w).std()
                df[f"{col}_z{w}"] = (df[col] - df[f"{col}_ma{w}"]) / (df[f"{col}_std{w}"] + 1e-8)
                df[f"{col}_mom{w}"] = df[col].diff(w)

        # Policy rate spreads
        if "sofr" in df.columns and "fed_funds" in df.columns:
            df["fed_sofr_spread"] = df["fed_funds"] - df["sofr"]
        if "prime_rate" in df.columns and "sofr" in df.columns:
            df["prime_sofr_spread"] = df["prime_rate"] - df["sofr"]
        if "iorb" in df.columns and "sofr" in df.columns:
            df["iorb_sofr_spread"] = df["iorb"] - df["sofr"]

        return df

    # ── 2. Yield curve shape features ─────────────────────────────────────

    def _curve_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        tenors = {
            "t_1mo": df.get("t_1mo"),
            "t_3mo": df.get("t_3mo"),
            "t_6mo": df.get("t_6mo"),
            "t_1yr": df.get("t_1yr"),
            "t_2yr": df.get("t_2yr"),
            "t_5yr": df.get("t_5yr"),
            "t_10yr": df.get("t_10yr"),
            "t_30yr": df.get("t_30yr"),
        }
        available = {k: v for k, v in tenors.items() if v is not None}

        # Classic spread measures
        if "t_10yr" in available and "t_2yr" in available:
            df["curve_10y2y"] = df["t_10yr"] - df["t_2yr"]
            df["curve_10y2y_inverted"] = (df["curve_10y2y"] < 0).astype(int)
            df["days_since_inversion"] = (
                df["curve_10y2y_inverted"]
                .groupby((df["curve_10y2y_inverted"] != df["curve_10y2y_inverted"].shift()).cumsum())
                .cumcount()
                * df["curve_10y2y_inverted"]
            )

        if "t_10yr" in available and "t_3mo" in available:
            df["curve_10y3m"] = df["t_10yr"] - df["t_3mo"]

        if "t_5yr" in available and "t_2yr" in available:
            df["curve_5y2y"] = df["t_5yr"] - df["t_2yr"]

        # Butterfly curvature: 2*5y - (2y + 10y)
        if all(k in available for k in ["t_2yr", "t_5yr", "t_10yr"]):
            df["curve_butterfly"] = 2 * df["t_5yr"] - (df["t_2yr"] + df["t_10yr"])

        # Level: average of key tenors
        level_tenors = [available[k] for k in ["t_2yr", "t_5yr", "t_10yr"] if k in available]
        if level_tenors:
            df["curve_level"] = pd.concat(level_tenors, axis=1).mean(axis=1)

        # Rolling momentum on slope
        for slope_col in ["curve_10y2y", "curve_10y3m"]:
            if slope_col in df.columns:
                for w in [5, 10, 20]:
                    df[f"{slope_col}_mom{w}d"] = df[slope_col].diff(w)
                    df[f"{slope_col}_z60"] = self._rolling_z(df[slope_col], 60)

        return df

    # ── 3. Spread features ─────────────────────────────────────────────────

    def _spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        spread_cols = ["hy_oas", "ig_oas", "ted_spread", "mortgage_30y"]

        for col in spread_cols:
            if col not in df.columns:
                continue
            for w in self.zscore_windows:
                df[f"{col}_z{w}"] = self._rolling_z(df[col], w)
            for w in [5, 10, 20]:
                df[f"{col}_mom{w}d"] = df[col].diff(w)

        # Spread differentials
        if "hy_oas" in df.columns and "ig_oas" in df.columns:
            df["hy_ig_diff"] = df["hy_oas"] - df["ig_oas"]
            df["hy_ig_diff_z60"] = self._rolling_z(df["hy_ig_diff"], 60)

        # MBS basis
        if "mortgage_30y" in df.columns and "t_10yr" in df.columns:
            df["mbs_basis"] = df["mortgage_30y"] - df["t_10yr"]
            df["mbs_basis_z60"] = self._rolling_z(df["mbs_basis"], 60)

        return df

    # ── 4. Liquidity features ──────────────────────────────────────────────

    def _liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "vix" in df.columns:
            for w in [10, 20, 60]:
                df[f"vix_z{w}"] = self._rolling_z(df["vix"], w)
            df["vix_mom5d"] = df["vix"].diff(5)
            df["vix_regime"] = pd.cut(
                df["vix"],
                bins=[0, 15, 20, 30, 40, 999],
                labels=["Low", "Normal", "Elevated", "High", "Extreme"],
            ).cat.codes

        if "rrp_balance" in df.columns:
            df["rrp_balance_bn"] = df["rrp_balance"] / 1e9
            for w in [5, 10, 20]:
                df[f"rrp_change{w}d"] = df["rrp_balance"].diff(w)
                df[f"rrp_pct_change{w}d"] = df["rrp_balance"].pct_change(w)
            df["rrp_z60"] = self._rolling_z(df["rrp_balance"], 60)

        if "reserve_balances" in df.columns:
            df["reserves_yoy"] = df["reserve_balances"].pct_change(252)
            df["reserves_z60"] = self._rolling_z(df["reserve_balances"], 60)

        if "m2" in df.columns:
            df["m2_yoy"] = df["m2"].pct_change(12)  # m2 is monthly
            df["m2_z60"] = self._rolling_z(df["m2"].ffill(), 60)

        return df

    # ── 5. Calendar features ───────────────────────────────────────────────

    def _calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        df["day_of_week"] = idx.dayofweek                               # 0=Mon
        df["month"] = idx.month
        df["quarter"] = idx.quarter

        # Month-end: last 3 business days of month
        df["month_end_flag"] = self._rolling_end_flag(idx, period="M", n_days=3)
        # Quarter-end: last 5 business days
        df["quarter_end_flag"] = self._rolling_end_flag(idx, period="Q", n_days=5)
        # Year-end: last 10 business days
        df["year_end_flag"] = self._rolling_end_flag(idx, period="Y", n_days=10)

        return df

    # ── 6. Composite signal indices ────────────────────────────────────────

    def _composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct composite stress and quality indices from weighted z-scores."""
        stress_components = {
            "ted_spread_z60": self.stress_weights.get("ted_spread_z", 0.30),
            "vix_z20": self.stress_weights.get("vix_z", 0.25),
            "hy_oas_z60": self.stress_weights.get("hy_oas_z", 0.30),
            "rrp_change5d": self.stress_weights.get("rrp_change_z", 0.15),
        }
        available_stress = {k: v for k, v in stress_components.items() if k in df.columns}
        if available_stress:
            total_weight = sum(available_stress.values())
            df["funding_stress_index"] = sum(
                df[k] * (w / total_weight) for k, w in available_stress.items()
            )

        quality_components = {
            "ig_oas_z60": self.quality_weights.get("ig_oas_z_inv", 0.40),
            "curve_level": self.quality_weights.get("curve_level_inv", 0.30),
            "vix_z20": self.quality_weights.get("vix_z_inv", 0.30),
        }
        available_quality = {k: v for k, v in quality_components.items() if k in df.columns}
        if available_quality:
            total_weight = sum(available_quality.values())
            # Inverted — higher score = better quality environment
            df["collateral_quality_index"] = -sum(
                df[k] * (w / total_weight) for k, w in available_quality.items()
            )

        # Macro momentum: 5-day z-score of key rate changes
        momentum_cols = [c for c in df.columns if c.endswith("_mom5d")]
        if momentum_cols:
            mom_matrix = df[momentum_cols].dropna(how="all")
            if not mom_matrix.empty:
                df["macro_momentum"] = mom_matrix.mean(axis=1)

        return df

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_z(series: pd.Series, window: int) -> pd.Series:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return (series - mu) / (sigma + 1e-8)

    @staticmethod
    def _rolling_end_flag(idx: pd.DatetimeIndex, period: str, n_days: int) -> pd.Series:
        """Flag the last n_days business days of each period (M/Q/Y)."""
        if period == "M":
            period_end = idx.to_period("M").to_timestamp("M")
        elif period == "Q":
            period_end = idx.to_period("Q").to_timestamp("Q")
        else:
            period_end = idx.to_period("Y").to_timestamp("Y")

        bdays_to_end = pd.Series(
            [(period_end[i] - idx[i]).days for i in range(len(idx))],
            index=idx,
            dtype=float,
        )
        # Approximate: flag if within n_days calendar days (rough business day proxy)
        return (bdays_to_end <= n_days * 1.4).astype(int)
