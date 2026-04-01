"""
Feature Engineering — Enhanced feature set for the securities lending optimizer.

Builds three feature matrices:
  1. Demand features   — for DemandForecaster training/inference
  2. Recall features   — for RecallRiskClassifier training/inference
  3. Collateral features — for MultiObjectiveOptimizer
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = [5, 10, 20, 60]


class FeatureEngineer:
    """Build feature matrices for each model in the lending optimizer."""

    def __init__(self, config: dict) -> None:
        self.windows = config["features"].get("rolling_windows", ROLLING_WINDOWS)
        self.demand_horizons = config["features"].get("demand_horizons", [1, 3, 5])
        self.recall_horizon = config["features"].get("recall_horizon_days", 5)

    # ── 1. Demand features ────────────────────────────────────────────────────

    def build_demand_features(
        self,
        loan_book: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
        """Build demand feature matrix and forward-shifted targets.

        Returns:
            (X, y_dict) where y_dict maps horizon → target series.
        """
        df = loan_book.copy().sort_values(["ticker", "date"])
        df = df.merge(market_data, on="date", how="left")

        features = []
        groups = df.groupby("ticker")

        for ticker, grp in groups:
            grp = grp.sort_values("date").copy()
            grp["borrow_volume"] = grp.get("shares_on_loan", grp.get("notional", 0.0))
            grp["borrow_rate"] = grp.get("borrow_rate", grp.get("rate", 0.0))

            for w in self.windows:
                grp[f"vol_rolling_mean_{w}d"] = grp["borrow_volume"].rolling(w).mean()
                grp[f"vol_rolling_std_{w}d"] = grp["borrow_volume"].rolling(w).std()
                grp[f"rate_rolling_mean_{w}d"] = grp["borrow_rate"].rolling(w).mean()
                grp[f"rate_z_{w}d"] = (
                    (grp["borrow_rate"] - grp[f"rate_rolling_mean_{w}d"])
                    / (grp[f"vol_rolling_std_{w}d"] + 1e-8)
                )

            grp["vol_mom_1d"] = grp["borrow_volume"].diff(1)
            grp["vol_mom_5d"] = grp["borrow_volume"].diff(5)
            grp["short_interest_ratio"] = grp.get(
                "short_interest", pd.Series(np.nan, index=grp.index)
            )
            features.append(grp)

        full = pd.concat(features).dropna()

        # Forward-shift targets to avoid lookahead bias
        y_dict: dict[int, pd.Series] = {}
        for h in self.demand_horizons:
            y_dict[h] = full.groupby("ticker")["borrow_volume"].shift(-h)

        feature_cols = [c for c in full.columns if c not in
                        ["ticker", "date", "borrow_volume", "borrow_rate"] +
                        [f"target_{h}d" for h in self.demand_horizons]]
        X = full[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
        logger.info("Demand feature matrix: %s", X.shape)
        return X, y_dict

    # ── 2. Recall features ────────────────────────────────────────────────────

    def build_recall_features(
        self,
        loan_book: pd.DataFrame,
        counterparties: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build recall risk features and binary target.

        Returns:
            (X, y) where y=1 if recall occurred within recall_horizon days.
        """
        df = loan_book.copy().sort_values(["counterparty_id", "ticker", "date"])
        df = df.merge(counterparties, on="counterparty_id", how="left")
        df = df.merge(market_data, on="date", how="left")

        # Counterparty-level aggregates
        cp_agg = df.groupby("counterparty_id").agg(
            cp_recall_rate=("is_recall", "mean"),
            cp_avg_term=("trade_term_days", "mean"),
            cp_hhi=("asset_class", lambda s: (s.value_counts(normalize=True) ** 2).sum()),
        ).rename_axis("counterparty_id").reset_index()
        df = df.merge(cp_agg, on="counterparty_id", how="left")

        # Market stress proxy
        df["market_stress"] = (
            df.get("vix", pd.Series(0.0, index=df.index)).fillna(0.0) / 30.0
        )

        feature_cols = [
            "cp_recall_rate", "cp_avg_term", "cp_hhi", "market_stress",
            "trade_term_days", "borrow_rate",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)

        if "is_recall" in df.columns:
            # Forward-shift to define target: did recall happen within horizon?
            y = df.groupby(["counterparty_id", "ticker"])["is_recall"].transform(
                lambda s: s.rolling(self.recall_horizon).max().shift(-self.recall_horizon)
            ).fillna(0).astype(int)
        else:
            y = pd.Series(0, index=df.index)

        logger.info("Recall feature matrix: %s, recall_rate=%.3f", X.shape, y.mean())
        return X, y

    # ── 3. Collateral features ────────────────────────────────────────────────

    def build_collateral_features(
        self,
        collateral_pool: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the collateral feature matrix for the optimizer.

        Returns DataFrame with columns required by MultiObjectiveOptimizer.
        """
        df = collateral_pool.copy()
        df = df.merge(market_data.tail(1), on="date", how="cross") if "date" in market_data else df

        # Ensure required optimizer columns exist
        required = ["financing_cost", "fee_rate", "capital_charge_est",
                    "haircut", "quality_score", "asset_class"]
        for col in required:
            if col not in df.columns:
                logger.warning("Column '%s' missing from collateral pool — filling with 0", col)
                df[col] = 0.0

        df["quality_score"] = df["quality_score"].clip(0.0, 1.0)
        df["haircut"] = df["haircut"].clip(0.0, 0.5)
        df["financing_cost"] = df["financing_cost"].clip(0.0, None)

        logger.info("Collateral feature matrix: %s", df.shape)
        return df
