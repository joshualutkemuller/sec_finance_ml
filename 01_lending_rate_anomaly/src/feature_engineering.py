"""
feature_engineering.py
-----------------------
Derives a rich set of features from raw borrow-rate time series data for
use by the Isolation Forest and LSTM Autoencoder anomaly detectors.

Feature groups
--------------
1. Rolling statistics   — rolling mean, std and z-score over configurable windows.
2. Spread features      — borrow-rate spread vs a benchmark rate plus spread momentum.
3. Momentum features    — 1-day / 5-day rate of change and second-order acceleration.
4. Liquidity proxy      — rolling high-low range, coefficient of variation, and
                          an intraday volatility ratio proxy.

All methods operate in-place on a copy of the input DataFrame and return
the enriched copy.

Classes
-------
LendingRateFeatureEngineer
    Orchestrates all feature groups via the :meth:`build_features` method.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LendingRateFeatureEngineer:
    """Build features from a raw securities lending rate DataFrame.

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration dictionary.  The following keys are
        used:

        - ``config['data']['rate_col']``        — borrow-rate column name.
        - ``config['data']['benchmark_col']``   — benchmark rate column (optional).
        - ``config['data']['ticker_col']``       — ticker column name (used for
                                                   group-by operations).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})

        self.rate_col: str = data_cfg.get("rate_col", "borrow_rate")
        self.ticker_col: str = data_cfg.get("ticker_col", "ticker")
        self.benchmark_col: str | None = data_cfg.get("benchmark_col") or None

        logger.debug(
            "LendingRateFeatureEngineer initialised — rate_col=%s, benchmark_col=%s",
            self.rate_col,
            self.benchmark_col,
        )

    # ------------------------------------------------------------------
    # 1. Rolling statistics
    # ------------------------------------------------------------------

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add rolling mean, standard deviation, and z-score for each window.

        For each window *w* in ``windows`` the following columns are added:

        - ``rate_rolling_mean_{w}``  — rolling mean over *w* periods
        - ``rate_rolling_std_{w}``   — rolling standard deviation over *w* periods
        - ``rate_zscore_{w}``        — (rate − mean) / std  (NaN where std == 0)

        Rolling statistics are computed *per ticker* when the DataFrame
        contains multiple tickers (identified by ``ticker_col``).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least ``rate_col``.
        windows : list[int], optional
            List of rolling window sizes in number of rows (trading days).
            Defaults to ``[5, 10, 20]``.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with additional rolling-stat columns appended.
        """
        if windows is None:
            windows = [5, 10, 20]

        df = df.copy()

        def _rolling_for_group(group: pd.DataFrame) -> pd.DataFrame:
            for w in windows:
                mean_col = f"rate_rolling_mean_{w}"
                std_col = f"rate_rolling_std_{w}"
                zscore_col = f"rate_zscore_{w}"

                rolling = group[self.rate_col].rolling(window=w, min_periods=max(1, w // 2))
                group[mean_col] = rolling.mean()
                group[std_col] = rolling.std()

                # Avoid division by zero: where std == 0, z-score is 0
                std_safe = group[std_col].replace(0, np.nan)
                group[zscore_col] = (group[self.rate_col] - group[mean_col]) / std_safe
                group[zscore_col] = group[zscore_col].fillna(0.0)

            return group

        if self.ticker_col in df.columns and df[self.ticker_col].nunique() > 1:
            df = (
                df.groupby(self.ticker_col, group_keys=False)
                .apply(_rolling_for_group)
            )
        else:
            df = _rolling_for_group(df)

        added = [f"rate_rolling_mean_{w}" for w in windows]
        logger.debug("Rolling stats added: %s", added)
        return df

    # ------------------------------------------------------------------
    # 2. Spread features
    # ------------------------------------------------------------------

    def add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add borrow-rate spread vs a benchmark rate plus spread momentum.

        Columns added
        ~~~~~~~~~~~~~
        - ``spread_vs_benchmark``       — ``rate_col`` minus ``benchmark_col``
        - ``spread_rolling_mean_10``    — 10-period rolling mean of the spread
        - ``spread_rolling_std_10``     — 10-period rolling std of the spread
        - ``spread_zscore``             — normalised spread deviation
        - ``spread_change_1d``          — 1-period diff of spread

        If ``benchmark_col`` is not configured or not present in ``df``, a
        constant zero benchmark is assumed and the spread equals the raw rate.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with at least ``rate_col``.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with spread feature columns appended.
        """
        df = df.copy()

        if self.benchmark_col and self.benchmark_col in df.columns:
            df["spread_vs_benchmark"] = df[self.rate_col] - df[self.benchmark_col]
            logger.debug("Spread computed vs benchmark column '%s'", self.benchmark_col)
        else:
            # No benchmark available — use the raw rate as a self-relative spread
            df["spread_vs_benchmark"] = df[self.rate_col]
            logger.debug(
                "No benchmark column — spread_vs_benchmark set equal to %s",
                self.rate_col,
            )

        # Rolling stats on the spread
        spread_rolling = df["spread_vs_benchmark"].rolling(window=10, min_periods=5)
        df["spread_rolling_mean_10"] = spread_rolling.mean()
        df["spread_rolling_std_10"] = spread_rolling.std()

        std_safe = df["spread_rolling_std_10"].replace(0, np.nan)
        df["spread_zscore"] = (
            (df["spread_vs_benchmark"] - df["spread_rolling_mean_10"]) / std_safe
        ).fillna(0.0)

        # 1-day change in spread
        df["spread_change_1d"] = df["spread_vs_benchmark"].diff(1)

        return df

    # ------------------------------------------------------------------
    # 3. Momentum features
    # ------------------------------------------------------------------

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change and acceleration features.

        Columns added
        ~~~~~~~~~~~~~
        - ``rate_change_1d``    — 1-period difference: ``rate.diff(1)``
        - ``rate_change_5d``    — 5-period difference: ``rate.diff(5)``
        - ``rate_pct_change_1d``— 1-period percentage change
        - ``rate_acceleration`` — second derivative: ``rate_change_1d.diff(1)``
        - ``rate_mom_zscore``   — rolling z-score of ``rate_change_1d`` over 10 periods

        Momentum features are computed *per ticker* when multiple tickers are
        present.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with at least ``rate_col``.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with momentum feature columns appended.
        """
        df = df.copy()

        def _momentum_for_group(group: pd.DataFrame) -> pd.DataFrame:
            group["rate_change_1d"] = group[self.rate_col].diff(1)
            group["rate_change_5d"] = group[self.rate_col].diff(5)
            group["rate_pct_change_1d"] = group[self.rate_col].pct_change(1)
            group["rate_acceleration"] = group["rate_change_1d"].diff(1)

            # Rolling z-score of 1-day change
            mom_rolling = group["rate_change_1d"].rolling(window=10, min_periods=5)
            mom_mean = mom_rolling.mean()
            mom_std = mom_rolling.std().replace(0, np.nan)
            group["rate_mom_zscore"] = (
                (group["rate_change_1d"] - mom_mean) / mom_std
            ).fillna(0.0)

            return group

        if self.ticker_col in df.columns and df[self.ticker_col].nunique() > 1:
            df = (
                df.groupby(self.ticker_col, group_keys=False)
                .apply(_momentum_for_group)
            )
        else:
            df = _momentum_for_group(df)

        return df

    # ------------------------------------------------------------------
    # 4. Liquidity proxy features
    # ------------------------------------------------------------------

    def add_liquidity_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling liquidity proxy features derived from rate volatility.

        In the absence of volume or open-interest data, these features proxy
        for liquidity conditions based on rate dispersion and range.

        Columns added
        ~~~~~~~~~~~~~
        - ``rate_rolling_max_10``     — 10-period rolling maximum
        - ``rate_rolling_min_10``     — 10-period rolling minimum
        - ``rate_high_low_range``     — rolling max − rolling min (range proxy)
        - ``rate_volatility_ratio``   — rolling std / rolling mean (CoV)
                                        capped at 5 to avoid extreme values
        - ``rate_range_zscore``       — z-score of the high-low range over 20 periods

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with at least ``rate_col``.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with liquidity proxy columns appended.
        """
        df = df.copy()

        rolling_10 = df[self.rate_col].rolling(window=10, min_periods=5)
        df["rate_rolling_max_10"] = rolling_10.max()
        df["rate_rolling_min_10"] = rolling_10.min()
        df["rate_high_low_range"] = (
            df["rate_rolling_max_10"] - df["rate_rolling_min_10"]
        )

        # Coefficient of variation (volatility ratio) — std / mean
        rolling_10_std = df[self.rate_col].rolling(window=10, min_periods=5).std()
        rolling_10_mean = df[self.rate_col].rolling(window=10, min_periods=5).mean()
        mean_safe = rolling_10_mean.replace(0, np.nan)
        df["rate_volatility_ratio"] = (rolling_10_std / mean_safe).clip(upper=5.0).fillna(0.0)

        # Z-score of range over a 20-period window
        range_rolling = df["rate_high_low_range"].rolling(window=20, min_periods=10)
        range_mean = range_rolling.mean()
        range_std = range_rolling.std().replace(0, np.nan)
        df["rate_range_zscore"] = (
            (df["rate_high_low_range"] - range_mean) / range_std
        ).fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps and return the enriched DataFrame.

        Steps executed in order:

        1. :meth:`add_rolling_stats` — windows [5, 10, 20]
        2. :meth:`add_spread_features`
        3. :meth:`add_momentum_features`
        4. :meth:`add_liquidity_proxy`
        5. Drop rows with remaining NaN values in any feature column
           (introduced by the rolling windows at the start of each series).

        Parameters
        ----------
        df : pd.DataFrame
            Raw or partially processed lending rate DataFrame.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame with NaN rows removed and index reset.
            The original ``rate_col``, ``ticker_col``, and ``date_col`` columns
            are preserved alongside the new feature columns.
        """
        logger.info("Building features for %d input rows", len(df))
        initial_cols = set(df.columns)

        df = self.add_rolling_stats(df, windows=[5, 10, 20])
        df = self.add_spread_features(df)
        df = self.add_momentum_features(df)
        df = self.add_liquidity_proxy(df)

        new_cols = sorted(set(df.columns) - initial_cols)
        logger.debug("New feature columns added (%d): %s", len(new_cols), new_cols)

        # Drop rows where any feature column is NaN
        before_drop = len(df)
        df = df.dropna(subset=new_cols).reset_index(drop=True)
        dropped = before_drop - len(df)

        if dropped > 0:
            logger.info(
                "Dropped %d rows with NaN values in feature columns (warm-up period)",
                dropped,
            )

        logger.info(
            "Feature engineering complete — %d rows, %d total columns",
            len(df),
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Helper: extract numeric feature matrix
    # ------------------------------------------------------------------

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the numeric feature matrix from a feature-enriched DataFrame.

        Excludes the raw identifier / metadata columns (ticker, date) and
        returns only the numeric feature columns as a NumPy array suitable
        for feeding directly into the anomaly detection models.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produced by :meth:`build_features`.

        Returns
        -------
        np.ndarray
            2-D float64 array of shape ``(n_samples, n_features)``.
        """
        data_cfg = self.config.get("data", {})
        exclude = {
            data_cfg.get("ticker_col", "ticker"),
            data_cfg.get("date_col", "date"),
        }

        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        logger.debug(
            "Feature matrix: %d rows × %d feature columns", len(df), len(feature_cols)
        )
        return df[feature_cols].values.astype(np.float64)
