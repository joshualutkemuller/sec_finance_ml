"""
feature_engineering.py — Financing Cost Forecaster feature engineering layer.

Provides FinancingCostFeatureEngineer, which transforms a merged DataFrame
of repo rates, macro rates, and positions into a model-ready feature matrix
with lag features, rolling statistics, calendar flags, macro spreads, rate
momentum, and term-structure differentials.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinancingCostFeatureEngineer:
    """
    Builds the full feature matrix for multi-horizon financing cost forecasting.

    Parameters
    ----------
    config : dict
        The ``features`` section of config.yaml with keys:
        lag_windows, rolling_windows, use_calendar_features, use_macro_features.
    group_col : str
        Column name used to group by instrument type before computing
        lag/rolling features, so that lags don't bleed across instruments.
    """

    def __init__(self, config: dict, group_col: str = "instrument_type") -> None:
        self.config = config
        self.group_col = group_col
        self.lag_windows: list[int] = config.get("lag_windows", [1, 3, 5, 10, 20])
        self.rolling_windows: list[int] = config.get("rolling_windows", [5, 10, 20])
        self.use_calendar: bool = config.get("use_calendar_features", True)
        self.use_macro: bool = config.get("use_macro_features", True)

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def add_lag_features(
        self,
        df: pd.DataFrame,
        col: str,
        lags: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Add shifted (lag) versions of ``col`` grouped by instrument type.

        For each lag ``l`` in ``lags``, creates column ``{col}_lag_{l}``
        equal to the value of ``col`` ``l`` rows earlier within each
        instrument group (i.e. sorted by date within group).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame; must contain columns ``date``, ``group_col``,
            and ``col``.
        col : str
            Column to lag.
        lags : list[int], optional
            Lag periods to create. Defaults to ``self.lag_windows``.

        Returns
        -------
        pd.DataFrame
            DataFrame with lag columns appended in-place (copy returned).
        """
        if lags is None:
            lags = self.lag_windows
        df = df.copy()
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping lag features.", col)
            return df
        for lag in lags:
            new_col = f"{col}_lag_{lag}"
            df[new_col] = (
                df.groupby(self.group_col, sort=False)[col]
                .shift(lag)
            )
        return df

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        col: str,
        windows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Add rolling mean, standard deviation, minimum, and maximum of ``col``
        computed within each instrument group.

        Creates columns of the form ``{col}_roll{w}_{stat}`` for each
        window ``w`` and statistic in {mean, std, min, max}.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame; must be sorted by (group_col, date).
        col : str
            Column over which to compute rolling statistics.
        windows : list[int], optional
            Window sizes in rows. Defaults to ``self.rolling_windows``.

        Returns
        -------
        pd.DataFrame
            DataFrame with rolling stat columns appended (copy returned).
        """
        if windows is None:
            windows = self.rolling_windows
        df = df.copy()
        if col not in df.columns:
            logger.warning("Column '%s' not found — skipping rolling stats.", col)
            return df
        for w in windows:
            grouped = df.groupby(self.group_col, sort=False)[col]
            df[f"{col}_roll{w}_mean"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
            )
            df[f"{col}_roll{w}_std"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
            )
            df[f"{col}_roll{w}_min"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).min()
            )
            df[f"{col}_roll{w}_max"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).max()
            )
        return df

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features that capture repo-market seasonality.

        New columns:
        - ``day_of_week``       : integer 0 (Mon) – 4 (Fri)
        - ``is_month_end``      : 1 if within 3 business days of month end
        - ``is_quarter_end``    : 1 if within 3 business days of quarter end
        - ``days_to_month_end`` : calendar days remaining in current month

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``date`` column of dtype datetime64.

        Returns
        -------
        pd.DataFrame
            Copy of df with calendar columns appended.
        """
        df = df.copy()
        if "date" not in df.columns:
            logger.warning("'date' column not found — skipping calendar features.")
            return df

        dates: pd.Series = pd.to_datetime(df["date"])

        df["day_of_week"] = dates.dt.dayofweek.astype(np.int8)

        # Month-end flag: last trading day of the month or within 3 business days
        month_last_bday = dates.apply(
            lambda d: pd.offsets.BMonthEnd().rollforward(d)
        )
        days_to_month_end_bdays = (month_last_bday - dates).dt.days
        df["is_month_end"] = (days_to_month_end_bdays <= 3).astype(np.int8)
        df["days_to_month_end"] = (
            dates.apply(lambda d: (d + pd.offsets.MonthEnd(0)).day - d.day)
        ).astype(np.int16)

        # Quarter-end flag
        quarter_end_months = {3, 6, 9, 12}
        is_qtr_month = dates.dt.month.isin(quarter_end_months)
        df["is_quarter_end"] = (is_qtr_month & (df["is_month_end"] == 1)).astype(np.int8)

        return df

    # ------------------------------------------------------------------
    # Macro spread features
    # ------------------------------------------------------------------

    def add_macro_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute repo-SOFR spread and related macro basis features.

        New columns:
        - ``repo_sofr_spread``  : rate_bps - (sofr * 100)
        - ``repo_ff_spread``    : rate_bps - (fed_funds * 100)
        - ``ois_basis``         : (sofr - ois) * 100   [monetary policy signal]

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``rate_bps``, ``sofr``, ``fed_funds``, ``ois``.

        Returns
        -------
        pd.DataFrame
            Copy with spread columns appended.
        """
        df = df.copy()
        if "sofr" in df.columns and "rate_bps" in df.columns:
            df["repo_sofr_spread"] = df["rate_bps"] - df["sofr"] * 100.0
        else:
            df["repo_sofr_spread"] = np.nan

        if "fed_funds" in df.columns and "rate_bps" in df.columns:
            df["repo_ff_spread"] = df["rate_bps"] - df["fed_funds"] * 100.0
        else:
            df["repo_ff_spread"] = np.nan

        if "sofr" in df.columns and "ois" in df.columns:
            df["ois_basis"] = (df["sofr"] - df["ois"]) * 100.0
        else:
            df["ois_basis"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Rate momentum (acceleration)
    # ------------------------------------------------------------------

    def add_rate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rate change and second-order acceleration features.

        New columns (computed within each instrument group):
        - ``rate_change_1d``    : rate_bps - rate_bps_lag_1
        - ``rate_change_3d``    : rate_bps - rate_bps_lag_3
        - ``rate_accel_1d``     : rate_change_1d - prior rate_change_1d
          (i.e. second difference — rate acceleration signal)

        Parameters
        ----------
        df : pd.DataFrame
            Must already contain ``rate_bps`` and lag columns.

        Returns
        -------
        pd.DataFrame
            Copy with momentum columns appended.
        """
        df = df.copy()
        if "rate_bps" not in df.columns:
            return df

        if "rate_bps_lag_1" in df.columns:
            df["rate_change_1d"] = df["rate_bps"] - df["rate_bps_lag_1"]
        else:
            df["rate_change_1d"] = (
                df.groupby(self.group_col, sort=False)["rate_bps"]
                .diff(1)
            )

        if "rate_bps_lag_3" in df.columns:
            df["rate_change_3d"] = df["rate_bps"] - df["rate_bps_lag_3"]
        else:
            df["rate_change_3d"] = (
                df.groupby(self.group_col, sort=False)["rate_bps"]
                .diff(3)
            )

        # Acceleration: change in rate_change_1d
        df["rate_accel_1d"] = (
            df.groupby(self.group_col, sort=False)["rate_change_1d"]
            .diff(1)
        )

        return df

    # ------------------------------------------------------------------
    # Term structure features
    # ------------------------------------------------------------------

    def add_term_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute spread between tenors as a term-structure signal.

        When multiple tenors are present for the same instrument on the
        same date (e.g. O/N vs T/N vs 1W), this method computes the
        spread of each tenor's rate to the overnight (O/N) rate. If only
        one tenor is present, it falls back to the SOFR–OIS curve slope
        derived from macro rates as a proxy for term-structure shape.

        New columns:
        - ``tn_on_spread``      : T/N rate minus O/N rate (bps)
        - ``1w_on_spread``      : 1W rate minus O/N rate (bps)
        - ``curve_slope_proxy`` : (libor_3m - sofr) * 100 as curve shape proxy

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame; may contain a ``tenor`` column.

        Returns
        -------
        pd.DataFrame
            Copy with term-structure columns appended.
        """
        df = df.copy()

        # Curve slope proxy using macro rates
        if "libor_3m" in df.columns and "sofr" in df.columns:
            df["curve_slope_proxy"] = (df["libor_3m"] - df["sofr"]) * 100.0
        else:
            df["curve_slope_proxy"] = np.nan

        if "tenor" not in df.columns:
            df["tn_on_spread"] = np.nan
            df["1w_on_spread"] = np.nan
            return df

        # Pivot to compute inter-tenor spreads on the same date+instrument
        tenor_pivot: pd.DataFrame = (
            df.groupby(["date", self.group_col, "tenor"])["rate_bps"]
            .mean()
            .unstack("tenor")
            .reset_index()
        )
        on_col = next((c for c in tenor_pivot.columns if "O/N" in str(c)), None)
        tn_col = next((c for c in tenor_pivot.columns if "T/N" in str(c)), None)
        w1_col = next((c for c in tenor_pivot.columns if "1W" in str(c)), None)

        if on_col and tn_col:
            tenor_pivot["tn_on_spread"] = tenor_pivot[tn_col] - tenor_pivot[on_col]
        if on_col and w1_col:
            tenor_pivot["1w_on_spread"] = tenor_pivot[w1_col] - tenor_pivot[on_col]

        spread_cols = ["date", self.group_col]
        for sc in ["tn_on_spread", "1w_on_spread"]:
            if sc in tenor_pivot.columns:
                spread_cols.append(sc)

        df = df.merge(tenor_pivot[spread_cols], on=["date", self.group_col], how="left")

        # Fill any still-missing spread columns
        for sc in ["tn_on_spread", "1w_on_spread"]:
            if sc not in df.columns:
                df[sc] = np.nan

        return df

    # ------------------------------------------------------------------
    # Master builder
    # ------------------------------------------------------------------

    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str = "rate_bps",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply the full feature engineering pipeline and return X and y.

        Steps applied in order:
        1. Sort by (group_col, date).
        2. add_macro_spread (if use_macro).
        3. add_term_structure_features (if use_macro).
        4. add_lag_features on rate_bps and repo_sofr_spread.
        5. add_rolling_stats on rate_bps and repo_sofr_spread.
        6. add_rate_momentum.
        7. add_calendar_features (if use_calendar).
        8. Drop rows with NaN in target or core lag features.
        9. Separate numeric feature columns from target.

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame from FinancingDataLoader.merge_all().
        target_col : str
            Name of the column to use as the prediction target (y).

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (numeric columns, NaN-filled where sparse).
        y : pd.Series
            Target series aligned with X.
        """
        df = df.copy()
        df.sort_values([self.group_col, "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if self.use_macro:
            df = self.add_macro_spread(df)
            df = self.add_term_structure_features(df)

        # Lag and rolling features on primary target
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_stats(df, target_col)

        # Lag and rolling features on SOFR spread if available
        if "repo_sofr_spread" in df.columns:
            df = self.add_lag_features(df, "repo_sofr_spread")
            df = self.add_rolling_stats(df, "repo_sofr_spread")

        # Notional (demand pressure) lags
        if "notional" in df.columns:
            df = self.add_lag_features(df, "notional", lags=[1, 5])

        df = self.add_rate_momentum(df)

        if self.use_calendar:
            df = self.add_calendar_features(df)

        # Encode instrument type as integer category for LightGBM
        if self.group_col in df.columns:
            df["instrument_type_code"] = (
                df[self.group_col].astype("category").cat.codes.astype(np.int8)
            )

        # Drop rows where target is NaN
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # Identify feature columns: numeric only, exclude raw strings and date
        exclude_cols = {
            target_col,
            "date",
            self.group_col,
            "tenor",
            "budget_rate_bps",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        logger.info(
            "build_features: X shape=%s, feature count=%d, target='%s'",
            X.shape,
            len(feature_cols),
            target_col,
        )
        return X, y
