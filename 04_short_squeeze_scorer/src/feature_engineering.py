"""
feature_engineering.py — Compute short-squeeze-relevant features from merged
short interest, borrow availability, and OHLCV price data.

Features computed:
    - days_to_cover          shares_short / 20-day avg daily volume
    - utilization_change     1-day delta of utilization_rate
    - si_ratio_momentum_N    rolling pct_change in short_interest_ratio over N days
    - price_momentum_N       N-day simple return of close price
    - rsi_14                 14-day Wilder RSI
    - volume_ratio_20        current volume / 20-day average volume
    - borrow_rate_zscore     z-score of borrow_rate over rolling 20-day window

All methods operate per ticker via groupby, so the input DataFrame must
contain a ``ticker`` column alongside the numeric columns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# Default rolling window for avg daily volume used in days_to_cover
_ADV_WINDOW: int = 20

# Default z-score window for borrow cost spike detection
_BORROW_ZSCORE_WINDOW: int = 20


class ShortSqueezeFeatureEngineer:
    """Build squeeze-signal features for an XGBoost short squeeze scorer.

    All public methods accept and return a ``pd.DataFrame`` sorted by
    (ticker, date). Methods may be called individually or chained through
    :meth:`build_features` which applies them in the recommended order.

    Args:
        config: The ``features`` block from ``config.yaml``.  Expected keys:
            ``si_ratio_windows`` (list[int]), ``momentum_windows`` (list[int]),
            ``volume_windows`` (list[int]).

    Example::

        engineer = ShortSqueezeFeatureEngineer(config["features"])
        feature_df = engineer.build_features(merged_df)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self._si_ratio_windows: list[int] = cfg.get("si_ratio_windows", [5, 10, 20])
        self._momentum_windows: list[int] = cfg.get("momentum_windows", [5, 10, 20])
        self._volume_windows: list[int] = cfg.get("volume_windows", [10, 20])

    # ------------------------------------------------------------------
    # Days-to-cover
    # ------------------------------------------------------------------

    def add_days_to_cover(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``days_to_cover`` column: shares_short / 20-day avg daily volume.

        Days-to-cover measures how many trading days it would take for all
        short sellers to close their positions at average trading volume.
        Higher values indicate a more heavily shorted, less liquid security.

        Args:
            df: Merged DataFrame with ``shares_short`` and ``volume`` columns.

        Returns:
            pd.DataFrame: Input DataFrame with ``days_to_cover`` appended.
        """
        df = df.copy()

        avg_vol = (
            df.groupby("ticker")["volume"]
            .transform(lambda s: s.rolling(_ADV_WINDOW, min_periods=1).mean())
        )
        df["days_to_cover"] = df["shares_short"] / avg_vol.replace(0, np.nan)
        logger.debug("Added days_to_cover (ADV window=%d).", _ADV_WINDOW)
        return df

    # ------------------------------------------------------------------
    # Utilization change
    # ------------------------------------------------------------------

    def add_utilization_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``utilization_change``: 1-day delta of ``utilization_rate``.

        A sudden increase in borrow utilization is a leading indicator that
        available supply is being rapidly consumed, which precedes a squeeze.

        Args:
            df: DataFrame with ``utilization_rate`` column.

        Returns:
            pd.DataFrame: Input DataFrame with ``utilization_change`` appended.
        """
        df = df.copy()
        df["utilization_change"] = (
            df.groupby("ticker")["utilization_rate"].transform(lambda s: s.diff(1))
        )
        logger.debug("Added utilization_change.")
        return df

    # ------------------------------------------------------------------
    # SI ratio momentum
    # ------------------------------------------------------------------

    def add_si_ratio_momentum(
        self, df: pd.DataFrame, windows: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Add rolling pct_change features for ``short_interest_ratio``.

        For each window ``N`` in *windows*, adds
        ``si_ratio_momentum_N = (si_ratio_t - si_ratio_{t-N}) / si_ratio_{t-N}``.

        Args:
            df: DataFrame with ``short_interest_ratio`` column.
            windows: Rolling windows (in days) for which to compute momentum.
                Defaults to the windows specified in config (e.g. [5, 10, 20]).

        Returns:
            pd.DataFrame: Input DataFrame with ``si_ratio_momentum_N`` columns.
        """
        df = df.copy()
        windows = windows or self._si_ratio_windows
        for w in windows:
            col = f"si_ratio_momentum_{w}"
            df[col] = df.groupby("ticker")["short_interest_ratio"].transform(
                lambda s, _w=w: s.pct_change(periods=_w)
            )
            logger.debug("Added %s.", col)
        return df

    # ------------------------------------------------------------------
    # Price momentum
    # ------------------------------------------------------------------

    def add_price_momentum(
        self, df: pd.DataFrame, windows: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Add N-day simple price return features for ``close`` price.

        For each window ``N``, adds
        ``price_momentum_N = (close_t / close_{t-N}) - 1``.

        Negative momentum paired with high short interest is a classic
        precondition for a short squeeze as short sellers face mounting losses.

        Args:
            df: DataFrame with ``close`` column.
            windows: Rolling windows (in days). Defaults to config.

        Returns:
            pd.DataFrame: Input DataFrame with ``price_momentum_N`` columns.
        """
        df = df.copy()
        windows = windows or self._momentum_windows
        for w in windows:
            col = f"price_momentum_{w}"
            df[col] = df.groupby("ticker")["close"].transform(
                lambda s, _w=w: s.pct_change(periods=_w)
            )
            logger.debug("Added %s.", col)
        return df

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Wilder RSI computed on ``close`` price per ticker.

        RSI below 30 combined with high short interest can indicate an
        oversold squeeze candidate; RSI above 70 in a high-SI name may
        signal a squeeze already in progress.

        Uses the ``ta`` library (``ta.momentum.RSIIndicator``) which
        implements the standard exponentially-smoothed Wilder RSI.

        Args:
            df: DataFrame with ``close`` column.
            window: Lookback period for RSI computation. Default 14.

        Returns:
            pd.DataFrame: Input DataFrame with ``rsi_{window}`` appended.
        """
        df = df.copy()
        col = f"rsi_{window}"

        def _rsi_per_ticker(series: pd.Series) -> pd.Series:
            indicator = ta.momentum.RSIIndicator(close=series, window=window)
            return indicator.rsi()

        df[col] = df.groupby("ticker")["close"].transform(_rsi_per_ticker)
        logger.debug("Added %s.", col)
        return df

    # ------------------------------------------------------------------
    # Volume ratio
    # ------------------------------------------------------------------

    def add_volume_ratio(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volume ratio: current volume / N-day average volume.

        A sudden spike in volume (ratio >> 1) in a high-short-interest
        security often marks the onset of covering pressure.

        Args:
            df: DataFrame with ``volume`` column.
            window: Rolling average window. Default 20.

        Returns:
            pd.DataFrame: Input DataFrame with ``volume_ratio_{window}`` appended.
        """
        df = df.copy()
        col = f"volume_ratio_{window}"
        avg_vol = df.groupby("ticker")["volume"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[col] = df["volume"] / avg_vol.replace(0, np.nan)
        logger.debug("Added %s.", col)
        return df

    # ------------------------------------------------------------------
    # Borrow cost spike
    # ------------------------------------------------------------------

    def add_borrow_cost_spike(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-score of ``borrow_rate`` per ticker.

        ``borrow_rate_zscore = (borrow_rate - rolling_mean) / rolling_std``

        A z-score above 2 indicates that the current borrow rate is
        statistically elevated relative to recent history, suggesting
        rising demand to borrow shares and potential squeeze conditions.

        Args:
            df: DataFrame with ``borrow_rate`` column.

        Returns:
            pd.DataFrame: Input DataFrame with ``borrow_rate_zscore`` appended.
        """
        df = df.copy()

        def _zscore(series: pd.Series) -> pd.Series:
            roll = series.rolling(_BORROW_ZSCORE_WINDOW, min_periods=2)
            return (series - roll.mean()) / roll.std().replace(0, np.nan)

        df["borrow_rate_zscore"] = df.groupby("ticker")["borrow_rate"].transform(
            _zscore
        )
        logger.debug("Added borrow_rate_zscore (window=%d).", _BORROW_ZSCORE_WINDOW)
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps in order and return the result.

        Applies the following transformations in sequence:
            1. ``add_days_to_cover``
            2. ``add_utilization_change``
            3. ``add_si_ratio_momentum`` (all configured windows)
            4. ``add_price_momentum`` (all configured windows)
            5. ``add_rsi`` (window=14)
            6. ``add_volume_ratio`` for each configured volume window
            7. ``add_borrow_cost_spike``

        Any NaN values introduced by rolling computations are left in place;
        callers should drop or impute them before passing to the scorer.

        Args:
            df: Fully merged DataFrame from
                :meth:`~data_ingestion.ShortInterestDataLoader.merge_all`.

        Returns:
            pd.DataFrame: DataFrame with all engineered feature columns added.
        """
        logger.info("Building features for %d rows.", len(df))
        df = self.add_days_to_cover(df)
        df = self.add_utilization_change(df)
        df = self.add_si_ratio_momentum(df)
        df = self.add_price_momentum(df)
        df = self.add_rsi(df)
        for w in self._volume_windows:
            df = self.add_volume_ratio(df, window=w)
        df = self.add_borrow_cost_spike(df)
        logger.info(
            "Feature engineering complete. Output shape: %s", df.shape
        )
        return df

    def get_feature_columns(self) -> list[str]:
        """Return the list of engineered feature column names.

        Useful for selecting the X matrix before passing to the scorer.

        Returns:
            list[str]: Ordered list of feature column names produced by
                :meth:`build_features`.
        """
        cols: list[str] = ["days_to_cover", "utilization_change"]
        cols += [f"si_ratio_momentum_{w}" for w in self._si_ratio_windows]
        cols += [f"price_momentum_{w}" for w in self._momentum_windows]
        cols += ["rsi_14"]
        cols += [f"volume_ratio_{w}" for w in self._volume_windows]
        cols += ["borrow_rate_zscore"]
        # Base short interest features
        cols += ["short_interest_ratio", "utilization_rate", "borrow_rate"]
        return cols
