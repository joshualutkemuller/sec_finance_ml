"""
feature_engineering.py — CollateralFeatureEngineer

Transforms raw merged collateral data into the feature matrix consumed
by the LightGBM quality scorer.  Each ``add_*`` method is idempotent and
safe to call independently; ``build_features`` applies all transformations
in the canonical order.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rating map: S&P / Moody's notation → ordinal integer (lower = better)
# ---------------------------------------------------------------------------
RATING_MAP: Dict[str, int] = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "NR": 17,
    "D": 18,
}

RATING_MAX = max(RATING_MAP.values())  # used for normalisation

FEATURE_COLS: List[str] = [
    "duration_score",
    "rating_score",
    "liquidity_score",
    "concentration_pct",
    "cost_of_carry_norm",
    "haircut_pct",
    "eligible",
]


class CollateralFeatureEngineer:
    """Compute quality-scoring features from a merged collateral DataFrame.

    Parameters
    ----------
    config : dict
        The ``optimization`` sub-section of ``config.yaml``.  Used to
        retrieve ``max_concentration_pct`` for the concentration flag.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._today: date = datetime.utcnow().date()

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    def add_duration_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a normalised modified-duration proxy score.

        Modified duration is approximated as::

            modified_duration ≈ years_to_maturity / (1 + coupon_rate)

        The raw duration is then scaled to [0, 1] using MinMaxScaler,
        where **higher duration → lower score** (longer duration carries
        more interest-rate risk, so lower quality for repo collateral).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``maturity_date`` (datetime) and
            ``coupon_rate`` (float, expressed as a decimal e.g. 0.05).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``duration_score`` column appended.
        """
        df = df.copy()

        if "maturity_date" not in df.columns:
            logger.warning("maturity_date column missing; duration_score set to 0.5")
            df["duration_score"] = 0.5
            return df

        today = pd.Timestamp(self._today)
        years_to_maturity = (df["maturity_date"] - today).dt.days / 365.25
        years_to_maturity = years_to_maturity.clip(lower=0.0)

        coupon = df.get("coupon_rate", pd.Series(0.05, index=df.index)).fillna(0.05)
        mod_duration = years_to_maturity / (1 + coupon)

        # Invert: shorter duration → higher score
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(mod_duration.values.reshape(-1, 1)).flatten()
        df["duration_score"] = 1.0 - scaled  # invert
        logger.debug("duration_score computed; mean=%.4f", df["duration_score"].mean())
        return df

    def add_credit_rating_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map credit rating strings to a normalised quality score.

        Uses :data:`RATING_MAP` to convert e.g. ``"AAA"`` → 1, then
        inverts and normalises so that **AAA → rating_score ≈ 1.0**
        and **D → rating_score ≈ 0.0**.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``credit_rating`` (str).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``rating_numeric`` and ``rating_score``
            columns appended.
        """
        df = df.copy()

        if "credit_rating" not in df.columns:
            logger.warning("credit_rating column missing; rating_score set to 0.5")
            df["rating_numeric"] = 9  # BBB
            df["rating_score"] = 0.5
            return df

        df["rating_numeric"] = (
            df["credit_rating"].str.strip().map(RATING_MAP).fillna(RATING_MAX)
        )
        # Invert: lower ordinal → higher score
        df["rating_score"] = 1.0 - (df["rating_numeric"] - 1) / (RATING_MAX - 1)
        logger.debug("rating_score computed; mean=%.4f", df["rating_score"].mean())
        return df

    def add_liquidity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a liquidity proxy score from the haircut percentage.

        Lower haircut → more liquid collateral → higher liquidity score.
        Score is defined as::

            raw = 1 / (haircut_pct + epsilon)
            liquidity_score = MinMaxScaler(raw) clipped to [0, 1]

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``haircut_pct`` (float, expressed as 0–1 or 0–100;
            values > 1 are automatically divided by 100).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``liquidity_score`` column appended.
        """
        df = df.copy()

        if "haircut_pct" not in df.columns:
            logger.warning("haircut_pct column missing; liquidity_score set to 0.5")
            df["liquidity_score"] = 0.5
            return df

        haircut = df["haircut_pct"].copy()
        # Normalise percentage format: treat values > 1 as percentage points
        if haircut.max() > 1.0:
            haircut = haircut / 100.0

        epsilon = 1e-6
        raw = 1.0 / (haircut + epsilon)
        raw = raw.clip(upper=raw.quantile(0.99))  # cap outliers

        scaler = MinMaxScaler()
        df["liquidity_score"] = scaler.fit_transform(
            raw.values.reshape(-1, 1)
        ).flatten()
        logger.debug("liquidity_score computed; mean=%.4f", df["liquidity_score"].mean())
        return df

    def add_concentration_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute pool-level concentration metrics per asset type.

        Calculates::

            concentration_pct[i] = sum(market_value for asset_type==k)
                                    / total_market_value

        where ``k`` is the asset type of row ``i``.  Also flags rows
        where the asset type's pool share exceeds
        ``config["max_concentration_pct"]``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``market_value`` and ``asset_type``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``concentration_pct`` and
            ``is_concentrated`` columns appended.
        """
        df = df.copy()

        if "market_value" not in df.columns or "asset_type" not in df.columns:
            logger.warning(
                "market_value or asset_type missing; concentration_pct set to 0.0"
            )
            df["concentration_pct"] = 0.0
            df["is_concentrated"] = False
            return df

        total_mv = df["market_value"].sum()
        type_mv = df.groupby("asset_type")["market_value"].transform("sum")
        df["concentration_pct"] = (type_mv / total_mv).fillna(0.0)

        max_conc = self._cfg.get("max_concentration_pct", 0.30)
        df["is_concentrated"] = df["concentration_pct"] > max_conc
        logger.debug(
            "concentration_metrics computed; concentrated rows: %d",
            df["is_concentrated"].sum(),
        )
        return df

    def add_cost_of_carry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a normalised cost-of-carry feature.

        Cost of carry per collateral item is approximated as::

            cost_of_carry = haircut_pct × financing_rate_bps / 10_000

        The result is normalised to [0, 1] via MinMaxScaler.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``haircut_pct`` and ``financing_rate_bps``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``cost_of_carry`` and
            ``cost_of_carry_norm`` columns appended.
        """
        df = df.copy()

        haircut = df.get("haircut_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
        rate_bps = df.get(
            "financing_rate_bps", pd.Series(0.0, index=df.index)
        ).fillna(0.0)

        # Normalise haircut if stored as percentage points
        if haircut.max() > 1.0:
            haircut = haircut / 100.0

        df["cost_of_carry"] = haircut * rate_bps / 10_000.0

        scaler = MinMaxScaler()
        df["cost_of_carry_norm"] = scaler.fit_transform(
            df["cost_of_carry"].values.reshape(-1, 1)
        ).flatten()
        logger.debug(
            "cost_of_carry computed; mean=%.6f", df["cost_of_carry"].mean()
        )
        return df

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations and return the feature matrix.

        Transformations are applied in this order:
            1. :meth:`add_duration_score`
            2. :meth:`add_credit_rating_numeric`
            3. :meth:`add_liquidity_score`
            4. :meth:`add_concentration_metrics`
            5. :meth:`add_cost_of_carry`

        Parameters
        ----------
        df : pd.DataFrame
            Merged collateral DataFrame from
            :meth:`~data_ingestion.CollateralDataLoader.merge_all`.

        Returns
        -------
        pd.DataFrame
            Full DataFrame with all engineered feature columns added.
            Callers can then select ``FEATURE_COLS`` to get the model
            input matrix.
        """
        logger.info("Building features for %d collateral items", len(df))
        df = self.add_duration_score(df)
        df = self.add_credit_rating_numeric(df)
        df = self.add_liquidity_score(df)
        df = self.add_concentration_metrics(df)
        df = self.add_cost_of_carry(df)

        # Ensure eligible is numeric (1/0)
        if "eligible" in df.columns:
            df["eligible"] = df["eligible"].astype(int)

        logger.info("Feature engineering complete; shape: %s", df.shape)
        return df
