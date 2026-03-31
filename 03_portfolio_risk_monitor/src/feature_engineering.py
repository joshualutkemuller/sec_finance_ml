"""
feature_engineering.py — Portfolio Concentration Risk Early Warning Monitor

Transforms the raw exposure snapshot produced by
:class:`~src.data_ingestion.PortfolioDataLoader` into a feature matrix
suitable for the :class:`~src.model.RandomForestRiskScorer`.

Features produced
-----------------
Concentration metrics (per portfolio/date row):
    hhi_portfolio       — HHI across all positions in the portfolio
    hhi_sector          — HHI across sector weights
    hhi_counterparty    — HHI across counterparty weights
    top5_exposure_pct   — % of portfolio AUM in the five largest positions
    max_sector_weight   — largest single sector weight
    max_cp_weight       — largest single counterparty weight
    sector_limit_breach — 1 if any sector exceeds the configured limit
    cp_limit_breach     — 1 if any counterparty exceeds the configured limit

Rolling / dynamic features (windowed over lookback days):
    hhi_rolling_change          — change in portfolio HHI over the window
    top5_rolling_change         — change in top-5 exposure pct over window
    max_sector_rolling_change   — change in max sector weight over window

Liquidity and risk features:
    liquidity_adjusted_exposure — sum(market_value / bid_ask_spread)
    portfolio_var_proxy         — sum of position-weighted volatility * 1.645
    avg_credit_utilisation      — mean counterparty credit line utilisation
    max_credit_utilisation      — max counterparty credit line utilisation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Z-score for 95 % VaR
_VAR_Z_95 = 1.6449


class ConcentrationFeatureEngineer:
    """Compute concentration and risk features from an exposure snapshot.

    Parameters
    ----------
    config : dict
        The ``concentration`` section of config.yaml, expected keys:
        ``hhi_warning_threshold``, ``top5_exposure_warning_pct``,
        ``sector_limit_pct``, ``counterparty_limit_pct``.
    lookback_days : int, optional
        Rolling window in trading days for change features (default 5).
    """

    def __init__(self, config: dict, lookback_days: int = 5) -> None:
        self._config = config
        self._lookback_days = lookback_days
        self._hhi_warn: float = float(config.get("hhi_warning_threshold", 0.25))
        self._top5_warn: float = float(config.get("top5_exposure_warning_pct", 40.0))
        self._sector_limit: float = float(config.get("sector_limit_pct", 30.0))
        self._cp_limit: float = float(config.get("counterparty_limit_pct", 20.0))

    # ------------------------------------------------------------------
    # Core metric computations (stateless helpers)
    # ------------------------------------------------------------------

    def compute_hhi(self, df: pd.DataFrame, weight_col: str) -> float:
        """Compute the Herfindahl-Hirschman Index for a set of weights.

        HHI = sum of squared market shares.  Ranges from 1/N (fully
        diversified across N equal positions) to 1.0 (complete concentration
        in a single name).  Weights are re-normalised internally so raw
        market values can also be passed.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the weight column.
        weight_col : str
            Column name holding market values or weight fractions.

        Returns
        -------
        float
            HHI value in [0, 1].  Returns 0.0 if the total weight is zero.
        """
        values = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        total = values.sum()
        if total <= 0.0:
            return 0.0
        shares = values / total
        hhi = float((shares ** 2).sum())
        return round(hhi, 6)

    def compute_top_n_exposure(self, df: pd.DataFrame, n: int = 5) -> float:
        """Return the fraction of total portfolio AUM held in the top-N positions.

        Parameters
        ----------
        df : pd.DataFrame
            Single-portfolio slice with a ``market_value`` column.
        n : int, optional
            Number of top positions to include (default 5).

        Returns
        -------
        float
            Top-N exposure as a percentage of total (0 – 100).
        """
        if "market_value" not in df.columns:
            raise ValueError("DataFrame must contain a 'market_value' column.")
        mv = df["market_value"].fillna(0.0)
        total = mv.sum()
        if total <= 0.0:
            return 0.0
        top_n_sum = mv.nlargest(n).sum()
        return round(100.0 * top_n_sum / total, 4)

    # ------------------------------------------------------------------
    # Per-snapshot enrichment methods
    # ------------------------------------------------------------------

    def add_sector_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sector-level concentration columns to an exposure snapshot.

        Computes, per (portfolio_id, date):
            hhi_sector          — HHI of sector market values
            max_sector_weight   — largest single sector weight (fraction)
            sector_limit_breach — 1 if max_sector_weight * 100 > limit

        Parameters
        ----------
        df : pd.DataFrame
            Exposure snapshot with columns portfolio_id, date, sector,
            market_value.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with three new columns appended.
        """
        logger.debug("Computing sector concentration metrics")
        if "sector" not in df.columns:
            df["sector"] = "UNKNOWN"

        sector_mv = (
            df.groupby(["portfolio_id", "date", "sector"])["market_value"]
            .sum()
            .reset_index()
        )
        portfolio_totals = (
            sector_mv.groupby(["portfolio_id", "date"])["market_value"]
            .sum()
            .rename("total_mv")
            .reset_index()
        )
        sector_mv = sector_mv.merge(portfolio_totals, on=["portfolio_id", "date"])
        sector_mv["sector_weight"] = np.where(
            sector_mv["total_mv"] > 0,
            sector_mv["market_value"] / sector_mv["total_mv"],
            0.0,
        )

        hhi_sector = (
            sector_mv.groupby(["portfolio_id", "date"])
            .apply(lambda g: (g["sector_weight"] ** 2).sum())
            .rename("hhi_sector")
            .reset_index()
        )
        max_sector = (
            sector_mv.groupby(["portfolio_id", "date"])["sector_weight"]
            .max()
            .rename("max_sector_weight")
            .reset_index()
        )

        metrics = hhi_sector.merge(max_sector, on=["portfolio_id", "date"])
        metrics["sector_limit_breach"] = (
            metrics["max_sector_weight"] * 100 > self._sector_limit
        ).astype(int)

        df = df.merge(metrics, on=["portfolio_id", "date"], how="left")
        for col in ["hhi_sector", "max_sector_weight", "sector_limit_breach"]:
            df[col] = df[col].fillna(0.0)
        return df

    def add_counterparty_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add counterparty-level concentration columns to an exposure snapshot.

        Computes, per (portfolio_id, date):
            hhi_counterparty    — HHI of counterparty market values
            max_cp_weight       — largest single counterparty weight (fraction)
            cp_limit_breach     — 1 if max_cp_weight * 100 > configured limit

        Parameters
        ----------
        df : pd.DataFrame
            Exposure snapshot with columns portfolio_id, date, counterparty,
            market_value.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with three new columns appended.
        """
        logger.debug("Computing counterparty concentration metrics")
        if "counterparty" not in df.columns:
            df["counterparty"] = "UNKNOWN"

        cp_mv = (
            df.groupby(["portfolio_id", "date", "counterparty"])["market_value"]
            .sum()
            .reset_index()
        )
        portfolio_totals = (
            cp_mv.groupby(["portfolio_id", "date"])["market_value"]
            .sum()
            .rename("total_mv_cp")
            .reset_index()
        )
        cp_mv = cp_mv.merge(portfolio_totals, on=["portfolio_id", "date"])
        cp_mv["cp_weight"] = np.where(
            cp_mv["total_mv_cp"] > 0,
            cp_mv["market_value"] / cp_mv["total_mv_cp"],
            0.0,
        )

        hhi_cp = (
            cp_mv.groupby(["portfolio_id", "date"])
            .apply(lambda g: (g["cp_weight"] ** 2).sum())
            .rename("hhi_counterparty")
            .reset_index()
        )
        max_cp = (
            cp_mv.groupby(["portfolio_id", "date"])["cp_weight"]
            .max()
            .rename("max_cp_weight")
            .reset_index()
        )

        metrics = hhi_cp.merge(max_cp, on=["portfolio_id", "date"])
        metrics["cp_limit_breach"] = (
            metrics["max_cp_weight"] * 100 > self._cp_limit
        ).astype(int)

        df = df.merge(metrics, on=["portfolio_id", "date"], how="left")
        for col in ["hhi_counterparty", "max_cp_weight", "cp_limit_breach"]:
            df[col] = df[col].fillna(0.0)
        return df

    def add_rolling_exposure_change(
        self, df: pd.DataFrame, window: int = 5
    ) -> pd.DataFrame:
        """Add rolling-window change features for HHI and top-5 exposure.

        Computes, per portfolio_id sorted by date:
            hhi_rolling_change        — current HHI minus HHI ``window`` days ago
            top5_rolling_change       — same delta for top5_exposure_pct
            max_sector_rolling_change — same delta for max_sector_weight

        NaN values (first ``window`` rows per portfolio) are filled with 0.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio-level summary DataFrame (one row per portfolio/date)
            that already contains hhi_portfolio, top5_exposure_pct, and
            max_sector_weight columns.
        window : int, optional
            Number of trading days for the rolling window (default 5).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with three new delta columns.
        """
        logger.debug("Computing rolling exposure change features (window=%d)", window)
        df = df.sort_values(["portfolio_id", "date"]).copy()

        for col, new_col in [
            ("hhi_portfolio", "hhi_rolling_change"),
            ("top5_exposure_pct", "top5_rolling_change"),
            ("max_sector_weight", "max_sector_rolling_change"),
        ]:
            if col not in df.columns:
                df[new_col] = 0.0
                continue
            df[new_col] = df.groupby("portfolio_id")[col].transform(
                lambda s: s.diff(window)
            )
        df[
            ["hhi_rolling_change", "top5_rolling_change", "max_sector_rolling_change"]
        ] = df[
            ["hhi_rolling_change", "top5_rolling_change", "max_sector_rolling_change"]
        ].fillna(
            0.0
        )
        return df

    def add_liquidity_adjusted_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a liquidity-adjusted exposure metric per portfolio/date.

        Liquidity-adjusted exposure is defined as:

            sum(market_value / bid_ask_spread)

        A higher value indicates the portfolio holds large positions in
        illiquid instruments (wide bid-ask spread relative to market value).

        Parameters
        ----------
        df : pd.DataFrame
            Exposure snapshot with columns market_value and bid_ask_spread.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``liquidity_adjusted_exposure`` column added.
        """
        logger.debug("Computing liquidity-adjusted exposure")
        if "bid_ask_spread" not in df.columns:
            df["liquidity_adjusted_exposure"] = df.get("market_value", 0.0)
            return df

        # Protect against division by zero
        spread = df["bid_ask_spread"].replace(0, np.nan).fillna(
            df["bid_ask_spread"].median()
        )
        df["liq_adj_single"] = df["market_value"] / spread

        liq_agg = (
            df.groupby(["portfolio_id", "date"])["liq_adj_single"]
            .sum()
            .rename("liquidity_adjusted_exposure")
            .reset_index()
        )
        df = df.merge(liq_agg, on=["portfolio_id", "date"], how="left")
        df = df.drop(columns=["liq_adj_single"])
        return df

    # ------------------------------------------------------------------
    # Master builder
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature matrix from an exposure snapshot.

        This is the primary entry point used by the pipeline.  It runs all
        enrichment steps in order, then aggregates to one row per
        (portfolio_id, date) so the feature matrix has the correct shape
        for the Random Forest scorer.

        Steps
        -----
        1. Compute per-position HHI inputs.
        2. Add sector concentration metrics.
        3. Add counterparty concentration metrics.
        4. Add liquidity-adjusted exposure.
        5. Aggregate to portfolio/date level.
        6. Compute top-5 exposure %.
        7. Derive portfolio-level HHI.
        8. Compute portfolio VaR proxy.
        9. Add rolling change features.
        10. Drop intermediate columns and return feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Exposure snapshot from
            :meth:`~src.data_ingestion.PortfolioDataLoader.build_exposure_snapshot`.

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per (portfolio_id, date) and all
            feature columns required by the risk scorer.
        """
        logger.info("Building feature matrix from exposure snapshot (%d rows)", len(df))

        df = df.copy()

        # -- Step 2 & 3: concentration metrics on the raw snapshot
        df = self.add_sector_concentration(df)
        df = self.add_counterparty_concentration(df)
        df = self.add_liquidity_adjusted_exposure(df)

        # -- Step 5: aggregate to portfolio/date level
        agg_dict: dict[str, str | list] = {
            "market_value": "sum",
            "portfolio_aum": "first",
            "hhi_sector": "first",
            "max_sector_weight": "first",
            "sector_limit_breach": "first",
            "hhi_counterparty": "first",
            "max_cp_weight": "first",
            "cp_limit_breach": "first",
            "liquidity_adjusted_exposure": "first",
        }
        if "volatility_30d" in df.columns:
            agg_dict["volatility_30d"] = "mean"
        if "credit_utilisation" in df.columns:
            agg_dict["avg_credit_utilisation"] = pd.NamedAgg(
                column="credit_utilisation", aggfunc="mean"
            ) if False else "mean"  # use string key below

        # Build a clean aggregation dict for groupby
        safe_agg: dict = {}
        for col, func in agg_dict.items():
            if col in df.columns:
                safe_agg[col] = func

        if "credit_utilisation" in df.columns:
            safe_agg["credit_utilisation"] = ["mean", "max"]

        port_df = df.groupby(["portfolio_id", "date"]).agg(safe_agg).reset_index()

        # Flatten multi-level column names from multi-func aggregations
        flat_cols: list[str] = []
        for col in port_df.columns:
            if isinstance(col, tuple):
                parent, agg_fn = col
                if parent == "credit_utilisation":
                    if agg_fn == "mean":
                        flat_cols.append("avg_credit_utilisation")
                    elif agg_fn == "max":
                        flat_cols.append("max_credit_utilisation")
                    else:
                        flat_cols.append(f"{parent}_{agg_fn}")
                else:
                    flat_cols.append(parent if agg_fn in ("first", "sum", "mean") else f"{parent}_{agg_fn}")
            else:
                flat_cols.append(col)
        port_df.columns = flat_cols  # type: ignore[assignment]

        # -- Step 6: top-5 exposure % (needs raw snapshot)
        top5_records = []
        for (pid, date), grp in df.groupby(["portfolio_id", "date"]):
            top5_records.append(
                {
                    "portfolio_id": pid,
                    "date": date,
                    "top5_exposure_pct": self.compute_top_n_exposure(grp, n=5),
                }
            )
        top5_df = pd.DataFrame(top5_records)
        port_df = port_df.merge(top5_df, on=["portfolio_id", "date"], how="left")

        # -- Step 7: portfolio-level HHI
        hhi_records = []
        for (pid, date), grp in df.groupby(["portfolio_id", "date"]):
            hhi_records.append(
                {
                    "portfolio_id": pid,
                    "date": date,
                    "hhi_portfolio": self.compute_hhi(grp, "market_value"),
                }
            )
        hhi_df = pd.DataFrame(hhi_records)
        port_df = port_df.merge(hhi_df, on=["portfolio_id", "date"], how="left")

        # -- Step 8: VaR proxy = sum(aum_weight * volatility * z95)
        if "volatility_30d" in df.columns:
            df["var_contrib"] = (
                df.get("aum_weight", 0.0) * df["volatility_30d"] * _VAR_Z_95
            )
            var_records = (
                df.groupby(["portfolio_id", "date"])["var_contrib"]
                .sum()
                .rename("portfolio_var_proxy")
                .reset_index()
            )
            port_df = port_df.merge(var_records, on=["portfolio_id", "date"], how="left")
        else:
            port_df["portfolio_var_proxy"] = 0.0

        # -- Breach flag features derived from thresholds
        port_df["hhi_breach"] = (
            port_df["hhi_portfolio"] > self._hhi_warn
        ).astype(int)
        port_df["top5_breach"] = (
            port_df["top5_exposure_pct"] > self._top5_warn
        ).astype(int)

        # -- Step 9: rolling change features
        port_df = self.add_rolling_exposure_change(port_df, window=self._lookback_days)

        # Fill any remaining NaN values with 0
        numeric_cols = port_df.select_dtypes(include=[np.number]).columns
        port_df[numeric_cols] = port_df[numeric_cols].fillna(0.0)

        logger.info(
            "Feature matrix built: %d rows × %d columns", *port_df.shape
        )
        return port_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feature column list (used by model training / inference)
    # ------------------------------------------------------------------

    @staticmethod
    def feature_columns() -> list[str]:
        """Return the ordered list of feature column names expected by the scorer.

        Returns
        -------
        list[str]
            Feature column names in the expected order.
        """
        return [
            "hhi_portfolio",
            "hhi_sector",
            "hhi_counterparty",
            "top5_exposure_pct",
            "max_sector_weight",
            "max_cp_weight",
            "sector_limit_breach",
            "cp_limit_breach",
            "liquidity_adjusted_exposure",
            "portfolio_var_proxy",
            "avg_credit_utilisation",
            "max_credit_utilisation",
            "hhi_breach",
            "top5_breach",
            "hhi_rolling_change",
            "top5_rolling_change",
            "max_sector_rolling_change",
        ]
