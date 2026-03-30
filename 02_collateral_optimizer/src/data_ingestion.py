"""
data_ingestion.py — CollateralDataLoader

Responsible for loading all four raw data sources (collateral pool,
eligibility schedule, haircut table, transactions) from CSV files and
producing a single merged DataFrame that downstream modules consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CollateralDataLoader:
    """Load and merge all collateral data sources.

    Parameters
    ----------
    config : dict
        The ``data`` sub-section of ``config.yaml``, containing keys:
        ``collateral_pool_path``, ``eligibility_schedule_path``,
        ``haircut_table_path``, ``transaction_path``.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._collateral_pool: Optional[pd.DataFrame] = None
        self._eligibility: Optional[pd.DataFrame] = None
        self._haircuts: Optional[pd.DataFrame] = None
        self._transactions: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_collateral_pool(self) -> pd.DataFrame:
        """Load the collateral pool CSV.

        Expected columns:
            collateral_id, asset_type, isin, face_value, market_value,
            currency, maturity_date, coupon_rate, credit_rating,
            issuer, country, sector

        Returns
        -------
        pd.DataFrame
            Raw collateral pool with standard dtypes applied.
        """
        path = Path(self._cfg["collateral_pool_path"])
        logger.info("Loading collateral pool from %s", path)
        df = pd.read_csv(path, parse_dates=["maturity_date"])

        # Normalise column names to lowercase / snake_case
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Coerce numeric columns
        for col in ("face_value", "market_value", "coupon_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._collateral_pool = df
        logger.info("Collateral pool loaded: %d rows", len(df))
        return df

    def load_eligibility_schedule(self) -> pd.DataFrame:
        """Load the eligibility schedule CSV.

        Expected columns:
            counterparty_id, asset_type, eligible, min_credit_rating,
            max_maturity_years, accepted_currencies

        Returns
        -------
        pd.DataFrame
            Eligibility rules per counterparty and asset type.
        """
        path = Path(self._cfg["eligibility_schedule_path"])
        logger.info("Loading eligibility schedule from %s", path)
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Ensure boolean eligible column
        if "eligible" in df.columns:
            df["eligible"] = df["eligible"].astype(bool)

        self._eligibility = df
        logger.info("Eligibility schedule loaded: %d rows", len(df))
        return df

    def load_haircut_table(self) -> pd.DataFrame:
        """Load the haircut table CSV.

        Expected columns:
            asset_type, credit_rating_bucket, maturity_bucket, haircut_pct

        Returns
        -------
        pd.DataFrame
            Haircut percentages keyed by asset type, rating, and maturity bucket.
        """
        path = Path(self._cfg["haircut_table_path"])
        logger.info("Loading haircut table from %s", path)
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "haircut_pct" in df.columns:
            df["haircut_pct"] = pd.to_numeric(df["haircut_pct"], errors="coerce")

        self._haircuts = df
        logger.info("Haircut table loaded: %d rows", len(df))
        return df

    def load_transactions(self) -> pd.DataFrame:
        """Load the transactions CSV.

        Expected columns:
            transaction_id, counterparty_id, required_value,
            financing_rate_bps, settlement_date, current_collateral_id

        Returns
        -------
        pd.DataFrame
            Live / historical transaction records with the current
            collateral allocation.
        """
        path = Path(self._cfg["transaction_path"])
        logger.info("Loading transactions from %s", path)
        df = pd.read_csv(path, parse_dates=["settlement_date"])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        for col in ("required_value", "financing_rate_bps"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._transactions = df
        logger.info("Transactions loaded: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_all(self) -> pd.DataFrame:
        """Load all sources (if not already loaded) and join them.

        Join strategy:
            1. Start with collateral pool.
            2. LEFT JOIN eligibility schedule on ``asset_type``
               (takes the first matching eligibility rule; a more
               sophisticated implementation would join on counterparty_id
               as well, which requires a transaction context).
            3. LEFT JOIN haircut table on ``asset_type`` (aggregated to a
               single haircut per asset_type via the mean of
               ``haircut_pct``).
            4. LEFT JOIN transactions on any shared key present in both
               frames (falls back to a cross-join key if no shared column
               is found — useful for single-transaction runs).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame ready for feature engineering.  Missing
            numeric values are forward-filled then filled with 0.
        """
        # Ensure all sources are loaded
        if self._collateral_pool is None:
            self.load_collateral_pool()
        if self._eligibility is None:
            self.load_eligibility_schedule()
        if self._haircuts is None:
            self.load_haircut_table()
        if self._transactions is None:
            self.load_transactions()

        df = self._collateral_pool.copy()

        # --- eligibility: deduplicate to one rule per asset_type ------
        elig_dedup = (
            self._eligibility.drop_duplicates(subset=["asset_type"])
            [["asset_type", "eligible", "min_credit_rating", "max_maturity_years"]]
        )
        df = df.merge(elig_dedup, on="asset_type", how="left")

        # --- haircut: aggregate to mean haircut per asset_type --------
        haircut_agg = (
            self._haircuts.groupby("asset_type", as_index=False)["haircut_pct"]
            .mean()
        )
        df = df.merge(haircut_agg, on="asset_type", how="left")

        # --- transactions: broadcast first transaction's rate to all rows
        # (In production this join would be per counterparty_id)
        if not self._transactions.empty:
            tx = self._transactions.iloc[[0]][["financing_rate_bps", "required_value"]]
            for col in tx.columns:
                df[col] = tx[col].values[0]
        else:
            df["financing_rate_bps"] = 0.0
            df["required_value"] = 0.0

        # Fill remaining nulls
        df["eligible"] = df["eligible"].fillna(False)
        df["haircut_pct"] = df["haircut_pct"].fillna(0.0)
        df["financing_rate_bps"] = df["financing_rate_bps"].fillna(0.0)

        logger.info("Merged dataset shape: %s", df.shape)
        return df
