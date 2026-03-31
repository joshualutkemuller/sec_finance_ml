"""
data_ingestion.py — Portfolio Concentration Risk Early Warning Monitor

Responsible for loading raw data sources (positions, market data, counterparty
exposures) from CSV files and joining them into a unified exposure snapshot
DataFrame that is consumed by downstream feature engineering steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioDataLoader:
    """Load and join the three core data sources for portfolio risk monitoring.

    Parameters
    ----------
    config : dict
        The ``data`` section of config.yaml, expected keys:
        ``positions_path``, ``market_data_path``, ``counterparty_path``,
        ``lookback_days``.
    """

    # Required columns per source — used for validation after load
    _POSITIONS_REQUIRED: list[str] = [
        "portfolio_id",
        "isin",
        "sector",
        "counterparty",
        "quantity",
        "date",
    ]
    _MARKET_DATA_REQUIRED: list[str] = [
        "isin",
        "date",
        "price",
        "bid_ask_spread",
        "volatility_30d",
    ]
    _COUNTERPARTY_REQUIRED: list[str] = [
        "counterparty",
        "date",
        "credit_limit",
        "current_exposure",
    ]

    def __init__(self, config: dict) -> None:
        self._config = config
        self._positions_path = Path(config["positions_path"])
        self._market_data_path = Path(config["market_data_path"])
        self._counterparty_path = Path(config["counterparty_path"])
        self._lookback_days: int = int(config.get("lookback_days", 30))

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_positions(self) -> pd.DataFrame:
        """Load portfolio positions from CSV.

        Returns
        -------
        pd.DataFrame
            Positions with columns: portfolio_id, isin, sector, counterparty,
            quantity, date.

        Raises
        ------
        FileNotFoundError
            If the positions file does not exist at the configured path.
        ValueError
            If required columns are missing from the file.
        """
        logger.info("Loading positions from %s", self._positions_path)
        df = self._read_csv(self._positions_path)
        self._validate_columns(df, self._POSITIONS_REQUIRED, "positions")
        df["date"] = pd.to_datetime(df["date"])
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df = df.dropna(subset=["portfolio_id", "isin", "quantity"])
        logger.info("Loaded %d position records", len(df))
        return df

    def load_market_data(self) -> pd.DataFrame:
        """Load market data (prices, spreads, volatility) from CSV.

        Returns
        -------
        pd.DataFrame
            Market data with columns: isin, date, price, bid_ask_spread,
            volatility_30d.
        """
        logger.info("Loading market data from %s", self._market_data_path)
        df = self._read_csv(self._market_data_path)
        self._validate_columns(df, self._MARKET_DATA_REQUIRED, "market_data")
        df["date"] = pd.to_datetime(df["date"])
        for col in ["price", "bid_ask_spread", "volatility_30d"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Guard: negative prices are data errors
        invalid_price_mask = df["price"] <= 0
        if invalid_price_mask.any():
            logger.warning(
                "Dropping %d market data rows with non-positive price",
                invalid_price_mask.sum(),
            )
            df = df[~invalid_price_mask]
        logger.info("Loaded %d market data records", len(df))
        return df

    def load_counterparty_data(self) -> pd.DataFrame:
        """Load counterparty exposure limits and current exposures from CSV.

        Returns
        -------
        pd.DataFrame
            Counterparty data with columns: counterparty, date, credit_limit,
            current_exposure.
        """
        logger.info("Loading counterparty data from %s", self._counterparty_path)
        df = self._read_csv(self._counterparty_path)
        self._validate_columns(df, self._COUNTERPARTY_REQUIRED, "counterparty")
        df["date"] = pd.to_datetime(df["date"])
        df["credit_limit"] = pd.to_numeric(df["credit_limit"], errors="coerce")
        df["current_exposure"] = pd.to_numeric(
            df["current_exposure"], errors="coerce"
        )
        # Derive headroom ratio: how much of the credit limit is consumed
        df["credit_utilisation"] = np.where(
            df["credit_limit"] > 0,
            df["current_exposure"] / df["credit_limit"],
            np.nan,
        )
        logger.info("Loaded %d counterparty records", len(df))
        return df

    def build_exposure_snapshot(self) -> pd.DataFrame:
        """Join positions, market data, and counterparty data into a single
        exposure snapshot DataFrame.

        The join logic:
        1. Load all three sources.
        2. Join positions → market data on (isin, date).  When a market data
           row is missing for the exact date the most recent prior price is
           forward-filled within the lookback window.
        3. Compute ``market_value = quantity * price``.
        4. Join counterparty credit utilisation on (counterparty, date).
        5. Derive ``aum_weight`` per portfolio (market_value / total portfolio
           market_value).

        Returns
        -------
        pd.DataFrame
            Enriched exposure snapshot with all columns needed by
            :class:`~src.feature_engineering.ConcentrationFeatureEngineer`.

        Raises
        ------
        ValueError
            If the resulting snapshot is empty after joins.
        """
        positions = self.load_positions()
        market_data = self.load_market_data()
        counterparty = self.load_counterparty_data()

        # -- Join positions with market data (left join to preserve all positions)
        logger.info("Joining positions with market data")
        snap = positions.merge(
            market_data[
                ["isin", "date", "price", "bid_ask_spread", "volatility_30d"]
            ],
            on=["isin", "date"],
            how="left",
        )

        # Forward-fill missing prices within each isin using market data history
        if snap["price"].isna().any():
            logger.debug("Forward-filling missing prices from market data history")
            latest_prices = (
                market_data.sort_values("date")
                .groupby("isin")[["isin", "date", "price", "bid_ask_spread", "volatility_30d"]]
                .last()
                .reset_index(drop=True)
            )
            missing_mask = snap["price"].isna()
            fill_map = latest_prices.set_index("isin")
            snap.loc[missing_mask, "price"] = snap.loc[missing_mask, "isin"].map(
                fill_map["price"]
            )
            snap.loc[missing_mask, "bid_ask_spread"] = snap.loc[
                missing_mask, "isin"
            ].map(fill_map["bid_ask_spread"])
            snap.loc[missing_mask, "volatility_30d"] = snap.loc[
                missing_mask, "isin"
            ].map(fill_map["volatility_30d"])

        # -- Compute market value
        snap["market_value"] = snap["quantity"] * snap["price"]

        # -- AUM weight per portfolio
        portfolio_aum = (
            snap.groupby(["portfolio_id", "date"])["market_value"]
            .sum()
            .rename("portfolio_aum")
            .reset_index()
        )
        snap = snap.merge(portfolio_aum, on=["portfolio_id", "date"], how="left")
        snap["aum_weight"] = np.where(
            snap["portfolio_aum"] > 0,
            snap["market_value"] / snap["portfolio_aum"],
            0.0,
        )

        # -- Join counterparty utilisation
        logger.info("Joining counterparty credit utilisation")
        snap = snap.merge(
            counterparty[["counterparty", "date", "credit_utilisation", "credit_limit"]],
            on=["counterparty", "date"],
            how="left",
        )
        snap["credit_utilisation"] = snap["credit_utilisation"].fillna(0.0)

        if snap.empty:
            raise ValueError(
                "Exposure snapshot is empty after joins — check data sources."
            )

        logger.info(
            "Exposure snapshot built: %d rows, %d columns",
            len(snap),
            snap.shape[1],
        )
        return snap.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a CSV file with basic error handling.

        Parameters
        ----------
        path : Path
            File system path to the CSV.

        Returns
        -------
        pd.DataFrame
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise IOError(f"Failed to read {path}: {exc}") from exc
        return df

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame, required: list[str], source_name: str
    ) -> None:
        """Assert that all required columns are present in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        required : list[str]
            Column names that must be present.
        source_name : str
            Human-readable source name for the error message.

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Source '{source_name}' is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
