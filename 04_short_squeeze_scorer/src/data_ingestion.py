"""
data_ingestion.py — Load, validate, and merge all data sources required for
short squeeze feature engineering.

Data sources:
    - Short interest data  (ticker, date, shares_short, shares_float, short_interest_ratio)
    - Borrow availability  (ticker, date, available_shares, utilization_rate, borrow_rate)
    - Price / volume data  (ticker, date, open, high, low, close, volume)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected schema definitions
# ---------------------------------------------------------------------------

_SHORT_INTEREST_COLUMNS: list[str] = [
    "ticker",
    "date",
    "shares_short",
    "shares_float",
    "short_interest_ratio",
]

_BORROW_COLUMNS: list[str] = [
    "ticker",
    "date",
    "available_shares",
    "utilization_rate",
    "borrow_rate",
]

_PRICE_COLUMNS: list[str] = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class ShortInterestDataLoader:
    """Load and merge short interest, borrow availability, and price/volume data.

    All three data sources must share a common (ticker, date) key. The loader
    validates each source schema before attempting a merge, raising a
    ``ValueError`` for any missing required column.

    Args:
        config: The ``data`` block from ``config.yaml`` as a plain dict.
            Expected keys: ``short_interest_path``, ``borrow_availability_path``,
            ``price_data_path``, ``lookback_days``.

    Example::

        loader = ShortInterestDataLoader(config["data"])
        df = loader.merge_all()
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._short_interest_path = Path(config["short_interest_path"])
        self._borrow_path = Path(config["borrow_availability_path"])
        self._price_path = Path(config["price_data_path"])
        self._lookback_days: int = int(config.get("lookback_days", 90))

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_short_interest(self) -> pd.DataFrame:
        """Load short interest data from the configured CSV path.

        Returns a DataFrame with columns:
            ticker (str), date (datetime64), shares_short (float),
            shares_float (float), short_interest_ratio (float).

        Returns:
            pd.DataFrame: Short interest data filtered to the lookback window.

        Raises:
            FileNotFoundError: If the configured path does not exist.
            ValueError: If required columns are missing.
        """
        logger.info("Loading short interest data from %s", self._short_interest_path)
        df = self._read_csv(self._short_interest_path)
        df = self._parse_dates(df)
        df = self._filter_lookback(df)

        if not self.validate_schema(df, _SHORT_INTEREST_COLUMNS):
            raise ValueError(
                f"Short interest data is missing required columns. "
                f"Expected: {_SHORT_INTEREST_COLUMNS}, got: {df.columns.tolist()}"
            )

        # Derive short_interest_ratio if not present but source columns are
        if "short_interest_ratio" not in df.columns and {
            "shares_short",
            "shares_float",
        }.issubset(df.columns):
            df["short_interest_ratio"] = df["shares_short"] / df["shares_float"].replace(
                0, np.nan
            )

        logger.info(
            "Loaded %d short interest rows for %d tickers.",
            len(df),
            df["ticker"].nunique(),
        )
        return df[_SHORT_INTEREST_COLUMNS].copy()

    def load_borrow_availability(self) -> pd.DataFrame:
        """Load borrow availability data from the configured CSV path.

        Returns a DataFrame with columns:
            ticker (str), date (datetime64), available_shares (float),
            utilization_rate (float), borrow_rate (float).

        Returns:
            pd.DataFrame: Borrow availability data filtered to lookback window.

        Raises:
            FileNotFoundError: If the configured path does not exist.
            ValueError: If required columns are missing.
        """
        logger.info("Loading borrow availability data from %s", self._borrow_path)
        df = self._read_csv(self._borrow_path)
        df = self._parse_dates(df)
        df = self._filter_lookback(df)

        if not self.validate_schema(df, _BORROW_COLUMNS):
            raise ValueError(
                f"Borrow availability data is missing required columns. "
                f"Expected: {_BORROW_COLUMNS}, got: {df.columns.tolist()}"
            )

        logger.info(
            "Loaded %d borrow rows for %d tickers.",
            len(df),
            df["ticker"].nunique(),
        )
        return df[_BORROW_COLUMNS].copy()

    def load_price_data(self) -> pd.DataFrame:
        """Load OHLCV price data from the configured CSV path.

        Returns a DataFrame with columns:
            ticker (str), date (datetime64), open (float), high (float),
            low (float), close (float), volume (float).

        Returns:
            pd.DataFrame: Price data filtered to the lookback window.

        Raises:
            FileNotFoundError: If the configured path does not exist.
            ValueError: If required columns are missing.
        """
        logger.info("Loading price data from %s", self._price_path)
        df = self._read_csv(self._price_path)
        df = self._parse_dates(df)
        df = self._filter_lookback(df)

        if not self.validate_schema(df, _PRICE_COLUMNS):
            raise ValueError(
                f"Price data is missing required columns. "
                f"Expected: {_PRICE_COLUMNS}, got: {df.columns.tolist()}"
            )

        logger.info(
            "Loaded %d price rows for %d tickers.",
            len(df),
            df["ticker"].nunique(),
        )
        return df[_PRICE_COLUMNS].copy()

    def merge_all(self) -> pd.DataFrame:
        """Load and merge all three data sources on (ticker, date).

        Performs inner joins so only ticker/date pairs present across all
        three sources are retained. Logs a warning if the merge produces
        fewer tickers than the short interest source.

        Returns:
            pd.DataFrame: Unified DataFrame with all columns from all three
                sources, sorted by (ticker, date).

        Raises:
            ValueError: If any source fails schema validation.
        """
        si_df = self.load_short_interest()
        borrow_df = self.load_borrow_availability()
        price_df = self.load_price_data()

        logger.info("Merging data sources on (ticker, date).")

        merged = si_df.merge(borrow_df, on=["ticker", "date"], how="inner")
        merged = merged.merge(price_df, on=["ticker", "date"], how="inner")

        original_tickers = si_df["ticker"].nunique()
        merged_tickers = merged["ticker"].nunique()
        if merged_tickers < original_tickers:
            logger.warning(
                "Merge dropped %d tickers (no matching borrow/price data). "
                "%d tickers remain.",
                original_tickers - merged_tickers,
                merged_tickers,
            )

        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        logger.info(
            "Merged dataset: %d rows, %d tickers.", len(merged), merged["ticker"].nunique()
        )
        return merged

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
    ) -> bool:
        """Validate that a DataFrame contains all required columns.

        Also checks that the DataFrame is non-empty and that the ``date``
        and ``ticker`` columns (when present) have no null values.

        Args:
            df: DataFrame to validate.
            required_columns: List of column names that must be present.
                Defaults to the union of all three source schemas.

        Returns:
            bool: ``True`` if validation passes, ``False`` otherwise.
        """
        if required_columns is None:
            required_columns = list(
                dict.fromkeys(
                    _SHORT_INTEREST_COLUMNS + _BORROW_COLUMNS + _PRICE_COLUMNS
                )
            )

        if df.empty:
            logger.warning("DataFrame is empty.")
            return False

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning("Missing columns: %s", missing)
            return False

        for key_col in ("ticker", "date"):
            if key_col in df.columns and df[key_col].isnull().any():
                logger.warning("Null values found in key column '%s'.", key_col)
                return False

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read a CSV file, raising FileNotFoundError with a clear message."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the ``date`` column to ``datetime64[ns]``."""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
        return df

    def _filter_lookback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only rows within the configured lookback window."""
        if "date" not in df.columns:
            return df
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(
            days=self._lookback_days
        )
        return df[df["date"] >= cutoff].copy()
