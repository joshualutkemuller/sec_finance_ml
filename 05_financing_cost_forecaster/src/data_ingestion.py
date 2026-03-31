"""
data_ingestion.py — Financing Cost Forecaster data loading layer.

Provides FinancingDataLoader, which reads repo rates, macro rates, book
positions, and budget data from CSV files (or future API adapters) and
merges them into a single analysis-ready DataFrame.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinancingDataLoader:
    """
    Loads and merges all data sources required for financing cost forecasting.

    Parameters
    ----------
    config : dict
        The ``data`` section of config.yaml, with keys:
        repo_rates_path, macro_rates_path, positions_path, budget_path,
        lookback_days.
    base_dir : str or Path, optional
        Base directory from which relative paths in config are resolved.
        Defaults to the current working directory.
    """

    REPO_RATE_COLS: list[str] = ["date", "instrument_type", "rate_bps", "tenor"]
    MACRO_RATE_COLS: list[str] = ["date", "sofr", "fed_funds", "ois", "libor_3m"]
    POSITION_COLS: list[str] = ["date", "instrument_type", "notional"]
    BUDGET_COLS: list[str] = ["instrument_type", "budget_rate_bps"]

    def __init__(self, config: dict, base_dir: Optional[str | Path] = None) -> None:
        self.config = config
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.lookback_days: int = config.get("lookback_days", 365)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Return an absolute path, resolving relative paths against base_dir."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def _read_csv(self, path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Read a CSV with optional date parsing and basic validation.

        Parameters
        ----------
        path : str
            Path key as stored in config (may be relative).
        parse_dates : list[str], optional
            Column names to parse as datetime.

        Returns
        -------
        pd.DataFrame
        """
        full_path = self._resolve(path)
        if not full_path.exists():
            logger.warning("File not found: %s — returning empty DataFrame", full_path)
            return pd.DataFrame()
        logger.debug("Loading %s", full_path)
        df = pd.read_csv(full_path, parse_dates=parse_dates or [])
        return df

    def _apply_lookback(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Filter DataFrame to the configured lookback window."""
        if df.empty or date_col not in df.columns:
            return df
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=self.lookback_days)
        cutoff = cutoff.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = df[date_col].dt.tz_localize(None)
        return df[df[date_col] >= cutoff].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_repo_rates(self) -> pd.DataFrame:
        """
        Load historical repo rates.

        Returns
        -------
        pd.DataFrame
            Columns: date (datetime64), instrument_type (str),
            rate_bps (float), tenor (str).
        """
        path = self.config.get("repo_rates_path", "data/repo_rates.csv")
        df = self._read_csv(path, parse_dates=["date"])
        if df.empty:
            df = self._synthetic_repo_rates()
        df = self._apply_lookback(df)
        df["rate_bps"] = pd.to_numeric(df["rate_bps"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["instrument_type", "date"], inplace=True)
        logger.info("Loaded %d repo rate rows", len(df))
        return df.reset_index(drop=True)

    def load_macro_rates(self) -> pd.DataFrame:
        """
        Load macro rate time series (SOFR, Fed Funds, OIS, LIBOR-3M).

        Returns
        -------
        pd.DataFrame
            Columns: date (datetime64), sofr (float), fed_funds (float),
            ois (float), libor_3m (float).
        """
        path = self.config.get("macro_rates_path", "data/macro_rates.csv")
        df = self._read_csv(path, parse_dates=["date"])
        if df.empty:
            df = self._synthetic_macro_rates()
        df = self._apply_lookback(df)
        for col in ["sofr", "fed_funds", "ois", "libor_3m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        logger.info("Loaded %d macro rate rows", len(df))
        return df.reset_index(drop=True)

    def load_positions(self) -> pd.DataFrame:
        """
        Load book position data (demand-pressure proxy).

        Returns
        -------
        pd.DataFrame
            Columns: date (datetime64), instrument_type (str),
            notional (float).
        """
        path = self.config.get("positions_path", "data/positions.csv")
        df = self._read_csv(path, parse_dates=["date"])
        if df.empty:
            df = self._synthetic_positions()
        df = self._apply_lookback(df)
        df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["instrument_type", "date"], inplace=True)
        logger.info("Loaded %d position rows", len(df))
        return df.reset_index(drop=True)

    def load_budget(self) -> pd.DataFrame:
        """
        Load budget rate data (static per instrument type).

        Returns
        -------
        pd.DataFrame
            Columns: instrument_type (str), budget_rate_bps (float).
        """
        path = self.config.get("budget_path", "data/budget.csv")
        df = self._read_csv(path)
        if df.empty:
            df = self._synthetic_budget()
        df["budget_rate_bps"] = pd.to_numeric(df["budget_rate_bps"], errors="coerce")
        logger.info("Loaded %d budget rows", len(df))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_all(self) -> pd.DataFrame:
        """
        Load all data sources and merge them into a single analysis-ready
        DataFrame indexed by (date, instrument_type).

        Merge strategy:
        1. repo_rates LEFT JOIN macro_rates on date.
        2. Result LEFT JOIN positions on (date, instrument_type).
        3. Result LEFT JOIN budget on instrument_type.

        Returns
        -------
        pd.DataFrame
            Fully merged DataFrame sorted by instrument_type and date.
        """
        repo = self.load_repo_rates()
        macro = self.load_macro_rates()
        positions = self.load_positions()
        budget = self.load_budget()

        if repo.empty:
            logger.warning("repo_rates is empty; returning empty merged DataFrame")
            return pd.DataFrame()

        # Step 1: merge macro on date
        df = repo.merge(macro, on="date", how="left")

        # Step 2: merge positions on date + instrument_type
        if not positions.empty:
            df = df.merge(positions, on=["date", "instrument_type"], how="left")
        else:
            df["notional"] = np.nan

        # Step 3: merge budget on instrument_type (static)
        if not budget.empty:
            df = df.merge(budget, on="instrument_type", how="left")
        else:
            df["budget_rate_bps"] = np.nan

        df.sort_values(["instrument_type", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "Merged DataFrame: %d rows, %d columns, instruments: %s",
            len(df),
            len(df.columns),
            df["instrument_type"].unique().tolist(),
        )
        return df

    # ------------------------------------------------------------------
    # Synthetic data generators (used when CSVs are absent)
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_repo_rates() -> pd.DataFrame:
        """Generate synthetic repo rate data for development and testing."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=400)
        instrument_types = ["UST_GC", "Agency_GC", "Corp_Specials", "Equity_Specials"]
        records = []
        base_rates = {"UST_GC": 530, "Agency_GC": 545, "Corp_Specials": 590, "Equity_Specials": 650}
        for inst in instrument_types:
            rate = float(base_rates[inst])
            for dt in dates:
                rate += float(rng.normal(0, 2.5))
                rate = max(0, rate)
                records.append(
                    {"date": dt, "instrument_type": inst, "rate_bps": round(rate, 2), "tenor": "O/N"}
                )
        return pd.DataFrame(records)

    @staticmethod
    def _synthetic_macro_rates() -> pd.DataFrame:
        """Generate synthetic macro rate data for development and testing."""
        rng = np.random.default_rng(43)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=400)
        sofr = 5.30
        records = []
        for dt in dates:
            sofr += float(rng.normal(0, 0.005))
            sofr = max(0, sofr)
            records.append(
                {
                    "date": dt,
                    "sofr": round(sofr, 4),
                    "fed_funds": round(sofr + 0.01, 4),
                    "ois": round(sofr - 0.005, 4),
                    "libor_3m": round(sofr + 0.15, 4),
                }
            )
        return pd.DataFrame(records)

    @staticmethod
    def _synthetic_positions() -> pd.DataFrame:
        """Generate synthetic position notional data."""
        rng = np.random.default_rng(44)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=400)
        instrument_types = ["UST_GC", "Agency_GC", "Corp_Specials", "Equity_Specials"]
        records = []
        for inst in instrument_types:
            base_notional = rng.uniform(100e6, 2e9)
            for dt in dates:
                notional = base_notional * rng.uniform(0.9, 1.1)
                records.append({"date": dt, "instrument_type": inst, "notional": round(notional, 0)})
        return pd.DataFrame(records)

    @staticmethod
    def _synthetic_budget() -> pd.DataFrame:
        """Generate static budget rate data per instrument."""
        return pd.DataFrame(
            [
                {"instrument_type": "UST_GC", "budget_rate_bps": 535.0},
                {"instrument_type": "Agency_GC", "budget_rate_bps": 550.0},
                {"instrument_type": "Corp_Specials", "budget_rate_bps": 600.0},
                {"instrument_type": "Equity_Specials", "budget_rate_bps": 660.0},
            ]
        )
