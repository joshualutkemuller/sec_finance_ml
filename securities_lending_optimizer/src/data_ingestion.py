"""
Data Ingestion — Loads loan book, collateral pool, market data, and counterparty data.
Supports CSV, Databricks SQL, and REST API sources.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

REQUIRED_LOAN_BOOK_COLS = {"ticker", "date", "counterparty_id"}
REQUIRED_COLLATERAL_COLS = {"ticker", "date", "asset_class"}


class LendingDataIngestion:
    """Unified data loader for the enhanced securities lending optimizer."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["data"]
        self.lookback = self.cfg.get("lookback_days", 504)
        self.cutoff = datetime.utcnow() - timedelta(days=self.lookback)

    def load_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all four datasets.

        Returns:
            (loan_book, collateral_pool, market_data, counterparties)
        """
        source = self.cfg.get("source_type", "csv")
        if source == "csv":
            return self._load_csv()
        elif source == "databricks":
            return self._load_databricks()
        elif source == "api":
            return self._load_api()
        else:
            raise ValueError(f"Unknown source_type: {source}")

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _load_csv(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def _read(path: str, required: set[str]) -> pd.DataFrame:
            df = pd.read_csv(path, parse_dates=["date"])
            missing = required - set(df.columns)
            if missing:
                logger.warning("Missing columns in %s: %s", path, missing)
            return df

        loan_book = _read(self.cfg["loan_book_path"], REQUIRED_LOAN_BOOK_COLS)
        collateral = _read(self.cfg["collateral_pool_path"], REQUIRED_COLLATERAL_COLS)
        market_data = pd.read_csv(self.cfg["market_data_path"], parse_dates=["date"])
        counterparties = pd.read_csv(self.cfg["counterparty_path"])

        loan_book = loan_book[loan_book["date"] >= self.cutoff]
        collateral = collateral[collateral["date"] >= self.cutoff]
        market_data = market_data[market_data["date"] >= self.cutoff]

        logger.info(
            "CSV loaded — loan_book=%d, collateral=%d, market=%d, counterparties=%d",
            len(loan_book), len(collateral), len(market_data), len(counterparties),
        )
        return loan_book, collateral, market_data, counterparties

    # ── Databricks ────────────────────────────────────────────────────────────

    def _load_databricks(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            from databricks import sql as dbsql
        except ImportError as exc:
            raise ImportError("Install databricks-sql-connector: pip install databricks-sql-connector") from exc

        host = os.environ["DATABRICKS_HOST"]
        token = os.environ["DATABRICKS_TOKEN"]
        http_path = os.environ["DATABRICKS_HTTP_PATH"]
        cutoff_str = self.cutoff.strftime("%Y-%m-%d")

        with dbsql.connect(server_hostname=host, http_path=http_path,
                           access_token=token) as conn:
            def _query(sql: str) -> pd.DataFrame:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return pd.DataFrame(cur.fetchall(),
                                        columns=[d[0] for d in cur.description])

            loan_book = _query(f"SELECT * FROM fsip.loan_book WHERE date >= '{cutoff_str}'")
            collateral = _query(f"SELECT * FROM fsip.collateral_pool WHERE date >= '{cutoff_str}'")
            market_data = _query(f"SELECT * FROM fsip.market_data WHERE date >= '{cutoff_str}'")
            counterparties = _query("SELECT * FROM fsip.counterparties")

        for df in [loan_book, collateral, market_data]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        return loan_book, collateral, market_data, counterparties

    # ── REST API ──────────────────────────────────────────────────────────────

    def _load_api(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base_url = self.cfg.get("api_base_url", "")
        headers = {"Authorization": f"Bearer {os.environ.get('API_TOKEN', '')}"}
        cutoff_str = self.cutoff.strftime("%Y-%m-%d")

        def _fetch(endpoint: str) -> pd.DataFrame:
            url = f"{base_url}{endpoint}?from={cutoff_str}"
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return pd.DataFrame(resp.json())

        loan_book = _fetch("/v1/loan_book")
        collateral = _fetch("/v1/collateral_pool")
        market_data = _fetch("/v1/market_data")
        counterparties = pd.DataFrame(
            requests.get(f"{base_url}/v1/counterparties", headers=headers, timeout=30).json()
        )

        return loan_book, collateral, market_data, counterparties
