"""
data_ingestion.py
-----------------
Handles loading lending rate data from multiple source types (CSV, Databricks,
REST API) and validates the resulting DataFrame against the expected schema
defined in config/config.yaml.

Classes
-------
LendingRateDataLoader
    Dispatcher that routes to the appropriate loader based on
    ``config['data']['source_type']``.

Functions
---------
load_config(config_path)
    Convenience function to parse a YAML configuration file into a dict.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Configuration loaded from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Data loader class
# ---------------------------------------------------------------------------


class LendingRateDataLoader:
    """Load securities lending borrow-rate data from configured sources.

    The class supports three source types controlled by
    ``config['data']['source_type']``:

    - ``"csv"``        — pandas ``read_csv`` from a local/network path.
    - ``"databricks"`` — executes a SQL query via the Databricks SDK or a
                         live Spark session.
    - ``"api"``        — fetches JSON or CSV data from a REST endpoint via
                         the ``requests`` library.

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration dictionary (typically the result of
        :func:`load_config`).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_cfg = config.get("data", {})

        self.source_type: str = self.data_cfg.get("source_type", "csv")
        self.ticker_col: str = self.data_cfg.get("ticker_col", "ticker")
        self.rate_col: str = self.data_cfg.get("rate_col", "borrow_rate")
        self.date_col: str = self.data_cfg.get("date_col", "date")
        self.lookback_days: int = int(self.data_cfg.get("lookback_days", 365))

        logger.debug(
            "LendingRateDataLoader initialised — source_type=%s, lookback_days=%d",
            self.source_type,
            self.lookback_days,
        )

    # ------------------------------------------------------------------
    # Public source-specific loaders
    # ------------------------------------------------------------------

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load lending rate data from a CSV file.

        The CSV is expected to contain at minimum the columns specified by
        ``ticker_col``, ``rate_col``, and ``date_col`` in the configuration.

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Raw lending rate data with the date column parsed as
            ``datetime64[ns]`` and sorted ascending by date.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        logger.info("Loading data from CSV: %s", path)
        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)

        # Apply lookback filter
        df = self._apply_lookback_filter(df)

        logger.info("CSV load complete — %d rows loaded", len(df))
        return df

    def load_from_databricks(self, query: str) -> pd.DataFrame:
        """Load lending rate data by executing a SQL query on Databricks.

        Attempts to use the ``databricks-sdk`` WorkspaceClient first.
        Falls back to a Spark session obtained from ``pyspark.sql.SparkSession``
        if the SDK is unavailable or ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN``
        are not set.

        Parameters
        ----------
        query : str
            SQL query to execute (e.g. ``SELECT * FROM lending.borrow_rates``).

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame with the date column parsed
            and sorted ascending.

        Raises
        ------
        RuntimeError
            If neither ``databricks-sdk`` nor a live Spark session is available.
        """
        logger.info("Loading data from Databricks with query: %.120s...", query)

        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        # --- Attempt 1: databricks-sdk StatementExecutionAPI ---
        try:
            from databricks.sdk import WorkspaceClient  # type: ignore[import]
            from databricks.sdk.service.sql import StatementState  # type: ignore[import]

            client = WorkspaceClient(
                host=databricks_host,
                token=databricks_token,
            )
            warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
            response = client.statement_execution.execute_statement(
                statement=query,
                warehouse_id=warehouse_id,
                wait_timeout="30s",
            )
            if response.status.state != StatementState.SUCCEEDED:
                raise RuntimeError(
                    f"Databricks statement failed: {response.status.error}"
                )

            schema = [
                col.name
                for col in response.manifest.schema.columns  # type: ignore[union-attr]
            ]
            rows = [
                list(row.values)
                for row in (response.result.data_array or [])
            ]
            df = pd.DataFrame(rows, columns=schema)

        except ImportError:
            logger.warning(
                "databricks-sdk not installed — falling back to SparkSession"
            )
            # --- Attempt 2: Active Spark session (Databricks Runtime / cluster) ---
            try:
                from pyspark.sql import SparkSession  # type: ignore[import]

                spark = SparkSession.getActiveSession()
                if spark is None:
                    raise RuntimeError("No active Spark session found.")

                sdf = spark.sql(query)
                df = sdf.toPandas()

            except ImportError as exc:
                raise RuntimeError(
                    "Neither databricks-sdk nor pyspark is available. "
                    "Install databricks-sdk or run inside a Databricks cluster."
                ) from exc

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        df = self._apply_lookback_filter(df)

        logger.info("Databricks load complete — %d rows loaded", len(df))
        return df

    def load_from_api(self, endpoint: str) -> pd.DataFrame:
        """Load lending rate data from a REST API endpoint.

        Sends a GET request to ``endpoint``. The response is expected to be
        either:
        - JSON array of objects (each object maps column names to values), or
        - CSV text (if ``Content-Type`` is ``text/csv``).

        Optional query parameters (e.g. date range filters) should be encoded
        into ``endpoint`` prior to calling this method.

        Parameters
        ----------
        endpoint : str
            Full URL of the REST API endpoint.

        Returns
        -------
        pd.DataFrame
            Lending rate data with the date column parsed and sorted ascending.

        Raises
        ------
        requests.HTTPError
            If the server returns a non-2xx HTTP status code.
        ValueError
            If the response body cannot be parsed as JSON or CSV.
        """
        import requests  # local import to keep startup lightweight

        logger.info("Loading data from API endpoint: %s", endpoint)

        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        api_token = os.getenv("API_TOKEN")
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/csv" in content_type:
            import io

            df = pd.read_csv(io.StringIO(response.text), parse_dates=[self.date_col])
        else:
            # Default: parse as JSON
            data = response.json()
            if isinstance(data, dict):
                # e.g. { "data": [...] }
                records = data.get("data", data.get("rows", data.get("results", [])))
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(
                    f"Unexpected JSON structure from API — got {type(data)}"
                )
            df = pd.DataFrame(records)
            df[self.date_col] = pd.to_datetime(df[self.date_col])

        df = df.sort_values(self.date_col).reset_index(drop=True)
        df = self._apply_lookback_filter(df)

        logger.info("API load complete — %d rows loaded", len(df))
        return df

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame contains the required columns and
        meets basic data quality criteria.

        Checks performed
        ~~~~~~~~~~~~~~~~
        - Required columns (``ticker_col``, ``rate_col``, ``date_col``) are present.
        - ``rate_col`` is numeric (float or int dtype).
        - ``date_col`` is a datetime dtype.
        - No rows have a null value in ``rate_col``.
        - At least one row is present.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        bool
            ``True`` if all checks pass, ``False`` otherwise. Validation
            failures are logged at WARNING level rather than raising exceptions
            so that the caller can decide how to proceed.
        """
        passed = True

        required_cols = [self.ticker_col, self.rate_col, self.date_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning("Schema validation failed — missing columns: %s", missing)
            passed = False

        if len(df) == 0:
            logger.warning("Schema validation failed — DataFrame is empty")
            passed = False

        if self.rate_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.rate_col]):
                logger.warning(
                    "Schema validation failed — %s is not numeric (dtype=%s)",
                    self.rate_col,
                    df[self.rate_col].dtype,
                )
                passed = False

            null_count = df[self.rate_col].isna().sum()
            if null_count > 0:
                logger.warning(
                    "Schema validation warning — %d null values in %s",
                    null_count,
                    self.rate_col,
                )
                # Warn but do not fail — callers can decide to drop/fill nulls

        if self.date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                logger.warning(
                    "Schema validation failed — %s is not datetime (dtype=%s)",
                    self.date_col,
                    df[self.date_col].dtype,
                )
                passed = False

        if passed:
            logger.info(
                "Schema validation passed — %d rows, %d columns", len(df), len(df.columns)
            )
        return passed

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load data from the configured source and validate the schema.

        Dispatches to :meth:`load_from_csv`, :meth:`load_from_databricks`, or
        :meth:`load_from_api` based on ``config['data']['source_type']``.
        After loading, :meth:`validate_schema` is called and a warning is
        logged if validation fails (the raw DataFrame is still returned so
        callers can apply their own remediation).

        Returns
        -------
        pd.DataFrame
            Validated lending rate data sorted ascending by date.

        Raises
        ------
        ValueError
            If ``source_type`` is not one of ``"csv"``, ``"databricks"``,
            ``"api"``.
        """
        path: str = self.data_cfg.get("path", "")

        if self.source_type == "csv":
            df = self.load_from_csv(path)

        elif self.source_type == "databricks":
            # The path field doubles as the SQL query for Databricks
            query = self.data_cfg.get("query", f"SELECT * FROM {path}")
            df = self.load_from_databricks(query)

        elif self.source_type == "api":
            df = self.load_from_api(path)

        else:
            raise ValueError(
                f"Unknown source_type '{self.source_type}'. "
                "Must be one of: 'csv', 'databricks', 'api'."
            )

        self.validate_schema(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_lookback_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame to the configured lookback window.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted DataFrame containing a datetime column named
            ``self.date_col``.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only rows within the most recent
            ``lookback_days`` calendar days.
        """
        if self.date_col not in df.columns:
            return df

        cutoff = df[self.date_col].max() - pd.Timedelta(days=self.lookback_days)
        filtered = df[df[self.date_col] >= cutoff].reset_index(drop=True)

        dropped = len(df) - len(filtered)
        if dropped > 0:
            logger.debug(
                "Lookback filter: dropped %d rows older than %s",
                dropped,
                cutoff.date(),
            )
        return filtered
