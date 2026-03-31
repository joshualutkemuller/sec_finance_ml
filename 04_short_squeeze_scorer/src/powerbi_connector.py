"""
powerbi_connector.py — Push scored squeeze results to a PowerBI streaming
push dataset on the Financing Solutions Intelligence Platform.

The connector targets the PowerBI Push Datasets REST API
(``POST /datasets/{datasetId}/tables/{tableName}/rows``).  For streaming
datasets the full ``streaming_url`` can be provided directly.

Push dataset schema (per row):
    ticker                  string
    short_interest_ratio    float
    days_to_cover           float
    utilization_rate        float
    squeeze_probability     float
    risk_tier               string
    alert_flag              boolean
    timestamp               ISO-8601 datetime string

Credentials:
    POWERBI_ACCESS_TOKEN — Bearer token read from the environment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# PowerBI REST API base URL
_POWERBI_API_BASE = "https://api.powerbi.com/v1.0/myorg"

# Columns required in the input DataFrame before pushing
_REQUIRED_COLUMNS: list[str] = [
    "ticker",
    "short_interest_ratio",
    "days_to_cover",
    "utilization_rate",
    "squeeze_probability",
    "risk_tier",
    "alert_flag",
]


class PowerBIConnector:
    """Push scored short squeeze results to a PowerBI push dataset.

    Supports two modes:
        1. **Streaming URL mode** — POST directly to a pre-formed streaming
           endpoint URL stored in ``powerbi.streaming_url`` config key.
        2. **Workspace/Dataset mode** — Construct the rows endpoint from
           ``workspace_id``, ``dataset_id``, and ``table_name``.

    Rows are sent in configurable batches (``powerbi.batch_size``) to
    respect PowerBI API limits.

    Args:
        config: The ``powerbi`` block from ``config.yaml``.

    Example::

        connector = PowerBIConnector(config["powerbi"])
        connector.push_scores(scored_df)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._enabled: bool = bool(config.get("enabled", True))
        self._batch_size: int = int(config.get("batch_size", 500))
        self._table_name: str = config.get("table_name", "ShortSqueezeScores")

        # Resolve endpoint
        streaming_url: str = os.path.expandvars(
            config.get("streaming_url", "")
        )
        workspace_id: str = os.path.expandvars(
            config.get("workspace_id", "")
        )
        dataset_id: str = os.path.expandvars(
            config.get("dataset_id", "")
        )

        if streaming_url and not streaming_url.startswith("${"):
            self._endpoint: str = streaming_url
        elif workspace_id and dataset_id:
            self._endpoint = (
                f"{_POWERBI_API_BASE}/groups/{workspace_id}"
                f"/datasets/{dataset_id}/tables/{self._table_name}/rows"
            )
        else:
            self._endpoint = ""
            logger.warning(
                "PowerBI endpoint not configured. Push calls will be no-ops."
            )

        self._access_token: str = os.getenv("POWERBI_ACCESS_TOKEN", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push_scores(self, df: pd.DataFrame) -> int:
        """Serialise and push all scored rows to PowerBI.

        Validates required columns, formats the payload, and posts in
        batches.  Returns the total number of rows successfully pushed.

        Args:
            df: Scored DataFrame containing at minimum the columns defined
                in ``_REQUIRED_COLUMNS``.  A ``timestamp`` column is added
                automatically if absent.

        Returns:
            int: Number of rows successfully pushed.

        Raises:
            ValueError: If required columns are missing from ``df``.
        """
        if not self._enabled:
            logger.info("PowerBI push disabled. Skipping.")
            return 0

        if not self._endpoint:
            logger.warning("No PowerBI endpoint configured. Skipping push.")
            return 0

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required PowerBI columns: {missing}"
            )

        push_df = df.copy()
        if "timestamp" not in push_df.columns:
            push_df["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        rows = self._prepare_rows(push_df)
        total_pushed = self._push_in_batches(rows)
        logger.info("PowerBI push complete. %d rows pushed.", total_pushed)
        return total_pushed

    def push_rows(self, rows: list[dict]) -> bool:
        """Push a pre-built list of row dicts to PowerBI.

        Lower-level method for callers that have already serialised the
        payload.  Used by :meth:`push_scores` internally.

        Args:
            rows: List of dicts matching the push dataset schema.

        Returns:
            bool: ``True`` if the HTTP request succeeded, ``False`` otherwise.
        """
        if not self._endpoint:
            logger.warning("No PowerBI endpoint. Cannot push rows.")
            return False

        headers = self._build_headers()
        payload = {"rows": rows}

        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=30,
            )
            if resp.status_code in (200, 201):
                logger.debug(
                    "PowerBI push accepted %d rows. HTTP %d.",
                    len(rows),
                    resp.status_code,
                )
                return True
            else:
                logger.error(
                    "PowerBI push failed. HTTP %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        except requests.RequestException as exc:
            logger.error("PowerBI push network error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_schema_definition() -> dict:
        """Return the PowerBI push dataset table schema definition.

        The returned dict can be used with the PowerBI REST API to create
        or update a push dataset table:
        ``POST /datasets/{id}/tables/{name}``

        Returns:
            dict: PowerBI table schema with column names and data types.
        """
        return {
            "name": "ShortSqueezeScores",
            "columns": [
                {"name": "ticker", "dataType": "string"},
                {"name": "short_interest_ratio", "dataType": "double"},
                {"name": "days_to_cover", "dataType": "double"},
                {"name": "utilization_rate", "dataType": "double"},
                {"name": "squeeze_probability", "dataType": "double"},
                {"name": "risk_tier", "dataType": "string"},
                {"name": "alert_flag", "dataType": "bool"},
                {"name": "timestamp", "dataType": "DateTime"},
            ],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including Bearer token if available."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    @staticmethod
    def _prepare_rows(df: pd.DataFrame) -> list[dict]:
        """Serialise a DataFrame to a list of PowerBI-compatible row dicts.

        Converts ``risk_tier`` enum values to strings, ensures ``alert_flag``
        is a Python bool, and formats timestamps as ISO-8601 strings.

        Args:
            df: Scored DataFrame with all required columns present.

        Returns:
            list[dict]: List of row dicts ready for the PowerBI REST API.
        """
        output_cols = _REQUIRED_COLUMNS + ["timestamp"]
        subset = df[[c for c in output_cols if c in df.columns]].copy()

        # Normalise types
        if "risk_tier" in subset.columns:
            subset["risk_tier"] = subset["risk_tier"].apply(
                lambda v: v.value if hasattr(v, "value") else str(v)
            )
        if "alert_flag" in subset.columns:
            subset["alert_flag"] = subset["alert_flag"].astype(bool)
        if "timestamp" in subset.columns:
            subset["timestamp"] = subset["timestamp"].apply(
                lambda v: v.isoformat() if isinstance(v, datetime) else str(v)
            )

        return subset.to_dict(orient="records")

    def _push_in_batches(self, rows: list[dict]) -> int:
        """Send rows to PowerBI in chunks of ``_batch_size``.

        Args:
            rows: Complete list of rows to push.

        Returns:
            int: Total number of rows from successful batch posts.
        """
        total_pushed = 0
        for start in range(0, len(rows), self._batch_size):
            batch = rows[start : start + self._batch_size]
            success = self.push_rows(batch)
            if success:
                total_pushed += len(batch)
                logger.debug(
                    "Pushed batch rows %d–%d (%d rows).",
                    start,
                    start + len(batch) - 1,
                    len(batch),
                )
            else:
                logger.error(
                    "Batch %d–%d failed. Rows in this batch NOT pushed.",
                    start,
                    start + len(batch) - 1,
                )
        return total_pushed
