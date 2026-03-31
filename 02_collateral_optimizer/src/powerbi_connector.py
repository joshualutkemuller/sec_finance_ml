"""
powerbi_connector.py — PowerBIConnector

Pushes collateral optimization results to a PowerBI push dataset via the
PowerBI REST API.  Supports both streaming (one row at a time) and batch
push_rows operations.

Authentication uses the OAuth2 client_credentials flow; credentials are
read from environment variables ``POWERBI_CLIENT_ID``,
``POWERBI_CLIENT_SECRET``, and ``POWERBI_TENANT_ID``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# PowerBI push dataset row schema
POWERBI_COLUMNS = [
    "collateral_id",
    "asset_type",
    "haircut",
    "quality_score",
    "recommended",
    "optimization_cost",
    "timestamp",
]


class PowerBIConnector:
    """Push collateral optimization results to a PowerBI push dataset.

    Parameters
    ----------
    config : dict
        The ``powerbi`` sub-section of ``config.yaml``.  Required keys:
        ``workspace_id``, ``dataset_id``, ``table_name``,
        ``streaming_url``, ``enabled``.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._enabled: bool = config.get("enabled", False)
        self._workspace_id: str = config.get("workspace_id", "")
        self._dataset_id: str = config.get("dataset_id", "")
        self._table_name: str = config.get("table_name", "collateral_recommendations")
        self._streaming_url: str = config.get("streaming_url", "")
        self._auth_url_template: str = config.get(
            "auth_url",
            "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        )
        self._scope: str = config.get(
            "scope", "https://analysis.windows.net/powerbi/api/.default"
        )
        self._access_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Obtain an OAuth2 access token via client_credentials flow.

        Reads the following environment variables (populated by
        ``python-dotenv`` / ``.env``):
            - ``POWERBI_CLIENT_ID``
            - ``POWERBI_CLIENT_SECRET``
            - ``POWERBI_TENANT_ID``

        Returns
        -------
        str
            Bearer token string.

        Raises
        ------
        RuntimeError
            If any required credential environment variable is missing
            or the token request fails.
        """
        if self._access_token:
            return self._access_token

        client_id = os.getenv("POWERBI_CLIENT_ID", "")
        client_secret = os.getenv("POWERBI_CLIENT_SECRET", "")
        tenant_id = os.getenv("POWERBI_TENANT_ID", "")

        missing = [
            name
            for name, val in (
                ("POWERBI_CLIENT_ID", client_id),
                ("POWERBI_CLIENT_SECRET", client_secret),
                ("POWERBI_TENANT_ID", tenant_id),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing PowerBI credential environment variables: {missing}"
            )

        auth_url = self._auth_url_template.format(tenant_id=tenant_id)
        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": self._scope,
        }

        resp = requests.post(auth_url, data=payload, timeout=30)
        resp.raise_for_status()
        token_data = resp.json()

        if "access_token" not in token_data:
            raise RuntimeError(
                f"Token response did not contain access_token: {token_data}"
            )

        self._access_token = token_data["access_token"]
        logger.info("PowerBI access token obtained successfully.")
        return self._access_token

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_push_url(self) -> str:
        """Construct the PowerBI REST API rows endpoint URL.

        Returns
        -------
        str
            URL for POST /datasets/{datasetId}/tables/{tableName}/rows.
        """
        return (
            f"https://api.powerbi.com/v1.0/myorg/groups/{self._workspace_id}"
            f"/datasets/{self._dataset_id}/tables/{self._table_name}/rows"
        )

    @staticmethod
    def _row_to_pbi_dict(row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a recommendation dict to the PowerBI column schema.

        Parameters
        ----------
        row : dict
            A recommendation dict from
            :meth:`~model.CollateralOptimizer.generate_recommendations`.

        Returns
        -------
        dict
            Dict with exactly the keys defined in :data:`POWERBI_COLUMNS`.
        """
        timestamp = row.get("timestamp") or datetime.now(tz=timezone.utc).isoformat()
        return {
            "collateral_id": str(row.get("collateral_id", "UNKNOWN")),
            "asset_type": str(row.get("asset_type", "UNKNOWN")),
            "haircut": float(row.get("haircut", 0.0)),
            "quality_score": float(row.get("quality_score", 0.0)),
            "recommended": bool(row.get("recommended", False)),
            "optimization_cost": float(row.get("optimization_cost", 0.0)),
            "timestamp": str(timestamp),
        }

    # ------------------------------------------------------------------
    # Push methods
    # ------------------------------------------------------------------

    def push_rows(self, rows: List[Dict[str, Any]]) -> bool:
        """Batch-push a list of rows to the PowerBI push dataset.

        Uses the standard REST API endpoint (not streaming).  The
        endpoint accepts up to 10 000 rows per request; this method
        sends all rows in a single call.

        Parameters
        ----------
        rows : list of dict
            Each dict must be mappable via :meth:`_row_to_pbi_dict`.

        Returns
        -------
        bool
            ``True`` if the push succeeded, ``False`` otherwise.
        """
        if not self._enabled:
            logger.info("PowerBI push disabled; skipping push_rows.")
            return False

        if not rows:
            logger.warning("push_rows called with empty rows list; nothing to push.")
            return False

        try:
            token = self._get_access_token()
        except RuntimeError as exc:
            logger.error("Cannot push rows — authentication failed: %s", exc)
            return False

        pbi_rows = [self._row_to_pbi_dict(r) for r in rows]
        url = self._build_push_url()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"rows": pbi_rows}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            logger.info(
                "PowerBI push_rows succeeded; pushed %d rows (HTTP %d)",
                len(pbi_rows),
                resp.status_code,
            )
            return True
        except requests.RequestException as exc:
            logger.error("PowerBI push_rows failed: %s", exc)
            return False

    def stream_row(self, row: Dict[str, Any]) -> bool:
        """Stream a single row to the PowerBI streaming push dataset.

        Uses the ``streaming_url`` from config (which bypasses OAuth2
        and uses the dataset-specific push URL with an embedded key).

        Parameters
        ----------
        row : dict
            A single recommendation record.

        Returns
        -------
        bool
            ``True`` if the stream succeeded, ``False`` otherwise.
        """
        if not self._enabled:
            logger.debug("PowerBI push disabled; skipping stream_row.")
            return False

        streaming_url = self._streaming_url
        if not streaming_url or "REPLACE_ME" in streaming_url:
            logger.warning("streaming_url not configured; skipping stream_row.")
            return False

        pbi_row = self._row_to_pbi_dict(row)
        payload = [pbi_row]  # streaming API expects a JSON array

        try:
            resp = requests.post(streaming_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.debug(
                "Streamed row for collateral_id=%s (HTTP %d)",
                pbi_row["collateral_id"],
                resp.status_code,
            )
            return True
        except requests.RequestException as exc:
            logger.error("stream_row failed for %s: %s", pbi_row["collateral_id"], exc)
            return False

    def publish_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> bool:
        """Publish a full recommendations list to PowerBI.

        Attempts batch push first; streams individual rows as a fallback
        if the batch push fails (e.g. when using a streaming dataset that
        does not support the batch endpoint).

        Parameters
        ----------
        recommendations : list of dict
            Output of
            :meth:`~model.CollateralOptimizer.generate_recommendations`.

        Returns
        -------
        bool
            ``True`` if at least the streaming fallback partially
            succeeded, ``False`` if all channels failed.
        """
        if not self._enabled:
            logger.info("PowerBI push disabled; skipping publish_recommendations.")
            return False

        if not recommendations:
            logger.info("No recommendations to publish to PowerBI.")
            return True

        # Try batch push first
        success = self.push_rows(recommendations)
        if success:
            return True

        # Fallback: stream one row at a time
        logger.warning(
            "Batch push failed; falling back to row-by-row streaming for %d rows.",
            len(recommendations),
        )
        streamed = 0
        for rec in recommendations:
            if self.stream_row(rec):
                streamed += 1

        logger.info("Streamed %d / %d rows via fallback.", streamed, len(recommendations))
        return streamed > 0

    def clear_table(self) -> bool:
        """Delete all rows from the push dataset table.

        Useful for daily refresh workflows where the table should be
        truncated before pushing a new batch.

        Returns
        -------
        bool
            ``True`` if the DELETE succeeded, ``False`` otherwise.
        """
        if not self._enabled:
            return False

        try:
            token = self._get_access_token()
        except RuntimeError as exc:
            logger.error("Cannot clear table — authentication failed: %s", exc)
            return False

        url = self._build_push_url()
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.delete(url, headers=headers, timeout=30)
            resp.raise_for_status()
            logger.info("PowerBI table '%s' cleared (HTTP %d)", self._table_name, resp.status_code)
            return True
        except requests.RequestException as exc:
            logger.error("Failed to clear PowerBI table: %s", exc)
            return False
