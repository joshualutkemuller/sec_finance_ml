"""PowerBI connector for the Securities Lending Rate Anomaly Detector.

Pushes scored anomaly results to the Financing Solutions Intelligence Platform
via the PowerBI REST API (Push Datasets and Streaming Datasets).
"""

import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PowerBIConnector:
    """Authenticates with Azure AD and publishes rows to PowerBI datasets.

    Supports both Push Datasets (historical rows queryable in reports) and
    Streaming Datasets (real-time dashboard tiles).

    Args:
        config: The ``powerbi`` section of config.yaml.
    """

    _TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    _PUSH_URL = (
        "https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}"
        "/datasets/{dataset_id}/tables/{table_name}/rows"
    )

    def __init__(self, config: dict) -> None:
        self.config = config
        self.workspace_id: str = config["workspace_id"]
        self.dataset_id: str = config["dataset_id"]
        self.table_name: str = config["table_name"]
        self.streaming_url: str = config.get("streaming_url", "")
        self.enabled: bool = config.get("enabled", True)

        # Read credentials from environment (never from config file)
        self._tenant_id: str = os.environ["POWERBI_TENANT_ID"]
        self._client_id: str = os.environ["POWERBI_CLIENT_ID"]
        self._client_secret: str = os.environ["POWERBI_CLIENT_SECRET"]
        self._token: str | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Obtain a Bearer token via Azure AD client credentials flow.

        Returns:
            Access token string.

        Raises:
            RuntimeError: If token acquisition fails.
        """
        url = self._TOKEN_URL.format(tenant_id=self._tenant_id)
        payload = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        }
        resp = requests.post(url, data=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to obtain PowerBI token: {resp.status_code} {resp.text}"
            )
        self._token = resp.json()["access_token"]
        return self._token

    def _auth_headers(self) -> dict[str, str]:
        token = self._token or self._get_access_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Push Dataset (queryable historical rows)
    # ------------------------------------------------------------------

    def push_rows(self, rows: list[dict[str, Any]]) -> bool:
        """POST rows to a PowerBI Push Dataset table.

        Args:
            rows: List of dicts matching the dataset table schema:
                  ticker, rate, anomaly_score, alert_flag, timestamp.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled or not rows:
            return False

        url = self._PUSH_URL.format(
            workspace_id=self.workspace_id,
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        payload = {"rows": rows}
        try:
            resp = requests.post(
                url, json=payload, headers=self._auth_headers(), timeout=30
            )
            if resp.status_code == 200:
                logger.info("Pushed %d rows to PowerBI dataset.", len(rows))
                return True
            # Token may have expired — refresh once and retry
            if resp.status_code == 401:
                self._token = None
                resp = requests.post(
                    url, json=payload, headers=self._auth_headers(), timeout=30
                )
                if resp.status_code == 200:
                    return True
            logger.error("PowerBI push failed: %s %s", resp.status_code, resp.text)
            return False
        except requests.RequestException as exc:
            logger.error("PowerBI push exception: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Streaming Dataset (real-time tiles)
    # ------------------------------------------------------------------

    def stream_row(self, row: dict[str, Any]) -> bool:
        """POST a single row to a PowerBI Streaming Dataset endpoint.

        Streaming datasets power real-time dashboard tiles and do not require
        authentication (URL contains a built-in key).

        Args:
            row: Dict matching the streaming dataset schema.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled or not self.streaming_url:
            return False
        try:
            resp = requests.post(
                self.streaming_url, json=[row], timeout=10
            )
            if resp.status_code == 200:
                return True
            logger.warning(
                "Streaming push failed: %s %s", resp.status_code, resp.text
            )
            return False
        except requests.RequestException as exc:
            logger.warning("Streaming push exception: %s", exc)
            return False

    # ------------------------------------------------------------------
    # High-level publisher
    # ------------------------------------------------------------------

    def publish_alerts(self, df: pd.DataFrame) -> None:
        """Convert a scored DataFrame and publish to PowerBI.

        Pushes all rows to the Push Dataset and streams the most recent
        row to the Streaming Dataset tile.

        Args:
            df: DataFrame with columns: ticker, rate, anomaly_score,
                alert_flag, timestamp.
        """
        if not self.enabled:
            logger.debug("PowerBI connector is disabled; skipping publish.")
            return

        required_cols = {"ticker", "rate", "anomaly_score", "alert_flag", "timestamp"}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error("DataFrame missing columns for PowerBI publish: %s", missing)
            return

        rows = df[list(required_cols)].to_dict(orient="records")
        self.push_rows(rows)

        # Stream the latest row for real-time tile refresh
        if rows:
            self.stream_row(rows[-1])
