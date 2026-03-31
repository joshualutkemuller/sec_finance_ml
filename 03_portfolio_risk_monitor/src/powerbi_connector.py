"""PowerBI connector for the Portfolio Concentration Risk Early Warning Monitor."""

import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PowerBIConnector:
    """Publishes portfolio risk scores to the Financing Solutions Intelligence Platform.

    Push Dataset schema: portfolio_id, sector, counterparty, concentration_score,
    risk_level, alert_flag, timestamp.
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

        self._tenant_id = os.environ["POWERBI_TENANT_ID"]
        self._client_id = os.environ["POWERBI_CLIENT_ID"]
        self._client_secret = os.environ["POWERBI_CLIENT_SECRET"]
        self._token: str | None = None

    def _get_access_token(self) -> str:
        url = self._TOKEN_URL.format(tenant_id=self._tenant_id)
        resp = requests.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "https://analysis.windows.net/powerbi/api/.default",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Token error: {resp.status_code} {resp.text}")
        self._token = resp.json()["access_token"]
        return self._token

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token or self._get_access_token()}",
            "Content-Type": "application/json",
        }

    def push_rows(self, rows: list[dict[str, Any]]) -> bool:
        """POST rows to a PowerBI Push Dataset table."""
        if not self.enabled or not rows:
            return False
        url = self._PUSH_URL.format(
            workspace_id=self.workspace_id,
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        try:
            resp = requests.post(url, json={"rows": rows}, headers=self._auth_headers(), timeout=30)
            if resp.status_code == 401:
                self._token = None
                resp = requests.post(url, json={"rows": rows}, headers=self._auth_headers(), timeout=30)
            success = resp.status_code == 200
            if not success:
                logger.error("PowerBI push failed: %s %s", resp.status_code, resp.text)
            return success
        except requests.RequestException as exc:
            logger.error("PowerBI push exception: %s", exc)
            return False

    def stream_row(self, row: dict[str, Any]) -> bool:
        """POST a row to a PowerBI Streaming Dataset."""
        if not self.enabled or not self.streaming_url:
            return False
        try:
            resp = requests.post(self.streaming_url, json=[row], timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def publish_risk_scores(self, df: pd.DataFrame) -> None:
        """Publish risk scores DataFrame to PowerBI.

        Args:
            df: Must contain portfolio_id, sector, counterparty,
                concentration_score, risk_level, alert_flag, timestamp.
        """
        if not self.enabled:
            return
        required = {"portfolio_id", "sector", "counterparty",
                    "concentration_score", "risk_level", "alert_flag", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing columns: %s", missing)
            return
        rows = df[list(required)].to_dict(orient="records")
        self.push_rows(rows)
        if rows:
            self.stream_row(rows[-1])
