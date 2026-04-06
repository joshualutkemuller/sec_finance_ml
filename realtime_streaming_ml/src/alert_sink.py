"""
Alert Sink — Sub-500ms alert dispatcher for streaming anomaly events.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class AlertSink:
    """
    Asynchronous multi-channel alert dispatcher designed for < 500ms latency.
    Uses a background thread per channel to avoid blocking the main stream loop.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("alerts", {})
        self.log_path = Path(self.cfg.get("log_path", "logs/streaming_alerts.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._enabled = self.cfg.get("enabled", True)
        self._target_ms = self.cfg.get("target_latency_ms", 500)

    def send(self, alert: dict) -> None:
        """Route alert asynchronously. Non-blocking."""
        if not self._enabled:
            return

        alert["alert_id"] = f"{int(time.time() * 1000)}"
        alert["dispatch_utc"] = datetime.now(timezone.utc).isoformat()

        # Always write to log synchronously (fast — local disk write)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

        logger.warning("[%s] %s | %s | rate=%.5f | sigma=%.2f",
                       alert.get("severity", "?"),
                       alert.get("alert_type", "?"),
                       alert.get("ticker", "?"),
                       alert.get("rate", 0),
                       alert.get("sigma_multiple", 0))

        # Teams async (fire and forget)
        threading.Thread(
            target=self._send_teams, args=(alert,), daemon=True
        ).start()

    def _send_teams(self, alert: dict) -> None:
        t0 = time.time()
        webhook = os.environ.get("TEAMS_WEBHOOK_URL", self.cfg.get("teams_webhook_url", ""))
        if not webhook or webhook.startswith("${"):
            return

        severity_color = {"CRITICAL": "FF0000", "WARNING": "FFA500"}.get(
            alert.get("severity", "INFO"), "0078D4"
        )
        facts = [{"name": k, "value": str(v)} for k, v in alert.items()
                 if k not in ("dispatch_utc", "alert_id")]

        payload = {
            "@type": "MessageCard",
            "themeColor": severity_color,
            "summary": f"[{alert.get('severity')}] {alert.get('alert_type')} — {alert.get('ticker')}",
            "sections": [{"activityTitle": f"Streaming Anomaly: {alert.get('ticker')}",
                          "facts": facts[:8]}],
        }
        try:
            requests.post(webhook, json=payload, timeout=0.4)
            elapsed_ms = (time.time() - t0) * 1000
            if elapsed_ms > self._target_ms:
                logger.warning("Teams alert exceeded latency target: %.0fms", elapsed_ms)
        except Exception as exc:
            logger.debug("Teams alert failed: %s", exc)
