"""
Macro Alert Dispatcher — Sends regime-change and spread-spike alerts via
Email (SMTP), Microsoft Teams webhook, and structured JSONL log.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "INFO": "#3498db",
    "WARNING": "#f39c12",
    "CRITICAL": "#e74c3c",
}


class MacroAlertDispatcher:
    """Multi-channel alert dispatcher for macro regime events."""

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("alerts", {})
        self.enabled = self.cfg.get("enabled", True)
        self.log_path = Path(self.cfg.get("log_path", "logs/macro_alerts.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_regime: str | None = None

    def dispatch_regime_change(
        self, current_regime: str, regime_df: pd.DataFrame
    ) -> None:
        """Fire alert if regime has changed since last run."""
        if not self.cfg.get("regime_change_alert", True):
            return

        prev_regime = regime_df["regime_label"].iloc[-2] if len(regime_df) > 1 else current_regime
        if current_regime == prev_regime:
            return

        severity = "CRITICAL" if current_regime == "Stress / Crisis" else "WARNING"
        alert = {
            "alert_id": str(uuid.uuid4()),
            "alert_type": "REGIME_CHANGE",
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "from_regime": prev_regime,
            "to_regime": current_regime,
            "message": f"Macro regime changed: {prev_regime} → {current_regime}",
        }
        self._dispatch(alert)

    def dispatch_spread_spike(
        self,
        lgbm_forecasts: dict[str, dict[int, np.ndarray]],
        features: pd.DataFrame,
    ) -> None:
        """Alert when any spread forecast exceeds its z-score threshold."""
        threshold = self.cfg.get("spread_spike_threshold_z", 2.0)
        spike_cols = {
            "hy_oas": "hy_oas_z60",
            "ig_oas": "ig_oas_z60",
            "sofr": "sofr_z60",
        }
        latest = features.iloc[-1]
        for target, z_col in spike_cols.items():
            if z_col not in latest.index:
                continue
            z_val = float(latest[z_col])
            if abs(z_val) > threshold:
                severity = "CRITICAL" if abs(z_val) > threshold * 1.5 else "WARNING"
                forecast_1d = float(lgbm_forecasts.get(target, {}).get(1, [np.nan])[-1])
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "alert_type": "SPREAD_SPIKE",
                    "severity": severity,
                    "timestamp": datetime.utcnow().isoformat(),
                    "target": target.upper(),
                    "z_score": round(z_val, 3),
                    "forecast_1d": round(forecast_1d, 4) if not np.isnan(forecast_1d) else None,
                    "message": (
                        f"{target.upper()} z-score={z_val:+.2f} exceeds threshold ±{threshold}"
                    ),
                }
                self._dispatch(alert)

    def _dispatch(self, alert: dict) -> None:
        """Route alert to all configured channels."""
        if not self.enabled:
            return

        # Always write to structured log
        with open(self.log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")
        logger.info("[%s] %s: %s", alert["severity"], alert["alert_type"], alert["message"])

        min_severity = self.cfg.get("min_severity", "INFO")
        severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        if severity_order.get(alert["severity"], 0) < severity_order.get(min_severity, 0):
            return

        self._send_email(alert)
        self._send_teams(alert)

    def _send_email(self, alert: dict) -> None:
        email_cfg = self.cfg.get("email", {})
        smtp_host = os.environ.get("SMTP_HOST", email_cfg.get("smtp_host", ""))
        smtp_port = int(os.environ.get("SMTP_PORT", email_cfg.get("port", 587)))
        smtp_user = os.environ.get("SMTP_USER", email_cfg.get("user", ""))
        smtp_pass = os.environ.get("SMTP_PASSWORD", "")
        from_addr = os.environ.get("ALERT_FROM", email_cfg.get("from_addr", smtp_user))
        recipients_raw = os.environ.get(
            "ALERT_EMAIL_RECIPIENTS",
            email_cfg.get("recipients", ""),
        )
        recipients = [r.strip() for r in str(recipients_raw).split(",") if r.strip()]

        if not smtp_host or not recipients:
            return

        color = SEVERITY_COLORS.get(alert["severity"], "#95a5a6")
        html_body = f"""
        <html><body>
        <div style='border-left:4px solid {color};padding:12px;font-family:Arial,sans-serif'>
          <h3 style='color:{color}'>[{alert['severity']}] {alert['alert_type']}</h3>
          <p>{alert['message']}</p>
          <p style='color:#7f8c8d;font-size:12px'>{alert['timestamp']} UTC</p>
        </div>
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[MACRO ALERT] {alert['alert_type']} — {alert['severity']}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, recipients, msg.as_string())
            logger.debug("Email alert sent: %s", alert["alert_type"])
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)

    def _send_teams(self, alert: dict) -> None:
        webhook_url = os.environ.get(
            "TEAMS_WEBHOOK_URL", self.cfg.get("teams_webhook_url", "")
        )
        if not webhook_url or webhook_url.startswith("${"):
            return

        color = SEVERITY_COLORS.get(alert["severity"], "#95a5a6").lstrip("#")
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": alert["message"],
            "sections": [{
                "activityTitle": f"[{alert['severity']}] {alert['alert_type']}",
                "activityText": alert["message"],
                "facts": [
                    {"name": k, "value": str(v)}
                    for k, v in alert.items()
                    if k not in ("alert_id", "alert_type", "severity", "message")
                ],
            }],
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Teams alert failed: %s", exc)
