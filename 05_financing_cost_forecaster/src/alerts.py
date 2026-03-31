"""Alert system for the Repo & Financing Cost Forecaster.

Fires alerts when:
  - A forecasted rate exceeds the spike threshold (bps jump vs current).
  - A forecasted rate breaches the budget allocation.
  - A rapid sequential rate increase is detected across horizons.
"""

import logging
import smtplib
import os
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class ForecastAlertType(Enum):
    """Classification of financing cost alert types."""
    RATE_SPIKE = "RATE_SPIKE"
    BUDGET_BREACH = "BUDGET_BREACH"
    RAPID_INCREASE = "RAPID_INCREASE"


@dataclass
class ForecastAlert:
    """Represents a single financing cost forecast alert.

    Attributes:
        instrument_type: Repo/financing instrument identifier.
        current_rate_bps: Current observed rate in basis points.
        forecast_bps: Forecasted rate in basis points.
        horizon_days: Forecast horizon (1, 3, or 5 days).
        budget_bps: Budget/target rate in basis points.
        breach_bps: Amount by which forecast exceeds budget.
        alert_type: Classification of the alert.
        timestamp: UTC timestamp when the alert was generated.
        message: Human-readable alert description.
    """
    instrument_type: str
    current_rate_bps: float
    forecast_bps: float
    horizon_days: int
    budget_bps: float
    breach_bps: float
    alert_type: ForecastAlertType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"[{self.alert_type.value}] {self.instrument_type} | "
                f"Forecast ({self.horizon_days}d): {self.forecast_bps:.1f} bps | "
                f"Current: {self.current_rate_bps:.1f} bps | "
                f"Budget: {self.budget_bps:.1f} bps | "
                f"Breach: {self.breach_bps:+.1f} bps"
            )


class ForecastAlertFactory:
    """Creates ForecastAlert objects from forecast vs budget comparison."""

    def from_forecasts(
        self,
        forecast_df: pd.DataFrame,
        budget_df: pd.DataFrame,
        config: dict,
    ) -> list[ForecastAlert]:
        """Generate alerts for rate spikes and budget breaches.

        Args:
            forecast_df: DataFrame with columns: instrument_type,
                current_rate_bps, forecast_1d_bps, forecast_3d_bps,
                forecast_5d_bps, timestamp.
            budget_df: DataFrame with columns: instrument_type, budget_rate_bps.
            config: Config dict containing alerts section with
                spike_threshold_bps and budget_breach_threshold_bps.

        Returns:
            List of ForecastAlert objects.
        """
        alerts: list[ForecastAlert] = []
        alert_cfg = config.get("alerts", {})
        spike_thresh = alert_cfg.get("spike_threshold_bps", 20.0)
        budget_breach_thresh = alert_cfg.get("budget_breach_threshold_bps", 10.0)

        # Merge budget into forecast df
        merged = forecast_df.merge(budget_df, on="instrument_type", how="left")
        merged["budget_rate_bps"] = merged.get("budget_rate_bps", float("nan"))

        for _, row in merged.iterrows():
            instrument = row["instrument_type"]
            current = float(row.get("current_rate_bps", 0.0))
            budget = float(row.get("budget_rate_bps", float("nan")))
            ts = row.get("timestamp", datetime.utcnow())

            for horizon, col in [(1, "forecast_1d_bps"), (3, "forecast_3d_bps"),
                                  (5, "forecast_5d_bps")]:
                if col not in row.index:
                    continue
                forecast = float(row[col])

                # Rate spike alert
                spike = forecast - current
                if spike >= spike_thresh:
                    alerts.append(ForecastAlert(
                        instrument_type=instrument,
                        current_rate_bps=current,
                        forecast_bps=forecast,
                        horizon_days=horizon,
                        budget_bps=budget,
                        breach_bps=forecast - budget if not pd.isna(budget) else 0.0,
                        alert_type=ForecastAlertType.RATE_SPIKE,
                        timestamp=ts,
                    ))

                # Budget breach alert
                if not pd.isna(budget) and (forecast - budget) >= budget_breach_thresh:
                    alerts.append(ForecastAlert(
                        instrument_type=instrument,
                        current_rate_bps=current,
                        forecast_bps=forecast,
                        horizon_days=horizon,
                        budget_bps=budget,
                        breach_bps=forecast - budget,
                        alert_type=ForecastAlertType.BUDGET_BREACH,
                        timestamp=ts,
                    ))

        return alerts


class ForecastAlertDispatcher:
    """Dispatches financing cost alerts via email, MS Teams, and log file.

    Args:
        config: The ``alerts`` section of config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.enabled: bool = config.get("enabled", True)

    def _send_email(self, alert: ForecastAlert) -> bool:
        """Send an alert via SMTP email.

        Returns:
            True if successful, False otherwise.
        """
        email_cfg = self.config.get("email", {})
        if not email_cfg:
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"[FSIP Alert] {alert.alert_type.value} — {alert.instrument_type}"
            )
            msg["From"] = os.environ.get("SMTP_USER", email_cfg.get("from", ""))
            recipients = os.environ.get(
                "ALERT_EMAIL_RECIPIENTS",
                ",".join(email_cfg.get("to", [])),
            )
            msg["To"] = recipients
            body = MIMEText(alert.message, "plain")
            msg.attach(body)

            smtp_host = os.environ.get("SMTP_HOST", email_cfg.get("smtp_host", "localhost"))
            smtp_port = int(os.environ.get("SMTP_PORT", email_cfg.get("port", 587)))
            smtp_user = os.environ.get("SMTP_USER", "")
            smtp_password = os.environ.get("SMTP_PASSWORD", "")

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], recipients.split(","), msg.as_string())
            return True
        except smtplib.SMTPException as exc:
            logger.error("Email send failed: %s", exc)
            return False

    def _send_teams(self, alert: ForecastAlert) -> bool:
        """Post alert to MS Teams via Incoming Webhook.

        Returns:
            True if successful, False otherwise.
        """
        webhook_url = os.environ.get(
            "TEAMS_WEBHOOK_URL", self.config.get("teams_webhook_url", "")
        )
        if not webhook_url:
            return False
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000" if alert.alert_type == ForecastAlertType.BUDGET_BREACH else "FFA500",
            "summary": f"FSIP Financing Alert: {alert.instrument_type}",
            "sections": [
                {
                    "activityTitle": f"**{alert.alert_type.value}** — {alert.instrument_type}",
                    "facts": [
                        {"name": "Horizon", "value": f"{alert.horizon_days}d"},
                        {"name": "Current Rate", "value": f"{alert.current_rate_bps:.1f} bps"},
                        {"name": "Forecast Rate", "value": f"{alert.forecast_bps:.1f} bps"},
                        {"name": "Budget Rate", "value": f"{alert.budget_bps:.1f} bps"},
                        {"name": "Breach", "value": f"{alert.breach_bps:+.1f} bps"},
                        {"name": "Timestamp", "value": str(alert.timestamp)},
                    ],
                }
            ],
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except requests.RequestException as exc:
            logger.error("Teams webhook failed: %s", exc)
            return False

    def _log_alert(self, alert: ForecastAlert) -> None:
        """Write alert to the configured log file."""
        log_path = self.config.get("log_path", "alerts.log")
        try:
            with open(log_path, "a") as f:
                f.write(f"{alert.timestamp.isoformat()}  {alert.message}\n")
        except OSError as exc:
            logger.warning("Could not write to alert log: %s", exc)

    def dispatch(self, alerts: list[ForecastAlert]) -> None:
        """Send all alerts through every configured channel.

        Args:
            alerts: List of ForecastAlert objects to dispatch.
        """
        if not self.enabled or not alerts:
            return
        for alert in alerts:
            self._log_alert(alert)
            self._send_email(alert)
            self._send_teams(alert)
            logger.info("Alert dispatched: %s", alert.message)
