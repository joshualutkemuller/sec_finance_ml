"""
alerts.py — Collateral alert dataclasses, dispatcher, and factory.

Three alert channels are supported:
    - Email (SMTP via smtplib / ssl)
    - Microsoft Teams (Incoming Webhook)
    - Structured JSONL log file

Alert levels and types mirror the thresholds defined in ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ===========================================================================
# Enums & Dataclasses
# ===========================================================================


class AlertLevel(Enum):
    """Severity levels for collateral alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Named alert categories."""

    SUBOPTIMAL_COLLATERAL = "SUBOPTIMAL_COLLATERAL"
    OPTIMISATION_FAILURE = "OPTIMISATION_FAILURE"
    COST_SAVINGS_AVAILABLE = "COST_SAVINGS_AVAILABLE"
    CONCENTRATION_BREACH = "CONCENTRATION_BREACH"


@dataclass
class CollateralAlert:
    """Represents a single collateral alert event.

    Attributes
    ----------
    collateral_id : str
        Identifier of the collateral item that triggered the alert.
    asset_type : str
        Asset class (e.g. ``"Govt Bond"``, ``"Equity"``).
    quality_score : float
        ML model quality score for this item (0–1).
    cost_savings_bps : float
        Estimated cost savings in basis points if substitution is made.
    alert_level : AlertLevel
        Severity of the alert.
    alert_type : AlertType
        Category of the alert.
    timestamp : str
        UTC ISO-8601 timestamp when the alert was created.
    message : str
        Human-readable description of the alert.
    metadata : dict
        Optional additional key-value pairs (e.g. allocation_weight).
    """

    collateral_id: str
    asset_type: str
    quality_score: float
    cost_savings_bps: float
    alert_level: AlertLevel
    alert_type: AlertType
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        d = asdict(self)
        d["alert_level"] = self.alert_level.value
        d["alert_type"] = self.alert_type.value
        return d


# ===========================================================================
# Alert Factory
# ===========================================================================


class CollateralAlertFactory:
    """Create :class:`CollateralAlert` instances from pipeline outputs."""

    @staticmethod
    def from_recommendations(
        recommendations: List[Dict[str, Any]],
        config: dict,
    ) -> List[CollateralAlert]:
        """Convert a list of recommendation dicts into alert objects.

        Logic
        -----
        - If ``quality_score < quality_score_threshold`` → CRITICAL /
          SUBOPTIMAL_COLLATERAL.
        - If ``cost_savings_bps >= cost_savings_threshold_bps`` → WARNING or
          CRITICAL (CRITICAL when savings >= 2× threshold) /
          COST_SAVINGS_AVAILABLE.
        - Otherwise → INFO (no alert dispatched unless log-only).

        Parameters
        ----------
        recommendations : list of dict
            Output of
            :meth:`~model.CollateralOptimizer.generate_recommendations`.
        config : dict
            Full pipeline config dict.  Uses ``model.quality_score_threshold``
            and ``alerts.cost_savings_threshold_bps``.

        Returns
        -------
        list of CollateralAlert
        """
        quality_threshold: float = config.get("model", {}).get(
            "quality_score_threshold", 0.55
        )
        savings_threshold: float = config.get("alerts", {}).get(
            "cost_savings_threshold_bps", 5.0
        )

        alerts: List[CollateralAlert] = []
        for rec in recommendations:
            quality_score = float(rec.get("quality_score", 0.0))
            savings_bps = float(rec.get("cost_savings_bps", 0.0))
            collateral_id = rec.get("collateral_id", "UNKNOWN")
            asset_type = rec.get("asset_type", "UNKNOWN")

            # Determine alert type and level
            if quality_score < quality_threshold:
                alert_type = AlertType.SUBOPTIMAL_COLLATERAL
                level = AlertLevel.CRITICAL
                msg = (
                    f"Collateral {collateral_id} ({asset_type}) has quality score "
                    f"{quality_score:.4f}, below threshold {quality_threshold:.4f}."
                )
            elif savings_bps >= 2 * savings_threshold:
                alert_type = AlertType.COST_SAVINGS_AVAILABLE
                level = AlertLevel.CRITICAL
                msg = (
                    f"Significant cost saving of {savings_bps:.2f} bps available "
                    f"by substituting collateral {collateral_id} ({asset_type})."
                )
            elif savings_bps >= savings_threshold:
                alert_type = AlertType.COST_SAVINGS_AVAILABLE
                level = AlertLevel.WARNING
                msg = (
                    f"Cost saving of {savings_bps:.2f} bps available "
                    f"for collateral {collateral_id} ({asset_type})."
                )
            else:
                alert_type = AlertType.COST_SAVINGS_AVAILABLE
                level = AlertLevel.INFO
                msg = (
                    f"Collateral {collateral_id} ({asset_type}) recommended with "
                    f"quality score {quality_score:.4f}."
                )

            alerts.append(
                CollateralAlert(
                    collateral_id=collateral_id,
                    asset_type=asset_type,
                    quality_score=quality_score,
                    cost_savings_bps=savings_bps,
                    alert_level=level,
                    alert_type=alert_type,
                    message=msg,
                    metadata={
                        "allocation_weight": rec.get("allocation_weight"),
                        "optimization_cost": rec.get("optimization_cost"),
                    },
                )
            )

        return alerts

    @staticmethod
    def optimisation_failure_alert(solver_status: str) -> CollateralAlert:
        """Create an OPTIMISATION_FAILURE alert for infeasible solver runs.

        Parameters
        ----------
        solver_status : str
            CVXPY problem status string (e.g. ``"infeasible"``).

        Returns
        -------
        CollateralAlert
        """
        return CollateralAlert(
            collateral_id="N/A",
            asset_type="N/A",
            quality_score=0.0,
            cost_savings_bps=0.0,
            alert_level=AlertLevel.CRITICAL,
            alert_type=AlertType.OPTIMISATION_FAILURE,
            message=(
                f"CVXPY optimisation returned status '{solver_status}'. "
                "No feasible collateral allocation was found. "
                "Review eligibility constraints and haircut limits."
            ),
        )


# ===========================================================================
# Alert Dispatcher
# ===========================================================================


class CollateralAlertDispatcher:
    """Dispatch :class:`CollateralAlert` objects via email, Teams, and log.

    Parameters
    ----------
    config : dict
        The ``alerts`` sub-section of ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._enabled: bool = config.get("enabled", True)
        self._log_path: Path = Path(config.get("log_path", "logs/collateral_alerts.jsonl"))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dispatch router
    # ------------------------------------------------------------------

    def dispatch(self, alerts: List[CollateralAlert]) -> None:
        """Dispatch all alerts through all configured channels.

        Skips dispatch if ``alerts.enabled`` is ``False`` in config.
        INFO-level alerts are written to the log only (not emailed or
        posted to Teams).

        Parameters
        ----------
        alerts : list of CollateralAlert
        """
        if not self._enabled:
            logger.info("Alert dispatch disabled in config; skipping.")
            return

        for alert in alerts:
            self.dispatch_log(alert)
            if alert.alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
                self.dispatch_email(alert)
                self.dispatch_teams(alert)

        logger.info("Dispatched %d alert(s)", len(alerts))

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    def dispatch_email(self, alert: CollateralAlert) -> None:
        """Send an HTML email alert via SMTP.

        Reads SMTP credentials from environment variables
        ``SMTP_USER`` and ``SMTP_PASSWORD`` (loaded via python-dotenv).
        Falls back silently if credentials are absent.

        Parameters
        ----------
        alert : CollateralAlert
        """
        import os
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        email_cfg = self._cfg.get("email", {})
        if not email_cfg.get("enabled", True):
            return

        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        if not smtp_user or not smtp_password:
            logger.warning("SMTP credentials not set; skipping email alert.")
            return

        host = email_cfg.get("smtp_host", "smtp.office365.com")
        port = int(email_cfg.get("smtp_port", 587))
        sender = email_cfg.get("sender", smtp_user)
        recipients: List[str] = email_cfg.get("recipients", [])
        subject_prefix = email_cfg.get("subject_prefix", "[Collateral Optimizer]")

        subject = (
            f"{subject_prefix} [{alert.alert_level.value}] "
            f"{alert.alert_type.value} — {alert.collateral_id}"
        )

        html_body = f"""
        <html><body>
        <h2 style="color:{'red' if alert.alert_level == AlertLevel.CRITICAL else 'orange'}">
            {alert.alert_level.value}: {alert.alert_type.value}
        </h2>
        <table border="1" cellpadding="5">
            <tr><td><b>Collateral ID</b></td><td>{alert.collateral_id}</td></tr>
            <tr><td><b>Asset Type</b></td><td>{alert.asset_type}</td></tr>
            <tr><td><b>Quality Score</b></td><td>{alert.quality_score:.4f}</td></tr>
            <tr><td><b>Cost Savings (bps)</b></td><td>{alert.cost_savings_bps:.2f}</td></tr>
            <tr><td><b>Timestamp</b></td><td>{alert.timestamp}</td></tr>
        </table>
        <p>{alert.message}</p>
        </body></html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.sendmail(sender, recipients, msg.as_string())
            logger.info("Email alert sent for collateral %s", alert.collateral_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send email alert: %s", exc)

    # ------------------------------------------------------------------
    # Microsoft Teams
    # ------------------------------------------------------------------

    def dispatch_teams(self, alert: CollateralAlert) -> None:
        """Post a Teams Adaptive Card via an Incoming Webhook.

        Parameters
        ----------
        alert : CollateralAlert
        """
        webhook_url: str = self._cfg.get("teams_webhook_url", "")
        if not webhook_url or webhook_url.startswith("https://outlook.office.com/webhook/REPLACE"):
            logger.debug("Teams webhook URL not configured; skipping.")
            return

        colour_map = {
            AlertLevel.INFO: "00B0F0",       # blue
            AlertLevel.WARNING: "FFA500",    # orange
            AlertLevel.CRITICAL: "FF0000",   # red
        }
        themeColor = colour_map.get(alert.alert_level, "00B0F0")

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": themeColor,
            "summary": alert.message,
            "sections": [
                {
                    "activityTitle": (
                        f"**{alert.alert_level.value}** — {alert.alert_type.value}"
                    ),
                    "activitySubtitle": f"Collateral ID: {alert.collateral_id}",
                    "facts": [
                        {"name": "Asset Type", "value": alert.asset_type},
                        {"name": "Quality Score", "value": f"{alert.quality_score:.4f}"},
                        {
                            "name": "Cost Savings (bps)",
                            "value": f"{alert.cost_savings_bps:.2f}",
                        },
                        {"name": "Timestamp", "value": alert.timestamp},
                    ],
                    "text": alert.message,
                }
            ],
        }

        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(
                "Teams alert posted for collateral %s (HTTP %d)",
                alert.collateral_id,
                resp.status_code,
            )
        except requests.RequestException as exc:
            logger.error("Failed to post Teams alert: %s", exc)

    # ------------------------------------------------------------------
    # Structured log
    # ------------------------------------------------------------------

    def dispatch_log(self, alert: CollateralAlert) -> None:
        """Append the alert as a JSONL entry to the configured log file.

        Parameters
        ----------
        alert : CollateralAlert
        """
        entry = alert.to_dict()
        entry["entry_type"] = "alert"

        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            logger.debug(
                "Alert logged for collateral %s [%s]",
                alert.collateral_id,
                alert.alert_level.value,
            )
        except OSError as exc:
            logger.error("Failed to write alert to log: %s", exc)
