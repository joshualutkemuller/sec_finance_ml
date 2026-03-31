"""
alerts.py — Portfolio Concentration Risk Early Warning Monitor

Handles the construction and dispatch of risk alerts when portfolio
concentration scores cross HIGH or CRITICAL thresholds.

Classes
-------
RiskAlert           — Dataclass representing a single risk alert event.
RiskAlertDispatcher — Sends alerts via email, Teams webhook, and log file.
RiskAlertFactory    — Builds :class:`RiskAlert` objects from scored DataFrames.
"""

from __future__ import annotations

import json
import logging
import smtplib
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

from src.model import RiskLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RiskAlert:
    """A single risk alert event for one portfolio/date combination.

    Attributes
    ----------
    portfolio_id : str
        Identifier of the portfolio that triggered the alert.
    sector : str
        Most concentrated GICS sector (or "ALL" for portfolio-level).
    counterparty : str
        Most concentrated counterparty name (or "ALL").
    concentration_score : float
        Raw Random Forest probability score in [0, 1].
    risk_level : RiskLevel
        Classified risk tier (LOW / MEDIUM / HIGH / CRITICAL).
    top_drivers : list[str]
        Ordered list of top SHAP feature names driving the score.
    timestamp : datetime
        UTC timestamp at which the alert was generated.
    message : str
        Human-readable alert body.
    alert_flag : bool
        True when this alert was (or should be) dispatched externally.
    """

    portfolio_id: str
    sector: str
    counterparty: str
    concentration_score: float
    risk_level: RiskLevel
    top_drivers: list[str]
    timestamp: datetime
    message: str
    alert_flag: bool = False

    def to_dict(self) -> dict:
        """Serialise the alert to a JSON-compatible dictionary.

        Returns
        -------
        dict
        """
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_powerbi_row(self) -> dict:
        """Return a dict matching the PowerBI push dataset schema.

        Returns
        -------
        dict
            Keys: portfolio_id, sector, counterparty, concentration_score,
            risk_level, alert_flag, timestamp.
        """
        return {
            "portfolio_id": self.portfolio_id,
            "sector": self.sector,
            "counterparty": self.counterparty,
            "concentration_score": round(self.concentration_score, 6),
            "risk_level": self.risk_level.value,
            "alert_flag": self.alert_flag,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class RiskAlertFactory:
    """Build :class:`RiskAlert` instances from scored pipeline outputs."""

    @staticmethod
    def from_scores(
        df: "pd.DataFrame",  # noqa: F821 — avoid circular import at module level
        risk_levels: list[RiskLevel],
        shap_df: "pd.DataFrame",  # noqa: F821
        config: dict,
    ) -> list[RiskAlert]:
        """Construct a list of :class:`RiskAlert` objects from scored results.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio-level feature / results DataFrame.  Must contain columns:
            ``portfolio_id``, and optionally ``sector``, ``counterparty``,
            ``date``.
        risk_levels : list[RiskLevel]
            Classified risk levels from
            :meth:`~src.model.RandomForestRiskScorer.classify`, same length as
            ``df``.
        shap_df : pd.DataFrame
            SHAP explanation DataFrame from
            :meth:`~src.model.RandomForestRiskScorer.explain`.  Columns are
            feature names; rows align with ``df``.  May be empty.
        config : dict
            Full pipeline config (used for threshold labels in alert text).

        Returns
        -------
        list[RiskAlert]
            One :class:`RiskAlert` per row in ``df``, with ``alert_flag``
            set to True for HIGH and CRITICAL rows.
        """
        import pandas as pd  # local import to avoid top-level circular dep

        alerts: list[RiskAlert] = []
        now = datetime.now(tz=timezone.utc)

        scores_col = "concentration_score" if "concentration_score" in df.columns else None

        for idx, row in df.iterrows():
            risk_level = risk_levels[idx] if isinstance(risk_levels, list) else risk_levels.iloc[idx]
            score: float = float(row[scores_col]) if scores_col else 0.0
            portfolio_id: str = str(row.get("portfolio_id", "UNKNOWN"))
            sector: str = str(row.get("dominant_sector", row.get("sector", "ALL")))
            counterparty: str = str(
                row.get("dominant_counterparty", row.get("counterparty", "ALL"))
            )

            # Extract top SHAP drivers for this row
            top_drivers: list[str] = []
            if not shap_df.empty and idx in shap_df.index:
                shap_row = shap_df.loc[idx].abs().sort_values(ascending=False)
                top_drivers = shap_row.index.tolist()

            # Build alert message
            message = RiskAlertFactory._build_message(
                portfolio_id=portfolio_id,
                sector=sector,
                counterparty=counterparty,
                score=score,
                risk_level=risk_level,
                top_drivers=top_drivers,
                timestamp=now,
            )

            alert = RiskAlert(
                portfolio_id=portfolio_id,
                sector=sector,
                counterparty=counterparty,
                concentration_score=score,
                risk_level=risk_level,
                top_drivers=top_drivers,
                timestamp=now,
                message=message,
                alert_flag=risk_level.is_actionable(),
            )
            alerts.append(alert)

        logger.debug(
            "RiskAlertFactory built %d alerts (%d actionable)",
            len(alerts),
            sum(1 for a in alerts if a.alert_flag),
        )
        return alerts

    @staticmethod
    def _build_message(
        portfolio_id: str,
        sector: str,
        counterparty: str,
        score: float,
        risk_level: RiskLevel,
        top_drivers: list[str],
        timestamp: datetime,
    ) -> str:
        """Format a human-readable alert message.

        Parameters
        ----------
        portfolio_id : str
        sector : str
        counterparty : str
        score : float
        risk_level : RiskLevel
        top_drivers : list[str]
        timestamp : datetime

        Returns
        -------
        str
        """
        drivers_text = (
            ", ".join(top_drivers[:5]) if top_drivers else "N/A"
        )
        return textwrap.dedent(
            f"""
            PORTFOLIO CONCENTRATION RISK ALERT
            ===================================
            Portfolio:           {portfolio_id}
            Risk Level:          {risk_level.value}
            Concentration Score: {score:.4f}
            Sector:              {sector}
            Counterparty:        {counterparty}
            Timestamp (UTC):     {timestamp.isoformat()}

            Top Risk Drivers:
              {drivers_text}

            Action required for HIGH/CRITICAL levels. Please review position
            concentrations and consider rebalancing or hedging.
            """
        ).strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class RiskAlertDispatcher:
    """Dispatch :class:`RiskAlert` objects via email, Teams, and log file.

    Parameters
    ----------
    config : dict
        The ``alerts`` section of config.yaml.
    dry_run : bool, optional
        When True, logs what would be sent but does not actually dispatch
        emails or Teams messages (default False).
    """

    def __init__(self, config: dict, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run
        self._master_enabled: bool = bool(config.get("enabled", True))
        self._email_config: dict = config.get("email", {})
        self._teams_webhook_url: str = config.get("teams_webhook_url", "")
        self._teams_enabled: bool = bool(config.get("teams_enabled", True))
        self._log_path: Path = Path(config.get("log_path", "logs/risk_alerts.jsonl"))

        # Ensure the log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, alerts: list[RiskAlert]) -> dict[str, int]:
        """Dispatch all alerts through all enabled channels.

        Only alerts with ``alert_flag=True`` (HIGH / CRITICAL) are sent via
        email and Teams.  All alerts are written to the structured log.

        Parameters
        ----------
        alerts : list[RiskAlert]
            Alerts to dispatch.

        Returns
        -------
        dict[str, int]
            Counts of dispatches: ``{"email": n, "teams": n, "log": n}``.
        """
        counts: dict[str, int] = {"email": 0, "teams": 0, "log": 0}

        if not alerts:
            logger.info("No alerts to dispatch.")
            return counts

        if not self._master_enabled:
            logger.info("Alert dispatch disabled in config — skipping all channels.")
            return counts

        actionable = [a for a in alerts if a.alert_flag]

        # Email
        if self._email_config.get("enabled", True) and actionable:
            sent = self._dispatch_email(actionable)
            counts["email"] = sent

        # Teams
        if self._teams_enabled and self._teams_webhook_url and actionable:
            sent = self._dispatch_teams(actionable)
            counts["teams"] = sent

        # Log — all alerts regardless of level
        logged = self._dispatch_log(alerts)
        counts["log"] = logged

        logger.info(
            "Alert dispatch complete: email=%d, teams=%d, log=%d",
            counts["email"],
            counts["teams"],
            counts["log"],
        )
        return counts

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    def _dispatch_email(self, alerts: list[RiskAlert]) -> int:
        """Send a single digest email containing all actionable alerts.

        Parameters
        ----------
        alerts : list[RiskAlert]
            HIGH / CRITICAL alerts to include.

        Returns
        -------
        int
            1 if the email was sent successfully, 0 otherwise.
        """
        cfg = self._email_config
        smtp_host: str = cfg.get("smtp_host", "localhost")
        smtp_port: int = int(cfg.get("smtp_port", 587))
        use_tls: bool = bool(cfg.get("use_tls", True))
        from_addr: str = cfg.get("from_address", "risk-monitor@example.com")
        to_addrs: list[str] = cfg.get("to_addresses", [])
        subject_prefix: str = cfg.get("subject_prefix", "[RISK ALERT]")

        if not to_addrs:
            logger.warning("Email dispatch skipped: no recipient addresses configured.")
            return 0

        subject = (
            f"{subject_prefix} {len(alerts)} Portfolio Concentration Alert(s) — "
            f"{alerts[0].timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        body = "\n\n" + "=" * 60 + "\n\n".join(a.message for a in alerts)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.attach(MIMEText(body, "plain"))

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would send email to %s: %s", to_addrs, subject
            )
            return 1

        # Read credentials from environment (set via .env / python-dotenv)
        import os

        username: Optional[str] = os.getenv("ALERT_EMAIL_USER")
        password: Optional[str] = os.getenv("ALERT_EMAIL_PASSWORD")

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                if use_tls:
                    server.starttls()
                if username and password:
                    server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            logger.info("Email alert sent to %d recipients", len(to_addrs))
            return 1
        except smtplib.SMTPException as exc:
            logger.error("Failed to send email alert: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def _dispatch_teams(self, alerts: list[RiskAlert]) -> int:
        """Post alert cards to Microsoft Teams via an incoming webhook.

        Builds an Adaptive Card payload (Teams-compatible JSON) containing
        a summary table of all actionable alerts.

        Parameters
        ----------
        alerts : list[RiskAlert]
            HIGH / CRITICAL alerts to post.

        Returns
        -------
        int
            Number of alerts successfully posted (0 or len(alerts)).
        """
        if not self._teams_webhook_url:
            logger.debug("Teams webhook URL not configured — skipping Teams dispatch.")
            return 0

        # Build a simple message card payload
        facts = [
            {
                "name": f"{a.portfolio_id} ({a.risk_level.value})",
                "value": f"Score: {a.concentration_score:.4f} | {a.sector} | {a.counterparty}",
            }
            for a in alerts
        ]

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000" if any(a.risk_level == RiskLevel.CRITICAL for a in alerts) else "FFA500",
            "summary": f"{len(alerts)} Portfolio Concentration Risk Alert(s)",
            "sections": [
                {
                    "activityTitle": f"**Portfolio Concentration Risk Alert** — {alerts[0].timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
                    "activitySubtitle": f"{len(alerts)} portfolio(s) require immediate review",
                    "facts": facts,
                    "markdown": True,
                }
            ],
        }

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would POST %d alert(s) to Teams webhook", len(alerts)
            )
            return len(alerts)

        try:
            response = requests.post(
                self._teams_webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            logger.info("Teams alert posted (%d items)", len(alerts))
            return len(alerts)
        except requests.RequestException as exc:
            logger.error("Failed to post Teams alert: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _dispatch_log(self, alerts: list[RiskAlert]) -> int:
        """Write all alerts to the structured JSONL log file.

        Each line in the log is a valid JSON object (one per alert).

        Parameters
        ----------
        alerts : list[RiskAlert]

        Returns
        -------
        int
            Number of alerts written.
        """
        written = 0
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                for alert in alerts:
                    fh.write(json.dumps(alert.to_dict()) + "\n")
                    written += 1
        except IOError as exc:
            logger.error("Failed to write to alert log %s: %s", self._log_path, exc)
        return written
