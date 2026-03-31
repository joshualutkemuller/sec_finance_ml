"""
alerts.py — Alert dataclass, multi-channel dispatcher, and factory for the
short squeeze probability scorer.

Classes:
    SqueezeAlert           — Dataclass carrying all fields for one alert event.
    SqueezeAlertDispatcher — Sends alerts via email, Teams webhook, and log file.
    SqueezeAlertFactory    — Builds SqueezeAlert lists from scored DataFrames.

Alerts are only dispatched when a security's risk tier is HIGH or CRITICAL.
Each dispatch channel (email, teams, log) can be independently toggled via
the ``alerts`` block of ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

from .model import RiskTier

logger = logging.getLogger(__name__)

# Risk tiers that trigger an alert dispatch
_ALERT_TIERS: frozenset[RiskTier] = frozenset({RiskTier.HIGH, RiskTier.CRITICAL})


# ---------------------------------------------------------------------------
# SqueezeAlert dataclass
# ---------------------------------------------------------------------------


@dataclass
class SqueezeAlert:
    """Represents a single short squeeze alert event for one security.

    Attributes:
        ticker: Security ticker symbol (e.g. "GME").
        short_interest_ratio: Shares short as a fraction of float (e.g. 0.42).
        days_to_cover: Estimated days to cover the short position.
        utilization_rate: Fraction of available borrow that is currently lent.
        squeeze_probability: XGBoost-predicted squeeze probability (0–1).
        risk_tier: Classified risk tier (LOW / ELEVATED / HIGH / CRITICAL).
        timestamp: UTC datetime of the scoring run that generated the alert.
        message: Human-readable summary string.
    """

    ticker: str
    short_interest_ratio: float
    days_to_cover: float
    utilization_rate: float
    squeeze_probability: float
    risk_tier: RiskTier
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    message: str = field(init=False)

    def __post_init__(self) -> None:
        self.message = (
            f"[{self.risk_tier.value}] {self.ticker} | "
            f"Squeeze Prob: {self.squeeze_probability:.2%} | "
            f"SI Ratio: {self.short_interest_ratio:.2%} | "
            f"Days-to-Cover: {self.days_to_cover:.1f} | "
            f"Utilization: {self.utilization_rate:.2%}"
        )

    def to_dict(self) -> dict:
        """Serialise the alert to a plain dict (JSON-friendly).

        Returns:
            dict: All fields with ``risk_tier`` as its string value and
                ``timestamp`` as an ISO-8601 string.
        """
        d = asdict(self)
        d["risk_tier"] = self.risk_tier.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# SqueezeAlertDispatcher
# ---------------------------------------------------------------------------


class SqueezeAlertDispatcher:
    """Dispatch :class:`SqueezeAlert` objects to email, Teams, and a log file.

    Each channel is enabled/disabled independently via the ``alerts`` config
    block.  Credentials are read from environment variables (loaded by
    ``python-dotenv`` at application startup).

    Args:
        config: The ``alerts`` block from ``config.yaml``.

    Example::

        dispatcher = SqueezeAlertDispatcher(config["alerts"])
        for alert in alerts:
            dispatcher.dispatch(alert)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._enabled: bool = bool(config.get("enabled", True))
        self._email_cfg: dict = config.get("email", {})
        self._teams_url: str = os.path.expandvars(
            config.get("teams_webhook_url", "")
        )
        self._log_path: Path = Path(config.get("log_path", "logs/alerts.jsonl"))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public dispatch entry point
    # ------------------------------------------------------------------

    def dispatch(self, alert: SqueezeAlert) -> None:
        """Send an alert to all configured channels.

        Only dispatches if the master ``alerts.enabled`` flag is True.

        Args:
            alert: The :class:`SqueezeAlert` to dispatch.
        """
        if not self._enabled:
            logger.debug("Alerts disabled. Skipping dispatch for %s.", alert.ticker)
            return

        self._send_log(alert)

        if alert.risk_tier in _ALERT_TIERS:
            if self._email_cfg.get("enabled", False):
                self._send_email(alert)
            if self._teams_url:
                self._send_teams(alert)

    def dispatch_many(self, alerts: list[SqueezeAlert]) -> None:
        """Dispatch a list of alerts, logging a summary at completion.

        Args:
            alerts: List of :class:`SqueezeAlert` objects to dispatch.
        """
        for alert in alerts:
            self.dispatch(alert)
        logger.info(
            "Dispatched %d alert(s) (%d HIGH, %d CRITICAL).",
            len(alerts),
            sum(1 for a in alerts if a.risk_tier == RiskTier.HIGH),
            sum(1 for a in alerts if a.risk_tier == RiskTier.CRITICAL),
        )

    # ------------------------------------------------------------------
    # Log channel
    # ------------------------------------------------------------------

    def _send_log(self, alert: SqueezeAlert) -> None:
        """Append alert as a JSON line to the configured log file.

        Args:
            alert: Alert to write.
        """
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict()) + "\n")
            logger.debug("Alert logged to %s for %s.", self._log_path, alert.ticker)
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)

    # ------------------------------------------------------------------
    # Email channel
    # ------------------------------------------------------------------

    def _send_email(self, alert: SqueezeAlert) -> None:
        """Send an HTML email alert via SMTP.

        Reads SMTP credentials from environment variables expanded at
        config load time.  Silently logs errors rather than raising so that
        a misconfigured email channel does not break the pipeline.

        Args:
            alert: Alert to send.
        """
        smtp_host: str = os.path.expandvars(
            self._email_cfg.get("smtp_host", "")
        )
        smtp_port: int = int(self._email_cfg.get("smtp_port", 587))
        smtp_user: str = os.path.expandvars(
            self._email_cfg.get("smtp_user", "")
        )
        smtp_password: str = os.path.expandvars(
            self._email_cfg.get("smtp_password", "")
        )
        from_addr: str = os.path.expandvars(
            self._email_cfg.get("from_address", smtp_user)
        )
        to_addrs: list[str] = [
            os.path.expandvars(a)
            for a in self._email_cfg.get("to_addresses", [])
        ]
        subject_prefix: str = self._email_cfg.get(
            "subject_prefix", "[SHORT SQUEEZE ALERT]"
        )

        if not (smtp_host and smtp_user and to_addrs):
            logger.warning(
                "Email channel misconfigured for %s. Skipping.", alert.ticker
            )
            return

        subject = (
            f"{subject_prefix} {alert.ticker} — "
            f"{alert.risk_tier.value} ({alert.squeeze_probability:.0%})"
        )
        html_body = _build_email_html(alert)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.attach(MIMEText(alert.message, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            logger.info(
                "Email alert sent for %s (%s) to %s.",
                alert.ticker,
                alert.risk_tier.value,
                to_addrs,
            )
        except smtplib.SMTPException as exc:
            logger.error("SMTP error sending alert for %s: %s", alert.ticker, exc)
        except OSError as exc:
            logger.error("Network error sending email for %s: %s", alert.ticker, exc)

    # ------------------------------------------------------------------
    # Teams channel
    # ------------------------------------------------------------------

    def _send_teams(self, alert: SqueezeAlert) -> None:
        """POST an adaptive card to a Microsoft Teams incoming webhook.

        Args:
            alert: Alert to dispatch.
        """
        if not self._teams_url or self._teams_url.startswith("${"):
            logger.warning(
                "Teams webhook URL not configured for %s. Skipping.", alert.ticker
            )
            return

        tier_color = {
            RiskTier.HIGH: "warning",
            RiskTier.CRITICAL: "attention",
        }.get(alert.risk_tier, "default")

        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": f"Short Squeeze Alert — {alert.ticker}",
                                "weight": "bolder",
                                "size": "large",
                                "color": tier_color,
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Risk Tier", "value": alert.risk_tier.value},
                                    {
                                        "title": "Squeeze Probability",
                                        "value": f"{alert.squeeze_probability:.1%}",
                                    },
                                    {
                                        "title": "Short Interest Ratio",
                                        "value": f"{alert.short_interest_ratio:.1%}",
                                    },
                                    {
                                        "title": "Days-to-Cover",
                                        "value": f"{alert.days_to_cover:.1f}",
                                    },
                                    {
                                        "title": "Utilization Rate",
                                        "value": f"{alert.utilization_rate:.1%}",
                                    },
                                    {
                                        "title": "Timestamp (UTC)",
                                        "value": alert.timestamp.strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        ),
                                    },
                                ],
                            },
                        ],
                    },
                }
            ],
        }

        try:
            resp = requests.post(
                self._teams_url,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(
                "Teams alert sent for %s (%s). HTTP %d.",
                alert.ticker,
                alert.risk_tier.value,
                resp.status_code,
            )
        except requests.RequestException as exc:
            logger.error(
                "Failed to send Teams alert for %s: %s", alert.ticker, exc
            )


# ---------------------------------------------------------------------------
# SqueezeAlertFactory
# ---------------------------------------------------------------------------


class SqueezeAlertFactory:
    """Build :class:`SqueezeAlert` objects from scored DataFrames.

    This is a stateless factory; all methods are classmethods.
    """

    @classmethod
    def from_scores(
        cls,
        df: "import pandas; pandas.DataFrame",
        risk_tiers: list[RiskTier],
        config: Optional[dict] = None,
    ) -> list[SqueezeAlert]:
        """Create :class:`SqueezeAlert` objects for HIGH and CRITICAL tickers.

        Iterates over the scored DataFrame and its corresponding risk tier
        list, constructing one ``SqueezeAlert`` for every row whose tier is
        HIGH or CRITICAL.

        Args:
            df: Scored DataFrame.  Must contain columns:
                ``ticker``, ``short_interest_ratio``, ``days_to_cover``,
                ``utilization_rate``, ``squeeze_probability``.
            risk_tiers: List of :class:`RiskTier` labels aligned with the
                rows of ``df`` (same length and order).
            config: Optional config dict (unused currently, reserved for
                future per-tier threshold overrides).

        Returns:
            list[SqueezeAlert]: One alert per HIGH/CRITICAL row, sorted
                descending by ``squeeze_probability``.

        Raises:
            ValueError: If ``df`` length and ``risk_tiers`` length differ.
        """
        import pandas as pd  # local import to avoid circular deps at module level

        if len(df) != len(risk_tiers):
            raise ValueError(
                f"df length ({len(df)}) != risk_tiers length ({len(risk_tiers)})"
            )

        now = datetime.now(tz=timezone.utc)
        alerts: list[SqueezeAlert] = []

        for (_, row), tier in zip(df.iterrows(), risk_tiers):
            if tier not in _ALERT_TIERS:
                continue
            alert = SqueezeAlert(
                ticker=str(row["ticker"]),
                short_interest_ratio=float(row.get("short_interest_ratio", 0.0)),
                days_to_cover=float(row.get("days_to_cover", 0.0)),
                utilization_rate=float(row.get("utilization_rate", 0.0)),
                squeeze_probability=float(row.get("squeeze_probability", 0.0)),
                risk_tier=tier,
                timestamp=now,
            )
            alerts.append(alert)

        # Sort most dangerous first
        alerts.sort(key=lambda a: a.squeeze_probability, reverse=True)
        logger.info(
            "Alert factory produced %d alert(s) from %d scored rows.",
            len(alerts),
            len(df),
        )
        return alerts


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_email_html(alert: SqueezeAlert) -> str:
    """Build a minimal HTML email body for a squeeze alert.

    Args:
        alert: The alert to render.

    Returns:
        str: HTML string suitable for use in a MIME multipart message.
    """
    tier_color = {
        RiskTier.LOW: "#28a745",
        RiskTier.ELEVATED: "#ffc107",
        RiskTier.HIGH: "#fd7e14",
        RiskTier.CRITICAL: "#dc3545",
    }.get(alert.risk_tier, "#6c757d")

    return f"""
    <html><body style="font-family:Arial,sans-serif;font-size:14px;">
      <h2 style="color:{tier_color};">
        Short Squeeze Alert &mdash; {alert.ticker}
        <span style="font-size:14px;">({alert.risk_tier.value})</span>
      </h2>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        <tr><th style="text-align:left;">Metric</th><th>Value</th></tr>
        <tr><td>Ticker</td><td><strong>{alert.ticker}</strong></td></tr>
        <tr><td>Squeeze Probability</td>
            <td style="color:{tier_color};"><strong>{alert.squeeze_probability:.1%}</strong></td></tr>
        <tr><td>Risk Tier</td>
            <td style="color:{tier_color};"><strong>{alert.risk_tier.value}</strong></td></tr>
        <tr><td>Short Interest Ratio</td><td>{alert.short_interest_ratio:.2%}</td></tr>
        <tr><td>Days-to-Cover</td><td>{alert.days_to_cover:.1f}</td></tr>
        <tr><td>Utilization Rate</td><td>{alert.utilization_rate:.2%}</td></tr>
        <tr><td>Timestamp (UTC)</td>
            <td>{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
      </table>
      <p style="color:#6c757d;font-size:12px;">
        Generated by Short Squeeze Probability Scorer &mdash;
        Financing Solutions Intelligence Platform
      </p>
    </body></html>
    """
