"""
alerts.py
---------
Alert dataclasses, severity levels, multi-channel dispatcher, and factory
for the Securities Lending Rate Anomaly Detector.

Classes
-------
AlertLevel
    Enum defining INFO, WARNING, and CRITICAL severity levels.

Alert
    Dataclass holding all fields for a single alert event.

AlertFactory
    Creates :class:`Alert` objects from a scored results DataFrame.

AlertDispatcher
    Sends alerts via email (SMTP), Microsoft Teams webhook, and/or log file.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Alert Level Enum
# ============================================================================


class AlertLevel(Enum):
    """Severity level for a lending-rate anomaly alert.

    Levels
    ------
    INFO
        Combined anomaly score in ``[threshold, threshold + 0.1)``.
    WARNING
        Combined anomaly score in ``[threshold + 0.1, threshold + 0.2)``.
    CRITICAL
        Combined anomaly score ``>= threshold + 0.2``.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# Alert Dataclass
# ============================================================================


@dataclass
class Alert:
    """A single anomaly alert for one security at one point in time.

    Attributes
    ----------
    ticker : str
        Security identifier (CUSIP / ISIN / ticker symbol).
    rate : float
        Observed borrow rate at the time of the alert (as a decimal, e.g.
        0.035 for 3.5%).
    anomaly_score : float
        Combined anomaly score [0, 1].
    alert_level : AlertLevel
        Severity classification derived from the score magnitude.
    timestamp : datetime
        UTC timestamp of the observation.
    message : str
        Human-readable description of the alert.
    if_score : float
        Isolation Forest component score [0, 1].
    lstm_score : float
        LSTM reconstruction-error component score [0, 1].
    """

    ticker: str
    rate: float
    anomaly_score: float
    alert_level: AlertLevel
    timestamp: datetime
    message: str
    if_score: float = 0.0
    lstm_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the alert to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with all alert fields, suitable for JSON serialisation
            or PowerBI row format.
        """
        return {
            "ticker": self.ticker,
            "rate": self.rate,
            "anomaly_score": round(self.anomaly_score, 6),
            "if_score": round(self.if_score, 6),
            "lstm_score": round(self.lstm_score, 6),
            "alert_level": self.alert_level.value,
            "alert_flag": True,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }

    def __str__(self) -> str:
        return (
            f"[{self.alert_level.value}] {self.ticker} | "
            f"rate={self.rate:.4f} | score={self.anomaly_score:.4f} | "
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )


# ============================================================================
# Alert Factory
# ============================================================================


class AlertFactory:
    """Creates :class:`Alert` objects from a scored results DataFrame.

    The factory expects a DataFrame that contains at minimum:
    ``ticker``, ``rate``, ``combined_score``, ``is_anomaly``, and
    ``timestamp`` columns.  It classifies each anomalous row into an
    :class:`AlertLevel` based on the combined score relative to the
    configured threshold.
    """

    @staticmethod
    def _classify_level(score: float, threshold: float) -> AlertLevel:
        """Map a combined anomaly score to an :class:`AlertLevel`.

        Parameters
        ----------
        score : float
            Combined anomaly score [0, 1].
        threshold : float
            Base threshold from ``alerts.anomaly_score_threshold``.

        Returns
        -------
        AlertLevel
        """
        if score >= threshold + 0.20:
            return AlertLevel.CRITICAL
        elif score >= threshold + 0.10:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def from_scores(
        self,
        df: "import pandas as pd; pd.DataFrame",  # type: ignore[name-defined]
        config: dict[str, Any],
    ) -> list[Alert]:
        """Build a list of :class:`Alert` objects from scored DataFrame rows.

        Only rows where ``is_anomaly`` is ``True`` are converted to alerts.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame produced by :meth:`~model.CombinedAnomalyDetector.predict`,
            merged with the original data columns (``ticker``, ``rate``,
            ``date``/``timestamp``).  Expected columns:

            - ``ticker``          — security identifier
            - ``rate`` (or configured ``rate_col``)  — borrow rate
            - ``combined_score``  — blended anomaly score
            - ``is_anomaly``      — boolean flag
            - ``if_score``        — IF component score (optional)
            - ``lstm_score``      — LSTM component score (optional)
            - ``timestamp`` or ``date`` — observation time

        config : dict[str, Any]
            Full application configuration dictionary.

        Returns
        -------
        list[Alert]
            One :class:`Alert` per anomalous row.  Returns an empty list if
            no anomalies are present.
        """
        import pandas as pd  # local import

        alerts_cfg = config.get("alerts", {})
        data_cfg = config.get("data", {})

        threshold: float = float(alerts_cfg.get("anomaly_score_threshold", 0.70))
        ticker_col: str = data_cfg.get("ticker_col", "ticker")
        rate_col: str = data_cfg.get("rate_col", "borrow_rate")

        # Determine the timestamp column
        date_col: str = data_cfg.get("date_col", "date")
        ts_col = "timestamp" if "timestamp" in df.columns else date_col

        anomaly_df = df[df["is_anomaly"] == True].copy()

        if anomaly_df.empty:
            logger.info("AlertFactory: no anomalies in scored DataFrame — no alerts created")
            return []

        alerts: list[Alert] = []
        for _, row in anomaly_df.iterrows():
            score = float(row.get("combined_score", 0.0))
            level = AlertFactory._classify_level(score, threshold)

            ticker = str(row.get(ticker_col, row.get("ticker", "UNKNOWN")))
            rate = float(row.get(rate_col, row.get("rate", 0.0)))
            if_score = float(row.get("if_score", 0.0))
            lstm_score = float(row.get("lstm_score", 0.0))

            # Parse timestamp
            raw_ts = row.get(ts_col, row.get("date", None))
            if isinstance(raw_ts, datetime):
                ts = raw_ts.replace(tzinfo=timezone.utc) if raw_ts.tzinfo is None else raw_ts
            elif raw_ts is not None:
                try:
                    ts = pd.Timestamp(raw_ts).to_pydatetime().replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.now(tz=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)

            message = (
                f"Anomalous borrow rate detected for {ticker}: "
                f"rate={rate:.4f}, combined_score={score:.4f} "
                f"(IF={if_score:.4f}, LSTM={lstm_score:.4f}), "
                f"level={level.value}"
            )

            alert = Alert(
                ticker=ticker,
                rate=rate,
                anomaly_score=score,
                alert_level=level,
                timestamp=ts,
                message=message,
                if_score=if_score,
                lstm_score=lstm_score,
            )
            alerts.append(alert)

        logger.info("AlertFactory created %d alerts from %d anomalous rows", len(alerts), len(anomaly_df))
        return alerts


# ============================================================================
# Alert Dispatcher
# ============================================================================


class AlertDispatcher:
    """Multi-channel alert dispatcher.

    Dispatches :class:`Alert` objects to one or more configured channels:

    - **Email**: plain-text + HTML email via SMTP (STARTTLS on port 587).
    - **MS Teams**: JSON Adaptive Card posted to an incoming webhook URL.
    - **Log file**: JSONL structured log appended to the configured path.

    All channels are attempted independently; a failure in one channel does
    not prevent dispatch to other channels.

    Parameters
    ----------
    config : dict[str, Any]
        Full application configuration.  Uses the ``alerts`` section.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        alerts_cfg = config.get("alerts", {})

        self.enabled: bool = bool(alerts_cfg.get("enabled", True))
        self.threshold: float = float(alerts_cfg.get("anomaly_score_threshold", 0.70))

        # Email config
        email_cfg = alerts_cfg.get("email", {})
        self.smtp_host: str = email_cfg.get("smtp_host", "")
        self.smtp_port: int = int(email_cfg.get("port", 587))
        self.email_from: str = email_cfg.get("from", "")
        self.email_to: list[str] = email_cfg.get("to", [])
        self.smtp_password: str = os.getenv("SMTP_PASSWORD", "")
        self.smtp_username: str = os.getenv("SMTP_USERNAME", self.email_from)

        # Teams config
        self.teams_webhook_url: str = alerts_cfg.get("teams_webhook_url", "")

        # Log file
        self.log_path: str = alerts_cfg.get("log_path", "logs/alerts.jsonl")

        logger.debug(
            "AlertDispatcher initialised — enabled=%s, smtp_host=%s, teams=%s",
            self.enabled,
            self.smtp_host or "(none)",
            "configured" if self.teams_webhook_url else "(none)",
        )

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    def _send_email(self, alert: Alert) -> bool:
        """Send an HTML alert email via SMTP (STARTTLS).

        Parameters
        ----------
        alert : Alert
            The alert to dispatch.

        Returns
        -------
        bool
            ``True`` if the email was sent successfully, ``False`` otherwise.
        """
        if not self.smtp_host or not self.email_to:
            logger.debug("Email alert skipped — smtp_host or email_to not configured")
            return False

        subject = (
            f"[{alert.alert_level.value}] Securities Lending Anomaly: {alert.ticker}"
        )

        html_body = f"""
        <html><body>
        <h2 style="color: {'red' if alert.alert_level == AlertLevel.CRITICAL else 'orange'};">
            {alert.alert_level.value} — Borrow Rate Anomaly Detected
        </h2>
        <table border="1" cellpadding="6" cellspacing="0">
            <tr><th>Ticker</th><td><b>{alert.ticker}</b></td></tr>
            <tr><th>Borrow Rate</th><td>{alert.rate:.4f}</td></tr>
            <tr><th>Combined Score</th><td>{alert.anomaly_score:.4f}</td></tr>
            <tr><th>Isolation Forest Score</th><td>{alert.if_score:.4f}</td></tr>
            <tr><th>LSTM Score</th><td>{alert.lstm_score:.4f}</td></tr>
            <tr><th>Alert Level</th><td>{alert.alert_level.value}</td></tr>
            <tr><th>Timestamp (UTC)</th><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>
        <p><i>{alert.message}</i></p>
        <hr><p style="font-size:11px;">
            Securities Lending Rate Anomaly Detector — Financing Solutions Intelligence Platform
        </p>
        </body></html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = ", ".join(self.email_to)
        msg.attach(MIMEText(alert.message, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            logger.info(
                "Email alert sent for %s to %d recipients", alert.ticker, len(self.email_to)
            )
            return True

        except smtplib.SMTPException as exc:
            logger.error("Failed to send email alert for %s: %s", alert.ticker, exc)
            return False

    # ------------------------------------------------------------------
    # Microsoft Teams webhook
    # ------------------------------------------------------------------

    def _send_teams(self, alert: Alert) -> bool:
        """Post an Adaptive Card to a Microsoft Teams incoming webhook.

        Parameters
        ----------
        alert : Alert
            The alert to dispatch.

        Returns
        -------
        bool
            ``True`` if the webhook POST returned HTTP 200, ``False`` otherwise.
        """
        if not self.teams_webhook_url:
            logger.debug("Teams alert skipped — webhook URL not configured")
            return False

        colour_map = {
            AlertLevel.INFO: "0076D7",
            AlertLevel.WARNING: "FFA500",
            AlertLevel.CRITICAL: "CC0000",
        }
        theme_colour = colour_map.get(alert.alert_level, "0076D7")

        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": theme_colour,
            "summary": f"[{alert.alert_level.value}] Lending Rate Anomaly: {alert.ticker}",
            "sections": [
                {
                    "activityTitle": (
                        f"**[{alert.alert_level.value}] Anomalous Borrow Rate — "
                        f"{alert.ticker}**"
                    ),
                    "activitySubtitle": (
                        f"{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    ),
                    "facts": [
                        {"name": "Ticker", "value": alert.ticker},
                        {"name": "Borrow Rate", "value": f"{alert.rate:.4f}"},
                        {"name": "Combined Score", "value": f"{alert.anomaly_score:.4f}"},
                        {"name": "IF Score", "value": f"{alert.if_score:.4f}"},
                        {"name": "LSTM Score", "value": f"{alert.lstm_score:.4f}"},
                        {"name": "Alert Level", "value": alert.alert_level.value},
                    ],
                    "markdown": True,
                }
            ],
        }

        try:
            response = requests.post(
                self.teams_webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("Teams alert sent for %s", alert.ticker)
                return True
            else:
                logger.warning(
                    "Teams webhook returned HTTP %d for %s",
                    response.status_code,
                    alert.ticker,
                )
                return False

        except requests.RequestException as exc:
            logger.error("Failed to send Teams alert for %s: %s", alert.ticker, exc)
            return False

    # ------------------------------------------------------------------
    # Log file
    # ------------------------------------------------------------------

    def _log_alert(self, alert: Alert) -> None:
        """Append a structured JSON log entry for the alert.

        Creates parent directories for ``log_path`` if they do not exist.
        Each alert is written as a single JSON object on its own line
        (JSONL format).

        Parameters
        ----------
        alert : Alert
            The alert to log.
        """
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        entry = {
            "event": "ANOMALY_ALERT",
            "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **alert.to_dict(),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            logger.debug("Alert logged to %s for %s", self.log_path, alert.ticker)

        except OSError as exc:
            logger.error("Failed to write alert log for %s: %s", alert.ticker, exc)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def dispatch(self, alerts: list[Alert]) -> None:
        """Dispatch a list of alerts to all configured channels.

        For each alert, attempts are made in order:
        1. Log file (always attempted if log_path is set)
        2. Email (if smtp_host and email_to are configured)
        3. Teams (if teams_webhook_url is configured)

        A failure in one channel does not prevent attempts on other channels.

        Parameters
        ----------
        alerts : list[Alert]
            Alerts to dispatch.  No-op if ``alerts`` is empty or
            ``alerts.enabled`` is ``False``.
        """
        if not self.enabled:
            logger.info("Alert dispatching is disabled (alerts.enabled=false)")
            return

        if not alerts:
            logger.info("No alerts to dispatch")
            return

        logger.info("Dispatching %d alert(s)", len(alerts))

        for alert in alerts:
            logger.info("Processing alert: %s", alert)

            # 1. Always log
            self._log_alert(alert)

            # 2. Email
            self._send_email(alert)

            # 3. Teams
            self._send_teams(alert)

        logger.info("Alert dispatch complete for %d alert(s)", len(alerts))
