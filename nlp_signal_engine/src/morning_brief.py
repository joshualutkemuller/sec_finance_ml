"""
Morning Brief — LLM-synthesized daily securities finance intelligence report.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Uses the Claude API to synthesize structured signals from all NLP pipelines,
FRED macro data, and model outputs into a concise, desk-ready morning brief.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


MORNING_BRIEF_SYSTEM_PROMPT = """You are a senior securities finance analyst writing the daily morning brief
for a securities lending and agency/prime brokerage desk. You synthesize macro data,
NLP signals from SEC filings and news, and quantitative model outputs into a concise,
actionable 400-600 word report.

Your brief must include:
1. MACRO REGIME section — current rate regime and 1-sentence outlook
2. TOP RISK NAMES section — up to 5 securities with elevated signals, with specific data points
3. LENDING DESK OUTLOOK — directional demand forecast and collateral recommendation
4. COLLATERAL QUALITY — one sentence on current quality environment

Style: Direct, data-driven, institutional. No hedging language or disclaimers.
Format with clear section headers in ALL CAPS. Include specific numbers from the data provided.
Sign the brief: "Joshua Lutkemuller, Managing Director — Securities Finance & Agency Lending"
"""


class MorningBriefGenerator:
    """
    Generates the daily Securities Finance Morning Brief using the Claude API.

    Inputs:
      - NLP signals DataFrame (per-ticker sentiment, risk scores)
      - FRED macro signals (regime, SOFR forecast, spreads)
      - Model outputs (short squeeze scores, collateral quality, demand forecast)

    Output:
      - Structured text brief (~500 words)
      - Pushed to email, Teams, and saved to file
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("morning_brief", {})
        self.model = self.cfg.get("model", "claude-opus-4-6")
        self.max_tokens = self.cfg.get("max_tokens", 800)
        self._client = None

    def generate(
        self,
        nlp_signals: pd.DataFrame,
        macro_signals: dict,
        model_outputs: dict,
        date: datetime | None = None,
    ) -> str:
        """Generate the morning brief.

        Args:
            nlp_signals: DataFrame with ticker-level NLP signals.
            macro_signals: Dict of macro indicators (regime, SOFR, spreads, etc.).
            model_outputs: Dict of model scores (squeeze scores, demand forecast, etc.).
            date: Report date (defaults to today).

        Returns:
            Formatted morning brief string.
        """
        date = date or datetime.utcnow()
        context = self._build_context(nlp_signals, macro_signals, model_outputs, date)

        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=MORNING_BRIEF_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            brief = response.content[0].text
            logger.info("Morning Brief generated (%d chars)", len(brief))
        except Exception as exc:
            logger.error("Claude API call failed: %s", exc)
            brief = self._fallback_brief(date, macro_signals, nlp_signals)

        self._save(brief, date)
        self._distribute(brief, date)
        return brief

    # ── Private ────────────────────────────────────────────────────────────

    def _build_context(
        self,
        nlp_signals: pd.DataFrame,
        macro_signals: dict,
        model_outputs: dict,
        date: datetime,
    ) -> str:
        date_str = date.strftime("%Y-%m-%d")
        lines = [f"DATE: {date_str}", "", "=== MACRO SIGNALS ==="]

        for k, v in macro_signals.items():
            lines.append(f"  {k}: {v}")

        lines += ["", "=== TOP NLP RISK SIGNALS (top 10 tickers by risk score) ==="]
        if not nlp_signals.empty and "ticker" in nlp_signals.columns:
            top = (
                nlp_signals.sort_values("risk_keyword_score", ascending=False)
                .head(10)
            )
            for _, row in top.iterrows():
                lines.append(
                    f"  {row.get('ticker', 'N/A')}: "
                    f"sentiment={row.get('sentiment_score', 0):.3f}, "
                    f"risk_score={row.get('risk_keyword_score', 0):.3f}, "
                    f"uncertainty={row.get('uncertainty_score', 0):.3f}"
                )

        lines += ["", "=== MODEL OUTPUTS ==="]
        for k, v in model_outputs.items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _fallback_brief(
        self, date: datetime, macro_signals: dict, nlp_signals: pd.DataFrame
    ) -> str:
        """Simple template fallback if Claude API is unavailable."""
        date_str = date.strftime("%Y-%m-%d")
        regime = macro_signals.get("current_regime", "Unknown")
        sofr = macro_signals.get("sofr", "N/A")
        return (
            f"SECURITIES FINANCE MORNING BRIEF — {date_str}\n"
            f"Author: Joshua Lutkemuller, MD\n\n"
            f"MACRO REGIME: {regime}\n"
            f"SOFR: {sofr}\n\n"
            f"[LLM synthesis unavailable — check ANTHROPIC_API_KEY]\n\n"
            f"NLP signals processed: {len(nlp_signals)} records.\n"
        )

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    raise EnvironmentError("ANTHROPIC_API_KEY not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError as exc:
                raise ImportError("Install anthropic: pip install anthropic") from exc
        return self._client

    def _save(self, brief: str, date: datetime) -> None:
        output_path = Path(self.cfg.get("output_path", "outputs/morning_brief.txt"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dated_path = output_path.parent / f"morning_brief_{date.strftime('%Y%m%d')}.txt"
        dated_path.write_text(brief)
        output_path.write_text(brief)  # also overwrite the "latest" symlink
        logger.info("Morning Brief saved to %s", dated_path)

    def _distribute(self, brief: str, date: datetime) -> None:
        """Send brief via email and Teams."""
        import smtplib
        from email.mime.text import MIMEText
        import requests

        date_str = date.strftime("%Y-%m-%d")

        # Email
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASSWORD", "")
        recipients_raw = os.environ.get("ALERT_EMAIL_RECIPIENTS", self.cfg.get("email_recipients", ""))
        recipients = [r.strip() for r in str(recipients_raw).split(",") if r.strip()]

        if smtp_host and recipients:
            msg = MIMEText(brief, "plain")
            msg["Subject"] = f"Securities Finance Morning Brief — {date_str}"
            msg["From"] = os.environ.get("ALERT_FROM", smtp_user)
            msg["To"] = ", ".join(recipients)
            try:
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    if smtp_pass:
                        server.login(smtp_user, smtp_pass)
                    server.sendmail(msg["From"], recipients, msg.as_string())
                logger.info("Morning Brief emailed to %d recipients", len(recipients))
            except Exception as exc:
                logger.warning("Morning Brief email failed: %s", exc)

        # Teams
        webhook = os.environ.get("TEAMS_WEBHOOK_URL", self.cfg.get("teams_webhook_url", ""))
        if webhook and not webhook.startswith("${"):
            payload = {
                "@type": "MessageCard",
                "themeColor": "0078D4",
                "summary": f"Morning Brief {date_str}",
                "sections": [{"text": f"<pre>{brief[:1200]}...</pre>"}],
            }
            try:
                requests.post(webhook, json=payload, timeout=10)
            except Exception as exc:
                logger.warning("Teams delivery failed: %s", exc)
