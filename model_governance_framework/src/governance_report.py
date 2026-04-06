"""
Governance Report — HTML report and PowerBI push for model health dashboard.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>FSIP Model Governance Report — {date}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #2c3e50; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
    h2 {{ color: #34495e; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .ok {{ color: #27ae60; font-weight: bold; }}
    .warning {{ color: #f39c12; font-weight: bold; }}
    .critical {{ color: #e74c3c; font-weight: bold; }}
    .footer {{ margin-top: 40px; color: #7f8c8d; font-size: 12px; border-top: 1px solid #ecf0f1; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>FSIP Model Governance Report</h1>
  <p><strong>Author:</strong> Joshua Lutkemuller, Managing Director — Securities Finance &amp; Agency Lending</p>
  <p><strong>Generated:</strong> {datetime_utc} UTC</p>

  <h2>Model Drift Status</h2>
  {drift_table}

  <h2>Performance Metrics (Rolling 30d)</h2>
  {performance_table}

  <h2>Recent Alerts</h2>
  {alert_table}

  <div class="footer">
    FSIP Financing Solutions Intelligence Platform | SR 11-7 Compliant Model Governance
    | Joshua Lutkemuller, MD
  </div>
</body>
</html>
"""


class GovernanceReportGenerator:
    """Generates HTML governance report and pushes summary to PowerBI."""

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("governance_report", {})
        self.output_html = Path(self.cfg.get("output_html", "outputs/governance_report.html"))

    def generate(
        self,
        drift_reports: list,
        performance_records: list[dict],
        recent_alerts: list[dict],
    ) -> str:
        """Build and save the HTML governance report."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        datetime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        drift_table = self._drift_table_html(drift_reports)
        perf_table = self._perf_table_html(performance_records)
        alert_table = self._alert_table_html(recent_alerts)

        html = HTML_TEMPLATE.format(
            date=date_str,
            datetime_utc=datetime_str,
            drift_table=drift_table,
            performance_table=perf_table,
            alert_table=alert_table,
        )

        self.output_html.parent.mkdir(parents=True, exist_ok=True)
        self.output_html.write_text(html)
        logger.info("Governance report saved to %s", self.output_html)

        if self.cfg.get("powerbi_push") and os.environ.get("POWERBI_WORKSPACE_ID"):
            self._push_to_powerbi(drift_reports)

        return str(self.output_html)

    def _drift_table_html(self, drift_reports: list) -> str:
        if not drift_reports:
            return "<p>No drift reports available.</p>"
        rows = []
        for r in drift_reports:
            css = r.drift_status.lower()
            rows.append(
                f"<tr><td>{r.model_id}</td>"
                f"<td class='{css}'>{r.drift_status}</td>"
                f"<td>{r.max_psi:.4f}</td>"
                f"<td>{r.mean_psi:.4f}</td>"
                f"<td>{r.n_features_drifted}/{r.n_features_tested}</td>"
                f"<td>{'Yes' if r.concept_drift_detected else 'No'}</td></tr>"
            )
        header = "<tr><th>Model</th><th>Status</th><th>Max PSI</th><th>Mean PSI</th><th>Drifted Features</th><th>Concept Drift</th></tr>"
        return f"<table>{header}{''.join(rows)}</table>"

    def _perf_table_html(self, records: list[dict]) -> str:
        if not records:
            return "<p>No performance records available.</p>"
        df = pd.DataFrame(records)
        return df.to_html(index=False, classes="", border=0).replace(
            "<table", '<table style="border-collapse:collapse;width:100%"'
        )

    def _alert_table_html(self, alerts: list[dict]) -> str:
        if not alerts:
            return "<p>No recent governance alerts.</p>"
        rows = []
        for a in alerts[:20]:
            sev = a.get("severity", "INFO").lower()
            rows.append(
                f"<tr><td>{a.get('timestamp', '')[:19]}</td>"
                f"<td>{a.get('model_id', '')}</td>"
                f"<td class='{sev}'>{a.get('severity', '')}</td>"
                f"<td>{a.get('alert_type', '')}</td>"
                f"<td>{a.get('message', '')[:80]}</td></tr>"
            )
        header = "<tr><th>Timestamp</th><th>Model</th><th>Severity</th><th>Type</th><th>Message</th></tr>"
        return f"<table>{header}{''.join(rows)}</table>"

    def _push_to_powerbi(self, drift_reports: list) -> None:
        """Push drift summary rows to PowerBI."""
        logger.info("Pushing %d drift records to PowerBI", len(drift_reports))
        # PowerBI push would follow the same pattern as existing powerbi_connector.py
