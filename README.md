# Financing Solutions Intelligence Platform — ML Alert Engine

A collection of five standalone machine learning projects that power the **Financing Solutions Intelligence Platform (FSIP)** — a PowerBI-based analytics dashboard for securities finance, lending, and portfolio operations.

Each project is fully self-contained with its own dependencies, configuration, data pipeline, ML model, alert system, and a PowerBI connector layer that pushes results to the FSIP dashboard via the PowerBI REST API (Push Datasets / Streaming Datasets).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Data Sources                                │
│  Market Data │ Loan Books │ Collateral DB │ Repo/Rate Feeds │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────▼───────────────┐
           │     Per-Project ML Pipeline    │
           │  Ingest → Features → Model →  │
           │  Score → Alert → Publish      │
           └───┬───────────────────────┬───┘
               │                       │
    ┌──────────▼──────────┐  ┌─────────▼──────────┐
    │   Alert Engine      │  │  PowerBI Connector  │
    │  Email / Teams /    │  │  Push Dataset API   │
    │  Webhook / Log      │  │  Streaming Tiles    │
    └─────────────────────┘  └────────────────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  Financing Solutions   │
                            │  Intelligence Platform │
                            │       (PowerBI)        │
                            └───────────────────────┘
```

---

## Projects

| # | Folder | Description | ML Approach |
|---|--------|-------------|-------------|
| 1 | [`01_lending_rate_anomaly`](./01_lending_rate_anomaly/) | Detect anomalous securities lending borrow rates | Isolation Forest + LSTM |
| 2 | [`02_collateral_optimizer`](./02_collateral_optimizer/) | Recommend optimal collateral substitutions | Gradient Boosting + CVXPY |
| 3 | [`03_portfolio_risk_monitor`](./03_portfolio_risk_monitor/) | Early warning for portfolio concentration risk | Random Forest Ensemble |
| 4 | [`04_short_squeeze_scorer`](./04_short_squeeze_scorer/) | Score securities for short squeeze probability | XGBoost |
| 5 | [`05_financing_cost_forecaster`](./05_financing_cost_forecaster/) | Forecast repo and financing costs | LightGBM Time Series |

---

## PowerBI Integration

All five projects share a common integration pattern with the Financing Solutions Intelligence Platform:

1. **Push Datasets** — each project pushes scored rows and alert metadata to a dedicated PowerBI dataset table.
2. **Streaming Tiles** — real-time alert counts and risk scores are streamed to dashboard tiles.
3. **Authentication** — uses Azure AD service principal (`POWERBI_CLIENT_ID`, `POWERBI_CLIENT_SECRET`, `POWERBI_TENANT_ID`) configured via environment variables.

See each project's `src/powerbi_connector.py` and `config/config.yaml` for dataset-specific schemas.

---

## Shared Environment Setup

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate

# Each project has its own requirements.txt — install per project:
cd 01_lending_rate_anomaly
pip install -r requirements.txt
```

### Common Environment Variables

```bash
# PowerBI
export POWERBI_CLIENT_ID="<azure-ad-app-client-id>"
export POWERBI_CLIENT_SECRET="<azure-ad-app-secret>"
export POWERBI_TENANT_ID="<azure-ad-tenant-id>"
export POWERBI_WORKSPACE_ID="<powerbi-workspace-guid>"

# Alerting
export SMTP_HOST="smtp.yourdomain.com"
export SMTP_PORT="587"
export SMTP_USER="alerts@yourdomain.com"
export SMTP_PASSWORD="<password>"
export ALERT_EMAIL_RECIPIENTS="team@yourdomain.com"
export TEAMS_WEBHOOK_URL="<ms-teams-incoming-webhook-url>"
```

---

## Databricks Deployment

Each project can be scheduled as a Databricks Job with a Notebook or Python script task. The recommended pattern:

```
Cluster: Single-node or small cluster (4-8 cores)
Schedule: Every 15–60 minutes depending on data freshness
Library: install requirements.txt as a cluster library or init script
Entry point: src/main.py (each project)
```

---

## Repository Structure

```
sec_finance_ml/
├── README.md                         ← this file
├── 01_lending_rate_anomaly/
│   ├── README.md
│   ├── requirements.txt
│   ├── process_map.md
│   ├── config/config.yaml
│   ├── src/
│   │   ├── main.py
│   │   ├── data_ingestion.py
│   │   ├── feature_engineering.py
│   │   ├── model.py
│   │   ├── alerts.py
│   │   └── powerbi_connector.py
│   ├── tests/test_model.py
│   └── notebooks/exploration.ipynb
├── 02_collateral_optimizer/          ← same structure
├── 03_portfolio_risk_monitor/        ← same structure
├── 04_short_squeeze_scorer/          ← same structure
└── 05_financing_cost_forecaster/     ← same structure
```

---

## Contributing

- Each project is independently versioned and deployable.
- Follow PEP 8. Type hints required on all public functions.
- Add tests under `tests/` before merging new model versions.
- Keep secrets out of code — use environment variables or a secrets manager.
