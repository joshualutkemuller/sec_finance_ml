# Portfolio Concentration Risk Early Warning Monitor

## Overview

This project monitors securities finance portfolios for **emerging concentration risk** using a
Random Forest ensemble trained on historical risk events. The system ingests position data, market
values, and counterparty exposures on a scheduled or on-demand basis, calculates a suite of
concentration metrics (HHI, top-N exposure percentages, sector and counterparty weights), engineers
rolling features, and feeds them into a trained Random Forest scorer. Each portfolio/sector/
counterparty dimension receives a continuous risk score between 0 and 1. When scores cross
configurable thresholds the system fires alerts and pushes results to the **Financing Solutions
Intelligence Platform** (PowerBI).

---

## Features

1. **Multi-dimensional concentration scoring** — Risk scores are computed independently per
   portfolio, sector, and counterparty dimension so that localised build-ups surface immediately.
2. **Herfindahl-Hirschman Index (HHI) engine** — Calculates and tracks HHI across all dimensions
   with configurable warning thresholds.
3. **SHAP-powered explainability** — Every alert includes the top-N risk drivers (feature
   contributions) so analysts understand *why* the score changed, not just that it changed.
4. **Four-level risk classification** — Scores are mapped to LOW / MEDIUM / HIGH / CRITICAL risk
   levels; only HIGH and CRITICAL trigger active alerts, reducing alert fatigue.
5. **Multi-channel alerting** — Dispatch alerts via email (SMTP), Microsoft Teams webhooks, and
   structured log files; channels are independently enable/disable in config.
6. **PowerBI streaming push** — Scored results are pushed in real-time to a PowerBI push dataset
   so the Financing Solutions Intelligence Platform dashboard refreshes automatically.
7. **Fully configurable via YAML** — All thresholds, model hyper-parameters, data paths, and
   integration endpoints live in `config/config.yaml`; no code changes required for operational
   tuning.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and populate the environment file
cp .env.example .env
# Edit .env with SMTP credentials, Teams webhook, PowerBI token, etc.

# 4. Review / adjust config
nano config/config.yaml
```

---

## Running the Pipeline

```bash
# Score today's snapshot and dispatch alerts
python src/main.py --run-date today

# Score a specific date
python src/main.py --run-date 2025-06-01

# Train / retrain the model from historical labelled data
python src/main.py --train --train-data data/historical_risk_events.csv

# Dry-run: score and log but suppress all external dispatches
python src/main.py --run-date today --dry-run
```

---

## Configuration (`config/config.yaml`)

| Section | Key | Description |
|---|---|---|
| `data` | `positions_path` | Path to positions CSV / Parquet |
| `data` | `lookback_days` | Days of history for rolling features |
| `model` | `n_estimators` | Number of trees in the Random Forest |
| `model` | `risk_score_threshold_*` | Score cut-offs for MEDIUM / HIGH / CRITICAL |
| `concentration` | `hhi_warning_threshold` | HHI level that triggers a warning flag |
| `concentration` | `top5_exposure_warning_pct` | % of AUM in top-5 names that triggers flag |
| `alerts` | `enabled` | Master on/off for all alert dispatch |
| `alerts` | `teams_webhook_url` | Teams incoming webhook URL |
| `powerbi` | `streaming_url` | PowerBI push dataset REST endpoint |
| `shap` | `top_n_features` | Number of top SHAP drivers per alert |

---

## Alert Types

| Channel | Trigger | Config key |
|---|---|---|
| **Email** | HIGH or CRITICAL risk level | `alerts.email.*` |
| **Teams** | HIGH or CRITICAL risk level | `alerts.teams_webhook_url` |
| **Log** | All risk levels (INFO / WARNING / ERROR) | `alerts.log_path` |

---

## PowerBI Push Dataset Schema

The system pushes one row per scored dimension to the PowerBI streaming dataset:

| Column | Type | Description |
|---|---|---|
| `portfolio_id` | string | Portfolio identifier |
| `sector` | string | GICS sector or "ALL" for portfolio-level |
| `counterparty` | string | Counterparty LEI / name or "ALL" |
| `concentration_score` | float | Model risk score 0–1 |
| `risk_level` | string | LOW / MEDIUM / HIGH / CRITICAL |
| `alert_flag` | boolean | True if alert was dispatched |
| `timestamp` | datetime (ISO 8601) | Scoring run timestamp (UTC) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Positions   │  │  Market Values   │  │  Counterparty Exp.   │  │
│  │  (CSV/DB)    │  │  (CSV/API)       │  │  (CSV/DB)            │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘  │
└─────────┼──────────────────┼────────────────────────┼─────────────┘
          │                  │                        │
          └──────────────────▼────────────────────────┘
                      ┌─────────────────┐
                      │  DataLoader     │
                      │  (data_ingest.) │
                      └────────┬────────┘
                               │  exposure_snapshot DataFrame
                               ▼
                      ┌─────────────────────────────┐
                      │  ConcentrationFeatureEngineer│
                      │  • HHI per dimension         │
                      │  • top-N exposure %          │
                      │  • rolling 5-day changes     │
                      │  • liquidity-adjusted exp.   │
                      └────────────┬────────────────┘
                                   │  feature matrix
                                   ▼
                      ┌─────────────────────────────┐
                      │  RandomForestRiskScorer      │
                      │  • score()  → 0-1 prob       │
                      │  • classify() → RiskLevel    │
                      │  • explain() → SHAP values   │
                      └────────────┬────────────────┘
                                   │  scores + risk levels + SHAP
                     ┌─────────────┴────────────────┐
                     ▼                              ▼
          ┌──────────────────┐          ┌───────────────────────┐
          │  RiskAlertDisp.  │          │  PowerBI Connector    │
          │  • email         │          │  • push_rows()        │
          │  • Teams webhook │          │  • streaming REST API │
          │  • log file      │          └───────────────────────┘
          └──────────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │  Audit Log       │
          │  (JSONL)         │
          └──────────────────┘
```

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn (RandomForestClassifier) |
| Explainability | shap |
| Configuration | PyYAML, python-dotenv |
| HTTP (Teams / PowerBI) | requests |
| Plotting (SHAP) | matplotlib |
| Testing | pytest |
| Alerting (email) | smtplib (stdlib) |
| Alerting (Teams) | Incoming Webhook (HTTP POST) |
| Dashboard | Microsoft PowerBI push datasets |
