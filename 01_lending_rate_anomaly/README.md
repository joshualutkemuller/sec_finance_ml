# Securities Lending Rate Anomaly Detector

## Overview

This project detects anomalous borrow rates in securities lending books using a two-stage anomaly detection pipeline:

1. **Isolation Forest** — static, feature-based anomaly detection that scores each observation independently without temporal context.
2. **LSTM Autoencoder** — temporal pattern detection that learns normal rate sequences and flags observations whose reconstruction error exceeds a learned threshold.

Alerts fire when the combined anomaly score exceeds configurable thresholds. Results are pushed to the **Financing Solutions Intelligence Platform** (PowerBI) via both push datasets and streaming datasets for near-real-time dashboarding.

---

## Features

- **Dual-model anomaly detection**: Isolation Forest for point anomalies + LSTM Autoencoder for sequence anomalies, combined into a single blended score.
- **Rich feature engineering**: Rolling statistics (mean, std, z-score), spread vs. benchmark, momentum, rate-of-change, and liquidity proxies.
- **Multi-channel alerting**: Email (SMTP), Microsoft Teams webhook, and structured log file alerts with configurable severity levels (INFO / WARNING / CRITICAL).
- **PowerBI integration**: Pushes scored results to the Financing Solutions Intelligence Platform via REST API push datasets and streaming datasets.
- **Configurable thresholds**: All model hyperparameters, feature windows, and alert thresholds are controlled through `config/config.yaml` — no code changes required to tune the system.
- **Audit logging**: Every pipeline run, model fit, scoring event, and alert dispatch is written to an audit log for compliance and reproducibility.
- **Multiple data sources**: Supports CSV files, Databricks SQL queries, and REST API endpoints as data ingestion sources.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd 01_lending_rate_anomaly
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and fill in secrets:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|---|---|
| `SMTP_PASSWORD` | Password for the alert email account |
| `POWERBI_CLIENT_ID` | Azure AD application (client) ID |
| `POWERBI_CLIENT_SECRET` | Azure AD client secret |
| `POWERBI_TENANT_ID` | Azure AD tenant ID |
| `DATABRICKS_HOST` | Databricks workspace URL (if using Databricks source) |
| `DATABRICKS_TOKEN` | Databricks personal access token (if using Databricks source) |

### 5. Edit configuration

Update `config/config.yaml` with your environment-specific settings (paths, tickers, thresholds). See the **Configuration** section below.

---

## How to Run

```bash
# Train both models and score current data
python src/main.py --config config/config.yaml --mode both

# Train only
python src/main.py --config config/config.yaml --mode train

# Score only (models must already be trained)
python src/main.py --config config/config.yaml --mode score
```

---

## Configuration

All settings live in `config/config.yaml`. Key sections:

### `data`

| Field | Type | Description |
|---|---|---|
| `source_type` | str | One of `csv`, `databricks`, `api` |
| `path` | str | File path (csv) or API endpoint URL |
| `ticker_col` | str | Column name for the security ticker |
| `rate_col` | str | Column name for the borrow rate |
| `date_col` | str | Column name for the observation date/timestamp |
| `lookback_days` | int | Number of days of history to load |

### `isolation_forest`

| Field | Type | Description |
|---|---|---|
| `contamination` | float | Expected proportion of anomalies (e.g. 0.05) |
| `n_estimators` | int | Number of isolation trees |
| `random_state` | int | Random seed for reproducibility |

### `lstm`

| Field | Type | Description |
|---|---|---|
| `hidden_size` | int | LSTM hidden layer dimensionality |
| `num_layers` | int | Number of stacked LSTM layers |
| `epochs` | int | Training epochs for the autoencoder |
| `batch_size` | int | Training batch size |
| `learning_rate` | float | Adam optimizer learning rate |
| `sequence_length` | int | Number of timesteps per input sequence |
| `threshold_percentile` | float | Percentile of training reconstruction errors used as anomaly threshold (e.g. 95) |

### `alerts`

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Master switch for alerting |
| `anomaly_score_threshold` | float | Combined score above which an alert fires |
| `email.smtp_host` | str | SMTP server hostname |
| `email.port` | int | SMTP port (typically 587 or 465) |
| `email.from` | str | Sender email address |
| `email.to` | list[str] | Recipient email addresses |
| `teams_webhook_url` | str | MS Teams incoming webhook URL |
| `log_path` | str | Path to the alert log file |

### `powerbi`

| Field | Type | Description |
|---|---|---|
| `workspace_id` | str | PowerBI workspace (group) GUID |
| `dataset_id` | str | PowerBI push dataset GUID |
| `table_name` | str | Table name within the dataset |
| `streaming_url` | str | PowerBI streaming dataset push URL |
| `enabled` | bool | Master switch for PowerBI publishing |

---

## Alert Types

| Channel | Trigger | Description |
|---|---|---|
| **Email** | `combined_score > threshold` | SMTP email with alert details sent to configured recipients |
| **MS Teams Webhook** | `combined_score > threshold` | Adaptive card posted to Teams channel via incoming webhook |
| **Log file** | Every alert | Structured JSON log entry appended to `alerts.log_path` |

---

## PowerBI Integration

Results are pushed to the **Financing Solutions Intelligence Platform** PowerBI workspace.

### Push Dataset Schema

Table: configured via `powerbi.table_name`

| Column | Type | Description |
|---|---|---|
| `ticker` | String | Security identifier (CUSIP / ISIN / ticker) |
| `rate` | Double | Observed borrow rate (%) |
| `anomaly_score` | Double | Combined anomaly score [0, 1] |
| `alert_flag` | Boolean | True if score exceeds threshold |
| `timestamp` | DateTime | Observation timestamp (UTC) |

### Push Methods

- **Push Dataset** (`push_rows`): Batched historical rows via `POST /datasets/{datasetId}/tables/{tableName}/rows`
- **Streaming Dataset** (`stream_row`): Individual real-time rows via the streaming dataset push URL

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Securities Lending Rate Anomaly Detector         │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
  │  Data Source │     │  Feature Engineering │     │  Anomaly Models  │
  │              │     │                      │     │                  │
  │  CSV  ───────┤     │  Rolling Mean/Std    │     │  Isolation       │
  │  Databricks ─┼────▶│  Z-Score             │────▶│  Forest          │
  │  REST API ───┤     │  Spread vs Benchmark │     │                  │
  └──────────────┘     │  Momentum / ROC      │     │  LSTM            │
                       │  Liquidity Proxy     │     │  Autoencoder     │
                       └─────────────────────┘     └────────┬─────────┘
                                                             │
                                                             ▼
                                                   ┌──────────────────┐
                                                   │  Combined Score  │
                                                   │  (IF + LSTM      │
                                                   │   weighted avg)  │
                                                   └────────┬─────────┘
                                                             │
                                               ┌─────────────▼──────────────┐
                                               │    Threshold Evaluation     │
                                               └──────┬──────────────┬───────┘
                                                      │              │
                                              Anomaly │              │ Normal
                                                      ▼              ▼
                                          ┌───────────────────┐  ┌──────────┐
                                          │  Alert Dispatcher │  │  Log &   │
                                          │                   │  │  Archive │
                                          │  Email            │  └──────────┘
                                          │  Teams Webhook    │
                                          │  Log File         │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │  PowerBI Connector│
                                          │                   │
                                          │  Push Dataset     │
                                          │  Streaming Dataset│
                                          └───────────────────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │   Financing        │
                                          │   Solutions        │
                                          │   Intelligence     │
                                          │   Platform (PowerBI│
                                          └───────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| Isolation Forest | scikit-learn |
| LSTM Autoencoder | PyTorch |
| Configuration | PyYAML |
| HTTP / API calls | requests |
| Email alerts | smtplib (stdlib) |
| Teams alerts | MS Teams Incoming Webhook (JSON) |
| PowerBI | PowerBI REST API (push + streaming datasets) |
| Azure AD auth | OAuth2 client credentials (requests) |
| Secret management | python-dotenv |
| Data sources | CSV, Databricks SDK, REST API |
| Testing | pytest |
| Optional compute | Databricks (databricks-sdk) |
