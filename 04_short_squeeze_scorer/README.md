# Short Squeeze Probability Scorer

A machine learning system that scores individual securities for the probability of a short squeeze event using XGBoost trained on historical squeeze events. Key signals include short interest ratio, days-to-cover, borrow availability, utilization rate, and price momentum. Alerts fire when a security crosses into HIGH or CRITICAL squeeze probability. Results are pushed to the Financing Solutions Intelligence Platform (PowerBI).

---

## Architecture

```
+---------------------------+
|     Data Sources          |
|  - Short Interest Feed    |
|  - Borrow Availability    |
|  - Price / Volume Data    |
+----------+----------------+
           |
           v
+----------+----------------+
|    Data Ingestion         |
|  ShortInterestDataLoader  |
|  merge_all / validate     |
+----------+----------------+
           |
           v
+----------+----------------+
|  Feature Engineering      |
|  - Days-to-Cover          |
|  - Utilization Change     |
|  - SI Ratio Momentum      |
|  - Price Momentum         |
|  - RSI                    |
|  - Volume Ratio           |
|  - Borrow Cost Z-Score    |
+----------+----------------+
           |
           v
+----------+----------------+
|   XGBoost Scorer          |
|  squeeze_probability 0-1  |
+----------+----------------+
           |
           v
+----------+----------------+
|  Risk Tier Classification |
|  LOW / ELEVATED /         |
|  HIGH / CRITICAL          |
+----------+----------------+
           |
      +----+----+
      |         |
      v         v
+-----+---+ +---+--------+
|  Alert  | |  PowerBI   |
| Dispatch| |   Push     |
| email / | | Streaming  |
| teams / | | Dataset    |
|   log   | +------------+
+---------+
      |
      v
+-----+---+
|  Audit  |
|   Log   |
+---------+
```

---

## Features

- **XGBoost-Based Scoring** — Trained on labeled historical short squeeze events to output a calibrated probability 0–1 for each ticker.
- **Multi-Signal Feature Engineering** — Combines short interest ratio, days-to-cover, utilization rate, borrow rate z-score, RSI, volume ratio, and price/SI momentum across configurable rolling windows.
- **Four-Tier Risk Classification** — Tickers are bucketed into LOW, ELEVATED, HIGH, or CRITICAL based on configurable probability thresholds.
- **Multi-Channel Alerting** — Alerts for HIGH and CRITICAL tiers dispatch via email, Microsoft Teams webhook, and structured log file.
- **PowerBI Streaming Push** — Scored results stream directly into the Financing Solutions Intelligence Platform PowerBI dataset using the Push Datasets API.
- **Fully Configurable** — All model hyperparameters, data paths, windows, thresholds, and alert destinations are controlled from `config/config.yaml` with secrets in `.env`.
- **Audit Trail** — Every pipeline run and alert dispatch is written to a structured log for compliance and reproducibility.

---

## Tech Stack

| Component              | Library / Service                        |
|------------------------|------------------------------------------|
| ML Model               | `xgboost` 2.x                            |
| Feature Engineering    | `pandas`, `numpy`, `ta`                  |
| Data Validation        | `pandas` schema checks                   |
| Configuration          | `pyyaml`, `python-dotenv`                |
| Alerting — HTTP        | `requests` (Teams webhook, PowerBI API)  |
| Alerting — Email       | `smtplib` (stdlib)                       |
| Model Persistence      | `xgboost` native JSON / pickle           |
| Testing                | `pytest`, `numpy` synthetic data         |
| Orchestration          | `argparse` CLI entry point               |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

Required variables:

```
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USER=alerts@company.com
SMTP_PASSWORD=secret
ALERT_FROM=alerts@company.com
ALERT_TO=risk-team@company.com
POWERBI_ACCESS_TOKEN=eyJ...
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
```

### 3. Edit configuration

Open `config/config.yaml` and update data paths, model hyperparameters, and PowerBI dataset IDs for your environment.

---

## Running the Pipeline

```bash
# Score all tickers with current data
python src/main.py --config config/config.yaml

# Train (or retrain) the model on labeled data
python src/main.py --config config/config.yaml --train --labels data/squeeze_labels.csv

# Score a specific ticker
python src/main.py --config config/config.yaml --ticker BBBY

# Dry-run (skip alert dispatch and PowerBI push)
python src/main.py --config config/config.yaml --dry-run
```

---

## Configuration Reference (`config/config.yaml`)

| Section         | Key                            | Description                                          |
|-----------------|--------------------------------|------------------------------------------------------|
| `data`          | `short_interest_path`          | CSV path for short interest data                     |
| `data`          | `borrow_availability_path`     | CSV path for borrow availability data                |
| `data`          | `price_data_path`              | CSV path for OHLCV price data                        |
| `data`          | `lookback_days`                | Rolling lookback window for feature computation      |
| `model`         | `n_estimators`                 | Number of XGBoost trees                              |
| `model`         | `max_depth`                    | Maximum tree depth                                   |
| `model`         | `learning_rate`                | XGBoost eta                                          |
| `model`         | `squeeze_prob_threshold_*`     | Probability thresholds for ELEVATED / HIGH / CRITICAL|
| `features`      | `si_ratio_windows`             | Rolling windows for SI ratio momentum features       |
| `features`      | `momentum_windows`             | Rolling windows for price return features            |
| `features`      | `volume_windows`               | Rolling windows for volume ratio features            |
| `alerts`        | `enabled`                      | Master alert toggle                                  |
| `alerts`        | `email`                        | Email alert configuration block                      |
| `alerts`        | `teams_webhook_url`            | Microsoft Teams incoming webhook URL                 |
| `alerts`        | `log_path`                     | File path for structured alert log                   |
| `powerbi`       | `workspace_id`                 | PowerBI workspace GUID                               |
| `powerbi`       | `dataset_id`                   | PowerBI push dataset GUID                            |
| `powerbi`       | `table_name`                   | Target table within the push dataset                 |
| `powerbi`       | `streaming_url`                | Full streaming endpoint URL (overrides workspace/dataset) |
| `powerbi`       | `enabled`                      | Toggle PowerBI push on/off                           |

---

## Alert Types

| Type     | Trigger                        | Destination                       |
|----------|--------------------------------|-----------------------------------|
| `email`  | HIGH or CRITICAL risk tier     | SMTP relay to configured recipient |
| `teams`  | HIGH or CRITICAL risk tier     | Teams channel via incoming webhook |
| `log`    | All tiers (configurable)       | Structured JSON lines in log file  |

---

## PowerBI Push Dataset Schema

| Column                  | Type      | Description                              |
|-------------------------|-----------|------------------------------------------|
| `ticker`                | string    | Security ticker symbol                   |
| `short_interest_ratio`  | float     | Shares short / shares float              |
| `days_to_cover`         | float     | Shares short / avg daily volume          |
| `utilization_rate`      | float     | Shares on loan / shares available        |
| `squeeze_probability`   | float     | XGBoost output probability 0–1           |
| `risk_tier`             | string    | LOW / ELEVATED / HIGH / CRITICAL         |
| `alert_flag`            | boolean   | True if HIGH or CRITICAL                 |
| `timestamp`             | datetime  | UTC timestamp of scoring run             |

---

## Running Tests

```bash
pytest tests/ -v
```
