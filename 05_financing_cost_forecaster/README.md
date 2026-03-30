# Repo & Financing Cost Forecaster

## Overview

This project forecasts short-term (1-day, 3-day, 5-day horizon) repo rates and total financing costs
for a securities finance book using LightGBM gradient-boosted trees. It incorporates macro rate
signals (SOFR, Fed Funds), supply/demand indicators from book positions, and historical rate patterns
via lag and rolling features. When forecasted costs exceed budget thresholds or rate spikes are
predicted, the system fires configurable alerts (email, Microsoft Teams webhook, log file). Final
forecast results and budget comparisons are pushed to the Financing Solutions Intelligence Platform
(PowerBI) for real-time dashboard consumption.

---

## Features

1. **Multi-Horizon Forecasting** — Independent LightGBM models trained per forecast horizon (1d, 3d,
   5d), each with horizon-shifted targets to avoid lookahead bias.
2. **Macro Rate Integration** — SOFR, Fed Funds, OIS, and legacy LIBOR-3M ingested and used to
   compute repo-SOFR spread and curve shape features that capture monetary policy dynamics.
3. **Lag & Rolling Feature Engineering** — Configurable lag windows (1, 3, 5, 10, 20 days) and
   rolling statistics (mean, std, min, max) applied to rate and spread columns to encode momentum
   and regime context.
4. **Calendar Awareness** — Day-of-week, month-end, quarter-end, and days-to-month-end flags account
   for known seasonal patterns in repo market supply/demand.
5. **Budget Breach & Spike Alerts** — Rule-based post-processing layer compares forecasts to budget
   rates and prior-day actuals; dispatches typed alerts via email SMTP, Teams incoming webhook, or
   structured log entries.
6. **PowerBI Streaming Push** — Forecast results formatted and pushed to a PowerBI streaming dataset
   on each pipeline run, enabling live dashboard refresh without scheduled refreshes.
7. **Instrument-Level Granularity** — Separate forecasts per instrument type (UST, Agency, Corp,
   Equity) with book position notional used as a demand-pressure proxy feature.

---

## Architecture

```
+------------------------------------------------------------------+
|                  Repo & Financing Cost Forecaster                |
+------------------------------------------------------------------+
|                                                                  |
|  [Data Sources]         [Feature Layer]       [Model Layer]      |
|                                                                  |
|  repo_rates.csv  -----> Lag Features    ----> LightGBM (1d)     |
|  macro_rates.csv -----> Rolling Stats   ----> LightGBM (3d)     |
|  positions.csv   -----> Calendar Flags  ----> LightGBM (5d)     |
|  budget.csv      -----> Macro Spreads   |                        |
|                         Momentum        |                        |
|                         Term Structure  |                        |
|                                         |                        |
|                   [Post-Processing]     v                        |
|                                                                  |
|                   Forecasts ---------> Budget Comparison         |
|                       |                     |                    |
|                       v                     v                    |
|               PowerBI Push           Alert Dispatcher           |
|            (Streaming Dataset)    (Email / Teams / Log)         |
|                                                                  |
|                   [Audit Log]                                    |
+------------------------------------------------------------------+
```

---

## Tech Stack

| Component            | Technology                          |
|----------------------|-------------------------------------|
| ML Framework         | LightGBM 4.x                        |
| Data Processing      | pandas, numpy                       |
| Feature Engineering  | Custom (statsmodels for stationarity checks) |
| Config Management    | PyYAML + python-dotenv              |
| Alerting             | smtplib (email), requests (Teams)   |
| PowerBI Integration  | requests (REST streaming push)      |
| Testing              | pytest                              |
| Python Version       | 3.10+                               |

---

## Setup

```bash
# 1. Clone / navigate to project
cd 05_financing_cost_forecaster

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment secrets
cp .env.example .env
# Edit .env with SMTP credentials, Teams webhook, PowerBI token

# 5. Edit config
vi config/config.yaml
```

---

## Running

```bash
# Train models only
python src/main.py --config config/config.yaml --mode train

# Generate forecasts only (requires saved models)
python src/main.py --config config/config.yaml --mode forecast

# Full pipeline: train + forecast + alerts + PowerBI push
python src/main.py --config config/config.yaml --mode both
```

---

## Configuration (`config/config.yaml`)

| Section    | Key                       | Description                                      |
|------------|---------------------------|--------------------------------------------------|
| `data`     | `repo_rates_path`         | Path to historical repo rate CSV                 |
| `data`     | `lookback_days`           | Days of history to load for training             |
| `model`    | `horizons`                | Forecast horizons in days (default: [1, 3, 5])   |
| `model`    | `n_estimators`            | Number of boosting rounds                        |
| `features` | `lag_windows`             | Lag periods for rate features                    |
| `features` | `rolling_windows`         | Rolling stat window sizes                        |
| `alerts`   | `spike_threshold_bps`     | Bps increase vs prior day that triggers alert    |
| `alerts`   | `budget_breach_threshold_bps` | Bps over budget to trigger breach alert      |
| `powerbi`  | `streaming_url`           | PowerBI REST streaming push URL                  |

---

## Alert Types

| Alert Type      | Trigger Condition                                            | Channels           |
|-----------------|--------------------------------------------------------------|--------------------|
| `RATE_SPIKE`    | Forecasted rate exceeds prior day by `spike_threshold_bps`  | Email, Teams, Log  |
| `BUDGET_BREACH` | Forecasted rate exceeds budget by `budget_breach_threshold_bps` | Email, Teams, Log |
| `RAPID_INCREASE`| 3d forecast > 1d forecast by spike threshold (acceleration) | Teams, Log         |

---

## PowerBI Push Dataset Schema

| Field                | Type      | Description                                 |
|----------------------|-----------|---------------------------------------------|
| `instrument_type`    | string    | Instrument category (UST, Agency, Corp, etc)|
| `current_rate_bps`   | float     | Most recent observed rate in basis points   |
| `forecast_1d_bps`    | float     | 1-day ahead forecast in basis points        |
| `forecast_3d_bps`    | float     | 3-day ahead forecast in basis points        |
| `forecast_5d_bps`    | float     | 5-day ahead forecast in basis points        |
| `cost_vs_budget_bps` | float     | Forecast vs budget delta (5d horizon)       |
| `alert_flag`         | boolean   | True if any alert fired for this instrument |
| `timestamp`          | datetime  | Pipeline run timestamp (UTC)                |

---

## Project Structure

```
05_financing_cost_forecaster/
├── README.md
├── requirements.txt
├── process_map.md
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── alerts.py
│   ├── powerbi_connector.py
│   └── main.py
└── tests/
    └── test_model.py
```
