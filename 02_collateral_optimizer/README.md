# Collateral Optimization & Substitution Recommender

## Overview

This project ranks and recommends optimal collateral substitutions for securities finance
transactions. A **Gradient Boosting** model (LightGBM) scores each piece of collateral on
eligibility and quality, and those scores feed into a **CVXPY** optimization problem that
minimises financing cost subject to eligibility, concentration, and haircut constraints.
When suboptimal collateral is detected — or when the optimiser fails to find a feasible
solution — alerts are dispatched via email, Microsoft Teams, and/or log file. Final results
are pushed to the **Financing Solutions Intelligence Platform** (PowerBI).

---

## Features

- **ML-based quality scoring** — LightGBM regressor trained on historical collateral
  performance; outputs a continuous quality score (0–1) per collateral item.
- **Convex portfolio optimisation** — CVXPY formulates the substitution decision as a
  quadratic programme; minimises `financing_cost @ x − quality_weight * quality_scores @ x`
  subject to feasibility constraints.
- **Concentration & haircut guardrails** — hard constraints prevent over-concentration in
  any single asset type and reject collateral whose haircut exceeds the configured ceiling.
- **Multi-channel alerting** — configurable thresholds trigger email, Teams webhook, and
  structured log alerts when cost savings are available or the optimiser is infeasible.
- **PowerBI streaming push** — every optimisation run streams a row per collateral item to
  a configured PowerBI push dataset for live dashboard updates.
- **Fully configurable** — all model hyper-parameters, constraint bounds, alert thresholds,
  and integration URLs live in `config/config.yaml`; secrets are loaded from `.env`.
- **Reproducible audit log** — every recommendation round-trip is written to a structured
  JSONL audit log for compliance and back-testing.

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| Data wrangling | pandas, numpy |
| ML scoring | lightgbm, scikit-learn |
| Optimisation | cvxpy (ECOS / SCS solver) |
| Config | pyyaml, python-dotenv |
| Alerting | smtplib (email), requests (Teams) |
| Reporting | PowerBI REST API (requests) |
| Testing | pytest |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Collateral Optimizer Pipeline                    │
│                                                                         │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
│  │  Collateral  │   │  Eligibility     │   │  Haircut Table /       │  │
│  │  Pool CSV    │   │  Schedule CSV    │   │  Transaction CSV       │  │
│  └──────┬──────┘   └────────┬─────────┘   └───────────┬────────────┘  │
│         └──────────────┬────┘                         │               │
│                        ▼                              │               │
│              ┌──────────────────┐                     │               │
│              │  Data Ingestion  │◄────────────────────┘               │
│              │  (merge_all)     │                                      │
│              └────────┬─────────┘                                      │
│                       │                                                │
│                       ▼                                                │
│              ┌──────────────────┐                                      │
│              │ Feature Engineer │  duration, rating, liquidity,        │
│              │                  │  concentration, cost_of_carry        │
│              └────────┬─────────┘                                      │
│                       │                                                │
│                       ▼                                                │
│              ┌──────────────────┐                                      │
│              │  LightGBM        │  quality_score ∈ [0, 1]             │
│              │  Quality Scorer  │                                      │
│              └────────┬─────────┘                                      │
│                       │                                                │
│                       ▼                                                │
│              ┌──────────────────┐  min  financing_cost @ x            │
│              │  CVXPY           │      − quality_weight * q @ x       │
│              │  Optimizer       │  s.t. Σx=1, x≥0,                   │
│              │                  │       concentration ≤ max_conc,     │
│              │                  │       haircut ≤ max_haircut,        │
│              │                  │       q @ x ≥ min_quality           │
│              └────────┬─────────┘                                      │
│                       │                                                │
│                  ┌────┴──────┐                                         │
│                  ▼           ▼                                         │
│          ┌──────────┐  ┌───────────┐                                   │
│          │  Alert   │  │  PowerBI  │                                   │
│          │Dispatcher│  │ Connector │                                   │
│          └──────────┘  └───────────┘                                   │
│                  │           │                                         │
│            Email/Teams   Streaming                                     │
│            /Log file     Push Dataset                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone / navigate

```bash
cd /home/user/sec_finance_ml/02_collateral_optimizer
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure secrets

Copy `.env.example` to `.env` and populate:

```
POWERBI_CLIENT_ID=...
POWERBI_CLIENT_SECRET=...
POWERBI_TENANT_ID=...
SMTP_USER=...
SMTP_PASSWORD=...
```

### 4. Edit `config/config.yaml`

Update file paths under `data:` and set integration URLs under `powerbi:` and `alerts:`.

---

## How to Run

```bash
python src/main.py                          # default: full pipeline
python src/main.py --config config/config.yaml --mode score_only
python src/main.py --mode optimize_only
```

### Modes

| Mode | Description |
|---|---|
| `full` | Ingest → Feature Engineer → Score → Optimise → Alert → Push |
| `score_only` | Ingest → Feature Engineer → Score (no optimisation) |
| `optimize_only` | Load pre-scored data → Optimise → Alert → Push |

---

## Configuration Reference (`config/config.yaml`)

| Key | Description |
|---|---|
| `data.collateral_pool_path` | Path to collateral pool CSV |
| `data.eligibility_schedule_path` | Path to eligibility rules CSV |
| `data.haircut_table_path` | Path to haircut schedule CSV |
| `data.transaction_path` | Path to live/historical transactions CSV |
| `model.n_estimators` | Number of LightGBM trees |
| `model.quality_score_threshold` | Minimum acceptable quality score |
| `optimization.max_concentration_pct` | Max weight per asset_type bucket |
| `optimization.solver` | CVXPY solver: `ECOS` or `SCS` |
| `alerts.cost_savings_threshold_bps` | Alert when savings exceed N bps |
| `powerbi.streaming_url` | PowerBI push dataset REST endpoint |

---

## Alert Types

| Type | Trigger | Channel |
|---|---|---|
| `SUBOPTIMAL_COLLATERAL` | Quality score below threshold | Email, Teams, Log |
| `OPTIMISATION_FAILURE` | CVXPY returns infeasible/unbounded | Email, Teams, Log |
| `COST_SAVINGS_AVAILABLE` | Savings ≥ `cost_savings_threshold_bps` | Email, Teams, Log |
| `CONCENTRATION_BREACH` | Allocation exceeds `max_concentration_pct` | Log |

---

## PowerBI Push Dataset Schema

Table name configured via `powerbi.table_name` (default: `collateral_recommendations`).

| Column | Type | Description |
|---|---|---|
| `collateral_id` | String | Unique collateral identifier |
| `asset_type` | String | Asset class (e.g. Govt Bond, Equity) |
| `haircut` | Double | Applied haircut percentage |
| `quality_score` | Double | ML model output score (0–1) |
| `recommended` | Boolean | True if selected by optimiser |
| `optimization_cost` | Double | Weighted financing cost contribution |
| `timestamp` | DateTime | UTC timestamp of optimisation run |
