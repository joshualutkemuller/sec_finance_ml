# Hooks, Startup & Stress Testing Guide

This folder contains operational tooling for the **Financing Solutions Intelligence Platform (FSIP)** ML pipeline — covering Git hooks for code quality enforcement, environment startup scripts for each of the five ML projects, and pointers to the stress testing framework.

---

## Folder Structure

```
hooks_and_startup/
├── README.md                          ← this file
├── hooks/
│   ├── pre-commit                     ← lint, type-check, secrets scan before every commit
│   ├── pre-push                       ← run full test suite before push
│   ├── post-merge                     ← auto-reinstall deps when requirements change
│   └── install_hooks.sh               ← one-time script to wire all hooks into .git/hooks/
└── startup/
    ├── setup_env.sh                   ← create venv + export all required env vars
    ├── validate_env.sh                ← verify all required env vars are set before run
    ├── startup_all_projects.sh        ← score mode on all 5 projects in sequence
    └── per_project/
        ├── 01_lending_rate_anomaly.sh
        ├── 02_collateral_optimizer.sh
        ├── 03_portfolio_risk_monitor.sh
        ├── 04_short_squeeze_scorer.sh
        └── 05_financing_cost_forecaster.sh
```

---

## 1. Git Hooks

### Overview

| Hook | Trigger | Purpose |
|------|---------|---------|
| `pre-commit` | `git commit` | Lint (flake8), type-check (mypy), scan for secrets |
| `pre-push` | `git push` | Run full pytest suite for changed projects |
| `post-merge` | `git pull` / merge | Reinstall requirements if `requirements.txt` changed |

### Installation

```bash
# From the repository root — run once per clone
chmod +x hooks_and_startup/hooks/install_hooks.sh
./hooks_and_startup/hooks/install_hooks.sh
```

This copies all hooks from `hooks_and_startup/hooks/` into `.git/hooks/` and makes them executable.

### Skipping Hooks (Emergency Only)

```bash
git commit --no-verify   # bypass pre-commit (document reason in commit message)
git push --no-verify     # bypass pre-push
```

> Do not bypass hooks routinely — they exist to prevent broken models or secrets from reaching the repo.

---

## 2. Environment Setup

### Prerequisites

- Python 3.10+
- `pip` (pip 23+)
- Access to environment variables (see below)

### Full Setup (First Time)

```bash
# From repo root
chmod +x hooks_and_startup/startup/setup_env.sh
./hooks_and_startup/startup/setup_env.sh
```

This script will:
1. Create a `.venv` virtual environment at the repo root
2. Export all required environment variables from a local `.env` file
3. Install each project's `requirements.txt` in sequence

### Environment Variable Reference

All secrets must be set as environment variables. **Never commit them.**

```bash
# ── PowerBI / Azure AD ────────────────────────────────────────
POWERBI_CLIENT_ID        # Azure AD app registration client ID
POWERBI_CLIENT_SECRET    # Azure AD app registration secret
POWERBI_TENANT_ID        # Azure AD tenant (directory) ID
POWERBI_WORKSPACE_ID     # PowerBI workspace GUID

# ── Email Alerts (SMTP) ───────────────────────────────────────
SMTP_HOST                # e.g. smtp.office365.com
SMTP_PORT                # e.g. 587
SMTP_USER                # sending account
SMTP_PASSWORD            # SMTP password / app password
ALERT_FROM               # From address
ALERT_EMAIL_RECIPIENTS   # Comma-separated To addresses

# ── Microsoft Teams ───────────────────────────────────────────
TEAMS_WEBHOOK_URL        # Incoming webhook URL for the alert channel

# ── Databricks (optional) ─────────────────────────────────────
DATABRICKS_HOST          # e.g. https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN         # Personal access token or service principal token
DATABRICKS_HTTP_PATH     # SQL warehouse HTTP path
```

Create a `.env` file at the repo root (it is git-ignored):

```bash
cp hooks_and_startup/startup/.env.example .env
# Edit .env with your actual values
```

### Validate Environment

Before running any project, validate that all required variables are set:

```bash
./hooks_and_startup/startup/validate_env.sh
```

---

## 3. Running the Projects

### Run All Projects (Score Mode)

```bash
./hooks_and_startup/startup/startup_all_projects.sh
```

This runs each project in `--mode score` sequentially. Suitable for a scheduled batch job.

### Run Individual Projects

```bash
# Lending Rate Anomaly Detector
./hooks_and_startup/startup/per_project/01_lending_rate_anomaly.sh [train|score|both]

# Collateral Optimizer
./hooks_and_startup/startup/per_project/02_collateral_optimizer.sh [full|score_only|optimize_only]

# Portfolio Risk Monitor
./hooks_and_startup/startup/per_project/03_portfolio_risk_monitor.sh [train|score|both]

# Short Squeeze Scorer
./hooks_and_startup/startup/per_project/04_short_squeeze_scorer.sh [train|score|both]

# Financing Cost Forecaster
./hooks_and_startup/startup/per_project/05_financing_cost_forecaster.sh [train|score|both]
```

Default mode is `score` if no argument is supplied.

---

## 4. Databricks Deployment

For scheduled runs on Databricks, point the job to the individual per-project startup scripts or use each project's `src/main.py` directly:

```
Job type:        Python script
Entry point:     src/main.py
Parameters:      --config config/config.yaml --mode score
Cluster:         Single-node (Standard_DS3_v2 or equivalent)
Schedule:        Every 15–60 min (per project SLA)
Init script:     pip install -r requirements.txt
Environment:     Set all env vars in Databricks Secrets + cluster env config
```

---

## 5. Stress Testing

See the companion folder for what-if analysis and stress-testing the balance sheet and collateral optimization framework:

```
../stress_testing_framework/
```

---

## 6. Enhanced Securities Lending Optimizer

A high-level ML design for a reinforcement-learning and multi-objective enhanced optimizer is available in:

```
../securities_lending_optimizer/
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `KeyError: POWERBI_CLIENT_ID` | Env var not set | Run `validate_env.sh` |
| `ModuleNotFoundError` | venv not activated | `source .venv/bin/activate` |
| `cvxpy.error.SolverError` | ECOS not installed | `pip install ecos` |
| `OSError: [Errno 111] Connection refused` | SMTP host unreachable | Check `SMTP_HOST` / firewall |
| `DatabricksError: 403` | Token expired or wrong scope | Rotate `DATABRICKS_TOKEN` |
| Pre-commit hook fails | Lint / type errors | Fix flake8/mypy errors before committing |
| Pre-push hook fails | Test failures | Run `pytest` locally and fix failures |
