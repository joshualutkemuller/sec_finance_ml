# Model Governance Framework — SR 11-7 Compliant ML Operations

**Author:** Joshua Lutkemuller
**Title:** Managing Director, Securities Finance & Agency Lending
**Created:** 2026

---

## Overview

Any model touching P&L at a regulated institution requires sign-off from
Model Risk Management under **Federal Reserve SR 11-7** (Guidance on Model
Risk Management). This framework provides the complete governance layer for
all nine ML models in this platform.

An MD who builds the governance infrastructure alongside the models is the
one who gets sign-off from Risk, Compliance, and the Board — and the one
whose platform survives the model validation process.

**This is what separates a quant research project from an enterprise platform.**

---

## Governance Components

### 1. Model Cards
Standardized documentation for each model: intended use, training data,
known limitations, performance benchmarks, fairness considerations.
One model card per model, version-controlled alongside the code.

### 2. Data Drift Detection (Evidently AI)
Daily PSI (Population Stability Index) test comparing today's feature
distribution to the training distribution. PSI > 0.10 = WARNING.
PSI > 0.25 = CRITICAL (trigger retraining).

### 3. Concept Drift Detection
Monitors prediction distribution and rolling model error rate.
Fires alert when performance degrades below minimum acceptable threshold.

### 4. Shadow Mode Deployment
New model version runs in parallel with production, outputs compared
before cutover. No production impact until challenger beats champion.

### 5. Immutable Audit Log
Every model prediction logged with: feature snapshot (hash), model version,
run ID, output value, and timestamp. Queryable for SR 11-7 evidence.

### 6. Model Risk Dashboard (PowerBI)
- PSI heatmap across all 9 models
- Performance metrics rolling 30d
- Retraining history and sign-off log
- Alert counts by severity and model

---

## SR 11-7 Compliance Checklist

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Model inventory | `model_registry.yaml` | ✓ |
| Use case documentation | Model cards per model | ✓ |
| Conceptual soundness | README architecture docs | ✓ |
| Data quality | `data_validator.py` schema checks | ✓ |
| Performance monitoring | `drift_detector.py` | ✓ |
| Independent validation | Shadow mode output comparison | ✓ |
| Ongoing monitoring | Daily PSI + performance alerts | ✓ |
| Change management | Version-controlled model cards | ✓ |
| Audit trail | Immutable JSONL logs per model | ✓ |

---

## Module Structure

```
model_governance_framework/
├── README.md
├── requirements.txt
├── config/config.yaml
├── model_cards/
│   ├── 01_lending_rate_anomaly.md
│   ├── 02_collateral_optimizer.md
│   ├── 03_portfolio_risk_monitor.md
│   ├── 04_short_squeeze_scorer.md
│   ├── 05_financing_cost_forecaster.md
│   ├── securities_lending_optimizer.md
│   ├── fred_macro_intelligence.md
│   ├── counterparty_graph_risk.md
│   └── nlp_signal_engine.md
└── src/
    ├── main.py                   ← CLI: validate / drift / report / full
    ├── model_registry.py         ← central model inventory + version tracking
    ├── drift_detector.py         ← Evidently PSI + concept drift
    ├── performance_monitor.py    ← rolling performance metrics per model
    ├── shadow_mode.py            ← champion/challenger comparison framework
    ├── audit_logger.py           ← immutable prediction audit log
    └── governance_report.py      ← PowerBI push + HTML governance report
```
