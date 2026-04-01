# Implementation Roadmap

## Phase Overview

| Phase | Focus | Effort | Priority |
|-------|-------|--------|----------|
| Phase 1 | Shock engine + scenario loader | 2 sprints | High |
| Phase 2 | Re-optimization under stress | 2 sprints | High |
| Phase 3 | Results analyzer + reporting | 1 sprint | Medium |
| Phase 4 | Monte Carlo simulation | 2 sprints | Medium |
| Phase 5 | Interactive what-if (PowerBI/Jupyter) | 2 sprints | Low |
| Phase 6 | Regulatory scenario alignment | 1 sprint | High (compliance) |

---

## Phase 1 — Shock Engine & Scenario Loader

**Goal**: Build the core machinery to apply any combination of shocks to the current balance sheet snapshot.

**Deliverables**:
- `src/scenario_loader.py` — reads YAML scenario files, validates schema
- `src/shock_engine.py` — applies rate, credit, equity, liquidity, counterparty, regulatory shocks
- All 15 predefined scenario YAML files in `scenarios/`
- Unit tests covering each shock type

**Key design decision**: Shock engine must be **non-destructive** — it returns a stressed copy of the input DataFrames, never modifying the originals. This allows easy comparison of stressed vs. base case.

```python
# Interface contract
stressed_inputs = shock_engine.apply(
    base_inputs=BaseInputs(loan_book, collateral_pool, market_data),
    scenario=scenario,
)
# base_inputs unchanged; stressed_inputs is a new copy
```

---

## Phase 2 — Stress Re-Optimization

**Goal**: Feed stressed inputs back through the existing optimizer and securities lending optimizer to get stressed allocations.

**Deliverables**:
- `src/stress_optimizer.py` — wraps both optimizers, runs stressed versions
- `reverse_stress()` method for Phase 1.5 reverse stress testing
- Integration test: for each scenario, assert stressed result differs from baseline by expected direction

**Key integration points**:
- Must call `02_collateral_optimizer/src/model.py` with stressed quality scores
- Must call `02_collateral_optimizer/src/main.py` CVXPY optimizer with stressed cost vectors
- Must pass stressed recall probabilities and demand forecasts to `securities_lending_optimizer`

---

## Phase 3 — Results Analyzer & Reporting

**Goal**: Compute delta metrics and produce reports.

**Deliverables**:
- `src/results_analyzer.py` — computes all delta metrics from the table in README
- `src/stress_report.py` — generates structured HTML/JSON reports, pushes to PowerBI
- PowerBI dataset table: `StressTestResults` (scenario_id, run_date, metric, baseline_value, stressed_value, delta, breach_flag)
- Alert dispatch for any metric breach

---

## Phase 4 — Monte Carlo Simulation

**Goal**: Sample from joint shock distribution and build output distributions.

**Deliverables**:
- Historical covariance matrix of (SOFR, IG spread, equity) estimated from 5yr window
- Monte Carlo engine: 10,000 samples, parallel execution via `concurrent.futures`
- VaR (95th, 99th) and ES (CVaR) reporting
- Convergence check: run 5,000 vs 10,000 samples to validate stability

---

## Phase 5 — Interactive What-If

**Goal**: Enable interactive exploration via Jupyter or PowerBI.

**Deliverables**:
- Jupyter notebook: `notebooks/interactive_stress_testing.ipynb`
  - `ipywidgets` sliders for each shock parameter
  - Tornado chart of sensitivities
  - Live optimizer rerun on slider change
- PowerBI:
  - `What-If Parameter` tables for key shocks (rate, equity, spread)
  - DAX measures computing stressed collateral values
  - Heatmap visual: scenario × metric grid

---

## Phase 6 — Regulatory Alignment

**Goal**: Ensure scenario outputs meet SFTR, Basel IV, SEC Rule 15c3-3 requirements.

**Deliverables**:
- ESMA SFTR adverse scenario (S11) certified against ESMA technical standards
- Basel IV SA-CCR capital charge calculation under each stress scenario
- SEC Rule 15c3-3 customer reserve formula stress analysis
- Automated regulatory report generation (PDF or structured XML)
- Compliance review checklist embedded in `src/stress_report.py`

---

## Dependencies on Existing Projects

```
stress_testing_framework
├── depends on: 02_collateral_optimizer/src/model.py (quality scorer)
├── depends on: 02_collateral_optimizer/src/main.py (CVXPY optimizer)
├── depends on: 03_portfolio_risk_monitor/src/model.py (risk scorer)
├── depends on: 05_financing_cost_forecaster/src/model.py (rate forecaster)
├── depends on: securities_lending_optimizer/src/multi_objective_optimizer.py
└── depends on: securities_lending_optimizer/src/recall_risk_classifier.py
```

All dependencies are called as library imports — the stress testing framework does NOT run standalone pipelines. It calls the existing models' `predict()` / `optimize()` methods with stressed input DataFrames.

---

## Testing Strategy

| Test Type | Coverage |
|-----------|---------|
| Unit tests | Each shock function independently |
| Integration tests | Stressed inputs → optimizer → results, direction check |
| Regression tests | Predefined scenarios produce expected output ranges |
| Boundary tests | Zero shock = base case; extreme shocks = bounded results |
| Performance tests | Full scenario suite must complete < 5 minutes |

---

## Operational Scheduling

Once built, the stress testing framework should run:

| Schedule | Scenarios | Trigger |
|----------|-----------|---------|
| Daily (EOD) | S01-S07, S13 | Cron after market close |
| Weekly | All 15 scenarios | Saturday morning |
| On-demand | Any scenario | Risk manager request |
| Pre-trade | User-defined what-if | Before large allocation decision |
| Regulatory | S11 (SFTR) | Monthly, reported to regulator |
