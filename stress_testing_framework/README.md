# What-If Analysis & Stress Testing Framework
## Balance Sheet & Collateral Optimization

This folder provides a conceptual and implementation blueprint for integrating **what-if analysis** and **stress testing** into the existing collateral optimizer and balance sheet management workflows. It is designed to complement the five core ML projects by stress-testing their outputs under adverse market scenarios before allocations are acted upon.

---

## Why Stress Testing Matters Here

The existing ML projects (`02_collateral_optimizer`, `03_portfolio_risk_monitor`, `05_financing_cost_forecaster`) produce point estimates — optimal allocations, risk scores, and rate forecasts under current market conditions. They do not natively answer questions like:

- *"What happens to our collateral pool value if rates rise 200bps overnight?"*
- *"How does our financing cost change if we lose access to a top-3 counterparty?"*
- *"What is our worst-case margin call exposure under a 2008-style credit event?"*
- *"If the Fed hikes 75bps and equity markets fall 20%, does our collateral still meet haircut requirements?"*

This framework answers those questions systematically.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   Scenario Library                                  │
│  Historical │ Hypothetical │ Reverse Stress │ Regulatory Mandated  │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ shock parameters
         ┌─────────────▼─────────────────────┐
         │         Shock Engine              │
         │  Applies rate / spread / price /  │
         │  liquidity / counterparty shocks  │
         │  to balance sheet snapshot        │
         └─────────────┬─────────────────────┘
                       │ stressed inputs
    ┌──────────────────▼──────────────────────────────┐
    │            Re-optimization Layer                 │
    │  Reruns 02_collateral_optimizer with stressed    │
    │  inputs to find optimal allocation under stress  │
    └──────────────────┬──────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────┐
    │           Results & Reporting                    │
    │  P&L impact │ Margin call estimate │ VaR delta   │
    │  Collateral shortfall │ Capital charge delta     │
    │  PowerBI heatmaps │ Alert dispatch               │
    └──────────────────────────────────────────────────┘
```

---

## Folder Structure

```
stress_testing_framework/
├── README.md                         ← this file
├── what_if_analysis.md               ← detailed what-if methodology
├── stress_scenarios.md               ← predefined scenario catalog
├── implementation_roadmap.md         ← phased build plan
├── scenarios/
│   ├── base_case.yaml                ← no shocks (baseline)
│   ├── rate_shock_200bps.yaml
│   ├── credit_crunch_2008.yaml
│   ├── equity_crash_20pct.yaml
│   ├── counterparty_default.yaml
│   ├── liquidity_squeeze.yaml
│   └── regulatory_stress_lseg.yaml   ← ESMA/SEC mandated test
└── src/
    ├── scenario_loader.py            ← load YAML scenario definitions
    ├── shock_engine.py               ← apply shocks to balance sheet
    ├── stress_optimizer.py           ← re-run optimizer under stress
    ├── results_analyzer.py           ← compute P&L, shortfall, VaR delta
    └── stress_report.py              ← generate reports + push to PowerBI
```

---

## Quick Start

```bash
# Run a single named scenario
python src/stress_optimizer.py \
    --scenario scenarios/rate_shock_200bps.yaml \
    --optimizer-config ../02_collateral_optimizer/config/config.yaml

# Run all scenarios in batch and generate a report
python src/stress_report.py \
    --scenario-dir scenarios/ \
    --optimizer-config ../02_collateral_optimizer/config/config.yaml \
    --output-dir reports/
```

---

## What-If Analysis — Key Dimensions

See `what_if_analysis.md` for full methodology. Summary of the six stress dimensions:

| Dimension | What Changes | Impact Channel |
|-----------|-------------|----------------|
| Interest Rate | Parallel / twist shifts in yield curve | Collateral value, repo cost, financing cost |
| Credit Spread | Widening of issuer / sector spreads | Haircut eligibility, quality score, margin calls |
| Equity Market | Index-level and name-level price shocks | Equity collateral value, short squeeze risk |
| Liquidity | Bid-ask widening, market depth reduction | Substitution feasibility, fire-sale discount |
| Counterparty | Loss of one or more counterparties | Book concentration, recall risk |
| Regulatory | Haircut floor increases, LCR constraints | Eligible collateral universe shrinkage |

---

## Scenario Library

See `stress_scenarios.md` for the full catalog. Core scenarios:

| Scenario | Description | Severity |
|----------|-------------|----------|
| Base Case | No shocks — baseline optimizer output | — |
| Rate Shock +200bps | Instantaneous parallel rate increase | Severe |
| Rate Shock +100bps | Moderate rate rise | Moderate |
| Rate Twist (short +50, long -25) | Curve flattener | Moderate |
| Credit Crunch 2008 | Spread widening + liquidity freeze replica | Extreme |
| Equity Crash -20% | Broad equity market decline | Severe |
| Equity Crash -35% | Tail equity shock | Extreme |
| Top-1 Counterparty Default | Lose largest lending counterparty | Severe |
| Top-3 Counterparty Loss | Simultaneous loss of top 3 counterparties | Extreme |
| Regulatory Haircut Floor +5% | All haircuts increase by 5pp | Moderate |
| ESMA SFTR Adverse | Regulatory scenario per SFTR guidelines | Regulatory |
| Liquidity Squeeze | Market depth drops 70%, spreads triple | Extreme |
| Combined Stress | Rates +150bps + credit widens + equity -15% | Extreme |

---

## Integration with Existing Projects

### With `02_collateral_optimizer`
- Feed stressed collateral values and haircuts into the LightGBM quality scorer
- Rerun CVXPY optimizer with stressed cost vectors and tightened constraints
- Compare stressed vs. unstressed allocation and cost delta

### With `03_portfolio_risk_monitor`
- Stress HHI concentrations by removing counterparties or sectors
- Check whether stressed portfolio triggers HIGH/CRITICAL risk thresholds
- Measure how many additional SHAP-driven alerts would fire

### With `05_financing_cost_forecaster`
- Apply rate shocks to SOFR/repo-SOFR spread inputs
- Rerun LightGBM forecaster with shocked feature set
- Report stressed 1d/3d/5d financing cost vs. budget delta

### With `securities_lending_optimizer`
- Stress the recall probability inputs (e.g., assume 3x historical recall rate under crisis)
- Stress demand forecast (assume demand drops 40% in liquidity squeeze)
- Rerun the multi-objective optimizer and compare capital charge delta

---

## Regulatory Alignment

This framework is designed to support:

- **ESMA SFTR (Securities Financing Transactions Regulation)** — stress testing of collateral transformation and concentration limits
- **Basel IV / SA-CCR** — capital charge sensitivity to collateral value changes
- **FRTB Internal Models Approach (IMA)** — ES (Expected Shortfall) calculation on repo book
- **SEC Rule 15c3-3 (Customer Protection Rule)** — customer reserve formula sensitivity
- **FINRA Rule 4210** — margin requirement stress analysis

---

## Key Metrics Produced

| Metric | Definition |
|--------|-----------|
| `collateral_shortfall_usd` | Max(0, required_coverage - stressed_collateral_value) |
| `financing_cost_delta_bps` | Stressed financing cost − baseline cost in basis points |
| `capital_charge_delta_usd` | Stressed RWA contribution − baseline RWA |
| `margin_call_estimate_usd` | Estimated variation margin call under stress |
| `eligible_pool_shrinkage_pct` | % reduction in eligible collateral pool size |
| `recall_risk_multiplier` | Factor increase in expected recalls under stress |
| `pnl_impact_usd` | Total estimated P&L impact of stress scenario |
| `days_to_cover_delta` | Change in days-to-cover for shorted securities |
| `hhi_delta` | Change in concentration HHI under stressed book |
