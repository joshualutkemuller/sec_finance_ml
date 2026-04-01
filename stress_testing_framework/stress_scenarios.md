# Stress Scenario Catalog

## Overview

This catalog defines all predefined stress scenarios for the FSIP platform. Each scenario is described by:
- **Narrative**: what market event it represents
- **Shock parameters**: quantitative inputs to the shock engine
- **Expected impacts**: which projects and metrics are most affected
- **Severity**: Moderate / Severe / Extreme

Scenarios are stored as YAML files in the `scenarios/` subdirectory and loaded by `scenario_loader.py`.

---

## Scenario Index

| ID | Name | Severity | YAML File |
|----|------|----------|-----------|
| S01 | Base Case | — | `base_case.yaml` |
| S02 | Rate Shock +200bps | Severe | `rate_shock_200bps.yaml` |
| S03 | Rate Shock +100bps | Moderate | `rate_shock_100bps.yaml` |
| S04 | Rate Twist (Flattener) | Moderate | `rate_twist_flattener.yaml` |
| S05 | Credit Crunch 2008 | Extreme | `credit_crunch_2008.yaml` |
| S06 | Equity Crash -20% | Severe | `equity_crash_20pct.yaml` |
| S07 | Equity Crash -35% | Extreme | `equity_crash_35pct.yaml` |
| S08 | Top-1 Counterparty Default | Severe | `counterparty_default.yaml` |
| S09 | Top-3 Counterparty Loss | Extreme | `counterparty_top3_loss.yaml` |
| S10 | Regulatory Haircut +5pp | Moderate | `regulatory_haircut_floor.yaml` |
| S11 | ESMA SFTR Adverse | Regulatory | `regulatory_stress_lseg.yaml` |
| S12 | Liquidity Squeeze | Extreme | `liquidity_squeeze.yaml` |
| S13 | Combined Stress | Extreme | `combined_stress.yaml` |
| S14 | Short Squeeze Cascade | Severe | `short_squeeze_cascade.yaml` |
| S15 | Reverse Stress | Extreme | `reverse_stress.yaml` |

---

## Scenario Definitions

---

### S01 — Base Case

**Narrative**: No shocks applied. Baseline optimizer output under current market conditions.

**Purpose**: Reference point for all other scenarios. Used to calibrate delta calculations.

**Shocks**: None.

**Key metrics**: All at baseline values.

---

### S02 — Rate Shock +200bps

**Narrative**: Instantaneous, parallel upward shift of the yield curve by 200 basis points. Replicates a rapid rate normalization shock (e.g., 1994 bond market crash, 2022 Fed hiking cycle).

**Shocks**:
- `rate_shift_bps: 200`
- `sofr_shock_bps: 200`

**Expected impacts**:
- Fixed-income collateral mark-to-market loss: ~8-12% for 5yr duration assets
- Financing cost increase: 200bps × repo_beta
- Haircut eligibility: longer-duration bonds may breach quality floor after repricing
- Capital charge increase due to collateral value decline

**Primary projects affected**: `02_collateral_optimizer`, `05_financing_cost_forecaster`, `securities_lending_optimizer`

**Key watch metrics**:
- `collateral_shortfall_usd`
- `financing_cost_delta_bps`
- `eligible_pool_shrinkage_pct`

---

### S03 — Rate Shock +100bps

**Narrative**: Moderate rate rise — single Fed hike of unusual size (e.g., 100bps surprise).

**Shocks**:
- `rate_shift_bps: 100`
- `sofr_shock_bps: 100`

**Expected impacts**: Moderate collateral value decline, contained financing cost increase.

---

### S04 — Rate Twist (Flattener)

**Narrative**: Short rates rise sharply while long rates fall — a bull flattener. Common in flight-to-quality episodes.

**Shocks**:
- `short_end_shift_bps: 100` (2yr and under)
- `long_end_shift_bps: -50` (10yr and over)
- `sofr_shock_bps: 75`

**Expected impacts**: Short-term repo costs increase; long-dated bond collateral gains value; curve-dependent strategies significantly impacted.

---

### S05 — Credit Crunch 2008

**Narrative**: Replicates the 2008-2009 global financial crisis shock profile. Severe credit spread widening combined with equity market crash, liquidity evaporation, and counterparty stress.

**Shocks**:
- `ig_spread_shock_bps: 400`
- `hy_spread_shock_bps: 1500`
- `equity_shock_pct: -0.38`
- `vol_shock_multiplier: 4.0`
- `bid_ask_multiplier: 8.0`
- `market_depth_reduction_pct: 0.65`
- `remove_counterparty_ids: [top_1_by_exposure]`

**Expected impacts**:
- Massive eligible collateral pool shrinkage (IG bonds repriced, HY disqualified)
- Financing cost spikes (LIBOR-OIS spread replication)
- Short squeeze risk elevated for heavily shorted names
- Potential breach of regulatory capital buffers

**This is the "DOOMSDAY" scenario** — used to test portfolio resilience under extreme tail risk.

---

### S06 — Equity Crash -20%

**Narrative**: Broad equity market decline of 20% (approx. March 2020 COVID shock, October 2002 dot-com bust).

**Shocks**:
- `equity_shock_pct: -0.20`
- `vol_shock_multiplier: 2.5`
- `ig_spread_shock_bps: 80`

**Expected impacts**:
- Equity collateral values decline 20%
- Short squeeze scores increase for heavily shorted names (price decline → covering pressure)
- Margin call risk elevated for equity-heavy counterparties
- Portfolio risk monitor may trigger CRITICAL alerts

---

### S07 — Equity Crash -35%

**Narrative**: Severe equity drawdown — tail risk scenario.

**Shocks**:
- `equity_shock_pct: -0.35`
- `vol_shock_multiplier: 4.0`
- `hy_spread_shock_bps: 500`

---

### S08 — Top-1 Counterparty Default

**Narrative**: The largest lending counterparty (by notional exposure) defaults or becomes non-operational overnight. Replicates Lehman Brothers-style event.

**Shocks**:
- `remove_counterparty_ids: [top_1_counterparty_by_notional]`
- `recall_risk_spillover_factor: 2.0` (other counterparties increase recall rates)
- `replacement_cost_bps: 20`

**Expected impacts**:
- Immediate book concentration increase in remaining counterparties
- Recall risk elevated across book (market-wide stress signal)
- Financing cost increases as replacement counterparties charge higher rates
- HHI concentration metric spikes → portfolio risk monitor triggers HIGH/CRITICAL

---

### S09 — Top-3 Counterparty Loss

**Narrative**: Simultaneous or rapid sequential loss of the three largest counterparties. Extreme systemic scenario.

**Shocks**:
- `remove_counterparty_ids: [top_3_counterparties_by_notional]`
- `recall_risk_spillover_factor: 3.0`
- `replacement_cost_bps: 40`

---

### S10 — Regulatory Haircut Floor +5pp

**Narrative**: Regulator mandates a 5 percentage point increase in minimum haircuts across all collateral. Replicates potential post-crisis regulatory tightening.

**Shocks**:
- `haircut_floor_increase_pp: 5`

**Expected impacts**:
- Eligible collateral pool value decreases (same assets, higher haircut)
- Some borderline-eligible assets become ineligible
- Optimizer must substitute to maintain collateral coverage
- Capital charge may increase

---

### S11 — ESMA SFTR Adverse Scenario

**Narrative**: Regulatory stress scenario aligned with ESMA's Securities Financing Transactions Regulation (SFTR) adverse scenario specifications.

**Shocks** (per ESMA guidelines):
- `ig_spread_shock_bps: 150`
- `equity_shock_pct: -0.15`
- `rate_shift_bps: 75`
- `haircut_floor_increase_pp: 2`
- `concentration_limit_reduction_pct: 10`

**Reporting**: Results must be included in SFTR regulatory reporting.

---

### S12 — Liquidity Squeeze

**Narrative**: Market liquidity evaporates — bid-ask spreads triple and market depth falls 70%. Replicates March 2020 US Treasury liquidity crisis or 1998 LTCM collapse.

**Shocks**:
- `bid_ask_multiplier: 7.0`
- `market_depth_reduction_pct: 0.70`
- `demand_surge_factor: 2.0`
- `sofr_shock_bps: 50`

**Expected impacts**:
- Effective collateral liquidation values fall 5-15% below mark-to-market
- Inventory cover days compress rapidly
- Short squeeze risk spikes for thinly-traded names
- Financing cost forecaster may under-estimate actual repo rates (model trained on liquid conditions)

---

### S13 — Combined Stress

**Narrative**: Multi-factor adverse scenario combining moderate versions of rate, credit, equity, and liquidity shocks.

**Shocks**:
- `rate_shift_bps: 150`
- `ig_spread_shock_bps: 200`
- `equity_shock_pct: -0.15`
- `vol_shock_multiplier: 2.0`
- `bid_ask_multiplier: 3.0`
- `sofr_shock_bps: 100`

**Purpose**: Tests portfolio resilience under realistic but not catastrophic combined adverse conditions.

---

### S14 — Short Squeeze Cascade

**Narrative**: A chain of short squeezes across heavily-shorted names causes forced covering, rapid price spikes, and elevated borrow rates. Replicates GameStop/AMC-style event at portfolio scale.

**Shocks**:
- `equity_shock_pct: +0.30` (for shorted names only)
- `borrow_rate_shock_multiplier: 5.0` (for names with SI > 20%)
- `recall_risk_spillover_factor: 2.5`
- `demand_surge_factor: 3.0` (borrowing demand surges)

**Expected impacts**:
- Borrow rate anomaly detector fires on multiple names simultaneously
- Short squeeze scorer triggers CRITICAL alerts across the book
- Collateral substitution may be forced as high-cost borrows consume fee budget

---

### S15 — Reverse Stress Test

**Narrative**: Instead of applying predefined shocks, find the minimum shock that causes a specific loss threshold to be breached. Works backwards from the outcome.

**Target**: `collateral_shortfall_usd > 50_000_000` OR `financing_cost_delta_bps > 100`

**Method**: Binary search / gradient-based optimization over shock parameters to find the minimum shock magnitude that triggers the target.

**Purpose**: Identifies which risks are closest to threshold breaches — most useful for risk appetite calibration and limit-setting.

**Implementation**: See `src/stress_optimizer.py → reverse_stress()` method.

---

## Adding Custom Scenarios

Create a YAML file in `scenarios/` following this template:

```yaml
scenario_id: CUSTOM_01
name: My Custom Scenario
description: Describe what market event this represents
severity: Moderate  # Moderate | Severe | Extreme | Regulatory

shocks:
  rate_shift_bps: 0
  sofr_shock_bps: 0
  short_end_shift_bps: 0
  long_end_shift_bps: 0
  ig_spread_shock_bps: 0
  hy_spread_shock_bps: 0
  sector_spread_shocks: {}       # {"Financials": 100, "Energy": 200}
  equity_shock_pct: 0.0
  name_specific_equity_shocks: {}  # {"AAPL": -0.10, "TSLA": -0.20}
  vol_shock_multiplier: 1.0
  bid_ask_multiplier: 1.0
  market_depth_reduction_pct: 0.0
  demand_surge_factor: 1.0
  remove_counterparty_ids: []
  recall_risk_spillover_factor: 1.0
  replacement_cost_bps: 0
  haircut_floor_increase_pp: 0
  new_concentration_limit_pct: null
  exclude_asset_classes: []

reporting:
  push_to_powerbi: true
  generate_pdf: false
  alert_on_breach: true
  breach_thresholds:
    collateral_shortfall_usd: 10000000
    financing_cost_delta_bps: 50
    eligible_pool_shrinkage_pct: 0.15
```
