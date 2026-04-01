# What-If Analysis — Methodology

## 1. Purpose & Scope

What-if analysis allows the desk to interactively probe the sensitivity of:
- Collateral valuations and haircut eligibility
- Financing costs and repo rates
- Portfolio concentration risk scores
- Capital charge estimates
- Short squeeze probability scores

to user-defined parameter changes before committing to an allocation decision.

Unlike batch stress testing (which runs predefined regulatory or historical scenarios), what-if analysis is **interactive and ad hoc** — a trader or risk manager can adjust a slider and immediately see the impact on P&L and collateral coverage.

---

## 2. What-If Dimensions

### 2.1 Interest Rate Shocks

**Parallel shift** — move the entire yield curve up or down by Δbps:
```
rate_shocked[t] = rate_base[t] + Δbps / 10000
```

**Twist** — steepen or flatten the curve:
```
rate_shocked[t] = rate_base[t] + α * duration_sensitivity[t] + β
```

**Impact on collateral value** (duration-based approximation):
```
ΔP ≈ -ModDuration * ΔRate * P_base
```

**Impact on financing cost** (repo rate sensitivity):
```
financing_cost_stressed = repo_rate_base + SOFR_shock * repo_beta
```

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `rate_shift_bps` | -300 to +400 | 0 |
| `short_end_shift_bps` | -300 to +400 | 0 |
| `long_end_shift_bps` | -300 to +400 | 0 |
| `sofr_shock_bps` | -200 to +300 | 0 |

---

### 2.2 Credit Spread Shocks

Widens or tightens credit spreads by sector, rating, or individually:

```
spread_stressed[i] = spread_base[i] * spread_multiplier[sector_i]
ΔP[i] ≈ -SpreadDuration[i] * Δspread[i] * P_base[i]
```

**Haircut impact** — quality score re-scoring after spread shock:
```
quality_score_stressed = f(spread_stressed, duration, rating_after_shock)
```

Bonds that breach haircut eligibility thresholds are removed from the eligible pool.

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `ig_spread_shock_bps` | 0 to +500 | 0 |
| `hy_spread_shock_bps` | 0 to +2000 | 0 |
| `sector_multiplier` | dict by GICS | 1.0 |
| `downgrade_trigger_spread` | bps threshold | 300 |

---

### 2.3 Equity Price Shocks

Applies a uniform or name-specific equity price decline:

```
price_stressed[i] = price_base[i] * (1 + equity_shock_pct)
collateral_value_stressed[i] = price_stressed[i] * shares[i] * (1 - haircut[i])
```

**Short squeeze impact** (feeds into `04_short_squeeze_scorer`):
- Large price spike (positive equity shock) increases short squeeze probability
- Short squeeze scorer re-evaluated with shocked price momentum features

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `equity_shock_pct` | -50% to +30% | 0% |
| `vol_shock_multiplier` | 1.0 to 5.0 | 1.0 |
| `name_specific_shocks` | dict[ticker, pct] | {} |

---

### 2.4 Liquidity Shocks

Models a market liquidity squeeze — wider bid-ask spreads and reduced market depth:

```
liquidation_cost[i] = base_spread[i] * liquidity_multiplier + market_impact[i]
effective_collateral_value[i] = market_value[i] - liquidation_cost[i]
```

**Inventory cover deterioration:**
```
inventory_cover_stressed = inventory / (demand_forecast * demand_surge_factor)
```

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `bid_ask_multiplier` | 1.0 to 10.0 | 1.0 |
| `market_depth_reduction_pct` | 0% to 90% | 0% |
| `demand_surge_factor` | 1.0 to 3.0 | 1.0 |

---

### 2.5 Counterparty Stress

Simulates loss of one or more counterparties (default, withdrawal, or regulatory action):

```
eligible_book = loan_book[~loan_book.counterparty_id.isin(stressed_counterparties)]
```

**Re-optimization** runs on the remaining eligible book:
- Concentration HHI recalculated on remaining counterparties
- Recall risk elevated for remaining counterparties (concentration spillover)
- Financing cost may increase if lost counterparty provided below-market rates

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `remove_counterparty_ids` | list of IDs | [] |
| `recall_risk_spillover_factor` | 1.0 to 3.0 | 1.0 |
| `replacement_cost_bps` | 0 to 50 | 0 |

---

### 2.6 Regulatory / Haircut Shocks

Models regulatory tightening — higher minimum haircuts, new concentration limits:

```
haircut_stressed[i] = max(haircut_base[i], haircut_floor_stressed)
eligible[i] = (haircut_stressed[i] <= haircut_limit) AND (quality_score[i] >= quality_floor)
```

**Key parameters:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `haircut_floor_increase_pp` | 0 to 20 | 0 |
| `new_concentration_limit_pct` | 10% to 50% | current |
| `exclude_asset_classes` | list | [] |

---

## 3. What-If Analysis Workflow

```
Step 1: Load base case snapshot (current book + market data)
Step 2: User specifies shock parameters (YAML or interactive UI)
Step 3: Shock engine applies shocks to relevant inputs
Step 4: Optimizer re-runs with stressed inputs
Step 5: Results compared to baseline (delta report)
Step 6: Alerts generated if any threshold breaches detected
Step 7: Results pushed to PowerBI for visualization
```

---

## 4. Interactive What-If (PowerBI / Jupyter)

### PowerBI Implementation
- Use PowerBI **What-If Parameters** (numeric range sliders) wired to DAX measures
- Example: `Rate Shock (bps)` slider → DAX recalculates collateral value in real-time
- DAX measures call the stressed value calculation formula

### Jupyter Implementation
- Use `ipywidgets` sliders for each shock parameter
- Callback reruns the optimizer and updates a matplotlib/plotly output cell
- Useful for exploratory analysis and model validation

### Example — Jupyter interactive widget:
```python
import ipywidgets as widgets
from IPython.display import display

rate_slider = widgets.IntSlider(min=-200, max=400, step=25, value=0,
                                 description='Rate Shock (bps)')
equity_slider = widgets.FloatSlider(min=-0.5, max=0.3, step=0.05, value=0.0,
                                     description='Equity Shock (%)')

def on_change(rate_shock, equity_shock):
    result = run_stress_scenario(rate_shock_bps=rate_shock,
                                  equity_shock_pct=equity_shock)
    display_results(result)

widgets.interact(on_change, rate_shock=rate_slider, equity_shock=equity_slider)
```

---

## 5. Sensitivity Analysis (Tornado Chart)

Run a one-at-a-time (OAT) sensitivity analysis to rank which inputs have the largest impact:

```python
# Perturb each parameter ±1 standard deviation and record output delta
sensitivities = []
for param in shock_params:
    result_up = run_stress(param=base_value + 1_sigma)
    result_dn = run_stress(param=base_value - 1_sigma)
    sensitivities.append({
        "param": param,
        "up_impact": result_up.pnl_impact - baseline.pnl_impact,
        "dn_impact": result_dn.pnl_impact - baseline.pnl_impact,
    })
```

Output: Tornado chart ranking parameters by |up_impact| + |dn_impact|.

---

## 6. Monte Carlo Simulation

For capturing joint distribution of shocks:
1. Sample N scenarios from a multivariate distribution (rates, spreads, equity) calibrated to historical covariance
2. Run the optimizer for each sample
3. Report the distribution of: collateral shortfall, P&L impact, capital charge
4. Derive VaR (95th/99th percentile) and Expected Shortfall (CVaR)

```python
# Simplified Monte Carlo loop
np.random.seed(42)
results = []
for _ in range(10_000):
    shocks = np.random.multivariate_normal(mean=mu, cov=sigma)
    scenario = build_scenario_from_shocks(shocks)
    result = run_stress_scenario(scenario)
    results.append(result.pnl_impact)

var_95 = np.percentile(results, 5)   # 5th percentile = worst 5%
es_95  = np.mean([r for r in results if r <= var_95])
```
