# PowerBI Dataset Schema & DAX Guide
## FRED Macro Intelligence — FSIP Integration

---

## Dataset Tables

### Table 1: `MacroRegimeSignals`

Pushed daily. One row per trading day.

| Column | Type | Description |
|--------|------|-------------|
| `date` | DateTime | Trading date |
| `regime_id` | Int64 | Regime cluster ID (0-4) |
| `regime_label` | Text | Human-readable regime name |
| `funding_stress_index` | Decimal | Composite stress score (higher = more stress) |
| `collateral_quality_index` | Decimal | Composite quality score (higher = better quality) |
| `macro_momentum` | Decimal | 5-day momentum across rate features |
| `curve_10y2y` | Decimal | 10Y-2Y Treasury spread (bps) |
| `curve_inverted` | True/False | True when 10Y-2Y < 0 |
| `vix` | Decimal | VIX closing level |
| `hy_oas` | Decimal | HY Option-Adjusted Spread (bps) |
| `sofr` | Decimal | SOFR rate (%) |
| `rrp_balance_bn` | Decimal | Fed RRP facility balance ($bn) |
| `ensemble_conflict` | Int64 | 1 if HMM and KMeans disagree on regime |
| `run_id` | Text | Pipeline run identifier (UUID) |

---

### Table 2: `SpreadForecasts`

Pushed daily. Multiple rows per day (one per target × horizon combination).

| Column | Type | Description |
|--------|------|-------------|
| `date` | DateTime | Forecast as-of date |
| `target` | Text | Forecast target (sofr, hy_oas, ig_oas) |
| `horizon_days` | Int64 | Forecast horizon (1, 3, 5, 10) |
| `forecast_date` | DateTime | Date being forecasted (date + horizon) |
| `lgbm_forecast` | Decimal | LightGBM point forecast |
| `lstm_mean` | Decimal | LSTM mean forecast |
| `lstm_lower_90` | Decimal | LSTM 5th percentile (lower 90% CI bound) |
| `lstm_upper_90` | Decimal | LSTM 95th percentile (upper 90% CI bound) |
| `regime_label` | Text | Current macro regime at time of forecast |
| `run_id` | Text | Pipeline run identifier |

---

### Table 3: `AgencyDemandForecast`

Pushed daily. One row per day with top SHAP drivers.

| Column | Type | Description |
|--------|------|-------------|
| `date` | DateTime | Forecast date |
| `demand_signal` | Decimal | Predicted agency demand (normalized change) |
| `demand_direction` | Text | "Increasing" / "Stable" / "Decreasing" |
| `shap_driver_1` | Text | Top SHAP feature name |
| `shap_value_1` | Decimal | SHAP value for top feature |
| `shap_driver_2` | Text | Second SHAP feature name |
| `shap_value_2` | Decimal | SHAP value for second feature |
| `shap_driver_3` | Text | Third SHAP feature name |
| `shap_value_3` | Decimal | SHAP value for third feature |
| `regime_label` | Text | Current macro regime |
| `run_id` | Text | Pipeline run identifier |

---

### Table 4: `YieldCurveSnapshot`

Pushed daily. One row per trading day with full curve.

| Column | Type | Description |
|--------|------|-------------|
| `date` | DateTime | Snapshot date |
| `t_1mo` | Decimal | 1-month yield (%) |
| `t_3mo` | Decimal | 3-month yield (%) |
| `t_6mo` | Decimal | 6-month yield (%) |
| `t_1yr` | Decimal | 1-year yield (%) |
| `t_2yr` | Decimal | 2-year yield (%) |
| `t_5yr` | Decimal | 5-year yield (%) |
| `t_10yr` | Decimal | 10-year yield (%) |
| `t_30yr` | Decimal | 30-year yield (%) |
| `curve_slope_10y2y` | Decimal | 10Y-2Y spread |
| `curve_butterfly` | Decimal | Butterfly curvature |
| `run_id` | Text | Pipeline run identifier |

---

## Recommended DAX Measures

### Current Regime
```dax
Current Regime =
CALCULATE(
    LASTNONBLANKVALUE(MacroRegimeSignals[date], MacroRegimeSignals[regime_label]),
    DATESINPERIOD(MacroRegimeSignals[date], MAX(MacroRegimeSignals[date]), -1, DAY)
)
```

### Funding Stress Level
```dax
Funding Stress Level =
SWITCH(
    TRUE(),
    CALCULATE(MAX(MacroRegimeSignals[funding_stress_index])) >= 2.0, "CRITICAL",
    CALCULATE(MAX(MacroRegimeSignals[funding_stress_index])) >= 1.0, "ELEVATED",
    CALCULATE(MAX(MacroRegimeSignals[funding_stress_index])) >= 0.0, "NORMAL",
    "LOW"
)
```

### Curve Inversion Duration
```dax
Days Inverted =
CALCULATE(
    COUNTROWS(MacroRegimeSignals),
    MacroRegimeSignals[curve_inverted] = TRUE(),
    DATESINPERIOD(MacroRegimeSignals[date], MAX(MacroRegimeSignals[date]), -365, DAY)
)
```

### SOFR Forecast vs Current
```dax
SOFR 5D Forecast Delta (bps) =
VAR latest_sofr = CALCULATE(MAX(MacroRegimeSignals[sofr]))
VAR forecast_sofr = CALCULATE(
    MAX(SpreadForecasts[lgbm_forecast]),
    SpreadForecasts[target] = "sofr",
    SpreadForecasts[horizon_days] = 5,
    DATESINPERIOD(SpreadForecasts[date], MAX(SpreadForecasts[date]), -1, DAY)
)
RETURN (forecast_sofr - latest_sofr) * 100
```

### HY Spread Z-Score Alert
```dax
HY Spread Alert Flag =
IF(
    CALCULATE(MAX(MacroRegimeSignals[hy_oas])) >
    CALCULATE(
        AVERAGE(MacroRegimeSignals[hy_oas]) + 2 * STDEV.P(MacroRegimeSignals[hy_oas]),
        DATESINPERIOD(MacroRegimeSignals[date], MAX(MacroRegimeSignals[date]), -252, DAY)
    ),
    "SPIKE",
    "NORMAL"
)
```

---

## Recommended Report Pages

### Page 1: Macro Overview
- Card visuals: Current Regime, Funding Stress Level, VIX, SOFR
- Line chart: SOFR + Fed Funds + Prime Rate (3yr history)
- Area chart: Regime timeline with color bands (custom visual or conditional formatting)
- KPI: Days since inversion (if inverted)

### Page 2: Yield Curve & Spreads
- Custom visual or R/Python visual: animated yield curve
- Scatter chart: HY OAS vs IG OAS (regime-colored dots)
- Line chart: TED Spread + RRP balance (dual axis)
- Table: Spread forecast for all targets × horizons

### Page 3: Agency/Prime Demand Signal
- Line chart: Agency demand signal (90d history + 5d forecast)
- Bar chart: SHAP drivers (horizontal, color-coded by direction)
- Table: Recent alerts with severity color
- Conditional column: regime label with background color matching regime

### Page 4: Stress & Risk
- Multi-row card: Funding Stress Index, Collateral Quality Index, Macro Momentum
- Scatter plot: VIX vs HY OAS (time as color gradient)
- Heatmap table (matrix visual): Regime transition probabilities
- Line chart: Rolling 60-day correlation SOFR ↔ HY OAS

---

## Slicer Recommendations

- `date` range slicer (relative: last 3M / 6M / 1Y / 2Y)
- `regime_label` multi-select (filter charts to specific regimes)
- `target` dropdown (filter SpreadForecasts table)
- `horizon_days` radio button (1d / 3d / 5d / 10d)
