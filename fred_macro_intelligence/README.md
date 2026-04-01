# FRED Macro Intelligence — Securities Finance & Agency/Prime Lending

## Overview

This module ingests **Federal Reserve Economic Data (FRED)** and applies machine learning to extract actionable signals for securities finance operations — specifically agency lending, prime brokerage, and repo/collateral management desks.

The pipeline is built on top of the existing vetted FRED data feed and answers the core question:

> *"What is the macro environment telling us about near-term securities lending conditions, rate regimes, and spread dynamics — and how should we adapt our book?"*

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FRED Data Pipeline (vetted)                        │
│  SOFR │ Fed Funds │ Treasury Curve │ Spreads │ Prime Rate │ VIX │ M2  │
└───────────────────────┬──────────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────────────────────┐
          │         Feature Engineering                │
          │  Rate factors │ Curve shape │ Spread regime│
          │  Momentum │ Calendar │ Macro composites    │
          └─────────────┬─────────────────────────────┘
                        │
     ┌──────────────────┼──────────────────────────────┐
     │                  │                              │
     ▼                  ▼                              ▼
┌──────────┐    ┌───────────────┐           ┌──────────────────┐
│  Regime  │    │ Spread/Rate   │           │  Agency Lending  │
│ Detector │    │  Forecaster   │           │ Demand Predictor │
│  (HMM +  │    │ (LightGBM +   │           │  (XGBoost +      │
│ KMeans)  │    │  LSTM)        │           │   SHAP)          │
└────┬─────┘    └──────┬────────┘           └────────┬─────────┘
     │                 │                             │
     └────────┬────────┘─────────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────┐
   │              Visualization Layer                  │
   │  Plotly Dashboards │ PowerBI │ Jupyter Widgets   │
   │  Regime Heatmaps │ Yield Curve Animation         │
   │  Rolling Correlation Matrix │ Tornado Charts     │
   └──────────────────────────────────────────────────┘
```

---

## Folder Structure

```
fred_macro_intelligence/
├── README.md                          ← this file
├── requirements.txt
├── config/
│   └── config.yaml                    ← FRED series IDs, model params, display settings
├── src/
│   ├── main.py                        ← CLI orchestration
│   ├── fred_loader.py                 ← FRED API + local cache loader
│   ├── feature_engineering.py         ← macro feature construction
│   ├── regime_detector.py             ← HMM + KMeans rate regime labeling
│   ├── spread_forecaster.py           ← LightGBM + LSTM spread/rate forecaster
│   ├── agency_demand_predictor.py     ← XGBoost agency/prime lending demand model
│   ├── visualization.py               ← Plotly chart library (10+ chart types)
│   ├── powerbi_connector.py           ← push macro signals to PowerBI
│   └── alerts.py                      ← macro regime-change alerts
├── dashboards/
│   ├── macro_dashboard.py             ← Plotly Dash application (standalone)
│   └── powerbi_schema.md              ← PowerBI dataset schema + DAX examples
└── notebooks/
    └── macro_eda.ipynb                ← Exploratory analysis template
```

---

## FRED Series Used

### Rate & Financing Benchmarks
| FRED ID | Description | Relevance |
|---------|-------------|-----------|
| `SOFR` | Secured Overnight Financing Rate | Primary repo benchmark |
| `FEDFUNDS` | Effective Federal Funds Rate | Policy rate baseline |
| `DFF` | Daily Fed Funds Rate | Intraday funding cost |
| `IOER` / `IORB` | Interest on Reserve Balances | Floor for GC repo |
| `DPRIME` | Bank Prime Loan Rate | Prime brokerage cost-of-funds |
| `DPCREDIT` | Discount Window Primary Rate | Stress funding signal |

### Treasury Yield Curve
| FRED ID | Description | Relevance |
|---------|-------------|-----------|
| `DGS1MO` | 1-Month Treasury | Short-end repo anchor |
| `DGS3MO` | 3-Month Treasury | Term repo pricing |
| `DGS6MO` | 6-Month Treasury | Semi-annual financing |
| `DGS1` | 1-Year Treasury | Short-duration collateral |
| `DGS2` | 2-Year Treasury | Key agency collateral tenor |
| `DGS5` | 5-Year Treasury | Mid-curve collateral |
| `DGS10` | 10-Year Treasury | Long-end agency collateral |
| `DGS30` | 30-Year Treasury | Long-duration MBS proxy |

### Spread & Credit Signals
| FRED ID | Description | Relevance |
|---------|-------------|-----------|
| `T10Y2Y` | 10Y-2Y Treasury Spread | Yield curve slope |
| `T10Y3M` | 10Y-3M Treasury Spread | Recession signal |
| `TEDRATE` | TED Spread (T-bill vs LIBOR) | Counterparty stress proxy |
| `BAMLH0A0HYM2` | HY Option-Adjusted Spread | Risk-off / credit stress |
| `BAMLC0A0CM` | IG OAS | IG credit environment |
| `MORTGAGE30US` | 30-Year Mortgage Rate | MBS collateral spread proxy |

### Liquidity & Market Conditions
| FRED ID | Description | Relevance |
|---------|-------------|-----------|
| `VIXCLS` | CBOE VIX | Market stress / collateral volatility |
| `M2SL` | M2 Money Supply | Liquidity environment |
| `WRESBAL` | Reserve Balances with Fed | Banking system liquidity |
| `RRPONTSYD` | Overnight Reverse Repo | RRP facility usage (excess liquidity) |
| `WSHOSHO` | Fed Balance Sheet (securities held) | QE/QT signal |

### Agency & Mortgage-Specific
| FRED ID | Description | Relevance |
|---------|-------------|-----------|
| `MORTGAGE15US` | 15-Year Mortgage Rate | Agency MBS pricing |
| `MSPNHSUS` | Median House Price | MBS underlying collateral |
| `DMORTGAGE30US` | 30Y Mortgage - 10Y Treasury Spread | MBS basis |
| `OBMMIFHA30YF` | FHA 30Y Fixed Rate | Agency lending benchmark |

---

## ML Components

### 1. Rate Regime Detector (`regime_detector.py`)

Labels each trading day as belonging to one of 4-5 macro rate regimes using unsupervised and semi-supervised methods:

**Method A — Hidden Markov Model (HMM):**
```
State space: 4 latent regimes
Observations: [sofr_level, curve_slope, hy_spread, vix, rrp_balance]
Emission model: Gaussian mixture per state
Transition: learned from historical data
```

**Method B — KMeans Clustering (interpretable labels):**
```
Features: PCA(5) of rate/spread/liquidity factors
K = 5 clusters → labeled by analyst:
  Regime 0: "Easing / QE"      — low rates, steep curve, tight spreads
  Regime 1: "Neutral / Normal"  — mid rates, moderate slope
  Regime 2: "Hiking Cycle"      — rising rates, flattening curve
  Regime 3: "Stress / Crisis"   — spread widening, VIX spike, inverted curve
  Regime 4: "Late Cycle"        — inverted curve, high rates, credit tightening
```

**Securities Finance Implications per Regime:**

| Regime | Borrow Rate Direction | Collateral Quality Bias | Recall Risk | Recommended Action |
|--------|----------------------|-------------------------|-------------|-------------------|
| Easing | Declining | Equity-friendly | Low | Extend loan terms, accumulate equity collateral |
| Normal | Stable | Balanced | Normal | Hold current allocations |
| Hiking | Rising | Fixed-income pressure | Elevated | Shorten duration, increase cash collateral % |
| Stress | Spiking (specials) | IG-only | Very high | Tighten concentration limits, increase haircuts |
| Late Cycle | Elevated | Treasury-only | High | Reduce HY exposure, maximize liquidity buffer |

---

### 2. Spread & Rate Forecaster (`spread_forecaster.py`)

**Dual-model approach:**

**LightGBM (1d/3d/5d horizons):**
- Features: full macro feature set (see below)
- Targets: next-day SOFR, repo-SOFR spread, IG OAS, HY OAS
- Loss: asymmetric quantile (α=0.7 — penalize under-estimates for risk management)
- Output feeds directly into `05_financing_cost_forecaster` as macro overlay

**LSTM (5d/10d horizon — trend capture):**
- Sequence length: 60 trading days
- Input: normalized macro feature matrix
- Output: multi-step forecast with uncertainty bands (MC Dropout, N=100 passes)
- Uncertainty bands shown as shaded confidence intervals in visualizations

---

### 3. Agency/Prime Lending Demand Predictor (`agency_demand_predictor.py`)

Predicts which macro environments drive increased/decreased demand for agency securities lending and prime brokerage activity:

- **Target**: Agency lending utilization rate (rolling 5d change)
- **Model**: XGBoost regressor + SHAP explainability
- **Key signal hypothesis**: High RRP balance → excess liquidity → reduced repo demand. Low RRP → tighter funding → increased borrow demand
- **Additional signals**: Curve inversion depth, Fed meeting proximity, quarter-end flag, mortgage prepayment speed

---

## Feature Engineering

Full macro feature set constructed in `feature_engineering.py`:

### Rate Level Features
- SOFR level, 5/10/20/60-day rolling mean and std
- Fed Funds - SOFR spread (policy transmission gap)
- Prime Rate - SOFR spread (prime brokerage margin proxy)
- All-in financing rate = SOFR + repo-SOFR spread estimate

### Yield Curve Shape Features
- Slope: `10Y - 2Y`, `10Y - 3M`, `5Y - 2Y`
- Curvature: `2*(5Y) - (2Y + 10Y)` (butterfly)
- Level: `(2Y + 5Y + 10Y) / 3`
- Inversion flag: `(10Y - 2Y) < 0`
- Days-since-inversion (rolling count)
- Curve momentum: 5/10/20-day change in slope

### Spread Features
- HY OAS, IG OAS, HY-IG differential
- TED Spread (counterparty stress indicator)
- MBS basis: `30Y Mortgage - 10Y Treasury`
- Spread z-scores over 60/120-day windows
- Spread momentum (5/10-day change)

### Liquidity Features
- RRP balance level and 5/10/20-day change
- Reserve balances level and rate-of-change
- M2 growth (YoY and MoM)
- Fed balance sheet size (QE/QT proxy)
- VIX level, 20-day z-score, VIX term structure slope

### Calendar Features
- Day of week (Mon-Fri encoding)
- Month-end flag (last 3 business days)
- Quarter-end flag (last 5 business days)
- Fed meeting flag (FOMC meeting day ± 1)
- Year-end flag (last 10 business days)
- Holiday proximity flag

### Derived Composite Signals
- **Funding Stress Index**: weighted composite of TED spread, VIX z-score, HY OAS z-score, RRP change
- **Collateral Quality Index**: IG OAS z-score inverted + Treasury curve level + VIX inverted
- **Macro Momentum Score**: PCA first component of 5-day changes across all rate features
- **Yield Curve Pressure**: sigmoid-scaled curve slope with inversion emphasis

---

## Visualization Library

Ten production-ready chart types in `visualization.py` (Plotly-based, PowerBI-compatible):

### Chart 1 — Rate Regime Timeline
Horizontal bar chart with color-coded regime bands overlaid on SOFR/Fed Funds history. Hover shows regime label, date range, avg borrow rate in that regime.

### Chart 2 — Yield Curve Animation
Animated Plotly line chart showing the yield curve morphing over time. Slider controls date. Key dates (Fed hikes, inversions, crises) marked as annotations.

### Chart 3 — Rolling Macro Correlation Matrix
Heatmap of rolling 60-day correlations between: SOFR, HY OAS, VIX, Curve Slope, RRP, Lending Utilization. Color scale diverging (red = negative, blue = positive). Updates daily.

### Chart 4 — Spread Forecast with Uncertainty Bands
Multi-line chart: historical SOFR (solid), 1d/3d/5d forecast (dashed), LSTM 90% confidence band (shaded). Regime color shown as background strip.

### Chart 5 — Funding Stress Dashboard (4-panel)
2×2 subplot grid:
- TED Spread history with z-score overlay
- RRP balance vs. SOFR spread (scatter + time color)
- VIX vs. HY OAS (regime-colored scatter)
- Reserve balance YoY change (bar)

### Chart 6 — Prime/Agency Demand Heatmap
Calendar heatmap (GitHub-style) where each cell = a trading day, color = agency lending utilization. Annotations on macro events. Filterable by year/quarter.

### Chart 7 — Macro Factor SHAP Waterfall
For each daily prediction, show the top-N SHAP feature contributions in a waterfall chart. Answers: *"Why did the model predict higher lending demand today?"*

### Chart 8 — Curve Shape PCA Biplot
PCA of the yield curve into 3 components (Level, Slope, Curvature). 3D scatter plot with regime color. Trajectory line shows path through factor space over time.

### Chart 9 — Sector Spread Tornado (What-If)
Horizontal tornado chart: if each macro input moves ±1σ, what is the estimated change in borrow cost? Bars sorted by sensitivity magnitude. Interactive — click bar to see time series.

### Chart 10 — Macro Regime Transition Matrix
Heatmap showing historical transition probabilities between regimes (from HMM). Row = current regime, column = next-day regime. Annotated with avg days spent per regime.

---

## PowerBI Integration

See `dashboards/powerbi_schema.md` for:
- Dataset table schemas for each signal type
- Recommended DAX measures for macro KPIs
- Suggested report page layouts for agency lending desk
- Drill-through from macro regime tile to securities-level impact

**Push tables:**
- `MacroRegimeSignals` — daily regime label, transition probability, composite indices
- `SpreadForecasts` — 1d/3d/5d SOFR, IG OAS, HY OAS forecasts + confidence intervals
- `AgencyDemandForecast` — predicted agency utilization + top SHAP drivers
- `YieldCurveSnapshot` — full curve levels daily for animated curve visual

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set FRED API key
export FRED_API_KEY=your_fred_api_key_here   # free at fred.stlouisfed.org

# Fetch latest FRED data, run all models, push to PowerBI
python src/main.py --config config/config.yaml --mode full

# Refresh data only
python src/main.py --config config/config.yaml --mode fetch

# Retrain models only (weekly cadence recommended)
python src/main.py --config config/config.yaml --mode train

# Launch interactive Dash dashboard (local)
python dashboards/macro_dashboard.py

# Run Jupyter EDA notebook
jupyter notebook notebooks/macro_eda.ipynb
```

---

## Scheduling Recommendations

| Job | Frequency | Mode | Notes |
|-----|-----------|------|-------|
| FRED data refresh | Daily 6:00 AM ET | `fetch` | After FRED daily update |
| Model scoring | Daily 7:00 AM ET | `score` | After data refresh |
| Model retraining | Weekly Saturday | `train` | Retrain on rolling 5yr window |
| Full pipeline | Daily 7:15 AM ET | `full` | Fetch + score + push + alert |
| Dashboard refresh | Every 4 hours | — | PowerBI auto-refresh |
