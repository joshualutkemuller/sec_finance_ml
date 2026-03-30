# Portfolio Concentration Risk Monitor — Process Map

```
╔══════════════════════════════════════════════════════════════════════════╗
║              PORTFOLIO CONCENTRATION RISK PIPELINE                      ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 1 — DATA INGESTION                                            │
  │                                                                     │
  │  Sources:                                                           │
  │    • positions_path    → portfolio holdings, quantities, ISIN       │
  │    • market_data_path  → prices, market values, bid-ask spreads     │
  │    • counterparty_path → counterparty exposures, LEI, credit lines  │
  │                                                                     │
  │  Output: raw DataFrames (positions, market_data, counterparties)    │
  │  Class:  PortfolioDataLoader.build_exposure_snapshot()              │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  exposure_snapshot (joined DataFrame)
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 2 — CONCENTRATION METRIC CALCULATION                         │
  │                                                                     │
  │  Metrics computed:                                                  │
  │    • HHI (Herfindahl-Hirschman Index) per portfolio / sector /      │
  │      counterparty dimension                                         │
  │    • Top-5 exposure % of total AUM                                  │
  │    • Sector weight % per GICS sector                                │
  │    • Counterparty notional exposure % of total book                 │
  │                                                                     │
  │  Thresholds read from: config.concentration.*                       │
  │  Class:  ConcentrationFeatureEngineer                               │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  raw concentration metrics
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 3 — FEATURE ENGINEERING                                      │
  │                                                                     │
  │  Additional features:                                               │
  │    • Rolling N-day change in HHI and top-N exposure                 │
  │    • VaR proxy: position-weighted volatility * z-score              │
  │    • Liquidity-adjusted exposure: market_value / bid_ask_spread     │
  │    • Sector momentum: rolling sector weight delta                   │
  │    • Counterparty credit headroom: exposure / credit_limit          │
  │                                                                     │
  │  Window: config.data.lookback_days (default 5)                      │
  │  Class:  ConcentrationFeatureEngineer.build_features()              │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  feature matrix  X  [n_rows × n_features]
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 4 — RANDOM FOREST SCORING                                    │
  │                                                                     │
  │  Model: RandomForestClassifier (scikit-learn)                       │
  │    • Trained on labelled historical risk events                     │
  │    • Outputs P(HIGH | CRITICAL risk) per row                        │
  │    • Hyper-params: n_estimators, max_depth, min_samples_leaf        │
  │                                                                     │
  │  Class:  RandomForestRiskScorer.score()                             │
  │  Output: scores array  [0.0 – 1.0]                                  │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  risk score per dimension
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 5 — SHAP EXPLAINABILITY                                      │
  │                                                                     │
  │  For every scored row:                                              │
  │    • Compute SHAP TreeExplainer values                              │
  │    • Rank features by |SHAP value|                                  │
  │    • Return top-N drivers (configurable via config.shap.top_n)      │
  │    • Optional: generate waterfall / beeswarm plots to disk          │
  │                                                                     │
  │  Class:  RandomForestRiskScorer.explain()                           │
  │  Output: SHAP DataFrame  [n_rows × top_n_features]                  │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  scores + SHAP drivers
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 6 — RISK LEVEL CLASSIFICATION                                │
  │                                                                     │
  │  Thresholds (from config.model):                                    │
  │    score < threshold_medium              → LOW                      │
  │    threshold_medium  <= score < high     → MEDIUM                   │
  │    threshold_high    <= score < critical → HIGH      ← alert        │
  │    score >= threshold_critical           → CRITICAL  ← alert        │
  │                                                                     │
  │  Class:  RandomForestRiskScorer.classify()                          │
  │  Enum:   RiskLevel  {LOW, MEDIUM, HIGH, CRITICAL}                   │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │  RiskLevel per row
                          ┌─────┴──────┐
                          │            │
                          ▼            ▼
  ┌───────────────────┐   ┌──────────────────────────────────────────┐
  │  LOW / MEDIUM     │   │  STEP 7 — ALERT DISPATCH (HIGH/CRITICAL) │
  │  → log only       │   │                                          │
  │  → PowerBI push   │   │  Channels:                               │
  └───────────────────┘   │    • Email   (SMTP via smtplib)          │
                          │    • Teams   (Incoming Webhook HTTP POST) │
                          │    • Log     (structured JSONL file)      │
                          │                                          │
                          │  Class:  RiskAlertDispatcher             │
                          │  Factory: RiskAlertFactory.from_scores() │
                          └─────────────┬────────────────────────────┘
                                        │  dispatched alerts
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 8 — POWERBI PUSH                                             │
  │                                                                     │
  │  All rows (all risk levels) are pushed to PowerBI streaming dataset │
  │                                                                     │
  │  Payload schema:                                                    │
  │    portfolio_id, sector, counterparty, concentration_score,         │
  │    risk_level, alert_flag, timestamp                                │
  │                                                                     │
  │  Class:  PowerBIConnector.push_rows()                               │
  │  Config: config.powerbi.streaming_url / dataset_id                  │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 9 — AUDIT LOG                                                │
  │                                                                     │
  │  Written to: config.alerts.log_path  (JSONL, one entry per run)    │
  │                                                                     │
  │  Fields:                                                            │
  │    run_id, run_timestamp, rows_scored, alerts_fired,               │
  │    powerbi_rows_pushed, model_version, pipeline_duration_s          │
  └─────────────────────────────────────────────────────────────────────┘
```
