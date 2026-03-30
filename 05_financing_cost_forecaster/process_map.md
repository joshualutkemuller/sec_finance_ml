# Repo & Financing Cost Forecaster — Process Map

```
+==============================================================================+
|           REPO & FINANCING COST FORECASTER — END-TO-END PROCESS             |
+==============================================================================+

  STEP 1: DATA INGESTION
  +----------------------------------------------------------------------+
  |  Sources loaded by FinancingDataLoader                               |
  |                                                                      |
  |  [repo_rates.csv]   --> date, instrument_type, rate_bps, tenor       |
  |  [macro_rates.csv]  --> date, sofr, fed_funds, ois, libor_3m        |
  |  [positions.csv]    --> date, instrument_type, notional              |
  |  [budget.csv]       --> instrument_type, budget_rate_bps             |
  |                                                                      |
  |  All sources merged on (date x instrument_type) -> master DataFrame  |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 2: LAG FEATURE ENGINEERING
  +----------------------------------------------------------------------+
  |  FinancingCostFeatureEngineer.add_lag_features()                     |
  |                                                                      |
  |  Lag windows: [1d, 3d, 5d, 10d, 20d]                                |
  |  Applied to: rate_bps, sofr_spread, fed_funds, notional              |
  |                                                                      |
  |  rate_bps_lag_1, rate_bps_lag_3, rate_bps_lag_5 ...                 |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 3: CALENDAR FEATURES
  +----------------------------------------------------------------------+
  |  FinancingCostFeatureEngineer.add_calendar_features()                |
  |                                                                      |
  |  day_of_week        (0=Mon ... 4=Fri)                                |
  |  is_month_end       (flag: 1 if last 3 business days of month)       |
  |  is_quarter_end     (flag: 1 if last 3 business days of quarter)     |
  |  days_to_month_end  (integer countdown)                              |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 4: MACRO FEATURE INTEGRATION
  +----------------------------------------------------------------------+
  |  FinancingCostFeatureEngineer.add_macro_spread()                     |
  |  FinancingCostFeatureEngineer.add_term_structure_features()          |
  |                                                                      |
  |  repo_sofr_spread   = rate_bps - sofr * 100                         |
  |  curve_slope        = (10y_proxy - 2y_proxy) in bps                 |
  |  ois_basis          = sofr - ois                                     |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 5: MULTI-HORIZON LIGHTGBM TRAINING
  +----------------------------------------------------------------------+
  |  MultiHorizonForecaster.fit()                                        |
  |                                                                      |
  |  For each horizon h in [1, 3, 5]:                                    |
  |    target_h = rate_bps.shift(-h)   (forward-looking target)         |
  |    train LGBMRegressor(horizon=h) on X_train, y_train_h             |
  |                                                                      |
  |  Models stored: models/lgbm_h1.pkl, lgbm_h3.pkl, lgbm_h5.pkl        |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 6: FORECAST GENERATION
  +----------------------------------------------------------------------+
  |  MultiHorizonForecaster.predict_df()                                 |
  |                                                                      |
  |  Output columns:                                                     |
  |    instrument_type, current_rate_bps,                                |
  |    forecast_1d_bps, forecast_3d_bps, forecast_5d_bps                |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 7: BUDGET VS FORECAST COMPARISON
  +----------------------------------------------------------------------+
  |  ForecastAlertFactory.from_forecasts()                               |
  |                                                                      |
  |  Merge forecast_df with budget_df on instrument_type                 |
  |  Compute cost_vs_budget_bps = forecast_5d_bps - budget_rate_bps     |
  |  Compute rate_delta_1d = forecast_1d_bps - current_rate_bps         |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 8: ALERT DISPATCH
  +----------------------------------------------------------------------+
  |  ForecastAlertDispatcher                                             |
  |                                                                      |
  |  RATE_SPIKE     --> forecast_1d > current + spike_threshold_bps      |
  |  BUDGET_BREACH  --> forecast_5d > budget + budget_breach_threshold   |
  |  RAPID_INCREASE --> forecast_3d > forecast_1d + spike_threshold      |
  |                                                                      |
  |  Dispatch channels (configurable):                                   |
  |  [Email SMTP]  [Teams Webhook]  [Log File]                           |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 9: POWERBI PUSH
  +----------------------------------------------------------------------+
  |  PowerBIConnector.push_rows()                                        |
  |                                                                      |
  |  Payload fields:                                                     |
  |    instrument_type, current_rate_bps, forecast_1d_bps,              |
  |    forecast_3d_bps, forecast_5d_bps, cost_vs_budget_bps,            |
  |    alert_flag, timestamp                                             |
  |                                                                      |
  |  HTTP POST --> PowerBI Streaming Dataset REST API                    |
  +----------------------------------------------------------------------+
                               |
                               v
  STEP 10: AUDIT LOG
  +----------------------------------------------------------------------+
  |  Structured JSON log entry written per run:                          |
  |                                                                      |
  |  {                                                                   |
  |    "run_id": "<uuid>",                                               |
  |    "timestamp": "<ISO-8601>",                                        |
  |    "mode": "train|forecast|both",                                    |
  |    "instruments_processed": [...],                                   |
  |    "alerts_fired": <count>,                                          |
  |    "powerbi_push_status": "ok|failed|disabled",                      |
  |    "model_metrics": {1: {...}, 3: {...}, 5: {...}}                   |
  |  }                                                                   |
  +----------------------------------------------------------------------+
```
