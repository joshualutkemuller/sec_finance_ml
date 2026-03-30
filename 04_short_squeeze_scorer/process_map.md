# Short Squeeze Scorer — Process Map

```
===========================================================================
  SHORT SQUEEZE PROBABILITY SCORER — END-TO-END PROCESS MAP
===========================================================================

STEP 1: DATA INGESTION
---------------------------------------------------------------------------
  Input Sources:
    [A] Short Interest Feed      (CSV / API)
        - ticker, date, shares_short, shares_float, short_interest_ratio
    [B] Borrow Availability Feed (CSV / API)
        - ticker, date, available_shares, utilization_rate, borrow_rate
    [C] Price / Volume Feed      (CSV / API)
        - ticker, date, open, high, low, close, volume

  Actions:
    - Load each source via ShortInterestDataLoader
    - Validate schemas (required columns, nulls, date range)
    - Merge on [ticker, date] -> unified DataFrame
    - Filter to lookback_days window

            [A] ──┐
            [B] ──┼──> merge_all() ──> validated_df
            [C] ──┘

---------------------------------------------------------------------------
STEP 2: TECHNICAL INDICATOR COMPUTATION
---------------------------------------------------------------------------
  Input:  validated_df (OHLCV + short interest + borrow data)
  Module: src/feature_engineering.py :: ShortSqueezeFeatureEngineer

  Computed Indicators:
    - RSI (default window=14)          add_rsi()
    - Volume Ratio (vol / avg_vol)     add_volume_ratio()
    - Price Momentum (N-day return)    add_price_momentum()

  Output columns added:
    rsi_14, volume_ratio_20,
    price_momentum_5, price_momentum_10, price_momentum_20

---------------------------------------------------------------------------
STEP 3: SHORT INTEREST FEATURE ENGINEERING
---------------------------------------------------------------------------
  Input:  df with price indicators from Step 2
  Module: src/feature_engineering.py :: ShortSqueezeFeatureEngineer

  Computed Features:
    - Days-to-Cover                    add_days_to_cover()
        = shares_short / avg_daily_volume (20-day)
    - Utilization Change               add_utilization_change()
        = utilization_rate.diff(1)
    - SI Ratio Momentum                add_si_ratio_momentum()
        = rolling pct_change over [5, 10, 20] day windows
    - Borrow Cost Spike (z-score)      add_borrow_cost_spike()
        = (borrow_rate - rolling_mean) / rolling_std

  Output columns added:
    days_to_cover, utilization_change,
    si_ratio_momentum_5, si_ratio_momentum_10, si_ratio_momentum_20,
    borrow_rate_zscore

  build_features() orchestrates Steps 2 + 3 in order.

---------------------------------------------------------------------------
STEP 4: XGBOOST SCORING
---------------------------------------------------------------------------
  Input:  feature_df (latest row per ticker, all feature columns)
  Module: src/model.py :: XGBoostSqueezeScorer

  Actions:
    - Load trained XGBoost model (or train if --train flag set)
    - Call scorer.score(X) -> np.ndarray of probabilities [0.0, 1.0]
    - Each value is the calibrated squeeze probability for that ticker

  Output: probs array (float64), shape (n_tickers,)

---------------------------------------------------------------------------
STEP 5: RISK TIER CLASSIFICATION
---------------------------------------------------------------------------
  Input:  probs array from Step 4
  Module: src/model.py :: XGBoostSqueezeScorer.classify()

  Thresholds (configurable in config.yaml):
    prob < elevated_threshold  ->  LOW
    prob < high_threshold      ->  ELEVATED
    prob < critical_threshold  ->  HIGH
    prob >= critical_threshold ->  CRITICAL

  Output: list[RiskTier]  (enum: LOW | ELEVATED | HIGH | CRITICAL)

---------------------------------------------------------------------------
STEP 6: ALERT DISPATCH
---------------------------------------------------------------------------
  Input:  scored_df with risk_tier column
  Module: src/alerts.py :: SqueezeAlertDispatcher

  Actions:
    - SqueezeAlertFactory.from_scores() builds SqueezeAlert objects
      for all HIGH and CRITICAL tickers
    - For each alert:
        [email]  -> send via SMTP relay
        [teams]  -> POST to Teams incoming webhook
        [log]    -> append JSON line to alert log file

  Trigger:  risk_tier in {HIGH, CRITICAL}

            scored_df
                |
                v
        SqueezeAlertFactory
                |
        +-----------+-----------+
        |           |           |
        v           v           v
     [email]    [Teams]      [log]
     SMTP       webhook      JSON
     relay      POST         line

---------------------------------------------------------------------------
STEP 7: POWERBI PUSH
---------------------------------------------------------------------------
  Input:  final_df with all scored columns
  Module: src/powerbi_connector.py :: PowerBIConnector

  Schema pushed per row:
    ticker, short_interest_ratio, days_to_cover, utilization_rate,
    squeeze_probability, risk_tier, alert_flag, timestamp

  Actions:
    - Serialize rows to JSON payload
    - POST to PowerBI Push Dataset streaming endpoint
    - Log HTTP response status

---------------------------------------------------------------------------
STEP 8: AUDIT LOG
---------------------------------------------------------------------------
  Actions:
    - Log pipeline start / end timestamps
    - Record row counts at each stage
    - Record alert counts by tier
    - Record PowerBI push status (rows pushed, HTTP status)
    - Write structured JSON entry to audit log file

===========================================================================
```
