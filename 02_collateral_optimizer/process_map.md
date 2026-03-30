# Collateral Optimization & Substitution Recommender — Process Map

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║              COLLATERAL OPTIMIZATION & SUBSTITUTION RECOMMENDER                    ║
║                           End-to-End Process Map                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — DATA INGESTION                                                            │
│                                                                                     │
│  Source A: Collateral Pool CSV                                                      │
│    ├── collateral_id, asset_type, isin, face_value, market_value                   │
│    ├── currency, maturity_date, coupon_rate, credit_rating                         │
│    └── issuer, country, sector                                                      │
│                                                                                     │
│  Source B: Eligibility Schedule CSV                                                 │
│    ├── counterparty_id, asset_type, eligible (bool)                                │
│    ├── min_credit_rating, max_maturity_years                                       │
│    └── accepted_currencies                                                          │
│                                                                                     │
│  Source C: Haircut Table CSV                                                        │
│    ├── asset_type, credit_rating_bucket, maturity_bucket                           │
│    └── haircut_pct                                                                  │
│                                                                                     │
│  Source D: Transaction CSV                                                          │
│    ├── transaction_id, counterparty_id, required_value                             │
│    ├── financing_rate_bps, settlement_date                                         │
│    └── current_collateral_id (existing allocation)                                 │
│                                                                                     │
│  Action: CollateralDataLoader.merge_all()                                           │
│    → LEFT JOIN collateral_pool ON eligibility_schedule (asset_type)                │
│    → LEFT JOIN haircut_table ON (asset_type, rating_bucket, maturity_bucket)       │
│    → LEFT JOIN transactions ON counterparty_id                                     │
│    → Output: merged_df  (N rows × M columns)                                       │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2 — FEATURE ENGINEERING                                                       │
│                                                                                     │
│  CollateralFeatureEngineer.build_features(merged_df)                               │
│                                                                                     │
│  2a. add_duration_score()                                                           │
│      ├── compute modified_duration from maturity_date & coupon_rate                │
│      └── normalise to [0, 1] using MinMaxScaler                                    │
│                                                                                     │
│  2b. add_credit_rating_numeric()                                                    │
│      ├── map: AAA→1, AA+→2, AA→3, AA-→4, A+→5, A→6, A-→7,                       │
│      │         BBB+→8, BBB→9, BBB-→10, NR→11                                      │
│      └── invert so higher = better: rating_score = 1 − (numeric − 1) / 10         │
│                                                                                     │
│  2c. add_liquidity_score()                                                          │
│      ├── proxy: 1/haircut_pct (lower haircut → more liquid)                       │
│      └── clip and normalise to [0, 1]                                              │
│                                                                                     │
│  2d. add_concentration_metrics()                                                    │
│      ├── pct_of_pool_by_asset_type = sum(market_value) per asset_type / total     │
│      └── is_concentrated flag if pct > max_concentration_pct                      │
│                                                                                     │
│  2e. add_cost_of_carry()                                                            │
│      ├── cost_of_carry = haircut_pct × financing_rate_bps / 10000                 │
│      └── normalise to [0, 1]                                                       │
│                                                                                     │
│  Output: feature_df with columns:                                                   │
│    [duration_score, rating_score, liquidity_score, concentration_pct,              │
│     cost_of_carry_norm, eligible, haircut_pct, market_value, ...]                  │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3 — GRADIENT BOOSTING QUALITY SCORING                                         │
│                                                                                     │
│  CollateralQualityScorer (LightGBM Regressor)                                      │
│                                                                                     │
│  Training (offline / periodic refit):                                               │
│    ├── X = feature_df[FEATURE_COLS]                                                │
│    ├── y = historical quality label (e.g., non-default / margin call flag)         │
│    ├── LGBMRegressor(n_estimators, max_depth, learning_rate, random_state)        │
│    ├── fit(X_train, y_train)                                                       │
│    └── save model to disk (joblib)                                                 │
│                                                                                     │
│  Inference:                                                                         │
│    ├── load model from disk                                                        │
│    ├── quality_scores = model.predict(X)                                           │
│    └── clip to [0, 1]                                                              │
│                                                                                     │
│  Output: quality_scores  (np.ndarray, shape [N])                                   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — CVXPY OPTIMIZATION                                                        │
│                                                                                     │
│  CollateralOptimizer.optimize(collateral_df, quality_scores, required_value)       │
│                                                                                     │
│  Decision variables:                                                                │
│    x ∈ R^N   (allocation weights for each collateral item)                         │
│                                                                                     │
│  Parameters (from config):                                                          │
│    c  = financing_cost vector    (cost_of_carry per item)                          │
│    q  = quality_scores           (ML model output)                                 │
│    H  = haircut_pct vector                                                         │
│    λ  = quality_weight                                                             │
│    α  = cost_weight                                                                │
│                                                                                     │
│  Objective:                                                                         │
│    minimise  α * (c @ x) − λ * (q @ x)                                            │
│                                                                                     │
│  Constraints:                                                                       │
│    1.  Σ x_i = 1                           (fully invested)                        │
│    2.  x_i ≥ 0  ∀ i                        (no short selling)                      │
│    3.  x_i = 0  if eligible[i] == False   (eligibility hard filter)               │
│    4.  Σ x_i [asset_type==k] ≤ max_conc  ∀ k  (concentration limit)              │
│    5.  H @ x ≤ max_haircut                 (blended haircut ceiling)               │
│    6.  q @ x ≥ min_quality_score           (quality floor)                         │
│                                                                                     │
│  Solver: ECOS (default) or SCS (fallback)                                          │
│                                                                                     │
│  Output: collateral_df with new columns:                                            │
│    [allocation_weight, recommended (bool), optimization_cost]                      │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5 — SUBSTITUTION RECOMMENDATION GENERATION                                    │
│                                                                                     │
│  CollateralOptimizer.generate_recommendations(optimized_df)                        │
│                                                                                     │
│  For each row where recommended == True:                                            │
│    ├── compute cost_savings_bps vs current collateral                              │
│    ├── build recommendation dict:                                                  │
│    │     { collateral_id, asset_type, haircut, quality_score,                     │
│    │       allocation_weight, optimization_cost, cost_savings_bps,                │
│    │       recommended, timestamp }                                                │
│    └── append to recommendations list                                              │
│                                                                                     │
│  Output: List[dict] of recommended substitutions                                   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6 — THRESHOLD EVALUATION                                                      │
│                                                                                     │
│  For each recommendation:                                                           │
│    ├── cost_savings_bps >= alerts.cost_savings_threshold_bps ?                     │
│    │     → YES: AlertLevel.WARNING or CRITICAL                                     │
│    │     → NO:  AlertLevel.INFO (log only)                                         │
│    ├── quality_score < model.quality_score_threshold ?                             │
│    │     → AlertLevel.CRITICAL                                                     │
│    └── optimization status == infeasible ?                                         │
│          → AlertLevel.CRITICAL (OPTIMISATION_FAILURE)                              │
│                                                                                     │
│  CollateralAlertFactory.from_recommendations(recommendations, config)              │
│  Output: List[CollateralAlert]                                                      │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7 — ALERT DISPATCH                                                            │
│                                                                                     │
│  CollateralAlertDispatcher(config)                                                  │
│                                                                                     │
│  For each CollateralAlert:                                                          │
│    ├── alerts.enabled == True?                                                      │
│    ├── dispatch_email()   → SMTP with HTML body summarising alert                  │
│    ├── dispatch_teams()   → HTTP POST to teams_webhook_url (JSON card)             │
│    └── dispatch_log()     → structured JSONL append to alerts.log_path             │
│                                                                                     │
│  Rate limiting: batch alerts per run to avoid spam                                 │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 8 — POWERBI PUSH                                                              │
│                                                                                     │
│  PowerBIConnector(config)                                                           │
│    ├── _get_access_token()  → OAuth2 client_credentials via MSAL / requests       │
│    ├── For each recommendation row:                                                 │
│    │     stream_row(row_dict)  → POST to streaming_url                             │
│    └── publish_recommendations(df) → batch push_rows() call                       │
│                                                                                     │
│  Payload schema per row:                                                            │
│    { collateral_id, asset_type, haircut, quality_score,                            │
│      recommended, optimization_cost, timestamp }                                   │
│                                                                                     │
│  PowerBI dashboard auto-refreshes from push dataset (real-time tile)               │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 9 — AUDIT LOG                                                                 │
│                                                                                     │
│  Written by main.py after every pipeline run:                                      │
│    ├── run_id (UUID)                                                               │
│    ├── timestamp (UTC ISO-8601)                                                    │
│    ├── n_collateral_items_evaluated                                                │
│    ├── n_recommended                                                               │
│    ├── optimization_status (optimal / infeasible / solver_error)                  │
│    ├── total_optimization_cost                                                     │
│    ├── n_alerts_dispatched                                                         │
│    ├── powerbi_push_status (success / failure)                                     │
│    └── config_snapshot (hash of config.yaml)                                       │
│                                                                                     │
│  Format: JSONL appended to alerts.log_path (same file, separate entry type)       │
│  Retention: managed externally (logrotate / S3 lifecycle)                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
