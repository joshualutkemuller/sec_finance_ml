# Securities Lending Rate Anomaly Detector — Process Map

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              SECURITIES LENDING RATE ANOMALY DETECTION PIPELINE                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA INGESTION                                                         │
│                                                                                 │
│  Source Selection (config: data.source_type)                                   │
│                                                                                 │
│  ┌───────────┐   ┌─────────────────┐   ┌──────────────────┐                    │
│  │  CSV File │   │   Databricks    │   │   REST API        │                    │
│  │           │   │  (SQL query /   │   │  (GET endpoint,  │                    │
│  │ pandas    │   │  spark session) │   │   JSON/CSV resp) │                    │
│  │ read_csv  │   │  databricks-sdk │   │   requests.get   │                    │
│  └─────┬─────┘   └────────┬────────┘   └────────┬─────────┘                    │
│        └─────────────────┬┘────────────────────┘                               │
│                          ▼                                                      │
│              ┌───────────────────────┐                                          │
│              │  Schema Validation    │                                          │
│              │  - ticker_col exists  │                                          │
│              │  - rate_col exists    │                                          │
│              │  - date_col exists    │                                          │
│              │  - no null rates      │                                          │
│              └───────────┬───────────┘                                          │
│                          │ Raw DataFrame (ticker, rate, date, ...)              │
└──────────────────────────┼──────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: FEATURE ENGINEERING                                                    │
│                                                                                 │
│  Input: Raw DataFrame                                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  2a. Rolling Statistics (windows: 5, 10, 20 days)                       │   │
│  │      • rate_rolling_mean_{w}   = df[rate].rolling(w).mean()             │   │
│  │      • rate_rolling_std_{w}    = df[rate].rolling(w).std()              │   │
│  │      • rate_zscore_{w}         = (rate - mean) / std                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  2b. Spread Features                                                    │   │
│  │      • spread_vs_benchmark     = rate - benchmark_rate                  │   │
│  │      • spread_rolling_mean_10  = spread.rolling(10).mean()              │   │
│  │      • spread_zscore           = (spread - spread_mean) / spread_std    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  2c. Momentum Features                                                  │   │
│  │      • rate_change_1d          = rate.diff(1)                           │   │
│  │      • rate_change_5d          = rate.diff(5)                           │   │
│  │      • rate_acceleration       = rate_change_1d.diff(1)                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  2d. Liquidity Proxy Features                                           │   │
│  │      • rate_high_low_range     = rolling_max - rolling_min              │   │
│  │      • rate_volatility_ratio   = rolling_std / rolling_mean             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Output: Feature Matrix (n_samples × n_features), NaN rows dropped             │
└──────────────────────────┼──────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌─────────────────────────┐   ┌──────────────────────────────────────────────────┐
│  STEP 3: ISOLATION      │   │  STEP 4: LSTM AUTOENCODER                        │
│  FOREST SCORING         │   │  RECONSTRUCTION ERROR                            │
│                         │   │                                                  │
│  Model: sklearn         │   │  Model: PyTorch LSTMAutoencoder                  │
│  IsolationForest        │   │                                                  │
│                         │   │  4a. Build sequences                             │
│  3a. Fit (train mode)   │   │      sliding window of length sequence_length    │
│      IF.fit(X_train)    │   │                                                  │
│                         │   │  4b. Fit (train mode)                            │
│  3b. Score              │   │      Encoder: LSTM → hidden states               │
│      raw = IF.decision_ │   │      Decoder: LSTM → reconstruct input           │
│            function(X)  │   │      Loss: MSELoss                               │
│      norm to [0,1]      │   │      Optimizer: Adam                             │
│      (high = anomalous) │   │                                                  │
│                         │   │  4c. Score                                       │
│  Output: if_score array │   │      recon_error = MSE(x, x_hat) per sequence    │
│                         │   │      normalize errors to [0,1]                   │
│                         │   │      threshold = percentile(train_errors, 95)    │
│                         │   │                                                  │
│                         │   │  Output: lstm_score array                        │
└────────────┬────────────┘   └──────────────────────┬───────────────────────────┘
             │                                        │
             └──────────────────┬─────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: COMBINED ANOMALY SCORING                                               │
│                                                                                 │
│  combined_score = α × if_score + (1 - α) × lstm_score                          │
│                                                                                 │
│  Default weighting: α = 0.5 (equal weight)                                     │
│  Both scores are in [0, 1] — higher values indicate greater anomalousness       │
│                                                                                 │
│  Output DataFrame columns:                                                      │
│    ticker | date | rate | if_score | lstm_score | combined_score | is_anomaly   │
└──────────────────────────┬──────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: THRESHOLD EVALUATION                                                   │
│                                                                                 │
│  Threshold: alerts.anomaly_score_threshold (default: 0.7)                      │
│                                                                                 │
│  is_anomaly = combined_score > threshold                                        │
│                                                                                 │
│  Alert Level assignment:                                                        │
│    combined_score ≥ 0.90  →  CRITICAL                                          │
│    combined_score ≥ 0.70  →  WARNING                                           │
│    combined_score ≥ 0.50  →  INFO                                              │
│    combined_score  < 0.50  →  (no alert)                                       │
└──────────────────────────┬──────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │ Anomalies only          │ All rows
              ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: ALERT DISPATCH                                                         │
│                                                                                 │
│  AlertFactory.from_scores(df, config) → list[Alert]                            │
│                                                                                 │
│  AlertDispatcher.dispatch(alerts):                                              │
│                                                                                 │
│  ┌─────────────────┐   ┌───────────────────┐   ┌──────────────────────────┐   │
│  │  Email Alert    │   │  Teams Webhook    │   │  Log File                │   │
│  │                 │   │                   │   │                          │   │
│  │  smtplib SMTP   │   │  POST JSON card   │   │  JSON line appended      │   │
│  │  TLS/STARTTLS   │   │  to webhook URL   │   │  to alerts.log_path      │   │
│  │  HTML body      │   │  Adaptive Card    │   │  (structured logging)    │   │
│  │  with ticker,   │   │  format with      │   │  Fields: ticker, rate,   │   │
│  │  rate, score    │   │  rate, score,     │   │  score, level, timestamp │   │
│  │                 │   │  alert level      │   │                          │   │
│  └─────────────────┘   └───────────────────┘   └──────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: POWERBI PUSH                                                           │
│                                                                                 │
│  PowerBIConnector.publish_alerts(df):                                          │
│                                                                                 │
│  8a. Authentication                                                             │
│      POST https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token      │
│      grant_type=client_credentials                                              │
│      → Bearer token (cached per run)                                           │
│                                                                                 │
│  8b. Push Dataset (batch historical rows)                                       │
│      POST /v1.0/myorg/groups/{workspaceId}/datasets/{datasetId}/               │
│           tables/{tableName}/rows                                               │
│      Payload: { "rows": [ {ticker, rate, anomaly_score,                        │
│                             alert_flag, timestamp}, ... ] }                    │
│                                                                                 │
│  8c. Streaming Dataset (real-time individual rows)                              │
│      POST streaming_url (configured push URL)                                  │
│      Payload: [ {ticker, rate, anomaly_score, alert_flag, timestamp} ]         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │   Financing Solutions Intelligence Platform (PowerBI Dashboard)         │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │   │
│  │   │ Rate Heatmap │  │ Anomaly Score│  │ Alert History│                 │   │
│  │   │ by ticker    │  │ Time Series  │  │ Table        │                 │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: AUDIT LOGGING                                                          │
│                                                                                 │
│  Logged at every stage:                                                         │
│    • Pipeline start/end (mode, config path, timestamp)                         │
│    • Data load: rows loaded, source type, schema validation result             │
│    • Feature engineering: feature count, NaN rows dropped                      │
│    • Model fit: hyperparameters, training duration                             │
│    • Scoring: n_anomalies detected, score distribution stats                  │
│    • Alert dispatch: channel, success/fail, recipient/webhook, timestamp       │
│    • PowerBI push: rows pushed, HTTP status, dataset ID, timestamp            │
│    • Errors: full stack trace, input shape at failure point                   │
│                                                                                 │
│  Output: Python logging to stdout + rotating file handler                      │
│          Structured JSON entries for machine-readable audit trail              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Raw Rate Data
    │
    ▼
[Validated DataFrame]
    │
    ▼
[Feature Matrix: n_samples × ~15 features]
    │
    ├──────────────────────────────┐
    ▼                              ▼
[IF Score: 0–1]           [LSTM Recon Error: 0–1]
    │                              │
    └──────────────┬───────────────┘
                   ▼
          [Combined Score: 0–1]
                   │
          ┌────────┴────────┐
          ▼                 ▼
    [is_anomaly=True]  [is_anomaly=False]
          │
    ┌─────┴──────┐
    ▼            ▼
[Alerts]   [PowerBI Push]
```
