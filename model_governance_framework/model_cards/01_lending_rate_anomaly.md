# Model Card — Lending Rate Anomaly Detector

**Model ID:** `01_lending_rate_anomaly`
**Version:** 1.0.0
**Author:** Joshua Lutkemuller, Managing Director — Securities Finance & Agency Lending
**Created:** 2026
**Last Updated:** 2026
**SR 11-7 Status:** Approved for production use (pending annual validation)

---

## Model Overview

| Field | Value |
|-------|-------|
| **Purpose** | Detect statistically anomalous borrow rates in the securities lending book |
| **Output** | Binary anomaly flag + anomaly score [0,1] per ticker per day |
| **Model Type** | Ensemble: Isolation Forest + LSTM Autoencoder |
| **Prediction Horizon** | Same-day (T+0) |
| **Update Frequency** | Daily, after market close |

---

## Intended Use

**In-scope:**
- Daily monitoring of borrow rates for specials detection
- Alert generation for desks when rates deviate from expected range
- Input to short squeeze probability scorer (04_short_squeeze_scorer)

**Out-of-scope:**
- Intraday rate decisions (model is daily, not real-time)
- Causal explanation of rate moves (descriptive, not causal)
- Use on securities outside the training universe

---

## Training Data

| Attribute | Value |
|-----------|-------|
| Date range | 5 years of historical borrow rates |
| Universe | All securities in the lending book with ≥ 252 trading days history |
| Features | Rolling statistics (5/10/20d), z-scores, spread vs benchmark, momentum |
| Label | Rule-based anomaly labels from desk knowledge (contamination = 0.05) |
| Train/validation split | 80/20 time-series split (no future leakage) |

---

## Performance Metrics (Validation Set)

| Metric | Value | Threshold |
|--------|-------|-----------|
| Precision | 0.84 | ≥ 0.75 |
| Recall | 0.79 | ≥ 0.70 |
| F1 Score | 0.81 | ≥ 0.72 |
| False Positive Rate | 4.2% | ≤ 6% |
| AUC-ROC | 0.91 | ≥ 0.80 |

---

## Known Limitations

1. **New listings**: Model has no history for recently listed securities — defaults to high anomaly score for first 30 trading days
2. **Structural breaks**: Model trained before 2024 may not reflect post-LIBOR transition rate dynamics without retraining
3. **Specials regime**: Model calibrated on normal market conditions; may have elevated false positive rate during specials market events
4. **Thin markets**: Low-liquidity names with infrequent borrow activity may produce unreliable scores

---

## Monitoring & Retraining Triggers

| Trigger | Action |
|---------|--------|
| PSI > 0.25 on any feature | CRITICAL alert, immediate retraining review |
| F1 drops below 0.72 | WARNING alert, schedule retraining within 5 business days |
| False positive rate > 8% | Review anomaly threshold calibration |
| More than 10 consecutive false positives on same ticker | Exclude ticker from model temporarily |

---

## Regulatory Notes

- Model output is advisory only — all alerts require human review before action
- Model version and run ID logged with every alert for SR 11-7 traceability
- Annual independent model validation required per Model Risk Policy
