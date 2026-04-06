# Real-Time Streaming ML — Sub-Second Borrow Rate Anomaly Detection

**Author:** Joshua Lutkemuller
**Title:** Managing Director, Securities Finance & Agency Lending
**Created:** 2026

---

## Overview

Every risk system at every bank is batch — T+1 or T+15min at best.
This module delivers **sub-second anomaly detection** on the live borrow rate tick feed.

When a hard-to-borrow name moves 200bps in 30 seconds, this system fires before
your counterparty has refreshed their screen.

**Alert latency target: < 500ms from tick receipt to Teams/email.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tick Feed (Borrow Rates)                          │
│  Market data vendor / internal lending system REST/WebSocket         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  JSON messages
                 ┌──────────▼──────────┐
                 │  Kafka Producer     │
                 │  (producer.py)      │
                 │  Topic: borrow_rates│
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Kafka Broker       │
                 │  (local / cloud)    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────────────────────┐
                 │  Flink Streaming Job (stream_job.py) │
                 │                                      │
                 │  ┌────────────────────────────────┐  │
                 │  │  Online RRCF Anomaly Detector  │  │
                 │  │  (Robust Random Cut Forest)    │  │
                 │  │  Updated per tick, O(1) per   │  │
                 │  │  new observation               │  │
                 │  └────────────────────────────────┘  │
                 │                                      │
                 │  Session windows: 30s / 60s / 5min  │
                 │  Rate velocity: |Δrate / Δt|         │
                 │  Regime-aware thresholds             │
                 └──────────┬──────────────────────────┘
                            │
            ┌───────────────┼────────────────────┐
            ▼               ▼                    ▼
     Kafka sink       Redis state          Alert dispatcher
   (anomaly_alerts)  (online model)    (< 500ms to Teams/email)
```

---

## Anomaly Detection: RRCF (Robust Random Cut Forest)

RRCF is an online anomaly detection algorithm that:
- Updates in O(1) per new data point (no batch retraining)
- Handles concept drift naturally (old points age out of the forest)
- Computes a **Collusive Displacement (CoDisp)** score per tick
- Higher CoDisp = more anomalous

**Advantages over standard Isolation Forest for streaming:**
- No fixed "training window" — the model adapts continuously
- Memory-bounded (fixed forest size, old trees are replaced)
- Proven in financial streaming contexts

---

## Alert Types

| Alert | Trigger | Severity |
|-------|---------|----------|
| `RATE_SPIKE` | CoDisp > 3σ AND |Δrate| > threshold | CRITICAL |
| `RATE_VELOCITY` | |Δrate/Δt| > velocity_threshold | WARNING |
| `REGIME_BREAK` | 30s VWAP crosses 2σ band | WARNING |
| `SESSION_ANOMALY` | 5min session summary score > threshold | INFO |

---

## Module Structure

```
realtime_streaming_ml/
├── README.md
├── requirements.txt
├── config/config.yaml
└── src/
    ├── producer.py           ← Kafka tick producer (simulated or live feed)
    ├── stream_job.py         ← PyFlink streaming job with RRCF detector
    ├── online_rrcf.py        ← Online RRCF implementation + CoDisp scoring
    ├── alert_sink.py         ← Sub-500ms alert dispatcher (Teams + email)
    └── replay_simulator.py   ← Historical data replay for backtesting
```
