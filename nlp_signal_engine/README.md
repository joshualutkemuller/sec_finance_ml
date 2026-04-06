# NLP Signal Engine — LLM Intelligence Layer for Securities Finance

**Author:** Joshua Lutkemuller
**Title:** Managing Director, Securities Finance & Agency Lending
**Created:** 2026

---

## Overview

The NLP Signal Engine transforms unstructured text — SEC filings, Fed communications,
earnings call transcripts, and real-time news — into structured, quantitative signals
that directly feed the securities lending and agency/prime lending workflows.

The flagship output is the **Securities Finance Morning Brief**: a daily AI-generated
500-word narrative combining macro regime, top risk names, borrow rate outlook, and
lending desk recommendations — delivered every morning before market open.

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Text Sources                                  │
│  SEC EDGAR (10-K/10-Q) │ FOMC Statements │ Earnings Calls │ News RSS │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │         Ingestion Layer             │
              │  sec_edgar_loader.py                │
              │  fomc_loader.py                     │
              │  news_loader.py                     │
              └────────────────┬───────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │          NLP Layer                  │
              │  FinBERT sentiment scoring          │
              │  spaCy NER (tickers, regulators)    │
              │  Risk keyword extraction            │
              │  Hawkish/dovish FOMC scoring        │
              └────────────────┬───────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │        Signal Generation            │
              │  Per-ticker sentiment z-score       │
              │  Short-squeeze precursor score      │
              │  Fed hawkishness index              │
              │  Macro narrative summary            │
              └────────────────┬───────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │      LLM Synthesis (Claude API)     │
              │  Morning Brief generation           │
              │  Risk factor summarization          │
              │  Structured desk recommendations   │
              └────────────────┬───────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │        Output & Distribution        │
              │  PowerBI push │ Email │ Teams       │
              │  Feeds 04_short_squeeze_scorer      │
              │  Feeds fred_macro_intelligence      │
              └────────────────────────────────────┘
```

---

## Signals Produced

| Signal | Source | Securities Finance Use |
|--------|--------|----------------------|
| `ticker_sentiment_z` | News + earnings NLP | Short squeeze precursor input |
| `risk_language_score` | 10-K/10-Q risk factors | Going-concern / liquidity flags |
| `fomc_hawkishness_index` | FOMC statements | Rate regime input, SOFR forecast |
| `management_uncertainty_score` | Earnings call NLP | Pre-event borrow demand signal |
| `analyst_skepticism_score` | Earnings Q&A tone | Short interest momentum signal |
| `morning_brief` | LLM synthesis | Daily desk narrative |

---

## Morning Brief Example Output

```
SECURITIES FINANCE MORNING BRIEF — 2026-04-07
Author: Joshua Lutkemuller, MD | Generated: 06:45 ET

MACRO REGIME: Hiking Cycle (confidence: 87%)
The Fed's March statement showed a hawkishness index of 0.71 (+0.12 vs Feb),
consistent with 1-2 further hikes priced. SOFR 5d forecast: +8bps.

TOP RISK NAMES:
• ACME Corp (ACME): Risk language score 0.84 — 10-Q flagged "going concern"
  language. SI ratio 24%, days-to-cover 6.2. Elevated short squeeze risk.
• XYZ Inc (XYZI): Management uncertainty score 0.79 on earnings call.
  Analyst skepticism elevated. Borrow rate z-score: +2.3.

LENDING DESK OUTLOOK:
Demand expected to increase 15-20% vs. 5d avg driven by earnings season
and tightening funding conditions. Recommend reducing HY collateral concentration
ahead of FOMC meeting (Apr 9). Agency collateral preferred.

COLLATERAL QUALITY: Neutral. IG OAS stable, MBS basis -2bps.
```

---

## Module Structure

```
nlp_signal_engine/
├── README.md
├── requirements.txt
├── config/config.yaml
└── src/
    ├── main.py                    ← CLI orchestration
    ├── sec_edgar_loader.py        ← SEC EDGAR 10-K/10-Q ingestion
    ├── fomc_loader.py             ← FOMC statement fetching + scoring
    ├── news_loader.py             ← RSS/NewsAPI real-time feed
    ├── nlp_processor.py           ← FinBERT + spaCy NLP pipeline
    ├── signal_generator.py        ← converts NLP outputs to numeric signals
    ├── morning_brief.py           ← Claude API LLM synthesis
    └── alerts.py                  ← NLP-driven alert dispatch
```
