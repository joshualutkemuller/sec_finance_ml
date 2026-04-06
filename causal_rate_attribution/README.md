# Causal Rate Attribution — Financing Cost Decomposition via Causal ML

**Author:** Joshua Lutkemuller
**Title:** Managing Director, Securities Finance & Agency Lending
**Created:** 2026

---

## Overview

Correlation-based models tell you *what* happened to financing costs.
Causal models tell you *why* — separating the effect of Fed policy from market-driven
spread widening from internal book composition changes.

This module answers the three questions every CFO and treasury head asks:

1. **"Why did our Q4 financing cost increase $47M vs budget?"**
   → Causal decomposition: $31M from Fed policy, $9M from spread widening, $7M from book mix

2. **"If the Fed hadn't hiked in November, what would our costs have been?"**
   → Counterfactual: estimated $38M lower financing cost

3. **"Which counterparties and asset classes are most causally sensitive to rate changes?"**
   → Heterogeneous treatment effects: Agency UST +0.92 sensitivity, HY Corp +0.34

This is publishable-grade analysis. EconML counterfactual work is currently produced
by quant researchers at central banks and top hedge funds — almost nobody at a
securities finance desk has it.

---

## Methodology

### Framework: DoWhy + EconML (Microsoft Research)

```
Treatment (T):   Fed Funds Rate change (Δbps per meeting)
Outcome (Y):     Daily financing cost (bps above SOFR)
Confounders (W): Market stress index, curve slope, book notional,
                 asset class mix, counterparty concentration
Instrument (Z):  FOMC meeting indicator (exogenous to Y, affects T)
```

### Causal Graph (DAG)

```
FOMC Meeting ──→ Fed Funds Change ──→ Financing Cost
                        │                    ↑
                Market Stress ───────────────┘
                Curve Slope ─────────────────┘
                Book Composition ────────────┘
```

Fed meetings are used as **instrumental variables** — they are exogenous shocks to
the Fed Funds Rate that are not directly caused by current market conditions,
allowing causal identification of the rate effect.

### Three Analyses

| Analysis | Method | Output |
|----------|--------|--------|
| Average Treatment Effect (ATE) | DoWhy IV estimation | $X per 25bps hike |
| Counterfactual | EconML DML | "If no hike, cost = $Y" |
| Heterogeneous TE (HTE) | EconML CausalForest | Per-counterparty / per-asset-class sensitivity |

---

## Module Structure

```
causal_rate_attribution/
├── README.md
├── requirements.txt
├── config/config.yaml
└── src/
    ├── main.py                   ← CLI: estimate / counterfactual / hte / report
    ├── data_builder.py           ← builds causal dataset from loan book + FRED
    ├── causal_estimator.py       ← DoWhy ATE + IV estimation
    ├── counterfactual.py         ← EconML DML counterfactual analysis
    ├── heterogeneous_effects.py  ← EconML CausalForest HTE
    ├── attribution_report.py     ← CFO-ready cost decomposition narrative
    └── visualization.py          ← waterfall, HTE forest plot, counterfactual band
```

---

## Sample Attribution Report Output

```
FINANCING COST ATTRIBUTION REPORT — Q4 2025
Author: Joshua Lutkemuller, MD

TOTAL FINANCING COST INCREASE VS Q3: +$47.2M (+18.3%)

CAUSAL DECOMPOSITION:
  Fed Policy Effect (3 hikes, +75bps total):     +$31.4M  (66.5%)
    └─ IV estimate: $418K per 1bps hike
  Market Spread Widening (HY OAS +180bps):       +$9.1M   (19.3%)
  Book Composition Change (HY mix +8ppts):        +$6.7M   (14.2%)
  Residual / Model Error:                         -$0.0M   (0.0%)

COUNTERFACTUAL — "No November Hike" Scenario:
  Actual Q4 cost:        $304.2M
  Counterfactual cost:   $273.1M
  Estimated saving:      -$31.1M  (if Nov hike had been skipped)

HETEROGENEOUS SENSITIVITY (top asset classes):
  Agency UST:       0.92 (highly sensitive to Fed)
  Agency MBS:       0.78
  IG Corp:          0.51
  HY Corp:          0.34 (partially insulated — driven by credit spreads)
  Equity Collateral: 0.18
```
