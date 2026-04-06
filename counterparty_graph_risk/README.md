# Counterparty Graph Risk — Systemic Contagion Detection via Graph Neural Networks

**Author:** Joshua Lutkemuller
**Title:** Managing Director, Securities Finance & Agency Lending
**Created:** 2026

---

## Overview

Traditional concentration risk metrics (HHI, top-N exposure) treat counterparties as
independent nodes. In reality, counterparties share underlying securities, collateral
pools, and funding chains — creating *hidden* systemic linkages that only become visible
when stress propagates through the network.

This module models the entire lending book as a **directed weighted graph** and trains
a **Graph Attention Network (GAT)** to:

1. Score each counterparty's **systemic contagion risk** — if they default, how far
   does the stress propagate?
2. Detect **hidden concentration** — two counterparties that appear separate but share
   80%+ of the same underlying securities
3. Compute a **network fragility index** measuring how close the overall graph is to a
   cascade failure

This is the only securities-finance-specific GNN implementation designed for production
lending desk use.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Graph Construction Layer                          │
│                                                                       │
│  Nodes: Counterparties (CP) + Securities (SEC)                       │
│  Edges:                                                               │
│    CP → SEC  : open borrow trade (weight = notional)                 │
│    SEC → CP  : collateral pledge (weight = haircut-adj value)        │
│    CP → CP   : shared-security co-exposure (weight = overlap ratio)  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  Graph Attention Network (GAT) │
            │                               │
            │  Layer 1: 2-hop neighborhood  │
            │  Layer 2: 3-hop aggregation   │
            │  Attention: learned per edge  │
            │  Output: node embeddings      │
            └───────────────┬───────────────┘
                            │
   ┌────────────────────────┼────────────────────────┐
   ▼                        ▼                        ▼
Contagion Score      Hidden Concentration      Network Fragility
(per CP node)        (CP-CP similarity)        Index (graph-level)
```

---

## Key Outputs

| Output | Description | Securities Finance Use |
|--------|-------------|----------------------|
| `contagion_score` | P(default cascade reaches ≥N nodes) | Counterparty exposure limits |
| `hidden_overlap_pct` | Shared security exposure between CP pairs | Diversification audit |
| `network_fragility_index` | Graph spectral gap (Fiedler value) | Systemic risk early warning |
| `node_centrality` | PageRank of each node in the lending graph | Key node identification |
| `community_labels` | Louvain community detection clusters | Hidden sector concentrations |

---

## Module Structure

```
counterparty_graph_risk/
├── README.md
├── requirements.txt
├── config/config.yaml
└── src/
    ├── main.py                  ← CLI: build / train / score
    ├── graph_builder.py         ← constructs PyG HeteroData from loan book
    ├── gat_model.py             ← Graph Attention Network (GAT) implementation
    ├── contagion_scorer.py      ← node-level contagion risk scoring
    ├── network_analytics.py     ← centrality, community detection, fragility index
    ├── visualization.py         ← network graph visualization (Plotly + networkx)
    └── alerts.py                ← regime-aware contagion alerts
```

---

## Graph Construction

```python
# Bipartite heterogeneous graph
# CounterpartyNode features: credit_spread, utilization_rate, recall_rate,
#                            avg_tenor, notional_usd, hhi
# SecurityNode features:     short_interest, borrow_rate, days_to_cover,
#                            squeeze_score, liquidity_score

G = HeteroData()
G['counterparty'].x  = cp_features           # [N_cp, F_cp]
G['security'].x      = sec_features          # [N_sec, F_sec]
G['counterparty', 'borrows', 'security'].edge_index  = borrow_edges
G['counterparty', 'borrows', 'security'].edge_attr   = notional_weights
G['security', 'pledged_to', 'counterparty'].edge_index = collateral_edges
G['counterparty', 'co_exposed', 'counterparty'].edge_index = overlap_edges
```

---

## Network Fragility Index

Based on the **algebraic connectivity** (Fiedler value λ₂) of the lending graph
Laplacian. A higher λ₂ indicates a more robust, better-connected graph that is
harder to fragment by removing nodes. A declining λ₂ is an early warning indicator
of increasing fragility.

```
Fragility Index = 1 / λ₂(L)   where L is the normalized graph Laplacian
Critical threshold: Fragility Index > 5.0 → alert
```

---

## Regulatory Context

The OFR (Office of Financial Research) and Federal Reserve have explicitly called for
network-based systemic risk measurement in securities financing markets (OFR Brief
Series 2022-01, "Network Analysis of Repo Markets"). This module directly addresses
that regulatory direction.
