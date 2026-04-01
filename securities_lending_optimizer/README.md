# Enhanced Securities Lending Optimizer — ML Framework

## Overview

This module describes and partially implements a **next-generation, multi-objective ML optimizer** for securities lending operations. It extends the existing collateral optimizer (`02_collateral_optimizer`) from a single-shot convex program into a **reinforcement learning (RL) + multi-objective optimization** framework that learns from observed lending book outcomes over time.

### Problem Statement

Securities lending desks must continuously solve a multi-objective problem:

```
Maximize:   Revenue (borrow fee income)
Minimize:   Financing cost, collateral risk, counterparty concentration
Subject to: Regulatory capital constraints, collateral eligibility, settlement limits
```

The current approach (LightGBM quality scorer + one-shot CVXPY) treats each day independently and cannot:
- Learn from counterparty behaviour over time
- Adapt to market regime changes dynamically
- Jointly optimize across multiple competing objectives
- Handle inventory constraints and recall risk

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Data Layer                                          │
│  Loan Book │ Collateral Pool │ Market Data │ Counterparty Profiles  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   Feature Engineering     │
         │  (enhanced — see below)   │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼──────────────────────────┐
         │           ML Layer                      │
         │                                         │
         │  ┌──────────────────────────────────┐   │
         │  │  Demand Forecaster (LightGBM)    │   │
         │  │  Predicts borrow demand per      │   │
         │  │  security 1d/3d/5d ahead         │   │
         │  └──────────────┬───────────────────┘   │
         │                 │                        │
         │  ┌──────────────▼───────────────────┐   │
         │  │  Recall Risk Classifier (XGBoost)│   │
         │  │  P(recall within T days) per     │   │
         │  │  counterparty-security pair       │   │
         │  └──────────────┬───────────────────┘   │
         │                 │                        │
         │  ┌──────────────▼───────────────────┐   │
         │  │  RL Policy (PPO / SAC)           │   │
         │  │  State: book snapshot + signals  │   │
         │  │  Action: allocation deltas       │   │
         │  │  Reward: risk-adjusted revenue   │   │
         │  └──────────────┬───────────────────┘   │
         └─────────────────┼────────────────────────┘
                           │
         ┌─────────────────▼────────────────────┐
         │     Multi-Objective Optimizer         │
         │  Pareto-front search via NSGA-II or  │
         │  scalarized CVXPY with RL-guided      │
         │  objective weights                    │
         └─────────────────┬────────────────────┘
                           │
         ┌─────────────────▼────────────────────┐
         │     Output & Alert Layer             │
         │  Allocation recommendations          │
         │  Substitution candidates             │
         │  Risk budget utilization             │
         │  PowerBI push + alert dispatch       │
         └──────────────────────────────────────┘
```

---

## Module Structure

```
securities_lending_optimizer/
├── README.md                     ← this file
├── requirements.txt
├── config/
│   └── config.yaml               ← all non-secret parameters
└── src/
    ├── main.py                   ← CLI: train / score / optimize
    ├── data_ingestion.py         ← loan book + collateral pool loader
    ├── feature_engineering.py    ← enhanced feature set (see below)
    ├── demand_forecaster.py      ← LightGBM multi-horizon demand model
    ├── recall_risk_classifier.py ← XGBoost recall probability model
    ├── rl_policy.py              ← PPO/SAC RL agent (stable-baselines3)
    ├── multi_objective_optimizer.py  ← CVXPY + NSGA-II Pareto solver
    ├── alerts.py                 ← alert dataclasses + dispatcher
    └── powerbi_connector.py      ← Azure AD OAuth2 + push/stream
```

---

## Enhanced Feature Set

Beyond the current collateral optimizer features, this framework adds:

### Counterparty Features
| Feature | Description |
|---------|-------------|
| `cp_utilization_rate` | Fraction of counterparty's eligible collateral currently pledged |
| `cp_recall_rate_30d` | Historical recall frequency over 30 days |
| `cp_credit_spread` | CDS spread or internal credit score proxy |
| `cp_concentration_hhi` | HHI of counterparty's collateral across asset classes |
| `cp_avg_term_days` | Average tenor of open trades with counterparty |

### Market Regime Features
| Feature | Description |
|---------|-------------|
| `market_stress_index` | Composite of VIX, credit spreads, repo-SOFR spread |
| `specials_ratio` | Fraction of book on special vs. GC rates |
| `squeeze_risk_score` | Output from `04_short_squeeze_scorer` for each security |
| `inventory_cover_days` | Available inventory / avg daily borrow demand |
| `rate_momentum_5d` | 5-day rate momentum for each security |

### Book-Level Features
| Feature | Description |
|---------|-------------|
| `book_pnl_rolling_30d` | Rolling 30-day realized P&L per security |
| `capital_charge_est` | Estimated RWA contribution of each position |
| `settlement_fail_rate` | Historical settlement failure rate per security |
| `open_recall_count` | Number of pending recalls on the book |

---

## ML Components

### 1. Demand Forecaster (`demand_forecaster.py`)

Predicts borrow demand (notional, number of trades) per security per horizon.

- **Model**: Multi-output LightGBM (one model per horizon: 1d, 3d, 5d)
- **Target**: Next-day / next-3d / next-5d borrow demand (shares)
- **Loss**: Asymmetric quantile loss (over-forecasting is costlier than under)
- **Key inputs**: Historical borrow volume, short interest, analyst event calendar, macro rates

```python
class DemandForecaster:
    def __init__(self, horizons: list[int] = [1, 3, 5]): ...
    def fit(self, X: pd.DataFrame, y_dict: dict[int, pd.Series]) -> None: ...
    def predict(self, X: pd.DataFrame) -> dict[int, np.ndarray]: ...
    def feature_importance(self) -> pd.DataFrame: ...
```

### 2. Recall Risk Classifier (`recall_risk_classifier.py`)

Estimates probability of a counterparty recalling collateral within T days.

- **Model**: XGBoost binary classifier with isotonic calibration
- **Target**: 1 if recall occurs within T days, else 0
- **Features**: Counterparty history, market stress, trade tenor, specials status
- **Output**: Calibrated P(recall) per (counterparty, security) pair

```python
class RecallRiskClassifier:
    def __init__(self, horizon_days: int = 5): ...
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
```

### 3. RL Policy Agent (`rl_policy.py`)

Learns an allocation policy that maximizes risk-adjusted revenue over an episode (e.g., 1 quarter of trading days).

- **Algorithm**: Proximal Policy Optimization (PPO) via `stable-baselines3`
- **State space**: Flattened snapshot of book features + market signals (continuous)
- **Action space**: Continuous delta vector — how much to increase/decrease each allocation bucket
- **Reward function**:

```
r_t = fee_income_t
      - λ₁ * financing_cost_t
      - λ₂ * capital_charge_t
      - λ₃ * recall_risk_penalty_t
      - λ₄ * concentration_penalty_t
      + bonus if regulatory constraints satisfied
```

- **Simulation environment**: Custom `gym.Env` wrapping a synthetic or historical book replay
- **Training regime**: Train on 2-3 years of historical data; evaluate on held-out periods

```python
class LendingEnv(gym.Env):
    """OpenAI Gym environment for lending book simulation."""
    def __init__(self, book_data: pd.DataFrame, config: dict): ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...

class RLPolicyAgent:
    def __init__(self, env: LendingEnv, config: dict): ...
    def train(self, total_timesteps: int = 500_000) -> None: ...
    def recommend(self, state: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### 4. Multi-Objective Optimizer (`multi_objective_optimizer.py`)

Given RL-guided objective weights and ML-generated signals, solves the final allocation problem.

**Two modes:**

**Mode A — Weighted Scalarization (fast, production-grade):**
```
min  w₁·financing_cost(x) - w₂·fee_revenue(x) + w₃·capital_charge(x) + w₄·recall_risk(x)
s.t. sum(x) = 1
     x[i] ≥ 0  ∀i
     x[i] ≤ max_concentration  for each asset class
     haircut_adjusted_value(x) ≥ min_collateral_value
     quality_score(x) ≥ min_quality_floor
     capital_charge(x) ≤ capital_budget
```
Weights `w₁..w₄` are output by the RL policy. Solved via CVXPY (ECOS solver).

**Mode B — NSGA-II Pareto Exploration (periodic, strategic):**
- Non-dominated sorting genetic algorithm for 3-4 objective Pareto front
- Returns a set of Pareto-optimal allocations for human review
- Useful for end-of-day strategic rebalancing decisions

```python
class MultiObjectiveOptimizer:
    def __init__(self, mode: str = "scalarized", config: dict = {}): ...
    def optimize(
        self,
        collateral_df: pd.DataFrame,
        objective_weights: np.ndarray,
        recall_probs: np.ndarray,
        demand_forecast: dict[int, np.ndarray],
    ) -> OptimizationResult: ...
```

---

## Training Pipeline

```bash
# Train all ML components
python src/main.py --config config/config.yaml --mode train

# Score and optimize (production run)
python src/main.py --config config/config.yaml --mode score

# Train then score
python src/main.py --config config/config.yaml --mode both

# RL training only (long-running — recommend GPU or Databricks cluster)
python src/main.py --config config/config.yaml --mode train_rl --timesteps 1000000
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| PPO over DQN | Continuous action space (allocation deltas); PPO handles better |
| Separate demand + recall models | Modularity; each can be retrained independently |
| Scalarized CVXPY in production | Deterministic, fast (<1s), auditable for compliance |
| NSGA-II for strategic rebalancing | Provides Pareto front for trader decision support |
| Asymmetric quantile loss | Borrow demand over-estimates are more costly than under-estimates |
| Isotonic calibration on recall classifier | Ensures probabilities are well-calibrated for risk weighting |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Demand forecast MAPE (1d horizon) | < 8% |
| Recall risk AUC | > 0.80 |
| RL policy Sharpe ratio (simulated) | > 1.5 vs. baseline |
| Optimizer solve time | < 2 seconds per run |
| Cost savings vs. current optimizer | > 5 bps per annum |

---

## Regulatory Considerations

- All allocation recommendations are logged with run ID and config hash for audit
- Capital charge estimates follow SA-CCR / Basel IV RWA formulas
- Concentration limits configurable per jurisdiction (ESMA, SEC, FCA)
- Model explainability: SHAP values available for demand forecaster and recall classifier
- RL policy decisions surfaced via SHAP on the state features at each timestep

---

## Dependencies

See `requirements.txt`. Key additions beyond the existing projects:

```
stable-baselines3>=2.2
gymnasium>=0.29
pymoo>=0.6           # NSGA-II implementation
torch>=2.0           # RL neural network backend
optuna>=3.4          # Hyperparameter search for RL policy
```
