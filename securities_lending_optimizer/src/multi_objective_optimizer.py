"""
Multi-Objective Optimizer — CVXPY scalarized solver + optional NSGA-II Pareto mode.

Two modes:
  scalarized  — single CVXPY solve with RL-guided weights (fast, auditable)
  pareto      — NSGA-II genetic algorithm for Pareto-front exploration (strategic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not installed — scalarized optimizer unavailable")

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger.warning("pymoo not installed — NSGA-II Pareto mode unavailable")


@dataclass
class OptimizationResult:
    """Result of a multi-objective optimization run."""

    status: str                          # "optimal" | "infeasible" | "error"
    allocation: np.ndarray               # portfolio weights, shape (n_assets,)
    total_cost: float
    fee_revenue: float
    capital_charge: float
    recall_risk_weighted: float
    objective_weights: np.ndarray        # weights used (from RL policy)
    alerts_data: list[dict] = field(default_factory=list)

    def build_alerts(self, run_id: str) -> list[dict]:
        """Return alert dictionaries for the dispatcher."""
        alerts = []
        cfg = {"run_id": run_id, "status": self.status}

        if self.status != "optimal":
            alerts.append({**cfg, "alert_type": "OPTIMIZATION_FAILURE",
                           "severity": "CRITICAL", "message": f"Solver status: {self.status}"})
        if self.recall_risk_weighted > 0.15:
            alerts.append({**cfg, "alert_type": "HIGH_RECALL_RISK",
                           "severity": "WARNING",
                           "message": f"Weighted recall risk={self.recall_risk_weighted:.3f}"})
        return alerts + self.alerts_data


class MultiObjectiveOptimizer:
    """CVXPY scalarized or NSGA-II Pareto optimizer for lending allocation."""

    def __init__(self, config: dict) -> None:
        self.config = config["optimizer"]
        self.mode = self.config.get("mode", "scalarized")
        self.solver = getattr(cp, self.config.get("solver", "ECOS"), None) if CVXPY_AVAILABLE else None

    def optimize(
        self,
        collateral_df: pd.DataFrame,
        objective_weights: np.ndarray,
        recall_probs: np.ndarray,
        demand_forecast: dict[int, np.ndarray],
    ) -> OptimizationResult:
        """Run the optimizer.

        Args:
            collateral_df: DataFrame with one row per asset, columns include
                financing_cost, fee_rate, capital_charge_est, asset_class,
                haircut, quality_score, concentration_limit.
            objective_weights: 4-element array [w_cost, w_revenue, w_capital, w_recall]
                from the RL policy.
            recall_probs: Recall probability per asset, shape (n_assets,).
            demand_forecast: {horizon: demand_array} mapping.

        Returns:
            OptimizationResult with allocation, costs, and alert data.
        """
        if self.mode == "scalarized":
            return self._solve_scalarized(
                collateral_df, objective_weights, recall_probs, demand_forecast
            )
        elif self.mode == "pareto":
            return self._solve_pareto(
                collateral_df, objective_weights, recall_probs, demand_forecast
            )
        else:
            raise ValueError(f"Unknown optimizer mode: {self.mode}")

    def _solve_scalarized(
        self,
        df: pd.DataFrame,
        weights: np.ndarray,
        recall_probs: np.ndarray,
        demand_forecast: dict[int, np.ndarray],
    ) -> OptimizationResult:
        if not CVXPY_AVAILABLE:
            raise ImportError("Install cvxpy: pip install cvxpy ecos")

        n = len(df)
        x = cp.Variable(n, nonneg=True)

        # Cost vectors from DataFrame
        financing_cost = df["financing_cost"].to_numpy(dtype=float)
        fee_rate = df["fee_rate"].to_numpy(dtype=float)
        capital_charge = df["capital_charge_est"].to_numpy(dtype=float)
        quality_score = df["quality_score"].to_numpy(dtype=float)
        haircut = df["haircut"].to_numpy(dtype=float)

        w1, w2, w3, w4 = weights[0], weights[1], weights[2], weights[3]

        objective = cp.Minimize(
            w1 * (financing_cost @ x)
            - w2 * (fee_rate @ x)
            + w3 * (capital_charge @ x)
            + w4 * (recall_probs @ x)
        )

        constraints = [
            cp.sum(x) == 1.0,
            quality_score @ x >= self.config["min_quality_floor"],
            (1.0 - haircut) @ x >= self.config["min_collateral_coverage"] - 1.0,
            capital_charge @ x <= self.config["capital_budget_usd"],
            recall_probs @ x <= self.config["max_recall_risk_weight"],
        ]

        # Per-asset-class concentration limits
        if "asset_class" in df.columns:
            for asset_class in df["asset_class"].unique():
                mask = (df["asset_class"] == asset_class).to_numpy()
                limit = self.config["max_concentration_per_asset_class"]
                constraints.append(cp.sum(x[mask]) <= limit)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self.solver, warm_start=True)
        except Exception as exc:
            logger.error("CVXPY solver error: %s", exc)
            return OptimizationResult(
                status="error",
                allocation=np.zeros(n),
                total_cost=np.nan,
                fee_revenue=0.0,
                capital_charge=0.0,
                recall_risk_weighted=0.0,
                objective_weights=weights,
            )

        status = prob.status or "unknown"
        if x.value is None:
            alloc = np.zeros(n)
        else:
            alloc = np.maximum(x.value, 0.0)
            alloc /= alloc.sum() if alloc.sum() > 0 else 1.0

        return OptimizationResult(
            status=status,
            allocation=alloc,
            total_cost=float(financing_cost @ alloc),
            fee_revenue=float(fee_rate @ alloc),
            capital_charge=float(capital_charge @ alloc),
            recall_risk_weighted=float(recall_probs @ alloc),
            objective_weights=weights,
        )

    def _solve_pareto(
        self,
        df: pd.DataFrame,
        weights: np.ndarray,
        recall_probs: np.ndarray,
        demand_forecast: dict[int, np.ndarray],
    ) -> OptimizationResult:
        """Run NSGA-II for Pareto-front exploration."""
        if not PYMOO_AVAILABLE:
            raise ImportError("Install pymoo: pip install pymoo")

        n = len(df)
        financing_cost = df["financing_cost"].to_numpy(dtype=float)
        fee_rate = df["fee_rate"].to_numpy(dtype=float)
        capital_charge = df["capital_charge_est"].to_numpy(dtype=float)

        nsga_config = self.config.get("nsga2", {})

        class LendingProblem(Problem):
            def __init__(self_inner) -> None:
                # 3 objectives: minimize cost, maximize revenue (→ minimize -rev), minimize capital
                super().__init__(
                    n_var=n, n_obj=3, n_ieq_constr=1,
                    xl=0.0, xu=1.0
                )

            def _evaluate(self_inner, x_pop, out, *args, **kwargs) -> None:  # type: ignore[override]
                # Normalize rows to sum to 1
                row_sums = x_pop.sum(axis=1, keepdims=True).clip(min=1e-9)
                x_norm = x_pop / row_sums

                f1 = x_norm @ financing_cost          # minimize
                f2 = -(x_norm @ fee_rate)             # minimize (neg revenue)
                f3 = x_norm @ capital_charge          # minimize
                out["F"] = np.column_stack([f1, f2, f3])

                # Inequality constraint: recall_risk_weighted <= threshold
                g1 = (x_norm @ recall_probs) - self.config["max_recall_risk_weight"]
                out["G"] = g1[:, np.newaxis]

        problem = LendingProblem()
        algorithm = NSGA2(
            pop_size=nsga_config.get("pop_size", 200),
            eliminate_duplicates=True,
        )
        res = pymoo_minimize(
            problem,
            algorithm,
            ("n_gen", nsga_config.get("n_gen", 100)),
            seed=nsga_config.get("seed", 42),
            verbose=False,
        )

        # Pick the Pareto solution closest to RL-guided weights direction
        pareto_x = res.X
        row_sums = pareto_x.sum(axis=1, keepdims=True).clip(min=1e-9)
        pareto_alloc = pareto_x / row_sums

        # Score each solution with scalarized RL weights
        w1, w2, w3 = weights[0], weights[2], weights[2]
        scores = (
            w1 * (pareto_alloc @ financing_cost)
            - w2 * (pareto_alloc @ fee_rate)
            + w3 * (pareto_alloc @ capital_charge)
        )
        best_idx = int(np.argmin(scores))
        alloc = pareto_alloc[best_idx]

        return OptimizationResult(
            status="optimal",
            allocation=alloc,
            total_cost=float(financing_cost @ alloc),
            fee_revenue=float(fee_rate @ alloc),
            capital_charge=float(capital_charge @ alloc),
            recall_risk_weighted=float(recall_probs @ alloc),
            objective_weights=weights,
        )
