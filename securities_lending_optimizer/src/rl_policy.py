"""
RL Policy — PPO/SAC agent for dynamic securities lending allocation.

LendingEnv:    Custom OpenAI Gym environment wrapping a lending book simulator.
RLPolicyAgent: Wrapper around stable-baselines3 PPO/SAC with training/inference helpers.

The RL agent learns to output objective weights (w₁..w₄) that are fed into
the multi-objective CVXPY optimizer each step, rather than directly outputting
allocations. This keeps the final allocation feasible by construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Type stubs — actual imports deferred to avoid hard dependency at import time ──
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — LendingEnv will raise if instantiated")

try:
    from stable_baselines3 import PPO, SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed — RLPolicyAgent will raise if instantiated")


class LendingEnv:
    """
    OpenAI Gym-compatible environment for lending book simulation.

    State (observation):
        A flattened vector containing:
        - Per-security features: borrow demand forecast, recall probability,
          current allocation, specials flag, rate z-score, squeeze score
        - Book-level features: total notional, capital utilization, P&L rolling 30d
        - Market features: market stress index, SOFR, repo spread

    Action:
        Continuous vector of length N_OBJECTIVES representing objective weights
        [w_financing_cost, w_fee_revenue, w_capital_charge, w_recall_risk].
        These weights are passed to the CVXPY optimizer to produce the actual allocation.

    Reward:
        r_t = fee_income - λ₁*financing_cost - λ₂*capital_charge
              - λ₃*recall_penalty - λ₄*concentration_penalty
    """

    N_OBJECTIVES = 4

    def __init__(
        self,
        loan_book: pd.DataFrame,
        collateral_pool: pd.DataFrame,
        market_data: pd.DataFrame,
        config: dict,
    ) -> None:
        if not GYM_AVAILABLE:
            raise ImportError("Install gymnasium: pip install gymnasium")

        self.loan_book = loan_book
        self.collateral_pool = collateral_pool
        self.market_data = market_data
        self.config = config
        self.reward_weights = config["rl_policy"]["reward_weights"]

        # Derive observation dimension from feature set
        self._n_securities = len(collateral_pool)
        self._obs_dim = self._n_securities * 6 + 3 + 3  # see docstring
        self._episode_dates = sorted(loan_book["date"].unique())
        self._step_idx = 0

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.N_OBJECTIVES,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._step_idx = 0
        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: apply action (objective weights), simulate, get reward."""
        # Normalize objective weights to sum to 1
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum()

        # Simulate book outcome with these weights
        reward, info = self._simulate_step(weights)

        self._step_idx += 1
        done = self._step_idx >= len(self._episode_dates) - 1
        obs = self._build_obs()

        return obs, float(reward), done, False, info

    def get_current_state(self) -> np.ndarray:
        """Return the current observation for inference."""
        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        """Build the observation vector for the current step."""
        # Placeholder: in production, pull real features from data
        rng = np.random.default_rng(self._step_idx)
        obs = rng.standard_normal(self._obs_dim).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _simulate_step(self, weights: np.ndarray) -> tuple[float, dict]:
        """Compute realized reward for given objective weights at current step."""
        rw = self.reward_weights
        # Placeholder simulation — replace with actual book P&L calculation
        fee_income = np.random.uniform(0.001, 0.005)
        financing_cost = np.random.uniform(0.0005, 0.002)
        capital_charge = np.random.uniform(0.0002, 0.001)
        recall_penalty = np.random.uniform(0.0, 0.002)
        concentration_penalty = np.random.uniform(0.0, 0.001)

        reward = (
            rw["fee_income"] * fee_income
            - rw["financing_cost"] * financing_cost
            - rw["capital_charge"] * capital_charge
            - rw["recall_risk"] * recall_penalty
            - rw["concentration"] * concentration_penalty
        )

        info = {
            "fee_income": fee_income,
            "financing_cost": financing_cost,
            "capital_charge": capital_charge,
        }
        return reward, info


class RLPolicyAgent:
    """Wrapper around stable-baselines3 PPO/SAC for lending book optimization."""

    def __init__(self, env: LendingEnv, config: dict) -> None:
        if not SB3_AVAILABLE:
            raise ImportError("Install stable-baselines3: pip install stable-baselines3")

        self.config = config["rl_policy"]
        algo_name = self.config.get("algorithm", "PPO").upper()
        AlgoClass = PPO if algo_name == "PPO" else SAC

        self._model = AlgoClass(
            policy=self.config.get("policy", "MlpPolicy"),
            env=env,
            n_steps=self.config.get("n_steps", 2048),
            batch_size=self.config.get("batch_size", 64),
            learning_rate=self.config.get("learning_rate", 3e-4),
            gamma=self.config.get("gamma", 0.99),
            ent_coef=self.config.get("ent_coef", 0.01),
            verbose=1,
        )
        logger.info("RLPolicyAgent initialized with %s", algo_name)

    def train(self, total_timesteps: int) -> None:
        logger.info("Starting RL training for %d timesteps", total_timesteps)
        self._model.learn(total_timesteps=total_timesteps)
        logger.info("RL training complete")

    def recommend(self, state: np.ndarray) -> np.ndarray:
        """Return normalized objective weights for the given state."""
        action, _ = self._model.predict(state, deterministic=True)
        weights = np.clip(action, 1e-6, None)
        weights /= weights.sum()
        return weights.astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        logger.info("RLPolicyAgent saved to %s", path)

    def load(self, path: str) -> None:
        algo_name = self.config.get("algorithm", "PPO").upper()
        AlgoClass = PPO if algo_name == "PPO" else SAC
        self._model = AlgoClass.load(path)
        logger.info("RLPolicyAgent loaded from %s", path)
