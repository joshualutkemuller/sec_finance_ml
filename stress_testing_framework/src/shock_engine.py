"""
Shock Engine — Applies predefined or custom shocks to balance sheet inputs.

All methods return a *stressed copy* of the inputs — originals are never modified.
This design allows straightforward baseline vs. stressed comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BaseInputs:
    """Container for the unshocked balance sheet snapshot."""
    loan_book: pd.DataFrame
    collateral_pool: pd.DataFrame
    market_data: pd.DataFrame

    def copy(self) -> "BaseInputs":
        return BaseInputs(
            loan_book=self.loan_book.copy(),
            collateral_pool=self.collateral_pool.copy(),
            market_data=self.market_data.copy(),
        )


@dataclass
class StressedInputs(BaseInputs):
    """Balance sheet inputs after shocks applied, with metadata."""
    scenario_id: str = ""
    shocks_applied: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.shocks_applied is None:
            self.shocks_applied = {}


class ShockEngine:
    """
    Applies a configurable set of shocks to a BaseInputs snapshot.

    Usage:
        engine = ShockEngine()
        stressed = engine.apply(base_inputs, scenario)
    """

    def apply(self, base: BaseInputs, scenario: dict) -> StressedInputs:
        """Apply all shocks defined in the scenario to the base inputs.

        Args:
            base: Unshocked balance sheet snapshot.
            scenario: Loaded YAML scenario dict (from ScenarioLoader).

        Returns:
            StressedInputs — stressed copy with metadata.
        """
        result = StressedInputs(
            loan_book=base.loan_book.copy(),
            collateral_pool=base.collateral_pool.copy(),
            market_data=base.market_data.copy(),
            scenario_id=scenario.get("scenario_id", "unknown"),
            shocks_applied={},
        )

        shocks = scenario.get("shocks", {})

        # Apply each shock type in a deterministic order
        if shocks.get("rate_shift_bps", 0) != 0 or shocks.get("sofr_shock_bps", 0) != 0:
            result = self._apply_rate_shock(result, shocks)

        if shocks.get("ig_spread_shock_bps", 0) != 0 or shocks.get("hy_spread_shock_bps", 0) != 0:
            result = self._apply_credit_spread_shock(result, shocks)

        if shocks.get("equity_shock_pct", 0.0) != 0.0:
            result = self._apply_equity_shock(result, shocks)

        if shocks.get("bid_ask_multiplier", 1.0) != 1.0:
            result = self._apply_liquidity_shock(result, shocks)

        if shocks.get("remove_counterparty_ids"):
            result = self._apply_counterparty_shock(result, shocks)

        if shocks.get("haircut_floor_increase_pp", 0) != 0:
            result = self._apply_regulatory_shock(result, shocks)

        logger.info(
            "Scenario %s applied — %d shock types, collateral_pool rows=%d",
            result.scenario_id,
            len(result.shocks_applied),
            len(result.collateral_pool),
        )
        return result

    # ── Rate shock ────────────────────────────────────────────────────────────

    def _apply_rate_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Reprice fixed-income collateral using duration-based approximation."""
        shift_bps = shocks.get("rate_shift_bps", 0)
        shift_decimal = shift_bps / 10_000

        cp = inputs.collateral_pool
        if "modified_duration" in cp.columns and "market_value" in cp.columns:
            # ΔP ≈ -ModDuration * ΔRate * P
            price_change_pct = -cp["modified_duration"] * shift_decimal
            mask_fixed = cp.get("asset_class", pd.Series(["OTHER"] * len(cp))) \
                           .isin(["UST", "Agency", "Corp", "Muni"])
            cp.loc[mask_fixed, "market_value"] *= (1 + price_change_pct[mask_fixed])
            # Recompute effective collateral value after haircut
            if "haircut" in cp.columns:
                cp["collateral_value"] = cp["market_value"] * (1 - cp["haircut"])

        # Shock SOFR / repo rate in market data
        sofr_shock = shocks.get("sofr_shock_bps", shift_bps)
        md = inputs.market_data
        for rate_col in ["sofr", "fed_funds", "repo_rate", "financing_rate"]:
            if rate_col in md.columns:
                md[rate_col] = md[rate_col] + sofr_shock / 10_000

        inputs.shocks_applied["rate"] = {"rate_shift_bps": shift_bps, "sofr_shock_bps": sofr_shock}
        logger.debug("Rate shock applied: %d bps", shift_bps)
        return inputs

    # ── Credit spread shock ───────────────────────────────────────────────────

    def _apply_credit_spread_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Widen credit spreads and reprice affected bonds."""
        ig_shock = shocks.get("ig_spread_shock_bps", 0) / 10_000
        hy_shock = shocks.get("hy_spread_shock_bps", 0) / 10_000
        sector_shocks = {k: v / 10_000 for k, v in
                         shocks.get("sector_spread_shocks", {}).items()}

        cp = inputs.collateral_pool
        if "rating" in cp.columns and "spread_duration" in cp.columns and "market_value" in cp.columns:
            ig_mask = cp["rating"].isin(["AAA", "AA", "A", "BBB"])
            hy_mask = ~ig_mask & cp["rating"].notna()

            cp.loc[ig_mask, "market_value"] *= (
                1 - cp.loc[ig_mask, "spread_duration"] * ig_shock
            )
            cp.loc[hy_mask, "market_value"] *= (
                1 - cp.loc[hy_mask, "spread_duration"] * hy_shock
            )

            for sector, shock in sector_shocks.items():
                sector_mask = cp.get("gics_sector", pd.Series([""] * len(cp))) == sector
                if sector_mask.any():
                    cp.loc[sector_mask, "market_value"] *= (
                        1 - cp.loc[sector_mask, "spread_duration"] * shock
                    )

        inputs.shocks_applied["credit"] = {
            "ig_spread_shock_bps": shocks.get("ig_spread_shock_bps", 0),
            "hy_spread_shock_bps": shocks.get("hy_spread_shock_bps", 0),
        }
        logger.debug("Credit spread shock applied: IG=%dbps, HY=%dbps",
                     shocks.get("ig_spread_shock_bps", 0),
                     shocks.get("hy_spread_shock_bps", 0))
        return inputs

    # ── Equity price shock ────────────────────────────────────────────────────

    def _apply_equity_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Apply uniform and/or name-specific equity price shocks."""
        uniform_shock = shocks.get("equity_shock_pct", 0.0)
        name_shocks = shocks.get("name_specific_equity_shocks", {})

        cp = inputs.collateral_pool
        equity_mask = cp.get("asset_class", pd.Series([""] * len(cp))) == "Equity"

        if equity_mask.any() and "market_value" in cp.columns:
            cp.loc[equity_mask, "market_value"] *= (1 + uniform_shock)

            for ticker, shock_pct in name_shocks.items():
                name_mask = equity_mask & (cp.get("ticker", pd.Series([""] * len(cp))) == ticker)
                if name_mask.any():
                    # Override the uniform shock with name-specific shock
                    cp.loc[name_mask, "market_value"] /= (1 + uniform_shock)  # undo uniform
                    cp.loc[name_mask, "market_value"] *= (1 + shock_pct)       # apply name shock

        inputs.shocks_applied["equity"] = {"equity_shock_pct": uniform_shock}
        logger.debug("Equity shock applied: %.1f%%", uniform_shock * 100)
        return inputs

    # ── Liquidity shock ───────────────────────────────────────────────────────

    def _apply_liquidity_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Model market illiquidity via liquidation cost discounts."""
        bid_ask_mult = shocks.get("bid_ask_multiplier", 1.0)
        depth_reduction = shocks.get("market_depth_reduction_pct", 0.0)
        demand_surge = shocks.get("demand_surge_factor", 1.0)

        cp = inputs.collateral_pool
        if "bid_ask_spread_pct" in cp.columns and "market_value" in cp.columns:
            stressed_spread = cp["bid_ask_spread_pct"] * bid_ask_mult
            liquidation_discount = stressed_spread * (1 + depth_reduction)
            cp["market_value"] = cp["market_value"] * (1 - liquidation_discount / 2)

        # Increase borrow demand forecast to simulate demand surge
        lb = inputs.loan_book
        if "borrow_volume" in lb.columns:
            lb["borrow_volume"] = lb["borrow_volume"] * demand_surge

        inputs.shocks_applied["liquidity"] = {
            "bid_ask_multiplier": bid_ask_mult,
            "market_depth_reduction_pct": depth_reduction,
        }
        logger.debug("Liquidity shock applied: bid-ask x%.1f, depth -%d%%",
                     bid_ask_mult, depth_reduction * 100)
        return inputs

    # ── Counterparty shock ────────────────────────────────────────────────────

    def _apply_counterparty_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Remove specified counterparties from the loan book."""
        remove_ids = shocks.get("remove_counterparty_ids", [])
        spillover = shocks.get("recall_risk_spillover_factor", 1.0)

        lb = inputs.loan_book
        pre_count = len(lb)
        if "counterparty_id" in lb.columns and remove_ids:
            lb_mask = ~lb["counterparty_id"].isin(remove_ids)
            inputs.loan_book = lb[lb_mask].copy()
            removed = pre_count - len(inputs.loan_book)
            logger.info("Removed %d trades from %d counterparties", removed, len(remove_ids))

        # Elevate recall risk for remaining counterparties
        if "recall_prob" in inputs.collateral_pool.columns and spillover > 1.0:
            inputs.collateral_pool["recall_prob"] = (
                inputs.collateral_pool["recall_prob"] * spillover
            ).clip(upper=1.0)

        inputs.shocks_applied["counterparty"] = {
            "removed_counterparties": remove_ids,
            "recall_risk_spillover_factor": spillover,
        }
        return inputs

    # ── Regulatory shock ──────────────────────────────────────────────────────

    def _apply_regulatory_shock(self, inputs: StressedInputs, shocks: dict) -> StressedInputs:
        """Apply regulatory tightening — higher haircut floors, tighter concentration limits."""
        haircut_increase = shocks.get("haircut_floor_increase_pp", 0) / 100
        exclude_classes = shocks.get("exclude_asset_classes", [])

        cp = inputs.collateral_pool
        if "haircut" in cp.columns and haircut_increase > 0:
            cp["haircut"] = (cp["haircut"] + haircut_increase).clip(upper=1.0)

        if exclude_classes and "asset_class" in cp.columns:
            cp["eligible"] = ~cp["asset_class"].isin(exclude_classes)

        inputs.shocks_applied["regulatory"] = {
            "haircut_floor_increase_pp": shocks.get("haircut_floor_increase_pp", 0),
            "excluded_asset_classes": exclude_classes,
        }
        logger.debug("Regulatory shock applied: haircut +%dpp", shocks.get("haircut_floor_increase_pp", 0))
        return inputs
