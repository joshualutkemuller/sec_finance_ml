"""
Graph Builder — Constructs a heterogeneous PyTorch Geometric graph from the lending book.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Graph schema:
  Node types:  'counterparty', 'security'
  Edge types:
    (counterparty, borrows, security)      — open borrow trades
    (security, pledged_to, counterparty)   — collateral pledges
    (counterparty, co_exposed, counterparty) — shared security overlap
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("torch-geometric not installed — graph construction unavailable")


@dataclass
class GraphMetadata:
    """Index mappings for graph nodes."""
    cp_index: dict[str, int]       # counterparty_id → node index
    sec_index: dict[str, int]      # ticker → node index
    cp_ids: list[str]              # ordered list of counterparty IDs
    sec_ids: list[str]             # ordered list of tickers


class LendingGraphBuilder:
    """
    Builds a PyG HeteroData graph from a securities lending book.

    Counterparty node features:
      - utilization_rate, recall_rate_30d, credit_spread, avg_tenor_days,
        total_notional_usd_bn, hhi, cp_count_securities

    Security node features:
      - short_interest_ratio, borrow_rate, days_to_cover, squeeze_score,
        liquidity_score, bid_ask_spread_pct, market_cap_bn
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["graph"]
        self.min_edge_usd = self.cfg.get("min_edge_weight_usd", 1_000_000)
        self.overlap_thresh = self.cfg.get("overlap_threshold", 0.20)

    def build(
        self,
        loan_book: pd.DataFrame,
        collateral: pd.DataFrame,
        counterparties: pd.DataFrame,
        securities: pd.DataFrame,
    ) -> tuple["HeteroData", GraphMetadata]:
        """Build the heterogeneous lending graph.

        Returns:
            (graph, metadata) tuple.
        """
        if not PYG_AVAILABLE:
            raise ImportError("Install torch-geometric: pip install torch-geometric")

        meta = self._build_index(loan_book, counterparties, securities)
        data = HeteroData()

        # ── Node features ──────────────────────────────────────────────────
        data["counterparty"].x = self._cp_features(counterparties, loan_book, meta)
        data["security"].x = self._sec_features(securities, meta)

        # ── Borrow edges (CP → SEC) ────────────────────────────────────────
        borrow_src, borrow_dst, borrow_w = self._borrow_edges(loan_book, meta)
        data["counterparty", "borrows", "security"].edge_index = torch.stack(
            [borrow_src, borrow_dst]
        )
        data["counterparty", "borrows", "security"].edge_attr = borrow_w.unsqueeze(1)

        # ── Collateral edges (SEC → CP) ────────────────────────────────────
        if collateral is not None and len(collateral) > 0:
            col_src, col_dst, col_w = self._collateral_edges(collateral, meta)
            data["security", "pledged_to", "counterparty"].edge_index = torch.stack(
                [col_src, col_dst]
            )
            data["security", "pledged_to", "counterparty"].edge_attr = col_w.unsqueeze(1)

        # ── Co-exposure edges (CP ↔ CP) ────────────────────────────────────
        co_src, co_dst, co_w = self._co_exposure_edges(loan_book, meta)
        if len(co_src) > 0:
            data["counterparty", "co_exposed", "counterparty"].edge_index = torch.stack(
                [co_src, co_dst]
            )
            data["counterparty", "co_exposed", "counterparty"].edge_attr = co_w.unsqueeze(1)

        logger.info(
            "Graph built: %d CP nodes, %d SEC nodes, %d borrow edges, %d co-exposure edges",
            len(meta.cp_ids),
            len(meta.sec_ids),
            len(borrow_src),
            len(co_src),
        )
        return data, meta

    # ── Private: index construction ───────────────────────────────────────

    def _build_index(
        self,
        loan_book: pd.DataFrame,
        counterparties: pd.DataFrame,
        securities: pd.DataFrame,
    ) -> GraphMetadata:
        cp_ids = sorted(
            set(loan_book["counterparty_id"].unique()) |
            set(counterparties.get("counterparty_id", pd.Series()).unique())
        )
        sec_ids = sorted(
            set(loan_book["ticker"].unique()) |
            set(securities.get("ticker", pd.Series()).unique())
        )
        return GraphMetadata(
            cp_index={cp: i for i, cp in enumerate(cp_ids)},
            sec_index={sec: i for i, sec in enumerate(sec_ids)},
            cp_ids=cp_ids,
            sec_ids=sec_ids,
        )

    def _cp_features(
        self,
        counterparties: pd.DataFrame,
        loan_book: pd.DataFrame,
        meta: GraphMetadata,
    ) -> torch.Tensor:
        n = len(meta.cp_ids)
        features = np.zeros((n, 7), dtype=np.float32)

        cp_agg = loan_book.groupby("counterparty_id").agg(
            total_notional=("notional_usd", "sum"),
            n_securities=("ticker", "nunique"),
            avg_tenor=("trade_term_days", "mean") if "trade_term_days" in loan_book else ("notional_usd", "count"),
        )

        for cp, idx in meta.cp_index.items():
            row = counterparties[counterparties["counterparty_id"] == cp]
            agg = cp_agg.loc[cp] if cp in cp_agg.index else None
            features[idx, 0] = float(row["utilization_rate"].iloc[0]) if len(row) and "utilization_rate" in row else 0.5
            features[idx, 1] = float(row["recall_rate_30d"].iloc[0]) if len(row) and "recall_rate_30d" in row else 0.05
            features[idx, 2] = float(row["credit_spread"].iloc[0]) if len(row) and "credit_spread" in row else 0.01
            features[idx, 3] = float(agg["avg_tenor"]) / 365 if agg is not None else 0.1
            features[idx, 4] = float(agg["total_notional"]) / 1e9 if agg is not None else 0.0
            n_sec = float(agg["n_securities"]) if agg is not None else 0
            features[idx, 5] = 1.0 / n_sec if n_sec > 0 else 0  # HHI proxy
            features[idx, 6] = n_sec / max(len(meta.sec_ids), 1)

        return torch.from_numpy(features)

    def _sec_features(
        self, securities: pd.DataFrame, meta: GraphMetadata
    ) -> torch.Tensor:
        n = len(meta.sec_ids)
        features = np.zeros((n, 7), dtype=np.float32)

        for sec, idx in meta.sec_index.items():
            row = securities[securities["ticker"] == sec] if "ticker" in securities.columns else pd.DataFrame()
            features[idx, 0] = float(row["short_interest_ratio"].iloc[0]) if len(row) and "short_interest_ratio" in row else 0.05
            features[idx, 1] = float(row["borrow_rate"].iloc[0]) if len(row) and "borrow_rate" in row else 0.003
            features[idx, 2] = float(row["days_to_cover"].iloc[0]) if len(row) and "days_to_cover" in row else 2.0
            features[idx, 3] = float(row["squeeze_score"].iloc[0]) if len(row) and "squeeze_score" in row else 0.1
            features[idx, 4] = float(row["liquidity_score"].iloc[0]) if len(row) and "liquidity_score" in row else 0.5
            features[idx, 5] = float(row["bid_ask_spread_pct"].iloc[0]) if len(row) and "bid_ask_spread_pct" in row else 0.001
            features[idx, 6] = float(row["market_cap_bn"].iloc[0]) if len(row) and "market_cap_bn" in row else 1.0

        return torch.from_numpy(features)

    def _borrow_edges(
        self, loan_book: pd.DataFrame, meta: GraphMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filtered = loan_book[loan_book.get("notional_usd", pd.Series(float("inf"), index=loan_book.index)) >= self.min_edge_usd]
        src, dst, weights = [], [], []
        for _, row in filtered.iterrows():
            cp_idx = meta.cp_index.get(row["counterparty_id"])
            sec_idx = meta.sec_index.get(row["ticker"])
            if cp_idx is not None and sec_idx is not None:
                src.append(cp_idx)
                dst.append(sec_idx)
                notional = row.get("notional_usd", 1e6)
                weights.append(float(notional) / 1e9)
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )

    def _collateral_edges(
        self, collateral: pd.DataFrame, meta: GraphMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst, weights = [], [], []
        for _, row in collateral.iterrows():
            sec_idx = meta.sec_index.get(row.get("ticker", ""))
            cp_idx = meta.cp_index.get(row.get("counterparty_id", ""))
            if sec_idx is not None and cp_idx is not None:
                src.append(sec_idx)
                dst.append(cp_idx)
                val = float(row.get("collateral_value_usd", row.get("market_value", 1e6)))
                weights.append(val / 1e9)
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )

    def _co_exposure_edges(
        self, loan_book: pd.DataFrame, meta: GraphMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build CP↔CP edges where they share ≥ overlap_threshold of the same securities."""
        cp_securities: dict[str, set] = (
            loan_book.groupby("counterparty_id")["ticker"]
            .apply(set)
            .to_dict()
        )
        src, dst, weights = [], [], []
        cp_list = list(cp_securities.keys())
        for i, cp_a in enumerate(cp_list):
            for cp_b in cp_list[i + 1:]:
                set_a, set_b = cp_securities[cp_a], cp_securities[cp_b]
                union = set_a | set_b
                if not union:
                    continue
                overlap = len(set_a & set_b) / len(union)
                if overlap >= self.overlap_thresh:
                    idx_a = meta.cp_index.get(cp_a)
                    idx_b = meta.cp_index.get(cp_b)
                    if idx_a is not None and idx_b is not None:
                        src.extend([idx_a, idx_b])
                        dst.extend([idx_b, idx_a])
                        weights.extend([overlap, overlap])
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )
