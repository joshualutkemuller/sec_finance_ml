"""
Network Analytics — Centrality, community detection, and fragility index.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class NetworkReport:
    """Complete network analytics report for a single snapshot."""
    fragility_index: float
    fiedler_value: float
    top_central_nodes: list[dict]         # [{"id": cp_id, "pagerank": float}, ...]
    community_assignments: dict[str, int]  # cp_id → community_id
    co_exposure_pairs: list[dict]          # high-overlap CP pairs
    is_fragile: bool
    n_communities: int
    network_density: float


class NetworkAnalytics:
    """
    Computes graph-theoretic risk metrics on the lending counterparty network.

    Fragility Index = 1 / λ₂(L)
      where λ₂ is the Fiedler value (algebraic connectivity) of the
      normalized graph Laplacian. Higher = more fragile.

    A declining Fiedler value over consecutive days is a leading indicator
    of increasing systemic fragility — like a slowly fraying safety net.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config["network_analytics"]
        self.fragility_threshold = self.cfg.get("fragility_alert_threshold", 5.0)
        self.top_n = self.cfg.get("centrality_top_n", 5)
        self.resolution = self.cfg.get("community_resolution", 1.0)

    def analyze(self, loan_book: pd.DataFrame, cp_ids: list[str]) -> NetworkReport:
        """Run full network analysis on the current lending book.

        Args:
            loan_book: DataFrame with counterparty_id, ticker, notional_usd columns.
            cp_ids: List of all counterparty IDs (ordered, matching GAT node indices).
        """
        G = self._build_networkx_graph(loan_book, cp_ids)

        fiedler = self._fiedler_value(G)
        fragility_index = 1.0 / (fiedler + 1e-9)
        is_fragile = fragility_index > self.fragility_threshold

        pagerank = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
        top_central = sorted(
            [{"id": k, "pagerank": round(v, 6)} for k, v in pagerank.items()],
            key=lambda x: x["pagerank"],
            reverse=True,
        )[:self.top_n]

        communities = self._detect_communities(G)
        co_exposure = self._high_overlap_pairs(loan_book)

        report = NetworkReport(
            fragility_index=round(fragility_index, 4),
            fiedler_value=round(fiedler, 6),
            top_central_nodes=top_central,
            community_assignments=communities,
            co_exposure_pairs=co_exposure,
            is_fragile=is_fragile,
            n_communities=len(set(communities.values())),
            network_density=round(nx.density(G), 6),
        )

        if is_fragile:
            logger.warning(
                "NETWORK FRAGILITY ALERT: fragility_index=%.2f (threshold=%.1f)",
                fragility_index,
                self.fragility_threshold,
            )

        return report

    def rolling_fragility(
        self, daily_books: list[tuple[pd.Timestamp, pd.DataFrame]], cp_ids: list[str]
    ) -> pd.DataFrame:
        """Compute fragility index over a time series of daily loan books."""
        records = []
        for dt, book in daily_books:
            G = self._build_networkx_graph(book, cp_ids)
            fiedler = self._fiedler_value(G)
            records.append({
                "date": dt,
                "fiedler_value": fiedler,
                "fragility_index": 1.0 / (fiedler + 1e-9),
                "network_density": nx.density(G),
                "n_edges": G.number_of_edges(),
            })
        return pd.DataFrame(records).set_index("date")

    # ── Private ────────────────────────────────────────────────────────────

    def _build_networkx_graph(
        self, loan_book: pd.DataFrame, cp_ids: list[str]
    ) -> nx.Graph:
        """Build undirected weighted CP-CP graph via shared security co-exposure."""
        G = nx.Graph()
        G.add_nodes_from(cp_ids)

        cp_securities: dict[str, set] = (
            loan_book.groupby("counterparty_id")["ticker"].apply(set).to_dict()
        )
        cp_notional: dict[str, float] = {}
        if "notional_usd" in loan_book.columns:
            cp_notional = loan_book.groupby("counterparty_id")["notional_usd"].sum().to_dict()

        cp_list = [cp for cp in cp_ids if cp in cp_securities]
        for i, cp_a in enumerate(cp_list):
            for cp_b in cp_list[i + 1:]:
                set_a, set_b = cp_securities[cp_a], cp_securities[cp_b]
                union = set_a | set_b
                if not union:
                    continue
                overlap = len(set_a & set_b) / len(union)
                if overlap > 0:
                    weight = overlap * np.sqrt(
                        cp_notional.get(cp_a, 1e6) * cp_notional.get(cp_b, 1e6)
                    ) / 1e12
                    G.add_edge(cp_a, cp_b, weight=float(weight), overlap=overlap)

        return G

    def _fiedler_value(self, G: nx.Graph) -> float:
        """Compute algebraic connectivity (Fiedler value λ₂) of the graph Laplacian."""
        if G.number_of_nodes() < 2:
            return 0.0
        if not nx.is_connected(G):
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            if G.number_of_nodes() < 2:
                return 0.0
        try:
            return float(nx.algebraic_connectivity(G, weight="weight", method="tracemin_lu"))
        except Exception:
            return float(nx.algebraic_connectivity(G, weight=None))

    def _detect_communities(self, G: nx.Graph) -> dict[str, int]:
        """Louvain community detection on the CP co-exposure graph."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                G, weight="weight", resolution=self.resolution, random_state=42
            )
            return partition
        except ImportError:
            logger.warning("python-louvain not installed — using connected components as communities")
            result: dict[str, int] = {}
            for cid, component in enumerate(nx.connected_components(G)):
                for node in component:
                    result[node] = cid
            return result

    def _high_overlap_pairs(
        self, loan_book: pd.DataFrame, threshold: float = 0.40
    ) -> list[dict]:
        """Return CP pairs with shared-security overlap above threshold."""
        cp_securities = (
            loan_book.groupby("counterparty_id")["ticker"].apply(set).to_dict()
        )
        pairs = []
        cp_list = list(cp_securities.keys())
        for i, cp_a in enumerate(cp_list):
            for cp_b in cp_list[i + 1:]:
                set_a, set_b = cp_securities[cp_a], cp_securities[cp_b]
                union = set_a | set_b
                if not union:
                    continue
                overlap = len(set_a & set_b) / len(union)
                if overlap >= threshold:
                    pairs.append({
                        "cp_a": cp_a,
                        "cp_b": cp_b,
                        "overlap_pct": round(overlap * 100, 1),
                        "shared_securities": sorted(set_a & set_b),
                    })
        return sorted(pairs, key=lambda x: x["overlap_pct"], reverse=True)
