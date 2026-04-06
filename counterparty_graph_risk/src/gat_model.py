"""
Graph Attention Network (GAT) — Heterogeneous node embedding for lending graph.

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Architecture:
  - HeteroConv wrapping GATv2Conv per edge type
  - 3 layers, 4 attention heads per layer
  - Node embeddings used for contagion scoring and CP similarity
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv, Linear, to_hetero
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class LendingGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network for the lending book graph.

    Inputs:  HeteroData with node types 'counterparty' and 'security'
    Outputs: Node embeddings for both types (used downstream for scoring)
    """

    def __init__(
        self,
        cp_in_channels: int,
        sec_in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        if not PYG_AVAILABLE:
            raise ImportError("Install torch-geometric: pip install torch-geometric")
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # Input projections to hidden_channels
        self.cp_proj = Linear(cp_in_channels, hidden_channels)
        self.sec_proj = Linear(sec_in_channels, hidden_channels)

        # GAT convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("counterparty", "borrows", "security"): GATv2Conv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False,
                ),
                ("security", "pledged_to", "counterparty"): GATv2Conv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False,
                ),
                ("counterparty", "co_exposed", "counterparty"): GATv2Conv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=True,
                ),
            }, aggr="sum")
            self.convs.append(conv)

        # Output heads
        self.cp_contagion_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.fragility_head = nn.Linear(hidden_channels, 1)

    def encode(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Return node embeddings for all node types."""
        h = {
            "counterparty": F.elu(self.cp_proj(x_dict["counterparty"])),
            "security": F.elu(self.sec_proj(x_dict["security"])),
        }
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {k: F.elu(v) for k, v in h_new.items()}
            h = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in h.items()}
        return h

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cp_contagion_scores: [N_cp, 1] — P(contagion) per counterparty
            cp_embeddings:       [N_cp, hidden] — embeddings for similarity analysis
        """
        h = self.encode(x_dict, edge_index_dict)
        cp_emb = h["counterparty"]
        contagion_scores = self.cp_contagion_head(cp_emb)
        return contagion_scores, cp_emb


class GATTrainer:
    """Self-supervised trainer using structural graph properties as supervision signal."""

    def __init__(self, config: dict) -> None:
        self.cfg = config["gat_model"]
        self._model: LendingGAT | None = None

    def build_model(self, cp_feat_dim: int, sec_feat_dim: int) -> LendingGAT:
        self._model = LendingGAT(
            cp_in_channels=cp_feat_dim,
            sec_in_channels=sec_feat_dim,
            hidden_channels=self.cfg.get("hidden_channels", 64),
            num_heads=self.cfg.get("num_heads", 4),
            num_layers=self.cfg.get("num_layers", 3),
            dropout=self.cfg.get("dropout", 0.2),
        )
        return self._model

    def train(self, data: "HeteroData") -> None:
        """Self-supervised training via link prediction on co-exposure edges."""
        model = self._model
        if model is None:
            raise RuntimeError("Call build_model() first")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.get("learning_rate", 5e-4),
            weight_decay=self.cfg.get("weight_decay", 1e-5),
        )
        epochs = self.cfg.get("epochs", 150)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            scores, emb = model(data.x_dict, data.edge_index_dict)

            # Self-supervised loss: high-centrality nodes should have high contagion score
            # Use degree centrality as proxy label
            cp_degree = self._compute_degree(data)
            cp_degree_norm = (cp_degree - cp_degree.min()) / (cp_degree.max() - cp_degree.min() + 1e-8)
            loss = F.binary_cross_entropy(scores.squeeze(), cp_degree_norm)

            # Contrastive loss on embeddings (co-exposed CPs should be similar)
            if ("counterparty", "co_exposed", "counterparty") in data.edge_index_dict:
                co_edges = data["counterparty", "co_exposed", "counterparty"].edge_index
                src_emb = emb[co_edges[0]]
                dst_emb = emb[co_edges[1]]
                contrastive_loss = -F.cosine_similarity(src_emb, dst_emb).mean()
                loss = loss + 0.1 * contrastive_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 30 == 0:
                logger.info("GAT epoch %d/%d — loss: %.5f", epoch + 1, epochs, loss.item())

        logger.info("GAT training complete")

    def _compute_degree(self, data: "HeteroData") -> torch.Tensor:
        n = data["counterparty"].x.shape[0]
        degree = torch.zeros(n)
        for et in data.edge_types:
            if et[0] == "counterparty":
                src = data[et].edge_index[0]
                degree.scatter_add_(0, src, torch.ones(len(src)))
        return degree

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("GAT model saved to %s", path)

    def load(self, path: str, cp_feat_dim: int, sec_feat_dim: int) -> LendingGAT:
        model = self.build_model(cp_feat_dim, sec_feat_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
