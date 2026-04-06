"""
Counterparty Graph Risk — Main Entry Point

Author:  Joshua Lutkemuller
         Managing Director, Securities Finance & Agency Lending
Created: 2026

Modes:
  build  → build graph from current loan book (no ML, just graph analytics)
  train  → train GAT model on graph
  score  → run full pipeline: graph + GAT scoring + network analytics + alerts
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterparty Graph Risk — Joshua Lutkemuller, MD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["build", "train", "score"], default="score")
    args = parser.parse_args()

    config = _load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_id = str(uuid.uuid4())
    logger = logging.getLogger("graph_risk.main")
    logger.info("run_id=%s | mode=%s | Counterparty Graph Risk", run_id, args.mode)

    from graph_builder import LendingGraphBuilder
    from network_analytics import NetworkAnalytics

    loan_book = pd.read_csv(config["data"]["loan_book_path"])
    collateral = pd.read_csv(config["data"]["collateral_path"]) if Path(config["data"]["collateral_path"]).exists() else pd.DataFrame()
    counterparties = pd.read_csv(config["data"]["counterparty_path"])
    securities = pd.DataFrame()  # supplement with market data if available

    builder = LendingGraphBuilder(config)
    graph, meta = builder.build(loan_book, collateral, counterparties, securities)

    analytics = NetworkAnalytics(config)
    report = analytics.analyze(loan_book, meta.cp_ids)

    logger.info(
        "Network: fragility=%.3f, fiedler=%.6f, communities=%d, density=%.4f, fragile=%s",
        report.fragility_index,
        report.fiedler_value,
        report.n_communities,
        report.network_density,
        report.is_fragile,
    )

    if args.mode in ("train", "score"):
        from gat_model import GATTrainer
        trainer = GATTrainer(config)
        cp_feat_dim = graph["counterparty"].x.shape[1]
        sec_feat_dim = graph["security"].x.shape[1]
        trainer.build_model(cp_feat_dim, sec_feat_dim)

        if args.mode == "train":
            trainer.train(graph)
            trainer.save(config["gat_model"]["model_path"])
        else:
            model_path = config["gat_model"]["model_path"]
            if Path(model_path).exists():
                model = trainer.load(model_path, cp_feat_dim, sec_feat_dim)
                scores, embeddings = model(graph.x_dict, graph.edge_index_dict)
                import torch
                score_arr = scores.squeeze().detach().numpy()
                for i, cp_id in enumerate(meta.cp_ids):
                    logger.info("  CP %-30s contagion_score=%.4f", cp_id, score_arr[i])

    # Write audit log
    log_path = Path(config["logging"]["audit_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "fragility_index": report.fragility_index,
            "fiedler_value": report.fiedler_value,
            "is_fragile": report.is_fragile,
            "n_communities": report.n_communities,
        }
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
