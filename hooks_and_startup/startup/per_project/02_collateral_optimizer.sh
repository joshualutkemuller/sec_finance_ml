#!/usr/bin/env bash
# =============================================================================
# 02_collateral_optimizer.sh — Start the Collateral Optimizer
# Usage: ./02_collateral_optimizer.sh [full|score_only|optimize_only]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${REPO_ROOT}/02_collateral_optimizer"
MODE="${1:-full}"

[ -d "${REPO_ROOT}/.venv" ] && source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Collateral Optimizer & Substitution Recommender | mode=${MODE} ==="
cd "$PROJECT"
python src/main.py --config config/config.yaml --mode "$MODE"
