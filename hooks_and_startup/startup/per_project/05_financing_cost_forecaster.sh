#!/usr/bin/env bash
# =============================================================================
# 05_financing_cost_forecaster.sh — Start the Repo & Financing Cost Forecaster
# Usage: ./05_financing_cost_forecaster.sh [train|score|both]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${REPO_ROOT}/05_financing_cost_forecaster"
MODE="${1:-score}"

[ -d "${REPO_ROOT}/.venv" ] && source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Repo & Financing Cost Forecaster | mode=${MODE} ==="
cd "$PROJECT"
python src/main.py --config config/config.yaml --mode "$MODE"
