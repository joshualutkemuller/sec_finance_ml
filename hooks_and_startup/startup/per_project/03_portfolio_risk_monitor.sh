#!/usr/bin/env bash
# =============================================================================
# 03_portfolio_risk_monitor.sh — Start the Portfolio Concentration Risk Monitor
# Usage: ./03_portfolio_risk_monitor.sh [train|score|both]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${REPO_ROOT}/03_portfolio_risk_monitor"
MODE="${1:-score}"

[ -d "${REPO_ROOT}/.venv" ] && source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Portfolio Concentration Risk Monitor | mode=${MODE} ==="
cd "$PROJECT"
python src/main.py --config config/config.yaml --mode "$MODE"
