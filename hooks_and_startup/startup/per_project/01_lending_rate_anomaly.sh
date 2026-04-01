#!/usr/bin/env bash
# =============================================================================
# 01_lending_rate_anomaly.sh — Start the Lending Rate Anomaly Detector
# Usage: ./01_lending_rate_anomaly.sh [train|score|both]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${REPO_ROOT}/01_lending_rate_anomaly"
MODE="${1:-score}"

[ -d "${REPO_ROOT}/.venv" ] && source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Lending Rate Anomaly Detector | mode=${MODE} ==="
cd "$PROJECT"
python src/main.py --config config/config.yaml --mode "$MODE"
