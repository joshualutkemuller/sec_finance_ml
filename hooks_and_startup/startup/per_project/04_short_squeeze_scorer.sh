#!/usr/bin/env bash
# =============================================================================
# 04_short_squeeze_scorer.sh — Start the Short Squeeze Probability Scorer
# Usage: ./04_short_squeeze_scorer.sh [train|score|both]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${REPO_ROOT}/04_short_squeeze_scorer"
MODE="${1:-score}"

[ -d "${REPO_ROOT}/.venv" ] && source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Short Squeeze Probability Scorer | mode=${MODE} ==="
cd "$PROJECT"
python src/main.py --config config/config.yaml --mode "$MODE"
