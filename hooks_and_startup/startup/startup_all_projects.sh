#!/usr/bin/env bash
# =============================================================================
# startup_all_projects.sh — Run all 5 FSIP ML projects in score mode.
# Suitable for a scheduled batch job (cron, Databricks job, etc.).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="${REPO_ROOT}/hooks_and_startup/startup/per_project"
LOG_DIR="${REPO_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

# Activate virtualenv if present
if [ -d "${REPO_ROOT}/.venv" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Validate environment before starting
echo "Validating environment..."
"${REPO_ROOT}/hooks_and_startup/startup/validate_env.sh"

echo ""
echo "=== FSIP Batch Run — ${TIMESTAMP} ==="
echo ""

FAILED_PROJECTS=()
PASSED_PROJECTS=()

run_project() {
    local SCRIPT="$1"
    local NAME="$2"
    echo "──────────────────────────────────────────────"
    echo "Starting: ${NAME}"
    LOG_FILE="${LOG_DIR}/${NAME}_${TIMESTAMP}.log"

    if bash "$SCRIPT" score >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✓ ${NAME} completed. Log: ${LOG_FILE}${NC}"
        PASSED_PROJECTS+=("$NAME")
    else
        echo -e "${RED}✗ ${NAME} FAILED. See log: ${LOG_FILE}${NC}"
        FAILED_PROJECTS+=("$NAME")
    fi
}

run_project "${SCRIPT_DIR}/01_lending_rate_anomaly.sh"    "01_lending_rate_anomaly"
run_project "${SCRIPT_DIR}/02_collateral_optimizer.sh"    "02_collateral_optimizer"
run_project "${SCRIPT_DIR}/03_portfolio_risk_monitor.sh"  "03_portfolio_risk_monitor"
run_project "${SCRIPT_DIR}/04_short_squeeze_scorer.sh"    "04_short_squeeze_scorer"
run_project "${SCRIPT_DIR}/05_financing_cost_forecaster.sh" "05_financing_cost_forecaster"

echo ""
echo "=== Batch Run Summary ==="
echo "Passed: ${#PASSED_PROJECTS[@]} / 5"
for P in "${PASSED_PROJECTS[@]:-}"; do echo -e "  ${GREEN}✓ ${P}${NC}"; done
if [ "${#FAILED_PROJECTS[@]}" -gt 0 ]; then
    echo "Failed: ${#FAILED_PROJECTS[@]}"
    for F in "${FAILED_PROJECTS[@]}"; do echo -e "  ${RED}✗ ${F}${NC}"; done
    exit 1
fi

echo -e "${GREEN}=== All projects completed successfully ===${NC}"
exit 0
