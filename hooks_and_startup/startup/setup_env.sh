#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — First-time environment setup for the FSIP ML platform.
# Creates a shared virtual environment and installs all project dependencies.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENV_DIR="${REPO_ROOT}/.venv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== FSIP ML Platform — Environment Setup ==="
echo "Repository root: ${REPO_ROOT}"

# ── 1. Python version check ───────────────────────────────────────────────────
PYTHON_MIN="3.10"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} detected${NC}"
else
    echo -e "${RED}✗ Python ${PYTHON_MIN}+ required. Found: ${PYTHON_VERSION}${NC}"
    exit 1
fi

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists at ${VENV_DIR}${NC}"
else
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# ── 3. Install per-project requirements ───────────────────────────────────────
PROJECTS=(
    "01_lending_rate_anomaly"
    "02_collateral_optimizer"
    "03_portfolio_risk_monitor"
    "04_short_squeeze_scorer"
    "05_financing_cost_forecaster"
    "securities_lending_optimizer"
)

for PROJECT in "${PROJECTS[@]}"; do
    REQ="${REPO_ROOT}/${PROJECT}/requirements.txt"
    if [ -f "$REQ" ]; then
        echo "Installing ${PROJECT}/requirements.txt..."
        pip install -q -r "$REQ"
        echo -e "${GREEN}✓ ${PROJECT} dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ No requirements.txt found for ${PROJECT} — skipping${NC}"
    fi
done

# Install dev tools
pip install -q flake8 mypy pytest

echo -e "${GREEN}✓ Dev tools installed (flake8, mypy, pytest)${NC}"

# ── 4. Load .env if present ───────────────────────────────────────────────────
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from ${ENV_FILE}..."
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo -e "${GREEN}✓ Environment variables loaded from .env${NC}"
else
    echo -e "${YELLOW}⚠ No .env file found. Copy hooks_and_startup/startup/.env.example to .env and fill in values.${NC}"
fi

# ── 5. Validate environment ───────────────────────────────────────────────────
echo ""
echo "Running environment validation..."
"${REPO_ROOT}/hooks_and_startup/startup/validate_env.sh" || true

echo ""
echo -e "${GREEN}=== Setup complete. Activate with: source .venv/bin/activate ===${NC}"
