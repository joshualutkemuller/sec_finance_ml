#!/usr/bin/env bash
# =============================================================================
# install_hooks.sh — copies all hooks from hooks_and_startup/hooks/ into
#                    .git/hooks/ and makes them executable.
# Run once after cloning the repository.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="${REPO_ROOT}/hooks_and_startup/hooks"
HOOKS_DST="${REPO_ROOT}/.git/hooks"

GREEN='\033[0;32m'
NC='\033[0m'

echo "Installing Git hooks from ${HOOKS_SRC} → ${HOOKS_DST}"

HOOKS=("pre-commit" "pre-push" "post-merge")

for HOOK in "${HOOKS[@]}"; do
    SRC="${HOOKS_SRC}/${HOOK}"
    DST="${HOOKS_DST}/${HOOK}"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST"
        chmod +x "$DST"
        echo -e "${GREEN}✓ Installed ${HOOK}${NC}"
    else
        echo "⚠ Hook not found: ${SRC} — skipping"
    fi
done

echo ""
echo -e "${GREEN}All hooks installed. Run 'git commit' or 'git push' to verify.${NC}"
