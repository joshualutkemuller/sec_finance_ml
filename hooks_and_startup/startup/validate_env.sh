#!/usr/bin/env bash
# =============================================================================
# validate_env.sh — Verifies all required environment variables are set.
# Exit code 0 = all present; Exit code 1 = missing vars detected.
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "── Validating environment variables ────────────────────────"

MISSING=0

check_var() {
    local VAR="$1"
    local REQUIRED="$2"  # "required" or "optional"
    if [ -z "${!VAR:-}" ]; then
        if [ "$REQUIRED" = "required" ]; then
            echo -e "${RED}  ✗ MISSING (required): ${VAR}${NC}"
            MISSING=1
        else
            echo -e "${YELLOW}  ⚠ missing (optional): ${VAR}${NC}"
        fi
    else
        echo -e "${GREEN}  ✓ ${VAR}${NC}"
    fi
}

echo ""
echo "PowerBI / Azure AD:"
check_var "POWERBI_CLIENT_ID"      "required"
check_var "POWERBI_CLIENT_SECRET"  "required"
check_var "POWERBI_TENANT_ID"      "required"
check_var "POWERBI_WORKSPACE_ID"   "required"

echo ""
echo "Email Alerts (SMTP):"
check_var "SMTP_HOST"                  "required"
check_var "SMTP_PORT"                  "required"
check_var "SMTP_USER"                  "required"
check_var "SMTP_PASSWORD"              "required"
check_var "ALERT_FROM"                 "required"
check_var "ALERT_EMAIL_RECIPIENTS"     "required"

echo ""
echo "Microsoft Teams:"
check_var "TEAMS_WEBHOOK_URL"      "required"

echo ""
echo "Databricks (optional — only required when source_type: databricks):"
check_var "DATABRICKS_HOST"        "optional"
check_var "DATABRICKS_TOKEN"       "optional"
check_var "DATABRICKS_HTTP_PATH"   "optional"

echo ""
if [ "$MISSING" -eq 1 ]; then
    echo -e "${RED}✗ One or more required variables are missing.${NC}"
    echo "   Copy hooks_and_startup/startup/.env.example to .env and fill in values."
    exit 1
else
    echo -e "${GREEN}✓ All required environment variables are set.${NC}"
    exit 0
fi
