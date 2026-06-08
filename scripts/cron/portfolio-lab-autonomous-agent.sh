#!/bin/bash
# cron-autonomous-portfolio-agent.sh - Autonomous Hermes agent for portfolio-lab
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
#
# NOTE: The LLM-powered agent is dispatched by Hermes cron, NOT by this script.
# This placeholder runs the lightweight pre-flight checks only.
# The cron_guard load-gate prevents dispatch when CPU is already saturated.
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

# Use lower load threshold for autonomous agent — it's the most expensive job
CRON_GUARD_MAX_LOAD=3
if cron_guard_start "pf-autonomous" 300; then
START=$(date +%s)

cd "$PROJECT_DIR"
PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"

echo "[$(date)] Autonomous agent cycle starting..."
echo "[$(date)] Checking for work items and triggers..."

PENDING_COUNT=$(find ~/wiki/projects/portfolio-lab/work -name "*.md" -type f 2>/dev/null | wc -l)
echo "[$(date)] Found $PENDING_COUNT work items"

if [ -f data/.regime_trigger ]; then
    echo "[$(date)] REGIME_TRIGGER_DETECTED: $(cat data/.regime_trigger)"
fi

if [ -f data/.health_report.json ]; then
    HEALTH_STATUS=$("$PYTHON_RUNTIME" -c "import json; print(json.load(open('data/.health_report.json'))['status'])" 2>/dev/null || echo "unknown")
    echo "[$(date)] HEALTH_STATUS: $HEALTH_STATUS"
fi

echo "[$(date)] Autonomous cycle complete. Run with LLM via Hermes cron for full agent capabilities."
END=$(date +%s)
DUR=$((END - START))
record_hermes_cron_status "portfolio-lab-autonomous-agent" 0 "$DUR"
cron_guard_end "pf-autonomous" 0
fi
