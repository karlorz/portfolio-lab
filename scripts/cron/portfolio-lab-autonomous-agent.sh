#!/bin/bash
# cron-autonomous-portfolio-agent.sh - Autonomous Hermes agent for portfolio-lab
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
#
# NOTE: The LLM-powered agent is dispatched by Hermes cron, NOT by this script.
# This placeholder runs the lightweight pre-flight checks only.
# The cron_guard load-gate prevents dispatch when CPU is already saturated.
source /root/projects/portfolio-lab/scripts/cron_guard.sh

# Use lower load threshold for autonomous agent — it's the most expensive job
CRON_GUARD_MAX_LOAD=3 cron_guard_start "pf-autonomous" 300 || exit $?

cd /root/projects/portfolio-lab

echo "[$(date)] Autonomous agent cycle starting..."
echo "[$(date)] Checking for work items and triggers..."

PENDING_COUNT=$(find ~/wiki/projects/portfolio-lab/work -name "*.md" -type f 2>/dev/null | wc -l)
echo "[$(date)] Found $PENDING_COUNT work items"

if [ -f data/.regime_trigger ]; then
    echo "[$(date)] REGIME_TRIGGER_DETECTED: $(cat data/.regime_trigger)"
fi

if [ -f data/.health_report.json ]; then
    HEALTH_STATUS=$(python3 -c "import json; print(json.load(open('data/.health_report.json'))['status'])" 2>/dev/null || echo "unknown")
    echo "[$(date)] HEALTH_STATUS: $HEALTH_STATUS"
fi

echo "[$(date)] Autonomous cycle complete. Run with LLM via Hermes cron for full agent capabilities."
cron_guard_end "pf-autonomous" $?
