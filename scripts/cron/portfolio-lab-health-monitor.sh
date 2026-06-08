#!/bin/bash
# cron-health-monitor.sh - Monitor system health and escalate issues
# Protected by cron_guard: load-gate (max 5), flock, 60s timeout, 1GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
CRON_GUARD_MEMORY_MB=1024 source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-health" 60; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    set +e
    "$PYTHON_RUNTIME" -m src.monitor.health_check 2>&1 | tee -a data/health.log
    health_pipeline_status=("${PIPESTATUS[@]}")
    set -e

    health_status="${health_pipeline_status[0]}"
    tee_status="${health_pipeline_status[1]}"
    if [ "$health_status" -ne 0 ]; then
        EXIT="$health_status"
    else
        EXIT="$tee_status"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-health" "$EXIT" "$DUR"
    cron_guard_end "pf-health" "$EXIT"
fi
