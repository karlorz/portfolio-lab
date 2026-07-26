#!/bin/bash
# cron-dashboard-generator.sh - Generate dashboard JSON files for Vite app
# Protected by cron_guard: load-gate (max 5), flock, 180s timeout, 3GB ulimit
# Batch II/IU DF3+DT2: align shell guard with Makefile + tasker + cron_compat (180s)
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-dashboard" 180; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    export REGIME_ALLOC_ENABLED=1

    set +e
    "$PYTHON_RUNTIME" src/dashboard/generator.py 2>&1 | tee -a data/dashboard.log
    dashboard_pipeline_status=("${PIPESTATUS[@]}")
    EXIT="$(cron_pipeline_exit "${dashboard_pipeline_status[0]}" "${dashboard_pipeline_status[1]}")"
    set -e

    if [ "$EXIT" -eq 0 ]; then
        echo "Dashboard data updated at $(date)"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-dashboard" "$EXIT" "$DUR"
    cron_guard_end "pf-dashboard" "$EXIT"
fi
