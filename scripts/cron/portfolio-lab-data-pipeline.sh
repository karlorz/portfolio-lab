#!/bin/bash
# cron-data-pipeline.sh - Hourly data fetch and regime detection
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-data" 300; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    set +e
    bun run fetch-data 2>&1 | tee -a data/cron.log
    EXIT=${PIPESTATUS[0]}

    if [ "$EXIT" -eq 0 ]; then
        "$PYTHON_RUNTIME" -m src.monitor.performance_attribution report --save 2>&1 | tee -a data/attribution.log
        EXIT=${PIPESTATUS[0]}
    fi
    if [ "$EXIT" -eq 0 ]; then
        "$PYTHON_RUNTIME" -m src.strategy.adaptive_ensemble_weights update --regime normal 2>&1 | tee -a data/adaptive_weights.log
        EXIT=${PIPESTATUS[0]}
    fi
    set -e

    if [ -f data/.regime_trigger ]; then
        echo "REGIME_CHANGE: $(cat data/.regime_trigger)"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-data" "$EXIT" "$DUR"
    cron_guard_end "pf-data" "$EXIT"
fi
