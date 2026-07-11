#!/bin/bash
# cron-wiki-sync.sh - Sync research findings to wiki
# Protected by cron_guard: load-gate (max 5), flock, 120s timeout, 1GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
CRON_GUARD_MEMORY_MB=1024 source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-wiki-sync" 120; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    set +e
    "$PYTHON_RUNTIME" src/research/wiki_sync.py 2>&1 | tee -a data/wiki_sync.log
    wiki_sync_pipeline_status=("${PIPESTATUS[@]}")
    EXIT="$(cron_pipeline_exit "${wiki_sync_pipeline_status[0]}" "${wiki_sync_pipeline_status[1]}")"
    set -e

    if [ "$EXIT" -eq 0 ]; then
        echo "Wiki sync completed at $(date)"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-wiki-sync" "$EXIT" "$DUR"
    cron_guard_end "pf-wiki-sync" "$EXIT"
fi
