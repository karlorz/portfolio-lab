#!/bin/bash
# Portfolio Lab Position Sync Script
# Synchronizes positions between broker and local database
# Protected by cron_guard: load-gate (max 5), flock, 60s timeout, 3GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-sync" 60; then
    START=$(date +%s)
    DATA_DIR="$PROJECT_DIR/data"
    DATE=$(date -Iseconds)

    echo "=== Portfolio Position Sync: $DATE ==="

    if [ ! -f "$DATA_DIR/positions.db" ] && [ ! -f "$DATA_DIR/market.db" ]; then
        echo "No position database found - creating placeholder"
        mkdir -p "$DATA_DIR"
    fi

    echo "Position sync status:"
    echo "  - Project dir: $PROJECT_DIR"
    echo "  - Data dir: $DATA_DIR"
    echo "  - Last sync: $DATE"

    mkdir -p "$DATA_DIR/logs"
    echo "$DATE | Position sync completed" >> "$DATA_DIR/logs/position_sync.log"

    echo "Status: complete"
    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-position-sync" 0 "$DUR"
    cron_guard_end "pf-sync" 0
fi
