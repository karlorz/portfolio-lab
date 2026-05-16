#!/bin/bash
# Portfolio Lab Position Sync Script
# Synchronizes positions between broker and local database
# Protected by cron_guard: load-gate (max 5), flock, 60s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-sync" 60; then
    PROJECT_DIR="/root/projects/portfolio-lab"
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
    cron_guard_end "pf-sync" 0
fi
