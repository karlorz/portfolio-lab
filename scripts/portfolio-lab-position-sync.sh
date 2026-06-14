#!/bin/bash
# Position sync cron job for portfolio-lab
# Runs hourly to reconcile Alpaca positions with local database

PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"

cd "$PROJECT_DIR" || exit 1

# Check if Alpaca is configured
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
    echo "[$(date)] Alpaca API not configured, skipping position sync"
    exit 0
fi

# Run position sync
"$PYTHON_RUNTIME" src/broker/position_sync.py sync 2>&1 | tee -a data/cron_position_sync.log

# Keep log file manageable (last 1000 lines)
tail -n 1000 data/cron_position_sync.log > data/cron_position_sync.log.tmp
mv data/cron_position_sync.log.tmp data/cron_position_sync.log

echo "[$(date)] Position sync completed"
