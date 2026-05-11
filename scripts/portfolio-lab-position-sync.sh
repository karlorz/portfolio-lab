#!/bin/bash
# Position sync cron job for portfolio-lab
# Runs hourly to reconcile Alpaca positions with local database

cd /root/projects/portfolio-lab || exit 1

# Check if Alpaca is configured
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
    echo "[$(date)] Alpaca API not configured, skipping position sync"
    exit 0
fi

# Run position sync
python3 src/broker/position_sync.py sync 2>&1 | tee -a data/cron_position_sync.log

# Keep log file manageable (last 1000 lines)
tail -n 1000 data/cron_position_sync.log > data/cron_position_sync.log.tmp
mv data/cron_position_sync.log.tmp data/cron_position_sync.log

echo "[$(date)] Position sync completed"
