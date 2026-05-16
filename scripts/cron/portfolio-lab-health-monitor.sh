#!/bin/bash
# cron-health-monitor.sh - Monitor system health and escalate issues
# Protected by cron_guard: load-gate (max 5), flock, 60s timeout, 1GB ulimit
CRON_GUARD_MEMORY_MB=1024 source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-health" 60; then
    cd /root/projects/portfolio-lab
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    python3 src/monitor/health.py 2>&1 | tee -a data/health.log
    cron_guard_end "pf-health" $?
fi
