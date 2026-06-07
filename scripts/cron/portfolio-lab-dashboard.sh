#!/bin/bash
# cron-dashboard-generator.sh - Generate dashboard JSON files for Vite app
# Protected by cron_guard: load-gate (max 5), flock, 120s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-dashboard" 120; then
    cd /root/projects/portfolio-lab
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-/root/projects/portfolio-lab/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    "$PYTHON_RUNTIME" src/dashboard/generator.py 2>&1 | tee -a data/dashboard.log
    echo "Dashboard data updated at $(date)"
    cron_guard_end "pf-dashboard" $?
fi
