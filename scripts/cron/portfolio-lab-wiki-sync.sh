#!/bin/bash
# cron-wiki-sync.sh - Sync research findings to wiki
# Protected by cron_guard: load-gate (max 5), flock, 120s timeout, 1GB ulimit
CRON_GUARD_MEMORY_MB=1024 source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-wiki-sync" 120; then
    cd /root/projects/portfolio-lab
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-/root/projects/portfolio-lab/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    "$PYTHON_RUNTIME" src/research/wiki_sync.py 2>&1 | tee -a data/wiki_sync.log
    echo "Wiki sync completed at $(date)"
    cron_guard_end "pf-wiki-sync" $?
fi
