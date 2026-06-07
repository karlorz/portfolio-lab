#!/bin/bash
# cron-data-pipeline.sh - Hourly data fetch and regime detection
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-data" 300; then
    cd /root/projects/portfolio-lab
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-/root/projects/portfolio-lab/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    bun run fetch-data 2>&1 | tee -a data/cron.log
    "$PYTHON_RUNTIME" -m src.monitor.performance_attribution report --save 2>&1 | tee -a data/attribution.log
    "$PYTHON_RUNTIME" -m src.strategy.adaptive_ensemble_weights update --regime normal 2>&1 | tee -a data/adaptive_weights.log

    if [ -f data/.regime_trigger ]; then
        echo "REGIME_CHANGE: $(cat data/.regime_trigger)"
    fi

    cron_guard_end "pf-data" $?
fi
