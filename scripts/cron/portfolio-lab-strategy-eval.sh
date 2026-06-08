#!/bin/bash
# cron-strategy-eval.sh - Periodic strategy evaluation (paper trading)
# Protected by cron_guard: load-gate (max 5), flock, 600s timeout, 3GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-eval" 600; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    export ALPHALAB_MODE="${ALPHALAB_MODE:-paper}"
    export REGIME_ALLOC_ENABLED=1

    set +e
    "$PYTHON_RUNTIME" src/strategy/evaluator.py 2>&1 | tee -a data/eval.log
    EXIT=${PIPESTATUS[0]}
    set -e

    if [ -f data/.promote_to_live ]; then
        echo "PROMOTION_CANDIDATE: $(cat data/.promote_to_live)"
    fi
    if [ -f data/.kill_switch_paper ]; then
        echo "KILL_SWITCH: $(cat data/.kill_switch_paper)"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-eval" "$EXIT" "$DUR"
    cron_guard_end "pf-eval" "$EXIT"
fi
