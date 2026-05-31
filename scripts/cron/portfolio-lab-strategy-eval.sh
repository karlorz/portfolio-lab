#!/bin/bash
# cron-strategy-eval.sh - Periodic strategy evaluation (paper trading)
# Protected by cron_guard: load-gate (max 5), flock, 600s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-eval" 600; then
    cd /root/projects/portfolio-lab
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    export ALPHALAB_MODE="${ALPHALAB_MODE:-paper}"
    export REGIME_ALLOC_ENABLED=1

    python3 src/strategy/evaluator.py 2>&1 | tee -a data/eval.log

    if [ -f data/.promote_to_live ]; then
        echo "PROMOTION_CANDIDATE: $(cat data/.promote_to_live)"
    fi
    if [ -f data/.kill_switch_paper ]; then
        echo "KILL_SWITCH: $(cat data/.kill_switch_paper)"
    fi

    cron_guard_end "pf-eval" $?
fi
