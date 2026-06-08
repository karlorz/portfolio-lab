#!/bin/bash
# cron-research-agent.sh - Research agent for regime analysis
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-research" 300; then
    START=$(date +%s)
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    set +e
    "$PYTHON_RUNTIME" src/research/agent.py 2>&1 | tee -a data/research.log
    EXIT=${PIPESTATUS[0]}
    set -e

    for prompt in work/claude_*.md; do
        if [ -f "$prompt" ]; then
            echo "CLAUDE_PROMPT_READY: $prompt"
        fi
    done

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-research" "$EXIT" "$DUR"
    cron_guard_end "pf-research" "$EXIT"
fi
