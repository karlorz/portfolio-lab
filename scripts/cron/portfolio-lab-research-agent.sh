#!/bin/bash
# cron-research-agent.sh - Research agent for regime analysis
# Protected by cron_guard: load-gate (max 5), flock, 300s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-research" 300; then
    cd /root/projects/portfolio-lab
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-/root/projects/portfolio-lab/scripts/python_runtime.sh}"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

    "$PYTHON_RUNTIME" src/research/agent.py 2>&1 | tee -a data/research.log

    for prompt in work/claude_*.md; do
        if [ -f "$prompt" ]; then
            echo "CLAUDE_PROMPT_READY: $prompt"
        fi
    done

    cron_guard_end "pf-research" $?
fi
