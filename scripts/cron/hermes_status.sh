#!/bin/bash
# Helper for Hermes no-agent cron wrappers to update data/cron_status.json.

cron_status_from_exit() {
    local exit_code="${1:-0}"

    if [ "$exit_code" -eq 0 ]; then
        echo "ok"
    elif [ "$exit_code" -eq 124 ]; then
        echo "timeout"
    elif [ "$exit_code" -eq 137 ]; then
        echo "oom"
    else
        echo "error"
    fi
}

record_hermes_cron_status() {
    local job_name="${1:?job name required}"
    local exit_code="${2:-0}"
    local duration_seconds="${3:-0}"
    local backend="${CRON_BACKEND:-hermes}"

    if [ "$backend" != "hermes" ]; then
        return 0
    fi

    local project_dir="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
    local python_runtime="${PYTHON_RUNTIME:-$project_dir/scripts/python_runtime.sh}"
    local status
    status="$(cron_status_from_exit "$exit_code")"

    "$python_runtime" scripts/cron_update.py "$job_name" "$status" "$duration_seconds" hermes || {
        echo "WARN: failed to update cron status for $job_name" >&2
        return 0
    }
}
