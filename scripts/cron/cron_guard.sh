#!/bin/bash
# cron_guard.sh — Shared guard library for all portfolio-lab cron jobs.
# Source this at the top of every cron script to get:
#   - Load-based gating (defer if loadavg > threshold)
#   - flock-based overlap prevention
#   - Hard timeout (SIGKILL after grace period)
#   - Memory limit (ulimit -v)
#
# Usage:
#   source "$(dirname "$0")/cron_guard.sh"
#   if cron_guard_start "my-job" 300; then
#       # ... do work ...
#       cron_guard_end "my-job" $?
#   fi

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
GUARD_MAX_LOAD=${CRON_GUARD_MAX_LOAD:-5}
GUARD_DEFAULT_TIMEOUT=${CRON_GUARD_DEFAULT_TIMEOUT:-600}
GUARD_MEMORY_MB=${CRON_GUARD_MEMORY_MB:-3072}  # 3GB default
GUARD_LOCK_DIR=${CRON_GUARD_LOCK_DIR:-/tmp/portfolio-lab-locks}
GUARD_PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-${PROJECT_DIR:-/root/projects/portfolio-lab}}"
GUARD_LOG_DIR=${CRON_GUARD_LOG_DIR:-$GUARD_PROJECT_DIR/data}

mkdir -p "$GUARD_LOCK_DIR"

# ── cron_guard_start <job_name> [timeout_seconds] ──────────────────────────
# Returns 0 if safe to proceed, exits with status 75 (TEMPFAIL) if deferred.
# Callers should use: if cron_guard_start "name" 300; then ... fi
cron_guard_start() {
    local job_name="${1:-unknown}"
    local timeout_secs="${2:-$GUARD_DEFAULT_TIMEOUT}"
    local lock_file="$GUARD_LOCK_DIR/${job_name}.lock"
    local start_time
    start_time=$(date +%s)

    # ── Layer 1: Load-based gating ─────────────────────────────────────
    local load1=""
    if [ -r /proc/loadavg ]; then
        load1=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    fi
    if [ -n "$load1" ] && [ "$load1" -gt "$GUARD_MAX_LOAD" ]; then
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name DEFERRED (load ${load1} > ${GUARD_MAX_LOAD})"
        exit 75  # TEMPFAIL — caller can retry later
    fi

    # ── Layer 2: Overlap prevention (flock) ────────────────────────────
    exec {lock_fd}>"$lock_file"
    if command -v flock >/dev/null 2>&1; then
        if ! flock -n "$lock_fd"; then
            echo "[$(date -Iseconds)] CRON_GUARD: $job_name SKIPPED (previous run still active, lock held)"
            exit 0  # Not an error — previous cycle still running
        fi
    elif command -v python3 >/dev/null 2>&1; then
        # Portable fallback for hosts without flock(1) (e.g. macOS): take
        # the same flock(2) lock on the inherited lock fd via Python 3, so
        # contention with any fcntl.flock holder is still detected.
        if ! GUARD_LOCK_FD="$lock_fd" python3 -c '
import fcntl
import os
import sys

try:
    fcntl.flock(int(os.environ["GUARD_LOCK_FD"]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
' 2>/dev/null; then
            echo "[$(date -Iseconds)] CRON_GUARD: $job_name SKIPPED (previous run still active, lock held)"
            exit 0  # Not an error — previous cycle still running
        fi
    else
        local fallback_lock_dir="${lock_file}.d"
        if ! mkdir "$fallback_lock_dir" 2>/dev/null; then
            echo "[$(date -Iseconds)] CRON_GUARD: $job_name SKIPPED (previous run still active, lock held)"
            exit 0  # Not an error — previous cycle still running
        fi
        export GUARD_LOCK_FALLBACK_DIR="$fallback_lock_dir"
    fi

    # ── Layer 3: Memory limit ──────────────────────────────────────────
    ulimit -v $((GUARD_MEMORY_MB * 1024)) 2>/dev/null || true

    # ── Layer 4: Hard timeout (background watchdog) ─────────────────────
    # Spawn a subshell that SIGTERMs the parent after timeout_secs,
    # then SIGKILLs 30s later if still alive.
    (
        trap 'exit 0' TERM INT
        if [[ "$lock_fd" =~ ^[0-9]+$ ]]; then
            eval "exec ${lock_fd}>&-" 2>/dev/null || true
        fi
        # Do not hold the parent's captured stdout/stderr pipes open after
        # the parent exits: drop them when they are pipes (e.g. a test
        # harness), keeping cron-log/terminal output otherwise.
        exec </dev/null
        if [ -p /dev/fd/1 ] || [ -p /dev/fd/2 ]; then
            exec >/dev/null 2>&1
        fi
        sleep "$timeout_secs" </dev/null >/dev/null 2>&1 &
        watchdog_sleep_pid=$!
        wait "$watchdog_sleep_pid" || exit 0
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name TIMEOUT after ${timeout_secs}s — sending SIGTERM" >&2
        kill -TERM $$ 2>/dev/null || true
        sleep 30
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name FORCE KILL after $((timeout_secs + 30))s — sending SIGKILL" >&2
        kill -KILL $$ 2>/dev/null || true
    ) &
    GUARD_WATCHDOG_PID=$!

    # Export state for cron_guard_end
    export GUARD_JOB_NAME="$job_name"
    export GUARD_START_TIME="$start_time"
    export GUARD_LOCK_FD="$lock_fd"
    export GUARD_TIMEOUT="$timeout_secs"

    echo "[$(date -Iseconds)] CRON_GUARD: $job_name STARTED (timeout=${timeout_secs}s, mem_limit=${GUARD_MEMORY_MB}MB, max_load=${GUARD_MAX_LOAD})"
    return 0
}

# ── cron_guard_end <job_name> <exit_code> ──────────────────────────────────
# Clean up: kill watchdog, release lock, log duration.
cron_guard_end() {
    local job_name="${1:-$GUARD_JOB_NAME}"
    local exit_code="${2:-0}"
    local end_time
    end_time=$(date +%s)
    local start_time="${GUARD_START_TIME:-$end_time}"
    local duration=$((end_time - start_time))

    # Kill the watchdog if still running
    if [ -n "${GUARD_WATCHDOG_PID:-}" ]; then
        if command -v pkill >/dev/null 2>&1; then
            pkill -TERM -P "$GUARD_WATCHDOG_PID" 2>/dev/null || true
        fi
        kill "$GUARD_WATCHDOG_PID" 2>/dev/null || true
        wait "$GUARD_WATCHDOG_PID" 2>/dev/null || true
    fi

    # Release flock
    if [ -n "${GUARD_LOCK_FD:-}" ]; then
        if [[ "$GUARD_LOCK_FD" =~ ^[0-9]+$ ]]; then
            eval "exec ${GUARD_LOCK_FD}>&-" 2>/dev/null || true
        fi
    fi
    if [ -n "${GUARD_LOCK_FALLBACK_DIR:-}" ]; then
        rmdir "$GUARD_LOCK_FALLBACK_DIR" 2>/dev/null || true
    fi

    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name COMPLETED (${duration}s)"
    elif [ "$exit_code" -eq 75 ]; then
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name DEFERRED (${duration}s)"
    else
        echo "[$(date -Iseconds)] CRON_GUARD: $job_name FAILED exit=$exit_code (${duration}s)" >&2
    fi

    return "$exit_code"
}
