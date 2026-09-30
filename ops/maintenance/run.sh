#!/bin/bash
# run.sh — Entrypoint for S1/S3 maintenance playbooks on cursor-box.
#
# Manual Invocation:
#   ops/maintenance/run.sh daily [--dry-run]
#   ops/maintenance/run.sh weekly [--dry-run]
#
# Box-user crontab (S2) already registers this entrypoint. Repo dual-mode
# (Makefile / crontab / src/cron_compat.py) stays deferred: this is box-level
# ops, not a Tasker job. After grok exits, report.py always writes
# last-<cycle>.json and posts fail-only wiki_capture.

set -euo pipefail

# ── Argument Parsing ───────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 daily|weekly [--dry-run]" >&2
    exit 2
}

if [ "$#" -lt 1 ]; then
    usage
fi

CYCLE="$1"
shift

DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            usage
            ;;
    esac
done

case "$CYCLE" in
    daily)
        JOB_NAME="ops-maint-daily"
        TIMEOUT_SECS=1200
        ;;
    weekly)
        JOB_NAME="ops-maint-weekly"
        TIMEOUT_SECS=2700
        ;;
    *)
        usage
        ;;
esac

# ── Environment & Path Setup ───────────────────────────────────────────────
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

# Export project directory for cron_guard and portfolio-lab tooling
export PORTFOLIO_LAB_PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-$REPO_ROOT}"
export PORTFOLIO_LAB_ENABLE_ML=0

# Allow overriding PATH or grok binary path (for testing and flexible deployment)
# Prepend $HOME/.local/bin to PATH unless GROK_BIN or OPS_SKIP_PATH_PREPEND is set
if [ -z "${OPS_SKIP_PATH_PREPEND:-}" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Allow environment override for the ops maintenance directory (used by tests)
OPS_DIR="${OPS_MAINT_DIR:-$SCRIPT_DIR}"
RUNBOOK="$OPS_DIR/${CYCLE}.md"

# ── Static Validation (Fail Closed) ────────────────────────────────────────
if ! command -v grok >/dev/null 2>&1; then
    echo "ERROR: 'grok' executable not found in PATH ($PATH)" >&2
    exit 1
fi

if [ ! -f "$RUNBOOK" ]; then
    echo "ERROR: Runbook file not found: $RUNBOOK" >&2
    exit 1
fi

# ── Source cron_guard ──────────────────────────────────────────────────────
GUARD_LIB="$REPO_ROOT/scripts/cron/cron_guard.sh"
if [ ! -f "$GUARD_LIB" ]; then
    echo "ERROR: cron_guard library not found at $GUARD_LIB" >&2
    exit 1
fi
# shellcheck source=scripts/cron/cron_guard.sh
source "$GUARD_LIB"

# ── Dry-Run Mode ───────────────────────────────────────────────────────────
DATA_DIR="${OPS_MAINT_DATA_DIR:-$REPO_ROOT/data/ops-maintenance}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TRANSCRIPT="$DATA_DIR/run-${CYCLE}-${TIMESTAMP}.json"
LAST_REPORT="$DATA_DIR/last-${CYCLE}.json"
REPORT_PY="$SCRIPT_DIR/report.py"

# Model route is configurable: cursor-box has no default-route auth (xAI 401),
# so it must run via its configured gateway model (flash-max via NEW_API key in
# ~/.grok/config.toml). Override with OPS_MAINT_MODEL on hosts whose default
# model route is authenticated.
OPS_MAINT_MODEL="${OPS_MAINT_MODEL:-flash-max}"

WOULD_RUN_CMD=(grok --agent minimal --model "$OPS_MAINT_MODEL" --prompt-file "$RUNBOOK" --always-approve --max-turns "${OPS_MAINT_MAX_TURNS:-80}" --output-format json --disable-web-search)
REPORT_CMD=("${OPS_MAINT_PYTHON:-python3}" "$REPORT_PY" --cycle "$CYCLE" --transcript "$TRANSCRIPT" --agent-exit 0 --output-dir "$DATA_DIR")

if [ ! -f "$REPORT_PY" ]; then
    echo "ERROR: Reporter not found: $REPORT_PY" >&2
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== Ops Maintenance Dry-Run ==="
    echo "Cycle: $CYCLE"
    echo "Job Name: $JOB_NAME"
    echo "Timeout: ${TIMEOUT_SECS}s"
    echo "Project Dir: $PORTFOLIO_LAB_PROJECT_DIR"
    echo "Runbook: $RUNBOOK"
    echo "Grok Binary: $(command -v grok)"
    echo "Transcript Destination: $TRANSCRIPT"
    echo "Report Destination: $LAST_REPORT"
    echo "Command: ${WOULD_RUN_CMD[*]}"
    echo "Reporter: ${REPORT_CMD[*]}"
    exit 0
fi

# ── Real Execution (Protected by cron_guard) ───────────────────────────────
mkdir -p "$DATA_DIR"

# Prune transcripts older than 14 days
find "$DATA_DIR" -type f -name "run-*.json" -mtime +14 -delete 2>/dev/null || true

if cron_guard_start "$JOB_NAME" "$TIMEOUT_SECS"; then
    EXEC_RC=0
    set +e
    "${WOULD_RUN_CMD[@]}" > "$TRANSCRIPT" 2>&1
    AGENT_EXIT=$?
    "${OPS_MAINT_PYTHON:-python3}" "$REPORT_PY" \
        --cycle "$CYCLE" \
        --transcript "$TRANSCRIPT" \
        --agent-exit "$AGENT_EXIT" \
        --output-dir "$DATA_DIR"
    REPORT_RC=$?
    set -e

    if [ "$AGENT_EXIT" -ne 0 ]; then
        echo "ERROR: grok agent execution failed with exit code $AGENT_EXIT" >&2
        EXEC_RC="$AGENT_EXIT"
    fi
    if [ "$REPORT_RC" -ne 0 ]; then
        echo "ERROR: ops-maintenance reporter failed with exit code $REPORT_RC" >&2
        EXEC_RC="$REPORT_RC"
    fi

    cron_guard_end "$JOB_NAME" "$EXEC_RC"
    exit "$EXEC_RC"
fi
