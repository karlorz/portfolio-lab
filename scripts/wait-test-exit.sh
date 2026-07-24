#!/usr/bin/env bash
# wait-test-exit.sh — wait for a full `make test` exit stamp without 10m tool traps.
#
# make test writes data/test_last_exit.json on completion:
#   {"exit":0,"ts":"...","memory_class":false}
#
# This helper:
#   - records baseline mtime (or requires --since-start if no prior stamp)
#   - polls until the stamp is fresher than baseline OR max wait elapses
#   - FAILS FAST if no pytest/make-test process is running and stamp is still stale
#     (the common agent bug: waiting on a dead/orphaned suite)
#
# Usage:
#   ./scripts/wait-test-exit.sh                  # wait up to 3600s
#   ./scripts/wait-test-exit.sh --max-sec 600
#   ./scripts/wait-test-exit.sh --poll-sec 30
#   ./scripts/wait-test-exit.sh --baseline-mtime 1784829521
#
# Exit codes:
#   0  — fresh stamp with exit==0
#   1  — fresh stamp with non-zero suite exit, or dead suite / max wait
#   2  — usage / missing project paths

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXIT_JSON="${PORTFOLIO_LAB_TEST_EXIT_JSON:-$PROJECT_DIR/data/test_last_exit.json}"

MAX_SEC=3600
POLL_SEC=30
BASELINE_MTIME=""
QUIET=0

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \?//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-sec) MAX_SEC="${2:?}"; shift 2 ;;
    --poll-sec) POLL_SEC="${2:?}"; shift 2 ;;
    --baseline-mtime) BASELINE_MTIME="${2:?}"; shift 2 ;;
    --exit-json) EXIT_JSON="${2:?}"; shift 2 ;;
    --quiet|-q) QUIET=1; shift ;;
    --help|-h) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

log() {
  if [[ "$QUIET" -eq 0 ]]; then
    echo "$*"
  fi
}

mtime_of() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  # portable: GNU stat first, BSD fallback
  stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || echo 0
}

suite_procs() {
  # Count likely full-suite / pytest parents for this project (exclude this script).
  # Must never fail under `set -o pipefail` when grep matches nothing.
  local lines
  lines="$(pgrep -af 'pytest|make test|uv run pytest|timeout 3600' 2>/dev/null || true)"
  if [[ -z "$lines" ]]; then
    echo 0
    return 0
  fi
  printf '%s\n' "$lines" \
    | grep -F "$PROJECT_DIR" \
    | grep -v 'wait-test-exit' \
    | grep -v 'pgrep -af' \
    | grep -c . \
    || true
}

read_exit_code() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo ""
    return
  fi
  # Prefer python for robust JSON; fall back to sed.
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("exit", ""))
except Exception:
    print("")
PY
  else
    sed -n 's/.*"exit"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$path" | head -1
  fi
}

if [[ -z "$BASELINE_MTIME" ]]; then
  BASELINE_MTIME="$(mtime_of "$EXIT_JSON")"
fi

log "wait-test-exit: project=$PROJECT_DIR"
log "  exit_json=$EXIT_JSON"
log "  baseline_mtime=$BASELINE_MTIME"
log "  max_sec=$MAX_SEC poll_sec=$POLL_SEC"
log "  suite_procs_now=$(suite_procs)"

START_EPOCH=$(date +%s)
DEAD_STREAK=0
POLL_N=0

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_EPOCH))
  CUR_MTIME="$(mtime_of "$EXIT_JSON")"
  PROCS="$(suite_procs)"
  POLL_N=$((POLL_N + 1))

  if [[ "$CUR_MTIME" -gt "$BASELINE_MTIME" ]]; then
    CODE="$(read_exit_code "$EXIT_JSON")"
    log "wait-test-exit: fresh stamp after ${ELAPSED}s (poll $POLL_N) exit=${CODE:-?} mtime=$CUR_MTIME"
    if [[ "$CODE" == "0" ]]; then
      exit 0
    fi
    log "wait-test-exit: suite finished with non-zero exit=${CODE:-unknown}"
    exit 1
  fi

  if [[ "$PROCS" -eq 0 ]]; then
    DEAD_STREAK=$((DEAD_STREAK + 1))
  else
    DEAD_STREAK=0
  fi

  # Two consecutive empty process snapshots → suite is gone without a fresh stamp.
  if [[ "$DEAD_STREAK" -ge 2 ]]; then
    log "wait-test-exit: FAIL — no pytest/make-test process and exit json still stale"
    log "  (baseline_mtime=$BASELINE_MTIME current_mtime=$CUR_MTIME elapsed=${ELAPSED}s)"
    log "  Suite was abandoned or never wrote data/test_last_exit.json."
    log "  Unstick: do not keep polling. Use make test-gate or focused pytest."
    exit 1
  fi

  if [[ "$ELAPSED" -ge "$MAX_SEC" ]]; then
    log "wait-test-exit: FAIL — max wait ${MAX_SEC}s exceeded without fresh stamp"
    log "  suite_procs=$PROCS current_mtime=$CUR_MTIME"
    exit 1
  fi

  log "  poll $POLL_N (${ELAPSED}s): procs=$PROCS exit_json_fresh=no"
  sleep "$POLL_SEC"
done
