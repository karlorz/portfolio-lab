#!/bin/sh
# Daily operational evidence collector for the live cursor-box Portfolio Lab.
# Strict BusyBox/POSIX /bin/sh only. Read-only: never restarts or stops
# services and never sources secret files; it only runs the evidence CLI and
# keeps a per-UTC-day stamp. The collector is invoked at most once per UTC
# day; the stamp is written only after the collector exits 0 so a later
# scheduled cycle can retry (exit 2 leaves no stamp).
#
# One RFC3339 UTC timestamp is computed here and passed to the CLI as
# --now, so the evidence day and the stamp day can never straddle midnight.
# flock exit 1 means the lock is contended (skip, exit 0); any other flock
# failure is a static error (exit 1, no stamp). A missing flock binary is
# detected before the lock file is created so nothing leaks on failure.
#
# Overrides (absolute paths; tests only): PLDE_PYTHON, PLDE_SCRIPT,
# PLDE_FLOCK (flock(1) binary), PLDE_LOCK_FILE, PLDE_STAMP_FILE,
# PLDE_ROOT. The collector's own PLDE_* overrides flow through the
# environment unchanged.
set -eu

PYTHON="${PLDE_PYTHON:-/home/box/.local/bin/python3}"
ROOT="${PLDE_ROOT:-/home/box/.local/share/portfolio-lab}"
SCRIPT="${PLDE_SCRIPT:-$ROOT/app/scripts/portfolio_lab_daily_evidence.py}"
FLOCK_BIN="${PLDE_FLOCK:-flock}"
LOCK_FILE="${PLDE_LOCK_FILE:-$ROOT/run/portfolio-lab-daily-evidence.lock}"
STAMP_FILE="${PLDE_STAMP_FILE:-$ROOT/run/portfolio-lab-daily-evidence-last-utc-day}"

NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TODAY="${NOW_UTC%T*}"

# Already collected for this UTC day: exit 0 without mutating anything.
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE" 2>/dev/null || true)" = "$TODAY" ]; then
    echo "portfolio-lab-daily-evidence: already collected for $TODAY" >&2
    exit 0
fi

# Probe flock before creating the lock file: a missing flock must fail
# statically without leaving an empty lock artifact behind.
if ! command -v "$FLOCK_BIN" >/dev/null 2>&1; then
    echo "portfolio-lab-daily-evidence: flock not found: $FLOCK_BIN" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOCK_FILE")" "$(dirname "$STAMP_FILE")"
exec 9>"$LOCK_FILE"
set +e
"$FLOCK_BIN" -n 9
flock_rc=$?
set -e
if [ "$flock_rc" -eq 1 ]; then
    echo "portfolio-lab-daily-evidence: lock held; skipping" >&2
    exit 0
fi
if [ "$flock_rc" -ne 0 ]; then
    echo "portfolio-lab-daily-evidence: flock failed ($flock_rc); not collecting" >&2
    exit 1
fi

export PORTFOLIO_LAB_ENABLE_ML=0
set +e
"$PYTHON" "$SCRIPT" --now "$NOW_UTC"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    printf '%s\n' "$TODAY" >"$STAMP_FILE"
    exit 0
fi
exit "$rc"