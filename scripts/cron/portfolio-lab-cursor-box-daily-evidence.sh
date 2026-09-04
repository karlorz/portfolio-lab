#!/bin/sh
# Daily operational evidence collector for the live cursor-box Portfolio Lab.
# Strict BusyBox/POSIX /bin/sh only. Read-only: never restarts or stops
# services and never sources secret files; it only runs the evidence CLI and
# keeps a per-UTC-day stamp. The collector is invoked at most once per UTC
# day; the stamp is written only after the collector exits 0 so a later
# scheduled cycle can retry (exit 2 leaves no stamp).
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

TODAY="$(date -u +%Y-%m-%d)"

# Already collected for this UTC day: exit 0 without mutating anything.
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE" 2>/dev/null || true)" = "$TODAY" ]; then
    echo "portfolio-lab-daily-evidence: already collected for $TODAY" >&2
    exit 0
fi

mkdir -p "$(dirname "$LOCK_FILE")" "$(dirname "$STAMP_FILE")"
exec 9>"$LOCK_FILE"
if ! "$FLOCK_BIN" -n 9; then
    echo "portfolio-lab-daily-evidence: lock held; skipping" >&2
    exit 0
fi

export PORTFOLIO_LAB_ENABLE_ML=0
set +e
"$PYTHON" "$SCRIPT"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    printf '%s\n' "$TODAY" >"$STAMP_FILE"
    exit 0
fi
exit "$rc"