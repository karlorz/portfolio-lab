#!/bin/sh
set -e

# ─────────────────────────────────────────────────────────────
#  Portfolio-Lab Docker Entrypoint
#  Runs the project-local tasker service by default.
#  Set TASKER_ENTRYPOINT_MODE=cron for legacy crontab fallback.
# ─────────────────────────────────────────────────────────────

SHUTDOWN_PENDING=0

# ── Cleanup handler ─────────────────────────────────────────
cleanup() {
    SHUTDOWN_PENDING=1
    echo "[entrypoint] SIGTERM/SIGINT received — initiating graceful shutdown..."

    # Remove PID file
    if [ -f /app/data/pipeline.pid ]; then
        rm -f /app/data/pipeline.pid
        echo "[entrypoint] PID file removed"
    fi

    # Allow the foreground service or legacy cron mode a brief cleanup window.
    sleep 2

    echo "[entrypoint] Shutdown complete"
    exit 0
}

# ── Trap signals ────────────────────────────────────────────
# SIGTERM (docker stop) and SIGINT (docker stop with timeout)
# Using EXIT ensures cleanup even on unexpected exit paths.
trap cleanup TERM INT

ENTRYPOINT_MODE="${TASKER_ENTRYPOINT_MODE:-tasker}"

# ── Write PID file for healthcheck ──────────────────────────
echo $$ > /app/data/pipeline.pid
echo "[entrypoint] PID $$ written to /app/data/pipeline.pid"

if [ "$ENTRYPOINT_MODE" = "cron" ]; then
    # ── Legacy cron fallback ────────────────────────────────
    if [ -f /app/crontab ]; then
        crontab /app/crontab
        echo "[entrypoint] Crontab installed"
    else
        echo "[entrypoint] WARNING: No crontab found at /app/crontab"
    fi

    echo "[entrypoint] Starting legacy cron in foreground (TASKER_ENTRYPOINT_MODE=cron)..."
    exec cron -f
fi

if [ "$ENTRYPOINT_MODE" != "tasker" ]; then
    echo "[entrypoint] ERROR: unsupported TASKER_ENTRYPOINT_MODE=$ENTRYPOINT_MODE" >&2
    exit 64
fi

# ── Start tasker in foreground ──────────────────────────────
export CRON_BACKEND="${CRON_BACKEND:-tasker}"
export PORTFOLIO_LAB_ENABLE_ML="${PORTFOLIO_LAB_ENABLE_ML:-0}"
export PORTFOLIO_LAB_PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/app}"
echo "[entrypoint] Starting tasker service in foreground..."
exec /app/scripts/python_runtime.sh -m src.tasker.service --host 0.0.0.0 --port "${TASKER_PORT:-8000}"
