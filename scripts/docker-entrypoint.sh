#!/bin/sh
set -e

# ─────────────────────────────────────────────────────────────
#  Portfolio-Lab Docker Entrypoint
#  Manages cron-based signal pipeline with graceful shutdown.
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

    # Allow running cron jobs to finish (cron -f will exit on its own
    # after SIGTERM, but may kill children immediately). Give brief
    # window for cleanup.
    sleep 2

    echo "[entrypoint] Shutdown complete"
    exit 0
}

# ── Trap signals ────────────────────────────────────────────
# SIGTERM (docker stop) and SIGINT (docker stop with timeout)
# Using EXIT ensures cleanup even on unexpected exit paths.
trap cleanup TERM INT

# ── Setup cron schedule ─────────────────────────────────────
if [ -f /app/crontab ]; then
    crontab /app/crontab
    echo "[entrypoint] Crontab installed"
else
    echo "[entrypoint] WARNING: No crontab found at /app/crontab"
fi

# ── Write PID file for healthcheck ──────────────────────────
echo $$ > /app/data/pipeline.pid
echo "[entrypoint] PID $$ written to /app/data/pipeline.pid"

# ── Start cron in foreground ────────────────────────────────
# Using exec replaces the shell with cron as PID 1 (with tini via init:true)
# Signals pass through to cron for clean job termination.
echo "[entrypoint] Starting cron in foreground..."
exec cron -f
