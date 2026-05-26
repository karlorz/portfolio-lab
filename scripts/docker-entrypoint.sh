#!/bin/sh
set -e

# Setup cron schedule
crontab /app/crontab

# Start cron as PID 1 (exec replaces shell so signals are forwarded)
exec cron -f
