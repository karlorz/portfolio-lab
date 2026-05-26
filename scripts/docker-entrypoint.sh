#!/bin/sh
set -e

# Setup cron schedule
crontab /app/crontab

# Write PID file for healthcheck
echo $$ > /app/data/pipeline.pid

# Start cron as PID 1 (exec replaces shell so signals are forwarded)
exec cron -f
