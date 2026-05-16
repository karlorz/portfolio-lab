#!/bin/bash
# cron-app-build.sh - Build Vite app with latest dashboard data
# Protected by cron_guard: load-gate (max 5), flock, 600s timeout, 3GB ulimit
source /root/projects/portfolio-lab/scripts/cron_guard.sh

if cron_guard_start "pf-build" 600; then
    export PATH="/root/.bun/bin:$PATH"
    cd /root/projects/portfolio-lab

    echo "Checking dashboard data..."
    if [ ! -f public/data/dashboard.json ]; then
        echo "Dashboard data missing, running generator..."
        export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
        python3 src/dashboard/generator.py
    fi

    echo "Running TypeScript check..."
    bun run tsc --noEmit 2>&1 | tee -a data/build.log || {
        echo "TypeScript errors detected, build aborted"
        cron_guard_end "pf-build" 1
        exit 1
    }

    echo "Building production app..."
    bun run build 2>&1 | tee -a data/build.log

    if [ -d "/var/www/portfolio-lab" ]; then
        echo "Syncing to deployment directory..."
        cp -r dist/* /var/www/portfolio-lab/
        echo "Deployed to /var/www/portfolio-lab/"
    fi

    echo "Build completed at $(date)"
    cron_guard_end "pf-build" $?
fi
