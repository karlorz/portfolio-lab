#!/bin/bash
# cron-app-build.sh - Build Vite app with latest dashboard data
# Protected by cron_guard: load-gate (max 5), flock, 600s timeout, 3GB ulimit
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
source "$PROJECT_DIR/scripts/cron_guard.sh"
source "$PROJECT_DIR/scripts/cron/hermes_status.sh"

if cron_guard_start "pf-build" 600; then
    START=$(date +%s)
    export PATH="/root/.bun/bin:$PATH"
    cd "$PROJECT_DIR"
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    EXIT=0

    echo "Checking dashboard data..."
    if [ ! -f public/data/dashboard.json ]; then
        echo "Dashboard data missing, running generator..."
        export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
        set +e
        "$PYTHON_RUNTIME" src/dashboard/generator.py
        EXIT=$?
        set -e
    fi

    if [ "$EXIT" -eq 0 ]; then
        echo "Running TypeScript check..."
        set +e
        bun run tsc --noEmit 2>&1 | tee -a data/build.log
        tsc_pipeline_status=("${PIPESTATUS[@]}")
        EXIT="$(cron_pipeline_exit "${tsc_pipeline_status[0]}" "${tsc_pipeline_status[1]}")"
        set -e
    fi

    if [ "$EXIT" -eq 0 ]; then
        echo "Building production app..."
        set +e
        bun run build 2>&1 | tee -a data/build.log
        build_pipeline_status=("${PIPESTATUS[@]}")
        EXIT="$(cron_pipeline_exit "${build_pipeline_status[0]}" "${build_pipeline_status[1]}")"
        set -e
    else
        echo "TypeScript errors detected, build aborted"
    fi

    if [ "$EXIT" -eq 0 ] && [ -d "/var/www/portfolio-lab" ]; then
        echo "Syncing to deployment directory..."
        set +e
        cp -r dist/* /var/www/portfolio-lab/
        EXIT=$?
        set -e
        if [ "$EXIT" -eq 0 ]; then
            echo "Deployed to /var/www/portfolio-lab/"
        fi
    fi

    if [ "$EXIT" -eq 0 ]; then
        echo "Build completed at $(date)"
    fi

    END=$(date +%s)
    DUR=$((END - START))
    record_hermes_cron_status "portfolio-lab-build" "$EXIT" "$DUR"
    cron_guard_end "pf-build" "$EXIT"
fi
