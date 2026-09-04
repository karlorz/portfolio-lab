#!/bin/bash
# portfolio-lab-s3-archive.sh - Daily backup creation, verification, S3 publish, and retention prune.
# NOTE: Must not run as a child of portfolio-lab-tasker because recovery create stops that unit.
# Protected by cron_guard: load-gate (max 5), flock, 2400s timeout, 3072MB ulimit.

set -euo pipefail

PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"
HOME="${HOME:-/root}"
CRON_GUARD_MEMORY_MB=3072 source "$PROJECT_DIR/scripts/cron_guard.sh"

if cron_guard_start "pf-s3-archive" 2400; then
    trap 'ec=$?; cron_guard_end "pf-s3-archive" "$ec"; exit "$ec"' EXIT
    START=$(date +%s)
    PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"
    SIDECAR="${PLR_S3_ARCHIVE_SIDECAR:-/var/backups/portfolio-lab-migration/sidecar}"
    ENDPOINT="${PLR_S3_ENDPOINT:-http://100.110.81.72:8333}"
    BUCKET="${PLR_S3_BUCKET:-portfolio-lab-archives}"
    CREDENTIALS_FILE="${PLR_S3_CREDENTIALS_FILE:-$HOME/.config/portfolio-lab/s3-credentials.env}"

    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p /var/backups/portfolio-lab-migration
    chmod 0700 /var/backups/portfolio-lab-migration 2>/dev/null || true
    ARCHIVE="/var/backups/portfolio-lab-migration/portfolio-lab-${STAMP}.portfolio-lab-recovery.tar"


    echo "[$(date -Iseconds)] Step 1: Creating recovery archive $ARCHIVE"
    "$PYTHON_RUNTIME" "$SIDECAR/scripts/portfolio_lab_recovery.py" create \
        --app-dir "$PROJECT_DIR" \
        --web-root /var/www/portfolio-lab \
        --tasker-service portfolio-lab-tasker \
        --archive "$ARCHIVE" \
        --storage-encryption-attested \
        --service-controller systemd \
        --materialize-generations-current

    echo "[$(date -Iseconds)] Step 2: Verifying recovery archive $ARCHIVE"
    "$PYTHON_RUNTIME" "$SIDECAR/scripts/portfolio_lab_recovery.py" verify \
        --archive "$ARCHIVE"

    echo "[$(date -Iseconds)] Step 3: Publishing archive to S3 via rclone"
    "$PYTHON_RUNTIME" "$SIDECAR/scripts/portfolio_lab_s3_archive.py" \
        --endpoint "$ENDPOINT" \
        --bucket "$BUCKET" \
        --credentials-file "$CREDENTIALS_FILE" \
        publish \
        --archive "$ARCHIVE" \
        --transport rclone

    echo "[$(date -Iseconds)] Step 4: Listing daily/ archives"
    # Credentials come from systemd EnvironmentFile or the 0600 credentials file via env; never print them.
    if [ -f "$CREDENTIALS_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$CREDENTIALS_FILE"
        set +a
    fi
    rclone lsf ":s3:${BUCKET}/daily/" \
        --s3-provider Other \
        --s3-force-path-style \
        --s3-env-auth \
        --s3-endpoint "$ENDPOINT" \
        --s3-no-check-bucket

    echo "[$(date -Iseconds)] Step 5: Pruning archives (keep 7 newest)"
    "$PYTHON_RUNTIME" "$SIDECAR/scripts/portfolio_lab_s3_archive.py" \
        --endpoint "$ENDPOINT" \
        --bucket "$BUCKET" \
        --credentials-file "$CREDENTIALS_FILE" \
        prune \
        --keep 7

    END=$(date +%s)
    DUR=$((END - START))
    echo "[$(date -Iseconds)] s3-archive finished duration=${DUR}s"
fi
