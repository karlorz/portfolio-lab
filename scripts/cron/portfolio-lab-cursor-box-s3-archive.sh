#!/bin/bash
# Daily SeaweedFS archive of the *live* cursor-box Portfolio Lab.
# Must not run inside the Tasker process: create stops production Tasker.
# cursor-box has no systemd; this is invoked from box-persist ensure.sh once per UTC day.
set -euo pipefail
export PATH="/home/box/.local/bin:/usr/local/bin:/usr/bin:/bin"
HOME="${HOME:-/home/box}"

ROOT="/home/box/.local/share/portfolio-lab"
APP="$ROOT/app"
WEB="$ROOT/www"
ARCHDIR="$ROOT/archives"
RUN="$ROOT/run"
LOCK="$RUN/s3-archive.lock"
STAMP_FILE="$RUN/s3-archive-last-utc-day"
CREDENTIALS_FILE="${PLR_S3_CREDENTIALS_FILE:-$HOME/.config/portfolio-lab/s3-credentials.env}"
ENDPOINT="${PLR_S3_ENDPOINT:-http://100.110.81.72:8333}"
BUCKET="${PLR_S3_BUCKET:-portfolio-lab-archives}"
BP="/home/box/.local/bin/portfolio-lab-box-persist"
SP="/home/box/.local/bin/portfolio-lab-static-persist"

mkdir -p "$ARCHDIR" "$RUN"
chmod 0700 "$ARCHDIR"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] s3-archive skipped (lock held)"
    exit 0
fi

restart_lab() {
    "$BP" ensure --mode production --app-dir "$APP" --web-root "$WEB" --service-name portfolio-lab-tasker
    "$SP" ensure --mode production --web-root "$WEB" --service-name portfolio-lab-static
}

STOPPED=0
trap 'if [ "$STOPPED" = 1 ]; then restart_lab || true; fi' EXIT

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$ARCHDIR/portfolio-lab-${STAMP}.tar"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stop production for quiesced archive"
"$BP" stop --mode production --app-dir "$APP" --web-root "$WEB" --service-name portfolio-lab-tasker
"$SP" stop --mode production --web-root "$WEB" --service-name portfolio-lab-static
STOPPED=1

# Quiesced copy: no .git, no venv, no restore metadata. Follow generation symlink.
tar -C "$ROOT" --exclude='app/.git' --exclude='app/.venv' \
    --exclude='app/.portfolio-lab-recovery' --exclude='app/__pycache__' \
    -cf "$ARCHIVE" app/data app/config/lab-app.env www
chmod 0600 "$ARCHIVE"
SUM="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$SUM" "$(basename "$ARCHIVE")" >"${ARCHIVE}.sha256"
chmod 0600 "${ARCHIVE}.sha256"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] restart production before rclone"
restart_lab
STOPPED=0

KEY="daily/$(date -u +%Y/%m/%d)/portfolio-lab-data-${STAMP}.tar"
set -a
# shellcheck disable=SC1090
source "$CREDENTIALS_FILE"
set +a
rclone copyto "$ARCHIVE" ":s3:${BUCKET}/${KEY}" \
    --s3-provider Other --s3-force-path-style --s3-env-auth \
    --s3-endpoint "$ENDPOINT" --s3-no-check-bucket --retries 3
rclone copyto "${ARCHIVE}.sha256" ":s3:${BUCKET}/${KEY}.sha256" \
    --s3-provider Other --s3-force-path-style --s3-env-auth \
    --s3-endpoint "$ENDPOINT" --s3-no-check-bucket --retries 3
rclone lsf ":s3:${BUCKET}/daily/" \
    --s3-provider Other --s3-force-path-style --s3-env-auth \
    --s3-endpoint "$ENDPOINT" --s3-no-check-bucket
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

date -u +%Y%m%d >"$STAMP_FILE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] published s3://${BUCKET}/${KEY} sha256=${SUM}"
