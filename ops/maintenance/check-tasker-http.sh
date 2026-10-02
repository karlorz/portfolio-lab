#!/bin/sh
# Probe Tasker :8000 /api/tasker/status (pass) and / (observational), plus :8001 /.
# MUST NOT restart prod app under /home/box/.local/share/portfolio-lab/app
# based on / 404 alone.
#
# Env:
#   TASKER_HTTP_BASE  default http://127.0.0.1:8000
#   STATIC_HTTP_BASE  default http://127.0.0.1:8001
set -eu

TASKER_HTTP_BASE="${TASKER_HTTP_BASE:-http://127.0.0.1:8000}"
STATIC_HTTP_BASE="${STATIC_HTTP_BASE:-http://127.0.0.1:8001}"
TASKER_HTTP_BASE="${TASKER_HTTP_BASE%/}"
STATIC_HTTP_BASE="${STATIC_HTTP_BASE%/}"

STATUS_URL="${TASKER_HTTP_BASE}/api/tasker/status"
ROOT_URL="${TASKER_HTTP_BASE}/"
STATIC_URL="${STATIC_HTTP_BASE}/"

http_code() {
    url=$1
    dest=$2
    code=$(curl -sS --connect-timeout 3 --max-time 5 -o "$dest" -w "%{http_code}" "$url" 2>/dev/null) || true
    case "$code" in
        [0-9][0-9][0-9]) printf '%s\n' "$code" ;;
        *) printf '%s\n' "000" ;;
    esac
}

json_has_tasker_identity() {
    f=$1
    [ -f "$f" ] && [ -s "$f" ] || return 1
    grep -q '"backend":"tasker"' "$f" && return 0
    grep -q '"backend": "tasker"' "$f" && return 0
    grep -q 'portfolio-lab-tasker' "$f" && return 0
    return 1
}

if ! command -v curl >/dev/null 2>&1; then
    echo "port=8000 path=/api/tasker/status http=000 class=fail-tasker-status"
    echo "port=8000 path=/ http=000 class=observe-root"
    echo "port=8001 path=/ http=000 class=fail-static"
    echo "tasker-http: fail"
    exit 1
fi

bodyfile=$(mktemp)
trap 'rm -f "$bodyfile"' EXIT

status_code=$(http_code "$STATUS_URL" "$bodyfile")
root_code=$(http_code "$ROOT_URL" /dev/null)
static_code=$(http_code "$STATIC_URL" /dev/null)

if [ "$status_code" = "200" ] && json_has_tasker_identity "$bodyfile"; then
    status_class="ok-tasker-status"
else
    status_class="fail-tasker-status"
fi

if [ "$static_code" = "200" ]; then
    static_class="ok-static"
else
    static_class="fail-static"
fi

if [ "$status_class" = "ok-tasker-status" ] && [ "$static_class" = "ok-static" ]; then
    overall="ok"
    rc=0
else
    overall="fail"
    rc=1
fi

echo "port=8000 path=/api/tasker/status http=${status_code} class=${status_class}"
echo "port=8000 path=/ http=${root_code} class=observe-root"
echo "port=8001 path=/ http=${static_code} class=${static_class}"
echo "tasker-http: ${overall}"
exit "$rc"
