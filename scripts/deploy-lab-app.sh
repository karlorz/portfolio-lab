#!/usr/bin/env bash
# Host-native Portfolio-Lab deploy/update for lab.karldigi.dev.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/config/lab-app.env"

env_file_value() {
  local key="$1"
  [ -f "$ENV_FILE" ] || return 0
  sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1
}

env_default() {
  env_file_value "$1" || true
}

APP_HOST="${PORTFOLIO_LAB_SITE_ADDRESS:-$(env_default PORTFOLIO_LAB_SITE_ADDRESS)}"
APP_HOST="${APP_HOST:-lab.karldigi.dev}"
APP_DIR="${PORTFOLIO_LAB_APP_DIR:-$(env_default PORTFOLIO_LAB_APP_DIR)}"
APP_DIR="${APP_DIR:-$PROJECT_ROOT}"
WEB_ROOT="${PORTFOLIO_LAB_WEB_ROOT:-$(env_default PORTFOLIO_LAB_WEB_ROOT)}"
WEB_ROOT="${WEB_ROOT:-/var/www/portfolio-lab}"
PUBLIC_ROOT="${PORTFOLIO_LAB_PUBLIC_ROOT:-$(env_default PORTFOLIO_LAB_PUBLIC_ROOT)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-${WEB_ROOT}}"
TASKER_HOST="${TASKER_HOST:-$(env_default TASKER_HOST)}"
TASKER_HOST="${TASKER_HOST:-127.0.0.1}"
TASKER_PORT="${TASKER_PORT:-$(env_default TASKER_PORT)}"
TASKER_PORT="${TASKER_PORT:-8000}"
SERVICE_NAME="${TASKER_SERVICE_NAME:-$(env_default TASKER_SERVICE_NAME)}"
SERVICE_NAME="${SERVICE_NAME:-portfolio-lab-tasker}"
CADDY_CONFIG="${CADDY_CONFIG:-$(env_default CADDY_CONFIG)}"
CADDY_CONFIG="${CADDY_CONFIG:-/etc/caddy/Caddyfile}"
CADDY_SERVICE="${CADDY_SERVICE_NAME:-$(env_default CADDY_SERVICE_NAME)}"
CADDY_SERVICE="${CADDY_SERVICE:-caddy}"
UPDATE_COMMAND_PATH="${UPDATE_COMMAND_PATH:-$(env_default UPDATE_COMMAND_PATH)}"
UPDATE_COMMAND_PATH="${UPDATE_COMMAND_PATH:-/usr/local/bin/portfolio-lab-update}"

SKIP_GIT="0"
SKIP_DEPS="0"
SKIP_DATA="0"
SKIP_BUILD="0"
SKIP_SERVICE="0"
SKIP_CADDY="0"
SKIP_UPDATE_COMMAND="0"
SKIP_MIRROR="0"
DRY_RUN="0"
PRINT_CADDY="0"

usage() {
  cat <<USAGE
Usage:
  ${SCRIPT_NAME} [options]

Host-native deploy/update for a Proxmox LXC or Linux VM. The default path:
  1. fast-forwards the local git checkout when possible
  2. installs Python/frontend dependencies
  3. builds dist/
  4. publishes dist/ to the configured web root
  5. installs/restarts a systemd tasker service
  6. writes a managed Caddy site block for ${APP_HOST}
  7. installs ${UPDATE_COMMAND_PATH} for future in-container updates

Options:
  --host <hostname>             Public hostname (default: ${APP_HOST})
  --app-dir <path>              Project checkout path (default: ${APP_DIR})
  --web-root <path>             Static web root (default: ${WEB_ROOT})
  --public-root <path>          Public data root for /data/* (default: ${PUBLIC_ROOT})
  --tasker-host <host>          Tasker bind host (default: ${TASKER_HOST})
  --tasker-port <port>          Tasker API port (default: ${TASKER_PORT})
  --service-name <name>         systemd service name (default: ${SERVICE_NAME})
  --caddy-config <path>         Caddyfile path (default: ${CADDY_CONFIG})
  --caddy-service <name>        Caddy systemd unit (default: ${CADDY_SERVICE})
  --update-command <path>       Update command path (default: ${UPDATE_COMMAND_PATH})
  --skip-git                   Do not run git pull --ff-only
  --skip-deps                  Do not run uv/bun dependency install
  --skip-data                  Do not refresh public/data via bun run fetch-data
  --skip-build                 Do not build frontend dist/
  --skip-service               Do not install/restart tasker systemd service
  --skip-caddy                 Do not write/reload Caddy config
  --skip-update-command        Do not install the in-container update command
  --skip-mirror                Do not mirror live public data → checkout public/data
  --print-caddy                Print the managed Caddy site block and exit
  --dry-run                    Print actions without changing the host
  -h, --help                   Show help

DNS for ${APP_HOST} is considered pre-completed and outside this script.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] WARN: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

is_root() {
  [ "$(id -u)" = "0" ]
}

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[%s] [dry-run]' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_app_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[%s] [dry-run] cd %q &&' "$(date '+%Y-%m-%d %H:%M:%S')" "$APP_DIR"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  (cd "$APP_DIR" && "$@")
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host) APP_HOST="${2:-}"; shift 2 ;;
    --app-dir) APP_DIR="${2:-}"; shift 2 ;;
    --web-root) WEB_ROOT="${2:-}"; shift 2 ;;
    --public-root) PUBLIC_ROOT="${2:-}"; shift 2 ;;
    --tasker-host) TASKER_HOST="${2:-}"; shift 2 ;;
    --tasker-port) TASKER_PORT="${2:-}"; shift 2 ;;
    --service-name) SERVICE_NAME="${2:-}"; shift 2 ;;
    --caddy-config) CADDY_CONFIG="${2:-}"; shift 2 ;;
    --caddy-service) CADDY_SERVICE="${2:-}"; shift 2 ;;
    --update-command) UPDATE_COMMAND_PATH="${2:-}"; shift 2 ;;
    --skip-git) SKIP_GIT="1"; shift ;;
    --skip-deps) SKIP_DEPS="1"; shift ;;
    --skip-data) SKIP_DATA="1"; shift ;;
    --skip-build) SKIP_BUILD="1"; shift ;;
    --skip-service) SKIP_SERVICE="1"; shift ;;
    --skip-caddy) SKIP_CADDY="1"; shift ;;
    --skip-update-command) SKIP_UPDATE_COMMAND="1"; shift ;;
    --skip-mirror) SKIP_MIRROR="1"; shift ;;
    --print-caddy) PRINT_CADDY="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[ -n "$APP_HOST" ] || die "--host must not be empty"
[ -n "$APP_DIR" ] || die "--app-dir must not be empty"
[ -n "$WEB_ROOT" ] || die "--web-root must not be empty"
[ -n "$PUBLIC_ROOT" ] || die "--public-root must not be empty"
[ -n "$TASKER_HOST" ] || die "--tasker-host must not be empty"
[ -n "$TASKER_PORT" ] || die "--tasker-port must not be empty"
[ -n "$SERVICE_NAME" ] || die "--service-name must not be empty"
[ -n "$CADDY_CONFIG" ] || die "--caddy-config must not be empty"
[ -n "$CADDY_SERVICE" ] || die "--caddy-service must not be empty"

if [ "$PRINT_CADDY" != "1" ] && [ "$DRY_RUN" != "1" ] && { [ "$SKIP_SERVICE" != "1" ] || [ "$SKIP_CADDY" != "1" ] || [ "$SKIP_UPDATE_COMMAND" != "1" ]; }; then
  is_root || die "Run as root for systemd/Caddy/update-command setup, or use --skip-service --skip-caddy --skip-update-command."
fi

generate_caddy_block() {
  cat <<EOF
# BEGIN portfolio-lab managed
${APP_HOST} {
	encode gzip

	handle /api/* {
		reverse_proxy ${TASKER_HOST}:${TASKER_PORT}
	}

	handle /assets/* {
		root * ${WEB_ROOT}
		header Cache-Control "public, max-age=31536000, immutable"
		file_server
	}

	handle /data/* {
		root * ${PUBLIC_ROOT}
		header Cache-Control "no-cache"
		file_server
	}

	handle {
		root * ${WEB_ROOT}
		header Cache-Control "no-cache"
		try_files {path} /index.html
		file_server
	}

	log {
		output stdout
		format json
	}
}
# END portfolio-lab managed
EOF
}

git_update() {
  [ "$SKIP_GIT" = "0" ] || return 0
  if [ ! -d "${APP_DIR}/.git" ]; then
    warn "No .git directory at ${APP_DIR}; skipping git update"
    return 0
  fi
  log "Fast-forwarding git checkout"
  run_cmd git -C "$APP_DIR" pull --ff-only
}

install_dependencies() {
  [ "$SKIP_DEPS" = "0" ] || return 0
  log "Installing Python dependencies"
  need_cmd uv
  run_app_cmd uv sync --frozen --no-dev --no-group ml

  log "Installing frontend dependencies"
  if command -v bun >/dev/null 2>&1; then
    run_app_cmd bun install --frozen-lockfile
  elif command -v npm >/dev/null 2>&1; then
    run_app_cmd npm install --no-audit --no-fund
  else
    die "Missing frontend package manager: install bun or npm"
  fi
}

build_frontend() {
  [ "$SKIP_BUILD" = "0" ] || return 0
  log "Building frontend dist"
  if command -v bun >/dev/null 2>&1; then
    run_app_cmd bun run build
  elif command -v npm >/dev/null 2>&1; then
    run_app_cmd npm run build
  else
    die "Missing frontend package manager: install bun or npm"
  fi
}

refresh_dashboard_data() {
  [ "$SKIP_DATA" = "0" ] || return 0
  log "Refreshing public dashboard data"
  if command -v bun >/dev/null 2>&1; then
    run_app_cmd bun run fetch-data
  else
    die "Missing frontend package manager: install bun to refresh dashboard data"
  fi
}

check_fred_readiness() {
  [ "$SKIP_DATA" = "0" ] || return 0
  log "Checking FRED readiness before refreshing public data"
  local readiness_mode
  readiness_mode="${PORTFOLIO_LAB_MODE:-lab}"
  run_app_cmd env PORTFOLIO_LAB_MODE="$readiness_mode" CRON_BACKEND="${CRON_BACKEND:-tasker}" ./scripts/python_runtime.sh -m src.monitor.fred_readiness --mode "$readiness_mode"
}

check_public_data_consistency() {
  log "Checking public data consistency before publish"
  if [ "$SKIP_DATA" = "1" ]; then
    warn "--skip-data set; validating existing public/data and dist/data artifacts"
  fi
  # Checkout public/data is intentional here (pre-publish). Live WWW SSOT is
  # separate; ops audits must set PUBLIC_DATA_DIR or --public-dir.
  run_app_cmd ./scripts/python_runtime.sh scripts/check_public_data_consistency.py \
    --app-dir "$APP_DIR" --allow-repo-public-data
}

mirror_repo_public_data_from_live() {
  # Batch BX: soft-gate static mirror of live PUBLIC_ROOT → checkout public/data.
  # Never fails deploy (ops runbook: soft gate + monitoring for mirror lag).
  [ "$SKIP_MIRROR" = "0" ] || return 0
  log "Mirroring live public data into checkout public/data (soft gate)"
  local live_src="${PUBLIC_ROOT}/data"
  if [ ! -d "$live_src" ]; then
    # WEB_ROOT often is /var/www/portfolio-lab with data/ nested
    if [ -d "${WEB_ROOT}/data" ]; then
      live_src="${WEB_ROOT}/data"
    else
      warn "No live public data dir at ${PUBLIC_ROOT}/data or ${WEB_ROOT}/data; skip mirror"
      return 0
    fi
  fi
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] Would mirror ${live_src} → ${APP_DIR}/public/data"
    return 0
  fi
  if ! (
    cd "$APP_DIR" &&
      ./scripts/python_runtime.sh scripts/mirror_repo_public_data.py \
        --source "$live_src" \
        --dest "${APP_DIR}/public/data"
  ); then
    warn "mirror_repo_public_data soft-failed (non-blocking); run make mirror-repo-public-data later"
  fi
}

publish_dist() {
  if [ "$DRY_RUN" != "1" ] && [ ! -d "${APP_DIR}/dist" ]; then
    die "Missing ${APP_DIR}/dist; run without --skip-build or build first"
  fi

  log "Publishing static app to ${WEB_ROOT}"
  run_cmd mkdir -p "$WEB_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    run_cmd rsync -a --delete "${APP_DIR}/dist/" "${WEB_ROOT}/"
  else
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] Would copy ${APP_DIR}/dist/ to ${WEB_ROOT}/ with tar"
    else
      tar -C "${APP_DIR}/dist" -cf - . | tar -C "$WEB_ROOT" -xf -
    fi
  fi
}

install_tasker_service() {
  [ "$SKIP_SERVICE" = "0" ] || return 0
  need_cmd systemctl

  local unit_path
  unit_path="/etc/systemd/system/${SERVICE_NAME}.service"
  log "Installing systemd service: ${unit_path}"

  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] Would write ${unit_path} and restart ${SERVICE_NAME}"
    return 0
  fi

  cat > "$unit_path" <<EOF
[Unit]
Description=Portfolio Lab tasker API and scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment=PORTFOLIO_LAB_ENABLE_ML=0
Environment=PORTFOLIO_LAB_MODE=lab
Environment=CRON_BACKEND=tasker
Environment=PORTFOLIO_LAB_PROJECT_DIR=${APP_DIR}
Environment=PUBLIC_DATA_DIR=${WEB_ROOT}/data
Environment=PYTHONPATH=${APP_DIR}
Environment=LOG_LEVEL=INFO
Environment=JSON_LOGS=1
Environment=PATH=/root/.bun/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=-${APP_DIR}/.env.local
ExecStart=${APP_DIR}/scripts/python_runtime.sh -m src.tasker.service --host ${TASKER_HOST} --port ${TASKER_PORT}
Restart=always
RestartSec=10
TimeoutStopSec=30
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME" >/dev/null
  systemctl restart "$SERVICE_NAME"
}

write_caddy_config() {
  [ "$SKIP_CADDY" = "0" ] || return 0
  need_cmd caddy
  need_cmd python3

  local block_file backup
  block_file="$(mktemp)"
  generate_caddy_block > "$block_file"

  if [ "$PRINT_CADDY" = "1" ]; then
    cat "$block_file"
    rm -f "$block_file"
    exit 0
  fi

  log "Writing managed Caddy site block to ${CADDY_CONFIG}"
  if [ "$DRY_RUN" = "1" ]; then
    cat "$block_file"
    log "[dry-run] Would validate ${CADDY_CONFIG} and reload ${CADDY_SERVICE}"
    rm -f "$block_file"
    return 0
  fi

  mkdir -p "$(dirname "$CADDY_CONFIG")"
  touch "$CADDY_CONFIG"
  backup="${CADDY_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
  cp "$CADDY_CONFIG" "$backup"

  python3 - "$CADDY_CONFIG" "$block_file" <<'PY'
from pathlib import Path
import re
import sys

config = Path(sys.argv[1])
block = Path(sys.argv[2]).read_text(encoding="utf-8").strip() + "\n"
begin = "# BEGIN portfolio-lab managed"
end = "# END portfolio-lab managed"
text = config.read_text(encoding="utf-8") if config.exists() else ""
pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.S)
if begin in text and end in text:
    text = pattern.sub(block, text)
else:
    if text and not text.endswith("\n"):
        text += "\n"
    if text:
        text += "\n"
    text += block
config.write_text(text, encoding="utf-8")
PY

  if ! caddy validate --config "$CADDY_CONFIG" --adapter caddyfile; then
    cp "$backup" "$CADDY_CONFIG"
    die "Caddy validation failed; restored ${backup}"
  fi

  systemctl reload "$CADDY_SERVICE" || systemctl restart "$CADDY_SERVICE"
  rm -f "$block_file"
}

install_update_command() {
  [ "$SKIP_UPDATE_COMMAND" = "0" ] || return 0

  log "Installing in-container update command: ${UPDATE_COMMAND_PATH}"
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] Would install ${UPDATE_COMMAND_PATH}"
    return 0
  fi

  local tmp
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$APP_DIR")
exec $(printf '%q' "$APP_DIR")/scripts/deploy-lab-app.sh "\$@"
EOF
  install -m 0755 "$tmp" "$UPDATE_COMMAND_PATH"
  rm -f "$tmp"
}

smoke_check() {
  if [ "$DRY_RUN" = "1" ] || [ "$SKIP_SERVICE" = "1" ]; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    log "Checking tasker API health"
    local attempt
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      if curl -fsS --max-time 5 "http://${TASKER_HOST}:${TASKER_PORT}/api/tasker/status" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    curl -fsS --max-time 10 "http://${TASKER_HOST}:${TASKER_PORT}/api/tasker/status" >/dev/null
  fi
}

main() {
  cd "$APP_DIR"
  if [ "$PRINT_CADDY" = "1" ]; then
    write_caddy_config
  fi
  git_update
  install_dependencies
  check_fred_readiness
  refresh_dashboard_data
  build_frontend
  check_public_data_consistency
  publish_dist
  # After publish: soft-mirror live WWW data into checkout public/data (H22b/BX)
  mirror_repo_public_data_from_live
  install_tasker_service
  write_caddy_config
  install_update_command
  smoke_check
  log "Lab app configured for https://${APP_HOST}"
}

main "$@"
