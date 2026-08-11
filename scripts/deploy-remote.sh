#!/usr/bin/env bash
# deploy-remote.sh
#
# Safe, reusable deploy script for preview/production Linux hosts.
# Works from macOS/Linux clients (bash 3.2+ compatible).
#
# Key safety guarantees:
# - No broad process kills (port-scoped only).
# - Staged releases under remote releases/<id> with current symlink.
# - Health checks before declaring success.
# - Optional rollback to previous release on preview-start failure.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="preview"
HOST=""
SSH_USER=""
SSH_PORT="22"
REMOTE_BASE=""
SYNC_METHOD="auto"   # auto|rsync|tar
PACKAGE_MANAGER="auto" # auto|bun|npm
PREVIEW_PORT="4173"
PREVIEW_HOST="127.0.0.1"
PREVIEW_SERVICE="portfolio-lab-preview"
PREVIEW_MEMORY_MAX="512M"
HEALTH_URL=""
PROD_WEB_ROOT="/var/www/portfolio-lab"
RELOAD_SERVICE=""
KEEP_RELEASES="5"
RELEASE_ID=""
SKIP_INSTALL="0"
SKIP_BUILD="0"
ALLOW_DIRTY="0"
RUN_GENERATOR="0"
BOOTSTRAP_PREVIEW_DATA="1"
DRY_RUN="0"
NO_STRICT_HOSTKEY="0"
EXCLUDE_FILE=""

usage() {
  cat <<USAGE
Usage:
  ${SCRIPT_NAME} --host <ssh-host> [options]

Required:
  --host <host>                 SSH host alias or IP

Core options:
  --mode <preview|production>   Deploy mode (default: preview)
  --user <user>                 SSH user override
  --ssh-port <port>             SSH port (default: 22)
  --remote-base <path>          Remote base dir (default by mode)
  --sync-method <auto|rsync|tar>
  --package-manager <auto|bun|npm>
  --keep-releases <n>           Number of releases to retain (default: 5)
  --release-id <id>             Custom release id (default: timestamp+gitsha)

Preview mode options:
  --preview-port <port>         Preview HTTP port (default: 4173)
  --preview-host <host>         Preview bind host (default: 127.0.0.1)
  --preview-service <name>      systemd service name (default: portfolio-lab-preview)
  --no-preview-service          Disable systemd service; use nohup fallback
  --preview-memory-max <size>   systemd MemoryMax for preview service (default: 512M)

Production mode options:
  --prod-web-root <path>        Static web root (default: /var/www/portfolio-lab)
  --reload-service <name>       Optional systemd service to reload/restart (e.g. caddy)

Health/options:
  --health-url <url>            Remote health URL; defaults per mode
  --skip-install                Skip dependency install
  --skip-build                  Skip frontend build step
  --allow-dirty                 Allow a build from an uncommitted working tree
  --run-generator               Run dashboard generator (best-effort, ML disabled)
  --no-bootstrap-preview-data   Disable preview /dist/data + /public/data JSON bootstrap
  --exclude-file <path>         Additional exclude patterns file
  --no-strict-host-key          Disable strict host key checks (not recommended)
  --dry-run                     Print actions only
  -h, --help                    Show help

Examples:
  # Private preview deploy on sg02
  ${SCRIPT_NAME} --host sg02 --mode preview \
    --remote-base \${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab-preview} \
    --preview-port 4173 --health-url http://127.0.0.1:4173/

  # Production deploy with static web root + Caddy reload
  ${SCRIPT_NAME} --host sg01 --mode production \
    --remote-base \${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab} \
    --prod-web-root /var/www/portfolio-lab \
    --reload-service caddy --health-url http://127.0.0.1/
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

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

is_integer() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --user) SSH_USER="${2:-}"; shift 2 ;;
    --ssh-port) SSH_PORT="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --sync-method) SYNC_METHOD="${2:-}"; shift 2 ;;
    --package-manager) PACKAGE_MANAGER="${2:-}"; shift 2 ;;
    --preview-port) PREVIEW_PORT="${2:-}"; shift 2 ;;
    --preview-host) PREVIEW_HOST="${2:-}"; shift 2 ;;
    --preview-service) PREVIEW_SERVICE="${2:-}"; shift 2 ;;
    --no-preview-service) PREVIEW_SERVICE=""; shift ;;
    --preview-memory-max) PREVIEW_MEMORY_MAX="${2:-}"; shift 2 ;;
    --health-url) HEALTH_URL="${2:-}"; shift 2 ;;
    --prod-web-root) PROD_WEB_ROOT="${2:-}"; shift 2 ;;
    --reload-service) RELOAD_SERVICE="${2:-}"; shift 2 ;;
    --keep-releases) KEEP_RELEASES="${2:-}"; shift 2 ;;
    --release-id) RELEASE_ID="${2:-}"; shift 2 ;;
    --exclude-file) EXCLUDE_FILE="${2:-}"; shift 2 ;;
    --skip-install) SKIP_INSTALL="1"; shift ;;
    --skip-build) SKIP_BUILD="1"; shift ;;
    --allow-dirty) ALLOW_DIRTY="1"; shift ;;
    --run-generator) RUN_GENERATOR="1"; shift ;;
    --no-bootstrap-preview-data) BOOTSTRAP_PREVIEW_DATA="0"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --no-strict-host-key) NO_STRICT_HOSTKEY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[ -n "$HOST" ] || { usage; die "--host is required"; }
[ "$MODE" = "preview" ] || [ "$MODE" = "production" ] || die "--mode must be preview or production"
[ "$SYNC_METHOD" = "auto" ] || [ "$SYNC_METHOD" = "rsync" ] || [ "$SYNC_METHOD" = "tar" ] || die "--sync-method must be auto|rsync|tar"
[ "$PACKAGE_MANAGER" = "auto" ] || [ "$PACKAGE_MANAGER" = "bun" ] || [ "$PACKAGE_MANAGER" = "npm" ] || die "--package-manager must be auto|bun|npm"

is_integer "$SSH_PORT" || die "--ssh-port must be an integer"
is_integer "$PREVIEW_PORT" || die "--preview-port must be an integer"
is_integer "$KEEP_RELEASES" || die "--keep-releases must be an integer"
[ -n "$PREVIEW_HOST" ] || die "--preview-host must not be empty"
[ -n "$PREVIEW_MEMORY_MAX" ] || die "--preview-memory-max must not be empty"

if [ "$SKIP_BUILD" = "0" ] && [ "$ALLOW_DIRTY" != "1" ]; then
  DIRTY_COUNT="$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$DIRTY_COUNT" -gt 0 ]; then
    die "Working tree has ${DIRTY_COUNT} uncommitted paths. Commit them or pass --allow-dirty."
  fi
fi

if [ -z "$REMOTE_BASE" ]; then
  if [ "$MODE" = "preview" ]; then
    REMOTE_BASE="${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab-preview}"
  else
    REMOTE_BASE="${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab}"
  fi
fi

if [ -z "$HEALTH_URL" ]; then
  if [ "$MODE" = "preview" ]; then
    HEALTH_URL="http://127.0.0.1:${PREVIEW_PORT}/"
  else
    HEALTH_URL="http://127.0.0.1/"
  fi
fi

if [ -n "$EXCLUDE_FILE" ] && [ ! -f "$EXCLUDE_FILE" ]; then
  die "Exclude file not found: $EXCLUDE_FILE"
fi

require_cmd ssh
require_cmd tar
require_cmd date

if [ -n "$SSH_USER" ]; then
  REMOTE="${SSH_USER}@${HOST}"
else
  REMOTE="${HOST}"
fi

SSH_ARGS=("-p" "$SSH_PORT" "-o" "BatchMode=yes" "-o" "ConnectTimeout=10")
if [ "$NO_STRICT_HOSTKEY" = "1" ]; then
  SSH_ARGS+=("-o" "StrictHostKeyChecking=no" "-o" "UserKnownHostsFile=/dev/null")
fi

ssh_exec() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[%s] [dry-run] ssh %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$REMOTE" "$*" >&2
    return 0
  fi
  # Values are intentionally expanded on the client before executing remotely.
  # shellcheck disable=SC2029
  ssh "${SSH_ARGS[@]}" "$REMOTE" "$@"
}

mktemp_compat() {
  local base
  base="${TMPDIR:-/tmp}/${SCRIPT_NAME}.tmp.$$.${RANDOM:-0}"
  echo "$base"
}

write_default_excludes() {
  local out_file
  out_file="$1"
  cat > "$out_file" <<'EXCLUDES'
.git/
.github/
.claude/
.hermes/
node_modules/
.venv/
dist/
__pycache__/
.pytest_cache/
.mypy_cache/
.DS_Store
._*
*.pyc
*.pyo
data/*.db
data/*.sqlite
data/*.sqlite3
logs/
EXCLUDES

  if [ -n "$EXCLUDE_FILE" ]; then
    cat "$EXCLUDE_FILE" >> "$out_file"
  fi
}

pick_sync_method() {
  if [ "$SYNC_METHOD" = "rsync" ] || [ "$SYNC_METHOD" = "tar" ]; then
    echo "$SYNC_METHOD"
    return
  fi

  if [ "$DRY_RUN" = "1" ]; then
    if command -v rsync >/dev/null 2>&1; then
      echo "rsync"
    else
      echo "tar"
    fi
    return
  fi

  if command -v rsync >/dev/null 2>&1; then
    if ssh_exec "command -v rsync >/dev/null 2>&1"; then
      echo "rsync"
      return
    fi
  fi

  echo "tar"
}

ensure_release_id() {
  if [ -n "$RELEASE_ID" ]; then
    echo "$RELEASE_ID"
    return
  fi

  local ts sha
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  sha="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
  echo "${ts}-${sha}"
}

main() {
  local release_id remote_release_dir excludes_tmp chosen_sync
  local -a local_tar_flags
  release_id="$(ensure_release_id)"
  remote_release_dir="${REMOTE_BASE}/releases/${release_id}"
  local_tar_flags=()

  # Avoid macOS xattr/metadata noise when streaming archives to Linux hosts.
  if tar -cf /dev/null --no-xattrs /dev/null >/dev/null 2>&1; then
    local_tar_flags+=(--no-xattrs)
  fi
  if tar -cf /dev/null --no-acls /dev/null >/dev/null 2>&1; then
    local_tar_flags+=(--no-acls)
  fi
  if tar -cf /dev/null --no-mac-metadata /dev/null >/dev/null 2>&1; then
    local_tar_flags+=(--no-mac-metadata)
  fi

  log "Local platform: $(uname -s)/$(uname -m)"
  log "Deploy mode: ${MODE}"
  log "Remote: ${REMOTE}"
  log "Release: ${release_id}"
  log "Remote base: ${REMOTE_BASE}"

  log "Checking SSH connectivity..."
  ssh_exec "echo connected: \$(hostname) \$(uname -s)/\$(uname -m)"

  log "Preparing remote directories..."
  ssh_exec "mkdir -p '${REMOTE_BASE}/releases' '${REMOTE_BASE}/shared' '${remote_release_dir}'"

  excludes_tmp="$(mktemp_compat)"
  write_default_excludes "$excludes_tmp"

  chosen_sync="$(pick_sync_method)"
  log "Sync method: ${chosen_sync}"

  if [ "$chosen_sync" = "rsync" ]; then
    require_cmd rsync
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] rsync project -> ${REMOTE}:${remote_release_dir}/"
    else
      rsync -az --delete \
        --exclude-from="$excludes_tmp" \
        "${PROJECT_ROOT}/" "${REMOTE}:${remote_release_dir}/"
    fi
  else
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] tar stream project -> ${REMOTE}:${remote_release_dir}"
    else
      # The remote release dir is intentionally expanded locally for the SSH command.
      # shellcheck disable=SC2029
      COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
      tar "${local_tar_flags[@]}" -C "$PROJECT_ROOT" --exclude-from="$excludes_tmp" -cf - . \
        | ssh "${SSH_ARGS[@]}" "$REMOTE" "tar -xf - -C '${remote_release_dir}'"
    fi
  fi

  rm -f "$excludes_tmp"

  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] Would now run remote deploy lifecycle"
    return 0
  fi

  # shellcheck disable=SC2029
  {
    printf 'MODE=%q\n' "$MODE"
    printf 'REMOTE_BASE=%q\n' "$REMOTE_BASE"
    printf 'RELEASE_ID=%q\n' "$release_id"
    printf 'KEEP_RELEASES=%q\n' "$KEEP_RELEASES"
    printf 'PACKAGE_MANAGER=%q\n' "$PACKAGE_MANAGER"
    printf 'SKIP_INSTALL=%q\n' "$SKIP_INSTALL"
    printf 'SKIP_BUILD=%q\n' "$SKIP_BUILD"
    printf 'RUN_GENERATOR=%q\n' "$RUN_GENERATOR"
    printf 'BOOTSTRAP_PREVIEW_DATA=%q\n' "$BOOTSTRAP_PREVIEW_DATA"
    printf 'PREVIEW_PORT=%q\n' "$PREVIEW_PORT"
    printf 'PREVIEW_HOST=%q\n' "$PREVIEW_HOST"
    printf 'PREVIEW_SERVICE=%q\n' "$PREVIEW_SERVICE"
    printf 'PREVIEW_MEMORY_MAX=%q\n' "$PREVIEW_MEMORY_MAX"
    printf 'PROD_WEB_ROOT=%q\n' "$PROD_WEB_ROOT"
    printf 'RELOAD_SERVICE=%q\n' "$RELOAD_SERVICE"
    printf 'HEALTH_URL=%q\n' "$HEALTH_URL"
    cat <<'REMOTE_SCRIPT'
set -euo pipefail

log() {
  printf '[remote %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[remote %s] WARN: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

release_dir="${REMOTE_BASE}/releases/${RELEASE_ID}"
current_link="${REMOTE_BASE}/current"
previous_target="$(readlink "$current_link" 2>/dev/null || true)"

if [ ! -d "$release_dir" ]; then
  echo "Release directory missing: $release_dir" >&2
  exit 1
fi

pick_pm() {
  local requested="$1"
  if [ "$requested" = "auto" ]; then
    if command -v bun >/dev/null 2>&1; then
      echo "bun"
      return
    fi
    if command -v npm >/dev/null 2>&1; then
      # bun canonical (CLAUDE.md/CI) — package-lock.json intentionally absent; npm install would regenerate
      echo "npm"
      return
    fi
    echo "none"
    return
  fi
  echo "$requested"
}

stop_port_listener() {
  local port="$1"
  local pids
  pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u || true)"
  if [ -z "$pids" ]; then
    return 0
  fi

  for pid in $pids; do
    if [ -n "$pid" ] && [ "$pid" != "$$" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 1
}

preview_service_unit() {
  local name="$PREVIEW_SERVICE"

  case "$name" in
    ""|"0"|"none") return 1 ;;
  esac

  case "$name" in
    *.service) ;;
    *) name="${name}.service" ;;
  esac

  case "$name" in
    ""|*[!A-Za-z0-9_.@-]*)
      warn "Invalid preview service name: ${PREVIEW_SERVICE}; using nohup fallback"
      return 1
      ;;
  esac

  printf '%s\n' "$name"
}

can_manage_systemd() {
  command -v systemctl >/dev/null 2>&1 && [ "$(id -u)" = "0" ] && [ -d /etc/systemd/system ]
}

stop_preview() {
  local port="$1"
  local unit

  unit="$(preview_service_unit 2>/dev/null || true)"
  if [ -n "$unit" ] && can_manage_systemd; then
    systemctl stop "$unit" >/dev/null 2>&1 || true
  fi

  stop_port_listener "$port"
}

write_preview_service() {
  local workdir="$1"
  local pm="$2"
  local port="$3"
  local host="$4"
  local unit="$5"
  local pm_bin exec_start unit_path service_path

  pm_bin="$(command -v "$pm" 2>/dev/null || true)"
  if [ -z "$pm_bin" ]; then
    warn "Could not resolve package manager binary for ${pm}; using nohup fallback"
    return 1
  fi

  if [ "$pm" = "bun" ]; then
    exec_start="${pm_bin} run preview --host ${host} --port ${port}"
  else
    exec_start="${pm_bin} run preview -- --host ${host} --port ${port}"
  fi

  unit_path="/etc/systemd/system/${unit}"
  service_path="$PATH"

  cat > "$unit_path" <<EOF
[Unit]
Description=Portfolio Lab protected preview app
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${workdir}
Environment=NODE_ENV=production
Environment=PORTFOLIO_LAB_ENABLE_ML=0
Environment=PATH=${service_path}
ExecStartPre=/usr/bin/test -f ${workdir}/package.json
ExecStartPre=/usr/bin/test -d ${workdir}/dist
ExecStart=${exec_start}
Restart=always
RestartSec=5
TimeoutStopSec=20
KillSignal=SIGTERM
SyslogIdentifier=${unit%.service}
MemoryMax=${PREVIEW_MEMORY_MAX}
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
}

start_preview_service() {
  local workdir="$1"
  local pm="$2"
  local port="$3"
  local host="$4"
  local unit

  unit="$(preview_service_unit 2>/dev/null || true)"
  if [ -z "$unit" ]; then
    return 1
  fi

  if ! can_manage_systemd; then
    warn "systemd root access unavailable; using nohup fallback"
    return 1
  fi

  log "Installing/restarting preview service: ${unit}"
  write_preview_service "$workdir" "$pm" "$port" "$host" "$unit" || return 1
  systemctl daemon-reload
  systemctl enable "$unit" >/dev/null
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  systemctl restart "$unit"
  sleep 2

  systemctl is-active --quiet "$unit" && ss -ltnp "sport = :${port}" >/dev/null 2>&1
}

start_preview_nohup() {
  local workdir="$1"
  local pm="$2"
  local port="$3"
  local host="$4"
  local cmd

  if [ "$pm" = "bun" ]; then
    cmd="bun run preview --host ${host} --port ${port}"
  else
    cmd="npm run preview -- --host ${host} --port ${port}"
  fi

  (cd "$workdir" && nohup sh -c "$cmd" > preview.log 2>&1 < /dev/null &)
  sleep 2

  ss -ltnp "sport = :${port}" >/dev/null 2>&1
}

start_preview() {
  local workdir="$1"
  local pm="$2"
  local port="$3"
  local host="$4"

  if start_preview_service "$workdir" "$pm" "$port" "$host"; then
    return 0
  fi

  warn "Starting preview with nohup fallback; service will not survive reboot"
  start_preview_nohup "$workdir" "$pm" "$port" "$host"
}

health_check() {
  local url="$1"
  local attempts=15
  local i

  for i in $(seq 1 "$attempts"); do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  return 1
}

json_is_valid() {
  local file="$1"
  if [ ! -s "$file" ]; then
    return 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$file" <<'PY' >/dev/null 2>&1
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    json.load(f)
PY
    return $?
  fi

  return 0
}

bootstrap_preview_data() {
  local workdir="$1"
  local data_root explain_dir
  local now_iso today

  now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  today="$(date -u +%Y-%m-%d)"

  for data_root in "${workdir}/dist/data" "${workdir}/public/data"; do
    explain_dir="${data_root}/explainability"
    mkdir -p "$data_root" "$explain_dir"

    ensure_json_file() {
      local rel="$1"
      local path="${data_root}/${rel}"

      if json_is_valid "$path"; then
        log "Preview data [$(basename "$(dirname "$data_root")")]: keeping existing ${rel}"
        return 0
      fi

      mkdir -p "$(dirname "$path")"

      case "$rel" in
      "signals.json")
        cat > "$path" <<EOF
{
  "timestamp": "${now_iso}",
  "generated_at": "${now_iso}",
  "regime": {"regime": "normal", "vix": null, "detected": null},
  "latest_prices": {},
  "current_positions": [],
  "target_allocations": {},
  "cash": 100000,
  "total_value": 100000,
  "recent_orders": [],
  "ml_signals": {
    "available": false,
    "timestamp": null,
    "predictions": {},
    "features": {},
    "grid_search": {
      "available": false,
      "timestamp": null,
      "top_allocation": null,
      "sharpe": null,
      "volatility": null
    }
  }
}
EOF
        ;;
      "dashboard.json")
        cat > "$path" <<EOF
{
  "prices": {},
  "regimes": [],
  "paper_portfolio": [],
  "generated_at": "${now_iso}"
}
EOF
        ;;
      "alerts.json")
        cat > "$path" <<EOF
{
  "alerts": [],
  "count": 0,
  "generated_at": "${now_iso}"
}
EOF
        ;;
      "stats.json")
        cat > "$path" <<EOF
{
  "asset_stats": {},
  "paper_portfolio": null,
  "spy_comparison": null,
  "generated_at": "${now_iso}"
}
EOF
        ;;
      "health.json")
        cat > "$path" <<EOF
{
  "cron_jobs": [],
  "data_freshness": {},
  "system_status": "warning",
  "generated_at": "${now_iso}",
  "error": "preview bootstrap placeholder"
}
EOF
        ;;
      "analytics.json")
        cat > "$path" <<EOF
{
  "status": "no_data",
  "message": "preview bootstrap placeholder",
  "generated_at": "${now_iso}",
  "data_points": 0,
  "date_range": {"start": null, "end": null},
  "drawdown": {
    "series": [],
    "max_drawdown": {
      "max_drawdown": 0,
      "max_drawdown_date": "${today}",
      "recovery_date": null,
      "underwater_days": 0,
      "peak_value": 100000,
      "trough_value": 100000
    }
  },
  "rolling_metrics": {
    "sharpe_63d": [],
    "sharpe_126d": [],
    "sharpe_252d": []
  },
  "benchmark_comparison": {
    "portfolio": {
      "start_date": "${today}",
      "end_date": "${today}",
      "start_value": 100000,
      "end_value": 100000,
      "total_return": 0,
      "cagr": null,
      "volatility": 0,
      "max_drawdown": 0,
      "sharpe": null
    }
  },
  "crisis_periods": []
}
EOF
        ;;
      "rebalance_health.json")
        cat > "$path" <<EOF
{
  "current_turnover_pct": 0,
  "max_daily_turnover": 0,
  "max_monthly_turnover": 0,
  "max_annual_turnover": 0,
  "daily_budget_used": 0,
  "monthly_budget_used": 0,
  "annual_budget_used": 0,
  "recent_rebalances": [],
  "cost_drag_bps": 0
}
EOF
        ;;
      "graduation.json")
        cat > "$path" <<EOF
{
  "criteria": [],
  "paper_trading": {
    "start_date": "${today}",
    "initial_capital": 100000,
    "current_value": 100000,
    "days_elapsed": 0,
    "days_required": 63
  },
  "readiness_pct": 0,
  "eligible": false
}
EOF
        ;;
      "adaptive_sizing.json"|"vixy_hedge.json"|"black_litterman.json"|"turnover_validator.json"|"regime_gate.json"|"tsmom.json"|"cross_asset_rv.json"|"explainability/explainability_latest.json")
        cat > "$path" <<EOF
{
  "generated_at": "${now_iso}",
  "status": "placeholder",
  "message": "preview bootstrap placeholder"
}
EOF
        ;;
      *)
        cat > "$path" <<EOF
{
  "generated_at": "${now_iso}",
  "status": "placeholder"
}
EOF
        ;;
      esac

      log "Preview data [$(basename "$(dirname "$data_root")")]: wrote placeholder ${rel}"
    }

    ensure_json_file "signals.json"
    ensure_json_file "dashboard.json"
    ensure_json_file "alerts.json"
    ensure_json_file "stats.json"
    ensure_json_file "health.json"
    ensure_json_file "analytics.json"
    ensure_json_file "rebalance_health.json"
    ensure_json_file "graduation.json"
    ensure_json_file "adaptive_sizing.json"
    ensure_json_file "vixy_hedge.json"
    ensure_json_file "black_litterman.json"
    ensure_json_file "turnover_validator.json"
    ensure_json_file "regime_gate.json"
    ensure_json_file "tsmom.json"
    ensure_json_file "cross_asset_rv.json"
    ensure_json_file "explainability/explainability_latest.json"
  done
}

cleanup_old_releases() {
  local keep="$1"
  local rel_dir="${REMOTE_BASE}/releases"
  local old
  old="$(ls -1dt "${rel_dir}"/* 2>/dev/null | tail -n +$((keep + 1)) || true)"
  if [ -z "$old" ]; then
    return 0
  fi

  while IFS= read -r line; do
    [ -n "$line" ] || continue
    rm -rf "$line"
  done <<EOF_OLD
$old
EOF_OLD
}

log "Deploy mode: ${MODE}"
log "Release dir: ${release_dir}"

pm="$(pick_pm "$PACKAGE_MANAGER")"
if [ "$pm" = "none" ]; then
  echo "No package manager found (bun/npm)" >&2
  exit 1
fi
log "Runtime package manager: ${pm}"

cd "$release_dir"

if [ "$RUN_GENERATOR" = "1" ]; then
  log "Running dashboard generator (best-effort)..."
  if command -v uv >/dev/null 2>&1; then
    PORTFOLIO_LAB_ENABLE_ML=0 PYTHONPATH="$release_dir" uv run python src/dashboard/generator.py \
      > data/generator_deploy.log 2>&1 || warn "generator failed; continuing"
  elif command -v python3 >/dev/null 2>&1; then
    PORTFOLIO_LAB_ENABLE_ML=0 PYTHONPATH="$release_dir" python3 src/dashboard/generator.py \
      > data/generator_deploy.log 2>&1 || warn "generator failed; continuing"
  else
    warn "python runtime not found; skipped generator"
  fi
fi

if [ "$SKIP_INSTALL" != "1" ]; then
  log "Installing dependencies..."
  if [ "$pm" = "bun" ]; then
    bun install --frozen-lockfile || bun install
  else
    if [ -f package-lock.json ]; then
      npm ci --no-audit --no-fund || npm install --no-audit --no-fund
    else
      npm install --no-audit --no-fund
    fi
  fi
fi

if [ "$SKIP_BUILD" != "1" ]; then
  log "Building frontend bundle..."
  if [ "$pm" = "bun" ]; then
    bun run build
  else
    npx vite build
  fi
fi

if [ "$MODE" = "preview" ]; then
  log "Stopping existing preview service/listener on port ${PREVIEW_PORT}"
  stop_preview "$PREVIEW_PORT"

  # Atomic current symlink switch
  ln -sfn "$release_dir" "${current_link}.new"
  mv -Tf "${current_link}.new" "$current_link"

  if [ "$BOOTSTRAP_PREVIEW_DATA" = "1" ]; then
    log "Bootstrapping preview dashboard JSON endpoints"
    bootstrap_preview_data "$current_link"
  fi

  log "Starting preview server on ${PREVIEW_HOST}:${PREVIEW_PORT}"
  if ! start_preview "$current_link" "$pm" "$PREVIEW_PORT" "$PREVIEW_HOST"; then
    warn "Preview start failed; attempting rollback"
    if [ -n "$previous_target" ] && [ -d "$previous_target" ]; then
      ln -sfn "$previous_target" "${current_link}.new"
      mv -Tf "${current_link}.new" "$current_link"
      stop_preview "$PREVIEW_PORT"
      start_preview "$current_link" "$pm" "$PREVIEW_PORT" "$PREVIEW_HOST" || true
    fi
    exit 1
  fi

  log "Running health check: ${HEALTH_URL}"
  if ! health_check "$HEALTH_URL"; then
    warn "Health check failed; attempting rollback"
    if [ -n "$previous_target" ] && [ -d "$previous_target" ]; then
      ln -sfn "$previous_target" "${current_link}.new"
      mv -Tf "${current_link}.new" "$current_link"
      stop_preview "$PREVIEW_PORT"
      start_preview "$current_link" "$pm" "$PREVIEW_PORT" "$PREVIEW_HOST" || true
    fi
    exit 1
  fi

else
  # production
  ln -sfn "$release_dir" "${current_link}.new"
  mv -Tf "${current_link}.new" "$current_link"

  if [ ! -d "$current_link/dist" ]; then
    echo "dist directory missing in release" >&2
    exit 1
  fi

  log "Publishing dist -> ${PROD_WEB_ROOT}"
  mkdir -p "$PROD_WEB_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$current_link/dist/" "$PROD_WEB_ROOT/"
  else
    tar -C "$current_link/dist" -cf - . | tar -C "$PROD_WEB_ROOT" -xf -
  fi

  if [ -n "$RELOAD_SERVICE" ]; then
    log "Reloading service: ${RELOAD_SERVICE}"
    systemctl reload "$RELOAD_SERVICE" || systemctl restart "$RELOAD_SERVICE"
  fi

  log "Running health check: ${HEALTH_URL}"
  health_check "$HEALTH_URL"
fi

cleanup_old_releases "$KEEP_RELEASES"

log "Deploy successful"
log "Current release: $(readlink "$current_link" 2>/dev/null || echo "$current_link")"
REMOTE_SCRIPT
  } | ssh "${SSH_ARGS[@]}" "$REMOTE" "bash -s"

  log "Deploy completed: ${release_id}"
}

main "$@"
