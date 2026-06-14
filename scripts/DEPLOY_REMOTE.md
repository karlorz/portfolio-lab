# Remote Deploy Runbook

This project now ships with `scripts/deploy-remote.sh` for resilient remote deploys to preview or production hosts.

## Why this is safer

- Port-scoped process stop only (`ss ... sport=:PORT`), no broad `pkill` patterns.
- Staged releases under `REMOTE_BASE/releases/<release-id>`.
- Atomic `current` symlink switch (`current.new` -> `current`).
- Health checks before success.
- Preview rollback to previous release if start/health fails.
- Preview systemd service by default (`portfolio-lab-preview.service`) so
  the app restarts after host reboot instead of depending on an SSH-owned
  `nohup` process.
- Preview binds to `127.0.0.1` by default; Caddy is the public entrypoint
  and enforces basic auth.
- Preview JSON bootstrap: missing dashboard endpoints in `dist/data/*.json`
  (preview runtime) and `public/data/*.json` are created as valid placeholders
  while preserving existing real JSON files.
- Local client compatible with macOS/Linux, arm64/amd64 (uses `bash`, `ssh`, `tar`; `rsync` optional).

## Quick Start

### Preview deploy (private)

```bash
scripts/deploy-remote.sh \
  --host sg02 \
  --mode preview \
  --remote-base "${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab-preview}" \
  --preview-port 4173 \
  --preview-host 127.0.0.1 \
  --preview-service portfolio-lab-preview \
  --health-url http://127.0.0.1:4173/
```

Or via Makefile:

```bash
make deploy-preview DEPLOY_HOST=sg02 DEPLOY_REMOTE_BASE="${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab-preview}"
```

### Production deploy

```bash
scripts/deploy-remote.sh \
  --host sg01 \
  --mode production \
  --remote-base "${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab}" \
  --prod-web-root /var/www/portfolio-lab \
  --reload-service caddy \
  --health-url http://127.0.0.1/
```

Or via Makefile:

```bash
make deploy-production DEPLOY_HOST=sg01 DEPLOY_REMOTE_BASE="${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab}" DEPLOY_RELOAD_SERVICE=caddy
```

## Useful Flags

- `--sync-method auto|rsync|tar` (default `auto`)
- `--package-manager auto|bun|npm` (default `auto`)
- `--keep-releases 5`
- `--preview-host 127.0.0.1` (default; use `0.0.0.0` only for temporary private-port QA)
- `--preview-service portfolio-lab-preview` (default systemd unit name)
- `--no-preview-service` (fallback to `nohup`; not reboot-resilient)
- `--preview-memory-max 512M` (systemd `MemoryMax` for the preview app)
- `--skip-install`
- `--skip-build`
- `--run-generator` (best-effort, ML disabled)
- `--no-bootstrap-preview-data` (disable placeholder JSON bootstrap for preview)
- `--dry-run`

## Preview data bootstrap

Preview deploys run a bootstrap step by default to prevent SPA fallback for missing
dashboard JSON endpoints.

- Keeps existing valid JSON files in `dist/data/` and `public/data/` (real generated data wins).
- Creates placeholders only for missing/invalid files:
  - `signals.json`, `dashboard.json`, `alerts.json`, `stats.json`, `health.json`,
    `analytics.json`, `rebalance_health.json`, `graduation.json`,
    `adaptive_sizing.json`, `vixy_hedge.json`, `black_litterman.json`,
    `turnover_validator.json`, `regime_gate.json`, `tsmom.json`,
    `cross_asset_rv.json`, `explainability/explainability_latest.json`.

Disable if needed:

```bash
scripts/deploy-remote.sh --host sg02 --mode preview --no-bootstrap-preview-data
```

## Rotate preview basic-auth password (Caddy)

Run from local workstation:

```bash
NEW_PASS="LabPreview-$(date -u +%Y%m%d)-$(openssl rand -hex 6)"
echo "NEW_PASS=$NEW_PASS"
ssh sg02 "NEW_PASS='$NEW_PASS' bash -s" <<'EOF'
set -euo pipefail
backup="/etc/caddy/Caddyfile.bak.$(date +%Y%m%d%H%M%S)"
cp /etc/caddy/Caddyfile "$backup"
hash="$(caddy hash-password --plaintext "$NEW_PASS")"
printf ':80, :8080 {\n  basicauth {\n    preview %s\n  }\n  reverse_proxy 127.0.0.1:4173\n}\n' "$hash" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl reload caddy
echo "backup=$backup"
EOF
```

## Smoke check matrix (preview + Caddy auth)

```bash
# Tailnet endpoint auth behavior
curl -sS -o /dev/null -w "%{http_code}\n" http://100.116.104.17/                   # expect 401
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{http_code}\n" http://100.116.104.17/  # expect 200

# Preview app behind Caddy only.
ssh sg02 "curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:4173/"         # expect 200
curl -sS -m 5 -o /dev/null -w "%{http_code}\n" http://100.116.104.17:4173/          # expect 000/connection failure

# Assets/data through Caddy
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{http_code}\n" http://100.116.104.17/assets/index-*.js
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{http_code}\n" http://100.116.104.17/assets/index-*.css
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{http_code}\n" http://100.116.104.17/data/prices.json
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{http_code}\n" http://100.116.104.17/data/signals.json
curl -sS -u "preview:<PASSWORD>" -o /dev/null -w "%{content_type}\n" http://100.116.104.17/data/signals.json
```

## Troubleshooting

### SSH reachable but preview down

On remote host:

```bash
systemctl status portfolio-lab-preview --no-pager -l
journalctl -u portfolio-lab-preview -n 120 --no-pager
ss -ltnp | grep 4173
ps -ef | grep "vite preview" | grep -v grep
tail -n 80 "${DEPLOY_REMOTE_BASE:-/root/projects/portfolio-lab-preview}/current/preview.log"
curl -I http://127.0.0.1:4173/
```

If `portfolio-lab-preview.service` is missing, rerun preview deploy. The deploy
script writes/enables the unit when run as root on a systemd host.

### SSH appears down after sg02 reboot

First confirm whether this is actually an OpenSSH failure or a full host
poweroff/reboot:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 sg02 'date; uptime; systemctl is-active ssh; journalctl --list-boots --no-pager | tail -5'
ssh sg02 "journalctl -b -1 --since '10 minutes ago' --no-pager | grep -Ei 'power|shutdown|reboot|oom|killed|memory|ssh|hermes' || true"
```

The 2026-05-31 sg02 incident was a host poweroff sequence (`Power key pressed
short` / `Powering off...`) with concurrent memory pressure from
`hermes-gateway.service` peaking around 5.4 GiB. OpenSSH was stopped because
systemd was powering the machine down; it was not the root service failure.

### SSH service recovery (if needed)

```bash
systemctl status ssh || systemctl status sshd
systemctl restart ssh || systemctl restart sshd
ss -ltnp | grep ':22'
```
