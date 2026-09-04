# Portfolio Lab Backup & Restore

> **Implementation status:** The commands in this guide apply once the
> repository recovery CLI is present and verified. They are an operator runbook,
> not evidence that a production recovery point or restore has been performed.

Operator-invoked, self-contained recovery of the lab application source, runtime
state, and served static release. This is **not** a redeployment procedure.

The recovery CLI is `scripts/portfolio_lab_recovery.py`. It creates a plaintext
`.tar` only at an explicitly selected destination that the operator attests is
encrypted at rest. It neither encrypts nor uploads the archive.

## Scope

| Included | Excluded |
|---|---|
| Git bundle for the recorded source commit | `.env`, `.env.local`, rclone and SkillWiki credentials |
| Quiesced `data/` tree, including SQLite WAL/SHM companions | Broker credentials/session state |
| Served static web tree, `_release.json`, and public data | Grok/Hermes/Claude agent state |
| `config/lab-app.env` only after a secret-key scan | SSH keys, Tailscale state, machine identity |
| Optional `logs/research-implement.md` | Caddy certificates/data, vault Git worktree, vault-sync runtime |
| Tasker unit/status and Portfolio Lab Caddy managed-block metadata | Unrelated host services and Caddy routes |

Secrets are separately supplied through the approved secret-management channel
only when a restore target needs them. Never commit or put recovery archives,
sidecars, manifests, or secrets in the repository or SkillWiki vault.

## Preconditions

- A named operator chooses an absolute destination on encrypted-at-rest storage.
- The source checkout is reproducible: all source changes are committed except
  the known generated runtime paths `data/ensemble_weights.json` and
  `data/vix_term_structure.json`.
- The source Tasker API is reachable; `create` captures its live
  `GET /api/tasker/status` response as `metadata/tasker-status.json` before
  Tasker is quiesced. The operator accepts a brief scheduled-job pause while
  Tasker drains and the recovery point is created.
- The source and archive destination have enough space for the served static
  tree, runtime data, Git bundle, and temporary archive.
- `git`, `systemctl`, and the Python runtime are available on the source
  host. `curl` is needed only for the attended development API drill shown
  below.
- Source creation, production restore and activation, and a development
  restore using `--start-dev-api` run as `root` (or through an explicitly
  approved equivalent wrapper); the recovery CLI invokes `systemctl` directly
  and does not elevate with `sudo`.

## Service controller and generation symlink options

The recovery CLI supports `--service-controller systemd|box-persist` (default `systemd`).
The `create` command remains source-side on sg01 and requires `systemd`; it rejects
`box-persist`. The `box-persist` controller option is valid only on restore targets
(both `dev` and `prod` targets, as well as `activate-prod`) where native supervisor
processes manage the Tasker lifecycle without systemd. In `dev` mode with `box-persist`,
Tasker runs in API-only shadow mode with dual scheduler-disable controls (`TASKER_DISABLE_SCHEDULER=1`
and `--no-scheduler`). In `prod` mode, staging proves the service is inactive, and
`activate-prod` requires explicit former-authority shutdown proof and enforces the
strict one-scheduler invariant. For full cursor-box host procedures, see the separate
runbook `scripts/PORTFOLIO_LAB_CURSOR_BOX_MIGRATION.md`.

When creating recovery archives where runtime data includes generation directories,
pass `--materialize-generations-current`. This option permits only the exact relative
directory symlink at `data/generations/current`, duplicates ordinary target file bytes
into the archive member tree, verifies metadata and member parity before creation, and
reconstructs the exact relative link during restore.

## Create and verify a recovery point

```bash
ARCHIVE_DIR=<ENCRYPTED_AT_REST_DESTINATION>
ARCHIVE="$ARCHIVE_DIR/portfolio-lab-$(date -u +%Y%m%dT%H%M%SZ).portfolio-lab-recovery.tar"

scripts/python_runtime.sh scripts/portfolio_lab_recovery.py create \
  --app-dir /root/projects/portfolio-lab \
  --web-root /var/www/portfolio-lab \
  --tasker-service portfolio-lab-tasker \
  --archive "$ARCHIVE" \
  --storage-encryption-attested \
  --service-controller systemd \
  --materialize-generations-current

scripts/python_runtime.sh scripts/portfolio_lab_recovery.py verify \
  --archive "$ARCHIVE"
```

Creation validates source and destination before stopping Tasker. It captures
metadata, calls `systemctl stop` so the service follows its SIGTERM drain
contract, creates the archive and SHA-256 sidecar, independently verifies the
archive, and restarts the source service if it was active before capture. A
recovery point is valid only if archive verification and required source restart
succeed.

Inspect the non-secret sidecar before transfer or restore:

```bash
cat "$ARCHIVE.sha256"
shasum -a 256 "$ARCHIVE"
```

If creation fails after Tasker stops, the CLI attempts to restart source Tasker
before returning a failure. Treat any create, verification, or restart failure
as an invalid recovery point. Retain the prior known-good archive.

## Development restore

Development restores require private, non-production paths and remain staged by
default. They never write Caddy configuration, reload Caddy, change DNS, or
start a scheduler.

```bash
scripts/python_runtime.sh scripts/portfolio_lab_recovery.py restore \
  --archive <ARCHIVE> \
  --app-dir /srv/portfolio-lab-recovery-dev \
  --web-root /srv/portfolio-lab-recovery-dev-www \
  --target-mode dev \
  --tasker-service portfolio-lab-tasker-recovery-dev
```

To inspect the Tasker API after validation, explicitly request the private
API-only service:

```bash
scripts/python_runtime.sh scripts/portfolio_lab_recovery.py restore \
  --archive <ARCHIVE> \
  --app-dir /srv/portfolio-lab-recovery-dev \
  --web-root /srv/portfolio-lab-recovery-dev-www \
  --target-mode dev \
  --tasker-service portfolio-lab-tasker-recovery-dev \
  --start-dev-api

curl -fsS http://127.0.0.1:8000/api/tasker/status
systemctl cat portfolio-lab-tasker-recovery-dev
```

The development unit must contain both `TASKER_DISABLE_SCHEDULER=1` and
`--no-scheduler`. A healthy API response does not prove scheduled jobs ran.
Validate the recovery report, Git commit, SQLite integrity, static manifest,
and archive sidecar before treating the drill as successful.

## Production restore and activation

A production restore only stages and validates content after it proves the
archived Tasker service is inactive. It does not start Tasker, change DNS, or
reload Caddy.

```bash
scripts/python_runtime.sh scripts/portfolio_lab_recovery.py restore \
  --archive <ARCHIVE> \
  --app-dir /root/projects/portfolio-lab \
  --web-root /var/www/portfolio-lab \
  --target-mode prod \
  --allow-production-paths \
  --tasker-service portfolio-lab-tasker
```

Before activation, separately provision required secrets, prove the former
authoritative Tasker scheduler is stopped, and confirm static release
provenance. `_release.json.source_git_sha` must exactly equal the archived Git
bundle commit. A public-data `generator_git_sha` may be an earlier commit only
when it is reachable in the archived bundle. If immutable source/static
provenance differs, activation remains blocked: rebuild and verify a coherent
static release from the archived source in an attended step. The recovery tool
never rebuilds automatically.

```bash
scripts/python_runtime.sh scripts/portfolio_lab_recovery.py activate-prod \
  --app-dir /root/projects/portfolio-lab \
  --web-root /var/www/portfolio-lab \
  --tasker-service portfolio-lab-tasker \
  --confirm-authoritative-activation \
  --former-authority-confirmed-stopped <FORMER_AUTHORITY_HOST>
```

After separately approved traffic activation, validate the public site on
**desktop and mobile**, SPA route fallback, `/_release.json`, `/data/index.json`,
`/data/signals.json`, `/api/tasker/status`, freshness, scheduler status, and
the expected kill/halt condition. Browser validation is an attended production
acceptance gate, not part of local tool tests.

## Rollback

Keep the source host, archive, sidecar, and restore-created rollback
directories until the acceptance window ends. If a target scheduler was
started, stop it before restoring previous authority. Do not attempt a
bidirectional runtime-data merge.

## Troubleshooting

- **Dirty source:** commit/stash source changes; only the two documented
generated runtime files may remain modified.
- **Destination rejected:** use an absolute operator-controlled destination
outside the app, web root, vault, and `/tmp`; provide
`--storage-encryption-attested` only after confirming encryption at rest.
- **Checksum, member, SQLite, or Git bundle failure:** verification fails before
target mutation; do not restore. Retain the previous verified archive and
investigate the failed evidence.
- **Static source SHA mismatch:** inspect with a development restore, then run
the attended static rebuild/verification process; do not bypass activation.
- **Missing secrets:** restore remains staged. Supply credentials through the
approved secret mechanism; never add them to the archive.
- **Source Tasker does not restart:** resolve source service health before
making another backup attempt.

> `make deploy-production` and `scripts/deploy-remote.sh --mode production`
> are deployment commands, **not** Portfolio Lab recovery procedures. They
> must not be used to restore a state-bearing recovery archive.

## Related documents

- `scripts/LAB_APP_DEPLOY.md` — normal host-native deployment and Tasker drain.
- `scripts/PORTFOLIO_LAB_CURSOR_BOX_MIGRATION.md` — cursor-box host-specific migration and shadow dry-run runbook.
- `projects/portfolio-lab/work/2026-08-14-portfolio-lab-backup-restore/` in
  the SkillWiki vault — planned implementation and attended drill evidence.
