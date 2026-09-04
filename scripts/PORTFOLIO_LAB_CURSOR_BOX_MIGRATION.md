# Portfolio Lab Cursor-Box Migration Runbook

> **Status:** Operator guidance for the cursor-box dry run and migration
> procedure. This document is **not evidence** that any migration, recovery
> archive, or cutover has occurred. Evidence of record lives in redacted
> manifests, recovery reports, and verified archives only.

## Authority and dry-run boundary

- sg01 remains the **authoritative** production host for Portfolio Lab.
- cursor-box is a **candidate-only** shadow host. Its scheduler must never run.
- During the dry run, cursor-box serves a shadow site behind Cloudflare Access.
- This runbook never authorizes DNS, Caddy/Cloudflare row changes, or traffic
  cutover; those require separate approval (see "Pause before cutover").

## Host constraints

- Alpine 3.22, dedicated user `box`.
- No systemd, no cron, no OpenRC, no runit, no s6: process supervision is
  performed by the box-persist controller only.
- No `apk`, no `sudo`, no Docker, and no base-system mutation.
- All tooling is user-owned below `/home/box/.local`; the bootstrap provides
  verify and uninstall commands and must be verified before first use.
- Bootstrap guidance: when a bare Alpine host environment provides neither
  `curl` nor `wget`, the bootstrap procedure explicitly requires the operator
  to provide `--stage0-python-archive=PATH`. This stage-0 Python archive is
  pre-fetched, checksum-verified, and transferred outside Git and the vault.

## Paths

| Path | Meaning |
|---|---|
| `/home/box/.local/share/portfolio-lab` | cursor-box state root |
| `app/` | repository checkout of the archived source commit |
| `www-candidate/` | candidate static web root (dry-run shadow site) |
| `www/` | production static web root (post-cutover) |
| `runtime/` | runtime data (SQLite, generations, config) |
| `run/` | box-persist process state |

## Tasker candidate controls

- Tasker on cursor-box must be disabled by **both** scheduler controls:
  environment (`TASKER_DISABLE_SCHEDULER=1`) and argument (`--no-scheduler`).
- One-scheduler invariant: exactly zero Tasker scheduler instances may exist on
  the candidate during the dry run; any observed scheduled start is blocking.
- sg01 keeps exactly one authoritative scheduler instance at all times.

## Static and API origins

- Static origin listens on loopback port **8001**; API origin listens on
  loopback port **8000**. Both must bind loopback only.
- Cloudflare must route `/api/*` (exact row, placed **before** the static
  catch-all) to port 8000 and the catch-all to port 8001.
- Cloudflare Access is required during the dry run; removing Access protection
  requires separate cutover approval.

## Recovery archives

- Create archives with the exact recovery flag
  `--materialize-generations-current` so the `data/generations/current`
  relative symlink is preserved as ordinary bytes with metadata/member parity;
  restore reconstructs the exact relative link.
- Seed and candidate verification gates: after restore, verify archive
  sidecar, Git bundle commit, SQLite integrity, static manifest
  `_release.json.source_git_sha`, scheduler disable controls, and loopback
  bindings before treating the candidate as dry-run ready.

## Evidence collection and comparison

- Evidence manifests use schema `portfolio-lab-migration-evidence/v1`
  (exact keys: `schema_version`, `role`, `host`, `collected_at`, `git`,
  `recovery`, `sqlite`, `digests`, `release`, `allocation`, `safety`,
  `tasker`, `schemas`, `freshness`, `endpoints`, `authority`).
- Collect evidence as an attended step on each host; never embed credentials
  or live secrets in manifests. Example safe command:

```bash
python3 scripts/portfolio_lab_migration_compare.py \
  --source sg01-evidence/ \
  --candidate cursor-box-evidence/ \
  --output-json comparison.json \
  --output-markdown comparison.md
```

- Differences are classified as: `expected`, `explained`,
  `blocking`, or `unavailable`.
- Terminal statements: on pass,
  `Dry run passed; cutover approval required.`; on blocked,
  `Dry run blocked` followed by the failed check IDs and
  `Retained safe state: sg01 remains authoritative; cursor-box scheduler remains disabled.`

## Browser verification

- Verify on desktop and mobile browsers with repeated real interaction across:
  - SPA root and shared routes (`/`, `/signals`, `/models`, `/status`, and direct deep-link route navigation).
  - API, data, and release probes: `/_release.json`, `/data/index.json`, `/data/signals.json`, and `/api/tasker/status`.
  - Component and network states: loading states, empty/unavailable dataset states, and graceful error presentation.
  - Console and layout hygiene: zero page errors, zero browser console exceptions, and no horizontal or vertical document overflow.
  - Driven interaction: at least one interactive form input or driven navigation causing an expected visible DOM/state change.
- Recheck every existing hostname and service, including authoritative `lab.karldigi.dev` and sg01 origins, before and after each candidate verification window.

## Pause before cutover

- Cutover is separately approved and requires all of:
  cursor-box persistence proof across restart, former-authority (sg01)
  scheduler stopped, a fresh recovery archive, sole scheduler activation on
  cursor-box, and explicit Access-removal approval.

## Rollback

- Rollback always stops cursor-box first, then restores the whole prior state
  from the archived recovery point. No bidirectional state merge is ever
  performed.

## Seven-day observation

- Observe for seven days across all operational dimensions:
  - Scheduler identity and one-scheduler invariant proof across runs.
  - Expected job executions, completed runs, and expected failures or dead-letter counts.
  - Public data and runtime state freshness within defined freshness envelopes.
  - Static release and runtime state consistency between artifacts.
  - Kill-switch state, incident trigger/resolve loops, and safety channels.
  - Disk headroom (including host APFS/ext4 volumes) and memory utilization over time.
  - API endpoint and public origin availability metrics.
  - Verified, reproducible recovery points and sidecars.
- Final old-domain choice (one of):
  - `lab.karldigi.dev` redirect,
  - retirement response, or
  - static archived notice.

## Exclusion note

- Never include real credentials, archive paths, sidecars, private host
  values, or live tokens in this runbook or in evidence manifests.
