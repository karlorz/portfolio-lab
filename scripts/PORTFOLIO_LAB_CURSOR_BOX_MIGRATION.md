# Portfolio Lab Cursor-Box Migration Runbook

> **Status:** Operator guidance covering the historical cursor-box dry run and
> cutover procedure (pre-cutover reference) plus the current post-cutover
> state. This document is **not evidence** that any migration, recovery
> archive, or cutover has occurred. Evidence of record lives in redacted
> manifests, recovery reports, and verified archives only.

## Current state (post-cutover)

- cursor-box is the current **authoritative** production host for Portfolio Lab.
- sg01 Tasker (scheduler) and the sg01 archive timer remain **stopped/disabled**.
- Cloudflare Access on the public origin remains **in place**.
- The old domain remains **non-authoritative**; no DNS, Caddy, or Cloudflare
  row change has been applied to it.
- Recycle persistence (production lifecycle surviving a box restart) and the
  seven-day acceptance observation remain **unproven/pending**.
- Attended gates remain required before: Access removal, any old-domain
  change, a recycle/restart exercise, a restore, or any authority change.
  This runbook alone authorizes none of them.

## Historical dry-run boundary

Before cutover, sg01 was the **authoritative** production host and cursor-box
was a **candidate-only** shadow host whose scheduler never ran, serving a
shadow site behind Cloudflare Access. The dry run never authorized DNS,
Caddy/Cloudflare row changes, or traffic cutover; those required separate
approval (see "Cutover and current attended gates" below). The rest of this runbook
documents that attended dry-run and cutover procedure for reference.

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

Historical dry-run requirement: Tasker on cursor-box had to be disabled by
**both** scheduler controls — environment (`TASKER_DISABLE_SCHEDULER=1`) and
argument (`--no-scheduler`) — and exactly zero candidate scheduler instances
were allowed; any observed scheduled start was blocking.

Post-cutover: the one-scheduler invariant applies to the current authority
(cursor-box); sg01 Tasker and the sg01 archive timer remain stopped/disabled.

## Static and API origins

- Static origin listens on loopback port **8001**; API origin listens on
  loopback port **8000**. Both must bind loopback only.
- Cloudflare must route `/api/*` (exact row, placed **before** the static
  catch-all) to port 8000 and the catch-all to port 8001.
- The dedicated `portfolio-lab-shadow` connector must use Cloudflare Tunnel
  transport `http2` as a mitigation for the observed QUIC/NAT expiry, not as a
  guarantee of connector or site availability. The cursor-box path repeatedly
  expired all four QUIC connections together, producing short public Error 1033
  windows even while the loopback origins stayed healthy. HTTP/2 does not cover
  whole-host or container pauses. Before switching an existing connector, run
  a bounded second-connector canary with `--protocol http2` and require all four
  connections to register successfully. Restart only the dedicated connector
  after the canary passes; do not restart Tasker, the static origin, broker
  services, or the separate shared cursor-box tunnel. Durable public
  availability additionally requires a redundant static edge origin independent
  of cursor-box.
- The broker brief route must remain ahead of the static catch-all:
  `/broker-brief*` routes to loopback port **8011**. Only the five approved
  HTML files belong in that webroot; private broker JSON/XML artifacts stay
  outside it.
- Cloudflare Access was required during the dry run and remains in place
  post-cutover; removing Access protection requires separate attended approval.

## Recovery archives

- Create archives with the exact recovery flag
  `--materialize-generations-current` so the `data/generations/current`
  relative symlink is preserved as ordinary bytes with metadata/member parity;
  restore reconstructs the exact relative link.
- Historical dry-run guidance: after restore, verify archive
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
  `Comparison passed; attended operational gates remain required.`; on blocked,
  `Dry run blocked` followed by the failed check IDs and
  `Read-only comparison: this tool did not change authority or scheduler state.`
  The comparison tool is read-only: it never changes authority or scheduler
  state.

## Browser verification

- Verify on desktop and mobile browsers with repeated real interaction across:
  - SPA root and shared routes (`/`, `/signals`, `/models`, `/status`, and direct deep-link route navigation).
  - API, data, and release probes: `/_release.json`, `/data/index.json`, `/data/signals.json`, and `/api/tasker/status`.
  - Component and network states: loading states, empty/unavailable dataset states, and graceful error presentation.
  - Console and layout hygiene: zero page errors, zero browser console exceptions, and no horizontal or vertical document overflow.
  - Driven interaction: at least one interactive form input or driven navigation causing an expected visible DOM/state change.
- During the dry run, recheck every existing hostname and service, including
  then-authoritative `lab.karldigi.dev` and sg01 origins, before and after each
  candidate verification window.

## Cutover and current attended gates

- The cutover was separately approved and required all of: cursor-box
  persistence proof across restart, former-authority (sg01) scheduler
  stopped, a fresh recovery archive, sole scheduler activation on cursor-box,
  and explicit Access-removal approval.
- Post-cutover, attended gates remain required for: Access removal, any
  old-domain change, a recycle/restart exercise, a restore, and any authority
  change. Each requires separate operator approval and must not be performed
  from this runbook alone.

## Rollback

- Rollback always stops cursor-box first, then restores the whole prior state
  from the archived recovery point. No bidirectional state merge is ever
  performed.
- Restore remains an attended gate post-cutover: it requires separate
  operator approval and a verified recovery archive.

## Seven-day observation

- **Status: pending/unproven until a full seven-day acceptance window
  completes.** Recycle persistence (production box-persist lifecycle surviving
  a host restart) is likewise unproven until exercised and observed.
- Observe for seven days across all operational dimensions:
  - Scheduler identity and one-scheduler invariant proof across runs.
  - Expected job executions, completed runs, and expected failures or dead-letter counts.
  - Public data and runtime state freshness within defined freshness envelopes.
  - Static release and runtime state consistency between artifacts.
  - Kill-switch state, incident trigger/resolve loops, and safety channels.
  - Disk headroom (including host APFS/ext4 volumes) and memory utilization over time.
  - API endpoint and public origin availability metrics.
  - Verified, reproducible recovery points and sidecars.
- Old-domain status: `lab.karldigi.dev` remains **non-authoritative** awaiting
  a final attended choice (one of):
  - `lab.karldigi.dev` redirect,
  - retirement response, or
  - static archived notice.
  No old-domain change may be applied without attended approval.

## Exclusion note

- Never include real credentials, archive paths, sidecars, private host
  values, or live tokens in this runbook or in evidence manifests.
