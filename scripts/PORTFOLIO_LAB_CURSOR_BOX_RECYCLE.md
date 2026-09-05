# Portfolio Lab Cursor-Box Controlled Recycle Runbook

> **PREPARATION ONLY — NOT AUTHORIZATION.** This runbook documents the
> attended precondition, observation, and evidence contract for a controlled
> recycle (host restart) of the cursor-box production host. It authorizes no
> reboot, no shutdown, no stop/start, no restore, no activation, and no
> authority change. Most checks below are read-only probes; the only writes
> are the bounded daily-evidence collector runs and the attended sg01 proof
> refresh (bounded evidence/proof artifacts, explicitly attended), and the
> operator recycle action is the only lifecycle mutation, requiring separate
> attended confirmation at execution time. This document is not evidence
> that any recycle occurred; evidence of record is the attended recycle
> proof (`recycle.json`) accepted by the seven-day acceptance checker.

## Scope and attended approval gate

- Host: **cursor-box** (Alpine, user `box`, no systemd/cron/OpenRC; process
  supervision by the box-persist controllers only). sg01 is the former
  authority and stays stopped/disabled throughout.
- A recycle is a host restart/power cycle of cursor-box that is expected to
  leave the production box-persist lifecycle intact (Tasker one-scheduler,
  static origin, tunnel, API, archive, evidence).
- **Attended approval gate:** a recycle requires separate operator approval
  at execution time. Nothing in this runbook is standing approval. The
  operator who executes the recycle must be present and must confirm the
  stop conditions immediately before the action and the post-recycle checks
  immediately after.
- Runbook placement: `scripts/PORTFOLIO_LAB_CURSOR_BOX_RECYCLE.md` in the
  repository; it is documentation only and ships no executable behavior.

## Host layout (cursor-box)

| Path | Meaning |
|---|---|
| `/home/box/.local/share/portfolio-lab` | state root |
| `app/` | repository checkout (scripts under `app/scripts/`) |
| `www/` | production static web root |
| `run/` | box-persist process state, archive stamps/log, activation + authority proofs |
| `archives/` | local archive tarballs + `.sha256` sidecars (daily continuity tar; v2 recovery archive when produced) |
| `evidence/` | daily evidence root (day dirs + `recycle.json` only) |
| `/home/box/.local/bin/portfolio-lab-box-persist` | Tasker lifecycle controller |
| `/home/box/.local/bin/portfolio-lab-static-persist` | static origin lifecycle controller |
| `/home/box/.local/share/box-persist/ensure.sh` | boot/lifecycle ensure hook (managed blocks; also invokes the once-per-UTC-day archive) |

Ports: API origin loopback `127.0.0.1:8000`; static origin loopback
`127.0.0.1:8001`. Public origin: `lab.termolo.com` via Cloudflare
(`/api/*` -> 8000 before the catch-all -> 8001).

## Preconditions (all must hold; re-verified in order immediately before stop conditions)

Probes below are read-only. The only writes anywhere in this runbook are the
bounded daily-evidence collector runs and the attended sg01 proof refresh
(both explicitly attended). `status --read-only` never cleans stale PID/state
records and never mutates services.

1. **Fresh same-day successful archive + SHA (continuity evidence only).**
   `run/s3-archive-last-utc-day` equals today's `%Y%m%d`; `run/s3-archive.log`
   latest success line is `published ... sha256=<64-hex>` on today's UTC day;
   the local tarball `archives/portfolio-lab-<STAMP>.tar` and its `.sha256`
   sidecar exist (mode 0600); the S3 object key
   `daily/YYYY/MM/DD/portfolio-lab-data-<utc>.tar` + `.sha256` published.
   This is a plain quiesced data tar: **not** a recovery archive and **not**
   verifiable by `portfolio_lab_recovery.py`. It is continuity evidence only
   — the daily-evidence `archive` category consumes its day and SHA; it is
   never a restore source.
2. **Verified `portfolio-lab-recovery/v2` recovery archive + sidecar
   (rollback readiness).** A recovery archive
   `<name>.portfolio-lab-recovery.tar` (schema `portfolio-lab-recovery/v2`,
   embedded `recovery-manifest.json`) and its `.sha256` sidecar must exist
   and verify. It must be created by compatible recovery tooling —
   `portfolio_lab_recovery.py create --materialize-generations-current`
   (preserving the `data/generations/current` link; `create` is source-side
   with the systemd controller per recovery-tooling constraints) — for the
   standard production layout, embedding no private paths, credentials, or
   live tokens. **Nothing scheduled produces it today** (the sg01 timer and
   the daily archive produce only the continuity tar from precondition 1);
   it requires an attended `create` + verify. Verify read-only:
   `python3 app/scripts/portfolio_lab_recovery.py verify --archive <RECOVERY_V2_ARCHIVE>`
   (the shipped `verify` CLI takes only `--archive`; the sidecar is derived
   at `<archive>.sha256`). **Absence of a verified v2 archive blocks the
   recycle.**
3. **Current daily evidence pass, not warning.** Today's UTC day evidence
   collects overall `pass` (every category `pass`), and the read-only
   acceptance report shows no blockers and no extend reasons, at worst
   `ready_except_recycle` when recycle.json is absent — note that verdict
   occurs only **without** `--require-recycle-proof` (with the flag and an
   absent proof the verdict is `extend`, which blocks). This precondition
   run uses the default flags:
   `python3 app/scripts/portfolio_lab_evidence_acceptance.py --evidence-root /home/box/.local/share/portfolio-lab/evidence`
   A warning (`extend`) or `investigate` blocks recycle preparation.
4. **Fresh mode-0600 sg01 proof, all false.** `run/former-authority-proof.json`
   (schema `portfolio-lab-former-authority-proof/v1`): regular non-symlink
   file owned by the collector uid, exactly mode 0600, `host_label` `sg01`,
   `collected_at` fresh under the freshness max age (default 21600 s) and not
   future-dated, and `tasker.active`, `tasker.enabled`,
   `archive_timer.active`, `archive_timer.enabled` all exactly `false`.
   Refresh it attended before the recycle; never assume an aged proof. The
   collector run that consumes it must use a pinned `--now` on the current
   UTC day no earlier than the proof's `collected_at`.
5. **Exact one scheduler.** Tasker controller reports `scheduler_instances: 1`:
   `/home/box/.local/bin/portfolio-lab-box-persist status --read-only --mode production --app-dir /home/box/.local/share/portfolio-lab/app --web-root /home/box/.local/share/portfolio-lab/www --service-name portfolio-lab-tasker`
   (`portfolio-lab-box-persist/v1`).
6. **Tasker active/exact** (same probe): `state: "active"` and
   `identity_exact: true`.
7. **Static active/exact**:
   `/home/box/.local/bin/portfolio-lab-static-persist status --read-only --mode production --web-root /home/box/.local/share/portfolio-lab/www --service-name portfolio-lab-static`
   (`portfolio-lab-static-persist/v1`, `state: "active"`,
   `identity_exact: true`).
8. **Exactly one tunnel connection.** Attended, host-specific operator
   procedure — the tunnel contract is not repo-managed, so no command
   template is supplied here: exactly one live tunnel connection/process
   serving `lab.termolo.com` to the loopback origins. Record the judgment
   `pass`/`fail` attended; a count other than one blocks.
9. **API and static HTTP 200.** `http://127.0.0.1:8000/api/tasker/status`
   and `http://127.0.0.1:8001/` both return HTTP 200 (loopback probes;
   bounded, e.g. 10 s default like the collector).
10. **Disk >= 15 GiB free.** `resources` evidence `disk.status: "pass"` with
    `free_bytes >= 15 GiB` (15 * 1024^3); below that the collector warns and
    the acceptance criterion fails.
11. **Boot ensure wiring independently confirmed.** The box-persist ensure
    hook (`/home/box/.local/share/box-persist/ensure.sh`, managed blocks
    installed by the controllers' `install-ensure-hook`) must be wired to run
    after a host boot. Confirm by attended read-only inspection, e.g.:
    `# read-only inspection placeholder: operator inspects ensure.sh managed blocks and its boot invocation per their existing procedure`
    No setup or wiring changes are made here. Active controller status after
    a recycle is consistent with ensure having run, but the wiring itself
    must be confirmed before the recycle — never inferred solely from active
    status.
12. **Backups and rollback commands identified.** Rollback uses the verified
    `portfolio-lab-recovery/v2` recovery archive and sidecar from
    precondition 2 — the daily continuity tar is never a restore source.
    Pre-identified rollback (not run from this runbook; requires separate
    approval per Failure path):
    `portfolio_lab_recovery.py restore --archive <RECOVERY_V2_ARCHIVE> --app-dir <app> --web-root <www> --target-mode prod --allow-production-paths --service-controller box-persist`,
    always after stopping cursor-box Tasker with the box-persist controller
    `stop` action first. Confirm the identifiers (tasker service
    `portfolio-lab-tasker`, static `portfolio-lab-static`, exact app/web
    paths) on the operator's copy before the recycle.

## Recycle proof artifact contract (`portfolio-lab-recycle-proof/v1`)

Written **only after both pre- and post-recycle observations exist** and only
if both phases fully pass. Never create it early, partially, or with
placeholders — an invalid or prematurely present `recycle.json` makes the
acceptance checker return `investigate`.

- Path: `<evidence-root>/recycle.json`, i.e.
  `/home/box/.local/share/portfolio-lab/evidence/recycle.json`.
- File: regular non-symlink, exactly mode **0600**, at most **262144 bytes**,
  valid UTF-8 JSON object, no secrets, no paths, no captured output.
- Window: `performed_at` must be an ISO-8601 timestamp with a UTC offset
  inside the current seven-day acceptance window (the seven UTC days ending
  at the checker's `--end-day`).
- Envelope exactly:

```json
{
  "schema": "portfolio-lab-recycle-proof/v1",
  "category": "recycle",
  "status": "pass",
  "collected_at": "<ISO-8601 with UTC offset>",
  "details": {
    "performed_at": "<ISO-8601 with UTC offset, inside the 7-day window>",
    "pre_recycle": {
      "scheduler_instances": 1,
      "tasker": "pass",
      "static": "pass",
      "tunnel": "pass",
      "api": "pass"
    },
    "post_recycle": {
      "scheduler_instances": 1,
      "tasker": "pass",
      "static": "pass",
      "tunnel": "pass",
      "api": "pass"
    }
  }
}
```

- Field meaning: `scheduler_instances` is exactly `1` in both phases;
  `tasker`/`static`/`tunnel`/`api` each carry the attended phase judgment,
  `pass` only when the corresponding observation in that phase passed.
- Validate read-only before finalizing:
  `python3 app/scripts/portfolio_lab_evidence_acceptance.py --evidence-root /home/box/.local/share/portfolio-lab/evidence --require-recycle-proof`
  must produce verdict `accept` once the proof is complete (phases fill only
  after the post-recycle checks). With the proof absent, this flagged run
  yields `extend`; an unflagged run (`--require-recycle-proof` off) yields
  `ready_except_recycle` — that verdict occurs only without the flag.

## Stop conditions before the recycle (all, in order)

1. Every precondition above re-verified and true immediately before the
   action (fresh probes, not cached results).
2. No incident in progress and no kill-authority/stop state observed.
3. The daily evidence collector is not mid-run (its flock is free:
   `run/portfolio-lab-daily-evidence.lock` uncontended).
4. Today's UTC archive cycle completed (precondition 1) and the recycle will
   not straddle an expected archive invocation in a way the operator has not
   accounted for.
5. The current UTC day and the seven-day acceptance window still align so a
   post-recycle `performed_at` can land inside the window.
6. The operator has the verified v2 recovery archive (precondition 2), the
   continuity tar, and the rollback command identities at hand, and another
   human or recorded approval exists per the attended approval gate.

If any stop condition fails: stop here; do not proceed to the operator
action.

## Operator action (placeholder — supplied/confirmed attended)

> **The actual recycle command is intentionally not embedded here.** The host
> restart/power-cycle command for cursor-box must be supplied and confirmed
> by the attending operator at execution time, and its scope must be exactly
> the cursor-box host restart. This runbook invents no reboot or shutdown
> command, automates none, and authorizes none. Execute it only after all
> stop conditions hold, with the attending operator present.

## Post-recycle bounded wait and check order

Bounded means: each probe has a fixed per-attempt timeout (default 10 s,
matching the collector) and the whole sequence has a fixed total budget
agreed before the action (default 30 min); on budget exhaustion go to the
Failure path. All probes read-only; the collector rerun and the attended sg01
proof refresh (steps 7 and 8) are the only writes.

1. **Host reachable.** cursor-box responds to an attended reachability probe
   (e.g. SSH/ICMP, bounded).
2. **box-persist ensure has run.** The boot ensure wiring was independently
   confirmed as precondition 11; here verify its observable consequence:
   production Tasker and static are only restarted from inactive by the
   ensure hook (production `ensure` requires the exact
   `run/production-activation.json` marker match), so both controllers
   reporting `active` with `identity_exact: true` plus the activation marker
   still matching exactly is the post-boot confirmation.
3. **Tasker active/exact, exactly one scheduler** (precondition 5/6 probes).
4. **Static active/exact** (precondition 7 probe).
5. **Exactly one tunnel connection** (precondition 8 attended, host-specific
   procedure).
6. **API and static HTTP 200** (precondition 9 probes).
7. **Evidence collection rerunnable/read-only.** Re-run today's collector
   with the current UTC day pinned (`--now` on that day): same-day reruns
   replace files atomically and observation-derived files are byte-identical
   under the same inputs (`resources.json` is a live snapshot — the one
   exception); the collector never starts/stops services. The rerun must
   complete and today's evidence must again be overall `pass`.
8. **sg01 remains disabled via fresh attended proof, then re-collect, in
   this order.** (a) Refresh `run/former-authority-proof.json` attended
   first: regular non-symlink file owned by the collector uid, exactly mode
   0600, fresh `collected_at`, all four booleans false, `host_label` `sg01`.
   (b) Choose the current RFC3339 `--now`: on the current UTC day and no
   earlier than the proof's `collected_at` (a `--now` earlier than
   `collected_at` future-dates the proof and makes the `authority` category
   warn). (c) Then re-run the collector with that `--now` so the current
   day's `authority` category is `pass` with `host_label: "sg01"`. Never
   reuse the stale pre-recycle pinned time.

Only after all eight pass (and a full seven-day `pass` window with the
written proof) is recycle persistence evidenced.

## Failure path

On any stop-condition failure, budget exhaustion, or any post-recycle check
not passing:

- **Stop.** Do not proceed, do not retry indefinitely, do not normalize
  degraded state to pass.
- **Do not activate sg01 and do not restore automatically.** Bringing sg01
  back or restoring state requires a separate attended approval; never
  automate it.
- **Preserve evidence.** Keep every day directory and any proof as-is; never
  delete or overwrite evidence, and never add files to the evidence root
  beyond the allowed day directories plus `recycle.json` (extra entries make
  the checker `investigate`). Record observations in timestamped operator
  notes outside the evidence root.
- **Rollback uses the verified `portfolio-lab-recovery/v2` recovery archive**
  and sidecar (precondition 2, and/or its S3 copy), never a live state merge
  and never the daily continuity tar.
- **Rollback always stops cursor-box Tasker first** (box-persist controller
  `stop` action, attended) before restoring, and restore itself requires a
  separate attended approval — the pre-identified command in precondition 12
  is the only acceptable restore form.

## Acceptance boundary

The recycle proof, when accepted, evidences only recycle persistence within
the seven-day window. It does **not** approve, and does not itself approve:
Cloudflare Access removal, any old-domain change, final acceptance of the
migration, or any authority change. Those remain separate attended gates.

## Referenced contracts (read-only)

| Contract | Source |
|---|---|
| `portfolio-lab-box-persist/v1` status | `scripts/portfolio_lab_box_persist.py` (`status --read-only`) |
| `portfolio-lab-static-persist/v1` status | `scripts/portfolio_lab_static_persist.py` (`status --read-only`) |
| `portfolio-lab-daily-evidence/v1` | `scripts/portfolio_lab_daily_evidence.py` (read-only collector) |
| `portfolio-lab-evidence-acceptance/v1` | `scripts/portfolio_lab_evidence_acceptance.py` (read-only checker; `--require-recycle-proof`) |
| `portfolio-lab-recycle-proof/v1` | acceptance checker `_recycle_problems`/`validate_recycle` + `tests/test_portfolio_lab_evidence_acceptance.py` |
| `portfolio-lab-former-authority-proof/v1` | daily evidence `collect_authority` (owner-uid, non-symlink, exact 0600, freshness) |
| Daily continuity tar + SHA | `scripts/cron/portfolio-lab-cursor-box-s3-archive.sh`, `scripts/portfolio_lab_s3_archive.py` (evidenced by `archive` category; not a recovery archive) |
| `portfolio-lab-recovery/v2` create/verify | `scripts/portfolio_lab_recovery.py` (`create --materialize-generations-current`, `verify` derives sidecar at `<archive>.sha256`; schema at `SCHEMA_VERSION`) |
| Restore/rollback | `scripts/portfolio_lab_recovery.py` (`restore --target-mode prod --service-controller box-persist`; uses the v2 recovery archive) |