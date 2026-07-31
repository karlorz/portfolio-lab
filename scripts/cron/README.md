# Portfolio-Lab Cron Scripts

Post-incident cron design — staggered, guard-protected, documented for agent review.

## Incident (2026-05-16)

Sustained loadavg 25–48 on sg01. Root cause: autonomous agent cron job dispatching LLM-powered Claude Code sessions every 15 minutes, each cycle 10+ min, overlapping runs consuming all CPU. Hermes gateway crashed 7× (SystemExit 75).

## Guard Architecture (cron_guard.sh)

Every script sources `cron_guard.sh` which provides 4-layer defense:

| Layer | Mechanism | Threshold |
|-------|-----------|-----------|
| 1. Load gate | Defer if 1-min loadavg > 5 | `CRON_GUARD_MAX_LOAD=5` |
| 2. Overlap | flock on `/tmp/portfolio-lab-locks/<name>.lock` | Exclusive lock |
| 3. Memory | `ulimit -v` | 3 GB (3072 MB) |
| 4. Timeout | Background watchdog → SIGTERM → SIGKILL | Per-job (see below) |

**Critical gap fixed**: The autonomous job must stay in Hermes `no-agent` mode so the guarded script runs directly. Use `scripts/cron/configure_autonomous_agent_job.py` after Hermes job edits to restore `script=portfolio-lab-autonomous-agent.sh`, `no_agent=true`, and the repo-backed wrapper copy.

## Job Inventory

| Script | Makefile Target | Guard Timeout | Duration | Schedule |
|--------|----------------|---------------|----------|----------|
| `portfolio-lab-health-monitor.sh` | `make health` | 120s | 120s | `0,30 * * * *` |
| `portfolio-lab-data-pipeline.sh` | `make data` | 300s | 5 min | `5 * * * *` |
| `portfolio-lab-dashboard.sh` | `make dashboard` | 180s | 3 min | `15 * * * *` |
| `portfolio-lab-strategy-eval.sh` | `make eval` | 600s | 10 min | `20 */2 * * *` |
| `portfolio-lab-research-agent.sh` | `make research` | 300s | 5 min | `25 */2 * * *` |
| `portfolio-lab-wiki-sync.sh` | `make wiki-sync` | 120s | 2 min | `35 */2 * * *` |
| `portfolio-lab-autonomous-agent.sh` | *(Hermes no-agent)* | 300s | 60s¹ | `40 */2 * * *` |
| `portfolio-lab-position-sync.sh` | `make sync` | 60s | 60s² | `55 */2 * * *` |
| `portfolio-lab-app-build.sh` | `make build` | 600s | — | **PAUSED**³ |

¹ Pre-flight checks only. LLM dispatch removed post-incident; the configurator enforces no-agent mode.
² Placeholder — no broker API wired yet.  
³ No web server consumers found (`/var/www/portfolio-lab` doesn't exist, no caddy/nginx config).

The app-build wrapper is the one memory-cap exception: Bun/Vite requires an 8 GB
virtual-address-space limit on the deployment host, so
`portfolio-lab-app-build.sh` overrides the shared 3 GB default with
`CRON_GUARD_MEMORY_MB=8192`. The other cron jobs retain the 3 GB guard.

## Staggered Schedule Design

**Principle**: No two jobs fire at the same minute. Minimum 5-minute gap between jobs. Heavy jobs (eval 10 min) get clear buffer on both sides.

### Even Hours (0,2,4,6,8,10,12,14,16,18,20,22)
```
:00  health        30m   120s
:05  data          hourly 300s
:15  dashboard     hourly 120s
:20  eval          2h     600s    ← heaviest
:25  research      2h     300s
:30  health        30m   120s
:35  wiki-sync     2h     120s
:40  autonomous    2h     60s     ← pre-flight only
:55  position-sync 2h     60s
```

### Odd Hours (1,3,5,7,9,11,13,15,17,19,21,23)
```
:00  health        30m   120s
:05  data          hourly 300s
:15  dashboard     hourly 120s
:30  health        30m   120s
```

## Frequency Rules

When adding new cron jobs or modifying existing ones:

1. **Interval**: 30 minutes minimum, 2 hours preferred for non-critical jobs.
2. **Staggering**: No two jobs at the same minute. Pick an unused minute slot.
3. **Heavy jobs first**: Jobs with runtime >3 minutes get slots ending in 0 or 5, with clear 5-minute buffer on both sides.
4. **Three-file sync**: When adding a job, update all three: Makefile target, `src/cron_compat.py` → `CRON_TARGETS` + `CRON_EXPECTED_DURATIONS`, and crontab file.
5. **Guard required**: Every new script must `source scripts/cron_guard.sh` and call `cron_guard_start`/`cron_guard_end`.

## Three Backend Support

| Backend | Env Var | How |
|---------|---------|-----|
| Hermes | `CRON_BACKEND=hermes` (default) | Jobs managed via `hermes cron` CLI |
| System crontab | `CRON_BACKEND=crontab` | `crontab "${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}/crontab"` |
| Manual | `CRON_BACKEND=manual` | `make <target>` from terminal |

## Related Docs

- `../cron_guard.sh` — Guard library source
- `../cron_update.py` — Cron status JSON updater
- `../cron_verify.py` — Status file integrity checker
- `../../src/cron_compat.py` — Backend feature flags + target registry
- `../../crontab` — System crontab file (standalone mode)
- `../../CLAUDE.md` — Project context (cron section)
- Wiki: `projects/portfolio-lab/work/2026-05-16-cron-job-audit/` — Full audit spec + tasks
- Wiki: `projects/portfolio-lab/compound/` — Compound research pages
