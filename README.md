# portfolio-lab

## Project Knowledge

Project specs, plans, compound notes, and ADRs live in the SkillWiki vault
returned by:

```bash
skillwiki path
```

Use `projects/portfolio-lab/` under that vault for project files. Do not
recreate repo-local `wiki/` or `work/` folders; they are legacy copies and the
vault is the canonical location.

Start here in the vault: `projects/portfolio-lab/knowledge.md`.

## Ops quick notes

- **Live authority:** `signals.json.target_allocations` → order router (champion SPY/GLD/TLT **46/38/16**). Ensemble is advisory unless separately promoted.
- **Dual-mode cron:** change jobs in `Makefile` + `crontab` + `src/cron_compat.py` + `config/tasker.yaml`; then `make verify-cron-sync`.
- **Daily brief:** `make daily-brief` (also tasker `portfolio-lab-daily-brief` at `:25` hourly) → `data/daily_brief.json`.
- **Ensemble inactivity:** do not force-wake or lower IC gates; classify first (B5 evidence in vault). Polarity follow-on is low priority / post-C1e.
- **Deploy lab host:** `make deploy-lab-app` (see `scripts/deploy-lab-app.sh`).
- **Backup/restore:** operator-invoked self-contained source/runtime/static recovery; secrets stay separate. See `scripts/LAB_APP_BACKUP_RESTORE.md` (`make deploy-production` is not recovery).
- **Agent test gate:** `make test-gate` mid-session; full `make test` merge-only.

## Agent cold path (cursor-box / Alpine / any clone)

Work in **this checkout**. On cursor-box that is `/workspace/code/portfolio-lab`
(symlink `/home/box/code/portfolio-lab`). Production lives at
`/home/box/.local/share/portfolio-lab/app` on `:8000`/`:8001` — do not edit it,
do not point `PYTHONPATH` at it, and never start a second Tasker scheduler
there.

### Python / uv

`~/.local/bin/uv` on cursor-box is a toolchain **wrapper** that injects Alpine
`LD_LIBRARY_PATH` (so `uv sync` can compile native wheels). That injection
breaks `uv run pytest` on a mixed glibc/musl host.

Use the clean resolver as the default agent path:

```bash
# one-time, from this checkout
scripts/agent_uv.sh sync

# mid-session gate (<2m, ensemble/signal). Makefile already uses scripts/agent_uv.sh.
PORTFOLIO_LAB_ENABLE_ML=0 make test-gate

# touched files
PORTFOLIO_LAB_ENABLE_ML=0 scripts/agent_uv.sh run pytest tests/<path> -q --tb=short
```

Override with `PORTFOLIO_LAB_UV=/path/to/clean/uv` if you already have a
standalone Astral binary. Do **not** `export LD_LIBRARY_PATH` from the
`~/.local/bin/uv` wrapper into the agent shell.

`scripts/python_runtime.sh` (cron / `make` jobs) defaults `PROJECT_DIR` to this
repo root. Do not assume `/root/projects/portfolio-lab`.

### Tests (safe)

| Intent | Command |
|--------|---------|
| Default agent gate | `PORTFOLIO_LAB_ENABLE_ML=0 make test-gate` |
| Touched files | `PORTFOLIO_LAB_ENABLE_ML=0 scripts/agent_uv.sh run pytest <paths> -q --tb=short` |
| Generator / dual-write | also `make test-generator` |
| Integration | `make test-integration` |
| Full merge suite | `make test` only (~30–45m). Do not stack a second full run. Wait with `scripts/wait-test-exit.sh` (60m max). |

Never default to `make test-unit` (still ~15k tests) or `make test-ml`.

### Side-dev Tasker (private deploy)

Production Tasker owns `data/tasker.lock` plus `:8000`/`:8001`. A side-dev API
uses a **sibling flock** (`data/tasker-side.lock`) so `--no-scheduler` does
not fight the live scheduler — but only when it runs from **this checkout's**
`data/` (private `TASKER_DB`). Starting an API sidecar from the production app
dir is refused.

```bash
export TASKER_DISABLE_SCHEDULER=1
export TASKER_HOST=127.0.0.1
export TASKER_PORT=8010   # anything other than 8000/8001
scripts/python_runtime.sh -m src.tasker.service --host 127.0.0.1 --port 8010 --no-scheduler
```

Do not set `PORTFOLIO_LAB_PROJECT_DIR` to the production app path. For a
fully isolated tree, clone or worktree this repo and run from there
(`PORTFOLIO_LAB_PROJECT_DIR` unset, or set to that worktree).

### Frontend (musl Bun vs Rolldown)

Vite 8 pulls `@rolldown/binding-linux-x64-gnu`. The cursor-box **musl** Bun
wrapper cannot load that glibc native binding, so `bun run dev`
(`bunx --bun vite`) fails on Alpine/cursor-box.

Reliable path: run Vite under **glibc Node**, not musl Bun:

```bash
# install once (bun install is fine for the lockfile; Node runs Vite)
bun install

bun run dev:node          # node node_modules/vite/bin/vite.js
# or:
node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4173
```

Point the Vite `/api` proxy at your **side-dev** Tasker port if you are not
using production `:8000`. `bun run fetch-data` / `bun test tests/ts/` stay
on Bun. Production/CI glibc Bun can keep `bun run dev` / `bun run build`.

This project was created using `bun init` in bun v1.3.11. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
