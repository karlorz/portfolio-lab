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

## Frontend

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run dev
```

This project was created using `bun init` in bun v1.3.11. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
