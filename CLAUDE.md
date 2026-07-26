# Portfolio Lab

## Agent instructions

Keep this file short: it is always injected into agent context. **Do not re-expand status dumps here** — put durable research, architecture maps, and historical findings in the SkillWiki vault and link them.

### Hard rules
- **No ML imports** without explicit user request (`torch` / `sklearn` / `xgboost` / `hmmlearn`). Default `PORTFOLIO_LAB_ENABLE_ML=0`. Safe work: `src/strategy/`, `src/signals/`, `src/broker/`, `src/monitor/`.
- **Live authority**: only `signals.json.target_allocations` → `src.broker.order_router`. Ensemble / overlays / MARL are advisory unless separately promoted (`marl_status.live_authoritative: false`).
- **Champion baseline**: SPY/GLD/TLT **46/38/16** (base-grid Sharpe 0.79; overlay research ~0.95). Challenger 44/36/20 is defensive only.
- **Paths**: import from `src.paths` (`DATA_DIR`, `MARKET_DB`, `WIKI_DIR`, …). Metrics: `src/backtest/metrics.py`.
- **Cron dual-mode**: when adding/changing jobs, update `Makefile` + `crontab` + `src/cron_compat.py` together; run `make verify-cron-sync`.
- **Tests (tiered — do not default to full suite)**:
  - **Default agent gate**: `make test-gate` (= `make test-fast`, &lt;2m, ensemble/signal). Prefer this mid-session.
  - **Touched files**: `PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest <paths> -q --tb=short`.
  - **Generator / dual-write edits**: also `make test-generator`. Integration paths: `make test-integration`.
  - **Full `make test`**: merge/pre-release only (~30–45m). Never stack a second full suite; never poll with a 10m Bash timeout. Wait with `scripts/wait-test-exit.sh` (60m max) or skip if `pgrep` empty + stale `data/test_last_exit.json`.
  - **`make test-unit`**: still ~15k tests (not a fast gate). `make test-ml` only when user asks for ML.
- **Frontend/data**: `bun run dev` / `build` / `fetch-data`. Python: `uv sync`, `uv run python …`.
- **Gotchas**: no `bc`; no bare `~/.hermes/` in app code (read `data/cron_status.json`); skillwiki pages need `started`/`updated`/`completed` frontmatter when validating.

### Knowledge index (canonical docs)
All project documentation, research status, architecture, ensemble weights, grid/FIRE tables, and compound notes live in the SkillWiki vault. **Start here:**

| What | Where |
|------|--------|
| **Project knowledge index** | SkillWiki: `projects/portfolio-lab/knowledge.md` (resolve root with `skillwiki path` or `WIKI_DIR`) |
| **Migrated CLAUDE.md body** | `projects/portfolio-lab/compound/claude-md-agent-reference.md` |
| **Compound notes index** | `projects/portfolio-lab/compound/README.md` |
| **Domain glossary** | repo `CONTEXT.md` (keep terms; not a status dump) |
| **Repo README** | `README.md` (points at vault, not local `wiki/`) |

Do **not** use the legacy empty tree `wiki/` inside the git repo. Vault is canonical (`src.paths.WIKI_DIR` / `PROJECT_WIKI_DIR`).

When a feature lands: update vault pages (and this hard-rules list if a rule changes). Do not grow this file into another status chronicle.
