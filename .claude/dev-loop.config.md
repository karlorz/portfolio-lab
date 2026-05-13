---
name: Dev loop project config — portfolio-lab
description: Drives the dev-loop skill for the portfolio-lab repo. PRD via superpowers, knowledge via skillwiki vault at ~/wiki, no publish/deploy (private research project).
type: config
---

# Dev Loop — portfolio-lab

> All-Season Portfolio backtesting + comparison lab.
> TypeScript/Vite/React dashboard + Python backtest engine + MARL agents.
> Research project — no public release artifact, no remote deploy.

## Identity

```yaml
slug: portfolio-lab
vault: /root/wiki
release_branch: main
```

## PRD layer

Full superpowers pipeline (brainstorm → spec → plan → execute → review).
The project already follows this rhythm: each `v2.xx` feature gets a
spec.md + plan.md under `projects/portfolio-lab/work/YYYY-MM-DD-<slug>/`
in the vault, then implementation lands in `src/`. Keep it.

```yaml
prd_layer: superpowers
prd_pipeline: full
```

### PRD backends registry

```yaml
prd_backends:
  superpowers:
    capabilities: [brainstorm, spec, plan, execute, review, subagent_dispatch]
    skills:
      brainstorm: superpowers:brainstorming
      plan: superpowers:writing-plans
      execute: superpowers:subagent-driven-development
      execute_fallback: superpowers:executing-plans
      review: simplify
```

### Cross-cutting disciplines

TDD is advisory rather than mandatory — backtest scripts and analysis
notebooks are not all test-first candidates, but new agents / signal
modules / risk code benefit from it. Systematic debugging fires
reactively on EXECUTE failures.

```yaml
prd_disciplines:
  - skill: superpowers:test-driven-development
    when: execute
    mode: advisory
  - skill: superpowers:systematic-debugging
    when: failure
    mode: reactive
  - skill: superpowers:verification-before-completion
    when: review
    mode: mandatory
```

## Knowledge layer

skillwiki vault at `/root/wiki`. Project workspace already exists at
`/root/wiki/projects/portfolio-lab/` with extensive history (60+ work
items, 70+ compound entries, ADRs). Continue writing there.

```yaml
knowledge_layer: skillwiki
```

### Knowledge backends registry

```yaml
knowledge_backends:
  skillwiki:
    vault: /root/wiki
    cli_entry: skillwiki
```

## Code layout

```yaml
cli_src: src/
cli_test: tests/
skills_glob:
cli_entry_override:
```

`src/` is sprawling — major subsystems include:

- `src/backtest/` — TypeScript + Python backtest harnesses
  (grid-search, rolling-window, monte-carlo-fire, walk-forward, factor-tilt)
- `src/agents/` — MARL controller + 5 specialist agents (Python, v2.51)
- `src/signals/`, `src/risk/`, `src/execution/` — signal integration,
  VaR/CVaR monitoring, execution timing
- `src/strategy/`, `src/optimization/`, `src/research/` — portfolio
  construction, optimizer, research notebooks
- `src/llm/`, `src/nlp/` — sentiment client + earnings/FOMC analyzers
- `src/dashboard/`, `src/components/`, `src/monitor/` — React UI
- `src/crypto/`, `src/options/`, `src/broker/`, `src/trading/` — asset
  class specific + broker abstraction + live trading prep

Python tests live in `tests/` (currently only `test_sentiment_client.py`).
TypeScript has no formal test harness yet — vite/vitest is not wired.

## E2E

No e2e scripts yet. The project relies on per-script CLI runs
(`python -m src.backtest.engine`, `bun run backtest`, etc.) and manual
dashboard verification. Leave empty; trivial fast-path applies to most
work items.

```yaml
e2e_scripts: []
```

## Release

Private research project. No npm publish, no remote deploy. Git push
to `origin/main` is the only "release" — and even that is local until
the user manually pushes.

```yaml
bump_script:
publish_via: none
deploy_script:
manifests_count: 0
remote_hosts: []
```

## Notes

```yaml
notes:
  canonical_spec: /root/wiki/projects/portfolio-lab/README.md
  python_runtime: python3 (no venv pinning yet — global interpreter)
  bun_runtime: bun 1.x; scripts in package.json use bunx --bun vite
  data_pipeline: |
    bun run fetch-data refreshes public/data/prices.json from Yahoo
    Finance v8. App + backtests load from prices.json — never re-fetch
    in-loop without explicit need.
  conventions: |
    - Work item slugs follow vXX-<feature> pattern (e.g., v292-etf-premium-monitor)
    - Compound pages in vault track every implemented strategy with
      backtest result snapshot (CAGR/Sharpe/MaxDD/crisis years)
    - CLAUDE.md is the canonical implementation status — keep it
      synced with the latest version bump after every feature lands
    - Subagent dispatch is preferred for multi-file feature work
      (typical: separate agents for signal module, risk module, dashboard,
      and integration glue)
  trivial_fast_path: |
    Use for backtest parameter sweeps, single-file analysis scripts,
    dashboard tweaks, README/CLAUDE.md edits, and compound-page writes.
    Escalate to full pipeline for new strategy modules, agent additions,
    or anything touching src/agents/ or src/signals/integrator.
  gotchas: |
    - Concurrent writes on src/llm/sentiment_client.py have been a
      problem before (see vault observation #307) — coordinate when
      dispatching parallel agents into that file.
    - ensemble_voter.py was specced but never implemented as of
      May 2026 (vault obs #301) — verify before referencing it.
    - Backtest data covers 2005-01-03 to 2026-05-08 (5371 trading days,
      15 symbols). Do not assume newer data without re-fetching.
```

## Gitignore

Not required — `knowledge_layer: skillwiki` keeps work items in the
vault, not the repo. No `.claude/dev-loop-work/` will be created.
