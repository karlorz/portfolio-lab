---
name: Dev loop project config — portfolio-lab
description: Drives the dev-loop skill for the portfolio-lab repo. PRD via TDD (test-first), knowledge via skillwiki path, no publish/deploy (private research project).
type: config
---

# Dev Loop — portfolio-lab

> All-Season Portfolio backtesting + comparison lab.
> TypeScript/Vite/React dashboard + Python backtest engine + MARL agents.
> Research project — no public release artifact, no remote deploy.

## Identity

```yaml
slug: portfolio-lab
release_branch: main
```

## PRD layer

TDD-first pipeline: plan (test design) → execute (red→green→refactor) → review → merge.
The plan step IS the test suite — write failing tests first, then implement to pass.
No brainstorm/spec steps; the test suite defines the requirements.

```yaml
prd_layer: tdd
prd_pipeline: tdd-first
```

### PRD backends registry

```yaml
prd_backends:
  tdd:
    capabilities: [plan, execute, review]
    skills:
      plan: superpowers:writing-plans
      execute: superpowers:test-driven-development
      review: simplify
```

### Cross-cutting disciplines

TDD is the pipeline itself (not a cross-cutting discipline). Systematic
debugging fires reactively on EXECUTE failures. Verification is mandatory
before completion.

```yaml
prd_disciplines:
  - skill: superpowers:systematic-debugging
    when: failure
    mode: reactive
  - skill: superpowers:verification-before-completion
    when: review
    mode: mandatory
```

## Knowledge layer

SkillWiki resolves the active vault via `skillwiki path`. Project workspace
content lives under `projects/portfolio-lab/` inside that vault, with extensive
history (60+ work items, 70+ compound entries, ADRs). Continue writing there.
Do not write specs/plans into repo-local `wiki/` or `work/` directories.

```yaml
knowledge_layer: skillwiki
```

### Knowledge backends registry

```yaml
knowledge_backends:
  skillwiki:
    vault: auto
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

Python tests: 157 test files, 12,429 safe tests (ML disabled, 3GB cap).
TypeScript tests: 14 test files, 191 tests (`bun test tests/ts/`).
Total: 13,248 tests safe. Run via `make test` (ML disabled) or `make test-ml`.

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

## Interview

```yaml
interview:
  backend: native
  trigger: auto
```

`native` backend asks 3 fixed questions per work item:
1. What are you building?
2. What constraints apply?
3. How will you know it's done?

`auto` trigger fires the interview only when ambiguity detection flags the
work item (conflicting prior decisions, vague description, zero prior art).
Clear, well-scoped tasks skip the interview entirely.

## Domain glossary

`CONTEXT.md` at repo root defines key terms: overlay, ensemble voting, regime
detection, MARL agents, VIX regime, VPIN, drift-based rebalancing, etc.

Agents dispatched by dev-loop should load CONTEXT.md as a reference before
writing tests or implementation code. Keep it updated as new concepts land.

```yaml
glossary:
  path: CONTEXT.md
  maintainer: dev-loop
  update_on: new_concept_landed
```

## CI

CI-enabled with runtime discovery — dev-loop queries GitHub branch protection
API to find required checks at MERGE time. No config duplication needed.

```yaml
ci_configured: true
ci_discovery: runtime
ci_workflow: .github/workflows/ci.yml
required_checks: []
```

## Critical paths

Three hot-spots derived from CLAUDE.md and project structure. The dev-loop
engine biases vault search, work-item priority, and research coverage gaps
toward these paths.

```yaml
critical_paths:
  signals-engine:
    code:
      - src/signals/**
    vault:
      - signal-architecture
      - ensemble-voting
    history_pins:
      - "UNIFIED_OVERLAY validated +0.014 Sharpe (v952)"
      - "6 active ensemble signals, 50% per-signal cap"
      - "behavioral_sentiment rejected: -0.216 Sharpe, 65.8% false positive rate"
  backtest-engine:
    code:
      - src/backtest/**
    vault:
      - backtest-architecture
    history_pins:
      - "Champion: SPY/GLD/TLT 46/38/16, Sharpe 0.79 (2005-2026, 94-config grid search)"
      - "Shared metrics module at src/backtest/metrics.py — use for new backtests"
  marl-agents:
    code:
      - src/agents/**
    vault:
      - marl-architecture
    history_pins:
      - "v2.51 ML-gated — never import without PORTFOLIO_LAB_ENABLE_ML=1"
      - "base_agent.py uses torch stubs (safe); execution_agent.py conditional imports"
```

## Fact-check tier

Full fact-checking enabled. Source order: local repo, context7 library docs,
vault queries, then web search (WebSearch/WebFetch built-in tools). Specs and
plans cite sources when consulting external docs.

```yaml
fact_check:
  enabled: true
  source_order:
    - local_repo
    - context7
    - vault
    - web
  web_tools:
    primary: WebSearch
  evidence_contract:
    require_sources_used_section: true
  triggers:
    - "version "
    - "deprecat"
    - "CVE-"
```

## Idle deep-research

When idle cycles find no claimable work, rotate through research topics
derived from critical paths. Cooldown: every 3rd idle cycle, max 4/day.

```yaml
idle_deep_research:
  enabled: true
  topic_seeds:
    - "portfolio-lab signal ensemble optimization techniques"
    - "multi-agent RL for portfolio allocation latest research"
    - "tail-risk hedging with VIX derivatives latest approaches"
    - "cross-asset momentum signal construction methods"
    - "volatility regime detection improvements"
  bias_toward: critical_paths
  cooldown_cycles: 3
  max_per_day: 4
  skip_if_recent_query_page_exists: 7
  budget:
    web_searches: 3
    deep_fetches: 3
    context7_calls: 3
```

## Browser verification

React + Vite dashboard detected but playwright-cli not installed. Skip
browser gate for now — rely on manual verification. Install playwright-cli
and re-run setup to enable.

```yaml
browser_verification:
  enabled: false
```

## Reactive debugging

Cap retries at 2, capture evidence on failure, escalate after 3 consecutive
idle cycles with the same error signature.

```yaml
reactive_debugging:
  enabled: true
  auto_retry_attempts: 2
  evidence_dir: .claude/dev-loop-debug/
  evidence_capture:
    - "make check 2>&1 | tee {evidence_dir}/{cycle}-check.log"
    - "git diff > {evidence_dir}/{cycle}-diff.patch"
  escalate_after:
    consecutive_idle_cycles: 3
    same_error_signature: true
```

## Code review

simplify-worker (sonnet) is the base reviewer. Codex is installed but
disabled by default — opt-in per intensity.

```yaml
code_review:
  parallel: true
  codex:
    enabled_in_normal: false
    enabled_in_high: false
    agent: dev-loop:codex-review-worker
```

## Notes

```yaml
notes:
  canonical_spec: projects/portfolio-lab/README.md
  python_runtime: python3 (no venv pinning yet — global interpreter)
  bun_runtime: bun 1.x; scripts in package.json use bunx --bun vite
  data_pipeline: |
    bun run fetch-data refreshes public/data/prices.json from Yahoo
    Finance v8. App + backtests load from prices.json — never re-fetch
    in-loop without explicit need.
  tdd_conventions: |
    - TDD is mandatory: write failing test first, watch it fail, implement minimal code, refactor
    - Python: pytest with --import-mode=importlib, tests/ directory
    - TypeScript: bun test tests/ts/
    - ML-gated: always PORTFOLIO_LAB_ENABLE_ML=0 unless user explicitly requests ML
    - Test run: make test (safe, 3GB cap) or make test-ml (full suite)
    - New signal modules: test signal generation, snapshot format, regime gating
    - New strategy modules: test weight computation, edge cases, integration with ensemble voter
    - Backtest scripts: test metric computation, data loading, not necessarily TDD for parameter sweeps
  conventions: |
    - Work item slugs follow vXX-<feature> pattern (e.g., v292-etf-premium-monitor)
    - Specs/plans route through `skillwiki path` under projects/portfolio-lab/work
    - Repo-local wiki/ and work/ directories are stale legacy copies and must not be recreated
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
    - ensemble_voter.py IS implemented (v2.58) — the earlier vault obs #301
      is stale. The 8-layer weight pipeline is production code.
    - Backtest data covers 2005-01-03 to 2026-05-08 (5371 trading days,
      15 symbols). Do not assume newer data without re-fetching.
    - pytest importlib mode (--import-mode=importlib via pyproject.toml addopts)
      eliminates sys.modules pollution — keep it.
    - 4-layer ML safety defense: collect_ignore, env var, import hook, leak check.
      Never bypass without explicit user request.
```

## Gitignore

```yaml
gitignore:
  - .claude/dev-loop-debug/
```

`knowledge_layer: skillwiki` keeps work items in the vault, not the repo.
Only `.claude/dev-loop-debug/` (reactive debugging evidence) needs gitignore.
