# Portfolio Lab - All-Season Strategies

## Status
- Research: **Complete** (11 wiki compound pages + grid search + rolling-window + Monte Carlo)
- Build: **Complete** — real Yahoo Finance data, working backtest engine + FIRE calculator
- **Champion**: SPY/GLD/TLT 46/38/16, Sharpe 0.79 (2005-2026, 94-config grid search)
- **Drift rebalancing**: 10% drift beats annual — Sharpe 0.83 vs 0.79
- Data: 5371 trading days (2005-01-03 to 2026-05-08), 15 symbols incl. EFA/VXUS/MTUM/VLUE/USMV
| - Test count: **12369 safe** (12369 Python + 214 TypeScript, 27 skipped, 0 failures, 42 BL mapper tests gated behind pypfopt)
|- **Signal snapshot coverage: 15/15** — all signal modules have get_signal_snapshot() for typed pipeline
|- **Gold allocation sweep**: 109 configs tested (GLD 20-55%) — champion 46/38/16 remains optimal; BofA/Goldman "more gold" thesis doesn't improve risk-adjusted returns
|- **GARCH-CVaR EWMA fallback**: 3-tier chain (GARCH → EWMA → historical) fixes zero-output bug for paper trading with few daily returns
|- **Overlay data pipeline**: overlay_dashboard data merged into signals.json — 9 panels now render with real data
|- **Graduation checklist v2**: 12 criteria (added regime_coverage, signal_diversity, sharpe_ci_lower)
|- **Periodic rebalancing**: forces rebalance if drift >2% and 30+ days since last trade
|- **LTTB downsampling**: src/utils/lttb.ts — auto-downsample charts from 5371 to 600 points preserving visual shape
|- **Signal staleness detection**: DashboardGenerator checks signal timestamps against 4h TTL (SIGNAL_STALENESS_TTL_HOURS), reports stale signals in signals.json
|- **External alerting**: src/monitor/alerting.py — webhook-based PASS→WARN→HALT state-transition alerting (ALERT_WEBHOOK_URL env var), staleness + drift checks
|- **pytest importlib mode**: --import-mode=importlib via addopts in pyproject.toml — eliminates sys.modules pollution class of bugs
|- **Dynamic MSM gating**: TSMOM is_gated_off now uses regime-based check instead of hardcoded True
|- **Staleness-weighted ensemble voting**: exponential decay (STALENESS_DECAY_TAU_HOURS=2h) degrades stale signal weights, recomputes weighted_consensus
|- **Regime confidence gating**: RegimeGate.gate_with_confidence() — defers gating when regime confidence < 0.7, combines with hysteresis
|- **SPC signal quality monitoring**: src/monitor/spc_monitor.py — Shewhart control charts flag 3-sigma distribution shifts for 3+ consecutive periods
|- **Rebalance health in signals.json**: DashboardGenerator.run() now includes rebalance_health section from src/monitor/rebalance_health
|- **JSONL tail reading**: generator.py uses `collections.deque(f, maxlen=N)` instead of full-file reads for orders.jsonl, features.jsonl, performance.jsonl, position_sync.jsonl, broker_orders.jsonl, grid_search_results.jsonl
|- **Regime overrides deduplication**: REGIME_OVERRIDES moved to src/paths.py (single source of truth), imported by evaluator.py and generator.py
|- **Evaluator config externalization**: PAPER_CONFIG and graduation criteria (MIN_SHARPE, MAX_DD, etc.) now read from env vars with hardcoded fallbacks (PAPER_INITIAL_CAPITAL, GRADUATION_MIN_SHARPE, etc.)
|- **React error boundaries**: PanelErrorBoundary wraps each dashboard tab panel — single-panel crashes don't kill the entire dashboard
|- **DashboardGenerator context manager**: `__enter__`/`__exit__` + `close()` + `try/finally` in `run()` prevents SQLite connection leaks on exceptions
|- **Evaluator print→logging**: check_graduation_criteria, kill switch, and trigger creation use logger instead of print() for production observability
|- **WIKI_DIR/WORK_DIR env vars**: configurable via environment variables with fallback defaults in src/paths.py
|- **TTL price cache**: src/data/price_cache.py — cachetools.TTLCache(maxsize=1, ttl=30s) eliminates redundant prices.json reads across 18 modules (PRICE_CACHE_TTL_SECONDS env var), ~10MB peak memory savings per cron cycle
|- **get_prices_df()**: cached pivoted DataFrame accessor with symbol subset parameter — eliminates duplicated pivot code across 8+ modules
|- **Shared strategy constants**: VOL_TARGET, MAX_DEVIATION, MIN_WEIGHT, REBALANCE_FREQ consolidated in src/paths.py (env-var configurable) — imported by tsmom_overlay.py and multi_speed_momentum.py
|- **Broker error handling**: alpaca.py submit_order() returns None on failure, get_orders() returns [] on failure
|- **SPC state persistence**: spc_monitor.py save_state/load_state — JSON serialization to DATA_DIR/spc_state.json, wired into DashboardGenerator
|- **VPIN query cache**: vpin_bvc.py TTLCache(maxsize=64, ttl=300s) for SQLite OHLCV queries
|- **Lazy SQLite connections**: ResearchAgent/WikiSync use lazy property with setter + close() + try/finally, generator.py close() narrows except
|- **MARKET_DB constant**: 7 modules consolidated from DATA_DIR/"market.db" to src/paths.MARKET_DB
|- **MSM transient error resilience**: _is_msm_gated caches last-known regime and defers to it on SQLite failures instead of gating off
|- **Dead import cleanup**: removed unused `import sqlite3` from 7 source files

### Key Findings
|- **TSMOM standalone (Sharpe 0.96) beats combined signal overlay (0.93)** — signal conflicts erode alpha
|- **MULTI_SPEED_MOM is net-negative as overlay** (-0.012 Sharpe vs baseline) — ensemble works because 46/38/16 base is strong
|- **DBC at 4% rejected**: Sharpe -0.057, hurts 2008/2020, marginal help 2022
|- **Factor rotation is defensive only**: reduces drawdowns, not an alpha generator
|- **Behavioral sentiment (VIX-proxy) is net-negative**: -0.216 Sharpe, 65.8% false positive rate
||- **UNIFIED_OVERLAY validated: +0.014 Sharpe vs baseline** — keep current weight (17-21%)
||- **Regime gating analysis v961 complete**: UNIFIED_OVERLAY needs gates (OFF in CRISIS/HIGH_VOL/NORMAL), CROSS_ASSET_RV needs adaptive params, CRYPTO_MOMENTUM needs none
||- **RECOVERY regime confirmed**: 5.5% of market time, Sharpe 4.17 — genuine distinct high-alpha regime
||- **LOW_VOL detection verified**: 35% detection rate, gates are activatable
|- **PyPortfolioOpt finds higher theoretical Sharpe**: Max Sharpe 0.87 (weights 40/34/26), HRP 0.93 (weights 29/28/43) vs champion 0.79 (46/38/16)
|- **Walk-forward validation (20 windows): WFE=1.02, mean OOS Sharpe=0.99** — champion portfolio validated out-of-sample
|- **Black-Litterman mapper**: maps ensemble biases → BL views with Idzorek confidence from health_scores, tau=0.15
|- **Deflated Sharpe Ratio**: DSR=0.979 with 94 grid search configs — champion survives multiple-testing correction

### Active Ensemble Signals (6)
MULTI_SPEED_MOM, CROSS_ASSET_RV, INTERNATIONAL_MOMENTUM, ALTERNATIVE_DATA, CROSS_ASSET_REGIME_ARB, UNIFIED_OVERLAY (all 6 active)
- **MULTI_SPEED_MOM health: 0.55** (below 0.60 viability floor) — 0.00 weight (disabled), gated OFF in HIGH_VOL/CRISIS by RegimeGate
- **RegimeGate**: behavioral_sentiment ON only in LOW_VOL; cross_asset_regime_arb OFF in LOW_VOL; LOW_VOL added to Regime enum (vol < 12% + momentum > 1%)

### Current Weights (NORMAL regime)
ALT_DATA 0.305, INTL_MOM 0.245, CROSS_RV 0.13, REGIME_ARB 0.13, UNIFIED 0.19
MSM 0.00 (disabled — net-negative -0.012 Sharpe, health 0.55)
Max per-signal cap: 50%

## Strategies Implemented (16 portfolios)
- SPY, QQQ, 60/40, All Weather (Dalio), Golden Butterfly ±Trend
- **SPY/GLD 55/45** — meets target (>=90% SPY return, <=70% vol)
- **SPY/GLD/TLT 58/32/10**, **50/35/15**, **50/40/10** — meet target
- **SPY/GLD/IEF 50/35/15** — best 2022 resilience with IEF
- **SPY/GLD/TLT 46/38/16** — Sharpe 0.79, fine-sweep champion
- SPY/EFA/GLD/TLT 36/10/38/16 — international tactical hedge

## Grid Search Results (2005-2026, 94 configs)

### Top 5 by Sharpe Ratio
| Portfolio | CAGR | Vol | Sharpe | Max DD | 2008 | 2020 | 2022 |
|-----------|------|-----|--------|--------|------|------|------|
| **SPY/GLD/TLT 46/38/16** | 10.6% | 11.1% | **0.79** | -26.2% | -12.3% | -7.1% | -13.0% |
| SPY/GLD/TLT 46/34/20 | 10.3% | 10.6% | **0.79** | -24.7% | -12.2% | -6.6% | -14.2% |
| SPY/GLD/TLT 48/32/20 | 10.3% | 10.6% | **0.79** | -25.3% | -13.6% | -7.0% | -14.6% |
| SPY/GLD/TLT 46/36/18 | 10.4% | 10.8% | **0.79** | -25.5% | -12.2% | -6.8% | -13.6% |
| SPY/GLD/TLT 48/34/18 | 10.4% | 10.8% | **0.79** | -25.9% | -13.6% | -7.2% | -14.0% |

### FIRE Key Results (Monte Carlo, 1000 sims x 30yr)
| Portfolio | 4% Survival | 5% Survival | 6% Survival | Safe Rate (95% conf) |
|-----------|-------------|-------------|-------------|---------------------|
| SPY/GLD/TLT 46/38/16 | 100% | 99% | 97% | 6.0% |
| SPY/GLD/TLT 50/35/15 | 100% | 100% | 98% | 6.0% |
| SPY/GLD 55/45 | 100% | 99% | 98% | 6.0% |
| SPY | 99% | 95% | 89% | 4.5% |
| 60/40 | 100% | 96% | 88% | 5.0% |

## Architecture

### Signal/Overlay Map
- `src/signals/` — signal generators (collar_signal, bond_duration_signal, crypto_momentum, calendar_seasonality, vix_term_structure, multi_speed_momentum, international_momentum, alternative_data_signal, cross_asset_relative_value, cross_asset_regime_arb, vpin_bvc, behavioral_sentiment, fed_policy_overlay, health_tracker, stacking_feature_engine/integrator, stacking_integrator, tsmom_overlay/tsmom_integration, signal_snapshot, regime_gate, multi_strategy_adapters)
- `src/strategy/` — overlays & strategy (unified_orchestrator, vixy_hedge_sizing, evaluator, ensemble_voter with BanditWeighter, factor_rotation, adaptive_ensemble_weights, turnover_validator, graduation_checklist, adaptive_sizing, risk_parity_weight_overlay, vol_parity_allocator, black_litterman_mapper)
- `src/backtest/` — backtest engines. **Use `src/backtest/metrics.py`** for BacktestResult/BacktestMetrics/compute_metrics()/compute_deflated_sharpe_ratio() (canonical shared dataclass + computation, eliminates copy-paste)
- `src/broker/` — broker integration (order_router, position_sync, collar_options_bridge, options_utils)
- `src/monitor/` — monitoring (garch_cvar, cvar_metrics, risk_decomposition [wired to dashboard], performance_attribution, unified_dashboard, daily_brief, rebalance_health)
- `src/agents/` — MARL system (ML-gated, see below)
- `src/data/` — data fetchers (Yahoo Finance, behavioral sentiment, international)
- `src/costs/` — transaction cost models (etf_cost_table with per-ETF costs and regime multipliers)
- `src/rebalancing/` — smart rebalancing (SmartRebalancingController with regime-adaptive drift thresholds)
- `src/research/` — research tools (agent, features, wiki_sync)

### AI Agents (src/agents/ v2.51, ML-gated)
- `ai_controller.py` — main entry (infer/train/status)
- `analyst_agent.py`, `sentiment_agent.py`, `risk_agent.py`, `execution_agent.py`, `controller_agent.py`
- `agent_graph.py` — LangGraph topology, `marl_trainer.py` — MAPPO training
- CLI: `python -m src.agents.ai_controller --mode status`

### Analysis Scripts (src/backtest/, TypeScript)
grid-search, rolling-window, correlation-regime, recovery-analysis, withdrawal-sweep, rebalance-tolerance, monte-carlo-fire, factor-tilt, commodities-sweep, tactical-rebalance, walk-forward-validation

### State Files
- `data/hedge_efficiency_state.json`, `data/vix_overlay_state.json`
- `data/cron_status.json` — backend-agnostic cron status

## Test Coverage

### 4-Layer ML Safety Defense
| Layer | Mechanism | Effect |
|-------|-----------|--------|
| 0 | `collect_ignore` in conftest.py | Heavy test files never opened |
| 1 | `PORTFOLIO_LAB_ENABLE_ML=0` | ML features disabled before import |
| 2 | `builtins.__import__` hook | Blocks torch/sklearn/xgboost/hmmlearn |
| 3 | Post-collection leak check | Warns if real ML libs evaded guards |

- **Python**: 12429 safe (0 failures, 27 skipped), 131 test files
- **TypeScript**: 191 tests across 10 files (`bun test tests/ts/`)
- **Safe run**: `make test` (ML disabled, 3GB ulimit cap)
- **ML run**: `make test-ml` or `PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/ --include-heavy`

## Dev Constraints (HARD RULES)

### No ML imports without explicit user override
- **NEVER** import torch/xgboost/sklearn/hmmlearn without user explicitly requesting ML
- torch 63MB + sklearn 78MB + hmmlearn 23MB = 164MB+, OOM-kills at ~49% on sg01
- **ALWAYS** `PORTFOLIO_LAB_ENABLE_ML=0` (default) for all test runs and dev work
- **ONLY** `PORTFOLIO_LAB_ENABLE_ML=1` when user explicitly asks

### ML-gated modules (do NOT import without user request)
`src/agents/ai_controller.py`, `analyst_agent.py`, `controller_agent.py`, `sentiment_agent.py`, `agent_graph.py`, `marl_trainer.py`, `risk_agent_hmm.py`
Safe: `base_agent.py` (uses torch stubs), `execution_agent.py` (conditional imports)

### Preferred dev targets (no ML, safe to test anytime)
- `src/strategy/` — unified_orchestrator, vixy_hedge_sizing, evaluator
- `src/signals/` — credit_spread, commodity_curve, etc.
- `src/broker/` — options_utils, order_router, position_sync
- `src/monitor/` — garch_cvar, etc.

## Quick Start
```bash
make test            # safe test suite (ML disabled, 3GB cap, 4155 passing)
make test-ml         # full suite including ML (needs >3GB RAM)
bash scripts/run-tests-safe           # standalone safe runner
PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/  # manual safe run
```

## To Run
```bash
bun run dev          # dev server
bun run build        # production build
bun run fetch-data   # refresh data from Yahoo Finance v8 API
```

## Data Pipeline
1. `bun run fetch-data` -> Yahoo Finance v8 chart API (auto-detects today)
2. Saves to `public/data/prices.json` (compact: {d, p} per symbol, ~2.4MB)
3. App loads `/data/prices.json` on startup, runs backtests client-side

## Python: uv Package Manager
Core deps in `pyproject.toml`, ML deps in `[dependency-groups] ml`.
```bash
uv sync                  # install core deps (no ML libs)
uv sync --group ml       # install core + ML deps
uv run python script.py  # run a script
```

## Environment Gotchas
- `bc` is NOT available -- use `date +%s` for duration math
- Makefile `define` with multiline Python is fragile -- use `scripts/cron_update.py`
- `skillwiki validate` requires `started:`, `updated:`, `completed:` frontmatter fields
- `make verify-cron-sync` catches backend drift -- run after changing Makefile/crontab

## Cron Compatibility Contract (dual-mode: Hermes + system crontab)

Three backends via `CRON_BACKEND` env var: `hermes` (default), `crontab`, `manual`

```python
from src.cron_compat import IS_HERMES, IS_CRONTAB, BACKEND, CRON_TARGETS
```

### When adding a new cron job
Update three files in lockstep:
1. **`Makefile`** -- add `.PHONY` target + `scripts/cron_update.py` call
2. **`crontab`** -- add crontab entry
3. **`src/cron_compat.py`** -- add job name to `CRON_TARGETS`

### When changing code that a cron job calls
- Makefile target is source of truth. Update it first.
- Run `make verify-cron-sync` after changes.
- No `~/.hermes/` paths in app code. Exception: `generator.py` reads `data/cron_status.json`.

### When changing generator.py or dashboard data
- Reads `data/cron_status.json` (not `~/.hermes/cron/state.json`)
- Format: `{jobs: [{name, status, last_run, duration_seconds, backend}]}`

### Verification
```bash
make verify-cron-sync          # check Makefile <-> crontab <-> cron_status.json sync
CRON_BACKEND=crontab make all  # test full pipeline with crontab backend
```

### Switching backends
```bash
hermes cron pause <ids> && crontab crontab    # to system crontab
crontab -r && hermes cron resume <ids>         # back to Hermes
```

## Wiki Compound Pages (97+ total)
- 11 original research pages + 86+ strategy/cycle pages in `wiki/projects/portfolio-lab/compound/`
- Full index: `wiki/projects/portfolio-lab/knowledge.md`
