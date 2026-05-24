# Portfolio Lab - All-Season Strategies

## Status
- Research: **Complete** (11 wiki compound pages + grid search + rolling-window + Monte Carlo)
- Build: **Complete** — real Yahoo Finance data, working backtest engine + FIRE calculator
- **Champion**: SPY/GLD/TLT 46/38/16, Sharpe 0.79 (2005-2026, 94-config grid search)
- **Drift rebalancing**: 10% drift beats annual — Sharpe 0.83 vs 0.79
- Data: 5371 trading days (2005-01-03 to 2026-05-08), 15 symbols incl. EFA/VXUS/MTUM/VLUE/USMV
- Test count: **4298 safe** (0 failures, 21 skipped, 4319 collected)

### Key Findings
|- **TSMOM standalone (Sharpe 0.96) beats combined signal overlay (0.93)** — signal conflicts erode alpha
|- **MULTI_SPEED_MOM is net-negative as overlay** (-0.012 Sharpe vs baseline) — ensemble works because 46/38/16 base is strong
|- **DBC at 4% rejected**: Sharpe -0.057, hurts 2008/2020, marginal help 2022
|- **Factor rotation is defensive only**: reduces drawdowns, not an alpha generator
|- **Behavioral sentiment (VIX-proxy) is net-negative**: -0.216 Sharpe, 65.8% false positive rate
|- **UNIFIED_OVERLAY validated: +0.014 Sharpe vs baseline** — keep current weight (17-21%)
|- **PyPortfolioOpt finds higher theoretical Sharpe**: Max Sharpe 0.87 (weights 40/34/26), HRP 0.93 (weights 29/28/43) vs champion 0.79 (46/38/16)

### Active Ensemble Signals (6)
MULTI_SPEED_MOM, CROSS_ASSET_RV, INTERNATIONAL_MOMENTUM, ALTERNATIVE_DATA, CROSS_ASSET_REGIME_ARB, UNIFIED_OVERLAY (all 6 active)
- **MULTI_SPEED_MOM health: 0.55** (below 0.60 viability floor) — 0.00 weight (disabled), gated OFF in HIGH_VOL/CRISIS by RegimeGate

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
- `src/signals/` — signal generators (collar_signal, bond_duration_signal, crypto_momentum, calendar_seasonality, vix_term_structure, multi_speed_momentum, international_momentum, alternative_data_signal, cross_asset_relative_value, cross_asset_regime_arb, vpin_bvc, behavioral_sentiment, fed_policy_overlay, health_tracker, stacking_feature_engine/integrator, stacking_integrator, tsmom_overlay/tsmom_integration, signal_pruner, signal_snapshot, regime_gate, multi_strategy_adapters)
- `src/strategy/` — overlays & strategy (unified_orchestrator, vixy_hedge_sizing, evaluator, ensemble_voter with BanditWeighter, factor_rotation, adaptive_ensemble_weights, turnover_validator, graduation_checklist, adaptive_sizing, risk_parity_weight_overlay, vol_parity_allocator)
- `src/backtest/` — backtest engines. **Use `src/backtest/metrics.py`** for BacktestResult/BacktestMetrics/compute_metrics() (canonical shared dataclass + computation, eliminates copy-paste)
- `src/broker/` — broker integration (order_router, position_sync, collar_options_bridge, options_utils)
- `src/monitor/` — monitoring (garch_cvar, cvar_metrics, risk_decomposition, performance_attribution, unified_dashboard, daily_brief, rebalance_health)
- `src/agents/` — MARL system (ML-gated, see below)
- `src/data/` — data fetchers (Yahoo Finance, behavioral sentiment, international)
- `src/research/` — research tools (agent, features, wiki_sync)

### AI Agents (src/agents/ v2.51, ML-gated)
- `ai_controller.py` — main entry (infer/train/status)
- `analyst_agent.py`, `sentiment_agent.py`, `risk_agent.py`, `execution_agent.py`, `controller_agent.py`
- `agent_graph.py` — LangGraph topology, `marl_trainer.py` — MAPPO training
- CLI: `python -m src.agents.ai_controller --mode status`

### Analysis Scripts (src/backtest/, TypeScript)
grid-search, rolling-window, correlation-regime, recovery-analysis, withdrawal-sweep, rebalance-tolerance, monte-carlo-fire, factor-tilt, commodities-sweep, tactical-rebalance

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

- **Python**: 4155 safe (0 failures, 21 skipped), 111 test files
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
