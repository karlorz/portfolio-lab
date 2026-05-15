# Portfolio Lab - All-Season Strategies

## Status
- Research: **Complete** (11 wiki compound pages + grid search + rolling-window + Monte Carlo)
- Build: **Complete** — real Yahoo Finance data, working backtest engine + FIRE calculator
- Grid Search: 94 configurations swept, Sharpe 0.79 validated on extended 2005-2026 data
- Rolling Window: 9 sub-periods tested, champion beats SPY in 6/9 windows
- Monte Carlo: 1000 bootstrap sims, 6% withdrawal at 95% confidence for all-season portfolios
- Data: 5371 trading days (2005-01-03 to 2026-05-08), 15 symbols incl. EFA/VXUS/MTUM/VLUE/USMV
- **Sharpe 0.79 champion validated with +1yr of new data (2005-2026)**
- **10% drift-based rebalancing beats annual: Sharpe 0.83 vs 0.79**
- **v2.55 Combined Signal Orchestrator + Backtest**: Multi-source aggregation with historical validation
  - **Key Finding**: TSMOM alone (Sharpe 0.96) outperforms combined overlay (0.93)
  - Signal conflicts and transaction costs erode alpha
  - Combined results 2006-2026: CAGR 10.71%, Sharpe 0.93 vs baseline 0.93, Max DD -25.7%
  - Crisis: 2008 -9.36%, 2020 +3.08%, 2022 -12.67%
  - Recommendation: Use TSMOM overlay standalone rather than combined signals
- **v2.54 Fed Policy Overlay**: Real-time FRED integration
  - Current regime: EASING (Fed 3.64%, real rate -0.31%, 10Y-2Y +0.47)
  - Allocation: SPY+2.6%, GLD+3%, TLT-5.5% from base 46/38/16
  - Signal integrator weight: 10% (regime-based tactical shifts)
- **v2.53 HMM-LSTM Regime Detector**: 5-state market classification (bull/bear/neutral/high_vol/crisis)
  - GaussianHMM trained on 26,225 samples (SPY/GLD/TLT/QQQ/IEF)
  - 4D features: momentum, volatility, trend strength, VIX proxy
  - Regime-based allocation shifts, transition matrix learned
  - CLI: train, detect, portfolio commands
- **v2.52 TSMOM Overlay**: AQR-style time-series momentum (12m formation, 1m skip, vol-scaled)
  - Sharpe 0.96 validated on 2006-2026 backtest (+0.17 vs baseline)
  - 243 rebalances, 10bps cost, max DD -20.83%
  - Signal integrator integration (5% weight)
- **v2.51 AI Agent Controller**: MARL system with 5 specialized agents (3,558 lines PyTorch)
  - Analyst Agent: fundamental/value analysis
  - Sentiment Agent: news/social signals with contrarian detection
  - Risk Agent: VaR/CVaR monitoring with drawdown alerts
  - Execution Agent: order timing with market impact modeling
  - Controller Agent: orchestration with centralized critic
  - Inference latency: 4.7ms (target: <50ms) ✓
  - Integrates with v2.24 signal integrator (5% weight in composite)

## Recent Implementation Updates (2026-05-16)

### v4.70 Crypto Tactical Allocation - COMPLETED
- **Signal Generator**: `src/signals/crypto_momentum.py` (340 lines) — BTC/ETH momentum + vol regime
  - 6-month/3-month/1-month momentum computation with 180-day lookback
  - 4 vol regimes: LOW (<40%), NORMAL (40-70%), HIGH (70-100%), EXTREME (>100%)
  - Vol-scaling: target 40% annualized, position range 0.25x-2.0x
  - BTC 60% / ETH 40% of crypto sleeve, funded from GLD
- **Tactical Overlay**: `src/strategy/crypto_allocation.py` (280 lines) — allocation + backtest
  - Entry: 6m momentum positive + vol regime normal/low
  - Exit: momentum negative OR vol extreme (>100% ann.)
  - Hard cap: 5% portfolio, 5% EnsembleVoter weight
  - Backtest engine with baseline vs crypto comparison
- **Tests**: `tests/test_crypto_momentum.py` (37 tests) + `tests/test_crypto_allocation.py` (23 tests) = 60 tests passing
- **State**: `data/crypto_allocation_state.json` — tracks current crypto allocation
- **Correlation**: BTC/ETH near-zero (0.05-0.15) to traditional 60/40 portfolio
- **Status**: All phases complete

### v3.50 Calendar Seasonality Overlay - COMPLETED
- **Signal Generator**: `src/signals/calendar_seasonality.py` (440 lines) — calendar-based execution timing
  - NYSECalendar with Easter computation, 12+ US market holidays, trading day logic
  - 8 calendar windows: TOM, Pre/Post-Holiday, Quarter-End, Monday, Pre-FOMC, December, OPEX
  - Composite urgency modifier (0.0-1.0, multiplicative) for rebalancing timing
  - FOMC schedule for 2026, options expiry (3rd Friday), future window prediction
- **Integration**: Convenience `get_calendar_modifier()` for rebalance scheduler
- **CLI**: `calendar <YYYY-MM>` calendar view, `check` current date assessment
- **Tests**: `tests/test_calendar_seasonality.py` (74 tests passing)
- **Expected Impact**: +0.01-0.02 Sharpe through 5-15 bps better execution annually
- **Status**: All phases complete

### v4.60 Cashless Collar Options Overlay - COMPLETED
- **Signal Generator**: `src/signals/collar_signal.py` (340 lines) — Black-Scholes pricing, strike selection
  - VIX-aware strike widening across 4 volatility regimes (NORMAL/ELEVATED/STRESS/CRISIS)
  - Binary search strike selection by target delta (30-delta call, 20-delta put)
  - No ML dependencies (scipy.stats.norm fallback to math.erf)
- **Tactical Overlay**: `src/strategy/collar_overlay.py` (340 lines) — roll logic, backtest engine
  - Monthly collar cycle: write OTM call, buy OTM put, net premium near zero
  - CRISIS freeze (VIX >40 disables collar — cost prohibitive)
  - Historical backtest: hedged vs unhedged comparison engine
- **Integration**: 10% weight in EnsembleVoter via CollarOverlayIntegrator
- **Tests**: `tests/test_collar_signal.py` (49 tests) + `tests/test_collar_overlay.py` (26 tests) = 75 tests passing
- **State**: `data/collar_overlay_state.json` — tracks current collar status
- **Target**: Max DD -26.2% → ≤-20%, Sharpe +0.03-0.05
- **Status**: All phases complete

## Recent Implementation Updates (2026-05-15)

### v4.50 VIX Term Structure Overlay - Phase 3 COMPLETED
- **Signal Generator**: `src/signals/vix_term_structure.py` (580 lines) — VIX/VIX3M/VIX6M slope analysis
- **Tactical Overlay**: `src/strategy/vix_overlay.py` (540 lines) — regime-based allocation shifts
  - Allocation shifts: SPY±10%, GLD±5%, TLT±5% based on term structure slope
  - Constraints: max 5% daily shift, 5-day holding period, VPIN freeze
  - VIX spike protection: >50% single-day spike disables overlay for 24h
- **Integration**: 15% weight in ensemble voter, SmartRebalanceGate coordination
- **Tests**: `tests/test_vix_overlay.py` (22 tests passing)
- **State**: `data/vix_overlay_state.json` — tracks current tactical allocation
- **Status**: Phase 3 complete, Phase 4 backtest validation ready

## Strategies Implemented (16 portfolios)
- SPY (S&P 500) — benchmark
- QQQ (Nasdaq-100) — growth benchmark
- 60/40 Portfolio — traditional stocks/bonds
- All Weather (Dalio) — 30/40/15/7.5/7.5 risk parity
- Golden Butterfly — 20/20/20/20/20 with SCV tilt
- Golden Butterfly + Trend — with 10-month SMA overlay
- **SPY/GLD 55/45** — ★ meets target (≥90% SPY return, ≤70% vol)
- **SPY/GLD/TLT 58/32/10** — ★ meets target
- **SPY/GLD/TLT 50/35/15** — ★ Sharpe 0.78, coarse-sweep winner
- **SPY/GLD/TLT 50/40/10** — ★ high CAGR + low vol
- **SPY/GLD/IEF 50/35/15** — ★ best 2022 resilience with IEF
- SPY/GLD 55/45 +Trend — trend overlay reduces max DD but increases vol
- SPY/GLD/TLT 50/35/15 +Trend — trend overlay variant
- SPY/GLD/TLT 50/35/15 +VolTarget — volatility targeting (12% target)
- **SPY/GLD/TLT 46/38/16 ★★** — ★★ Sharpe 0.79, fine-sweep champion
- SPY/EFA/GLD/TLT 36/10/38/16 — international tactical hedge

## Grid Search Results (2005-2026, 94 configs)

### Top 5 by Sharpe Ratio (all meet target)
| Portfolio | CAGR | Vol | Sharpe | Max DD | 2008 | 2020 | 2022 |
|-----------|------|-----|--------|--------|------|------|------|
| **SPY/GLD/TLT 46/38/16 ★★** | 10.6% | 11.1% | **0.79** | -26.2% | -12.3% | -7.1% | -13.0% |
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

## Recent Implementation Updates (2026-05-14)

### v2.65 VPIN Microstructure Signal - COMPLETED
- **Engine**: `src/signals/vpin_bvc.py` (564 lines) — BVC volume clock, VPIN toxicity scoring
- **Data**: Fetches real OHLCV from Yahoo Finance v8 API (market.db fallback for close-only)
- **Integration**: SmartRebalanceGate auto-computes VPIN from 60-day history
- **Behavior**: VPIN >0.5 triggers `defer_toxicity` — delays rebalances during informed trading
- **Current**: SPY VPIN = 0.57 (moderate-high toxicity)

### v2.3 Live Trading Prep - COMPLETED (All Phases)
- **Phase 1**: `src/broker/position_sync.py` — hourly broker↔local reconciliation
- **Phase 2**: `src/broker/order_router.py` — signal→order conversion with dry-run mode
- **Phase 3**: Exponential backoff retry (3 attempts), 300ms rate limiting, kill switch
- **Phase 4**: `BrokerPanel.tsx` — dashboard component showing broker positions, drift, orders
- **Commit**: `df8e606`

### v2.90 Smart Rebalancing - COMPLETED
- **Drift triggers**: Per-asset drift thresholds with urgency levels
- **VPIN timing**: Defers execution when microstructure toxicity is high
- **Cost budget**: 50bps annual limit with YTD tracking
- **Dashboard**: SmartRebalancePanel with drift bars, VPIN indicator, cost gauge

### v2.80 Phase 2 Multi-Asset Trend Following (DBC) - REJECTED
- **CLI**: Added 4-part `--portfolio SPY/GLD/TLT/DBC` support to `src/signals/multi_speed_momentum.py`
- **Backtest 2008-2026** (monthly rebalance, multi-speed momentum overlay):
  - 3-asset 46/38/16: CAGR 10.26%, Sharpe **0.904**, MaxDD -24.8%
  - 4-asset 46/34/16/4 (DBC funded by GLD): CAGR 9.52%, Sharpe **0.847**, MaxDD -27.0%
  - **Sharpe delta: -0.057** — DBC at 4% degrades risk-adjusted return
- **Crisis breakdown**: DBC hurts 2008 (-2.4pp) and 2020 (-1.0pp), helps 2022 (+1.9pp). Net negative.
- **Decision**: Reject Phase 2 at 4% weight. Phase 3 (synthetic short) deferred.
- **Follow-ups**: DBC weight sweep (2/3/5/6%), regime-gated DBC, or fund from SPY/TLT instead of GLD.
- **Spec**: `wiki/projects/portfolio-lab/work/2026-05-13-v280-multi-asset-trend-following/spec.md`

### v2.71 Intraday Seasonality Execution - COMPLETED
- **Phase 1**: Intraday cost model with symbol-specific profiles
- **Phase 2**: Rebalance scheduler with optimal window (11:00-14:00 ET) selection
- **Phase 3**: ExecutionAgent integration with urgency-based scheduling
- **Commit**: `6f0620d`

**Features**:
- Cost reduction: 5-15 bps per rebalancing trade
- Urgency mapping: >0.75 = immediate, <0.25 = wait for optimal window
- Dashboard status integration

### v3.16 Dual-Mode Cron Resilience - COMPLETED
- **Feature flag**: `CRON_BACKEND` env var (hermes/crontab/manual), `src/cron_compat.py`
- **Ops layer**: `Makefile` (8 targets + verify-cron-sync), project-local `data/cron_status.json`
- **Standalone**: `crontab` file for operation without Hermes Agent
- **ADR**: `wiki/projects/portfolio-lab/architecture/adr-dual-mode-cron-resilience.md`
- **Concept**: `wiki/concepts/dual-mode-cron-agent-resilience.md` (generalized pattern)

### v3.19-v3.22 Q3 2026 Trending Strategies Implementation - COMPLETED
- **v3.19 ML-Enhanced FX Carry Timing** ✅
  - RandomForest classifier for carry unwind prediction
  - 4701 training samples, 141 unwind events (3.0%), CV F1: 0.037
  - Feature importance: volatility_1m (0.32), carry_signal (0.17), momentum_1m (0.14)
  - Current UUP unwind risk: 0.187 (low) → 5.0% carry allocation
  - Data: FX tickers (UUP, UDN, FXE, FXY, FXB, FXA, FXC, FXF) added to fetcher.ts
  - Tests: 18 passing, model saved to `data/fx_carry_ml_model.pkl`

- **v3.20 Commodity Curve Overlay** ✅
  - Futures curve shape (contango/backwardation) gating for DBC allocation
  - Current: DBC in backwardation (+5.93%), allocation allowed at 5.0%
  - Contango → -12% expected returns, backwardation → +8% expected returns
  - Tests: 31 passing, `src/signals/commodity_curve.py` (330 lines)

- **v3.21 GARCH-Filtered CVaR Enhancement** ✅
  - GARCH(1,1) volatility filtering for tail risk estimation
  - 15-20% better tail risk estimates during volatility clustering
  - Current: CVaR 95% -1.9%, VaR 95% -1.27%, ratio 1.50x (moderate)
  - Tests: 45 passing, `src/monitor/garch_cvar.py` (443 lines)

- **v3.22 Entropy-Based Diversification Monitor** ✅
  - Shannon entropy + effective N + HHI for concentration risk
  - Current portfolio: H=1.02, N_eff=2.77, HHI=0.38 (good diversification)
  - Correlation structure entropy via eigenvalue decomposition
  - Tests: 38 passing, `src/monitor/entropy_monitor.py` (372 lines)

### v3.14 Credit Spread Signal - COMPLETED
- **Signal**: High-yield credit spread trend and level monitoring
- **Thresholds**: >500bps (distressed), 350-500bps (elevated), <350bps (normal)
- **Current**: 298bps (NORMAL) → Risk-on regime
- **Tests**: 24 passing, `src/signals/credit_spread.py` (387 lines)

## Test Coverage

### Test Safety: 4-Layer CPU Exhaust Defense
Heavy ML libraries (torch 63MB, sklearn 78MB, hmmlearn 23MB) can OOM-kill the test
suite on low-resource hosts (sg01). A 4-layer defense guarantees this never happens:

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| 0 | `collect_ignore` in conftest.py | Heavy test files **never opened** by pytest |
| 1 | `PORTFOLIO_LAB_ENABLE_ML=0` env var | ML features disabled before any import |
| 2 | `builtins.__import__` hook | Blocks torch/sklearn/xgboost/hmmlearn at interpreter level |
| 3 | Post-collection leak check | Warns if real ML libs evaded all guards |

**Layer 0 is the strongest**: `collect_ignore = ["test_execution_agent.py", ...]` in
`tests/conftest.py` prevents pytest from even opening those files during directory
listing. New heavy test files MUST be added to this list.

### Python (tests/)
- **3938 safe** tests (134 heavy excluded via collect_ignore, never imported)
- **4072 total** collected when `PORTFOLIO_LAB_ENABLE_ML=1 --include-heavy`
- ~3100 passing, pre-existing failures in yield curve and a few other suites
- 114 test files covering signals, strategy, dashboard, broker, agents, data, research
- **Safe**: `make test` or `bash scripts/run-tests-safe` (ML disabled, 3GB ulimit cap)
- **ML**: `make test-ml` or `PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/ --include-heavy`

### TypeScript (tests/ts/)
- **167 tests** across 9 files (DSR 24, duration-signals 35, purged-cv 21, car25 23, stress-validation 15, sector-attribution 19, sector-momentum 15, leveraged-treasury 7)
- Run: `bun test tests/ts/`
- Bun native test runner, zero configuration needed

## Analysis Scripts (src/backtest/)
- `grid-search.ts` — 94-config allocation sweep
- `rolling-window.ts` — 9 sub-period Sharpe validation
- `correlation-regime.ts` — 12-regime correlation analysis
- `recovery-analysis.ts` — drawdown events + 4% withdrawal GFC simulation
- `withdrawal-sweep.ts` — 7 rates × 8 portfolios × 4 scenarios
- `rebalance-tolerance.ts` — ±10% allocation tolerance + frequency comparison
- `monte-carlo-fire.ts` — 1000 bootstrap Monte Carlo FIRE simulation
- `factor-tilt.ts` — MTUM/VLUE/USMV factor tilt analysis (2013-2026)
- `commodities-sweep.ts` — DBC as partial GLD replacement sweep
- `tactical-rebalance.ts` — drift-based vs calendar rebalancing analysis

## AI Agents (src/agents/ v2.51)
- `analyst_agent.py` — Fundamental/value analysis with PPO policy (321 lines)
- `sentiment_agent.py` — News/social sentiment with contrarian detection (332 lines)
- `risk_agent.py` — VaR/CVaR monitoring with drawdown alerts (412 lines)
- `execution_agent.py` — Order timing with market impact modeling (379 lines)
- `controller_agent.py` — Master orchestration with centralized critic (458 lines)
- `agent_graph.py` — LangGraph-style communication topology (394 lines)
- `marl_trainer.py` — MAPPO training with GAE and value decomposition (543 lines)
- `ai_controller.py` — Main entry point with signal integrator bridge (469 lines)

CLI Usage:
```bash
python -m src.agents.ai_controller --mode status
python -m src.agents.ai_controller --mode infer --portfolio 46/38/16
python -m src.agents.ai_controller --mode train --episodes 500
```

## Wiki Compound Pages (97+ total)
- 11 original research: grid-search-results, rolling-window-analysis, correlation-regime-analysis, drawdown-recovery-fire, fire-withdrawal-rebalance-tolerance, monte-carlo-fire-simulation, decision-framework, factor-tilt-analysis, commodities-analysis, tactical-rebalancing
- 86+ strategy/cycle pages in `wiki/projects/portfolio-lab/compound/`
- Full index: `wiki/projects/portfolio-lab/knowledge.md` (auto-generated)

## Environment Gotchas
- `bc` is NOT available — use `date +%s` for duration math, avoid `date +%s%N | bc`
- Makefile `define` with multiline Python is fragile — use separate helper scripts (see `scripts/cron_update.py`)
- `skillwiki validate` requires `started:`, `updated:`, `completed:` (when status=completed) frontmatter fields
- `hermes chat -q "<prompt>"` gets one-shot advice from Hermes agent without interactive session
- `make verify-cron-sync` catches backend drift — run after changing Makefile targets or crontab

## Python: uv Package Manager

All Python dependencies managed via [uv](https://docs.astral.sh/uv/). Core deps
in `pyproject.toml`, ML deps (torch/xgboost) in `[dependency-groups] ml`.

```bash
uv sync                  # install core deps (no ML libs)
uv sync --group ml       # install core + ML deps
uv run python script.py  # run a script
uv run pytest tests/     # run tests (ML disabled by default)
```

**ML features disabled by default.** Set `PORTFOLIO_LAB_ENABLE_ML=1` to enable:
```bash
PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/ -m heavy
PORTFOLIO_LAB_ENABLE_ML=1 uv run python -m src.agents.ai_controller --mode status
```

## Dev Constraints (HARD RULES)

### No ML imports without explicit user override
- **NEVER** import `torch`, `xgboost`, `sklearn`, `hmmlearn` without user explicitly requesting ML
- torch 63MB + sklearn 78MB + hmmlearn 23MB = **164MB+** total, OOM-kills at ~49% on sg01
- **ALWAYS** keep `PORTFOLIO_LAB_ENABLE_ML=0` (the default) for all test runs and dev work
- **ONLY** set `PORTFOLIO_LAB_ENABLE_ML=1` when the user explicitly asks for ML agent work
- **Safe test run**: `make test` (4-layer defense: collect_ignore + env var + import hook + 3GB ulimit)

### ML-gated modules (do NOT import these without user request)
These modules import torch/sklearn/hmmlearn and will stall the machine without `PORTFOLIO_LAB_ENABLE_ML=1`:
- `src/agents/ai_controller.py` (492 lines) — MARL entry point
- `src/agents/analyst_agent.py` (321 lines) — PPO policy
- `src/agents/controller_agent.py` (458 lines) — centralized critic
- `src/agents/sentiment_agent.py` (332 lines) — sentiment RL agent
- `src/agents/agent_graph.py` (394 lines) — LangGraph topology
- `src/agents/marl_trainer.py` (543 lines) — MAPPO training
- `src/agents/risk_agent_hmm.py` (600+ lines) — HMM-LSTM regime detector (sklearn/hmmlearn)
- `src/strategy/regime_hmm.py` (500+ lines) — Wasserstein HMM regime (hmmlearn)
- `src/agents/base_agent.py` (266 lines) — uses torch stubs (safe without ML, tested)

### How the ML gate works
`tests/conftest.py` provides a 4-layer defense:
1. **`collect_ignore`** — known heavy test files never opened by pytest (0 CPU)
2. **`builtins.__import__` hook** — blocks torch/sklearn/xgboost/hmmlearn at interpreter level
3. **Post-collection check** — warns if real ML libs evaded the hook (checks `__file__`/`__version__`)
4. **`make test` ulimit -v** — OS kernel enforces 3GB virtual memory cap

`src/agents/base_agent.py` and `src/agents/execution_agent.py` use conditional imports
(`if os.environ.get("PORTFOLIO_LAB_ENABLE_ML") == "1": import torch else: stubs`).
`src/agents/risk_agent_hmm.py` and `src/strategy/regime_hmm.py` use the same pattern
for sklearn/hmmlearn. These stubs are registered in `sys.modules` so any subsequent
`import torch` finds the 0MB stub rather than the 63MB real library.

### Test coverage for ML-gated modules
- Tests for agent modules exist (`test_marl_trainer.py`, `test_base_agent.py`, etc.) but require mocking
- `test_base_agent.py` (54 tests) runs without ML — uses torch stubs
- Remaining agent modules (`analyst_agent`, `controller_agent`, `sentiment_agent`, `ai_controller`, `agent_graph`) lack dedicated test files — write them ONLY when the user explicitly requests ML agent work

### Preferred dev targets (no ML, safe to test anytime)
These modules have NO ML deps and are always safe to work on:
- `src/strategy/` — comparison, evaluator, dual_momentum, etc.
- `src/signals/` — signal modules (credit_spread, commodity_curve, etc.)
- `src/broker/` — broker integration (options_utils, order_router, position_sync)
- `src/monitor/` — entropy_monitor, garch_cvar, etc.

## Quick Start
```bash
make test            # safe test suite (ML disabled, 3GB memory cap, 3775 tests)
make test-ml         # full suite including ML (needs >3GB RAM)
bash scripts/run-tests-safe           # standalone safe runner with --ml flag
PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/  # manual safe run
```

## To Run
```bash
cd /Users/karlchow/Desktop/code/portfolio-lab
bun run dev          # dev server
bun run build        # production build
bun run fetch-data   # refresh data from Yahoo Finance v8 API
```

## Data Pipeline
1. `bun run fetch-data` → fetches from Yahoo Finance v8 chart API (auto-detects today's date)
2. Saves to `public/data/prices.json` (compact: {d, p} per symbol, ~2.4MB)
3. App loads `/data/prices.json` on startup, runs backtests client-side

## Cron Compatibility Contract (dual-mode: Hermes + system crontab)

portfolio-lab supports **three cron backends** via `CRON_BACKEND` env var:
- `hermes` (default) — Hermes Agent cron scheduler (11 jobs in `~/.hermes/scripts/`)
- `crontab` — system crontab (standalone, no Hermes needed)
- `manual` — `make <target>` from terminal or Claude Code

### Feature flag

Import from `src/cron_compat.py` — never hardcode Hermes paths in application code:
```python
from src.cron_compat import IS_HERMES, IS_CRONTAB, BACKEND, CRON_TARGETS
```

### When adding a new cron job

You MUST update three files in lockstep:
1. **`Makefile`** — add a `.PHONY` target that runs the module + calls `scripts/cron_update.py`
2. **`crontab`** — add a crontab entry for standalone mode
3. **`src/cron_compat.py`** — add the job name to `CRON_TARGETS` list

### When changing code that a cron job calls

- The Makefile target is the **source of truth** for how each job runs. If you change CLI flags, env vars, or module paths, update the Makefile target first.
- After changing a Makefile target, run `make verify-cron-sync` to confirm the crontab file still matches.
- Do NOT add `~/.hermes/` path dependencies — use project-relative paths only. The one exception is `src/dashboard/generator.py` which reads `data/cron_status.json` (backend-agnostic).

### When changing generator.py or dashboard data

- `generator.py` reads `data/cron_status.json` (not `~/.hermes/cron/state.json`). Keep the JSON format stable: `{jobs: [{name, status, last_run, duration_seconds, backend}]}`.
- The `backend` field in each job entry tracks which runner executed it (`hermes`, `crontab`, `manual`).

### Verification

```bash
make verify-cron-sync          # check Makefile ↔ crontab ↔ cron_status.json sync
CRON_BACKEND=crontab make all  # test full pipeline with crontab backend
python3 -c "from src.cron_compat import active_backend; print(active_backend())"  # discover active backend
```

### Switching backends

```bash
# To system crontab:
hermes cron pause <ids> && crontab crontab

# Back to Hermes:
crontab -r && hermes cron resume <ids>
```

See `compound/dual-mode-hermes-claude-code-resilience.md` in wiki for full architecture.
