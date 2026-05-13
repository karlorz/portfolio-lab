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

## Recent Implementation Updates (2026-05-13)

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

## Test Coverage (tests/)
- **201/201 passing** — all tests green
- `test_integrator.py` — 34 tests: data structures, normalization, composite signal aggregation, regime detection, allocation deltas, signal agreement, signal history
- `test_order_router.py` — 14 tests: signal-to-order conversion, kill switch, dry-run, retry logic, price fetching
- `test_position_sync.py` — 7 tests: drift calculation, edge cases
- `test_signal_execution_bridge.py` — 21 tests: urgency classification, allocation deltas, order generation
- `test_liquidity_checks.py` — 18 tests: premium thresholds, trade eligibility, force override
- `test_rebalance_scheduler.py` — 14 tests: order scheduling, urgency windows, batch rebalancing
- `test_vpin_rebalancer.py` — 13 tests: BVC calculator, VPIN engine, smart rebalancer integration
- `test_sentiment_client.py` — 30 tests: LLM sentiment client, cost tracking, retry logic, JSON parsing
- `test_tips_monitor.py` — 11 tests: TIPS yield tracking, signal generation, regime classification, allocation guidance
- `test_alternative_data.py` — 22 tests: data classes, database init, satellite/credit card/supply chain adapters, composite signals, earnings predictions, agreement detection

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

## Wiki Compound Pages (11 total)
- grid-search-results
- rolling-window-analysis
- correlation-regime-analysis
- drawdown-recovery-fire
- fire-withdrawal-rebalance-tolerance
- monte-carlo-fire-simulation
- decision-framework
- factor-tilt-analysis
- commodities-analysis
- tactical-rebalancing

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
