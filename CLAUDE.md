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

## Recent Implementation Updates (2026-05-21)

### v9.35 BanditWeighter Removal + Weight Rebalancing + Coverage Expansion - COMPLETED
- **BanditWeighter removed**: 266 lines of dead code deleted
  - update_bandit() had zero callers — bandit never received observations
  - get_blended_weights() always returned 100% static REGIME_WEIGHTS via cold-start path
  - Simplified to get_regime_weights() returning static weights directly
  - Deleted: tests/test_bandit_weighter.py (10 tests for dead class)
- **MSM weight reduced**: 21% → 10% across all regimes (net-negative -0.012 Sharpe per v9.24)
  - Redistributed to ALT_DATA (21→26%, +0.015 Sharpe) and INTL_MOM (17→21%, +0.02 Sharpe)
  - UNIFIED_OVERLAY increased from 15→17-21% depending on regime
- **ALT_DATA enum naming fix**: health_tracker.py ALT_DATA → ALTERNATIVE_DATA (matches ensemble_voter.py)
- **ODTE yield tests**: 170 tests for 3 previously untested modules (1293 lines)
  - test_odte_yield.py: ZeroDTECalculator, StrikeSelector, ZeroDTEPosition + dataclasses + enums
- **4 individual overlay backtests** (previously only group-tested via combined overlay):
  - src/backtest/vixy_hedge_backtest.py + 29 tests — VIX-based hedge allocation
  - src/backtest/collar_overlay_backtest.py + 44 tests — VIX-gated collar with CRISIS freeze
  - src/backtest/bond_duration_backtest.py + 46 tests — TLT momentum-driven duration rotation
  - src/backtest/crypto_allocation_backtest.py + 37 tests — SPY momentum-gated crypto, 5% cap
- **Test count**: 6424 safe (0 failures, 10 skipped)
- **Status**: All phases complete

### v9.31 UNIFIED_OVERLAY Activation + Dead Code Removal - COMPLETED
- **UNIFIED_OVERLAY activated**: Added 15% weight to all 4 regime weights (was dead code at 0%)
  - The orchestrator_ensemble_bridge.py was generating SignalReadings that compute_vote() silently discarded
  - UNIFIED_OVERLAY collection added to collect_signals() via bridge import
  - Weights redistributed proportionally from other signals (MSM 25→21%, ALT_DATA 25→21%, etc.)
- **vix_overlay.py removed**: 2000 lines of dead code (module + backtest + tests)
  - No production caller anywhere in src/ — fully disconnected from portfolio system
  - Removed: src/strategy/vix_overlay.py (570L), src/backtest/vix_overlay_backtest.py (590L), tests/test_vix_overlay.py (466L), tests/test_vix_overlay_backtest.py (374L)
- **Bug fix**: ResearchAgent.delegate_to_claude() returned `str(work_file)` instead of `Path` — broke create_claude_prompt() calling `.with_suffix()` on string
- **Tests**: 56 new for strategy/factor_rotation.py (1038L, largest untested non-ML module), 31 new for research/agent.py (328L), 14 backtest validation tests for vixy_hedge + unified_orchestrator
- **Test count**: 6189 safe (0 failures, 10 skipped)
- **Status**: All phases complete

### v9.18 Crypto Institutional Test Suite - COMPLETED
- **Tests**: 58 new tests for `src/crypto/institutional.py` (902-line module, previously 0 tests):
  - `tests/test_crypto_institutional.py` (58 tests): 4 dataclasses, init_database, TokenizedTreasuryStrategy (allocation + rebalance + circuit breaker + product performance + yield), CryptoRiskManager (risk assessment + compliance + rebalance deltas), CLI (4 commands), constants, edge cases
  - Circuit breaker: all 5 states tested (green/yellow/orange/red/black)
  - Edge cases: invalid risk profile fallback, zero portfolio, rebalance boundary
- **Test count**: 6324 → 6382 safe
- **Status**: All phases complete

### v9.21 Coverage Gap Closure Round 2 - COMPLETED
- **Tests**: 70 new tests across 2 previously undertested modules:
  - `tests/test_analytics_calculator.py` (52 tests): AnalyticsCalculator — drawdown series, max drawdown, rolling Sharpe, benchmark comparison, report generation, edge cases
  - `tests/test_rebalancing_backtest.py` (+18 tests, 30→41 total): Added run_smart_strategy, run_full_backtest, print_comparison, save_results, calendar/drift edge cases
- **Test count**: 6382 → 6452 safe (6405 total passed, 15 pre-existing yield/sentiment failures)
- **Status**: All phases complete

### v9.22 Test Pollution Fix + Coverage Expansion - COMPLETED
- **Fix**: `tests/test_rebalancing_backtest.py` — eliminated sys.modules pollution that caused 22 test failures across 7 downstream files
  - Root cause: module-level `sys.modules` eviction of `src.rebalancing.*` corrupted Python import namespace for all subsequent tests
  - Fix: replaced `sys.modules` manipulation + `importlib.reload` with `unittest.mock.patch.object` on backtest module symbols
  - Previously failing files now all pass: test_sentiment_client (8), test_vix_insurance_signal (5), test_yield_curve_regime (3), test_yield_dashboard (3), test_tsmom_overlay (1), test_visibility_graph (1), test_vix_futures (1)
- **Tests**: 56 new tests across 2 previously untested modules:
  - `tests/test_fx_fetcher.py` (31 tests): FXMetrics dataclass, FXFetcher DB init/cache, Yahoo API mock, metrics computation (carry regime/vol regime/USD strength), signal generation, save_metrics, edge cases
  - `tests/test_factor_correlation.py` (25 tests): calculate_returns, calculate_correlation (positive/negative/insufficient/zero variance), load_factor_prices, build_correlation_matrix (diagonal/symmetric/bounded), analyze_redundancy, generate_report
- **Test count**: 6452 → 6506 safe (0 failures, previously 22 pre-existing failures now all fixed)
- **Status**: All phases complete

### v9.23 Ensemble Weight Rebalancing + Zero-Weight Signal Skip - COMPLETED
- **Weight cap**: MULTI_SPEED_MOM capped at 50% max per regime (was 60-70%)
  - Reduces single-point-of-failure: if dominant signal degrades, ensemble can compensate
  - NORMAL: 60→50%, excess redistributed to CROSS_ASSET_RV (+3%), ALT_DATA (+2.5%), INTL_MOM (+2%), REGIME_ARB (+2.5%)
  - HIGH_VOL: 60→50%, excess redistributed to ALT_DATA (+3.75%), CROSS_ASSET_RV (+2.5%), INTL_MOM (+2.5%), REGIME_ARB (+1.25%)
  - CRISIS: 70→50%, excess redistributed to CROSS_ASSET_RV (+13.33%), REGIME_ARB (+6.67%)
  - RECOVERY: unchanged (already 43%)
- **Zero-weight skip**: collect_signals() now accepts regime parameter and skips sources with 0.000 weight
  - CRISIS regime: INTERNATIONAL_MOMENTUM and ALTERNATIVE_DATA computation skipped (was wasting CPU)
  - Backward compatible: CLI call without regime collects all signals
- **Tests**: Added `test_no_signal_exceeds_50_pct` to enforce weight cap invariant
- **Test count**: 6506 → 6508 safe (0 failures)
- **Status**: All phases complete

### v9.24 Individual Signal Backtest Validation - COMPLETED
- **Backtests**: 5 new backtest files for all active ensemble signals (previously 0 had individual validation)
  - `src/backtest/multi_speed_momentum_backtest.py` — dominant 50% weight signal
  - `src/backtest/alternative_data_backtest.py` — hardcoded regime→signal mapping (never validated before)
  - `src/backtest/cross_asset_regime_arb_backtest.py` — equity/bond/gold divergence detection
  - `src/backtest/international_momentum_backtest.py` — EFA/EEM vs SPY momentum
  - `src/backtest/cross_asset_rv_backtest.py` — z-score mean-reversion triggers
- **Key Findings** (2006-2026, baseline Sharpe 0.942):
  - MULTI_SPEED_MOM: Sharpe 0.930 (**-0.012 vs baseline**) — dominant signal is net-negative as overlay
  - ALTERNATIVE_DATA: Sharpe 0.957 (+0.015) — only signal showing positive alpha contribution
  - CROSS_ASSET_REGIME_ARB: Sharpe 0.942 (0.000) — neutral, no alpha added or subtracted
  - INTERNATIONAL_MOMENTUM: Sharpe ~0.96 (+0.02) — modest positive contribution
  - CROSS_ASSET_RV: Sharpe ~0.94 (-0.00x) — marginal, near-zero contribution
- **Implication**: The dominant signal (MULTI_SPEED_MOM at 50% weight) is actually a net-negative overlay. The ensemble performs well because the baseline allocation (46/38/16) is already strong, not because the signal adds value. This mirrors the earlier finding that TSMOM alone beats combined signals.
- **CLI**: Each backtest has `run` and `--save` commands
- **Test count**: 6189 safe (0 failures, 10 skipped)
- **Status**: All phases complete

### v9.26 Signal Weight Rebalancing - COMPLETED
- **Root cause**: v9.24 backtesting revealed MULTI_SPEED_MOM (50% weight) is net-negative (Sharpe -0.012 vs baseline 0.942)
- **Changes**: Reduced MSM from 50% to 25% across all regimes. Increased ALTERNATIVE_DATA (only positive alpha +0.015) and INTERNATIONAL_MOMENTUM (+0.02)
- **New weights** (NORMAL): MSM 0.25, ALT_DATA 0.25, INTL_MOM 0.20, CROSS_RV 0.15, REGIME_ARB 0.15
- **Tests**: Updated regime dominance assertions, 5839/5839 passing
- **Status**: All phases complete

### v9.30 Backtest Consolidation + Coverage Expansion - COMPLETED
- **Backtest refactor**: All 11 backtest files now import from `src/backtest/metrics.py` — 99 lines removed, zero duplicated BacktestResult dataclasses
- **agent_graph.py tests**: 79 tests for core MARL orchestration topology (NodeType, GraphEdge, message routing, topology viz, save/load)
- **research/features.py tests**: 76 tests for feature engineering pipeline (Features dataclass, FeaturePipeline, FeatureStore, CLI)
- **Test count**: 6059 → 6135 safe (0 failures, 6 pollution-affected in full suite)
- **Status**: All phases complete

### v9.29 Dead Enum Cleanup + Doc Accuracy - COMPLETED
- **SignalSource enum**: 13 → 6 entries (5 active + UNIFIED_OVERLAY). Removed 7 dead values kept only for stacking_feature_engine.py demo
- **CLAUDE.md fixes**: Marked 6 "EnsembleVoter weight" claims as "planned, NOT in REGIME_WEIGHTS" — collar (10%), crypto (5%), bond duration (8%), VIX (15%), factor rotation (5%), orchestrator (20%) were documented as completed but never implemented
- **Test count**: 5980 → 6059 safe (0 failures)
- **Status**: All phases complete

### v9.28 Coverage Expansion + Source Bug Fixes - COMPLETED
- **Tests**: 141 new tests across 3 previously untested modules:
  - `tests/test_wiki_sync.py` (51 tests): WikiSync init, save_raw_source, sync methods, regime/graduation helpers
  - `tests/test_pipeline.py` (33 tests): init_db, fetch_yahoo, detect_regime, check_data_quality
  - `tests/test_multi_strategy_adapters.py` (49 tests, rewritten): all 3 adapters, signals, confidence, portfolio
- **Source bugs fixed**: 5 bugs found by tests:
  - wiki_sync.py: f-string ValueError, unescaped `{` in citations (2), division by zero in win rate
  - pipeline.py: SELECT missing symbol column → detect_regime always returned None
  - multi_strategy_adapters.py: empty list `tickers=[]` fell through to defaults
- **Test count**: 5862 → 5980 safe (0 failures)
- **Status**: All phases complete

### v9.27 Shared Metrics Module + Wiki Sync Fix - COMPLETED
- **Metrics module**: `src/backtest/metrics.py` — shared BacktestMetrics, compute_metrics(), compute_crisis_returns(), save_results_json()
  - Eliminates ~800 lines of copy-paste metric logic across 11 backtest files
  - Future backtests should import from this module instead of defining their own
  - Tests: 23 new tests (5839 → 5862 safe)
- **Wiki sync fix**: `src/research/wiki_sync.py` — stable filenames + skip unchanged writes
  - Root cause: save_raw_source() created new timestamped JSON every 30-min cron run
  - Fix: use stable filename (name.json) and skip write when content hash matches existing file
  - Eliminates unbounded disk growth and citation-only diffs in compound pages
- **Test count**: 5862 safe (0 failures)
- **Status**: All phases complete

### v9.25 Dead Code Purge - COMPLETED
- **Removed**: 49 files (22,695 lines) — v9.19 pruning follow-through
  - 26 source files (12,917 lines): 12 signal modules, 4 data fetchers, 2 feature pipelines, 1 execution module, 1 monitor module, 5 backtest engines, 1 signal infrastructure
  - 23 test files (9,778 lines): corresponding tests for all removed modules
- **Enum cleanup**: Removed 15 deprecated SignalSource entries with zero external references. Kept 8 entries still referenced by ML-gated stacking_feature_engine.py and orchestrator_ensemble_bridge.py
- **SignalSource enum**: 13 entries (5 active + 8 kept-for-compat deprecated) — down from 29
- **Test count**: 6508 → 5839 safe (670 dead tests removed, 1 test updated)
- **Status**: All phases complete

### v9.19 Ensemble Voter Dead Signal Pruning - COMPLETED
- **Pruned**: 14 deprecated signal collection blocks from `collect_signals()` — 396 net lines removed (566→170 lines)
  - Removed: MACRO_MOMENTUM, CLOSING_AUCTION, FACTOR_ROTATION, MEAN_REVERSION, TRANSIENT_FACTORS, VISIBILITY_GRAPH, VP_MACD, FACTOR_TIMING, LLM_NARRATIVE, MACRO_REGIME_SYNTHESIS, FX_CARRY, COMMODITY_CURVE, ZERO_DTE
  - All had zero weight in REGIME_WEIGHTS — computation was discarded in compute_vote()
- **Removed from REGIME_WEIGHTS**: TSFM_MOMENTUM and DURATION_REGIME (had weight but no data feeds)
- **Active signals**: 5 — MULTI_SPEED_MOM, CROSS_ASSET_RV, INTERNATIONAL_MOMENTUM, ALTERNATIVE_DATA, CROSS_ASSET_REGIME_ARB
- **Weights renormalized** per regime to sum=1.0
- **Tests**: Updated 4 tests in test_ensemble_voter.py, 6382/6382 passing
- **Status**: All phases complete

### v9.16 Coverage Gap Closure - COMPLETED
- **Tests**: 60 new tests across 3 previously untested modules:
  - `tests/test_odte_executor.py` (27 tests): 0DTE options executor — enums, dataclasses, paper-mode simulation, exit logic
  - `tests/test_health_tracker.py` (20 tests): signal health tracking — DB init, prediction logging, health score formula, decay detection
  - `tests/test_health.py` (13 tests): HealthMonitor — kill switches, portfolio health, cron execution, data freshness, circuit breaker
- **Test count**: 6020 → 6080 safe
- **Bug found**: `paper_mode=False` overridden by env var in `ODTEExecutor.__init__` (False or True = True)
- **False positive**: `src/strategy/comparison.py` was flagged as untested but has `test_strategy_comparison.py` (71 tests)
- **Status**: All phases complete

## Recent Implementation Updates (2026-05-19)

### v9.11 Execution Module Test Coverage - COMPLETED
- **Tests**: 65 new tests across 3 untested non-ML modules:
  - `tests/test_goals.py` (35 tests): load/validate/risk_budget/account_type
  - `tests/test_rebalance_health.py` (14 tests): order parsing, schedule compliance
  - `tests/test_tca_scorecard.py` (16 tests): peer groups, trend analysis, quality distribution
- **Test count**: 5731 safe (5952 total)
- **Status**: All phases complete

### v9.10 GP-VCV Hybrid Gaussian Process Covariance Estimation - COMPLETED
- **Estimator**: `src/monitor/gp_vcv_estimator.py` (~380 lines) — GP-based VCV using sklearn GPR
  - Hybrid kernel: RBF (long-term) + Matérn 3/2 (short-term) + WhiteKernel (noise)
  - Per-asset variance prediction with correlation-based VCV reconstruction
  - EWMA baseline comparison, save/load state persistence
  - Based on arXiv:2605.17275
  - ML-gated: conditional sklearn import, collect_ignore for safe test mode
- **Integration**: `src/strategy/regime_optimizer.py` — estimator toggle
  - `estimator: 'ewma' | 'gp_vcv'` parameter (default: 'ewma' for backward compatibility)
  - Graceful ML fallback: GP-VCV → EWMA when ML disabled or import fails
- **Tests**: `tests/test_gp_vcv_estimator.py` (19 tests passing, ML-only, collect_ignore)
- **Expected Impact**: +0.01-0.02 Sharpe through smoother regime transitions
- **Status**: All phases complete

## Recent Implementation Updates (2026-05-16)

### v4.80 Dynamic Bond Duration Rotation - COMPLETED

### v4.90 Orchestrator-EnsembleVoter Bridge - COMPLETED
- **Bridge**: `src/strategy/orchestrator_ensemble_bridge.py` (300 lines) — orchestrator→ensemble voter
  - Converts 7-asset allocation weights to -1/+1 directional signals
  - Added `UNIFIED_OVERLAY` to `SignalSource` enum in ensemble_voter.py
  - Recommended 20% weight in ensemble (highest single-source) — NOT implemented in REGIME_WEIGHTS
  - Risk and execution signals from conflict count + calendar modifier
- **Tests**: `tests/test_orchestrator_ensemble_bridge.py` (21 tests passing)
- **Status**: All phases complete

### v4.90 Combined Overlay Backtest - COMPLETED
- **Backtest Engine**: `src/backtest/combined_overlay_backtest.py` (360 lines) — validates theoretical Sharpe
  - Runs all 4 overlays together on historical/synthetic data (2006-2026)
  - Synthetic data with realistic correlations and crisis regimes
  - Crisis decomposition: 2008, 2020, 2022 returns
  - Overlay activity tracking: % days active per overlay
  - Target validation: Sharpe >= 0.90, Max DD >= -22%
  - CLI: `run`, `--save` for JSON output
- **Tests**: `tests/test_combined_overlay_backtest.py` (20 tests passing)
- **Status**: All phases complete

### v4.90 Unified Overlay Orchestrator - COMPLETED
- **Orchestrator**: `src/strategy/unified_orchestrator.py` (430 lines) — multi-overlay integration
  - Combines 4 overlays: collar (25%), bond duration (25%), crypto (15%), calendar (10%)
  - Weighted-sum conflict resolution with hard constraint enforcement
  - 7-asset model: SPY, GLD, TLT, IEF, SHY, BTC, ETH
  - Hard bounds: SPY 36-56%, GLD 28-48%, bonds 6-26%, crypto 0-5%
  - Estimated Sharpe projection: baseline 0.79 → 0.91 target
- **Tests**: `tests/test_unified_orchestrator.py` (23 tests passing)
- **Status**: All phases complete

### v4.60 Collar Live Options Integration - COMPLETED
- **Bridge**: `src/broker/collar_options_bridge.py` (330 lines) — collar↔broker options chain
  - Async options chain fetch with graceful fallback (no PriceFetcher, no API keys)
  - Live chain delta search: call 0.30, put -0.20
  - Liquidity filtering: min volume 10, min OI 100, max spread 5%
  - Live-vs-theoretical comparison, numpy-safe JSON serialization
- **Tests**: `tests/test_collar_options_bridge.py` (15 tests passing)
- **Status**: All phases complete
- **Signal Generator**: `src/signals/bond_duration_signal.py` (280 lines) — yield curve + real rate analysis
  - 4 curve regimes: STEEP (>1.0%), NORMAL (0.3-1.0%), FLAT (0.0-0.3%), INVERTED (<0.0%)
  - 3 rate directions: FALLING, STABLE, RISING (6-month trend)
  - 12-cell strategy matrix: curve regime × rate direction → TLT/IEF/SHY weights
  - Real rate >2% shifts toward longer duration for carry
- **Rotation Strategy**: `src/strategy/bond_duration_rotator.py` (260 lines) — bond sleeve allocation
  - Rotates 16% bond sleeve across TLT (16yr dur), IEF (7yr), SHY (2yr)
  - Baseline comparison engine: static TLT vs dynamic rotation
  - 8% EnsembleVoter weight (planned, NOT in REGIME_WEIGHTS), state persistence
- **Tests**: `tests/test_bond_duration_signal.py` (36 tests) + `tests/test_bond_duration_rotator.py` (22 tests) = 58 tests passing
- **Expected Impact**: +0.02-0.03 Sharpe through better risk-adjusted fixed-income positioning
- **Status**: All phases complete

### v4.70 Crypto Tactical Allocation - COMPLETED
- **Signal Generator**: `src/signals/crypto_momentum.py` (340 lines) — BTC/ETH momentum + vol regime
  - 6-month/3-month/1-month momentum computation with 180-day lookback
  - 4 vol regimes: LOW (<40%), NORMAL (40-70%), HIGH (70-100%), EXTREME (>100%)
  - Vol-scaling: target 40% annualized, position range 0.25x-2.0x
  - BTC 60% / ETH 40% of crypto sleeve, funded from GLD
- **Tactical Overlay**: `src/strategy/crypto_allocation.py` (280 lines) — allocation + backtest
  - Entry: 6m momentum positive + vol regime normal/low
  - Exit: momentum negative OR vol extreme (>100% ann.)
  - Hard cap: 5% portfolio, 5% EnsembleVoter weight (planned, NOT in REGIME_WEIGHTS)
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
- **Integration**: 10% weight in EnsembleVoter via CollarOverlayIntegrator (planned, NOT in REGIME_WEIGHTS)
- **Tests**: `tests/test_collar_signal.py` (49 tests) + `tests/test_collar_overlay.py` (26 tests) = 75 tests passing
- **State**: `data/collar_overlay_state.json` — tracks current collar status
- **Target**: Max DD -26.2% → ≤-20%, Sharpe +0.03-0.05
- **Status**: All phases complete

## Recent Implementation Updates (2026-05-16)

### v6.00 Post-Trade TCA Engine - COMPLETED
- **Engine**: `src/execution/tca_engine.py` (~680 lines) — Order parsing from `orders.jsonl`, Almgren-Chriss impact decomposition, quality scoring (0-100), symbol/side aggregation
- **Scorecard**: `src/execution/tca_scorecard.py` (~300 lines) — Peer-group normalization, trend analysis, dashboard JSON export
- **Tests**: `tests/test_tca_engine.py` — 44 tests passing
- **Real Data**: 3 orders from May 11 rebalance analyzed (avg slip -10 bps, avg quality 43.3/100)
- **Commit**: `9e460c6`

### v5.72 Graduation Metrics Fix - COMPLETED
- **Fix**: `src/strategy/evaluator.py` — intra-day snapshot deduplication graduation bug
  - Added `_deduplicate_to_daily()` to filter 30-min snapshots to trading-day-level
  - Added vol floor `max(std, 0.0001)` and Sharpe cap `MAX_REALISTIC_SHARPE=3.0`
  - No more false PROMOTION CANDIDATE alerts from Sharpe 68+ blowup
- **Tests**: `tests/test_evaluator.py` — 10 new tests (41 total), all passing
- **Commit**: `fecedd4`
- **Status**: Complete

### v5.70+v5.71 Performance Attribution + Cross-Asset RV - COMPLETED
- **Performance Attribution**: Tracks each signal source contribution to P&L, hit rate, etc.
- **Cross-Asset Relative Value**: Z-score of N-asset rolling returns, mean-reversion triggers
- **Status**: Complete

### v7.04 Dynamic VIXY Hedge Sizing - COMPLETED
- **Sizing Engine**: `src/strategy/vixy_hedge_sizing.py` (~350 lines) — VIX-based hedge allocation
  - QuantPedia-derived formula: VIX/10 = target allocation% (VIX=28 → 2.8%)
  - 4 hedge regimes: NORMAL (VIX<20), ELEVATED (VIX 20-30), STRESS (VIX 30-40), CRISIS (VIX>40)
  - Regime-aware floors/ceilings: NORMAL(0-2%), ELEVATED(1-3.5%), STRESS(2-6%), CRISIS(3-10%)
  - Vol scaling: realized/implied vol ratio adjusts allocation
  - CRISIS freeze: collar complement disabled when VIXY >8%
  - CLI: `status`, `recommend`, `backtest`, `update` modes
- **Efficiency Monitor**: `src/monitor/hedge_efficiency.py` (~250 lines) — cost-benefit tracking
  - Drawdown detection from SPY cumulative returns
  - Running efficiency: YTD benefit / YTD cost with A-F grading
  - Strategy comparison: VIXY vs collar vs trend-following vs cash benchmarks
  - State persistence to `data/hedge_efficiency_state.json`
- **Ensemble Integration**: Added `VIXY_HEDGE` to `SignalSource` enum in ensemble_voter.py (removed in v9.25 enum cleanup)
  - Weights: NORMAL=5%, HIGH_VOL=10%, CRISIS=10%, RECOVERY=3% (were never in REGIME_WEIGHTS after v9.19 pruning)
- **Tests**: `tests/test_vixy_hedge_sizing.py` (34 tests) + `tests/test_hedge_efficiency.py` (25 tests) = 59 tests passing
- **Status**: All phases complete

### v3.10 ML Signal Stacking Ensemble — Phase 1-5 Complete

- **Feature Engineering**: `src/signals/stacking_feature_engine.py` — 84 features from 8 base signals
- **Training**: `src/ml/stacking_trainer.py` (688 lines) — XGBoost meta-learner, time-series CV
- **Inference**: `src/signals/stacking_integrator.py` (418 lines) — production inference with fallback
- **Backtest**: `src/backtest/stacking_ensemble_backtest.py` (~380 lines)
  - Monte Carlo simulation: 65% vs 76% directional accuracy impact
  - **Key Finding**: +11% accuracy produces +0.000 Sharpe when applied as +/-5% shift on 15% of days
  - Signal frequency and shift magnitude are the binding constraints, not accuracy
  - t-stat: -0.74 — not statistically significant
- **Dashboard**: `src/components/StackingEnsemblePanel.tsx` — Accuracy comparison, probability bars, feature importance
- **Tests**: 76 tests (20 feature + 17 trainer + 26 integrator + 13 backtest)
- **Status**: All phases complete

### v3.00 Factor Rotation — Backtest Validation Complete

- **Data**: `src/data/factor_data.py` (472 lines) — MTUM/QUAL/USMV/VLUE ETF data fetcher
- **Signal**: `src/signals/factor_rotation.py` (575 lines) — quality-momentum blend, regime-based allocation
- **Integration**: `FACTOR_ROTATION` in SignalSource enum, 5% ensemble weight (planned, NOT in REGIME_WEIGHTS — signal pruned in v9.19)
- **Backtest**: `src/backtest/factor_rotation_backtest.py` (~430 lines)
  - SPY baseline vs factor rotation on 2021-2026 data
  - Sharpe delta: -0.216 (defensive drag in bulls), DD improvement: +5.8pp (met)
  - Regime: bull -0.141, neutral 0.666, elevated 0.882, high_vol 1.474, crisis 0.588
  - Finding: Factor rotation is a defensive tool — reduces drawdowns, not an alpha generator
- **Tests**: `tests/test_factor_rotation.py` (29 tests) + `tests/test_factor_rotation_backtest.py` (16 tests) = 45 tests
- **Dashboard**: `src/components/FactorRotationPanel.tsx` — Factor pie chart, regime badge, Q+M score
- **Status**: All phases complete

### v2.70 Behavioral Sentiment Overlay — Phase 1-4 Complete

- **Data Fetcher**: `src/data/behavioral_sentiment_fetcher.py` (356 lines) — CBOE SKEW, VIX9D, P/C ratio, Reddit sentiment
  - `BehavioralSentimentSnapshot` dataclass with composite score (-3 to +3)
  - Sentiment weights: Options 35%, Retail 40%, Social 25%
  - SQLite cache with 4-hour TTL
- **Signal Generator**: `src/signals/behavioral_sentiment.py` (411 lines) — contrarian allocation signals
  - Z-score normalization against 90-day rolling window
  - Regime-gated suppression: VIX >30 disabled, VIX >25 half weight
  - Circuit breakers: 5-day churn control, earnings blackout, duplicate rejection
  - Historical backfill using VIX proxy for pre-2024 periods
- **Ensemble Integration**: 5% weight in combined_orchestrator.py
  - Conflict resolution: Trend wins over behavioral, Macro wins during Fed events
- **Walk-Forward Backtest**: `src/backtest/behavioral_sentiment_backtest.py` (~430 lines)
  - 2021-2026 validation: Sharpe delta **-0.216** — VIX-proxy contrarian signals degrade performance
  - False positive rate: 65.8%, 5-regime VIX bucket analysis
  - Finding: Simple VIX-level signals are net negative; real-time SKEW/PCR data needed
  - CLI: `run`, `--summary`, `--output` for JSON export
- **Tests**: `tests/test_behavioral_sentiment.py` (43 tests) + `tests/test_behavioral_sentiment_backtest.py` (19 tests) = 62 tests passing
- **Dashboard**: `src/components/BehavioralSentimentPanel.tsx` — Fear/Greed gauge, options/retail/social panels
- **Status**: All phases complete

### v5.80 Cron Guard Hardening - COMPLETED
- **Guard Script**: `scripts/cron_guard.sh` — shared library for all Hermes cron jobs
  - 4-layer defense: load gating (max load 5), flock overlap prevention, ulimit memory cap, timeout watchdog
  - TEMPFAIL exit code 75 for transient failures (Hermes-aware)
  - All 10 Hermes scripts + 11 Makefile targets updated
- **Config**: `src/cron_compat.py` — Added `CRON_EXPECTED_DURATIONS` and `CRON_GUARD_CONFIG`
- **Root Cause**: May 16 incident — autonomous agent cron job dispatching LLM-powered sessions every 15 min, sessions 10+ min, overlapping → loadavg 48.75
- **Status**: Complete

### v5.81 Cron Audit & Stagger - COMPLETED (2026-05-17)
- **Audit**: Verified wiki work item `work/2026-05-16-cron-job-audit/` — 5 items claimed DONE but not applied
- **Removed**: `portfolio-lab-cron.sh` (redundant monolithic script)
- **Paused**: `portfolio-lab-build` (no web server consumers)
- **Staggered**: 8 active jobs spread across unique minute slots — zero collisions
- **Reduced**: dashboard */10→hourly, autonomous */15→every 2h, eval/research/wiki-sync /30→every 2h
- **Fixed**: Added `portfolio-lab-attribution` to CRON_TARGETS + CRON_EXPECTED_DURATIONS + crontab
- **Repo**: All 9 Hermes scripts + guard copied to `scripts/cron/` with full `scripts/cron/README.md`
- **Wiki**: Compound page `projects/portfolio-lab/compound/cron-job-audit-staggered-schedule`

## Recent Implementation Updates (2026-05-15)

### v4.50 VIX Term Structure Overlay - Phase 3 COMPLETED
- **Signal Generator**: `src/signals/vix_term_structure.py` (580 lines) — VIX/VIX3M/VIX6M slope analysis
- **Tactical Overlay**: `src/strategy/vix_overlay.py` (540 lines) — regime-based allocation shifts
  - Allocation shifts: SPY±10%, GLD±5%, TLT±5% based on term structure slope
  - Constraints: max 5% daily shift, 5-day holding period, VPIN freeze
  - VIX spike protection: >50% single-day spike disables overlay for 24h
- **Integration**: 15% weight in ensemble voter (planned, NOT in REGIME_WEIGHTS), SmartRebalanceGate coordination
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
- **6424 safe** passed (0 failures, 10 skipped)
- **6561 total** collected when `PORTFOLIO_LAB_ENABLE_ML=1 --include-heavy` (6549 passed, 12 failed)
- ~4500+ passing, pre-existing failures in yield curve and a few other suites
- 190 test files + 4 new dashboard components covering signals, strategy, backtest, dashboard, broker, agents, data, research, chat, execution
- **New (v9.18)**: `test_crypto_institutional.py` (58 tests) — TokenizedTreasuryStrategy + CryptoRiskManager + CLI
- **New (v9.16)**: `test_odte_executor.py` (27 tests), `test_health_tracker.py` (20 tests), `test_health.py` (13 tests) — 60 tests for previously untested modules
- **New (v9.21)**: `test_analytics_calculator.py` (52 tests), expanded `test_rebalancing_backtest.py` (+18 tests) — 70 tests for undertested modules
- **Safe**: `make test` or `bash scripts/run-tests-safe` (ML disabled, 3GB ulimit cap)
- **ML**: `make test-ml` or `PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/ --include-heavy`
- **Test pollution fix (v9.17)**: Fixed 3 polluting test files that broke 205+ downstream tests:
  - `test_sentiment_client.py`: Replaced `@patch` decorators with `patch.object()` context managers + autouse fixture for SDK stubs when openai/anthropic not installed
  - `test_international_momentum.py`: Restored pandas/yfinance in `sys.modules` after mock usage (was only restoring numpy)
  - `test_sentiment_analyzer.py`: Made 3 tests resilient to mock eviction (analyzer may be None without API keys)
  - `test_base_agent.py`: Added `@pytest.mark.skipif` for 2 torch-dependent tests in safe mode
  - `test_rebalancing_backtest.py`: Evict `src.rebalancing.*` from sys.modules after mock restore
  - `test_generator.py`: Skip with clear message when MagicMock pollution detected
  - `test_cron_compat.py`, `test_daily_brief.py`, `test_vix_overlay_backtest.py`, `test_macro_regime_synthesis.py`, `test_yield_dashboard.py`: Fixed stale assertions and environment-dependent checks

### TypeScript (tests/ts/)
- **191 tests** across 10 files (DSR 24, duration-signals 35, purged-cv 21, car25 23, stress-validation 15, sector-attribution 19, sector-momentum 15, leveraged-treasury 7, overlay-panels 24)
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
make test            # safe test suite (ML disabled, 3GB memory cap, 6080p/251f tests, see test-isolation for pollution bypass)
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
