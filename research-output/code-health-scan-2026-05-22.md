# Code Health + Vault Health Scan: Ranked Work Items

## Summary

**Date**: 2026-05-22
**Scope**: Full scan of 220 source files, 191 test files, 25 backtest files, signal weights, architectural debt

---

## TL;DR

- **4 non-ML modules remain untested** (predictive_model.py 541L, regime_classifier.py 484L, bond_momentum_backtest.py 404L, macro_regime_cli.py 297L) — down from 14 flagged after filtering out false positives (factor_rotation, odte_yield_*, etc. all have tests)
- **6 backtest files have embedded logic** — 3 overlay backtests (collar, bond_duration, crypto) and 3 signal backtests (cross_asset_rv, international_momentum, alternative_data) do NOT import the actual production code. Only vixy_hedge, multi_speed_momentum, and cross_asset_regime_arb properly import live code.
- **Signal weights are well-tuned** — MULTI_SPEED_MOM reduced to 10% (from 50% peak), net-negative contribution contained. ALT_DATA (26%) and INTL_MOM (21%) are the alpha drivers.
- **27 files still use manual path construction** instead of from src.paths import — 12 strategy, 9 signal, 6 monitor files.
- **1 bare except:** remains in src/strategy/duration_overlay.py:288
- **factor_rotation_backtest.py deleted** — only .pyc remains, documented in CLAUDE.md but removed in v9.25 purge.

---

## Top 5 Recommended Work Items

| Rank | Title | Category | Files | Impact | Effort |
|------|-------|----------|-------|--------|--------|
| 1 | Migrate 3 overlay backtests to import live strategy code | backtest-missing | src/backtest/collar_overlay_backtest.py, bond_duration_backtest.py, crypto_allocation_backtest.py | CRITICAL - backtest results don't validate production code | HIGH (4-6h) |
| 2 | Test predictive_model.py (541L VAR(1) inference) | test-gap | src/agents/predictive_model.py | HIGH - production inference path | MEDIUM (2-3h) |
| 3 | Migrate 3 signal backtests to import live signal code | backtest-missing | src/backtest/cross_asset_rv_backtest.py, international_momentum_backtest.py, alternative_data_backtest.py | HIGH - same divergence risk as overlay | MEDIUM (3-4h) |
| 4 | Test regime_classifier.py (484L regime state) | test-gap | src/research/regime_classifier.py | MEDIUM - regime affects research conclusions | LOW (1-2h) |
| 5 | Migrate 27 files to src/paths.py imports | arch-debt | 12 strategy + 9 signal + 6 monitor files | MEDIUM - portability, refactoring velocity | MEDIUM (2-3h) |

---

## Key Findings by Category

### Test Gaps
**4 non-ML modules with NO test coverage:**
- `src/agents/predictive_model.py` (541L) — VAR(1) price prediction, used by FPILOT inference planning
- `src/research/regime_classifier.py` (484L) — market regime classification
- `src/research/bond_momentum_backtest.py` (404L) — research backtest
- `src/regime/macro_regime_cli.py` (297L) — CLI wrapper

### Backtest Gaps
**6 backtest files with embedded logic (not importing production code):**
- Overlay: collar_overlay_backtest.py, bond_duration_backtest.py, crypto_allocation_backtest.py
- Signal: cross_asset_rv_backtest.py, international_momentum_backtest.py, alternative_data_backtest.py
- Well-integrated: vixy_hedge_backtest.py, multi_speed_momentum_backtest.py, cross_asset_regime_arb_backtest.py

**9 backtest files not importing shared metrics.py:**
- alt_data_walkforward_stress, alternative_data_backfill, behavioral_sentiment_backtest, car25, dbc_weight_sweep, ensemble_backtest, real_data_backtest, run_actual_ubt_validation, stacking_ensemble_backtest

### Signal Quality
- MSM at 10% is appropriate (confirmed net-negative in v9.24 backtest)
- ALT_DATA (26%) and INTL_MOM (21%) are the alpha drivers with positive Sharpe contribution
- CROSS_ASSET_RV at 34% in CRISIS needs defensive justification
- UNIFIED_OVERLAY (17-21%) has no individual backtest validation

### Architectural Debt
- 27 files with manual path construction (12 strategy, 9 signal, 6 monitor)
- 1 bare `except:` remaining in duration_overlay.py:288
- Duplicated BacktestConfig and DailyPrices dataclasses across 10+ backtest files
- factor_rotation_backtest.py deleted (only .pyc), documented in CLAUDE.md but gone

---

## Sources Queried
- All 220 Python source files in src/
- All 191 test files in tests/
- All 25 backtest files in src/backtest/
- Full grep of src/ for TODO/FIXME/HACK/bare-except/hardcoded-paths
- src/strategy/ensemble_voter.py (current signal weights)
- CLAUDE.md (version history, research findings)
- Vault: /root/wiki

## Output
- Vault report: `/root/wiki/projects/portfolio-lab/compound/code-health-scan-2026-05-22.md`
- Log update: `/root/wiki/projects/portfolio-lab/log.md`
- Raw data: `/root/wiki/projects/portfolio-lab/raw/code-health-scan-2026-05-22.sh`
