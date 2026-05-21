# Top 5 Ranked Recommendations

| Rank | Recommendation | Impact | Effort | I/E Ratio | Key Files |
|------|---------------|--------|--------|-----------|-----------|
| 1 | Backtest tests (3 files, 1319 lines) | High | Medium | 1.2 | cross_asset_regime_arb_backtest.py, cross_asset_rv_backtest.py, international_momentum_backtest.py |
| 2 | Centralize 46/38/16 config (30+ files) | Medium | Low | 2.0 | ~50 edits across agents/, strategy/, backtest/, rebalancing/, monitor/, signals/ |
| 3 | Dead REGIME_WEIGHTS in integrator.py | Low | Low | 2.5 | src/signals/integrator.py lines 72-118 |
| 4 | Continuous VIX term structure mapping | Medium | Low | 1.8 | src/signals/vix_term_structure.py `get_allocation_shifts()` |
| 5 | Tests for 8 uncovered non-ML modules | High | High | 0.8 | rebalancing/backtest.py, 3x odte_yield_*, pipeline/integration.py, etc. |

## Summary

- **6246 safe tests passing, 0 failures** -- codebase is in strong shape
- **Zero** `except Exception: pass` instances (already hardened)
- **4,693 total uncovered non-ML lines** across backtest tests (1,319) + module tests (3,374)
- **30+ files** still hardcode 46/38/16 instead of importing from src/paths.py
- **2 copies** of REGIME_WEIGHTS with incompatible schemas (one dead in integrator.py)
- **1 signal** (vix_term_structure.py) still uses discrete allocation bins instead of continuous passthrough
- **245 cov matrix values** hardcoded in regime_optimizer.py with no recomputation mechanism
