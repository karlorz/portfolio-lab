# v2.42 Tail Risk Hedging - Deep Research Synthesis

**Date:** 2026-05-13
**Status:** **COMPLETE** - Phase 1 implemented (v2.42a)
**Session IDs:** c506957b4377 (tail hedge), 495a2c4c60be (vol targeting), 85f8c74831bc (ESG)

## Research Summary

### 1. Tail Risk Hedging Strategies (Session: c506957b4377)

**Key Findings:**
- **Optimal Strategy:** Hybrid approach combining protective puts + VIX calls
- **Allocation:** 0.5-2% of portfolio annually
- **Entry Timing:** Buy when VIX < 15-22 (low implied vol)
- **Strike Selection:** 10-30 delta OTM for cost efficiency

**Implementation v2.42a - COMPLETE:**
- [x] Protective put calculator (strike, expiration, delta selection)
- [x] VIX call overlay sizing with entry/exit thresholds
- [x] Cost/benefit analytics with 2% max portfolio budget
- [ ] Hybrid hedge optimizer (puts + VIX) - Phase 2

**Entry Conditions:**
- VIX < 15: Full size (5 contracts, ~1% premium)
- VIX 15-22: Scale linearly (3-5 contracts, 0.5-1% premium)
- VIX > 22: No entry (vol too expensive)

**Exit Conditions:**
- VIX > 35: Take profit (crisis level, monetize insurance)
- 30 DTE: Roll position (avoid gamma risk)
- VIX > 40: Emergency exit (tail event in progress)

**CLI:**
```bash
python -m src.risk.tail_hedge_calculator analyze --vix 18.5
python -m src.risk.tail_hedge_calculator vix-signal --current-vix 15.2
python -m src.risk.tail_hedge_calculator cost --underlying SPY --strike-pct 0.95
```

### 2. Volatility Targeting (Session: 495a2c4c60be)

**Key Findings:**
- **Target:** 8-10% annualized volatility
- **Mechanism:** Scale exposure inversely to realized vol
- **Performance:** Sharpe ~1.42 vs static approaches
- **Drawdown:** Max -27.7% vs -40%+ for 60/40

**Integration with Risk Parity:**
- S&P Risk Parity indices target 8% vol
- Dynamic versions use ML/LSTM for vol forecasting
- Rebalancing frequency: Weekly to monthly

### 3. ESG Integration (Session: 85f8c74831bc)

**Status:** Deferred - existing `src/analytics/esg_integration.py` provides WACI calculator

## Implementation Status

| Component | Status | File | Lines |
|-----------|--------|------|-------|
| Protective Put Calculator | **COMPLETE** | `src/risk/tail_hedge_calculator.py` | 588 |
| VIX Call Overlay | **COMPLETE** | `src/risk/tail_hedge_calculator.py` | 588 |
| Hybrid Optimizer | Phase 2 | TBD | — |
| Vol Targeting | Partial | `src/signals/integrator.py` | — |
| ESG Integration | Existing | `src/analytics/esg_integration.py` | 540 |

## References
- CBOE VXTH Index methodology
- S&P Risk Parity Index 8% Target Vol methodology
- CFA Institute ESG Integration 2025 report
- Commit: `796cecd`
