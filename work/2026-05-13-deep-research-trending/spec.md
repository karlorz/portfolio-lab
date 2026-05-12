# Deep Research: Portfolio Trending & Market Regime Analysis

**Type:** deep_research  
**Created:** 2026-05-13  
**Status:** synthesis-complete → implementing-p1-tsmom  

## Focus Areas
1. **Tactical Asset Allocation (TAA)** - Institutional approaches to trend following [COMPLETED]
2. **ML Signal Generation** - HMM-LSTM regime detection for Risk Agent [SYNTHESIZED]
3. **Inflation/Rate Regime Hedging** - Fed policy impacts on all-season portfolios [SYNTHESIZED]
4. **Alternative Data Integration** - Satellite, sentiment, flow data signals [DEFERRED]
5. **Risk Parity Enhancements** - Beyond Dalio's All Weather [DEFERRED]

## Deliverables
- [x] Research synthesis with institutional citations - `compound-synthesis.md`
- [x] Implementation opportunities ranked P1-P4 - See synthesis doc
- [ ] Code implementation of highest-priority strategy - TSMOM Overlay (P1)
- [ ] Wiki documentation with citations - Pending implementation

## Implementation Roadmap

### P1: Time-Series Momentum Overlay (WEEK 1)
**Source:** AQR Moskowitz et al. (2012, 2017, 2024)  
**Target:** Sharpe 0.79 → 0.88  
**Status:** READY FOR IMPLEMENTATION  

**Deliverables:**
- `src/signals/tsmom_overlay.py` - 12-month momentum with 1-month skip
- `src/signals/tsmom_backtest.py` - Standalone backtest validator
- Integration with `src/signals/integrator.py` (v2.24)
- Test: `python3 -m src.signals.tsmom_overlay backtest --portfolio 46/38/16`

### P2: HMM-LSTM Regime Detection (WEEK 2)
**Source:** arXiv 2407.19858, SSRN 5366835  
**Target:** Sharpe 0.88 → 0.95  
**Status:** PLANNED

### P3: Fed Policy Overlay (WEEK 2-3)
**Source:** Fed Research, Goldman, CME  
**Target:** Sharpe 0.95 → 0.98  
**Status:** PLANNED

## Current Market Regime (May 2026)
- **Fed Funds:** 3.50%-3.75% (effective ~3.64%)
- **CPI:** 3.8% YoY (up from 3.3%, energy-driven)
- **Real Rates:** Short-term ~0.97%, 10Y TIPS ~1.9-2.0%
- **Classification:** Hold/neutral with restrictive tilt
- **Tactical Implication:** Balanced gold/Treasury approach; elevated real yields support duration selective Treasuries

## Research Artifacts
- **Synthesis:** `compound-synthesis.md` (Full institutional citations, implementation specs)
- **References:** AQR, arXiv 2407.19858, SSRN 5366835, Fed Research

## Next Action
Begin P1 TSMOM implementation immediately.
