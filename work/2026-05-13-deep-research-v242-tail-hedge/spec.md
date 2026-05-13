---
skill_compatibility: 2
version: "2.0"
kind: feature
slug: v242-tail-hedge
status: completed
progress: 100
completed: 2026-05-13
started: 2026-05-13
---

# v2.42 Tail Risk Hedging - Work Item

**Date:** 2026-05-13  
**Status:** COMPLETE - Phase 1 implemented (v2.42)  
**Session IDs:** c506957b4377 (tail hedge), 495a2c4c60be (vol targeting), 85f8c74831bc (ESG)

## Research Summary

### 1. Tail Risk Hedging Strategies (Session: c506957b4377) ✓

**Key Findings:**
- **Optimal Strategy:** Hybrid approach combining protective puts + VIX calls
- **Allocation:** 0.5-2% of portfolio annually
- **Entry Timing:** Buy when VIX < 15-22 (low implied vol)
- **Strike Selection:** 10-30 delta OTM for cost efficiency

**Implementation v2.42 - COMPLETE:**
- [x] Protective put calculator (strike, expiration, delta selection)
- [x] VIX call overlay sizing with entry/exit thresholds
- [x] Cost/benefit analytics with 2% max portfolio budget
- [ ] Hybrid hedge optimizer (puts + VIX) - Phase 2 (deferred)

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

**Verification Results:**
- VIX 14.2 (18th percentile): ENTER 5 contracts, Strike 22.0, Premium $881 (0.88%), Expected Payout $9,000
- VIX 18.5 (44th percentile): ENTER 4 contracts, Premium $919 (0.92%), Expected Payout $7,200
- VIX 22.5 (62nd percentile): NO ENTRY (vol too expensive)

### 2. Volatility Targeting (Session: 495a2c4c60be) - SYNTHESIZED

**Key Findings:**
- **Target:** 8-10% annualized volatility
- **Mechanism:** Scale exposure inversely to realized vol
- **Performance:** Sharpe ~1.42 vs static approaches
- **Drawdown:** Max -27.7% vs -40%+ for 60/40

**Status:** Research synthesized, integration planned for v2.60+

### 3. ESG/Sustainable Investing (Session: 85f8c74831bc) - DEFERRED

**Status:** Research synthesized, implementation deferred due to data access requirements

## Files Created
- `src/risk/tail_hedge_calculator.py` (588 lines)

## Git Commit
`796cecd` - "feat: v2.42 Tail Risk Hedge Calculator - Phase 1"
