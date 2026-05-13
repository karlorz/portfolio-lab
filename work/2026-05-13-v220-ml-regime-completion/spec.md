---
kind: implementation
version: 2.20.1
status: completed
progress: 100
completed: 2026-05-13
priority: critical
dependencies:
  - v256-multi-speed-momentum
  - v257-macro-momentum
  - v258-ensemble-voter
---

# v2.20.1: Wasserstein HMM + Ensemble System - COMPLETE

## Phase 1: Wasserstein HMM Regime Detector ✓
**Status**: COMPLETE (pre-existing, verified working)
**File**: `src/strategy/regime_hmm.py` (23KB)

Already implemented with:
- GaussianHMM with 2-4 hidden states
- Wasserstein distance template tracking
- Feature inputs: VIX changes, 2s10s spread, SPY/GLD momentum
- CLI: `status`, `history`, `predict`, `train`

## Phase 2: Multi-Speed Momentum Ensemble ✓
**Status**: COMPLETE (new - v2.56)
**File**: `src/signals/multi_speed_momentum.py` (30KB)

Man AHL-style speed diversification:
- Fast: 63-day lookback (crisis alpha)
- Medium: 126-day lookback (balanced)
- Slow: 252-day lookback (trend persistence)
- Equal risk-weighting across speeds

CLI:
```bash
python -m src.signals.multi_speed_momentum compute --ticker SPY
python -m src.signals.multi_speed_momentum backtest --portfolio 46/38/16
```

## Phase 3: Macro Momentum Signals ✓
**Status**: COMPLETE (new - v2.57)
**File**: `src/signals/macro_momentum.py` (25KB)

Brooks et al. (2017) implementation:
- Business Cycle: 3m/6m momentum signals
- International Trade: GLD vs SPY relative strength
- Monetary Policy: Yield curve (TLT/SHY spread)
- Risk Sentiment: 12m equity momentum + gold fear

CLI:
```bash
python -m src.signals.macro_momentum signal
python -m src.signals.macro_momentum backtest
```

## Phase 4: Ensemble Signal Voter ✓
**Status**: COMPLETE (new - v2.58)
**File**: `src/strategy/ensemble_voter.py` (23KB)

Multi-source aggregation with regime-dependent weighting:
- Normal regime: MultiSpeed 25%, Macro 10%
- High vol regime: HMM 35%, MultiSpeed 20%
- Crisis regime: Circuit 35%, CTA 35%
- Recovery: MultiSpeed 30%, HMM 25%

CLI:
```bash
python -m src.strategy.ensemble_voter vote
python -m src.strategy.ensemble_voter recommend --portfolio 46/38/16
python -m src.strategy.ensemble_voter explain
```

## Current Ensemble Output (2026-05-13)
```
Regime: NORMAL (confidence: 79.1%)
Sources: 2
Consensus: +0.333
Agreement: 100.0%

Asset Biases:
  Equity (SPY):   +1.000
  Duration (TLT): -0.752
  Gold (GLD):     +1.000

Recommended Action: INCREASE_EQUITY
Confidence: 100.0%

Allocation (46/38/16 base → recommended):
  SPY: 46.0% → 49.8% (shift: +3.8%)
  GLD: 38.0% → 42.7% (shift: +4.7%)
  TLT: 16.0% → 7.5%  (shift: -8.5%)
```

## Implementation Complete
All major components of v2.20.1 are now functional:
1. ✅ Wasserstein HMM regime detection (pre-existing)
2. ✅ Multi-speed momentum ensemble (v2.56)
3. ✅ Macro momentum signals (v2.57)
4. ✅ Ensemble voter with regime-dependent weights (v2.58)

## Files Created
- `src/signals/multi_speed_momentum.py` - 30KB, 807 lines
- `src/signals/macro_momentum.py` - 25KB, 682 lines
- `src/strategy/ensemble_voter.py` - 23KB, 641 lines
- `work/2026-05-13-v258-ensemble-voter/spec.md`

## Next Steps
1. Backtest full ensemble vs individual components
2. Add CTA trend overlay to ensemble (existing v2.10)
3. Add TSFM factor momentum (existing v2.15)
4. Integrate with LiveDashboard allocation panel
5. Add circuit breaker feeds to crisis regime detection
