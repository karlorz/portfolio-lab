---
kind: implementation
version: 2.58
status: completed
progress: 100
priority: critical
dependencies:
  - v2.56-multi-speed-momentum
  - v2.57-macro-momentum
---

# v2.58: Ensemble Signal Voter

## Objective
Multi-source signal aggregation system with regime-dependent weighting for portfolio allocation decisions.

## Implementation
**File**: `src/strategy/ensemble_voter.py` (~450 lines)

### Components

1. **Regime Detection** (simple heuristic, can integrate HMM later)
   - Crisis: Vol > 30% or DD > 10%
   - High vol: Vol > 20% or DD > 5%
   - Recovery: Recent DD followed by positive momentum
   - Normal: Otherwise

2. **Signal Sources** (configured)
   - MULTI_SPEED_MOM (v2.56): Fast/medium/slow momentum ensemble
   - MACRO_MOMENTUM (v2.57): Business cycle, monetary policy, trade, sentiment
   - CTA_TREND (existing): Multi-timeframe trend following
   - HMM_REGIME (v2.53): Hidden Markov Model states
   - CIRCUIT_BREAKER (v2.14): Risk limits and controls

3. **Regime-Dependent Weights**
   - Normal: MultiSpeed 25%, TSFM 40%, CTA 20%, Macro 10%, Duration 5%
   - High vol: HMM 35%, CTA 30%, MultiSpeed 20%, Macro 10%, Circuit 5%
   - Crisis: Circuit 35%, CTA 35%, HMM 20%, Macro 10%
   - Recovery: MultiSpeed 30%, HMM 25%, CTA 20%, TSFM 15%, Macro 10%

4. **Consensus Logic**
   - Weighted average of active signals
   - Agreement ratio: % of signals agreeing with consensus
   - Asset-specific bias extraction (equity, duration, gold)
   - Action classification: increase_equity, decrease_equity, neutral, risk_off

### CLI Interface
```bash
python -m src.strategy.ensemble_voter vote        # Current vote summary
python -m src.strategy.ensemble_voter recommend   # Allocation recommendation  
python -m src.strategy.ensemble_voter explain     # Detailed reasoning
```

### Example Output
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
```

### Integration Points
- Consumes signals from multi_speed_momentum.get_signal_for_ticker()
- Consumes signals from macro_momentum.MacroMomentumEngine
- Stores votes in SQLite (data/ensemble_signals.db)
- Outputs allocation shifts compatible with base 46/38/16 portfolio

## Current Status
**COMPLETED** - Working with 2 signal sources:
- Multi-speed momentum (v2.56) 
- Macro momentum (v2.57)

Consensus +0.333 suggests positive equity/gold bias, negative duration.

## Performance Target
- Sharpe improvement: +0.05 to +0.10 from baseline 0.96
- Crisis detection: Early warning via regime shift to high_vol/crisis
- Signal diversification: Reduce reliance on single methodology

## Next Steps
1. Integrate HMM regime detector (v2.53) as signal source
2. Add CTA trend overlay signals (existing src/strategy/cta_overlay.py)
3. Add TSFM factor momentum signals (v2.15)
4. Backtest full ensemble vs individual components
5. Dashboard integration for real-time vote display

## References
- `/root/projects/portfolio-lab/work/2026-05-13-autonomous-trending-deep-research/compound-synthesis.md`
- Man AHL multi-speed momentum research (Sept 2025)
- Brooks et al. (2017) "A Half Century of Macro Momentum"
