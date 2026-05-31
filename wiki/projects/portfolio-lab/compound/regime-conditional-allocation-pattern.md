---
title: "Regime-Conditional Allocation Pattern"
created: 2026-05-28
updated: 2026-05-28
tags:
  - concept
  - regime
  - allocation
  - pattern
---

# Regime-Conditional Allocation Pattern

## Pattern Summary

Allocate portfolio weights dynamically based on detected market regime, rather than using static weights. This is the core architectural pattern for improving risk-adjusted returns beyond static 46/38/16.

## Current Implementation

The project already implements regime-conditional behavior in three places:
1. **Regime-conditional ensemble weights** (`REGIME_CONDITIONAL_WEIGHTS` in ensemble_voter.py) -- per-signal multipliers
2. **Regime-conditional vol targeting** -- 5%/7%/9%/11%/10% targets per regime
3. **Regime-conditional signal gating** (`regime_gate.py`) -- ON/OFF per signal per regime

## Missing: Regime-Conditional Base Allocation

The 46/38/16 SPY/GLD/TLT split is static across all regimes. Research suggests:

```
Regime          SPY    GLD    TLT    Rationale
CRISIS          30%    45%    25%    Maximum defense
HIGH_VOL        38%    42%    20%    Elevated defense
NORMAL          46%    38%    16%    Current champion
LOW_VOL         52%    30%    18%    Capture equity premium
RECOVERY        48%    35%    17%    Gradual normalization
```

## Implementation Pattern

```python
# Extend vol_parity_allocator.py or create new module
REGIME_ALLOCATION = {
    'crisis':    {'SPY': 0.30, 'GLD': 0.45, 'TLT': 0.25},
    'high_vol':  {'SPY': 0.38, 'GLD': 0.42, 'TLT': 0.20},
    'normal':    {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},  # champion
    'low_vol':   {'SPY': 0.52, 'GLD': 0.30, 'TLT': 0.18},
    'recovery':  {'SPY': 0.48, 'GLD': 0.35, 'TLT': 0.17},
}
```

## Key Insight

The regime classifier (two-stage k-means) is mature and validated. The gap is not in detection but in conditional response. This pattern requires:
- Existing regime detection (DONE)
- Conditional allocation weights (TO DO)
- Smooth transitions between regimes (TO DO -- hysteresis)
- Backtest validation (TO DO)

## Expected Impact

+0.01-0.02 Sharpe from regime-conditional gold allocation alone. Combined with regime-conditional SPY/TLT, potential +0.02-0.04 Sharpe total.

## Risks

- Regime misclassification risk (mitigated by existing ARI=1.0 stability)
- Overfitting to historical regime sequences (mitigated by walk-forward validation)
- Transition costs from regime changes (mitigated by existing transaction cost model)
