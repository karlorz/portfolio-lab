---
title: Trending Portfolio Strategies Q2 2026 - Plan
id: deep-research-trending-2026-Q2
status: in-progress
created: 2026-05-12
---

# Implementation Plan: Trending Strategies Q2 2026

## Research Deliverables [COMPLETED]
- [x] AQR Tactical Asset Allocation Research
- [x] ML Signal Generation Research
- [x] Inflation Hedging Research
- [x] Risk Management Research
- [x] Risk Parity & ARP Research
- [x] Synthesis document: compound-synthesis.md

## Implementation Priorities

### P1: Time-Series Factor Momentum (TSFM) Engine v2.15
**Status:** IN PROGRESS
**File:** src/strategy/factor_rotation.py enhancement
**Rationale:** Builds on existing v2.9 ML features; AQR-validated with 18%+ returns

**Changes Required:**
1. Add `calculate_ts_factor_momentum()` method
2. Implement volatility-scaled position sizing
3. Add factor momentum overlay to existing rotation engine
4. Update CLI with --tsfm flag

### P2: kNN Macro Regime Detector v2.16
**Status:** PENDING
**File:** src/research/regime_classifier.py enhancement OR new module
**Rationale:** Generali Investments validated +7.2 bps/month; 83% accuracy

**Changes Required:**
1. Integrate FRED-MD data source
2. Implement kNN classifier (k=5-10)
3. Add equity/bond allocation signal output
4. Connect to evaluator.py as regime input

### P3: Dynamic Real Asset Allocator v2.17
**Status:** PENDING
**File:** src/strategy/inflation_risk_parity.py enhancement
**Rationale:** PGIM "leaning in" strategy; 10.6% vs 10.1% returns, lower vol

**Changes Required:**
1. Add inflation regime detection (already exists)
2. Implement "leaning in" allocation logic
3. Dynamic 10-20% -> 40-60% allocation range
4. Backtest integration with historical inflation data

### P4: ML Signal Ensemble v2.18
**Status:** PENDING
**File:** New module src/strategy/ml_ensemble.py
**Rationale:** Combines multiple signals with confidence weighting

**Changes Required:**
1. Create ensemble architecture
2. Integrate existing signals (factor rotation, ARP, inflation RP)
3. Weighted voting/confidence scoring
4. Output final allocation recommendation

## Current Status
- Health Monitor: HEALTHY
- Circuit Breaker: GREEN (0.08% drawdown)
- Portfolio Value: $99,985.91
- Data Freshness: OK
- Wiki Sync: OK

## Work Item Log
- 2026-05-12 06:15: Work item created, research initiated
- 2026-05-12 06:30: Research completed (3 subagents parallel)
- 2026-05-12 06:45: Synthesis document written
- 2026-05-12 06:50: Implementation planning complete
- 2026-05-12 06:55: P1 TSFM implementation started
