# v2.20.1 Implementation Plan

## Research Source
`/root/projects/portfolio-lab/work/2026-05-12-v220-ml-regime-strategies/compound-synthesis.md`

## Phase 1: Wasserstein HMM Regime Detector (IN PROGRESS)
- [x] Create `src/strategy/regime_hmm.py` with GaussianHMM base
- [x] Implement Wasserstein distance template tracking
- [x] Add feature extraction (VIX, 2s10s, momentum)
- [x] Build regime classifier with 2-4 states
- [ ] CLI interface: status, history, predict
- [ ] Test on historical data

## Phase 2: CTA Trend Overlay (EXISTING - v2.10)
**Note:** CTA overlay already exists at `src/strategy/cta_overlay.py` (v2.10)
- [x] Multi-asset trend signals implemented
- [x] Time-series momentum (20d, 60d, 120d)
- [x] Volatility targeting position sizing
- [x] Crisis alpha detection
- [ ] Enhance with 1M/3M/12M timeframes from research

## Phase 3: Ensemble Signal Voting (PENDING)
1. Create `src/strategy/ensemble_voter.py`
2. Collect signals from all sources (TSFM, HMM, CTA, Duration, Circuit Breaker)
3. Implement soft voting with regime-dependent weights
4. Consensus threshold logic (2/3 signals agree)
5. CLI interface: vote, recommend, explain

## Phase 4: Integration (PENDING)
1. Integrate HMM with factor_rotation.py position sizing
2. Connect ensemble voter to LiveDashboard
3. Add regime transition alerts
4. Final testing

## Current Status
- [x] Research synthesis: COMPLETED (11762 bytes)
- [x] Work item created with spec.md, plan.md
- [x] Phase 1: IN PROGRESS (creating HMM detector)
- [ ] Phase 2: SKIPPED (already implemented)
- [ ] Phase 3: NOT STARTED
- [ ] Phase 4: NOT STARTED

## Blockers
None - ready to proceed
