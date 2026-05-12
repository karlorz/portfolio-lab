---
kind: implementation
version: 2.20.1
status: ready
priority: critical
dependencies:
  - v219-dashboard-yield-curve
  - v218-dynamic-duration
research_session: 114d636884a8
---

# v2.20.1: Wasserstein HMM Regime Detector + CTA Trend Overlay

## Objective
Implement institutional-grade regime detection and crisis alpha overlay based on Q2 2026 research synthesis. This is the highest-impact upgrade based on deep research findings:
- Wasserstein HMM: Sharpe 2.18, Max DD -5.43%, turnover 0.0079
- CTA Trend Overlay: +27% in 2022 when 60/40 failed

## Research Basis
From `/root/projects/portfolio-lab/work/2026-05-12-v220-ml-regime-strategies/compound-synthesis.md`:
- arXiv 2603.04441v1: Wasserstein HMM with template tracking prevents label switching
- CME/MS Research: Managed futures -0.54 correlation in declining quarters
- SG Trend Index 2022: +27.3% performance vs SPY -19%

## Implementation Tasks

### 1. Wasserstein HMM Regime Detector (src/strategy/regime_hmm.py)
- [ ] GaussianHMM implementation with 2-4 hidden states
- [ ] Wasserstein distance template tracking for label stability
- [ ] Feature inputs: VIX changes, 2s10s spread, SPY/GLD momentum
- [ ] Regime classification: Risk-On (SPY-heavy), Risk-Off (TLT/GLD), Crisis (BIL/SHY)
- [ ] Transition probability matrix computation
- [ ] Regime persistence statistics (avg duration per state)
- [ ] CLI: `regime status`, `regime history`, `regime predict`

### 2. CTA Trend Overlay Module (src/strategy/cta_overlay.py)
- [ ] Multi-asset trend signals: SPY, GLD, TLT, IEF, SHY, BIL, DBC, UUP
- [ ] Time-series momentum: 1-month, 3-month, 12-month lookbacks
- [ ] Moving average crossovers: 50/200-day
- [ ] Volatility-adjusted position sizing
- [ ] Aggregate CTA score: -1 (strong short) to +1 (strong long)
- [ ] Crisis alpha trigger: When equity trend negative + vol spike
- [ ] Recommended overlay allocation: 5-20% of portfolio
- [ ] CLI: `cta signals`, `cta score`, `cta recommend`

### 3. Ensemble Signal Voting (src/strategy/ensemble_voter.py)
- [ ] Signal collection from multiple sources:
  - TSFM v2.15 factor momentum (existing)
  - HMM regime detector (new)
  - CTA trend overlay (new)
  - Duration/yield curve regime (v2.17-2.18)
  - Circuit breaker status (v2.14)
- [ ] Soft voting with confidence weighting
- [ ] Regime-dependent signal weights:
  - Normal: TSFM 50%, CTA 30%, Duration 20%
  - High vol: HMM 40%, CTA 40%, TSFM 20%
  - Crisis: CTA 50%, HMM 30%, Circuit breaker 20%
- [ ] Consensus threshold: 2/3 signals agree for action
- [ ] Output: Final allocation recommendation with confidence
- [ ] CLI: `ensemble vote`, `ensemble recommend`, `ensemble explain`

### 4. Integration with Existing Systems
- [ ] HMM regime feeds into factor_rotation.py position sizing
- [ ] CTA overlay integrates with LiveDashboard allocation panel
- [ ] Ensemble voter updates dashboard signal strength display
- [ ] Regime transitions trigger alerts via existing alert system

### 5. Testing & Validation
- [ ] Backtest HMM on 2005-2026 data
- [ ] Validate CTA signals vs SG Trend Index correlation
- [ ] Test ensemble consensus accuracy in known crisis periods (2008, 2020, 2022)
- [ ] Compare turnover vs existing strategies

## Files to Create/Modify
- Create: `src/strategy/regime_hmm.py`
- Create: `src/strategy/cta_overlay.py`
- Create: `src/strategy/ensemble_voter.py`
- Modify: `src/strategy/factor_rotation.py` (integrate HMM regime)
- Modify: `src/components/LiveDashboard.tsx` (ensemble signal display)
- Modify: `src/strategy/circuit_breaker.py` (feed into ensemble)

## Acceptance Criteria
- [ ] HMM achieves >80% regime classification accuracy on historical crises
- [ ] Wasserstein template tracking prevents label switching (verified on rolling window)
- [ ] CTA overlay shows negative correlation to equities in bear markets
- [ ] Ensemble consensus improves Sharpe vs any single signal
- [ ] Turnover remains <0.01 daily (HMM advantage)
- [ ] All components have CLI interfaces with --help
- [ ] Integration tests pass with existing dashboard

## Performance Targets
| Metric | Target | Source |
|--------|--------|--------|
| HMM Sharpe | >1.5 | Research: 2.18 |
| HMM Max DD | <-10% | Research: -5.43% |
| CTA Crisis Alpha | >15% | Research: +27% in 2022 |
| Ensemble Sharpe | >1.2 | Combined benefit |
| Daily Turnover | <0.01 | Wasserstein advantage |

## ETA
8-12 hours

## References
- `/root/projects/portfolio-lab/work/2026-05-12-v220-ml-regime-strategies/compound-synthesis.md`
- arXiv 2603.04441v1: Wasserstein HMM
- CME Group: Managed Futures as Crisis Risk Offset
- SG Trend Index 2022 performance data
