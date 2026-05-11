# Portfolio-Lab 2026 Q3 Roadmap

**Date**: 2026-05-12
**Planning Mode**: Plan Only - No Execution
**Target Quarter**: Q3 2026 (July - September)

## Executive Summary

Portfolio-Lab v2.x is a mature autonomous trading system with:
- **6 active cron jobs** (data, eval, research, dashboard, health, wiki-sync)
- **$100K paper trading** operational, no graduation candidates yet
- **11 wiki compound pages** documenting research and decisions
- **Factor rotation + volatility targeting** strategies implemented (v2.7, v2.8)

Q3 2026 focus: **Strategy enhancement** and **live trading preparation** via ML complexity, CTA trend overlays, and graduation gate validation.

## Current State Assessment

### What's Working
| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | ✅ Healthy | 17 symbols, hourly fetch, fresh data |
| Paper Trading | ✅ Operational | SPY/GLD/TLT 46/38/16 baseline |
| Cron Jobs | ✅ 6/8 passing | Build script PATH issue fixed |
| Wiki Sync | ✅ Active | Auto-updating compound pages |
| Health Monitor | ✅ Passing | 6/6 checks OK |
| Factor Rotation | ✅ Implemented | v2.7 released |
| Vol Targeting | ✅ Implemented | v2.8 released |

### Technical Debt
| Issue | Priority | Estimated Effort |
|-------|----------|------------------|
| Build/position-sync crons | Low | Fixed (PATH export added) |
| Regime classifier training | Medium | 2-3 days |
| Walk-forward validation | Medium | 1-2 days |

## Q3 2026 Objectives

### 1. Strategy Enhancement (July)

**v2.9: ML Feature Engineering Enhancement**
- **Goal**: Implement nonlinear interaction features per AQR "Virtue of Complexity" research
- **Rationale**: Complex models with proper regularization outperform linear by 50-100%
- **Work Item**: `/root/wiki/projects/portfolio-lab/work/2026-05-12-v29-ml-feature-engineering/`

**Deliverables:**
- [ ] Enhance `factor_rotation.py` with interaction features:
  - Value-momentum synergy score
  - Extreme value flags (percentile > 90)
  - Momentum acceleration (3m vs 6m)
  - Volatility-adjusted momentum
  - Cross-asset correlation divergence
- [ ] Implement walk-forward validation framework
- [ ] A/B test: enhanced vs baseline factor rotation
- [ ] Target: +0.1 Sharpe improvement

**Files:** `src/strategy/factor_rotation.py`, `src/research/features.py`

---

**v2.10: CTA Trend-Following Overlay**
- **Goal**: Add CTA-style multi-timeframe trend overlay as strategy option
- **Rationale**: CTAs provide crisis alpha and diversification; volatility targeting reduces DD by 30%
- **Work Item**: `/root/wiki/projects/portfolio-lab/work/2026-05-12-v210-cta-trend-overlay/`

**Deliverables:**
- [ ] Create `src/strategy/cta_overlay.py`:
  - Multi-timeframe trend detection (20d/60d/120d SMA)
  - Volatility-targeted position sizing
  - Equal risk allocation across markets
  - Ensemble trend scoring
- [ ] Universe: SPY, QQQ, IWM, TLT, IEF, GLD, DBC, VIX
- [ ] Weekly rebalancing schedule
- [ ] Crisis alpha verification (2008, 2020, 2022 backtests)
- [ ] UI: Add "CTA Overlay" to strategy selector

**Target Performance:** Sharpe ~0.6, Max DD <20%, Crisis alpha positive

### 2. Risk Management Hardening (August)

**v2.11: Kill Switch Enhancement**
- Graduated response (reduce 50% at 10% DD, 100% at 15% DD)
- Whipsaw detection (3+ flips in 20 days → pause)
- Correlation spike alerts (>0.7 intra-market)

**v2.12: Paper Trading Graduation Gates**
- Formalize 63-day evaluation period
- Automated Sharpe >0.5 check
- Max DD <15% enforcement
- Win rate >45% tracking
- Human approval workflow integration

### 3. Live Trading Preparation (September)

**v3.0: Live Trading Beta**
- [ ] Alpaca live account integration
- [ ] Position reconciliation engine
- [ ] Order execution with slippage models
- [ ] Real-time P&L tracking
- [ ] Kill switch → live broker connection
- [ ] Paper/live toggle with 24h cool-off

**Prerequisites:**
- 63 days paper trading success
- 2+ Sharpe >0.5 strategies validated
- Risk limits tested under stress scenarios

## Research Pipeline (Ongoing)

### Q3 Research Topics
| Priority | Topic | Expected Output | Timeline |
|----------|-------|-----------------|----------|
| High | ML virtue of complexity | Feature engineering implementation | July |
| High | CTA risk management | Trend overlay strategy | July-Aug |
| Medium | ESG factor rotation | Alternative core holding options | August |
| Medium | Alternative data (crypto) | BTC/ETH signal integration | September |
| Low | HMM regime detection | Statistical vs rule-based regimes | September |

### Wiki Compound Pages to Create
1. `ml-virtue-of-complexity-2025` (done)
2. `cta-trend-following-risk-2025` (done)
3. `v29-feature-engineering-results` (July)
4. `v210-cta-overlay-backtest` (August)
5. `v30-live-trading-adr` (September)

## Resource Allocation

### Development Effort Estimate
| Phase | Weeks | Focus |
|-------|-------|-------|
| July (v2.9) | 2-3 | ML features + testing |
| July-Aug (v2.10) | 3-4 | CTA overlay + backtest |
| August (v2.11-2.12) | 2 | Risk hardening |
| September (v3.0) | 3-4 | Live trading prep |

### Dependencies
- Alpaca live account setup (human task)
- Walk-forward validation data (automated)
- CTA universe ETF data (already in pipeline)

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| ML overfitting | Medium | Walk-forward validation, regularization |
| CTA whipsaw losses | Medium | Multi-timeframe ensemble, vol targeting |
| Live trading errors | Low | Extensive paper testing, kill switches |
| Data pipeline failures | Low | Health monitoring, multiple fallbacks |

## Success Metrics

### Q3 Goals
- [ ] 2 new strategy variants with Sharpe >0.6
- [ ] All 8 cron jobs passing consistently
- [ ] Paper trading Sharpe >0.5 sustained 63 days
- [ ] Wiki compound pages: 15+ total
- [ ] Live trading beta ready (pending human approval)

### Definition of Done (Q3)
- Feature code merged to main
- Wiki documentation complete with citations
- Backtest results validated vs benchmarks
- No critical health check failures for 30 days

## Appendix: Work Items

### Active (Ready for Development)
1. `work/2026-05-12-v29-ml-feature-engineering/` - ML complexity features
2. `work/2026-05-12-v210-cta-trend-overlay/` - CTA trend overlay

### Pending (Q3 Planning)
3. v2.11 Kill switch enhancement (to be created)
4. v2.12 Graduation gates formalization (to be created)
5. v3.0 Live trading beta (to be created)

## Notes

- Build script PATH fix applied 2026-05-12
- Current paper portfolio: $100,000 (no trades executed yet)
- Health monitor passing all checks
- Research agent: no triggers, monitoring active

---

**Status**: Plan Ready for Review  
**Next Action**: Approve roadmap and begin v2.9 implementation
