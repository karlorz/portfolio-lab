# v2.42 Tail Risk Hedging - Deep Research Synthesis

**Date:** 2026-05-13
**Status:** 🔴 IN PROGRESS
**Session IDs:** c506957b4377 (tail hedge), 495a2c4c60be (vol targeting), 85f8c74831bc (ESG)

## Research Summary

### 1. Tail Risk Hedging Strategies (Session: c506957b4377)

**Key Findings:**
- **Optimal Strategy:** Hybrid approach combining protective puts + VIX calls
- **Allocation:** 0.5-2% of portfolio annually
- **Entry Timing:** Buy when VIX < 15-20 (low implied vol)
- **Strike Selection:** 10-30 delta OTM for cost efficiency

**Implementation Options:**
| Strategy | Cost | Convexity | Best For |
|----------|------|-----------|----------|
| Protective Puts | High (theta decay) | Direct equity hedge | Bear markets |
| VIX Calls | Moderate | Vol spike capture | Crash events |
| Put Spreads | Lower | Capped upside | Cost reduction |
| Collar | Zero/net credit | Capped both sides | Income needs |

**2025 Context:**
- Elevated valuations + AI concentration = tail risk elevated
- Expected drag: 0.5-2% annually
- Potential benefit: Enable higher equity exposure long-term

**Instruments:**
- SPX/XSP options (tax-efficient, cash-settled 60/40)
- VIX futures/options (basis risk exists)
- Cambria TAIL ETF (simpler access)
- LEAPs for longer-term hedges

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

**2025 Context:**
- Vol clustering means high vol follows high vol
- Leverage effect (negative return-vol correlation) enables better timing
- Machine learning enhances regime adaptation

### 3. ESG Integration (Session: 85f8c74831bc)

**Key Metrics:**
- **WACI:** Weighted Average Carbon Intensity (tCO2e/$M revenue)
- **Scope 1+2+3:** Direct + indirect emissions tracking
- **Decarbonization Target:** 20-70% reduction achievable via reallocation

**Integration Approaches:**
1. **Screening:** Negative/positive/best-in-class selection
2. **Quantitative:** Climate-adjusted factor models
3. **Scenario Analysis:** NGFS transition/physical risk scenarios
4. **Carbon Pair Trades:** Long low-carbon, short high-carbon

**2025 Context:**
- ESG now treated as priced risk factor (like vol/duration)
- Climate focus dominates (E pillar)
- Portfolios with lower WACI show crisis resilience

## Implementation Plan

### Phase 1: Tail Hedge Module (v2.42a)
- [ ] Protective put calculator (strike, expiration, delta selection)
- [ ] VIX call overlay sizing
- [ ] Hybrid hedge optimizer (puts + VIX)
- [ ] Cost/benefit analytics

### Phase 2: Vol Targeting Enhancement (v2.42b)
- [ ] Realized volatility calculator
- [ ] Target vol position sizer
- [ ] Risk parity integration
- [ ] Regime-based leverage adjustment

### Phase 3: ESG Overlay (v2.42c)
- [ ] WACI calculator for portfolio
- [ ] Carbon intensity scoring
- [ ] Climate risk stress testing
- [ ] ESG-tilted allocation optimizer

## References
- CBOE VXTH Index methodology
- S&P Risk Parity Index 8% Target Vol methodology
- CFA Institute ESG Integration 2025 report
- GMO TCFD Report 2025
