# Deep Research Synthesis: Next-Gen Portfolio Strategies

**Work Item:** 2026-05-13-deep-research-trending-next  
**Date:** 2026-05-13  
**Status:** SYNTHESIS COMPLETE

## Executive Summary

Research completed on 4 focus areas for next-generation portfolio strategies targeting Sharpe > 1.0, Max DD < 15%, Annual turnover < 150%.

## Focus Area 1: Intraday Microstructure Signals ✓

### Key Findings
JP Morgan is deploying intraday strategies on single names (previously only futures):
- **Intraday momentum** - Capturing microstructure effects in mega-caps
- **Mean reversion** - Short-term price dislocations
- **Order flow modeling** - TradeFM foundation model for trade-flow prediction

**TradeFM Paper (JPMorgan AI Research, 2026):**
- Generative foundation model for trade-flow and market microstructure
- Models atomic order flow: stream of buy/sell orders
- Addresses non-stationary dynamics across assets and liquidity regimes
- Each participant observes partial view of true market state

**Implementation Opportunity:**
- P2 priority (requires high-frequency data access)
- Target: Extend existing execution layer with microstructure features
- Potential alpha: 50-200 bps annually on large-cap exposures

## Focus Area 2: Alternative Risk Premia (ARP) ✓

### Key Findings - AQR Research

**Systematic Styles Across Asset Classes:**
- **Value**: Buy cheap assets (high book-to-price, high real-yield), sell expensive
- **Momentum**: Buy recent outperformers, sell underperformers (6-12 month returns)
- **Carry**: Buy higher-yielding assets, sell lower-yielding (currencies, roll yield)
- **Defensive**: Low-beta, quality, low-volatility exposures
- **Trend**: Multi-asset trend following

**Performance Characteristics (1990-2016 backtests):**
- Individual style Sharpe ratios: 0.3-1.7
- Value and momentum often negatively correlated (diversification benefit)
- Composite portfolios show low correlation to 60/40 portfolios
- **Cross-asset application enhances robustness**

**Economic/Behavioral Basis:**
- Risk-based: Compensation for bearing distress risk (value), global imbalances (carry)
- Behavioral: Anchoring, herding, leverage aversion create persistent mispricings

**Implementation Opportunity:**
- **P1 PRIORITY** - Can extend existing multi-speed framework
- Add carry signals to TSMOM overlay (already have momentum)
- Implement value factors for equity selection (MTUM complement: VLUE)
- Target: Sharpe 0.96 → 1.05 with ARP overlay

## Focus Area 3: Cross-Asset Arbitrage (Deferred)

**Status:** Requires sophisticated infrastructure for statistical arbitrage
- P3 priority due to data/compute requirements
- DE Shaw, Two Sigma implementations not directly replicable

## Focus Area 4: Factor Timing with ML (Partial)

**Existing Implementation:**
- v2.53 HMM-LSTM Regime Detector already provides 5-state classification
- Current performance: Regime-based allocation shifts

**Enhancement Opportunity:**
- P1 priority - Extend existing HMM with factor momentum overlay
- Implement "Value and Momentum Everywhere" signals (Asness, Moskowitz, Pedersen)
- Add carry signals to current momentum framework

## Implementation Roadmap

### P1: Alternative Risk Premia Overlay (v2.60)
**Timeline:** Week 1-2  
**Source:** AQR "Understanding Alternative Risk Premia", "Value and Momentum Everywhere"

**Components:**
1. **Carry Signal Module**
   - Bond carry: Real yield spreads (TLT vs IEF)
   - Equity carry: Dividend yield vs risk-free
   - Gold carry: Storage cost vs expected return
   
2. **Value Signal Module**
   - P/E, P/B percentile rankings
   - Real yield percentile for bonds
   - Complement existing MTUM with VLUE signals

3. **Cross-Asset Integration**
   - Extend SignalIntegrator with ARP source
   - Weight: 8-10% in composite (reduce technical/macro slightly)
   - Regime-dependent weighting: Value > Momentum in bear markets

**Expected Outcome:**
- Sharpe 0.96 → 1.05
- Max DD improvement through negative value-momentum correlation
- Maintain existing 8-source architecture

### P2: Microstructure Execution Enhancement (v2.61)
**Timeline:** Week 2-3  
**Source:** JP Morgan TradeFM paper, intraday momentum research

**Components:**
1. **Volume Profile Integration**
   - Extend existing seasonality execution (v2.71)
   - Add volume-weighted optimal windows
   
2. **Order Flow Signals** (if data available)
   - Tick-level imbalance indicators
   - Short-term mean reversion for execution timing

### P3: Factor Momentum + Value Integration (v2.62)
**Timeline:** Week 3-4  
**Source:** "Value and Momentum Everywhere" (Asness et al.)

**Components:**
1. **Factor Rotation Module**
   - HMM regime detection triggers factor weight shifts
   - Bull regime: Momentum 60%, Value 40%
   - Bear regime: Value 60%, Momentum 40%, Defensive 20%

## References

1. **JP Morgan AI Research** - TradeFM: A Generative Foundation Model for Trade-flow and Market Microstructure (arXiv:2602.23784, 2026)
   - https://arxiv.org/html/2602.23784v1

2. **AQR Capital Management** - Understanding Alternative Risk Premia
   - https://www.aqr.com/Insights/Research/White-Papers/Understanding-Alternative-Risk-Premia

3. **AQR** - Value and Momentum Everywhere (Asness, Moskowitz, Pedersen)
   - https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly
   - https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2018/11/Value-and-Momentum-Everywhere_000.pdf

4. **JP Morgan Markets** - Institutional Trading Research
   - https://www.jpmorgan.com/markets

## Next Steps

1. Implement ARP overlay module (`src/signals/arp_overlay.py`)
2. Extend SignalIntegrator with carry + value sources
3. Validate with 2005-2026 backtest
4. Target Sharpe > 1.0 for composite 8+2 source system
