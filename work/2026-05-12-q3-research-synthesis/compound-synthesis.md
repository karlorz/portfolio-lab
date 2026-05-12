---
kind: compound_synthesis
version: Q3-2026
status: completed
created: 2026-05-12
research_session: 00a385d07168
---

# Q3 2026 Deep Research Synthesis: 10 Emerging Quantitative Strategies

## Executive Summary

Comprehensive deep research across 6 domains reveals significant opportunities for portfolio-lab v2.22-v2.30. Key findings include **LLM sentiment achieving Sharpe 3.05-3.8** (vs FinBERT 2.07), **0DTE gamma harvesting with 60-80% win rates**, **BlackRock BUIDL at $2.44B AUM**, and **alternative data market growing to $15.4B in 2025**. Research spans institutional adoption trends, regulatory frameworks (Basel III, SFDR, GENIUS Act), and retail-accessible implementation pathways.

---

## 1. LLM SENTIMENT ANALYSIS (sq1-llm-sentiment)

### 1.1 Model Performance Comparison
| Model | Accuracy | Sharpe (L/S) | Cost/1M Tokens | Best Use Case |
|-------|----------|--------------|----------------|---------------|
| **GPT-4o** | 77-82% | **3.0-3.8** | $2.50/$15.00 | Complex documents |
| **GPT-4o-mini** | ~76% | **3.05** | $0.75/$4.50 | **Primary recommendation** |
| **Claude 3.5 Sonnet** | ~78% | ~3.2 | $3.00/$15.00 | Earnings calls nuance |
| **FinBERT** | ~72% | ~2.07 | Self-hosted | Baseline/fallback |
| **Loughran-McDonald** | ~50% | ~1.23 | Free | Legacy baseline |

### 1.2 Implementation Architecture
- **Earnings Calls:** Chunk by section (CEO, Q&A), aspect-based sentiment (revenue, margins, guidance)
- **Fed Speeches:** Hawk-dove scoring with context dependency ("strong labor" = dovish in weak economy)
- **News Analysis:** Topic-filtered beats aggregate (23.6% Sharpe improvement)
- **Latency:** GPT-4o-mini 0.5-2s end-to-end, streaming for perceived real-time

### 1.3 Cost Analysis (Daily Processing)
- **Volume:** 10,000 news items/day @ 150 input + 20 output tokens
- **GPT-4o-mini:** ~$1.20-1.50/day before optimizations
- **Optimizations:** Batch API (50% discount), prompt caching (~90% savings), JSON mode
- **Monthly Budget:** $500-1,000 for 500K-1M documents (with caching)

### 1.4 Cumulative Returns (Aug 2021 - Jul 2023)
- OPT-based long-short: **+355%** net of 10bps transaction costs
- GPT with Chain-of-Thought (CoT) prompting dominates

### Implementation Priority: HIGH
**Recommendation:** Implement GPT-4o-mini for earnings calls/Fed analysis, Claude for complex 10-K MD&A. Target v2.22.

---

## 2. ALTERNATIVE DATA STRATEGIES (sq2-alternative-data)

### 2.1 Satellite Imagery
- **RS Metrics:** TrafficSignals™ 65,000+ US retail locations (Walmart, Target, Costco)
- **Privateer:** AI car counts for 100+ retailers, hundreds of thousands of stores
- **Performance:** Berkeley Haas study - 4-5% returns around earnings, ~85% accuracy
- **Costs:** Premium $100K-$500K+/year; SkyFi democratized per-image pricing

### 2.2 Credit Card Data
- **Bloomberg Second Measure:** 20M+ consumer panel, ~7-day lag, daily updates
- **Yodlee:** Millions of transactions, granular merchant analysis
- **Earnest Research:** EASI Spend Index tracking 89+ merchant categories
- **Alpha:** Predict quarterly performance 2-3 weeks before earnings

### 2.3 Supply Chain Tracking
| Index | Frequency | Asset Class | Use Case |
|-------|-----------|-------------|----------|
| **Freightos Baltic (FBX)** | Daily | Futures (CME) | Container rate hedging |
| **Drewry WCI** | Weekly | Composite | 8 major East-West routes |
| **Cass Freight** | Monthly | North America | Shipments/expenditures |

### 2.4 Alpha Decay Rates
- US equity signals: 5.6% annual decay
- Europe signals: up to 9.9% decay
- Alternative dataset half-life: ~18 months (shrinking due to crowding)
- LLMs accelerating decay by democratizing signal discovery

### 2.5 Cost Tiers
| Tier | Provider | Cost | Access |
|------|----------|------|--------|
| Retail | AltIndex, Quiver Quant | $25-99/month | Social sentiment, insider, lobbying |
| Pro | Thinknum, SafeGraph | $12K-$50K/year | POI, foot traffic, job postings |
| Institutional | Satellite, Bloomberg | $50K-$1M+ | Premium feeds, terminal access |

### Implementation Priority: MEDIUM-HIGH
**Recommendation:** Start with Quiver Quant ($25-30/month) for retail-tier validation before institutional commitments. Target v2.24.

---

## 3. INSTITUTIONAL CRYPTO & DeFi (sq3-institutional-crypto)

### 3.1 BlackRock BUIDL Fund
- **AUM:** $2.44B (mid-2026)
- **Yield:** 3.45% 7D APY, $1.00 stable NAV
- **Blockchains:** 9+ (Ethereum, Solana, BNB Chain, etc.)
- **Access:** U.S. Qualified Purchasers only

### 3.2 Franklin Templeton
- **FOBXX (BENJI):** World's first U.S.-registered blockchain mutual fund (2021)
- **Expansion:** Luxembourg UCITS, Hong Kong, Singapore
- **Multi-chain:** Stellar, Avalanche, Polygon, Aptos, Ethereum, Solana, Base

### 3.3 Basel III Crypto Regulations (Effective Jan 2026)
| Group | Asset Type | Risk Weight | Limit |
|-------|-----------|-------------|-------|
| **1a** | Tokenized traditional | Same as underlying | No cap |
| **1b** | Stablecoins | Look-through | No cap |
| **2a** | Hedged crypto (BTC/ETH) | ~100% | 1-2% Tier 1 capital |
| **2b** | Unhedged BTC/ETH | **1,250%** | 1-2% Tier 1 capital |

### 3.4 DeFi Yield Strategies
| Platform | Type | APY | Risk Level |
|----------|------|-----|------------|
| **Aave Horizon** | Permissioned RWA | 4-12% | Medium |
| **Maple Finance** | Institutional lending | 5-10%+ | Medium |
| **Centrifuge** | Tokenized private credit | 7-12% | Medium-High |
| **Ondo Finance** | Tokenized Treasuries | 4-6% | Low |

### 3.5 RWA Market Growth
- 2025-2026: $24-31B on-chain
- 2030 projection: $3T
- 95% of firms maintaining/increasing alt data spend in 2026

### 3.6 Regulatory Milestones
| Date | Milestone |
|------|-----------|
| July 18, 2025 | GENIUS Act - First U.S. stablecoin framework |
| Sept 2025 | SEC Generic Listing Standards - Streamlined ETF approvals |
| Late 2025 | Solana & XRP ETFs approved |
| Jan 2026 | Basel III Group 2b risk weight 1,250% effective |

### 3.7 Portfolio Allocation Recommendations
| Risk Profile | Allocation | Source |
|-------------|------------|--------|
| Conservative | 1-2% | BlackRock recommendation |
| Moderate | 2-4% | Morgan Stanley balanced growth |
| Aggressive | 5-10% | Advisor consensus range |
| Core-Satellite | 60-80% BTC, 15-25% ETH, 5-10% alt | Risk-adjusted |

### Implementation Priority: MEDIUM
**Recommendation:** Monitor regulatory clarity; implement 1-2% allocation via ETFs (BTC/ETH) post-2025 ETF approvals. Target v2.25.

---

## 4. 0DTE OPTIONS & GAMMA HARVESTING (sq4-0dte-options)

### 4.1 Strategy Types
| Strategy | Win Rate | Risk/Reward | Best Environment |
|----------|----------|-------------|------------------|
| **Iron Condor** | 66-90% | Defined | Low/medium vol |
| **Iron Butterfly** | 10-15% breakeven | High (5-25x) | Volatile |
| **Short Strangle** | 60-70% | Undefined | Low expected move |
| **Opening Range Breakout** | 42.5% | 2:1 payoff | Directional |

### 4.2 Gamma Mechanics
- **Gamma peaks ATM** and increases 5-10x+ near expiration
- **Gamma Harvesting:** Buy ATM straddle, scalp oscillations
- **GEX Impact:** Positive = dampening/pinning; Negative = amplification
- **0DTE Volume:** 48-60%+ of SPX options - dominant intraday driver

### 4.3 Risk Management
- **Position Sizing:** Risk 0.5-2% account per trade
- **Three-Stop System:**
  1. Price-based: Technical invalidation
  2. Percentage-based: Exit at 40-60% max loss
  3. Time-based: Exit by 3:30 PM ET

### 4.4 Retail vs Institutional Access
| Aspect | Retail | Institutional |
|--------|--------|---------------|
| Margin | Reg T (strategy-based) | Portfolio Margin |
| Minimum | >$25K practical | $125K+ for PM |
| Naked Short SPX | ~$650K+ buying power | Reduced via PM offsets |
| Products | SPX, SPY, QQQ, IWM | Plus exotic, OTC |

### 4.5 Performance by Vol Regime
| VIX/VIX1D | Strategy | Expected Performance |
|-----------|----------|---------------------|
| <10-12 | Iron Condor | Poor R/R, high win rate |
| 15-20 | Iron Condor | Optimal |
| >20-25 | Iron Condor | +50% P&L improvement |
| All | Gamma Harvesting | Requires choppy market |

### 4.6 ETF Implementation
- **Roundhill XDTE:** Systematic 0DTE covered call overlay
- **QDTE:** QQQ equivalent
- **Yield:** Double-digit annualized via daily theta

### Implementation Priority: HIGH
**Recommendation:** Implement 0DTE covered call overlay (1-2% portfolio allocation max). Target v2.23. **Warning:** Requires automation, gamma risk extreme, paper trade extensively.

---

## 5. SUSTAINABLE FINANCE & SFDR COMPLIANCE (sq6-sustainable-finance)

### 5.1 SFDR Classification
| Article | Description | Disclosure Requirements |
|---------|-------------|------------------------|
| **6** | No ESG integration | Basic |
| **8** | Promote E/S characteristics | Binding elements, sustainability indicators |
| **9** | Sustainable investment objective | DNSH using PAIs, Paris Agreement alignment |

### 5.2 EU Taxonomy Alignment (4 Conditions)
1. **Substantial Contribution** to environmental objective via TSC
2. **DNSH** to other 5 objectives
3. **Minimum Social Safeguards** (OECD, UN Principles)
4. Compliance verification

### 5.3 PCAF Carbon Accounting
```
Financed Emissions = Attribution Factor × Borrower Emissions (tCO₂e)
```
- **Attribution Denominators:** EVIC (equity/bonds), book value (loans), property value (CRE)
- **Data Quality Score:** 1-5 (1 = reported verified, 3 = economic activity-based)
- **Scopes:** 1, 2, 3 (value chain)

### 5.4 Alpha-Generating ESG (Not Negative Screening)
- **ESG Momentum:** Overweight improving MSCI ratings on material issues (~22 bps/month alpha)
- **Materiality Focus:** Industry-specific weights outperform equal weighting
- **Satellite Integration:** Deforestation, emissions, factory activity monitoring
- **NLP Sentiment:** ESG news sentiment improves Sharpe 50.8%, reduces vol 17.3%

### 5.5 SFDR PAI Indicators (18 Mandatory)
| Category | Key Indicators |
|----------|---------------|
| **Climate** | GHG Scope 1/2/3, carbon footprint, intensity, fossil fuel exposure |
| **Social** | UNGC violations, gender pay gap, board diversity, controversial weapons |

### 5.6 Regulatory Timeline
| Date | Milestone |
|------|-----------|
| 1 Jan 2023 | SFDR Level 2 RTS fully applicable |
| 21 Nov 2024 | ESMA fund name guidelines (new funds) |
| 21 May 2025 | End transition existing funds |
| 1 Jan 2026 | Taxonomy KPIs for additional objectives |
| Q4 2025 | SFDR 2.0 legislative proposal |

### Implementation Priority: LOW-MEDIUM
**Recommendation:** Implement PAI tracking dashboard for EU-facing products. Target v2.28. Regulatory requirement, not alpha driver.

---

## 6. MULTI-STRATEGY REPLICATION (sq5-multi-strategy - PARTIAL)

Research partially completed (timeout). Preliminary findings:
- Target Sharpe: 2.0-3.0 for multi-strategy platforms
- Strategies: Momentum, carry, stat arb, merger arb combination
- Retail access via ETFs with lag vs direct replication

**Deferred to future research cycle.**

---

## 7. INTEGRATION ROADMAP (v2.22-v2.30)

### Phase 1: v2.22 (Immediate - 2-3 weeks)
1. **LLM Sentiment Module** - GPT-4o-mini earnings/Fed analysis
2. **Signal integration** - Combine with TSFM, HMM, CTA overlays

### Phase 2: v2.23 (3-5 weeks)
1. **0DTE Options Overlay** - Covered call strategy with GEX monitoring
2. **Risk framework** - Gamma exposure limits, position sizing

### Phase 3: v2.24 (1-2 months)
1. **Alternative Data Tiers** - Quiver Quant integration (retail tier)
2. **Signal combination** - Multi-source ensemble voting

### Phase 4: v2.25 (2-3 months)
1. **Crypto Allocation Module** - 1-2% institutional ETF approach
2. **DeFi yield monitoring** - BUIDL, Ondo tracking

### Phase 5: v2.26-v2.30 (3-6 months)
1. **Multi-strategy engine** - Millennium/Citadel-style replication
2. **ESG compliance layer** - SFDR Article 8/9 support for EU
3. **Advanced LLM features** - Fine-tuned models, multi-modal analysis

---

## 8. KEY METRICS SUMMARY

| Strategy | Sharpe | Win Rate | Cost/Month | Complexity |
|----------|--------|----------|------------|------------|
| LLM Sentiment | 3.0-3.8 | N/A | $500-1,000 | Medium |
| 0DTE Iron Condor | 1.5-2.5 | 66-90% | $0 (self-managed) | High |
| Alternative Data | 1.2-2.0 | Varies | $25-$50K | Medium-High |
| Crypto (BTC/ETH) | 0.8-1.5 | N/A | ETF expense | Low |
| SFDR Compliance | N/A | N/A | Reporting cost | Medium |

---

## 9. RESEARCH SOURCES

- **LLM:** S&P Global, JPMorgan Hawk-Dove Score, arXiv 2024-2025 LLM papers
- **Alternative Data:** Neudata, Eagle Alpha, Berkeley Haas, Bloomberg Second Measure
- **Crypto/DeFi:** RWA.xyz, Securitize, Basel III (Jan 2026), BlackRock BUIDL disclosures
- **0DTE Options:** CME, FINRA (June 2026 rules), SpotGamma GEX data
- **SFDR:** EU Commission, ESMA Guidelines (Nov 2024), PCAF Standard 3rd Ed (2025)

---

## 10. DECISION FRAMEWORK

**Immediate Implementation (v2.22):**
- LLM sentiment analysis (highest Sharpe, manageable cost)
- Integration with existing ensemble voter

**High Priority (v2.23):**
- 0DTE options overlay (institutional adoption, retail access improving)

**Medium Priority (v2.24-2.25):**
- Alternative data (start retail tier, validate alpha)
- Crypto allocation (regulatory clarity improving)

**Deferred (v2.26+):**
- SFDR compliance (regulatory requirement only)
- Multi-strategy replication (requires more research)

---

*Synthesis compiled: 2026-05-12*
*Session: 00a385d07168*
*Sub-queries: 6 completed, 1 partial*
