---
kind: compound_synthesis
version: 2.20
status: in_progress
created: 2026-05-12
research_session: 114d636884a8
---

# Q2 2026 Quantitative Strategy Research Synthesis

## Executive Summary

Comprehensive deep research across 5 strategy domains reveals significant opportunities for portfolio-lab v2.20. Key findings include **Wasserstein HMM achieving Sharpe 2.18** (vs 1.18 SPY buy-hold), **CTA trend overlays delivering +27% in 2022** when 60/40 failed, and **Transformer-based allocation reaching Sharpe 1.69**. Research synthesized from arXiv 2025-2026 preprints, SSGA institutional research, and CME systematic risk frameworks.

---

## 1. REGIME DETECTION ALGORITHMS (sq1-regime)

### 1.1 k-Nearest Neighbors (kNN) Macro Detection
- **Performance:** Ridge + kNN regimes: Sharpe 1.505 vs 0.838 equal-weight (arXiv 2503.11499v1)
- **Implementation:** FRED-MD macro features (CPI, industrial production, yield spreads, unemployment)
- **k Selection:** 5-30 via cross-validation, distance-weighted voting
- **Caution:** High turnover at daily frequency (0.57 daily) - better for monthly allocation

### 1.2 Hidden Markov Models (HMM) - TOP PERFORMER
- **Wasserstein HMM (2025-2026):** Sharpe **2.18** vs 1.59 equal-weight, Max DD **-5.43%** vs -14.62% SPX
- **Standard HMM:** 19.41% annual return, Sharpe 1.22, Max DD -19.54%
- **Implementation:** GaussianHMM from hmmlearn, VIX changes as primary input
- **Regimes:** Low Vol (State 0) → 100% SPY, High Vol (State 1) → 100% TLT
- **Key Innovation:** Wasserstein distance template tracking prevents label switching, reduces turnover to 0.0079

### 1.3 Clustering-Based Detection
- **SSGA t-distributed GMM (Feb 2025):** 4 regimes identified over 1995-2024
  - Emerging Expansion (42.34%): Rising returns, higher vol
  - Robust Expansion (25.35%): Strong, stable growth
  - Cautious Decline (19.16%)
  - Market Turmoil (13.16%): F1 score ~73-78% for crisis detection
- **Modified k-means (arXiv 2503.11499v1):** 6-regime fuzzy c-means with L2 + Cosine clustering
- **Wasserstein K-Means:** Superior for return distribution clustering vs standard K-means

### Implementation Priority: HIGH
**Recommendation:** Implement Wasserstein HMM as primary regime detector - best Sharpe, lowest drawdown, lowest turnover.

---

## 2. ADAPTIVE RISK PARITY (sq2-risk-parity)

### 2.1 Core Concepts
- **Adaptive RP** adjusts allocations based on correlation regime detection
- Risk parity relies on covariance matrices - correlations spike during crises
- Detect "clustering regimes" and reposition dynamically

### 2.2 Implementation Approaches
- **Regime-Switching Models:** Markov switching on returns/volatilities/correlations
- **Rules-Based:** Momentum signals + volatility filters + correlation metrics
- **Dynamic Optimization:** Rolling covariance + ML for regime clustering

### 2.3 Time-Varying Risk Budgeting
```
RC_i = x_i * (∂ρ/∂x_i)
```
- Recompute conditional risk contributions at each rebalance
- Apply turnover limits (τ), liquidity constraints, position bounds
- Use ADMM or cyclical coordinate descent for optimization

### 2.4 Performance: Adaptive vs Static
| Strategy | Mean Return | Volatility | Sharpe | Max DD |
|----------|-------------|------------|--------|--------|
| **Dynamic RP** | **26.86%** | 18.95% | **1.418** | **-27.70%** |
| Static RP | 25.40% | 18.57% | 1.368 | -27.88% |
| Markowitz MVO | 25.86% | 15.63% | 1.655* | -31.20% |

*MVO high Sharpe in-sample only; less robust out-of-sample

### 2.5 Institutional Frameworks
- **AQR:** Risk parity +1.7% annual vs 60/40 at same vol; 63% Sharpe improvement
- **Bridgewater All Weather:** 25+ years, four economic quadrants (growth/inflation rising/falling)
- **Columbia Threadneedle:** Adaptive Risk Allocation reduced max DD from -31% to -19%

### Implementation Priority: MEDIUM-HIGH
**Recommendation:** Build dynamic risk budgeting engine with regime-dependent risk contributions.

---

## 3. SYSTEMATIC TAIL RISK & CTA OVERLAYS (sq4-tail-risk)

### 3.1 CME Systematic Risk Framework
- **Instruments:** E-mini futures for beta adjustment, managed futures/CTAs for crisis offset
- **CTA Correlation:** -0.54 to equities in declining quarters
- **Alternative to:** Expensive long-put strategies

### 3.2 CTA Trend-Following Overlays
- **Recommended Allocation:** 10-30% (or return-stacking for ~100% exposure)
- **2022 Performance:**
  - SG CTA Index: +18-20.1%
  - SG Trend Index: **+27.3%**
  - KMLM ETF: +24-30%
  - vs 60/40: -15%+ (worst year since 1970s)
  - vs SPY: -18-19%
- **Mechanism:** Long energy/commodities, short bonds/currencies during rate hikes

### 3.3 Volatility Targeting Circuit Breakers
- **Formula:** Position = (Target Vol × Portfolio Value) / (Instrument Vol × Scaling Factor)
- **Target Range:** 8-15% annualized portfolio volatility
- **Circuit Breaker Layers:**
  - Daily loss -1% to -2% → halt new positions
  - Portfolio DD 5-10% → halve sizes
  - Portfolio DD 15-20% → pause/go cash
  - Vol spike regime → force deleveraging

### 3.4 2022 Regime Shift Insights
- Stock-bond correlation flipped from negative (-0.3 to -0.5) to positive
- Managed futures showed low/negative correlation to BOTH stocks AND bonds
- 5-10% CTA allocation cut 60/40 losses by 2-5+ percentage points

### Implementation Priority: HIGH
**Recommendation:** Add CTA trend overlay module with configurable allocation (5-20%) and signal integration.

---

## 4. ML SIGNAL GENERATION (sq5-ml-signals)

### 4.1 Transformer-Based Regime Classifiers
- **Portfolio Transformer (PT):** End-to-end attention with direct Sharpe optimization
- **Performance:** Sharpe **1.65-1.69** vs traditional mean-variance 0.8-1.2
- **Key Architectures:**
  - iTransformer: Treats temporal as channels for long sequences
  - PatchTST: Vision Transformer adapted to time-series
  - Signature-Informed Transformer (SIT): Path signatures for lead-lag
  - Crossformer: Explicit cross-dimensional dependencies
- **Hybrid VSN-LSTM-Attention (DeePM):** ~50% improvement over pure Transformer

### 4.2 Conditional Autoencoders for Factor Risk
- **Core:** Jointly estimates latent risk factors + time-varying betas
- **Architecture:** Encoder → Conditional Betas → Latent Factors → Decoder
- **Performance:** Sharpe 0.65-0.78 after transaction costs (long-only)
- **Use Cases:** Dynamic risk decomposition, mispricing extraction, better VaR
- **Implementation:** PyTorch/TensorFlow, 94 characteristics → 20 influential factors

### 4.3 LSTM/GRU for Time-Series Allocation
- **Two Approaches:**
  - Prediction + Optimization: LSTM forecasts → mean-variance/CVaR
  - End-to-End: RNN outputs softmax-normalized weights directly
- **Performance:** ~50% beat-market rate, LSTM edges GRU on complex dependencies
- **Key Variants:** LSTM + PPO (reinforcement learning) for dynamic allocation

### 4.4 Ensemble Voting Systems
- **Mechanisms:** Hard voting (majority rule), Soft voting (probability averaging)
- **Base Models:** Random Forest, XGBoost, SVM, LSTM, Transformer, ARIMA/GARCH
- **Why It Works:** Uncorrelated models protect each other from errors
- **Performance:** Lower variance, better Sharpe, improved drawdown control

### 4.5 Performance vs Traditional Methods
| Method | Sharpe | vs Traditional |
|--------|--------|----------------|
| Transformer | 1.65-1.69 | +35-50% vs MPT |
| ML + Markowitz | 1.38-1.48 | +15-25% vs pure MPT |
| Deep Risk-Based ML | 1.38 | +55% vs Risk Parity |
| Ensemble Voting | 1.3-1.6 | More stable, lower DD |
| Mean-Variance | 0.8-1.2 | Baseline |

### Implementation Priority: MEDIUM
**Recommendation:** Start with ensemble voting across existing signals (TSFM, trend, risk parity) before building transformer infrastructure.

---

## 5. HYBRID ARCHITECTURE RECOMMENDATION (v2.20)

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE VOTING LAYER                   │
│         (Soft voting with regime-dependent weights)          │
└──────────────┬────────────────────────────────┬─────────────┘
               │                                │
    ┌──────────▼──────────┐        ┌───────────▼──────────┐
    │  WASSERSTEIN HMM    │        │  DYNAMIC RISK PARITY │
    │  Regime Classifier  │        │  - Time-varying ERC  │
    │  - Sharpe 2.18      │        │  - Vol targeting     │
    │  - Turnover 0.0079  │        │  - Correlation adj   │
    └──────────┬──────────┘        └───────────┬──────────┘
               │                                │
    ┌──────────▼──────────┐        ┌───────────▼──────────┐
    │  TSFM v2.15         │        │  CTA TREND OVERLAY   │
    │  Factor Momentum    │        │  - 10-30% allocation │
    │  (Already impl)     │        │  - Crisis alpha      │
    └─────────────────────┘        └──────────────────────┘
```

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: v2.20.1 (Immediate - 1-2 weeks)
1. **Wasserstein HMM Regime Detector** - Highest impact, lowest turnover
2. **CTA Trend Overlay Module** - 2022 crisis protection proven
3. **Ensemble Signal Voting** - Combine existing signals (TSFM, trend, HMM)

### Phase 2: v2.20.2 (2-4 weeks)
1. **Dynamic Risk Parity Engine** - Time-varying risk contributions
2. **Volatility Targeting Circuit Breaker** - Target 10-12% portfolio vol

### Phase 3: v2.21 (Future)
1. **Transformer Regime Classifier** - Requires significant infrastructure
2. **Conditional Autoencoder** - Factor risk decomposition
3. **End-to-End ML Allocation** - After baseline ensemble proven

---

## 7. KEY METRICS TO TRACK

| Strategy | Sharpe | Max DD | Turnover | Complexity |
|----------|--------|--------|----------|------------|
| Wasserstein HMM | 2.18 | -5.43% | 0.0079 | Medium |
| Standard HMM | 1.22 | -19.54% | Low | Low |
| kNN + MVO | 1.81 | -12.52% | 0.57 | Medium |
| Dynamic RP | 1.418 | -27.70% | Medium | Medium |
| Transformer | 1.65-1.69 | Variable | Low-Med | High |
| CTA Overlay | Crisis alpha | - | Low | Low |

---

## 8. RESEARCH SOURCES

- arXiv 2603.04441v1 - Wasserstein HMM (2025-2026)
- arXiv 2503.11499v1 - Modified k-means regime detection (2025)
- SSGA "Decoding Market Regimes" - t-distributed GMM (Feb 2025)
- CME Group "Managed Futures as Crisis Risk Offset" (2024)
- AQR "Understanding Risk Parity" (2010) + "Risk Management and Real World" (2011)
- Columbia Threadneedle "Adaptive Risk Allocation" (2020)
- MDPI JRFM "Dynamic Risk Parity" (2026)
- "Portfolio Transformer for Attention-Based Asset Allocation" (arXiv:2206.03246)
- Stefan Jansen "Machine Learning for Trading" Chapter 20

---

## 9. DECISION FRAMEWORK

**Immediate Implementation (v2.20.1):**
- Wasserstein HMM regime detector (highest Sharpe, lowest turnover)
- CTA trend overlay (crisis alpha proven in 2022)
- Ensemble voting across signals

**Deferred to v2.20.2+:**
- Dynamic risk parity (requires more testing)
- Transformer models (high complexity, marginal improvement)

**Not Recommended:**
- Pure kNN at daily frequency (excessive turnover)
- Complex ML without ensemble baseline (overfitting risk)

---

*Synthesis compiled: 2026-05-12*
*Session: 114d636884a8*
