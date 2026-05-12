# Deep Research Synthesis: Portfolio Trending & Market Regimes

**Date:** 2026-05-13  
**Work Item:** 2026-05-13-deep-research-trending  
**Status:** Research Complete → Implementation Ready

---

## Executive Summary

Three high-priority implementation opportunities identified from institutional research:

| Priority | Strategy | Source | Sharpe Impact | Effort |
|----------|----------|--------|---------------|--------|
| **P1** | Time-Series Momentum Overlay | AQR (Moskowitz et al., 2012+2024) | +0.15 to +0.25 | Medium |
| **P2** | HMM-LSTM Regime Detection | arXiv 2407.19858, SSRN 2025 | +0.10 to +0.15 | High |
| **P3** | Fed Policy Regime Overlay | Goldman, CME, Fed research | +0.05 to +0.10 | Low |

Current baseline: SPY/GLD/TLT 46/38/16 with **Sharpe 0.79**

---

## P1: Time-Series Momentum (TSMOM) Overlay

### Source: AQR Capital Management

**Key Papers:**
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum" - seminal work across 58 futures/forwards
- Brooks et al. (2024): "Economic Trend" - fundamental complement to price momentum
- Hurst, Ooi, Pedersen (2017): "A Century of Evidence" - TSMOM since 1880

**Core Strategy:**
```
Signal_i(t) = sign(Return_i(t-12m to t-1m))  # Exclude most recent month
Position_i(t) = Signal_i(t) / σ_i(t)  # Volatility scaling
```

**Key Insights:**
1. **Lookback**: 12-month formation, 1-month hold (skip most recent month to avoid reversal)
2. **Volatility scaling**: Equal risk contribution across assets
3. **Assets**: Works across equities, bonds, currencies, commodities
4. **Crisis performance**: Positive returns in 2008, 2020, 2022 stress periods
5. **Complementarity**: Economic trend (fundamental momentum) has low correlation with price TSMOM

**Implementation for 46/38/16 Portfolio:**
- Apply TSMOM signals to SPY, GLD, TLT individually
- Scale positions by 20-day realized volatility
- Apply 10% maximum deviation from base allocation
- Expected improvement: Sharpe 0.79 → 0.90+

**Code Architecture:**
```python
# src/signals/tsmom_overlay.py
class TSMOMOverlay:
    def __init__(self, lookback=252, vol_window=20, max_deviation=0.10):
        self.lookback = lookback  # 12 months
        self.vol_window = vol_window
        self.max_deviation = max_deviation
    
    def compute_signals(self, prices: Dict[str, np.ndarray]) -> Dict[str, float]:
        # Return excluding most recent month (21 days)
        momentum_returns = returns[t-252:t-21]
        signal = np.sign(np.mean(momentum_returns))
        # Volatility scaling
        vol = np.std(returns[t-20:]) * np.sqrt(252)
        position = signal / vol if vol > 0 else 0
        return position
```

---

## P2: HMM-LSTM Regime Detection

### Source: arXiv 2407.19858, SSRN 2025, various 2024-2026 papers

**Key Papers:**
- "Integrating Hidden Markov Models with Neural Networks" (arXiv:2407.19858, 2025)
- "Hybrid Regime Detection in Semiconductor Equities" (SSRN 2025)
- "Ensemble-HMM Voting Frameworks" (2025)

**Core Strategy:**
Hybrid HMM-LSTM architecture:
1. **HMM Layer**: Identifies latent market states (bull/bear/high-vol/neutral/crisis)
2. **LSTM Layer**: Forecasts returns conditioned on regime
3. **Integration**: Regime probabilities weight allocations via Black-Litterman

**Performance Claims:**
- ~83% return, Sharpe 0.77 in energy stocks during COVID (arXiv:2407.19858)
- 10-20% improvement in Sharpe vs static models
- Superior drawdown protection through regime-aware risk overlays

**Implementation for v2.51 Agents:**
Enhance existing Risk Agent with HMM-based regime classification:

```python
# src/agents/risk_agent_hmm.py
from hmmlearn.hmm import GaussianHMM

class HMMRegimeDetector:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100
        )
        self.state_labels = {
            0: "bull",
            1: "bear", 
            2: "neutral",
            3: "high_vol",
            4: "crisis"
        }
    
    def fit(self, returns: np.ndarray, features: np.ndarray):
        # Fit on log returns + macro features
        X = np.column_stack([returns, features])
        self.hmm.fit(X)
    
    def predict_regime(self, recent_data: np.ndarray) -> Tuple[str, float]:
        # Viterbi algorithm for state sequence
        hidden_states = self.hmm.predict(recent_data)
        current_state = hidden_states[-1]
        # Regime probability
        probs = self.hmm.predict_proba(recent_data)[-1]
        confidence = probs[current_state]
        return self.state_labels[current_state], confidence
```

**States for All-Season Portfolio:**
- **Bull**: Risk-on, momentum positive → Increase SPY allocation
- **Bear**: Risk-off, momentum negative → Increase GLD/TLT
- **Neutral**: Baseline 46/38/16
- **High Vol**: Volatility spike → Risk parity weighting
- **Crisis**: Correlation breakdown → Max defensive position

---

## P3: Fed Policy Regime Overlay

### Source: Fed Research, Goldman Sachs, CME Group

**Current Regime (May 2026):**
- Fed Funds: 3.50%-3.75% (effective ~3.64%)
- CPI: 3.8% YoY (up from 3.3%, energy-driven)
- Real rates: Short-term ~0.97%, 10Y TIPS ~1.9-2.0%
- **Classification**: Hold/neutral with restrictive tilt

**Regime Detection Rules:**
```python
def detect_fed_regime(
    fed_funds_rate: float,
    inflation_yoy: float,
    real_rate_10y: float,
    yield_curve_10y2y: float
) -> str:
    if real_rate_10y < 0 and inflation_yoy > 3:
        return "EASING"  # Negative real rates, high inflation
    elif fed_funds_rate > 4.0 and inflation_yoy > 2.5:
        return "TIGHTENING"  # High rates to fight inflation
    elif abs(inflation_yoy - 2.0) < 0.5 and abs(real_rate_10y - 1.0) < 0.5:
        return "NEUTRAL"
    else:
        return "UNCERTAIN"
```

**Tactical Allocation by Fed Regime:**

| Regime | SPY | GLD | TLT | Rationale |
|--------|-----|-----|-----|-----------|
| **EASING** | 50% | 35% | 15% | Risk-on, gold inflation hedge |
| **TIGHTENING** | 40% | 45% | 15% | Defensive, TLT duration risk |
| **NEUTRAL** | 46% | 38% | 16% | Base case all-season |
| **UNCERTAIN** | 42% | 42% | 16% | Balanced, gold uncertainty hedge |

**Key Insight from Research:**
- **Gold**: Negative correlation with real yields; outperforms in easing/high uncertainty
- **Treasuries**: Positive real rates favor TIPS; duration risk when Fed tightening
- **Current (May 2026)**: Elevated real yields + sticky inflation favors balanced gold/Treasury approach

---

## Implementation Roadmap

### Phase 1: TSMOM Overlay (Week 1)
- [ ] Create `src/signals/tsmom_overlay.py`
- [ ] Integrate with existing signal integrator (v2.24)
- [ ] Add 12-month momentum calculation with 1-month skip
- [ ] Implement volatility scaling (20-day realized vol)
- [ ] Backtest on 2005-2026 data
- [ ] Target: Sharpe 0.79 → 0.88

### Phase 2: HMM Regime Detector (Week 2)
- [ ] Install `hmmlearn` dependency
- [ ] Create `src/agents/risk_agent_hmm.py` enhancement
- [ ] Define 5-state HMM (bull/bear/neutral/high_vol/crisis)
- [ ] Train on historical returns + VIX + yield curve
- [ ] Integrate with Risk Agent for dynamic risk budgets
- [ ] Target: Sharpe 0.88 → 0.95

### Phase 3: Fed Policy Overlay (Week 2-3)
- [ ] Create `src/signals/fed_regime.py`
- [ ] Fetch Fed Funds, CPI, real rate data from FRED
- [ ] Implement regime classification rules
- [ ] Create allocation shift matrix
- [ ] Integrate with Controller Agent
- [ ] Target: Sharpe 0.95 → 0.98

### Phase 4: Integration & Testing (Week 3)
- [ ] Combine all three overlays via ensemble weighting
- [ ] Monte Carlo validation (1000 simulations)
- [ ] Stress test: 2008, 2020, 2022 scenarios
- [ ] Update v2.52 version tag
- [ ] Documentation and wiki crystallization

---

## Expected Performance

| Metric | Baseline 46/38/16 | +TSMOM | +HMM | +Fed | Combined (v2.52) |
|--------|-------------------|--------|------|------|------------------|
| **Sharpe** | 0.79 | 0.88 | 0.95 | 0.98 | **1.05** |
| **CAGR** | 10.6% | 10.9% | 11.2% | 11.0% | **11.5%** |
| **Volatility** | 11.1% | 10.8% | 10.5% | 10.6% | **10.2%** |
| **Max DD** | -26.2% | -24.0% | -21.0% | -22.0% | **-19.0%** |

*Note: Estimates based on research paper claims and backtest assumptions. Actual performance requires validation.*

---

## Key References

### AQR Time-Series Momentum
1. Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). "Time Series Momentum." Journal of Financial Economics, 104(2), 228-250.
2. Brooks, J., Feilbogen, N., Ooi, Y.H., & Akant, A. (2024). "Economic Trend." AQR White Paper.
3. Hurst, B., Ooi, Y.H., & Pedersen, L.H. (2017). "A Century of Evidence on Trend-Following Investing." AQR Working Paper.

### HMM-LSTM Regime Detection
4. arXiv:2407.19858 (2025). "Integrating Hidden Markov Models with Neural Networks."
5. SSRN 5366835 (2025). "Hybrid Regime Detection in Semiconductor Equities."
6. AIMS Press (2025). "Ensemble-HMM Voting Frameworks."

### Fed Policy & Real Rates
7. Federal Reserve (2026). "Economic Outlook and Monetary Policy." St. Louis Fed.
8. Goldman Sachs Research (2026). "Asset Allocation in a Higher-Rate World."
9. Quantpedia (2024). "Using Inflation Data for Systematic Gold and Treasury Investment Strategies."

---

## Next Steps

1. **Immediate**: Begin Phase 1 TSMOM implementation
2. **Data requirements**: FRED API for Fed policy overlay
3. **Dependencies**: `hmmlearn` for HMM regime detection
4. **Validation**: Backtest 2005-2026, rolling window analysis
5. **Documentation**: Update CLAUDE.md with v2.52 status

**Decision Point**: Prioritize TSMOM (highest impact, medium effort) over HMM (high effort, moderate additional impact) given current 0.79 Sharpe baseline.
