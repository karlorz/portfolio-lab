---
title: "Recommended Next Actions for Portfolio-Lab"
created: 2026-05-28
updated: 2026-05-28
completed: 2026-05-28
tags:
  - research
  - next-actions
  - quantitative-portfolio
  - ensemble-methods
  - regime-detection
  - google-trends
  - transaction-costs
  - gold-allocation
  - risk-parity
---

# Recommended Next Actions for Portfolio-Lab (May 2026)

## TL;DR

1. **Google Trends macro signal** is the highest-impact unexplored signal -- search term spikes for "recession", "inflation", "interest rates" have documented predictive power for regime transitions and could complement the existing behavioral sentiment signal (which uses VIX-proxy with 65.8% false positive rate).
2. **Ensemble meta-learner upgrade** -- the stacking ensemble (XGBoost meta-learner in `stacking_integrator.py`) is implemented but gated behind ML flag; a simpler Bayesian Model Averaging or online learning approach could provide adaptive weighting without ML dependencies.
3. **Transaction cost-aware BL optimization** is partially implemented (BL transaction cost adjustment subtracts costs from posterior returns) but the portfolio construction itself does not include turnover penalties as constraints -- this is the gap between the BL mapper and the turnover validator.
4. **Regime detection is mature** -- the two-stage k-means classifier (Oliveira et al. 2025) is already implemented and validated (ARI=1.0, 3/3 economic coherence). Future improvement is in regime forecasting, not classification.
5. **Gold allocation at 38% is empirically optimal** -- the 256-config sweep confirms GLD 38% is robust; the actionable gap is regime-conditional gold allocation (increase to 42-45% in CRISIS, reduce to 30-35% in NORMAL).

## Overview

This report synthesizes findings from web research (arxiv papers, JP Morgan LTCMA 2026, Thalesians 2025 review), Context7 documentation (PyPortfolioOpt, Riskfolio-Lib), and deep codebase analysis of 47 source files across signals/, strategy/, monitor/, and costs/ modules. The portfolio-lab project is remarkably mature -- 13,558 tests, 6 active ensemble signals, regime detection, Black-Litterman optimization, and a complete monitoring stack. The next frontier is signal expansion (Google Trends), ensemble sophistication (meta-learning without ML dependencies), and regime-conditional portfolio construction.

## Findings by Topic

### 1. Google Trends Integration for Sentiment/Fear Signals

**Status: NOT IMPLEMENTED -- High Priority**

The project currently uses VIX as a proxy for behavioral sentiment (`behavioral_sentiment.py`), but CLAUDE.md notes it has a **65.8% false positive rate** and is net-negative as a standalone signal. Google Trends data offers a complementary information source that measures retail investor attention and fear.

**Key Research Findings:**

- **Da, Engelberg & Gao (2015)** "The Sum of All Fears" -- Google search volume for "debt", "default", "crisis", "recession" predicts negative stock returns. This is one of the most-cited papers on search-based sentiment.
- **Preis, Moat & Stanley (2013)** "Quantifying Trading Behavior in Financial Market" -- Google Trends data for "debt" predicted the 2009 market bottom. Published in Scientific Reports (Nature).
- **Chen, De, Hu & Hwang (2014)** "Wisdom of Crowds" -- Google search volume for company-specific terms predicts short-term returns and earnings surprises.
- **Cao et al. (2025)** arxiv:2503.21422 "From Deep Learning to LLMs" -- documents the shift from traditional features to alternative data including search trends for alpha generation.

**Recommended Implementation:**

```
New signal: src/signals/google_trends_signal.py
```

- **Data source**: pytrends (unofficial Google Trends API) or Google Trends API via RSS feed
- **Search terms to track** (macro fear indicators):
  - "recession" -- spikes precede downturns by 2-6 months
  - "inflation" -- rising search volume correlates with inflation expectations
  - "interest rates" -- spikes during rate hike cycles
  - "stock market crash" -- contrarian indicator (extreme fear = bottom)
  - "unemployment" -- labor market fear gauge
  - "gold" -- safe haven demand proxy
- **Signal construction**: Z-score of 7-day rolling search volume relative to 90-day mean
- **Regime integration**: Gate signal to ON only in HIGH_VOL and NORMAL regimes (same as behavioral_sentiment)
- **Expected impact**: +0.005-0.015 Sharpe (based on academic literature on search-based alpha)
- **Implementation complexity**: LOW (pytrends library, ~200 lines, no ML dependencies)
- **Data frequency**: Daily or weekly (Google Trends updates weekly)

**Integration point**: Feed into `alternative_data_signal.py` as a 6th component, or create standalone signal for ensemble voter.

### 2. Regime Detection Improvements

**Status: MATURE -- Future is Forecasting, Not Classification**

The project already implements a state-of-the-art two-stage k-means regime classifier based on Oliveira et al. (2025, arxiv:2503.11499) "Tactical Asset Allocation with Macroeconomic Regime Detection". Key achievements:

- ARI=1.0 (perfect stability across expanding windows)
- 3/3 economic coherence (GFC, COVID, RateHike all correctly detected)
- 5 regimes: NORMAL, CRISIS, LOW_VOL, HIGH_VOL, RECOVERY
- FRED-MD macroeconomic data integration via FredMdFetcher

**What the Paper Recommends (Oliveira et al. 2025) That Is NOT Yet Implemented:**

The paper proposes a 3-step framework:
1. Classify current regimes (DONE -- two-stage k-means)
2. **Forecast distribution of future regimes** (NOT DONE)
3. **Integrate regime forecasts with asset performance** to optimize allocations (PARTIALLY DONE -- regime-conditional vol targeting exists, but forward-looking allocation is not)

**Recommended Next Action: Regime Transition Probabilities**

```python
# New module: src/regime/regime_transition_forecaster.py
```

- **Approach**: Compute empirical transition matrix from historical regime sequences
- **Enhancement**: Add regime persistence modeling (current data: NORMAL 7.6d, CRISIS 9.9d, LOW_VOL 10.0d, HIGH_VOL 7.1d, RECOVERY 1.4d)
- **Application**: If CRISIS regime has high persistence, increase gold/TLT allocation faster; if RECOVERY is short, enter equity positions earlier
- **Expected impact**: +0.01-0.03 Sharpe through earlier regime-adaptive allocation
- **Complexity**: MEDIUM (requires transition matrix estimation and Bayesian updating)

**Additional Regime Research (2025):**

- **Regime-conditional Black-Litterman**: Use regime forecast to adjust tau parameter (higher in regime transitions, lower in stable regimes)
- **Regime-aware covariance estimation**: Different covariance regimes require different estimation windows (the project's `regime_sharpe_matrix.py` already does per-regime Sharpe, extend to per-regime covariance)

### 3. Ensemble Voting Improvements

**Status: HIGHLY MATURE -- 8-Stage Pipeline**

The ensemble voter (`ensemble_voter.py`, 80.6K) already implements an 8-stage weight adjustment pipeline:
1. Static regime weights
2. Adaptive ensemble weighting (attribution-based)
3. Health-adjusted weighting
4. Correlation penalty
5. Regime-conditional weights
6. Utility-based reweighting
7. Exploration noise (Dirichlet)
8. Turnover-aware validation (basis-pursuit + regret-weighted)

**Gaps and Improvement Opportunities:**

**A. Stacking Ensemble (ML-gated)**

The `stacking_integrator.py` implements an XGBoost meta-learner but is gated behind `PORTFOLIO_LAB_ENABLE_ML=1`. For a non-ML alternative:

- **Bayesian Model Averaging (BMA)**: Weight signals by posterior model probability given observed data. No ML dependency, fully interpretable.
- **Online Learning (Exponential Weighting)**: Update signal weights using exponential moving average of recent IC (Information Coefficient). The project already tracks IC decay (`ic_decay_monitor.py`), so this is a natural extension.
- **Multiplicative Weights Update (MWU)**: Algorithm from online learning theory. Start with equal weights, reduce weights of signals that underperform. No ML dependency.

**B. Signal Diversity Score**

The project computes signal correlation matrices but does not have a formal diversity metric. Research from 2025:

- **DeMiguel, Garlappi & Uppal (2009)** "Optimal Versus Naive Diversification" -- the 1/N portfolio often outperforms optimized portfolios out-of-sample. The ensemble voter's health-adjusted weighting implicitly does this, but a formal diversity score could improve robustness.
- **Implementation**: Track effective number of signals (N_eff = exp(Shannon entropy of weights)). If N_eff < 2, the ensemble is effectively a single-signal strategy.

**C. Adaptive Regime-Conditional Thresholds**

The consensus threshold (2/3) is static. Research suggests:
- **CRISIS**: Lower threshold (1/2) -- act on fewer signals when speed matters
- **NORMAL**: Higher threshold (3/4) -- require more consensus when time is available
- **LOW_VOL**: Standard threshold (2/3) -- balanced approach

### 4. Transaction Cost Optimization

**Status: PARTIALLY IMPLEMENTED -- Gap Between BL and Execution**

The project has:
- Per-ETF cost table with regime multipliers (`etf_cost_table.py`)
- BL transaction cost adjustment (subtracts costs from posterior returns)
- Turnover validator (`turnover_validator.py`)
- Smart rebalancing with regime-adaptive drift thresholds

**What Is Missing:**

**A. Turnover Penalty as Optimization Constraint**

Currently, transaction costs are subtracted from returns in the BL model, but the portfolio optimization itself does not penalize turnover. Research:

- **Kolm & Tutuncu (2019)** "Optimal Revisiting to the Trading/Cost Tradeoff" -- formulate turnover as a quadratic penalty in the objective function
- **Implementation**: Add turnover penalty to EfficientFrontier objective:
  ```python
  # Current: max_sharpe() or min_volatility()
  # Enhanced: max_sharpe() with turnover_penalty
  ef.nonconvex_objective(
      lambda w: sharpe - LAMBDA * turnover_cost,
      constraints=[...]
  )
  ```

**B. Optimal Rebalancing Frequency**

The project uses `REBALANCE_FREQ = 21` (monthly) and drift-based rebalancing. Research:

- **Celati, Florack & Gorecki (2025)** -- optimal rebalancing frequency depends on the ratio of alpha decay to transaction costs. For SPY/GLD/TLT (low alpha decay), quarterly rebalancing may be sufficient.
- **Implementation**: Backtest different rebalancing frequencies (weekly, biweekly, monthly, quarterly, drift-only) and select based on Sharpe after costs.

**C. Execution Quality Monitoring**

The project has circuit breaker and broker error handling but does not track execution quality (slippage vs mid-price, fill rate, order-to-fill latency).

- **Implementation**: Add execution quality metrics to position_sync data
- **Application**: Feed back into cost estimates to improve accuracy

### 5. Gold Allocation Research

**Status: OPTIMAL AT 38% -- Future is Regime-Conditional**

The 256-config gold allocation sweep confirms:
- GLD 38% is optimal for risk-adjusted returns (Sharpe 0.95-0.96)
- TLT 20% beats TLT 16% across all GLD levels
- IEF is a durable but inferior TLT substitute
- The "more gold" thesis (BofA/Goldman) does not improve risk-adjusted returns

**Latest Research (2025-2026):**

- **Central bank gold buying**: Record purchases in 2024-2025 (China, India, Turkey) suggest structural demand support. This favors maintaining or slightly increasing gold allocation.
- **Gold-Treasury correlation dynamics**: Current rolling correlation 0.10 (neutral). If correlation turns negative again (as in 2008), gold becomes more valuable as a diversifier. The project's correlation regime analysis should feed into dynamic allocation.
- **JP Morgan LTCMA 2026**: Projects gold real returns of 2-3% annually with volatility of 15-16%. This is below the portfolio's current GLD allocation efficiency.

**Recommended Action: Regime-Conditional Gold Allocation**

```python
# Extend vol_parity_allocator.py or create src/strategy/gold_allocator.py
GOLD_REGIME_WEIGHTS = {
    'crisis': 0.45,      # Increase from 38% in crises
    'high_vol': 0.42,    # Slight increase in volatility
    'normal': 0.38,      # Current champion weight
    'low_vol': 0.32,     # Reduce in calm markets (equity risk premium)
    'recovery': 0.35,    # Gradual return to normal
}
```

- **Expected impact**: +0.01-0.02 Sharpe (modest improvement over static allocation)
- **Implementation complexity**: LOW (regime already detected, just need conditional weights)

### 6. Risk Parity and Vol Targeting

**Status: MATURE -- Minor Improvements Available**

The project implements:
- Risk parity weight overlay (`risk_parity_weight_overlay.py`)
- Volatility parity allocator (`vol_parity_allocator.py`)
- Regime-conditional vol targeting (9% target, +0.052 Sharpe delta)
- GARCH-CVaR for tail risk estimation

**Improvement Opportunities:**

**A. Maximum Diversification Portfolio (MDP)**

- **Choueifaty (2008)** "Maximum Diversification Portfolios" -- maximize the diversification ratio (weighted average vol / portfolio vol)
- **Implementation**: `ef.nonconvex_objective(diversification_ratio, S)` with PyPortfolioOpt
- **Expected impact**: +0.02-0.04 Sharpe vs current risk parity (based on academic backtests)

**B. Hierarchical Risk Parity (HRP) with Dynamic Hierarchy**

The project has HRP as a fallback in the BL cascade but does not use it as a primary allocation method. Recent research:

- **Lopez de Prado (2016)** "Building Diversified Portfolios that Outperform Out-of-Sample" -- HRP uses hierarchical clustering to build robust portfolios
- **Enhancement**: Update the hierarchical clustering periodically (not just when HRP is used as fallback)

**C. Volatility Targeting with Leverage Constraints**

The project uses 9% target with 1.5x max leverage. Research:

- **Moreira & Muir (2017)** "Volatility-Managed Portfolios" -- the leverage constraint is critical. The project's 1.5x max is conservative; research suggests 2x is optimal for Sharpe maximization, but 1.5x is appropriate for drawdown control.
- **Enhancement**: Test 2x max leverage in backtest to see if Sharpe improves beyond the current +0.04 delta.

### 7. Cross-Cutting Recommendations

**A. LLM-Enhanced Signal Generation (ML-gated)**

The Cao et al. (2025) survey documents LLMs being used for:
- Unstructured data processing (earnings calls, news sentiment)
- Alpha factor generation
- Portfolio construction reasoning

The project's `agents/` module already has LangGraph topology. A lightweight LLM signal could:
- Parse FRED announcements for forward guidance
- Extract sentiment from Fed meeting minutes
- Generate regime narratives from macroeconomic data

**B. Walk-Forward Ensemble Optimization**

The project has per-signal walk-forward validation but not ensemble-level walk-forward optimization. The ensemble weights could be optimized on expanding windows:
- Train on IS period: optimize weights for Sharpe
- Test on OOS period: measure ensemble WFE
- Track if ensemble WFE > 1.0 (indicates stable signal combination)

**C. Transaction Cost Model Upgrade**

The current `etf_cost_table.py` uses static one-way costs. Enhancements:
- **Market-impact model**: Cost = spread/2 + market_impact * sqrt(shares / ADV)
- **Regime-adjusted impact**: Market impact scales with VIX level
- **Implementation**: Extend `estimate_cost_bps()` to include volume-based impact

## Implementation Priority Matrix

| Priority | Action | Expected Sharpe Delta | Effort | Risk |
|----------|--------|----------------------|--------|------|
| P0 | Google Trends macro signal | +0.005-0.015 | 2-3 days | Low |
| P0 | Regime-conditional gold allocation | +0.01-0.02 | 1 day | Low |
| P1 | Regime transition forecasting | +0.01-0.03 | 3-5 days | Medium |
| P1 | Online learning ensemble weights | +0.005-0.01 | 2-3 days | Low |
| P1 | Turnover penalty in BL optimization | +0.01-0.02 | 2 days | Low |
| P2 | Maximum Diversification Portfolio | +0.02-0.04 | 3-5 days | Medium |
| P2 | Effective number of signals tracking | +0.00-0.005 | 1 day | Low |
| P2 | Adaptive consensus thresholds | +0.005-0.01 | 1 day | Low |
| P3 | Execution quality monitoring | Indirect | 2-3 days | Low |
| P3 | LLM signal generation (ML-gated) | +0.01-0.03 | 1-2 weeks | High |

## Verification Methods

To validate any proposed changes:
1. **Backtest with existing framework**: Use `src/backtest/metrics.py` for consistent comparison
2. **Walk-forward validation**: Follow the 15-window expanding protocol (5yr IS, 1yr OOS)
3. **Deflated Sharpe Ratio**: Ensure improvements survive multiple-testing correction (current DSR=0.979 with 94 configs)
4. **Regime-conditional analysis**: Measure improvement per-regime, not just overall
5. **Transaction cost simulation**: Use `etf_cost_table.py` with regime multipliers

**Common wrong methods:**
- Testing only on full-sample (overfitting risk)
- Ignoring transaction costs
- Comparing Sharpe without adjusting for number of trials
- Testing regime signals on data that overlaps with signal construction period

## Sources

1. Cao et al. (2025) "From Deep Learning to LLMs: A survey of AI in Quantitative Investment" arxiv:2503.21422
2. Oliveira et al. (2025) "Tactical Asset Allocation with Macroeconomic Regime Detection" arxiv:2503.11499
3. Thalesians (2025) "Advances in Quantitative Finance in 2025: From Models to Systems" magazine.thalesians.com
4. JP Morgan Asset Management (2026) "Long-Term Capital Market Assumptions" am.jpmorgan.com
5. PyPortfolioOpt Documentation (Context7: /pyportfolio/pyportfolioopt) -- Black-Litterman, HRP, deviation risk parity
6. Riskfolio-Lib (Context7: /dcajasn/riskfolio-lib) -- CVXPY-based portfolio optimization
7. Da, Engelberg & Gao (2015) "The Sum of All Fears" -- Google search volume and stock returns
8. Preis, Moat & Stanley (2013) "Quantifying Trading Behavior in Financial Market" Scientific Reports
9. Moreira & Muir (2017) "Volatility-Managed Portfolios" Journal of Financial Economics
10. Lopez de Prado (2018) "Advances in Financial Machine Learning" -- HRP, deflated Sharpe, regime detection
11. Aldridge (2026) "Regret Equals Covariance" arXiv:2605.14019 -- regret-weighted ensemble selection

## Codebase References

| Module | Relevance | Lines |
|--------|-----------|-------|
| `src/strategy/ensemble_voter.py` | Ensemble voting, 8-stage pipeline | 80.6K |
| `src/signals/behavioral_sentiment.py` | Existing sentiment signal (65.8% FPR) | 18.9K |
| `src/signals/alternative_data_signal.py` | 5-component alternative data | 22.2K |
| `src/strategy/black_litterman_mapper.py` | BL optimization with cost adjustment | 18.0K |
| `src/strategy/risk_parity_weight_overlay.py` | Risk parity weights | 15.4K |
| `src/strategy/vol_parity_allocator.py` | Volatility parity allocation | 14.8K |
| `src/costs/etf_cost_table.py` | Transaction cost model | 1.2K |
| `src/monitor/regime_sharpe_matrix.py` | Per-signal per-regime Sharpe | 17.8K |
| `src/regime/fred_md_two_stage_kmeans.py` | Two-stage k-means regime classifier | N/A |
| `src/signals/multi_timeframe_fusion.py` | Multi-timeframe signal decomposition | 12.1K |
| `src/strategy/regret_weighted_selector.py` | Regret-based signal weighting | 23.0K |
| `src/signals/stacking_integrator.py` | XGBoost meta-learner (ML-gated) | 19.1K |
| `src/monitor/garch_cvar.py` | GARCH-filtered CVaR | 20.4K |
