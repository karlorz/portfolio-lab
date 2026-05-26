# Deep Research: FRED-MD Regime Detection for Tactical Asset Allocation

> Research conducted 2026-05-26

---

## TL;DR

1. **Oliveira et al. (arXiv:2503.11499)** proposes a two-layer modified k-means regime detection pipeline: L2 distance to isolate crisis months, then cosine distance to cluster normal regimes -- the paper does NOT use GaussianHMM, contrary to what the search query suggested
2. **Distance-to-probability** uses a fuzzy c-means style formula (Equation 1) -- normalized inverse distances from centroids, then combines layers via a log-scaled probability anchor (Equation 4)
3. **Temporal modeling** uses a Markov transition matrix computed directly from k-means labels, iterated as a Markov chain -- no HMM in the original paper
4. **Walk-forward validation** uses a 48-month fixed window with 1-month-ahead predictions, applied to 746 monthly observations (2000-2022) across 10 sector ETFs
5. **Two Python implementations exist**: a faithful replication by klmtseng and a related HMM-based system (RegimeX) by dreamrun24

---

## Overview

This paper (Oliveira, Sandfelder, Fujita, Dong, Cucuringu, 2025) is the first to apply regime detection from the **FRED-MD macroeconomic dataset** (127 variables, McCracken & Ng 2016) to tactical asset allocation. The authors develop a data-driven clustering approach that classifies months into 6 macroeconomic regimes, then uses these regimes to conditionally forecast ETF returns and construct portfolios.

---

## Findings

### 1. Two-Layer Modified K-Means Algorithm

**Layer 1 -- L2 Clustering for Outlier Detection (Section 3.1.1)**
- Applies k-means with k=2 using Euclidean (L2) distance
- Separates months into two groups: the smaller cluster contains "outlier" months representing extreme macroeconomic conditions (economic crises)
- The larger cluster (set B) represents "business as usual" periods
- Economic rationale: crisis months have macroeconomic vectors with extreme magnitudes (e.g., skyrocketing unemployment, collapsed production)

**Layer 2 -- Cosine Clustering for Regular Regimes (Section 3.1.2)**
- Applied only to set B (normal months) using cosine distance
- Cosine distance is magnitude-agnostic, measuring angular similarity between state vectors
- k determined by elbow heuristic (paper finds optimal r=5)
- Produces 5 distinct regimes: Economic Recovery, Expansionary Growth, Stagflationary Pressure, Pre-Recession Transition, Reflationary Boom

**Data preprocessing:**
- 127 FRED-MD variables, excludes group 6 (Interest & Exchange Rates) to focus on US macro
- t-code transformations applied (log, differencing) for stationarity
- PCA to reduce dimensionality (95% variance threshold = 61 components)
- Data from Dec 1959 to Jan 2023

### 2. Converting K-Means Distances to Probabilities (Fuzzy C-Means Formula)

**Equation 1 -- Distance-to-Probability:**
```
P(C_i) = (1 - d_i / SUM_j(d_j)) / SUM_m(1 - d_m / SUM_j(d_j))
```
This is essentially the fuzzy c-means membership formula applied to k-means centroids. The authors state: "We emphasize that this procedure is essentially the same as fuzzy c-mean, although implemented in a slightly different fashion."

**Combining L2 and Cosine Layers (Equations 2-4):**
Because L2 and cosine k-means operate on different subsets, combining probabilities requires care:
1. Compute cosine-cluster probabilities normally for Regimes 1..r
2. Find `P_max = max(P(Regime 1), ..., P(Regime r))`
3. Anchor crisis probability `P_R0` via three conditions:
   - P_R0 = 0 when P(Regime 0) = 0.0
   - P_R0 = P_max when P(Regime 0) = 0.5
   - P_R0 = infinity when P(Regime 0) = 1.0
4. Continuous function: **P_R0 = -P_max * log2(1 - P(Regime 0))** (Equation 4)
5. Renormalize the full distribution to sum to 1

### 3. GaussianHMM / Temporal Smoothing

**Critical finding: The paper does NOT use GaussianHMM.** Instead, it uses a simpler approach:

1. **Regime Transition Probability Matrix** (Section 3.3): Computed as empirical transition counts:
   `e_ij = TransitionCount(Regime i, Regime j) / |Regime i|`
2. **Markov Chain Forecasting** (Section 5.1): The current regime distribution is normalized and multiplied by the transition matrix:
   `p_tilde_t = p_t / |p_t|`
   `p_tilde_{t+1} = p_tilde_t^T * E_t`
3. The paper compares this to GMM (Section 4.3) and finds the modified k-means provides "more interpretable view of uncertainty"

**RegimeX repo** (separate project) does use `hmmlearn.hmm.GaussianHMM` for regime detection, with Viterbi decoding and regime interpretation (Bull/Bear/Crisis) based on trailing return and volatility characteristics.

### 4. Walk-Forward Validation Methodology

From Section 6 of the paper:
- **Data**: 746 monthly observations (Feb 2000 to Dec 2022)
- **Universe**: 10 sector ETFs (SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)
- **Estimation window**: 48 months (4 years), fixed-window
- **Prediction horizon**: 1 month ahead
- **Forecasting models tested**: Naive (regime-conditional Sharpe), Ridge regression, Mean-Variance Optimization, Black-Litterman
- **Portfolio types**: Long-only, Long-and-Short, Long-or-Short, Mixed
- **Metrics**: Sharpe ratio, Sortino ratio, Avg/Max drawdown, % positive returns
- **Control**: Random regime labels (proves signal isn't spurious)
- **Results**: All 4 models outperform random regime baselines at p<0.05 across multiple metrics

The **regime-conditioned-allocation** repo follows the identical approach: PCA + two-layer k-means fit on full sample per convention, then walk-forward Ridge regression. It also provides a fully walk-forward variant in `stress_validation.py` and crisis-window stress testing in `type_error_tradeoff.py`.

### 5. Python Implementations

**Primary: klmtseng/regime-conditioned-allocation**
- https://github.com/klmtseng/regime-conditioned-allocation
- Faithful replication of Oliveira et al. with detailed validation
- Full pipeline in `regime_pipeline.py`: FRED-MD (pandas_datareader) + PCA + two-layer k-means + Markov forecasting + Ridge + portfolio construction
- Additional validation scripts: `ic_validation.py`, `negative_control.py`, `benchmark_distance.py`, `stress_validation.py`, `type_error_tradeoff.py`, `gap_closing.py`
- Key finding: Signal is cross-sectional (ranking), not time-series
- OOS Sharpe 1.645 (narrow 2-year window), permutation test p ≈ 0.055
- Includes NBER recession comparison, regime characterization, and IC decomposition

**Secondary: dreamrun24/RegimeX**
- https://github.com/dreamrun24/RegimeX---Macro-Aware-Tactical-Asset-Allocation-Engine
- Related but different: uses GaussianHMM instead of two-layer k-means
- More modular: `models/hmm.py` (GaussianRegimeHMM wrapper), `features/engineering.py` (RobustZScoreScaler), `backtest/walk_forward.py`, `allocation/optimizer.py`
- Uses CVXPY for regime-specific convex optimization
- Features: yfinance + fredapi + VIX + realized volatility
- Transaction costs, turnover tracking, gross vs net performance
- Baselines: 60/40, equal-weight, volatility-switching

---

## Verification Methods

To verify any of these findings:
- Read the paper directly: https://arxiv.org/abs/2503.11499
- Run the primary replication: `git clone https://github.com/klmtseng/regime-conditioned-allocation.git && cd regime-conditioned-allocation && python regime_pipeline.py`
- Test the RegimeX pipeline: `git clone https://github.com/dreamrun24/RegimeX---Macro-Aware-Tactical-Asset-Allocation-Engine.git && cd RegimeX---Macro-Aware-Tactical-Asset-Allocation-Engine && python scripts/run_research.py --offline-demo`
- Validate the two-layer k-means directly with sklearn: `KMeans(n_clusters=2)` with default Euclidean distance, then `KMeans(n_clusters=r, algorithm='full')` on normalized cosine-transformed data
- FRED-MD data is accessible via `pandas_datareader.get_data_fred()` with no API key required for public endpoints

---

## Analysis

**Key insight for portfolio-lab integration**: The Oliveira paper's approach is complementary to the existing regime detection in portfolio-lab. The portfolio-lab currently uses asset-return-based regimes (volatility regimes via GARCH/MSM), while this paper uses purely macroeconomic data. A combined approach could:

1. Use the two-layer k-means on FRED-MD data to establish macro regimes (monthly)
2. Map these macro regimes to the existing daily volatility/momentum regimes
3. Use the fuzzy probability (Equation 1) as a soft signal rather than hard classification
4. The Markov transition matrix provides a natural way to forecast next-month regime distribution

**Caveats:**
- GaussianHMM is NOT part of the Oliveira paper -- adding it would be a separate contribution
- The 48-month fixed window is relatively short for macroeconomic regime detection; the regime-conditioned-allocation repo notes that PCA/kmmeans fit on full sample is convention
- The paper's OOS results come from a ~2-year window (2021-2023), which is narrow
- FRED data is revised and not point-in-time (noted in both repos)

---

## Sources

1. Oliveira et al. (2025), "Tactical Asset Allocation with Macroeconomic Regime Detection," arXiv:2503.11499. https://arxiv.org/abs/2503.11499
2. klmtseng/regime-conditioned-allocation (GitHub). https://github.com/klmtseng/regime-conditioned-allocation
3. dreamrun24/RegimeX (GitHub). https://github.com/dreamrun24/RegimeX---Macro-Aware-Tactical-Asset-Allocation-Engine
4. McCracken & Ng (2016), "FRED-MD: A Monthly Database for Macroeconomic Research," Journal of Business & Economic Statistics.
5. Aaron Tseng (2026), "Where Macro Regime Signals Actually Live" (Substack). https://aarontsengquant.substack.com/p/where-macro-regime-signals-actually
6. Bezdek (1981), "Pattern Recognition with Fuzzy Objective Function Algorithms," Springer.
