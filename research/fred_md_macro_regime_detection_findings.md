# FRED-MD Macro Regime Detection: Research Findings

## Source: Oliveira et al. 2025 arXiv 2503.11499

**Paper**: "Tactical Asset Allocation with Macroeconomic Regime Detection"
**Authors**: Daniel Cunha Oliveira, Dylan Sandfelder, Andre Fujita, Xiaowen Dong, Mihai Cucuringu
**URL**: https://arxiv.org/abs/2503.11499
**HTML (with figures)**: https://arxiv.org/html/2503.11499v2

---

## 1. Two-Layer Modified K-Means Algorithm (Core Contribution)

The paper's key innovation is a two-layer k-means that separates outlier detection from regime classification:

### Layer 1: L2 K-Means for Outlier Detection (k=2)
- **Input**: PCA-reduced FRED-MD monthly state vectors (127 series -> 61 components at 95% variance)
- **Distance metric**: Euclidean (L2) -- deliberately chosen because L2 is sensitive to outliers
- **k=2**: Splits months into two clusters: A (smaller) and B (larger)
- **Logic**: The smaller cluster (A) = Regime 0 (atypical months). L2 sensitivity ensures deviant months are captured.
- **Condition**: If |A| <= |B|, Regime 0 = A; otherwise swap labels
- **Rationale**: Outlier months (financial crises, structural breaks) should not contaminate the "normal" regime centroids

### Layer 2: Cosine K-Means for Typical Regimes (k=5)
- **Input**: Only the "typical" months from cluster B
- **Distance metric**: Cosine similarity -- magnitude-agnostic, focuses on pattern shape not intensity
- **k selection**: Standard k-means elbow heuristic applied in cosine space
- **Optimal k = 5** (5 distinct macroeconomic regime types)
- **Output**: Regimes 1 through 5, representing distinct "business as usual" states

### Algorithm 1 Pseudocode (from paper)

```
X = concat(x_t for t=1..T)           # FRED-MD monthly state vectors
{A, B} = KMeans_L2(X, 2)              # Layer 1: outlier split
if |A| <= |B|:
    R_0 = A                           # Regime 0 = outliers
else:
    swap A, B
    R_0 = A
r = ElbowHeuristic_Cosine(B)          # Elbow on cosine distance -> k=5
{R_1, ..., R_r} = KMeans_Cosine(B, r) # Layer 2: cosine clustering
return regime_probabilities(X)        # Soft assignment from centroid distances
```

### Soft Probability Assignment
- Converts centroid-to-point distances into probability distributions (fuzzy c-means style)
- For Regime 0: P(Regime 0) = 0.0 => distance 0; P(Regime 0) = 0.5 => P_max; P(Regime 0) = 1.0 => infinite distance
- Combines probabilities from both KMeans_L2 and KMeans_Cosine
- Output: probability distribution over all (r+1) regimes for each month

### Matching Clustering Algorithm
- Ensures cluster interpretation consistency across walk-forward windows
- Prevents label-switching (where the same regime gets different numbers in different windows)
- Uses centroid matching between successive estimation windows

---

## 2. FRED-MD Data Preprocessing Pipeline

### Raw Data
- **127 monthly variables** from FRED-MD database (McCracken & Ng, 2016)
- Date range: December 1959 to January 2023 (paper); final eval subsample 2000-2022 (746 months)
- 8 variable groups: Output/Income, Consumption/Orders/Inventories, Labor Market, Housing, Money/Credit, Interest/Exchange Rates, Prices, Stock Market

### 7 T-Code Transformations
Each FRED-MD indicator has a predefined t-code (1-7) for stationarity transformation:

| T-Code | Transformation | Formula | Example Variables |
|--------|---------------|---------|------------------|
| 1 | No transformation | x_t | Interest rates, ratios |
| 2 | First difference | x_t - x_{t-1} | Already stationary series |
| 3 | Second difference | (x_t-x_{t-1}) - (x_{t-1}-x_{t-2}) | Some prices |
| 4 | Natural log | ln(x_t) | Levels of positive series |
| 5 | Log first difference | ln(x_t) - ln(x_{t-1}) | Real GDP, IP, employment |
| 6 | Log second difference | ln(x_t) - 2*ln(x_{t-1}) + ln(x_{t-2}) | Some price indices |
| 7 | Percent change | (x_t - x_{t-1}) / x_{t-1} | Series with zeros/negatives |

### PCA Dimensionality Reduction
- Standardize all variables after t-code transformation (demean + unit variance)
- PCA on the standardized 127-dimensional space
- **Threshold**: 95% cumulative variance explained
- **Components**: ~61 (from 127 original dimensions)
- **Purpose**: Noise reduction, factor extraction, computational efficiency

---

## 3. Walk-Forward Validation

### Oliveira et al. Setup
- **Estimation window**: Fixed 48 months (4 years) -- NOT expanding
- **Prediction horizon**: 1 month ahead
- **Total test period**: 2000-2022 (746 monthly observations after merging FRED-MD + returns)
- **Re-estimation**: Regime detection + forecasting model re-fit at each step
- **Asset universe**: 9 sector ETFs (SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV) + TLT
- **Volatility scaling**: 10% annualized vol target

### 4 Forecasting Models Tested
1. **Naive**: Conditional expected returns given most probable regime, using Sharpe ratios from past regime-specific performance
2. **Black-Litterman**: Sample moments combined with regime-conditioned priors using BL framework
3. **Linear Ridge Regression**: Regime-specific parameter estimates, combined via current regime probabilities
4. **Mean-Variance Optimization**: Regime-conditioned expected returns + GARCH-like conditional volatility

### Position Sizing Variants
- **lo**: Long-only (weights 0 to 1)
- **lns**: Long and short (weights -1 to 1)
- **los**: Long or short (all-in one direction)
- **mx**: Mixed (combination)

### Evaluation Metrics
- Sharpe Ratio, Sortino Ratio, Average Drawdown, Maximum Drawdown, % Positive Returns
- Transaction costs modeled (1-10 bps)
- Slippage modeled (2 bps)
- Volatility-scaled to 10% annualized target

---

## 4. Performance Results

### Top Results (Table from paper)

| Strategy | Sharpe | Sortino | Max DD | %Pos |
|----------|--------|---------|--------|------|
| **Ridge LO (l=3)** | **1.505** | **4.449** | **-4.389** | -- |
| MVO LO (l=2) | 1.128 | 2.022 | -6.776 | 0.629 |
| MVO LO (l=4) | 1.129 | 2.132 | -6.933 | 0.629 |
| BL LO (l=2) | 1.177 | -- | -- | 0.665 |
| Naive LO (l=4) | 1.065 | 2.541 | -6.719 | 0.549 |
| Naive LO (l=2) | 0.981 | 2.532 | -6.163 | 0.540 |
| **SPY** | **0.818** | 1.331 | -33.492 | 0.662 |
| **EW** | **0.838** | 1.445 | -32.286 | 0.662 |

### Key Takeaways
1. **Ridge LO (l=3) is the champion**: Sharpe 1.505, Sortino 4.449, Max DD -4.4% -- massive improvement over SPY baseline of 0.818
2. **Long-only dominates**: Every long-short variant underperformed its long-only counterpart
3. **Lower dimensions better**: l=2 or l=3 (regimes as 2-3 dimensional probability vectors) beats l=4
4. **Max DD dramatically reduced**: Ridge LO -4.4% vs SPY -33.5% and EW -32.3%
5. **Regime-based strategies shine in downturns**: Outperform during 2008, 2020, 2022 by significant margins

---

## 5. Practical GitHub Implementations

### Repo A: EstherBD/Market-Regime-Detection (Most Relevant)
**URL**: https://github.com/EstherBD/Market-Regime-Detection-with-PCA-and-Clustering

**Implementation details**:
- **FRED data**: SPX, NASDAQ, yields (2y/10y/30y/3m), VIX, CPI, UNRATE, INDPRO
- **Feature engineering**: Log returns (1/5/21/63d), realized vol (21d), correlations (60d), yield curve level/slope/curvature, YoY CPI, unemployment level
- **Winsorization**: 1%/99% clipping on extreme features
- **StandardScaler**: suffix `_z` on standardized columns
- **PCA**: 5 components (~70% variance) for interpretability
- **KMeans**: k=3 (vs paper's k=2 + k=5), n_init=20, random_state=0
- **Results**: Crisis (7%), Risk-off (20%), Risk-on stable (73%)
- **Output**: Transition matrix, regime durations, regime timeline, cluster heatmaps

```python
# Key code pattern
scaler = StandardScaler()
features_std = pd.DataFrame(
    scaler.fit_transform(features_cleaned[std_cols]),
    index=features_cleaned.index,
    columns=[col + '_z' for col in std_cols]
)
pca = PCA(n_components=2)
X_pca = pca_2.fit_transform(X)
km = KMeans(n_clusters=3, random_state=0, n_init=20)
feat_df['Regime'] = km.fit_predict(X_pca[:, :3])
```

### Repo B: manav363/market-regime-detection (Walk-Forward Focus)
**URL**: https://github.com/manav363/market-regime-detection

**Implementation patterns**:
- Random Forest (not k-means) with walk-forward TimeSeriesSplit
- 5-fold walk-forward validation
- 14 features: RSI, ATR, Momentum, Volatility, Trend indicators
- Confidence threshold: 0.55 (signals below threshold suppressed)
- Risk: 1% per trade, volatility-adjusted sizing
- Costs: 1 bps transaction, 2 bps slippage
- OOS Sharpe: ~0.98 (AAPL)
- FastAPI dashboard for visualization

### Repo C: YuvrajChauhan-Fin/macro-liquidity-regime-strategy-v4
**URL**: https://github.com/YuvrajChauhan-Fin/macro-liquidity-regime-strategy-v4
- Walk-forward validation
- Risk budgeting and vol targeting
- Macro liquidity regime approach

---

## 6. Alternative Approaches

### Gaussian Mixture Models (GMM)
- Tested in Oliveira et al. Section 4.3 as direct comparison
- Inherently probabilistic (no heuristic distance conversion needed)
- Both methods produce similar regime classifications
- Minor differences in regimes 2 and 3 (transition periods)
- Paper suggests both are robust; GMM aligns slightly better with NBER recession dating

### DBSCAN
- Density-based; handles arbitrary shapes and noise
- Mentioned in related work, not directly tested
- Promising for outlier detection step (replacing L2 k-means)

### Silhouette Score
- `sklearn.metrics.silhouette_score(X, labels, metric='euclidean')`
- Range: -1 to +1 (higher = better)
- Used for post-fit cluster quality assessment
- Davies-Bouldin Index also available (lower = better)
- Paper used elbow heuristic instead for k selection

### HMM / Markov-Switching
- Hamilton (1989) classical approach
- boujeepants24/regime-factor-model repo uses HMM + multi-factor equity model
- Not directly compared in Oliveira et al.

### Spectral Clustering
- Not tested in reviewed sources
- Useful for non-convex regime boundaries

---

## 7. Recommended Implementation for Portfolio Lab

1. **FRED-MD data fetch**: Use fredapi or CSV downloads from St. Louis Fed
2. **T-code transformation**: Map each of 127 series via its t-code
3. **PCA**: 95% threshold (~61 components) or simpler 70% (~5 components for interpretability)
4. **Two-layer k-means**: L2(k=2) -> Cosine(k=5 by elbow)
5. **Soft assignment**: Distance->probability conversion
6. **Transition matrix**: Empirical regime transition counts
7. **Walk-forward**: 48-month fixed window, 1-step ahead
8. **Ridge regression** conditioned on regime probabilities
9. **Long-only, vol-scaled** positions at 10% target vol
10. **Evaluate**: Sharpe, Sortino, Max DD vs SPY benchmark
