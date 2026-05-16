"""
Portfolio-Lab v5.01: Transient Statistical Factors for Regime-Aware Risk Modeling

Extracts short-lived statistical factors via rolling PCA on residual returns to capture
time-varying correlation structure during regime transitions. Complements existing
persistent factor models (GARCH-CVaR, entropy monitor).

Key Insight:
During regime transitions, persistent factor models understate risk because they miss
short-lived correlation changes. Transient factors identified via PCA on 20-60 day
rolling windows capture these shifts.

Usage:
    from src.monitor.transient_factors import (
        TransientFactorExtractor, compute_transient_risk
    )

    extractor = TransientFactorExtractor(window=60, n_factors=3)
    metrics = extractor.compute(returns_matrix)  # shape: (n_assets, n_days)

    # Get risk contribution
    risk = compute_transient_risk(metrics)

    # CLI
    python -m src.monitor.transient_factors analyze    # Run analysis on market data
    python -m src.monitor.transient_factors signal     # Generate ensemble signal
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default symbols for multi-asset analysis
DEFAULT_SYMBOLS = ["SPY", "GLD", "TLT", "IEF", "QQQ", "EFA", "SHY"]


@dataclass
class TransientFactorMetrics:
    """Container for transient factor analysis results."""
    timestamp: str
    window: int                    # Rolling window used (days)
    n_assets: int                  # Number of assets analyzed
    n_factors_selected: int        # Number of factors selected via eigenvalue ratio
    explained_ratio: float         # Variance explained by selected factors
    factor_eigenvalues: List[float]  # All eigenvalues
    factor_stability: float        # 0-1: how stable factors are vs prior window
    risk_contribution: float       # Transient factor risk contribution (as % of total risk)
    regime_transition_score: float # 0-1: higher = more likely in transition
    individual_loadings: Dict[str, List[float]]  # Asset loadings on each factor
    stability_trend: str           # "stable", "shifting", "transition"
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert numpy types
        for k, v in result.items():
            if isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = {k2: v2.tolist() if isinstance(v2, np.ndarray) else float(v2) if isinstance(v2, np.floating) else v2 for k2, v2 in v.items()}
        return result


class TransientFactorExtractor:
    """
    Extract transient statistical factors from rolling PCA on residual returns.
    
    Features:
    - Rolling PCA with 20d/60d windows
    - Factor count selection via eigenvalue ratio (parallel analysis-inspired)
    - Factor stability metric comparing consecutive windows
    - Regime transition probability estimation
    
    Uses only numpy for PCA (no scikit-learn dependency required in no-ML mode).
    """
    
    def __init__(
        self,
        window: int = 60,
        min_window: int = 20,
        max_factors: int = 4,
        eigenvalue_ratio_threshold: float = 1.5,
        stability_lookback: int = 5,
        transition_threshold: float = 0.6,
    ):
        """
        Args:
            window: Primary rolling window for PCA (default: 60 trading days)
            min_window: Minimum window for PCA (fallback if data is short)
            max_factors: Maximum number of factors to extract
            eigenvalue_ratio_threshold: Min ratio between consecutive eigenvalues
                for factor selection (parallel analysis heuristic)
            stability_lookback: Number of windows to compare for stability
            transition_threshold: Stability score below this = regime transition
        """
        self.window = window
        self.min_window = min_window
        self.max_factors = max_factors
        self.eigenvalue_ratio_threshold = eigenvalue_ratio_threshold
        self.stability_lookback = stability_lookback
        self.transition_threshold = transition_threshold
        
        # State persistence
        self._last_eigenvectors: Optional[np.ndarray] = None
        self._stability_history: List[float] = []
    
    def compute(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> TransientFactorMetrics:
        """
        Compute transient factor analysis on return matrix.
        
        Args:
            returns: numpy array of shape (n_assets, n_days) with daily returns
            asset_names: Optional list of asset names for loadings
            timestamp: Optional timestamp string
            
        Returns:
            TransientFactorMetrics with all analysis results
        """
        if returns.ndim != 2:
            raise ValueError(f"Returns must be 2D (n_assets, n_days), got shape {returns.shape}")
        
        n_assets, n_days = returns.shape
        if n_assets < 2:
            raise ValueError(f"Need at least 2 assets, got {n_assets}")
        if n_days < self.min_window:
            raise ValueError(f"Need at least {self.min_window} days, got {n_days}")
        
        effective_window = min(self.window, n_days)
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        ts = timestamp or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Step 1: Compute residual returns by removing market factor
        residual_returns = self._compute_residuals(returns[:, -effective_window:])
        
        # Step 2: PCA on residual returns
        cov_matrix = np.cov(residual_returns, rowvar=True)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 3: Select number of factors
        n_factors = self._select_factor_count(eigenvalues, n_assets)
        explained_ratio = float(np.sum(eigenvalues[:n_factors]) / np.sum(eigenvalues))
        
        # Step 4: Compute factor stability
        stability = self._compute_stability(eigenvectors[:, :n_factors])
        self._stability_history.append(stability)
        
        # Step 5: Compute risk contribution
        risk_contribution = self._compute_risk_contribution(
            eigenvalues[:n_factors], eigenvectors[:, :n_factors], cov_matrix
        )
        
        # Step 6: Regime transition score
        transition_score = 1.0 - stability
        avg_stability = np.mean(self._stability_history[-5:]) if len(self._stability_history) >= 5 else stability
        if transition_score > self.transition_threshold and avg_stability < 0.5:
            stability_trend = "transition"
        elif stability > 0.7:
            stability_trend = "stable"
        else:
            stability_trend = "shifting"
        
        # Step 7: Per-asset factor loadings
        loadings: Dict[str, List[float]] = {}
        for i, name in enumerate(asset_names):
            loadings[name] = [float(eigenvectors[i, k]) for k in range(n_factors)]
        
        return TransientFactorMetrics(
            timestamp=ts,
            window=effective_window,
            n_assets=n_assets,
            n_factors_selected=n_factors,
            explained_ratio=explained_ratio,
            factor_eigenvalues=[float(e) for e in eigenvalues],
            factor_stability=float(stability),
            risk_contribution=float(risk_contribution),
            regime_transition_score=float(transition_score),
            individual_loadings=loadings,
            stability_trend=stability_trend,
        )
    
    def _compute_residuals(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute residual returns by removing the market factor (first PC).
        
        The market factor is approximated as the equal-weighted average of all assets.
        This is a simple approach that doesn't require a market proxy.
        
        Args:
            returns: (n_assets, n_days) array
            
        Returns:
            Residual returns: (n_assets, n_days)
        """
        # Market factor = equal-weighted average return
        market_return = np.mean(returns, axis=0)  # shape: (n_days,)
        
        # Regress each asset on market factor to get residuals
        residuals = np.zeros_like(returns)
        for i in range(returns.shape[0]):
            # Simple OLS: beta = cov(asset, market) / var(market)
            cov = np.cov(returns[i], market_return)[0, 1]
            var_mkt = np.var(market_return) + 1e-10  # avoid division by zero
            beta = cov / var_mkt
            residuals[i] = returns[i] - beta * market_return
        
        return residuals
    
    def _select_factor_count(self, eigenvalues: np.ndarray, n_assets: int) -> int:
        """
        Select number of factors using eigenvalue ratio (parallel analysis heuristic).
        
        Selects factors where eigenvalue ratio to next eigenvalue > threshold,
        keeps at least 1 factor and at most max_factors.
        """
        n_factors = 1  # At least 1 factor
        for i in range(min(len(eigenvalues) - 1, self.max_factors)):
            if eigenvalues[i] <= 0:
                break
            # Calculate ratio of consecutive eigenvalues
            ratio = eigenvalues[i] / (eigenvalues[i + 1] + 1e-10)
            if ratio >= self.eigenvalue_ratio_threshold:
                n_factors = i + 1
            else:
                break
        
        return min(n_factors, self.max_factors)
    
    def _compute_stability(self, current_eigenvectors: np.ndarray) -> float:
        """
        Compute factor stability between current and previous window.
        
        Uses absolute cosine similarity between eigenvector sets.
        Returns 1.0 for identical, 0.0 for completely different.
        """
        if self._last_eigenvectors is None:
            self._last_eigenvectors = current_eigenvectors.copy()
            return 1.0  # First run -> assume stable
        
        # Ensure shapes match for comparison
        n_prev = self._last_eigenvectors.shape[1]
        n_curr = current_eigenvectors.shape[1]
        n_factors = min(n_prev, n_curr)
        
        # Compute cosine similarity matrix between factor pairs
        similarities = []
        for i in range(n_factors):
            v_prev = self._last_eigenvectors[:, i]
            v_curr = current_eigenvectors[:, i]
            cos_sim = np.abs(np.dot(v_prev, v_curr)) / (
                np.linalg.norm(v_prev) * np.linalg.norm(v_curr) + 1e-10
            )
            similarities.append(float(cos_sim))
        
        # Update stored eigenvectors
        self._last_eigenvectors = current_eigenvectors.copy()
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def _compute_risk_contribution(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        full_cov: np.ndarray,
    ) -> float:
        """
        Compute transient factor risk contribution as proportion of total variance.
        
        Returns ratio of transient-factor-explained variance to total variance.
        Higher = transient factors dominate = higher unseen risk.
        """
        total_var = np.trace(full_cov)
        if total_var <= 0:
            return 0.0
        
        # Variance explained by selected transient factors
        factor_var = float(np.sum(eigenvalues))
        
        # Return as proportion (0-1 range)
        return min(factor_var / total_var, 1.0)
    
    def get_ensemble_signal_value(self, metrics: TransientFactorMetrics) -> float:
        """
        Convert transient factor analysis into an ensemble signal (-1 to +1).
        
        Negative = high regime transition risk -> reduce equity exposure
        Positive = stable regime -> maintain/increase equity exposure
        
        Logic:
        - stability_trend == "transition": signal = -0.5 to -0.8 (reduce equity)
        - stability_trend == "shifting": signal = -0.2 to -0.5 (cautious)
        - stability_trend == "stable": signal = +0.2 to +0.5 (maintain/aggressive)
        """
        if metrics.stability_trend == "transition":
            # Regime transition: high uncertainty, reduce equity exposure
            base = -0.5
            # Amplify if many factors (fragmentation) or high risk contribution
            if metrics.n_factors_selected >= 3:
                base -= 0.2
            if metrics.risk_contribution > 0.5:
                base -= 0.1
            return max(base, -0.8)
        
        elif metrics.stability_trend == "shifting":
            # Moderate uncertainty, slight caution
            base = -0.2
            if metrics.regime_transition_score > 0.4:
                base -= 0.2
            return max(base, -0.5)
        
        else:  # stable
            # Stable regime, slightly positive
            base = 0.2
            if metrics.explained_ratio > 0.7:
                base += 0.2  # Well-explained by few factors -> clean regime
            return min(base, 0.5)
    
    def save_state(self, path: Optional[Path] = None) -> None:
        """Save extractor state for persistence."""
        if path is None:
            path = DATA_DIR / "transient_factor_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "stability_history": [float(s) for s in self._stability_history[-50:]],
            "last_eigenvectors": self._last_eigenvectors.tolist() if self._last_eigenvectors is not None else None,
            "updated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Optional[Path] = None) -> bool:
        """Load saved state. Returns True if state loaded."""
        if path is None:
            path = DATA_DIR / "transient_factor_state.json"
        
        if not path.exists():
            return False
        
        try:
            with open(path) as f:
                state = json.load(f)
            self._stability_history = [float(s) for s in state.get("stability_history", [])]
            ev = state.get("last_eigenvectors")
            if ev is not None:
                self._last_eigenvectors = np.array(ev)
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False


def compute_transient_risk(
    metrics: TransientFactorMetrics,
    base_cvar: Optional[float] = None,
) -> Dict:
    """
    Compute transient-factor-adjusted risk estimate.
    
    Args:
        metrics: Transient factor analysis
        base_cvar: Current CVaR estimate (if available)
        
    Returns:
        Dict with adjusted risk metrics
    """
    # Risk multiplier based on transient factor instability
    stability_factor = metrics.factor_stability
    transition_score = metrics.regime_transition_score
    
    # Base risk multiplier from stability
    if stability_factor < 0.3:
        risk_multiplier = 1.25  # 25% risk uplift
    elif stability_factor < 0.5:
        risk_multiplier = 1.15  # 15% uplift
    elif stability_factor < 0.7:
        risk_multiplier = 1.05  # 5% uplift
    else:
        risk_multiplier = 1.0   # No uplift
    
    # Additional adjustment from transition probability
    if transition_score > 0.7:
        risk_multiplier *= 1.1  # Extra 10% during active transitions
    
    # Factor fragmentation adjustment
    if metrics.n_factors_selected >= 3:
        risk_multiplier *= 1.05  # Multiple factors = fragmented risk
    
    adjusted_cvar = None
    if base_cvar is not None:
        adjusted_cvar = round(base_cvar * risk_multiplier, 4)
    
    return {
        "risk_multiplier": round(risk_multiplier, 4),
        "stability_factor": round(stability_factor, 4),
        "transition_score": round(transition_score, 4),
        "stability_trend": metrics.stability_trend,
        "n_factors": metrics.n_factors_selected,
        "explained_ratio": round(metrics.explained_ratio, 4),
        "adjusted_cvar": adjusted_cvar,
        "transient_risk_contribution": round(metrics.risk_contribution, 4),
        "recommended_equity_adjustment": round(
            -risk_multiplier + 1.0 if stability_factor < 0.5 else stability_factor - 0.5,
            4
        ),
    }


def fetch_market_returns(
    symbols: Optional[List[str]] = None,
    days: int = 120,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Fetch market returns for transient factor analysis from market.db.
    
    Args:
        symbols: List of symbols (default: SPY, GLD, TLT, IEF, QQQ, EFA)
        days: Number of trading days to fetch
        
    Returns:
        Tuple of (returns_matrix, asset_names, date_strings)
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    db_path = DATA_DIR / "market.db"
    if not db_path.exists():
        # Try prices.json as fallback
        return _fetch_from_prices_json(symbols, days)
    
    import sqlite3
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if price_data table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'"
    )
    if not cursor.fetchone():
        conn.close()
        return _fetch_from_prices_json(symbols, days)
    
    # Get all dates and prices
    all_data: Dict[str, Dict[str, float]] = {}
    all_dates: set = set()
    
    for symbol in symbols:
        cursor.execute(
            "SELECT date, close FROM price_data WHERE symbol=? ORDER BY date",
            (symbol,)
        )
        rows = cursor.fetchall()
        if rows:
            all_data[symbol] = {row[0]: row[1] for row in rows}
            for row in rows:
                all_dates.add(row[0])
    
    conn.close()
    
    if not all_data:
        return _fetch_from_prices_json(symbols, days)
    
    # Find common dates, sorted
    common_dates = sorted(all_dates)
    if len(common_dates) > days:
        common_dates = common_dates[-days:]
    
    # Build returns matrix
    n_assets = len(symbols)
    n_dates = len(common_dates)
    
    price_matrix = np.zeros((n_assets, n_dates), dtype=np.float64)
    dates_list = common_dates
    
    for i, symbol in enumerate(symbols):
        if symbol not in all_data:
            continue
        for j, date in enumerate(common_dates):
            price_matrix[i, j] = all_data[symbol].get(date, np.nan)
    
    # Compute returns
    returns = np.diff(price_matrix, axis=1) / np.maximum(price_matrix[:, :-1], 1e-10)
    
    # Remove NaN columns
    valid_cols = ~np.any(np.isnan(returns), axis=0)
    returns = returns[:, valid_cols]
    
    return returns, symbols, [d for i, d in enumerate(dates_list[1:]) if i < len(valid_cols) and valid_cols[i]]


def _fetch_from_prices_json(
    symbols: List[str],
    days: int = 120,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Fallback: fetch from prices.json."""
    prices_path = Path(PROJECT_ROOT / "public" / "data" / "prices.json")
    if not prices_path.exists():
        # Generate synthetic data for testing
        np.random.seed(42)
        n_days = max(days, 60)
        n_assets = len(symbols)
        returns = np.random.randn(n_assets, n_days) * 0.01
        dates = [(datetime.now() - timedelta(days=n_days - i)).strftime("%Y-%m-%d") for i in range(n_days)]
        return returns, symbols, dates
    
    with open(prices_path) as f:
        data = json.load(f)
    
    # Format: {d: [dates], p: {symbol: [prices]}}
    all_dates = data.get("d", [])
    
    # Find symbol intersections
    price_map: Dict[str, List[float]] = {}
    price_keys = data.get("p", {})
    for symbol in symbols:
        if symbol in price_keys:
            price_map[symbol] = price_keys[symbol]
    
    if len(price_map) < 2:
        # Not enough data
        np.random.seed(42)
        n_days = max(days, 60)
        n_assets = len(symbols)
        returns = np.random.randn(n_assets, n_days) * 0.01
        dates = [(datetime.now() - timedelta(days=n_days - i)).strftime("%Y-%m-%d") for i in range(n_days)]
        return returns, symbols, dates
    
    n_assets = len(price_map)
    n_dates = min(len(all_dates), days)
    aligned_dates = all_dates[-n_dates:] if n_dates <= len(all_dates) else all_dates
    
    price_matrix = np.zeros((n_assets, n_dates))
    asset_names = list(price_map.keys())
    
    for i, name in enumerate(asset_names):
        prices = price_map[name][-n_dates:] if len(price_map[name]) >= n_dates else price_map[name]
        for j in range(min(len(prices), n_dates)):
            price_matrix[i, j] = prices[j]
    
    returns = np.diff(price_matrix, axis=1) / np.maximum(price_matrix[:, :-1], 1e-10)
    valid_cols = ~np.any(np.isnan(returns), axis=0)
    returns = returns[:, valid_cols]
    
    return returns, asset_names, aligned_dates[1:][:returns.shape[1]]


def analyze_transient_factors(output_json: bool = False) -> Dict:
    """
    Convenience function: fetch data and run transient factor analysis.
    
    Returns dict with results suitable for dashboard integration.
    """
    extractor = TransientFactorExtractor(window=60)
    extractor.load_state()
    
    returns, names, dates = fetch_market_returns(days=120)
    
    if returns.shape[1] < 20:
        # Not enough data, return empty
        return {"status": "insufficient_data", "n_days": returns.shape[1]}
    
    metrics = extractor.compute(returns, asset_names=names)
    signal_value = extractor.get_ensemble_signal_value(metrics)
    
    result = {
        "status": "ok",
        "timestamp": metrics.timestamp,
        "stability_trend": metrics.stability_trend,
        "factor_stability": round(metrics.factor_stability, 4),
        "regime_transition_score": round(metrics.regime_transition_score, 4),
        "n_factors_selected": metrics.n_factors_selected,
        "explained_ratio": round(metrics.explained_ratio, 4),
        "risk_contribution": round(metrics.risk_contribution, 4),
        "ensemble_signal": round(signal_value, 4),
        "n_assets": metrics.n_assets,
        "window": metrics.window,
    }
    
    extractor.save_state()
    
    if output_json:
        print(json.dumps(result, indent=2))
    
    return result


def generate_ensemble_signal(output_json: bool = False) -> Dict:
    """
    Generate an ensemble signal from transient factor analysis.
    Called by EnsembleVoter during collect_signals().
    
    Returns:
        Dict with signal_value (-1 to +1), confidence, and supporting info
    """
    result = analyze_transient_factors()
    
    if result.get("status") != "ok":
        return {
            "signal_value": 0.0,
            "confidence": 0.0,
            "reason": result.get("status", "error"),
        }
    
    # Signal value from extractor
    signal_value = result["ensemble_signal"]
    
    # Confidence: higher when stability is clear
    stability = result["factor_stability"]
    if stability > 0.7 or stability < 0.3:
        confidence = 0.7  # Clear signal
    elif 0.3 <= stability <= 0.5:
        confidence = 0.4  # Moderate confidence
    else:
        confidence = 0.5
    
    signal = {
        "signal_value": float(signal_value),
        "confidence": float(confidence),
        "stability": float(stability),
        "trend": result["stability_trend"],
        "n_factors": result["n_factors_selected"],
        "transition_score": result["regime_transition_score"],
        "reasoning": (
            f"Transient factor stability={stability:.2f}, "
            f"trend={result['stability_trend']}, "
            f"n_factors={result['n_factors_selected']}, "
            f"explained={result['explained_ratio']:.1%}"
        ),
    }
    
    if output_json:
        print(json.dumps(signal, indent=2))
    
    return signal


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_transient_factors(output_json=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "signal":
        generate_ensemble_signal(output_json=True)
    else:
        # Default: analyze and print summary
        result = analyze_transient_factors(output_json=True)
        if result.get("status") == "ok":
            print(f"\nRegime: {result['stability_trend'].upper()}")
            print(f"Stability: {result['factor_stability']:.2f}")
            print(f"Transition Score: {result['regime_transition_score']:.2f}")
            print(f"Signal: {result['ensemble_signal']:.2f}")
