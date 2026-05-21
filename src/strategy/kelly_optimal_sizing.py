#!/usr/bin/env python3
"""
v8.04: Kelly-Optimal Position Sizing with Sigmoidal Scaling

Implements multivariate Kelly optimization with sigmoidal scaling law
(Tepelyan & Lam, arXiv:2604.24723, Apr 2026).

Core logic:
1. Estimate edge (expected excess return) and odds (variance) from rolling window
2. Compute optimal Kelly fraction: f* = \u03a3\u207b\u00b9 \u03bc (multivariate)
3. Apply sigmoidal scaling: f_optimal = sigmoid(k * edge/odds) * f_max
4. Enforce hard bounds from adaptive_sizing.py
5. Output allocation weights for SPY/GLD/TLT

No ML dependencies \u2014 pure numpy/scipy optimization.
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.paths import PROJECT_ROOT, DATA_DIR, PRICES_JSON

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_PATH = DATA_DIR / "kelly_sizing_state.json"

# Base allocation (matching adaptive_sizing.py)
BASE_ALLOCATION = {
    "SPY": 0.46,
    "GLD": 0.38,
    "TLT": 0.16,
}

# Hard bounds
HARD_BOUNDS = {
    "SPY": (0.36, 0.56),
    "GLD": (0.28, 0.48),
    "TLT": (0.06, 0.26),
    "IEF": (0.00, 0.10),
    "SHY": (0.00, 0.10),
}

ASSETS = ["SPY", "GLD", "TLT"]

# Default parameters
DEFAULT_LOOKBACK = 252       # Trading days for estimation
DEFAULT_SIGMOID_K = 1.5      # Sigmoid steepness parameter
DEFAULT_FRACTION_MAX = 0.8   # Maximum Kelly fraction (fraction of full Kelly)
DEFAULT_RISK_FREE = 0.042    # ~4.2% annual (current Fed rate)
DEFAULT_MIN_DAYS = 63        # Minimum days for estimation


@dataclass
class KellyFactors:
    """Factor readings for Kelly computation."""
    timestamp: str
    lookback_days: int
    spy_mean_return: float
    gld_mean_return: float
    tlt_mean_return: float
    spy_volatility: float
    gld_volatility: float
    tlt_volatility: float
    avg_edge_to_odds: float
    kelly_magnitude: float  # Norm of Kelly vector (indicates aggressiveness)
    sigmoid_scale: float    # Sigmoid scaling factor applied


@dataclass
class KellyDecision:
    """Complete Kelly sizing decision."""
    timestamp: str
    base_allocation: Dict[str, float]
    kelly_allocation: Dict[str, float]  # Raw Kelly fractions
    scaled_allocation: Dict[str, float]  # After sigmoid scaling
    final_allocation: Dict[str, float]   # After hard bounds
    asset_edges: Dict[str, float]        # Estimated edge per asset
    asset_odds: Dict[str, float]         # Estimated odds (variance) per asset
    factors: KellyFactors

    def to_dict(self) -> dict:
        return asdict(self)


# \u2500\u2500 Sigmoidal Scaling \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def sigmoid_scaling(edge_to_odds: float, k: float = DEFAULT_SIGMOID_K) -> float:
    """
    Sigmoidal scaling function for Kelly fraction.
    
    From Tepelyan & Lam (2026): optimal Kelly fraction follows sigmoidal
    function of edge-to-odds ratio.
    
    f_scaled = tanh(k * edge_to_odds) for edge_to_odds > 0
    f_scaled = 0 for edge_to_odds <= 0 (no edge -> no bet)
    """
    if not isinstance(edge_to_odds, (int, float)):
        return 0.0
    if edge_to_odds <= 0 or math.isnan(edge_to_odds) or math.isinf(edge_to_odds):
        return 0.0
    result = float(np.tanh(k * edge_to_odds))
    if math.isnan(result):
        return 0.0
    return result


def compute_edge_to_odds(mean_return: float, variance: float,
                         risk_free: float = DEFAULT_RISK_FREE) -> float:
    """
    Compute edge-to-odds ratio for a single asset.
    
    edge = expected excess return (annualized)
    odds = variance (annualized)
    edge_to_odds = edge / odds = (\u03bc - r_f) / \u03c3\u00b2
    """
    edge = mean_return - risk_free
    if variance <= 0 or edge <= 0:
        return 0.0
    return edge / variance


# \u2500\u2500 Multivariate Kelly \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def multivariate_kelly(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                       risk_free: float = DEFAULT_RISK_FREE) -> np.ndarray:
    """
    Compute multivariate Kelly optimal fractions.
    
    f* = \u03a3\u207b\u00b9 (\u03bc - r_f \u00b7 1)
    """
    n = len(mean_returns)
    excess = mean_returns - risk_free
    
    try:
        L = np.linalg.cholesky(cov_matrix)
        y = np.linalg.solve(L, excess)
        f = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        logger.debug("Cholesky failed, using pseudo-inverse for Kelly computation")
        cov_pinv = np.linalg.pinv(cov_matrix)
        f = cov_pinv @ excess
    
    return f


# \u2500\u2500 Main Optimizer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

class KellyOptimizer:
    """
    Kelly-optimal position sizing with sigmoidal scaling.
    
    Computes optimal allocation weights for SPY/GLD/TLT using:
    1. Rolling window estimation of mean returns and covariance
    2. Multivariate Kelly optimization
    3. Sigmoidal scaling to prevent overbetting
    4. Hard bounds enforcement
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.state_path = self.data_dir / "kelly_sizing_state.json"
        self.prices: Optional[Dict] = None
        
        # Configuration
        self.lookback: int = DEFAULT_LOOKBACK
        self.sigmoid_k: float = DEFAULT_SIGMOID_K
        self.fraction_max: float = DEFAULT_FRACTION_MAX
        self.risk_free: float = DEFAULT_RISK_FREE
        
        # State
        self.last_allocation: Dict[str, float] = dict(BASE_ALLOCATION)
        self.last_decision: Optional[KellyDecision] = None
        self._load_state()
    
    # \u2500\u2500 Data Loading \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def load_prices(self) -> Optional[Dict]:
        """Load price data from JSON."""
        prices_path = PRICES_JSON
        if not prices_path.exists():
            prices_path = self.data_dir / "prices.json"
        if not prices_path.exists():
            logger.warning("Prices file not found, using fallback")
            return None
        try:
            with open(prices_path) as f:
                self.prices = json.load(f)
            return self.prices
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            return None
    
    def get_series(self, symbol: str) -> Optional[np.ndarray]:
        """Get price series as numpy array."""
        if self.prices is None:
            self.load_prices()
        if self.prices is None or symbol not in self.prices:
            return None
        return np.array([p["p"] for p in self.prices[symbol]])
    
    # \u2500\u2500 Return Estimation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def estimate_returns_and_cov(self, prices: Dict[str, np.ndarray],
                                 lookback: int = DEFAULT_LOOKBACK
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate mean returns and covariance from price history.
        
        Returns:
            means: (3,) array of annualized mean returns (SPY, GLD, TLT)
            cov: (3,3) annualized covariance matrix
            vols: (3,) array of annualized volatilities
        """
        returns_list = []
        for symbol in ASSETS:
            if symbol not in prices or prices[symbol] is None:
                logger.warning(f"No price data for {symbol}")
                returns_list.append(np.zeros(lookback))
                continue
            series = prices[symbol]
            if len(series) < 2:
                returns_list.append(np.zeros(lookback))
                continue
            r = np.diff(series) / series[:-1]
            if len(r) > lookback:
                r = r[-lookback:]
            elif len(r) < lookback:
                r = np.pad(r, (lookback - len(r), 0), 'constant')
            returns_list.append(r)
        
        returns_array = np.column_stack(returns_list)  # (T, 3)
        
        n = returns_array.shape[0]
        means = np.mean(returns_array, axis=0) * 252
        cov = np.cov(returns_array, rowvar=False) * 252
        vols = np.sqrt(np.diag(cov))
        
        return means, cov, vols
    
    # \u2500\u2500 Core Computation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def _compute_single_allocation(self, lb: int) -> KellyDecision:
        """Compute allocation given a lookback length. Internal helper."""
        # Get price series
        prices_series = {}
        for symbol in ASSETS:
            series = self.get_series(symbol)
            if series is not None and len(series) > DEFAULT_MIN_DAYS:
                prices_series[symbol] = series
            else:
                prices_series[symbol] = np.ones(lb + 1)
        
        # Estimate returns
        means, cov, vols = self.estimate_returns_and_cov(prices_series, lb)
        
        # Compute multivariate Kelly fractions
        kelly_fractions = multivariate_kelly(means, cov, self.risk_free)
        
        # Compute edge-to-odds per asset
        edges = {}
        odds = {}
        edge_to_odds_list = []
        for i, symbol in enumerate(ASSETS):
            edge = float(means[i] - self.risk_free)
            odd = float(max(cov[i, i], 1e-10))
            edges[symbol] = edge
            odds[symbol] = odd
            e_to_o = edge / odd if odd > 0 else 0.0
            edge_to_odds_list.append(e_to_o)
        
        avg_edge_to_odds = float(np.mean([e for e in edge_to_odds_list if e > 0]))
        if math.isnan(avg_edge_to_odds) or avg_edge_to_odds <= 0:
            avg_edge_to_odds = 0.0
        
        sigmoid_scale = sigmoid_scaling(avg_edge_to_odds, self.sigmoid_k)
        kelly_norm = float(np.linalg.norm(kelly_fractions))
        
        raw_kelly = {ASSETS[i]: float(kelly_fractions[i]) for i in range(len(ASSETS))}
        scaled_kelly = {}
        for symbol in ASSETS:
            k = raw_kelly.get(symbol, 0.0)
            scaled = k * sigmoid_scale * self.fraction_max
            scaled = max(-0.5, min(2.0, scaled))
            if math.isnan(scaled):
                scaled = 0.0
            scaled_kelly[symbol] = scaled
        
        # Blend ratio with Kelly norm dampening
        # High Kelly norms (>5) indicate estimation instability — reduce blend
        norm_dampener = 1.0
        if kelly_norm > 10.0:
            norm_dampener = 0.25
        elif kelly_norm > 5.0:
            norm_dampener = 0.5
        
        blend_ratio = sigmoid_scale * self.fraction_max * norm_dampener
        blend_ratio = max(0.0, min(0.5, blend_ratio))
        
        allocation = {}
        for symbol in BASE_ALLOCATION:
            base = BASE_ALLOCATION[symbol]
            kelly_w = scaled_kelly.get(symbol, 0.0)
            allocation[symbol] = base * (1 - blend_ratio) + kelly_w * blend_ratio
        
        allocation = self._apply_bounds(allocation)
        
        factors = KellyFactors(
            timestamp=datetime.now().isoformat(),
            lookback_days=lb,
            spy_mean_return=float(means[0]),
            gld_mean_return=float(means[1]),
            tlt_mean_return=float(means[2]),
            spy_volatility=float(vols[0]),
            gld_volatility=float(vols[1]),
            tlt_volatility=float(vols[2]),
            avg_edge_to_odds=avg_edge_to_odds,
            kelly_magnitude=kelly_norm,
            sigmoid_scale=sigmoid_scale,
        )
        
        decision = KellyDecision(
            timestamp=factors.timestamp,
            base_allocation=dict(BASE_ALLOCATION),
            kelly_allocation=raw_kelly,
            scaled_allocation=scaled_kelly,
            final_allocation=allocation,
            asset_edges=edges,
            asset_odds=odds,
            factors=factors,
        )
        
        return decision
    
    def _fallback_allocation(self, lb: int = DEFAULT_LOOKBACK) -> KellyDecision:
        """Return base allocation when price data is unavailable."""
        factors = KellyFactors(
            timestamp=datetime.now().isoformat(),
            lookback_days=lb,
            spy_mean_return=self.risk_free,
            gld_mean_return=self.risk_free,
            tlt_mean_return=self.risk_free,
            spy_volatility=0.15,
            gld_volatility=0.15,
            tlt_volatility=0.15,
            avg_edge_to_odds=0.0,
            kelly_magnitude=0.0,
            sigmoid_scale=0.0,
        )
        return KellyDecision(
            timestamp=factors.timestamp,
            base_allocation=dict(BASE_ALLOCATION),
            kelly_allocation={s: 0.0 for s in ASSETS},
            scaled_allocation={s: 0.0 for s in ASSETS},
            final_allocation=dict(BASE_ALLOCATION),
            asset_edges={s: 0.0 for s in ASSETS},
            asset_odds={s: 0.0 for s in ASSETS},
            factors=factors,
        )
    
    def compute_allocation(self, lookback: Optional[int] = None) -> KellyDecision:
        """
        Compute Kelly-optimal allocation.
        
        1. Load price data
        2. Estimate returns and covariance
        3. Compute multivariate Kelly fractions
        4. Apply sigmoidal scaling
        5. Enforce hard bounds
        6. Persist state
        """
        lb = lookback if lookback is not None else self.lookback
        
        if self.prices is None:
            self.load_prices()
        if self.prices is None:
            logger.warning("No price data available - returning base allocation")
            decision = self._fallback_allocation(lb=lb)
            self._save_state(decision)
            return decision
        
        decision = self._compute_single_allocation(lb)
        
        self.last_allocation = decision.final_allocation
        self.last_decision = decision
        self._save_state(decision)
        
        return decision
    
    # \u2500\u2500 Bounds \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def _apply_bounds(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply hard bounds and normalize."""
        result = {}
        for asset, weight in allocation.items():
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            result[asset] = max(lo, min(hi, weight))
        
        for extra in ["IEF", "SHY"]:
            if extra not in result:
                result[extra] = 0.0
        
        total = sum(result.values())
        if total > 0:
            for asset in result:
                result[asset] /= total
        
        return result
    
    # \u2500\u2500 State Persistence \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def _load_state(self):
        """Load persisted state."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.last_allocation = state.get("last_allocation", dict(BASE_ALLOCATION))
            self.lookback = state.get("lookback", DEFAULT_LOOKBACK)
            self.sigmoid_k = state.get("sigmoid_k", DEFAULT_SIGMOID_K)
            self.fraction_max = state.get("fraction_max", DEFAULT_FRACTION_MAX)
        except Exception as e:
            logger.warning(f"Failed to load Kelly state: {e}")
    
    def _save_state(self, decision: KellyDecision):
        """Save state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_allocation": decision.final_allocation,
            "raw_kelly": decision.kelly_allocation,
            "scaled_kelly": decision.scaled_allocation,
            "asset_edges": decision.asset_edges,
            "asset_odds": decision.asset_odds,
            "avg_edge_to_odds": decision.factors.avg_edge_to_odds,
            "kelly_magnitude": decision.factors.kelly_magnitude,
            "sigmoid_scale": decision.factors.sigmoid_scale,
            "lookback": self.lookback,
            "sigmoid_k": self.sigmoid_k,
            "fraction_max": self.fraction_max,
            "risk_free": self.risk_free,
            "last_updated": decision.timestamp,
        }
        try:
            self.state_path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save Kelly state: {e}")
    
    # \u2500\u2500 Backtest Simulation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def simulate(self, years: int = 3) -> Dict:
        """
        Simple backtest: compare Kelly vs static allocation over past N years.
        
        Uses rolling windows: at each rebalance point, estimate Kelly
        from prior lookback days, then hold for the next month.
        """
        if self.prices is None:
            self.load_prices()
        if self.prices is None:
            return {"error": "No price data for simulation"}
        
        series_dict = {}
        min_len = float('inf')
        for symbol in ASSETS:
            s = self.get_series(symbol)
            if s is not None:
                series_dict[symbol] = s
                min_len = min(min_len, len(s))
        
        if min_len < self.lookback + 63:
            return {"error": f"Insufficient data: need {self.lookback + 63} days"}
        
        for symbol in series_dict:
            series_dict[symbol] = series_dict[symbol][-int(min_len):]
        
        n = int(min_len)
        rebalance_freq = 21
        start_idx = self.lookback
        
        static_returns = []
        kelly_returns = []
        allocation_history = []
        
        for i in range(start_idx, n - rebalance_freq, rebalance_freq):
            est_prices = {}
            for symbol in ASSETS:
                est_prices[symbol] = series_dict[symbol][i - self.lookback:i + 1]
            
            means, cov, vols = self.estimate_returns_and_cov(est_prices, self.lookback)
            kelly_f = multivariate_kelly(means, cov, self.risk_free)
            
            edge_to_odds_list = []
            for j, sym in enumerate(ASSETS):
                e_to_o = compute_edge_to_odds(means[j], cov[j, j], self.risk_free)
                edge_to_odds_list.append(e_to_o)
            avg_e = float(np.mean([e for e in edge_to_odds_list if e > 0]))
            if math.isnan(avg_e) or avg_e <= 0:
                avg_e = 0.0
            sig_s = sigmoid_scaling(avg_e, self.sigmoid_k)
            
            # Norm dampening for estimation stability
            kelly_norm = float(np.linalg.norm(kelly_f))
            norm_dampener = 1.0
            if kelly_norm > 10.0:
                norm_dampener = 0.25
            elif kelly_norm > 5.0:
                norm_dampener = 0.5
            
            blend_ratio = sig_s * self.fraction_max * norm_dampener
            blend_ratio = max(0.0, min(0.5, blend_ratio))
            
            kelly_alloc = {}
            for j, symbol in enumerate(ASSETS):
                val = float(kelly_f[j]) * sig_s * self.fraction_max
                kelly_alloc[symbol] = 0.0 if math.isnan(val) else max(-0.5, min(2.0, val))
            
            alloc = {}
            for symbol in BASE_ALLOCATION:
                base = BASE_ALLOCATION[symbol]
                kelly_w = kelly_alloc.get(symbol, 0.0)
                alloc[symbol] = base * (1 - blend_ratio) + kelly_w * blend_ratio
            
            alloc = self._apply_bounds(alloc)
            
            for j in range(rebalance_freq):
                idx = i + j
                if idx >= n - 1:
                    break
                static_r = sum(BASE_ALLOCATION[sym] *
                              (series_dict[sym][idx + 1] / series_dict[sym][idx] - 1)
                              for sym in ASSETS)
                kelly_r = sum(alloc.get(sym, 0.0) *
                              (series_dict[sym][idx + 1] / series_dict[sym][idx] - 1)
                              for sym in ASSETS)
                static_returns.append(static_r)
                kelly_returns.append(kelly_r)
            
            allocation_history.append(alloc)
        
        static_ret_arr = np.array(static_returns)
        kelly_ret_arr = np.array(kelly_returns)
        
        def compute_metrics(returns: np.ndarray) -> Dict:
            if len(returns) < 10:
                return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}
            ann_ret = float(np.mean(returns) * 252)
            ann_vol = float(np.std(returns) * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            cum = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cum)
            dd = cum / peak - 1
            max_dd = float(np.min(dd))
            return {
                "sharpe": round(sharpe, 4),
                "cagr": round(ann_ret, 4),
                "volatility": round(ann_vol, 4),
                "max_dd": round(max_dd, 4),
            }
        
        static_metrics = compute_metrics(static_ret_arr)
        kelly_metrics = compute_metrics(kelly_ret_arr)
        
        total_turnover = 0.0
        for i in range(1, len(allocation_history)):
            prev = allocation_history[i - 1]
            curr = allocation_history[i]
            turnover = sum(abs(curr.get(sym, 0) - prev.get(sym, 0)) for sym in ASSETS)
            total_turnover += turnover
        avg_turnover = total_turnover / max(1, len(allocation_history) - 1)
        
        n_rebalances = len(allocation_history)
        
        return {
            "static": static_metrics,
            "kelly": kelly_metrics,
            "sharpe_delta": round(kelly_metrics.get("sharpe", 0) - static_metrics.get("sharpe", 0), 4),
            "dd_delta": round(kelly_metrics.get("max_dd", 0) - static_metrics.get("max_dd", 0), 4),
            "n_rebalances": n_rebalances,
            "avg_turnover": round(avg_turnover, 4),
            "avg_edge_to_odds": round(float(np.mean(
                [v.get("avg_edge_to_odds", 0) for v in allocation_history[:5]])), 4) if allocation_history else 0,
            "years_simulated": round(n * rebalance_freq / 252, 1),
        }
    
    # \u2500\u2500 CLI \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    
    def print_status(self, decision: Optional[KellyDecision] = None):
        """Print formatted Kelly status."""
        if decision is None:
            decision = self.compute_allocation()
        
        print("=" * 70)
        print("  KELLY-OPTIMAL POSITION SIZING v8.04")
        print("  (Tepelyan & Lam, arXiv:2604.24723)")
        print("=" * 70)
        print(f"  Timestamp:   {decision.timestamp[:19]}")
        print()
        
        print("  Estimated Returns (Ann.):")
        for sym in ASSETS:
            edge_val = decision.asset_edges.get(sym, 0.0) or 0.0
            odd_val = decision.asset_odds.get(sym, 0.0) or 0.0
            print(f"    {sym}: mean={edge_val + self.risk_free:.2%}, "
                  f"edge={edge_val:+.2%}, "
                  f"odds={odd_val:.4f}")
        print()
        
        print(f"  Avg Edge/Odds:  {decision.factors.avg_edge_to_odds:.4f}")
        print(f"  Sigmoid Scale:  {decision.factors.sigmoid_scale:.4f}")
        print(f"  Kelly Norm:     {decision.factors.kelly_magnitude:.4f}")
        print(f"  Risk-Free Rate: {self.risk_free:.2%}")
        print()
        
        print("  Allocation:")
        print(f"    {'Asset':6s} {'Base':>8s} {'Raw Kelly':>10s} {'Scaled':>8s} {'Final':>8s}")
        print(f"    {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for sym in ASSETS:
            base = decision.base_allocation.get(sym, 0.0)
            raw = decision.kelly_allocation.get(sym, 0.0)
            scaled = decision.scaled_allocation.get(sym, 0.0)
            final = decision.final_allocation.get(sym, 0.0)
            print(f"    {sym:6s} {base:>7.1%} {raw:>+9.2%} {scaled:>+7.2%} {final:>7.1%}")
        print()
        
        print("  Volatility (Ann.):")
        print(f"    SPY: {decision.factors.spy_volatility:.1%}")
        print(f"    GLD: {decision.factors.gld_volatility:.1%}")
        print(f"    TLT: {decision.factors.tlt_volatility:.1%}")
        print()
        
        print("  Interpretation:")
        sig = decision.factors.sigmoid_scale
        if sig < 0.1:
            print("    Low conviction - near base allocation (weak edge)")
        elif sig < 0.3:
            print("    Moderate conviction - modest Kelly tilt")
        elif sig < 0.5:
            print("    Strong conviction - significant Kelly tilt")
        else:
            print("    High conviction - full Kelly allocation")


# \u2500\u2500 CLI Entry Point \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def main():
    """CLI entry point."""
    import sys
    
    optimizer = KellyOptimizer()
    
    if len(sys.argv) < 2 or sys.argv[1] == "adjust":
        decision = optimizer.compute_allocation()
        optimizer.print_status(decision)
    
    elif sys.argv[1] == "status":
        if STATE_PATH.exists():
            print(json.dumps(json.loads(STATE_PATH.read_text()), indent=2))
        else:
            print("No state file found. Run 'adjust' first.")
    
    elif sys.argv[1] == "simulate":
        result = optimizer.simulate()
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print("=" * 70)
        print("  KELLY SIZING BACKTEST SIMULATION")
        print("=" * 70)
        print(f"  Years:          {result.get('years_simulated', 'N/A')}")
        print(f"  Rebalances:     {result.get('n_rebalances', 0)}")
        print(f"  Avg Turnover:   {result.get('avg_turnover', 0):.2%}")
        print()
        print(f"  {'Metric':20s} {'Static':>10s} {'Kelly':>10s} {'Delta':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        static = result.get("static", {})
        kelly = result.get("kelly", {})
        for metric in ["sharpe", "cagr", "volatility", "max_dd"]:
            s_val = static.get(metric, 0)
            k_val = kelly.get(metric, 0)
            if metric == "sharpe":
                delta = result.get("sharpe_delta", k_val - s_val)
            elif metric == "max_dd":
                delta = result.get("dd_delta", k_val - s_val)
            else:
                delta = k_val - s_val
            print(f"  {metric.title():20s} {s_val:>10.4f} {k_val:>10.4f} {delta:>+10.4f}")
        print()
    else:
        print("Usage: python -m src.strategy.kelly_optimal_sizing [adjust|status|simulate]")


if __name__ == "__main__":
    main()
