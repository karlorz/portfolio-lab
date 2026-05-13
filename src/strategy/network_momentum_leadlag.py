#!/usr/bin/env python3
"""
Portfolio-Lab v2.58: Network Momentum Lead-Lag Overlay

Imperial College "Follow the Leader" implementation based on:
- Li & Ferreira (2025): "Follow the Leader: Network Momentum in Cross-Asset Momentum"
- arXiv:2501.07135 (January 2025)

Network Momentum Strategy:
    1. Lead-lag detection via Dynamic Time Warping (DTW) and Lévy area signatures
    2. Graph learning: Sparse adjacency matrix via convex optimization
    3. Ensemble approach: Multiple lookback windows (22, 44, 66, 88, 110, 132 days)
    4. Aggregate momentum via learned lead-lag relationships

Key Insight:
    Assets don't move independently - they lead/lag each other via information
    spillover, funding flows, and cross-asset arbitrage. Network momentum
    captures these predictive relationships vs standalone TSMOM.

Performance (Paper Results):
    - Sharpe improvement: +29-33% vs MACD baseline
    - Sortino improvement: +33%
    - Better downside control, stronger positive skewness

Implementation for SPY/GLD/TLT:
    - Cross-asset lead-lag typically: SPY leads GLD/TLT in risk-on
    - GLD leads in inflation surprises, geopolitical stress
    - TLT leads during flight-to-quality
    - Network aggregation should improve timing of regime shifts

Expected for Portfolio-Lab:
    - Baseline Sharpe: 0.98 (Risk Parity v2.57)
    - Network Momentum Target: 1.15-1.25 (+0.17 to +0.27)

Usage:
    python -m src.strategy.network_momentum_leadlag compute --ticker SPY --window 44
    python -m src.strategy.network_momentum_leadlag backtest --ensemble
    python -m src.strategy.network_momentum_leadlag status
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
from itertools import combinations

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Constants
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

# Network Momentum Parameters (from paper: ensemble across these windows)
LOOKBACK_WINDOWS = [22, 44, 66, 88, 110, 132]  # days
DEFAULT_WINDOW = 66  # ~3 months - balanced choice

# Lead-lag detection parameters
DTW_RADIUS = 5  # Sakoe-Chiba constraint radius
LEVY_LAGS = [1, 5, 10, 21]  # Levy area lags for lead-lag detection

# Graph learning parameters
GRAPH_SPARSITY_ALPHA = 0.01  # Sparsity regularization
GRAPH_SMOOTHNESS_BETA = 0.01  # Smoothness regularization

# Allocation overlay parameters
MAX_DEVIATION = 0.15  # ±15% max deviation from base
MIN_WEIGHT = 0.05

ASSETS = ['SPY', 'GLD', 'TLT', 'CASH']
DEFAULT_BASE_ALLOCATION = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}


@dataclass
class LeadLagMatrix:
    """Lead-lag relationship matrix for asset pairs."""
    timestamp: str
    window: int
    
    # Lead-lag scores (positive = row leads column, negative = column leads row)
    leadlag_matrix: Dict[Tuple[str, str], float]  # (leader, follower): strength
    
    # DTW distances (lower = more similar paths)
    dtw_distances: Dict[Tuple[str, str], float]
    
    # Lévy area signatures
    levy_areas: Dict[Tuple[str, str], float]
    
    # Derived adjacency matrix (learned graph structure)
    adjacency: Dict[Tuple[str, str], float]
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'window': self.window,
            'leadlag_matrix': {f"{k[0]}->{k[1]}": v for k, v in self.leadlag_matrix.items()},
            'dtw_distances': {f"{k[0]}-{k[1]}": v for k, v in self.dtw_distances.items()},
            'levy_areas': {f"{k[0]}-{k[1]}": v for k, v in self.levy_areas.items()},
            'adjacency': {f"{k[0]}->{k[1]}": v for k, v in self.adjacency.items()},
        }


@dataclass
class WindowMomentumSignal:
    """Momentum signal for a specific lookback window."""
    ticker: str
    window: int
    timestamp: str
    
    # Standalone momentum
    momentum_return: float
    signal: int  # -1, 0, +1
    
    # Network-adjusted momentum (incorporates lead-lag)
    network_momentum: float  # Weighted average of leaders' momentum
    network_adjustment: float  # Delta from standalone
    
    # Allocation impact
    base_weight: float
    target_weight: float
    adjustment: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnsembleNetworkSignal:
    """Ensemble signal across multiple lookback windows."""
    ticker: str
    timestamp: str
    
    # Per-window signals
    window_signals: Dict[int, WindowMomentumSignal]
    
    # Ensemble aggregation (equal weight across windows per paper)
    ensemble_momentum: float
    ensemble_signal: int
    ensemble_confidence: float  # Agreement across windows
    
    # Allocation
    base_weight: float
    adjustment: float
    target_weight: float
    
    # Cross-asset network metrics
    leadership_score: float  # How often this asset leads others
    followership_score: float  # How often this asset follows others
    network_centrality: float  # Eigenvector centrality in lead-lag graph
    
    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'window_signals': {str(k): v.to_dict() for k, v in self.window_signals.items()},
            'ensemble_momentum': self.ensemble_momentum,
            'ensemble_signal': self.ensemble_signal,
            'ensemble_confidence': self.ensemble_confidence,
            'base_weight': self.base_weight,
            'adjustment': self.adjustment,
            'target_weight': self.target_weight,
            'leadership_score': self.leadership_score,
            'followership_score': self.followership_score,
            'network_centrality': self.network_centrality,
        }


@dataclass
class NetworkMomentumPortfolio:
    """Network momentum-adjusted portfolio allocation."""
    timestamp: str
    base_allocation: Dict[str, float]
    network_adjustments: Dict[str, float]
    target_allocation: Dict[str, float]
    
    # Lead-lag analysis
    leadlag_matrix: LeadLagMatrix
    
    # Per-asset signals
    ensemble_signals: Dict[str, EnsembleNetworkSignal]
    
    # Network topology metrics
    dominant_leader: str  # Asset with highest leadership score
    dominant_follower: str  # Asset with highest followership score
    network_efficiency: float  # How well-connected the network is
    
    overall_confidence: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'base_allocation': self.base_allocation,
            'network_adjustments': self.network_adjustments,
            'target_allocation': self.target_allocation,
            'leadlag_matrix': self.leadlag_matrix.to_dict(),
            'ensemble_signals': {k: v.to_dict() for k, v in self.ensemble_signals.items()},
            'dominant_leader': self.dominant_leader,
            'dominant_follower': self.dominant_follower,
            'network_efficiency': self.network_efficiency,
            'overall_confidence': self.overall_confidence,
        }


class NetworkMomentumLeadLag:
    """
    Network momentum overlay with lead-lag detection.
    
    Implements Imperial College "Follow the Leader" methodology:
    - DTW for lead-lag alignment detection
    - Lévy area signatures for directional lead-lag scoring
    - Graph learning for optimal momentum aggregation
    - Ensemble across multiple lookback windows
    """
    
    def __init__(
        self,
        prices_path: Path = PRICES_PATH,
        db_path: Path = DB_PATH,
        lookback_windows: List[int] = None,
        max_deviation: float = MAX_DEVIATION
    ):
        self.prices_path = prices_path
        self.db_path = db_path
        self.lookback_windows = lookback_windows or LOOKBACK_WINDOWS
        self.max_deviation = max_deviation
        self._prices_df: Optional[pd.DataFrame] = None
    
    def _load_prices(self) -> pd.DataFrame:
        """Load price data from JSON."""
        if self._prices_df is not None:
            return self._prices_df
        
        with open(self.prices_path, 'r') as f:
            data = json.load(f)
        
        records = []
        for symbol, entries in data.items():
            for entry in entries:
                records.append({
                    'date': entry['d'],
                    'ticker': symbol,
                    'price': entry['p']
                })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot(index='date', columns='ticker', values='price')
        df = df.sort_index()
        
        self._prices_df = df
        return df
    
    def _simple_dtw_distance(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        radius: int = DTW_RADIUS
    ) -> float:
        """
        Simplified DTW distance with Sakoe-Chiba constraint.
        
        DTW finds optimal alignment between two time series,
        accounting for temporal shifts (lead-lag).
        """
        n, m = len(series1), len(series2)
        
        # Normalize series
        s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-8)
        s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-8)
        
        # DTW matrix with window constraint
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0
        
        for i in range(1, n + 1):
            # Sakoe-Chiba window
            start_j = max(1, i - radius)
            end_j = min(m + 1, i + radius + 1)
            
            for j in range(start_j, end_j):
                cost = abs(s1[i-1] - s2[j-1])
                dtw[i, j] = cost + min(
                    dtw[i-1, j],      # insertion
                    dtw[i, j-1],      # deletion
                    dtw[i-1, j-1]     # match
                )
        
        return dtw[n, m]
    
    def _compute_levy_area_signature(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        lags: List[int] = None
    ) -> float:
        """
        Compute Lévy area signature for lead-lag detection.
        
        Lévy area measures the area between two paths - skew-symmetric
        indicator of which series leads the other.
        
        Positive = series1 leads series2
        Negative = series2 leads series1
        """
        if lags is None:
            lags = LEVY_LAGS
        
        # Cumulative returns (path)
        cumsum1 = np.cumsum(series1)
        cumsum2 = np.cumsum(series2)
        
        # Compute Lévy area (simplified - trapezoidal integration)
        # Lévy area = 1/2 * ∫(x dy - y dx)
        n = len(series1)
        levy_areas = []
        
        for lag in lags:
            if n <= lag:
                continue
            
            # Lagged series for lead-lag detection
            if lag > 0:
                x = cumsum1[:-lag] if lag > 0 else cumsum1
                y = cumsum2[lag:] if lag > 0 else cumsum2
            else:
                x = cumsum1[-lag:]
                y = cumsum2[:lag]
            
            # Ensure equal length
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]
            
            # Lévy area via trapezoidal rule
            levy = 0.5 * np.sum(x[:-1] * np.diff(y) - y[:-1] * np.diff(x))
            levy_areas.append(levy)
        
        # Average across lags
        return np.mean(levy_areas) if levy_areas else 0.0
    
    def _learn_adjacency_matrix(
        self,
        leadlag_scores: Dict[Tuple[str, str], float],
        assets: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Learn sparse adjacency matrix from lead-lag scores.
        
        Simplified implementation - full version uses convex optimization:
        minimize: tr(X^T(D-A)X) - alpha*1^T*log(A*1) + beta*||A||_F^2
        
        Where:
        - D is degree matrix
        - X is feature matrix (momentum scores)
        - alpha controls sparsity
        - beta controls smoothness
        """
        adjacency = {}
        
        # Normalize lead-lag scores to [0, 1] range
        scores = list(leadlag_scores.values())
        if not scores:
            return adjacency
        
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        for (leader, follower), score in leadlag_scores.items():
            # Normalize and apply sparsity threshold
            norm_score = (score - min_score) / score_range
            
            # Apply sparsity (only keep strong connections)
            if norm_score > 0.3:  # Sparsity threshold
                adjacency[(leader, follower)] = norm_score
            else:
                adjacency[(leader, follower)] = 0.0
        
        return adjacency
    
    def compute_leadlag_matrix(
        self,
        window: int,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[LeadLagMatrix]:
        """
        Compute lead-lag matrix for all asset pairs.
        
        Returns matrix where entry (i,j) indicates how much asset i
        leads asset j (positive) or follows it (negative).
        """
        if prices_df is None:
            prices_df = self._load_prices()
        
        assets = ['SPY', 'GLD', 'TLT']
        
        # Get returns for window
        end_idx = len(prices_df)
        start_idx = max(0, end_idx - window - 1)
        
        if end_idx - start_idx < window * 0.8:
            return None
        
        window_prices = prices_df.iloc[start_idx:end_idx]
        
        # Calculate returns
        returns = {}
        for asset in assets:
            if asset in window_prices.columns:
                price_series = window_prices[asset].dropna()
                if len(price_series) > 10:
                    returns[asset] = price_series.pct_change().dropna().values
        
        if len(returns) < 2:
            return None
        
        # Compute lead-lag for all pairs
        leadlag_matrix = {}
        dtw_distances = {}
        levy_areas = {}
        
        for asset1, asset2 in combinations(assets, 2):
            if asset1 not in returns or asset2 not in returns:
                continue
            
            # Match lengths
            r1, r2 = returns[asset1], returns[asset2]
            min_len = min(len(r1), len(r2))
            r1, r2 = r1[-min_len:], r2[-min_len:]
            
            # DTW distance
            dtw_dist = self._simple_dtw_distance(r1, r2)
            dtw_distances[(asset1, asset2)] = dtw_dist
            dtw_distances[(asset2, asset1)] = dtw_dist
            
            # Lévy area (directional lead-lag)
            levy = self._compute_levy_area_signature(r1, r2)
            
            # Positive levy = asset1 leads asset2
            # Negative levy = asset2 leads asset1
            levy_areas[(asset1, asset2)] = levy
            levy_areas[(asset2, asset1)] = -levy
            
            # Combined lead-lag score (normalized)
            leadlag_score = levy / (dtw_dist + 1e-6)  # Higher score = stronger lead
            leadlag_matrix[(asset1, asset2)] = leadlag_score
            leadlag_matrix[(asset2, asset1)] = -leadlag_score
        
        # Learn adjacency matrix
        adjacency = self._learn_adjacency_matrix(leadlag_matrix, assets)
        
        current_date = prices_df.index[-1]
        
        return LeadLagMatrix(
            timestamp=current_date.isoformat(),
            window=window,
            leadlag_matrix=leadlag_matrix,
            dtw_distances=dtw_distances,
            levy_areas=levy_areas,
            adjacency=adjacency
        )
    
    def compute_window_signal(
        self,
        ticker: str,
        window: int,
        base_weight: float,
        leadlag_matrix: LeadLagMatrix,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[WindowMomentumSignal]:
        """Compute momentum signal for a specific lookback window."""
        if prices_df is None:
            prices_df = self._load_prices()
        
        if ticker not in prices_df.columns:
            return None
        
        end_idx = len(prices_df)
        start_idx = max(0, end_idx - window - 1)
        
        if end_idx - start_idx < window * 0.5:
            return None
        
        # Calculate standalone momentum
        prices = prices_df[ticker].iloc[start_idx:end_idx].dropna()
        if len(prices) < 10:
            return None
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        momentum_return = (end_price / start_price) - 1
        signal = int(np.sign(momentum_return)) if momentum_return != 0 else 0
        
        # Network-adjusted momentum
        # Weight by adjacency strength from leading assets
        network_contributions = []
        total_weight = 0.0
        
        assets = ['SPY', 'GLD', 'TLT']
        for other_asset in assets:
            if other_asset == ticker or other_asset not in prices_df.columns:
                continue
            
            # Get adjacency strength (how much other_asset leads ticker)
            adj_strength = leadlag_matrix.adjacency.get((other_asset, ticker), 0.0)
            
            if adj_strength > 0.1:  # Significant connection
                # Get other asset's momentum
                other_prices = prices_df[other_asset].iloc[start_idx:end_idx].dropna()
                if len(other_prices) >= 10:
                    other_momentum = (other_prices.iloc[-1] / other_prices.iloc[0]) - 1
                    network_contributions.append(adj_strength * other_momentum)
                    total_weight += adj_strength
        
        # Network momentum = weighted average of leaders' momentum
        if total_weight > 0 and network_contributions:
            network_momentum = np.sum(network_contributions) / total_weight
        else:
            network_momentum = momentum_return
        
        network_adjustment = network_momentum - momentum_return
        
        # Allocation adjustment based on network momentum
        adjustment = np.clip(
            network_momentum * 0.5,  # Scale factor
            -self.max_deviation,
            self.max_deviation
        )
        target_weight = base_weight + adjustment
        target_weight = np.clip(target_weight, MIN_WEIGHT, 1.0)
        
        current_date = prices_df.index[-1]
        
        return WindowMomentumSignal(
            ticker=ticker,
            window=window,
            timestamp=current_date.isoformat(),
            momentum_return=momentum_return,
            signal=signal,
            network_momentum=network_momentum,
            network_adjustment=network_adjustment,
            base_weight=base_weight,
            target_weight=target_weight,
            adjustment=adjustment
        )
    
    def compute_ensemble_signal(
        self,
        ticker: str,
        base_weight: float,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[EnsembleNetworkSignal]:
        """
        Compute ensemble signal across all lookback windows.
        
        Per paper: Ensemble across 22, 44, 66, 88, 110, 132 days
        with equal weight aggregation.
        """
        if prices_df is None:
            prices_df = self._load_prices()
        
        # Compute lead-lag matrix for default window
        leadlag_matrix = self.compute_leadlag_matrix(DEFAULT_WINDOW, prices_df)
        if not leadlag_matrix:
            return None
        
        # Compute signals for each window
        window_signals = {}
        for window in self.lookback_windows:
            # Recompute lead-lag for different windows
            window_ll = self.compute_leadlag_matrix(window, prices_df)
            if window_ll:
                signal = self.compute_window_signal(
                    ticker, window, base_weight, window_ll, prices_df
                )
                if signal:
                    window_signals[window] = signal
        
        if not window_signals:
            return None
        
        # Equal-weight ensemble (per paper)
        ensemble_momentum = np.mean([s.network_momentum for s in window_signals.values()])
        ensemble_signal = int(np.sign(ensemble_momentum)) if ensemble_momentum != 0 else 0
        
        # Confidence = agreement across windows
        signals = [s.signal for s in window_signals.values()]
        if all(s == signals[0] for s in signals):
            ensemble_confidence = 1.0
        elif sum(signals) == 0:
            ensemble_confidence = 0.0
        else:
            ensemble_confidence = abs(sum(signals)) / len(signals)
        
        # Allocation based on ensemble momentum
        adjustment = np.clip(
            ensemble_momentum * 0.5,
            -self.max_deviation,
            self.max_deviation
        )
        target_weight = base_weight + adjustment
        target_weight = np.clip(target_weight, MIN_WEIGHT, 1.0)
        
        # Network centrality metrics
        assets = ['SPY', 'GLD', 'TLT']
        leadership_score = 0.0
        followership_score = 0.0
        
        for other in assets:
            if other == ticker:
                continue
            # Leadership: ticker leads other
            lead_strength = leadlag_matrix.adjacency.get((ticker, other), 0.0)
            leadership_score += lead_strength
            
            # Followership: other leads ticker
            follow_strength = leadlag_matrix.adjacency.get((other, ticker), 0.0)
            followership_score += follow_strength
        
        # Normalize
        leadership_score /= max(1, len(assets) - 1)
        followership_score /= max(1, len(assets) - 1)
        
        # Centrality = how connected (both leading and following)
        network_centrality = (leadership_score + followership_score) / 2
        
        current_date = prices_df.index[-1]
        
        return EnsembleNetworkSignal(
            ticker=ticker,
            timestamp=current_date.isoformat(),
            window_signals=window_signals,
            ensemble_momentum=ensemble_momentum,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            base_weight=base_weight,
            adjustment=adjustment,
            target_weight=target_weight,
            leadership_score=leadership_score,
            followership_score=followership_score,
            network_centrality=network_centrality
        )
    
    def get_current_recommendation(
        self,
        base_allocation: Dict[str, float]
    ) -> Optional[NetworkMomentumPortfolio]:
        """Get current network momentum portfolio recommendation."""
        prices_df = self._load_prices()
        current_date = prices_df.index[-1]
        
        # Compute lead-lag matrix
        leadlag_matrix = self.compute_leadlag_matrix(DEFAULT_WINDOW, prices_df)
        if not leadlag_matrix:
            return None
        
        # Compute ensemble signals for each asset
        ensemble_signals = {}
        network_adjustments = {}
        target_allocation = {'CASH': 0.0}
        
        for ticker, base_weight in base_allocation.items():
            if ticker == 'CASH':
                continue
            
            signal = self.compute_ensemble_signal(ticker, base_weight, prices_df)
            if signal:
                ensemble_signals[ticker] = signal
                network_adjustments[ticker] = signal.adjustment
                target_allocation[ticker] = signal.target_weight
            else:
                target_allocation[ticker] = base_weight
                network_adjustments[ticker] = 0.0
        
        # Normalize weights
        total_weight = sum(w for k, w in target_allocation.items() if k != 'CASH')
        if total_weight > 0:
            for ticker in target_allocation:
                if ticker != 'CASH':
                    target_allocation[ticker] /= total_weight
        
        # Identify dominant leader/follower
        leadership_scores = {t: s.leadership_score for t, s in ensemble_signals.items()}
        followership_scores = {t: s.followership_score for t, s in ensemble_signals.items()}
        
        dominant_leader = max(leadership_scores, key=leadership_scores.get) if leadership_scores else 'SPY'
        dominant_follower = max(followership_scores, key=followership_scores.get) if followership_scores else 'TLT'
        
        # Network efficiency = average centrality
        centralities = [s.network_centrality for s in ensemble_signals.values()]
        network_efficiency = np.mean(centralities) if centralities else 0.0
        
        # Overall confidence
        overall_confidence = np.mean([s.ensemble_confidence for s in ensemble_signals.values()]) if ensemble_signals else 0.0
        
        return NetworkMomentumPortfolio(
            timestamp=current_date.isoformat(),
            base_allocation=base_allocation,
            network_adjustments=network_adjustments,
            target_allocation=target_allocation,
            leadlag_matrix=leadlag_matrix,
            ensemble_signals=ensemble_signals,
            dominant_leader=dominant_leader,
            dominant_follower=dominant_follower,
            network_efficiency=network_efficiency,
            overall_confidence=overall_confidence
        )


class NetworkMomentumBacktester:
    """Backtester for network momentum lead-lag strategy."""
    
    def __init__(
        self,
        base_allocation: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: int = 21
    ):
        self.base_allocation = base_allocation
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.rebalance_freq = rebalance_freq
        self.network_momentum = NetworkMomentumLeadLag()
        self.prices_df = self.network_momentum._load_prices()
    
    def run_backtest(self) -> Dict:
        """Run full historical backtest."""
        prices = self.prices_df.copy()
        
        if self.start_date:
            prices = prices[prices.index >= self.start_date]
        if self.end_date:
            prices = prices[prices.index <= self.end_date]
        
        min_history = max(LOOKBACK_WINDOWS) + 50
        if len(prices) < min_history:
            return {'error': f'Insufficient data: {len(prices)} days < {min_history} required'}
        
        portfolio_value = 100000.0
        current_weights = self.base_allocation.copy()
        
        daily_values = []
        rebalance_dates = []
        
        for i in range(min_history, len(prices)):
            current_date = prices.index[i]
            history = prices.iloc[:i+1]
            
            # Monthly rebalancing
            if (i - min_history) % self.rebalance_freq == 0:
                self.network_momentum._prices_df = history
                
                recommendation = self.network_momentum.get_current_recommendation(
                    self.base_allocation
                )
                
                if recommendation:
                    current_weights = recommendation.target_allocation
                    rebalance_dates.append({
                        'date': current_date.isoformat(),
                        'weights': current_weights.copy(),
                        'dominant_leader': recommendation.dominant_leader,
                        'network_efficiency': recommendation.network_efficiency
                    })
            
            # Daily return
            daily_return = 0.0
            for ticker, weight in current_weights.items():
                if ticker == 'CASH' or ticker not in prices.columns:
                    continue
                if i > 0:
                    ticker_return = float(prices[ticker].iloc[i]) / float(prices[ticker].iloc[i-1]) - 1
                    daily_return += weight * ticker_return
            
            new_value = portfolio_value * (1 + daily_return)
            if np.isnan(new_value) or np.isinf(new_value):
                new_value = portfolio_value
            portfolio_value = new_value
            
            daily_values.append({
                'date': current_date.isoformat(),
                'value': portfolio_value,
                'return': daily_return
            })
        
        # Calculate metrics
        df_values = pd.DataFrame(daily_values)
        df_values['date'] = pd.to_datetime(df_values['date'])
        df_values.set_index('date', inplace=True)
        
        returns = df_values['return'].dropna()
        
        start_val = float(df_values['value'].iloc[0])
        end_val = float(df_values['value'].iloc[-1])
        years = len(df_values) / 252
        
        total_return = (end_val / start_val) - 1 if start_val > 0 else 0
        cagr = ((end_val / start_val) ** (1/years)) - 1 if start_val > 0 and years > 0 else 0
        volatility = float(returns.std()) * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Crisis analysis
        crisis_periods = {
            '2008': ('2008-01-01', '2008-12-31'),
            '2020': ('2020-02-01', '2020-05-31'),
            '2022': ('2022-01-01', '2022-12-31'),
        }
        
        crisis_returns = {}
        for crisis, (start, end) in crisis_periods.items():
            try:
                period_df = df_values.loc[start:end]
                if not period_df.empty:
                    crisis_return = period_df['value'].iloc[-1] / period_df['value'].iloc[0] - 1
                    crisis_returns[crisis] = crisis_return
            except:
                crisis_returns[crisis] = None
        
        # Baseline comparison
        baseline_values = [100000.0]
        for i in range(min_history, len(prices)):
            daily_return = 0.0
            for ticker, weight in self.base_allocation.items():
                if ticker == 'CASH' or ticker not in prices.columns:
                    continue
                if i > 0:
                    ticker_return = float(prices[ticker].iloc[i]) / float(prices[ticker].iloc[i-1]) - 1
                    daily_return += weight * ticker_return
            new_value = baseline_values[-1] * (1 + daily_return)
            if np.isnan(new_value) or np.isinf(new_value):
                new_value = baseline_values[-1]
            baseline_values.append(new_value)
        
        baseline_cagr = ((baseline_values[-1] / baseline_values[0]) ** (1/years)) - 1 if baseline_values[0] > 0 and years > 0 else 0
        baseline_returns = pd.Series(baseline_values).pct_change().dropna()
        baseline_vol = float(baseline_returns.std()) * np.sqrt(252)
        baseline_sharpe = baseline_cagr / baseline_vol if baseline_vol > 0 else 0
        
        return {
            'strategy': 'Network Momentum Lead-Lag v2.58 (arXiv:2501.07135)',
            'start_date': prices.index[min_history].isoformat(),
            'end_date': prices.index[-1].isoformat(),
            'trading_days': len(df_values),
            'rebalances': len(rebalance_dates),
            'start_value': 100000,
            'end_value': portfolio_value,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'baseline_cagr': baseline_cagr,
            'baseline_sharpe': baseline_sharpe,
            'excess_return': cagr - baseline_cagr,
            'sharpe_improvement': sharpe - baseline_sharpe,
            'crisis_2008_return': crisis_returns.get('2008'),
            'crisis_2020_return': crisis_returns.get('2020'),
            'crisis_2022_return': crisis_returns.get('2022'),
            'lookback_windows': LOOKBACK_WINDOWS,
            'dominant_leader_history': [r['dominant_leader'] for r in rebalance_dates[-12:]] if rebalance_dates else [],
        }


def main():
    parser = argparse.ArgumentParser(
        description='Network Momentum Lead-Lag v2.58'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Compute command
    compute_parser = subparsers.add_parser('compute', help='Compute lead-lag for ticker')
    compute_parser.add_argument('--ticker', required=True)
    compute_parser.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    backtest_parser.add_argument('--output', help='Output JSON')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Get current recommendation')
    live_parser.add_argument('--output', help='Output JSON')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        print("Network Momentum Lead-Lag v2.58 - Status")
        print("=" * 50)
        print("Source: Li & Ferreira (2025), arXiv:2501.07135")
        print()
        print("Lookback windows (ensemble):", LOOKBACK_WINDOWS)
        print(f"Default window: {DEFAULT_WINDOW} days")
        print()
        print("Lead-lag detection:")
        print(f"  - DTW with Sakoe-Chiba radius: {DTW_RADIUS}")
        print(f"  - L\u00e9vy area lags: {LEVY_LAGS}")
        print()
        print("Graph learning:")
        print(f"  - Sparsity alpha: {GRAPH_SPARSITY_ALPHA}")
        print(f"  - Smoothness beta: {GRAPH_SMOOTHNESS_BETA}")
        print()
        print(f"Max deviation: {MAX_DEVIATION:.0%}")
        print(f"Rebalance frequency: 21 days")
        print()
        print(f"Data source: {PRICES_PATH}")
        print(f"Prices exist: {PRICES_PATH.exists()}")
    
    elif args.command == 'compute':
        network_momentum = NetworkMomentumLeadLag()
        leadlag_matrix = network_momentum.compute_leadlag_matrix(args.window)
        
        if leadlag_matrix:
            print(json.dumps(leadlag_matrix.to_dict(), indent=2))
        else:
            print(json.dumps({'error': 'Could not compute lead-lag matrix'}))
    
    elif args.command == 'backtest':
        backtester = NetworkMomentumBacktester(
            base_allocation=DEFAULT_BASE_ALLOCATION,
            start_date=args.start,
            end_date=args.end
        )
        result = backtester.run_backtest()
        print(json.dumps(result, indent=2, default=str))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    
    elif args.command == 'live':
        network_momentum = NetworkMomentumLeadLag()
        recommendation = network_momentum.get_current_recommendation(DEFAULT_BASE_ALLOCATION)
        
        if recommendation:
            print(json.dumps(recommendation.to_dict(), indent=2))
        else:
            print(json.dumps({'error': 'Could not compute recommendation'}))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(recommendation.to_dict() if recommendation else {}, f, indent=2)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
