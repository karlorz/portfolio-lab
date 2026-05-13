"""
Correlation Regime Detector (v2.72)

Adaptive correlation regime detection for chameleon risk parity.
Switches between normal and stress-period correlation matrices
based on market volatility and regime classification.

Key Features:
- Regime-dependent correlation matrices (normal vs crisis)
- Hierarchical clustering for risk factor grouping
- Ledoit-Wolf shrinkage estimation
- Integration with risk parity allocation engine
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import sys
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class RegimeType(Enum):
    """Market regime classification for correlation matrices"""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class CorrelationRegime:
    """Correlation matrix for a specific regime"""
    regime: RegimeType
    assets: List[str]
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    volatilities: np.ndarray
    start_date: str
    end_date: str
    sample_size: int
    stability_score: float  # How stable is this correlation structure


@dataclass
class RegimeClassification:
    """Current regime classification with confidence"""
    timestamp: str
    current_regime: RegimeType
    regime_probability: Dict[RegimeType, float]
    confidence: float
    features: Dict[str, float]  # VIX, trend, volatility, etc.
    days_in_current_regime: int
    transition_probability: float  # Probability of switching


@dataclass
class AdaptiveWeights:
    """Risk parity weights adapted to current regime"""
    timestamp: str
    regime: RegimeType
    assets: List[str]
    weights: np.ndarray
    risk_contributions: np.ndarray
    effective_correlation: str  # Which correlation matrix was used
    diversification_ratio: float


class CorrelationRegimeDetector:
    """
    Detects correlation regimes and adapts risk parity allocations.
    
    Uses:
    1. VIX levels for regime identification
    2. Rolling correlation stability metrics
    3. Hierarchical clustering for factor-based grouping
    """
    
    REGIME_THRESHOLDS = {
        'vix_normal': 20.0,
        'vix_high': 30.0,
        'vix_crisis': 40.0,
        'volatility_percentile': 75.0,
        'correlation_shift_threshold': 0.15
    }
    
    def __init__(
        self,
        assets: List[str] = None,
        lookback_window: int = 252,
        min_observations: int = 63,
        db_path: str = "data/correlation_regimes.db"
    ):
        self.assets = assets or [
            'SPY', 'QQQ', 'IWM',  # US Equity
            'EFA', 'EEM',  # International
            'TLT', 'IEF', 'SHY',  # Treasuries
            'GLD', 'SLV',  # Precious metals
            'DBC', 'USO',  # Commodities
            'VNQ',  # REITs
            'AGG', 'LQD'  # Bonds
        ]
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        
        self.db_path = Path(project_root) / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # Regime correlation matrices cache
        self._regime_matrices: Dict[RegimeType, Optional[CorrelationRegime]] = {
            r: None for r in RegimeType
        }
        
        # Current regime
        self._current_regime: Optional[RegimeClassification] = None
        self._regime_start_date: Optional[datetime] = None
    
    def _init_db(self):
        """Initialize SQLite database for correlation regime tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Correlation matrices by regime
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlation_matrices (
                id INTEGER PRIMARY KEY,
                regime TEXT NOT NULL,
                assets TEXT NOT NULL,
                correlation_matrix TEXT NOT NULL,
                covariance_matrix TEXT NOT NULL,
                volatilities TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                sample_size INTEGER,
                stability_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(regime, start_date)
            )
        """)
        
        # Regime classifications
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_classifications (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                current_regime TEXT NOT NULL,
                regime_probability TEXT NOT NULL,
                confidence REAL,
                features TEXT,
                days_in_current_regime INTEGER,
                transition_probability REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        """)
        
        # Adaptive allocations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_allocations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                assets TEXT NOT NULL,
                weights TEXT NOT NULL,
                risk_contributions TEXT NOT NULL,
                effective_correlation TEXT,
                diversification_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def classify_regime(
        self,
        vix: float,
        realized_vol_30d: float,
        spy_trend_50d: float,
        correlation_stability: float,
        historical_vix_percentile: float
    ) -> RegimeClassification:
        """
        Classify current market regime based on multiple features.
        
        Args:
            vix: Current VIX level
            realized_vol_30d: 30-day realized volatility (annualized)
            spy_trend_50d: SPY 50-day trend (positive = uptrend)
            correlation_stability: Measure of correlation stability (0-1)
            historical_vix_percentile: VIX percentile vs historical (0-100)
        
        Returns:
            RegimeClassification with probabilities and confidence
        """
        now = datetime.now()
        
        # Calculate regime probabilities using feature-based rules
        probs = {}
        
        # CRISIS: VIX > 40 or 95th percentile, high realized vol
        crisis_score = 0.0
        if vix > self.REGIME_THRESHOLDS['vix_crisis']:
            crisis_score += 0.5
        if historical_vix_percentile > 95:
            crisis_score += 0.3
        if realized_vol_30d > 0.30:  # 30% annualized
            crisis_score += 0.2
        probs[RegimeType.CRISIS] = min(1.0, crisis_score)
        
        # HIGH_VOL: VIX 30-40 or 75-95th percentile
        high_vol_score = 0.0
        if self.REGIME_THRESHOLDS['vix_high'] <= vix <= self.REGIME_THRESHOLDS['vix_crisis']:
            high_vol_score += 0.4
        if 75 <= historical_vix_percentile < 95:
            high_vol_score += 0.3
        if realized_vol_30d > 0.20:
            high_vol_score += 0.2
        if correlation_stability < 0.3:  # Unstable correlations
            high_vol_score += 0.1
        probs[RegimeType.HIGH_VOL] = min(1.0, high_vol_score)
        
        # RECOVERY: High vol but positive trend
        recovery_score = 0.0
        if historical_vix_percentile > 70 and spy_trend_50d > 0.02:
            recovery_score += 0.4
        if vix > 25 and spy_trend_50d > 0.05:
            recovery_score += 0.3
        if correlation_stability > 0.5:  # Stabilizing correlations
            recovery_score += 0.2
        probs[RegimeType.RECOVERY] = min(1.0, recovery_score)
        
        # NORMAL: Everything else, low VIX, stable
        normal_score = 0.0
        if vix < self.REGIME_THRESHOLDS['vix_normal']:
            normal_score += 0.5
        if historical_vix_percentile < 50:
            normal_score += 0.3
        if correlation_stability > 0.7:
            normal_score += 0.2
        probs[RegimeType.NORMAL] = min(1.0, normal_score)
        
        # Normalize to probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {k: 0.25 for k in probs.keys()}
        
        # Determine current regime
        current = max(probs, key=probs.get)
        confidence = probs[current]
        
        # Calculate days in current regime
        if self._current_regime and self._current_regime.current_regime == current:
            if self._regime_start_date:
                days_in_regime = (now - self._regime_start_date).days
            else:
                days_in_regime = 1
        else:
            days_in_regime = 1
            self._regime_start_date = now
        
        # Transition probability (based on persistence and recent changes)
        transition_prob = self._calculate_transition_probability(
            current, probs, correlation_stability
        )
        
        classification = RegimeClassification(
            timestamp=now.isoformat(),
            current_regime=current,
            regime_probability=probs,
            confidence=confidence,
            features={
                'vix': vix,
                'realized_vol_30d': realized_vol_30d,
                'spy_trend_50d': spy_trend_50d,
                'correlation_stability': correlation_stability,
                'vix_percentile': historical_vix_percentile
            },
            days_in_current_regime=days_in_regime,
            transition_probability=transition_prob
        )
        
        self._current_regime = classification
        return classification
    
    def _calculate_transition_probability(
        self,
        current_regime: RegimeType,
        probs: Dict[RegimeType, float],
        stability: float
    ) -> float:
        """Calculate probability of regime transition"""
        # Higher if second-best probability is close to current
        sorted_probs = sorted(probs.values(), reverse=True)
        if len(sorted_probs) > 1:
            gap = sorted_probs[0] - sorted_probs[1]
            prob_near = 1 - gap  # Small gap = higher transition prob
        else:
            prob_near = 0.0
        
        # Higher if correlations are unstable
        stability_factor = 1 - stability
        
        # Lower if we've been in regime for a while (persistence)
        if self._current_regime:
            persistence = min(1.0, self._current_regime.days_in_current_regime / 30)
        else:
            persistence = 0.0
        
        transition_prob = 0.4 * prob_near + 0.4 * stability_factor - 0.2 * persistence
        return max(0.0, min(1.0, transition_prob))
    
    def estimate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        regime: RegimeType,
        use_shrinkage: bool = True
    ) -> CorrelationRegime:
        """
        Estimate regime-specific correlation matrix from returns.
        
        Uses Ledoit-Wolf shrinkage for better out-of-sample performance.
        """
        if len(returns) < self.min_observations:
            raise ValueError(f"Insufficient observations: {len(returns)} < {self.min_observations}")
        
        assets = list(returns.columns)
        
        # Calculate sample covariance
        sample_cov = returns.cov().values
        
        if use_shrinkage:
            # Ledoit-Wolf shrinkage estimation
            cov_matrix = self._ledoit_wolf_shrinkage(returns, sample_cov)
        else:
            cov_matrix = sample_cov
        
        # Extract volatilities and correlation
        vols = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vols, vols)
        
        # Calculate stability score (based on condition number)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-10)
        stability = 1 / (1 + np.log(condition_number))
        
        regime_obj = CorrelationRegime(
            regime=regime,
            assets=assets,
            correlation_matrix=corr_matrix,
            covariance_matrix=cov_matrix,
            volatilities=vols,
            start_date=returns.index[0].isoformat() if hasattr(returns.index[0], 'isoformat') else str(returns.index[0]),
            end_date=returns.index[-1].isoformat() if hasattr(returns.index[-1], 'isoformat') else str(returns.index[-1]),
            sample_size=len(returns),
            stability_score=stability
        )
        
        self._regime_matrices[regime] = regime_obj
        return regime_obj
    
    def _ledoit_wolf_shrinkage(
        self,
        returns: pd.DataFrame,
        sample_cov: np.ndarray
    ) -> np.ndarray:
        """
        Apply Ledoit-Wolf shrinkage to covariance matrix.
        
        Shrinks toward a constant correlation model.
        """
        n, p = returns.shape
        
        # Sample correlation
        sample_corr = np.corrcoef(returns.T)
        
        # Target: constant correlation (average off-diagonal)
        mask = ~np.eye(p, dtype=bool)
        avg_corr = sample_corr[mask].mean()
        target = np.full((p, p), avg_corr)
        np.fill_diagonal(target, 1.0)
        
        # Calculate shrinkage intensity
        # Simplified estimator
        delta = 0.3  # Standard shrinkage parameter
        
        # Shrink sample correlation toward target
        shrunk_corr = (1 - delta) * sample_corr + delta * target
        
        # Reconstruct covariance
        vols = np.sqrt(np.diag(sample_cov))
        shrunk_cov = shrunk_corr * np.outer(vols, vols)
        
        return shrunk_cov
    
    def hierarchical_clustering(
        self,
        corr_matrix: np.ndarray,
        asset_names: List[str]
    ) -> List[List[str]]:
        """
        Perform hierarchical clustering on correlation matrix.
        
        Returns clusters of assets grouped by risk factors.
        """
        # Convert correlation to distance
        distance = 1 - corr_matrix
        
        # Ensure diagonal is 0
        np.fill_diagonal(distance, 0)
        
        # Convert to condensed distance matrix
        condensed = squareform(distance, checks=False)
        
        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method='ward')
        
        # Get ordered leaves
        order = leaves_list(linkage_matrix)
        
        # Create clusters based on correlation threshold
        clusters = []
        threshold = 0.5  # Assets with corr > 0.5 cluster together
        
        current_cluster = [asset_names[order[0]]]
        
        for i in range(1, len(order)):
            asset_i = asset_names[order[i]]
            prev_asset = asset_names[order[i-1]]
            
            idx_i = asset_names.index(asset_i)
            idx_prev = asset_names.index(prev_asset)
            
            if corr_matrix[idx_i, idx_prev] > threshold:
                current_cluster.append(asset_i)
            else:
                clusters.append(current_cluster)
                current_cluster = [asset_i]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def calculate_risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        asset_names: List[str],
        target_risk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate risk parity weights using iterative optimization.
        
        Equal risk contribution from each asset.
        """
        n = len(asset_names)
        
        if target_risk is None:
            target_risk = np.ones(n) / n  # Equal risk target
        
        # Initial guess: inverse volatility
        inv_vols = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vols / inv_vols.sum()
        
        # Iterative optimization (simplified)
        for _ in range(100):
            # Portfolio variance
            port_var = weights @ cov_matrix @ weights
            
            # Risk contributions
            marginal_risk = cov_matrix @ weights
            risk_contrib = weights * marginal_risk / np.sqrt(port_var)
            
            # Adjust weights to equalize risk contributions
            adjustment = target_risk / (risk_contrib + 1e-10)
            weights = weights * adjustment
            weights = weights / weights.sum()
            
            # Convergence check
            if np.std(risk_contrib) < 0.001:
                break
        
        # Final risk contributions
        port_var = weights @ cov_matrix @ weights
        marginal_risk = cov_matrix @ weights
        risk_contrib = weights * marginal_risk / np.sqrt(port_var)
        
        return weights, risk_contrib
    
    def get_adaptive_allocation(
        self,
        regime: RegimeType,
        available_assets: List[str],
        returns_data: Optional[pd.DataFrame] = None
    ) -> AdaptiveWeights:
        """
        Get risk parity allocation adapted to current correlation regime.
        
        Uses stored regime matrix or estimates from provided returns.
        """
        if self._regime_matrices[regime] is None and returns_data is not None:
            # Estimate from provided data
            self.estimate_correlation_matrix(returns_data, regime)
        
        regime_matrix = self._regime_matrices.get(regime)
        
        if regime_matrix is None:
            # Fall back to sample covariance from recent data
            raise ValueError(f"No correlation matrix available for regime {regime}")
        
        # Filter to available assets
        asset_indices = []
        available_subset = []
        for asset in available_assets:
            if asset in regime_matrix.assets:
                asset_indices.append(regime_matrix.assets.index(asset))
                available_subset.append(asset)
        
        if not asset_indices:
            raise ValueError(f"No matching assets found in regime matrix")
        
        # Extract sub-matrices
        corr_subset = regime_matrix.correlation_matrix[np.ix_(asset_indices, asset_indices)]
        cov_subset = regime_matrix.covariance_matrix[np.ix_(asset_indices, asset_indices)]
        
        # Calculate risk parity weights
        weights, risk_contrib = self.calculate_risk_parity_weights(
            cov_subset, available_subset
        )
        
        # Calculate diversification ratio
        weighted_vol = weights @ np.sqrt(np.diag(cov_subset))
        portfolio_vol = np.sqrt(weights @ cov_subset @ weights)
        div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return AdaptiveWeights(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            assets=available_subset,
            weights=weights,
            risk_contributions=risk_contrib,
            effective_correlation=f"regime_{regime.value}",
            diversification_ratio=div_ratio
        )
    
    def save_regime_matrix(self, regime: CorrelationRegime):
        """Save correlation regime to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO correlation_matrices
            (regime, assets, correlation_matrix, covariance_matrix, volatilities,
             start_date, end_date, sample_size, stability_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            regime.regime.value,
            json.dumps(regime.assets),
            json.dumps(regime.correlation_matrix.tolist()),
            json.dumps(regime.covariance_matrix.tolist()),
            json.dumps(regime.volatilities.tolist()),
            regime.start_date,
            regime.end_date,
            regime.sample_size,
            regime.stability_score
        ))
        
        conn.commit()
        conn.close()
    
    def save_regime_classification(self, classification: RegimeClassification):
        """Save regime classification to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        probs_json = json.dumps({k.value: v for k, v in classification.regime_probability.items()})
        
        cursor.execute("""
            INSERT OR REPLACE INTO regime_classifications
            (timestamp, current_regime, regime_probability, confidence, features,
             days_in_current_regime, transition_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            classification.timestamp,
            classification.current_regime.value,
            probs_json,
            classification.confidence,
            json.dumps(classification.features),
            classification.days_in_current_regime,
            classification.transition_probability
        ))
        
        conn.commit()
        conn.close()
    
    def save_adaptive_allocation(self, allocation: AdaptiveWeights):
        """Save adaptive allocation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO adaptive_allocations
            (timestamp, regime, assets, weights, risk_contributions,
             effective_correlation, diversification_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            allocation.timestamp,
            allocation.regime.value,
            json.dumps(allocation.assets),
            json.dumps(allocation.weights.tolist()),
            json.dumps(allocation.risk_contributions.tolist()),
            allocation.effective_correlation,
            allocation.diversification_ratio
        ))
        
        conn.commit()
        conn.close()
    
    def get_regime_history(
        self,
        days: int = 90
    ) -> List[Dict]:
        """Get historical regime classifications"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, current_regime, confidence, features,
                   days_in_current_regime, transition_probability
            FROM regime_classifications
            WHERE timestamp >= date('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{days} days',))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': r[0],
                'regime': r[1],
                'confidence': r[2],
                'features': json.loads(r[3]),
                'days_in_regime': r[4],
                'transition_prob': r[5]
            }
            for r in rows
        ]


def cli():
    """CLI for correlation regime detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Correlation Regime Detector v2.72')
    parser.add_argument('--classify', action='store_true', help='Classify current regime')
    parser.add_argument('--history', action='store_true', help='Show regime history')
    parser.add_argument('--vix', type=float, default=18.0, help='Current VIX')
    parser.add_argument('--vol', type=float, default=0.16, help='30-day realized vol')
    parser.add_argument('--trend', type=float, default=0.05, help='50-day trend')
    
    args = parser.parse_args()
    
    detector = CorrelationRegimeDetector()
    
    if args.classify:
        # Simplified classification with provided inputs
        classification = detector.classify_regime(
            vix=args.vix,
            realized_vol_30d=args.vol,
            spy_trend_50d=args.trend,
            correlation_stability=0.7,
            historical_vix_percentile=45.0
        )
        
        print("\n" + "="*60)
        print("CORRELATION REGIME CLASSIFICATION")
        print("="*60)
        print(f"Timestamp: {classification.timestamp}")
        print(f"Current Regime: {classification.current_regime.value.upper()}")
        print(f"Confidence: {classification.confidence:.1%}")
        print(f"Days in Regime: {classification.days_in_current_regime}")
        print(f"Transition Probability: {classification.transition_probability:.1%}")
        
        print("\nRegime Probabilities:")
        for regime, prob in classification.regime_probability.items():
            marker = " ←" if regime == classification.current_regime else ""
            print(f"  {regime.value}: {prob:.1%}{marker}")
        
        print("\nFeatures:")
        for feature, value in classification.features.items():
            print(f"  {feature}: {value:.3f}")
        
        # Risk parity allocation recommendation
        print("\n" + "-"*60)
        print("RISK PARITY RECOMMENDATION")
        print("-"*60)
        
        if classification.current_regime == RegimeType.CRISIS:
            print("  → Use CRISIS period correlation matrix (bond-equity correlation ~0.5)")
            print("  → Reduce equity exposure, increase duration hedge")
            print("  → Expect higher volatility, wider risk parity bands")
        elif classification.current_regime == RegimeType.HIGH_VOL:
            print("  → Use HIGH_VOL correlation matrix (elevated correlations)")
            print("  → Tighten risk parity bands, increase rebalance frequency")
            print("  → Monitor for transition to CRISIS or RECOVERY")
        elif classification.current_regime == RegimeType.RECOVERY:
            print("  → Use RECOVERY period correlation matrix (stabilizing)")
            print("  → Gradually increase equity, maintain bond hedge")
            print("  → Watch for full transition to NORMAL regime")
        else:  # NORMAL
            print("  → Use NORMAL period correlation matrix (diversified)")
            print("  → Standard risk parity allocation")
            print("  → Standard rebalance frequency (quarterly)")
        
        print("\n" + "="*60)
    
    elif args.history:
        history = detector.get_regime_history(days=90)
        
        print("\nRegime History (Last 90 Days)")
        print("="*60)
        
        for entry in history[:20]:  # Show last 20
            print(f"  {entry['timestamp'][:10]}: "
                  f"{entry['regime']:12} "
                  f"conf={entry['confidence']:.1%} "
                  f"days={entry['days_in_regime']:2d}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
