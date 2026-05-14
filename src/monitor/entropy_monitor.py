"""
Entropy-Based Diversification Monitor v3.22
Shannon entropy calculation for portfolio concentration risk assessment.

Provides:
- Shannon entropy of portfolio weights
- Effective number of assets (exp(entropy))
- Herfindahl-Hirschman Index (concentration)
- Normalized diversification score (0-100)
- Correlation structure entropy
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class EntropyMetrics:
    """Container for entropy-based diversification metrics."""
    timestamp: str
    shannon_entropy: float           # Raw entropy value (natural log)
    effective_n: float               # Effective number of assets (exp(entropy))
    max_possible: float              # ln(n_assets) - maximum possible entropy
    normalized_score: float          # 0-100 scale
    concentration_risk: str        # low/medium/high/critical
    hhi_index: float                 # Herfindahl-Hirschman Index
    correlation_entropy: Optional[float] = None  # Cross-asset correlation structure
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'shannon_entropy': round(self.shannon_entropy, 4),
            'effective_n': round(self.effective_n, 2),
            'max_possible': round(self.max_possible, 4),
            'normalized_score': round(self.normalized_score, 1),
            'concentration_risk': self.concentration_risk,
            'hhi_index': round(self.hhi_index, 4),
            'correlation_entropy': round(self.correlation_entropy, 4) if self.correlation_entropy else None
        }


class EntropyCalculator:
    """Calculate diversification metrics using information entropy."""
    
    # Thresholds for concentration risk levels (raw entropy values)
    # These thresholds are designed for portfolios with 3-10 assets
    # For 3 assets: max entropy = ln(3) ≈ 1.10
    # Critical: entropy < 0.5 (~45% of max possible for 3 assets)
    # Warning: entropy 0.5-0.7 (~45-65% of max)  
    # Moderate: entropy 0.7-0.9 (~65-80% of max)
    # Good: entropy > 0.9 (>80% of max)
    RISK_THRESHOLDS = {
        'critical': 0.5,   # Severe concentration (<45% of max diversification)
        'warning': 0.7,    # Elevated concentration (45-65%)
        'moderate': 0.9,   # Below optimal (65-80%)
        'good': 1.0       # Good diversification (>80%)
    }
    
    def __init__(self, risk_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize entropy calculator.
        
        Args:
            risk_thresholds: Custom threshold values (optional)
        """
        self.thresholds = risk_thresholds or self.RISK_THRESHOLDS.copy()
    
    def shannon_entropy(self, weights: np.ndarray) -> float:
        """
        Calculate Shannon entropy of portfolio weights.
        
        H = -sum(w_i * ln(w_i)) for all w_i > 0
        
        Args:
            weights: Array of portfolio weights (must sum to ~1)
            
        Returns:
            Shannon entropy in nats (natural log)
        """
        # Filter positive weights only
        positive_weights = weights[weights > 0]
        
        if len(positive_weights) == 0:
            return 0.0
        
        # Normalize to ensure sum = 1
        normalized = positive_weights / positive_weights.sum()
        
        # Calculate entropy: -sum(p * ln(p))
        entropy = -np.sum(normalized * np.log(normalized))
        
        return float(entropy)
    
    def effective_n_assets(self, entropy: float) -> float:
        """
        Convert entropy to effective number of assets.
        
        N_eff = exp(entropy)
        
        This represents how many equally-weighted assets would
        produce the same diversification benefit.
        
        Args:
            entropy: Shannon entropy value
            
        Returns:
            Effective number of assets
        """
        return math.exp(entropy)
    
    def herfindahl_hirschman_index(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI).
        
        HHI = sum(w_i^2)
        
        Ranges from 1/n (equal weight) to 1 (single asset).
        Common thresholds:
        - HHI < 0.15: Competitive (diversified)
        - HHI 0.15-0.25: Moderate concentration  
        - HHI > 0.25: High concentration
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            HHI value (0 to 1)
        """
        return float(np.sum(weights ** 2))
    
    def normalized_diversification_score(
        self, 
        entropy: float, 
        n_assets: int
    ) -> float:
        """
        Normalize entropy to 0-100 scale.
        
        Score = 100 * entropy / ln(n_assets)
        
        Args:
            entropy: Shannon entropy value
            n_assets: Number of assets in portfolio
            
        Returns:
            Normalized score (0-100)
        """
        if n_assets <= 1:
            return 100.0 if entropy > 0 else 0.0
        
        max_entropy = math.log(n_assets)
        score = 100.0 * entropy / max_entropy
        
        return min(100.0, max(0.0, score))
    
    def concentration_risk_level(self, entropy: float) -> str:
        """
        Determine concentration risk level from entropy.
        
        Args:
            entropy: Shannon entropy value
            
        Returns:
            Risk level string: critical/warning/moderate/good
        """
        if entropy < self.thresholds['critical']:
            return 'critical'
        elif entropy < self.thresholds['warning']:
            return 'high'
        elif entropy < self.thresholds['moderate']:
            return 'medium'
        elif entropy < self.thresholds['good']:
            return 'low'
        else:
            return 'good'
    
    def correlation_entropy(
        self, 
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate entropy of correlation structure.
        
        Measures effective independence of assets based on
        their correlation structure. Lower entropy indicates
        higher correlation (less diversification).
        
        Uses eigenvalue entropy: H = -sum(λ_i * ln(λ_i))
        where λ_i are normalized eigenvalues of correlation matrix.
        
        Args:
            correlation_matrix: NxN correlation matrix
            
        Returns:
            Correlation entropy (higher = more independent)
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        
        # Normalize to sum to 1 (treat as probability distribution)
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Calculate entropy
        positive_eigen = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(positive_eigen * np.log(positive_eigen))
        
        return float(entropy)
    
    def calculate_metrics(
        self,
        weights: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> EntropyMetrics:
        """
        Calculate all entropy-based diversification metrics.
        
        Args:
            weights: Dict mapping asset symbols to weights
            correlation_matrix: Optional correlation matrix for cross-asset entropy
            
        Returns:
            EntropyMetrics dataclass with all calculated values
        """
        # Convert to numpy array
        weight_values = np.array(list(weights.values()))
        n_assets = len(weight_values)
        
        # Calculate metrics
        entropy = self.shannon_entropy(weight_values)
        effective_n = self.effective_n_assets(entropy)
        max_possible = math.log(n_assets) if n_assets > 1 else 0.0
        normalized_score = self.normalized_diversification_score(entropy, n_assets)
        risk_level = self.concentration_risk_level(entropy)
        hhi = self.herfindahl_hirschman_index(weight_values)
        
        # Correlation entropy if matrix provided
        corr_entropy = None
        if correlation_matrix is not None:
            corr_entropy = self.correlation_entropy(correlation_matrix)
        
        return EntropyMetrics(
            timestamp=datetime.now().isoformat(),
            shannon_entropy=entropy,
            effective_n=effective_n,
            max_possible=max_possible,
            normalized_score=normalized_score,
            concentration_risk=risk_level,
            hhi_index=hhi,
            correlation_entropy=corr_entropy
        )
    
    def check_alert(self, metrics: EntropyMetrics) -> Optional[Dict]:
        """
        Check if entropy metrics trigger an alert.
        
        Args:
            metrics: Calculated entropy metrics
            
        Returns:
            Alert dict if threshold breached, None otherwise
        """
        if metrics.concentration_risk == 'critical':
            return {
                'level': 'critical',
                'message': f"CRITICAL: Portfolio entropy {metrics.shannon_entropy:.2f} below danger threshold {self.thresholds['critical']}",
                'action': 'Immediate rebalancing recommended - severe concentration detected'
            }
        elif metrics.concentration_risk == 'high':
            return {
                'level': 'warning',
                'message': f"WARNING: Portfolio entropy {metrics.shannon_entropy:.2f} indicates elevated concentration",
                'action': 'Review allocation weights - consider rebalancing'
            }
        elif metrics.concentration_risk == 'medium':
            return {
                'level': 'monitor',
                'message': f"MONITOR: Portfolio entropy {metrics.shannon_entropy:.2f} below optimal",
                'action': 'Monitor for further concentration'
            }
        
        return None


class EntropyHistory:
    """Track entropy metrics over time."""
    
    def __init__(self, max_history: int = 90):
        """
        Initialize history tracker.
        
        Args:
            max_history: Maximum number of days to keep
        """
        self.history: List[EntropyMetrics] = []
        self.max_history = max_history
    
    def add(self, metrics: EntropyMetrics):
        """Add metrics to history."""
        self.history.append(metrics)
        
        # Trim old entries
        cutoff = datetime.now() - timedelta(days=self.max_history)
        self.history = [
            h for h in self.history 
            if datetime.fromisoformat(h.timestamp) > cutoff
        ]
    
    def get_trend(self, days: int = 30) -> Dict:
        """
        Get entropy trend over specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with trend statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            h for h in self.history 
            if datetime.fromisoformat(h.timestamp) > cutoff
        ]
        
        if not recent:
            return {'available': False}
        
        entropies = [h.shannon_entropy for h in recent]
        scores = [h.normalized_score for h in recent]
        
        return {
            'available': True,
            'period_days': days,
            'n_samples': len(recent),
            'entropy_mean': round(np.mean(entropies), 4),
            'entropy_std': round(np.std(entropies), 4),
            'entropy_min': round(min(entropies), 4),
            'entropy_max': round(max(entropies), 4),
            'score_mean': round(np.mean(scores), 1),
            'trend_direction': 'improving' if entropies[-1] > entropies[0] else 'declining',
            'current_vs_mean': round(entropies[-1] - np.mean(entropies), 4)
        }
    
    def to_json(self) -> str:
        """Export history to JSON."""
        return json.dumps([h.to_dict() for h in self.history], indent=2)


# Singleton calculator for reuse
_default_calculator: Optional[EntropyCalculator] = None

def get_calculator() -> EntropyCalculator:
    """Get default entropy calculator instance."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = EntropyCalculator()
    return _default_calculator


def calculate_portfolio_entropy(
    weights: Dict[str, float],
    correlation_matrix: Optional[np.ndarray] = None
) -> EntropyMetrics:
    """
    Convenience function to calculate portfolio entropy metrics.
    
    Args:
        weights: Dict mapping asset symbols to weights
        correlation_matrix: Optional correlation matrix
        
    Returns:
        EntropyMetrics with all calculated values
    """
    calc = get_calculator()
    return calc.calculate_metrics(weights, correlation_matrix)
