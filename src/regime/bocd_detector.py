"""
Bayesian Online Changepoint Detection (BOCD) for Regime Detection.

Implements Adams & MacKay (2007) for real-time structural break detection
in time series data without fixed observation windows.

Simplified implementation using conjugate Normal model for mean detection
(rather than full Normal-Inverse-Gamma for variance).

Algorithm:
1. At each time step t, maintain a distribution over run lengths r_t.
2. Calculate predictive probability of current observation given each run length.
3. Update run length distribution using Bayes' rule.
4. Probability of new changepoint is the prior probability of run length 0.

For regime detection, we monitor shifts in return distribution.

Usage:
    from src.regime.bocd_detector import BOCDDetector

    detector = BOCDDetector(hazard_rate=1/252)  # Daily hazard (expected 1 change per year)
    detector.fit(returns)  # Fit on return series
    signal = detector.get_signal()  # Get regime signal

References:
    Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
    arXiv:0710.3742.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "BOCDDetector",
    "BOCDResult",
]


@dataclass
class BOCDResult:
    """Result from BOCD detection."""
    run_length_probs: np.ndarray  # Probability distribution over run lengths at each time
    changepoint_probs: np.ndarray  # Probability of changepoint at each time
    regime_labels: np.ndarray  # Hard regime labels (0 = no change, 1 = change)
    timestamps: Optional[List] = None  # Optional timestamps for each observation


class BOCDDetector:
    """
    Bayesian Online Changepoint Detection for regime detection.
    
    Uses conjugate Normal model for mean detection.
    Maintains run length distribution and detects structural breaks.
    
    Parameters:
        hazard_rate: Prior probability of changepoint at each time step.
                     Default 1/252 assumes ~1 changepoint per year (daily data).
        threshold: Probability threshold for regime change detection (default 0.5).
        min_run_length: Minimum run length to consider stable (default 5).
    """
    
    def __init__(
        self,
        hazard_rate: float = 1.0 / 252,
        threshold: float = 0.5,
        min_run_length: int = 5
    ):
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.min_run_length = min_run_length
        
        # State
        self._run_length_probs: Optional[np.ndarray] = None
        self._changepoint_probs: Optional[np.ndarray] = None
        self._regime_labels: Optional[np.ndarray] = None
        self.timestamps: Optional[List] = None
        
        # Hyperparameters for Normal prior (conjugate for known variance)
        # We'll estimate variance from data
        self.mu_0 = 0.0
        self.kappa_0 = 1.0  # Prior precision (1/variance of prior)
        
        # Sufficient statistics for each run length
        self._sum_x: Optional[np.ndarray] = None  # Sum of observations for each run length
        self._count: Optional[np.ndarray] = None   # Count of observations for each run length
        
        self._fitted = False
    
    def fit(
        self,
        returns: np.ndarray,
        timestamps: Optional[List] = None
    ) -> 'BOCDDetector':
        """
        Fit BOCD detector on return series.
        
        Args:
            returns: 1D array of returns (e.g., daily log returns)
            timestamps: Optional list of timestamps for each observation
            
        Returns:
            self
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 1:
            raise ValueError("Returns must be 1D array")
        
        n = len(returns)
        if n < 2:
            raise ValueError("Need at least 2 observations")
        
        self.timestamps = timestamps
        
        # Estimate observation variance from data (for predictive distribution)
        self._obs_var = max(np.var(returns), 1e-10)
        
        # Initialize run length distribution
        # Row 0 = before any observations, Row t = after observing t-th point
        self._run_length_probs = np.zeros((n + 1, n + 1))
        self._changepoint_probs = np.zeros(n)
        self._regime_labels = np.zeros(n, dtype=int)
        
        # Sufficient statistics: for each run length r, maintain sum and count
        self._sum_x = np.zeros(n + 1)
        self._count = np.zeros(n + 1)
        
        # Initial distribution: run length 0 with probability 1
        self._run_length_probs[0, 0] = 1.0
        
        # Process each observation
        for t in range(n):
            x = returns[t]
            
            # Current run length probabilities (before update)
            rlp = self._run_length_probs[t, :t+1].copy()
            
            # 1. Calculate predictive probabilities for each run length
            # For conjugate Normal with known variance:
            # Predictive is Normal(mu_post, var_post) where:
            #   mu_post = (kappa_0 * mu_0 + sum_x) / (kappa_0 + count)
            #   var_post = obs_var * (1 + 1/(kappa_0 + count))
            
            counts = self._count[:t+1] + 1  # Will have count+1 after this observation
            sums = self._sum_x[:t+1] + x
            
            # Posterior mean after including x
            posterior_mu = (self.kappa_0 * self.mu_0 + sums) / (self.kappa_0 + counts)
            
            # Predictive variance (before observing x)
            # var_pred = obs_var * (1 + 1/(kappa_0 + count))
            pred_var = self._obs_var * (1.0 + 1.0 / (self.kappa_0 + self._count[:t+1]))
            pred_var = np.maximum(pred_var, 1e-10)  # Avoid zero
            
            # Predictive probability: Normal(x | prior_mean, pred_var)
            # prior_mean = (kappa_0 * mu_0 + sum_x) / (kappa_0 + count)
            prior_mu = (self.kappa_0 * self.mu_0 + self._sum_x[:t+1]) / (self.kappa_0 + self._count[:t+1])
            
            # Log predictive probability
            log_pred = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (x - prior_mu)**2 / pred_var
            
            # Convert to probability (unnormalized)
            pred_probs = np.exp(log_pred)
            
            # 2. Update step: multiply by run length probabilities
            growth_probs = rlp * pred_probs * (1 - self.hazard_rate)
            
            # Changepoint probability: sum of probabilities that would transition to run length 0
            changepoint_prob = np.sum(rlp * pred_probs * self.hazard_rate)
            
            # New run length distribution
            self._run_length_probs[t+1, 1:t+2] = growth_probs
            self._run_length_probs[t+1, 0] = changepoint_prob
            
            # Normalize
            total_prob = np.sum(self._run_length_probs[t+1, :t+2])
            if total_prob > 0:
                self._run_length_probs[t+1, :t+2] /= total_prob
            else:
                # Fallback: reset to run length 0
                self._run_length_probs[t+1, 0] = 1.0
            
            # Record changepoint probability
            self._changepoint_probs[t] = changepoint_prob
            
            # Update sufficient statistics for next step
            # For run length 0 (new run): reset
            self._sum_x[0] = 0.0
            self._count[0] = 0.0
            
            # For run length r > 0: update with new observation
            for r in range(1, t+2):
                if self._run_length_probs[t+1, r] > 1e-10:
                    self._sum_x[r] = self._sum_x[r-1] + x
                    self._count[r] = self._count[r-1] + 1
                else:
                    # Reset stats for negligible probability runs
                    self._sum_x[r] = 0.0
                    self._count[r] = 0.0
            
            # Determine regime label
            self._regime_labels[t] = 1 if changepoint_prob > self.threshold else 0
        
        self._fitted = True
        return self
    
    def get_signal(self) -> Dict:
        """
        Get regime signal in portfolio-lab format.
        
        Returns:
            Dict with regime information
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        # Calculate regime statistics
        n = len(self._regime_labels)
        changepoint_count = np.sum(self._changepoint_probs > self.threshold)
        
        # Determine current regime based on recent run length
        # If high probability of short run length, we're in a new regime
        recent_probs = self._run_length_probs[-1, :self.min_run_length+1]
        regime_change_prob = np.sum(recent_probs)
        
        # Map to portfolio-lab regime types
        # For volatility detection, we'll map to:
        # 0 = NORMAL, 1 = CRISIS, 2 = HIGH_VOL, 3 = LOW_VOL
        # Simplified: changepoint = regime change (CRISIS or HIGH_VOL)
        current_regime = 1 if regime_change_prob > self.threshold else 0
        
        return {
            "bocd_detector": {
                "regime": current_regime,
                "regime_change_prob": float(regime_change_prob),
                "changepoint_count": int(changepoint_count),
                "current_run_length": int(np.argmax(self._run_length_probs[-1])),
                "hazard_rate": self.hazard_rate,
                "threshold": self.threshold,
                "n_observations": n,
                "description": "Bayesian Online Changepoint Detection regime signal"
            }
        }
    
    def get_changepoint_timestamps(
        self,
        timestamps: Optional[List] = None
    ) -> List:
        """
        Get timestamps where changepoints were detected.
        
        Args:
            timestamps: Optional list of timestamps for each observation
            
        Returns:
            List of (timestamp, probability) tuples for detected changepoints
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        if timestamps is None:
            timestamps = self.timestamps
        
        if timestamps is None:
            # Return indices instead
            return [
                (i, float(self._changepoint_probs[i]))
                for i in range(len(self._changepoint_probs))
                if self._changepoint_probs[i] > self.threshold
            ]
        
        # Match timestamps to changepoints
        return [
            (timestamps[i], float(self._changepoint_probs[i]))
            for i in range(len(self._changepoint_probs))
            if self._changepoint_probs[i] > self.threshold
        ]