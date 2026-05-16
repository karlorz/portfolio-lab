#!/usr/bin/env python3
"""
Portfolio-Lab v5.00: FPILOT Predictive Model Module

Lightweight VAR-based multi-step price prediction for inference-time planning.
No ML dependencies — uses numpy-only OLS estimation with rolling window calibration.

Architecture:
  - fits a VAR(1) model on price returns (rolling 252-day window)
  - generates multi-step (5-10 day) price trajectory forecasts
  - estimates expected returns + covariance for the planning horizon
  - provides confidence intervals via residual bootstrapping

References:
  arxiv:2605.12653 — "Plan Before You Trade" (FPILOT)
  https://arxiv.org/abs/2605.12653
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PredictionResult:
    """Output from predictive model for a planning horizon."""
    # Expected returns over the horizon (n_assets,)
    expected_returns: np.ndarray
    # Covariance matrix over the horizon (n_assets, n_assets)
    covariance: np.ndarray
    # Multi-step price trajectories (n_steps, n_assets)
    trajectories: np.ndarray
    # Confidence scores for each step (0-1, lower = more uncertainty)
    step_confidence: np.ndarray
    # Whether the model has sufficient data for prediction
    valid: bool
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateTrajectory:
    """A candidate allocation trajectory for planning."""
    # Allocation weights at each step (n_steps, n_assets)
    allocations: np.ndarray
    # Expected cumulative return
    expected_return: float
    # Expected risk (std of returns)
    expected_risk: float
    # Sharpe-like score for this trajectory
    score: float
    # Whether this trajectory is feasible (obeys constraints)
    feasible: bool = True
    # Step-by-step scores
    step_scores: np.ndarray = field(default_factory=lambda: np.array([]))


class PredictiveModel:
    """
    Lightweight VAR(1) price prediction model for FPILOT planning.

    Fits a vector autoregression of order 1 on asset returns:
      r_{t+1} = c + A * r_t + epsilon

    Uses rolling window OLS estimation (numpy only).

    Attributes:
        window: Rolling calibration window in trading days (default 252)
        horizon: Planning horizon in days (default 5)
        n_assets: Number of assets being modeled
        _A: Fitted VAR coefficient matrix (n_assets, n_assets)
        _c: Fitted intercept vector (n_assets,)
        _residuals: Historical residuals from fitted model
        _is_fitted: Whether model has been calibrated
    """

    def __init__(
        self,
        n_assets: int = 4,
        window: int = 252,
        horizon: int = 5,
        use_bootstrap: bool = True,
        n_bootstrap: int = 100,
    ):
        self.n_assets = n_assets
        self.window = window
        self.horizon = horizon
        self.use_bootstrap = use_bootstrap
        self.n_bootstrap = n_bootstrap

        # Fitted parameters
        self._A: Optional[np.ndarray] = None  # (n_assets, n_assets)
        self._c: Optional[np.ndarray] = None  # (n_assets,)
        self._residuals: Optional[np.ndarray] = None  # (n_samples, n_assets)
        self._is_fitted = False

        # Training data (raw prices stored for rolling updates)
        self._price_history: List[np.ndarray] = []  # list of (n_assets,) price vectors

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been calibrated."""
        return self._is_fitted

    def update_prices(self, prices: np.ndarray):
        """
        Update price history with latest observation.

        Args:
            prices: (n_assets,) price vector for the current day
        """
        if prices.ndim != 1 or prices.shape[0] != self.n_assets:
            raise ValueError(
                f"Expected (n_assets={self.n_assets},) array, got {prices.shape}"
            )
        self._price_history.append(prices.copy())

        # Trim history to window + horizon
        max_len = self.window + self.horizon + 1
        if len(self._price_history) > max_len:
            self._price_history = self._price_history[-max_len:]

    def fit(self, prices: np.ndarray) -> bool:
        """
        Fit VAR(1) model on price history.

        Args:
            prices: (n_samples, n_assets) array of daily prices, oldest first

        Returns:
            True if fit succeeded, False if insufficient data
        """
        n = prices.shape[0]
        if n < self.window + 2:
            return False

        # Use most recent window observations
        fit_prices = prices[-self.window - 1:]

        # Compute returns: window+1 prices -> window returns
        returns = np.diff(fit_prices, axis=0) / (fit_prices[:-1] + 1e-10)

        # We need both r_t and r_{t+1} for VAR(1):
        #   r_{t+1} = c + A * r_t + epsilon
        # So we have (window - 1) observations
        n_obs = self.window - 1

        # Build design matrix: [1, r_t] for each t
        X = np.ones((n_obs, self.n_assets + 1))
        X[:, 1:] = returns[:-1]  # lagged returns (n_obs, n_assets)

        # Target: r_{t+1}
        y = returns[1:]  # (n_obs, n_assets)

        # OLS: beta = (X'X)^{-1} X'y
        # Handle potential singularity with pseudo-inverse
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.pinv(XtX)
        except np.linalg.LinAlgError:
            return False

        beta = XtX_inv @ X.T @ y  # (n_assets + 1, n_assets)

        self._c = beta[0]  # intercept (n_assets,)
        self._A = beta[1:]  # coefficient matrix (n_assets, n_assets)

        # Compute residuals
        residuals = y - X @ beta  # (n_obs, n_assets)
        self._residuals = residuals
        self._is_fitted = True

        # Store price history
        self._price_history = [prices[i].copy() for i in range(n)]

        return True

    def predict(
        self,
        horizon: Optional[int] = None,
        return_cov: bool = True,
        n_bootstrap: Optional[int] = None,
    ) -> PredictionResult:
        """
        Generate multi-step price trajectory forecasts.

        Args:
            horizon: Number of steps to predict (default: self.horizon)
            return_cov: Whether to compute covariance matrix
            n_bootstrap: Number of bootstrap samples for confidence (default: self.n_bootstrap)

        Returns:
            PredictionResult with trajectory, expected returns, covariance
        """
        if not self._is_fitted or self._A is None or self._c is None:
            return self._empty_result()

        h = horizon or self.horizon
        if h < 1:
            h = 1

        # Get latest return for initial condition
        if len(self._price_history) < 2:
            return self._empty_result()

        latest_prices = self._price_history[-1]
        prev_prices = self._price_history[-2]
        latest_return = (latest_prices - prev_prices) / (prev_prices + 1e-10)

        # Deterministic forecast (expected trajectory)
        trajectory = np.zeros((h, self.n_assets))
        current_return = latest_return.copy()

        for step in range(h):
            next_return = self._c + self._A @ current_return
            trajectory[step] = next_return
            current_return = next_return

        # Compute expected cumulative returns over horizon
        # (approximate: sum of expected returns, ignoring compounding)
        expected_returns = np.sum(trajectory, axis=0)

        # Bootstrap confidence estimation
        n_boot = n_bootstrap or self.n_bootstrap
        step_confidence = np.ones(h)

        if self._residuals is not None and self._residuals.shape[0] > 10 and n_boot > 0:
            boot_trajectories = np.zeros((n_boot, h, self.n_assets))
            for b in range(n_boot):
                sampled_residuals = self._residuals[
                    np.random.choice(self._residuals.shape[0], size=h, replace=True)
                ]
                current_return_boot = latest_return.copy()
                for step in range(h):
                    next_return_boot = (
                        self._c + self._A @ current_return_boot + sampled_residuals[step]
                    )
                    boot_trajectories[b, step] = next_return_boot
                    current_return_boot = next_return_boot

            # Confidence = 1 - (spread / max_spread)
            spreads = np.std(boot_trajectories, axis=0).mean(axis=1)  # (h,)
            max_spread = np.max(spreads) + 1e-10
            step_confidence = 1.0 - (spreads / max_spread)
            step_confidence = np.clip(step_confidence, 0.0, 1.0)
        elif self._residuals is not None and self._residuals.shape[0] > 3:
            # Fallback: use residual std as uncertainty proxy
            residual_std = np.std(self._residuals, axis=0).mean()
            scaling = np.tanh(1.0 / (residual_std + 1e-10))
            step_confidence = np.full(h, float(scaling))

        # Covariance from bootstrap samples
        covariance = np.eye(self.n_assets) * 0.01  # fallback
        if return_cov and self._residuals is not None:
            # Use residual covariance as estimate of single-step covariance
            covariance = np.cov(self._residuals, rowvar=False)
            # Scale by horizon for cumulative (approximate)
            covariance = covariance * h

        return PredictionResult(
            expected_returns=expected_returns,
            covariance=covariance,
            trajectories=trajectory,
            step_confidence=step_confidence,
            valid=True,
            metadata={
                "horizon": h,
                "n_bootstrap": n_boot if (self._residuals is not None and n_boot > 0) else 0,
                "fitted": True,
                "latest_return_std": float(
                    np.std(self._residuals) if self._residuals is not None else 0.0
                ),
            },
        )

    def _empty_result(self) -> PredictionResult:
        """Return an empty/fallback result when model is not ready."""
        return PredictionResult(
            expected_returns=np.zeros(self.n_assets),
            covariance=np.eye(self.n_assets) * 0.01,
            trajectories=np.zeros((self.horizon, self.n_assets)),
            step_confidence=np.zeros(self.horizon),
            valid=False,
            metadata={"fitted": False, "reason": "insufficient_data"},
        )

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "n_assets": self.n_assets,
            "window": self.window,
            "horizon": self.horizon,
            "A": self._A.tolist() if self._A is not None else None,
            "c": self._c.tolist() if self._c is not None else None,
            "is_fitted": self._is_fitted,
            "history_length": len(self._price_history),
        }

    def load_state(self, state: Dict[str, Any]):
        """Load model state from serialized dict."""
        self.n_assets = state.get("n_assets", self.n_assets)
        self.window = state.get("window", self.window)
        self.horizon = state.get("horizon", self.horizon)

        a = state.get("A")
        if a is not None:
            self._A = np.array(a)
        c = state.get("c")
        if c is not None:
            self._c = np.array(c)

        self._is_fitted = state.get("is_fitted", False)

    def clear_history(self):
        """Clear price history (keeps fitted parameters)."""
        self._price_history.clear()


class TrajectoryOptimizer:
    """
    Generates and scores candidate allocation trajectories for planning.

    Uses cross-entropy method (CEM)-style sampling to explore allocation
    paths over the planning horizon and select the optimal trajectory.
    """

    def __init__(
        self,
        n_assets: int = 4,
        n_candidates: int = 50,
        n_elite: int = 10,
        n_iterations: int = 3,
    ):
        self.n_assets = n_assets
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations

        # Default allocation weights
        self.default_weights = np.array([0.46, 0.38, 0.16, 0.0])
        if n_assets != 4:
            self.default_weights = np.array([1.0 / n_assets] * n_assets)

        # Constraints (auto-sized to n_assets)
        self.min_weights = np.array([0.30, 0.20, 0.05, 0.0])
        self.max_weights = np.array([0.60, 0.50, 0.25, 0.05])
        if n_assets != 4:
            self.min_weights = np.array([0.05] * n_assets)
            self.max_weights = np.array([0.70] * n_assets)
            self.min_weights[0] = 0.30  # SPY-like gets higher floor

    def set_constraints(
        self, min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        """Set allocation constraints."""
        if min_weights is not None:
            self.min_weights = np.array(min_weights)
        if max_weights is not None:
            self.max_weights = np.array(max_weights)

    def generate_trajectories(
        self,
        prediction: PredictionResult,
        current_weights: np.ndarray,
        horizon: int,
    ) -> List[CandidateTrajectory]:
        """
        Generate candidate allocation trajectories using CEM sampling.

        Args:
            prediction: Price prediction from PredictiveModel
            current_weights: Current portfolio allocation (n_assets,)
            horizon: Number of steps in trajectory

        Returns:
            List of CandidateTrajectory objects, sorted by score descending
        """
        if not prediction.valid:
            return self._default_trajectory(current_weights, horizon)

        candidates = []
        expected_returns = prediction.expected_returns
        covariance = prediction.covariance

        # CEM-style iterative refinement
        # Start with gaussian centered at current weights
        mu = current_weights.copy()
        sigma = np.eye(self.n_assets) * 0.05

        for iteration in range(self.n_iterations):
            iteration_candidates = []

            for _ in range(self.n_candidates):
                # Sample allocation adjustments
                delta = np.random.multivariate_normal(mu, sigma)
                # Clip to [0, 1]
                weights = np.clip(delta, 0.0, 1.0)
                # Normalize to sum to 1
                weights = weights / (weights.sum() + 1e-10)

                # Check constraints
                feasible = True
                if len(weights) == len(self.min_weights):
                    feasible = bool(
                        np.all(weights >= self.min_weights)
                        and np.all(weights <= self.max_weights)
                    )

                # Score trajectory
                score = self._score_allocation(weights, expected_returns, covariance)

                trajectory = np.tile(weights, (horizon, 1))

                step_scores = np.array([
                    self._score_allocation(weights, expected_returns, covariance)
                    for _ in range(horizon)
                ])

                step_return = self._expected_return(weights, expected_returns)
                step_risk = self._expected_risk(weights, covariance)

                candidate = CandidateTrajectory(
                    allocations=trajectory,
                    expected_return=step_return,
                    expected_risk=step_risk,
                    score=score,
                    feasible=feasible,
                    step_scores=step_scores,
                )
                iteration_candidates.append(candidate)

            # Sort by score
            iteration_candidates.sort(key=lambda c: c.score, reverse=True)

            # Select elites
            elite_scores = np.array([
                c.score for c in iteration_candidates[:self.n_elite]
            ])
            elite_allocations = np.array([
                c.allocations[0] for c in iteration_candidates[:self.n_elite]
            ])

            # Update CEM parameters
            if len(elite_allocations) > 1:
                mu = np.mean(elite_allocations, axis=0)
                sigma = np.cov(elite_allocations, rowvar=False) + np.eye(self.n_assets) * 0.01

            candidates = iteration_candidates

        # Return all sorted candidates
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def select_optimal(
        self,
        candidates: List[CandidateTrajectory],
        risk_aversion: float = 1.0,
    ) -> CandidateTrajectory:
        """
        Select the optimal trajectory from candidates.

        Args:
            candidates: Sorted list of candidate trajectories
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Selected CandidateTrajectory (fallback to first feasible)
        """
        if not candidates:
            return self._default_trajectory_single()

        # Prefer feasible candidates
        feasible = [c for c in candidates if c.feasible]
        if not feasible:
            feasible = candidates

        # Apply risk aversion scoring
        scored = []
        for c in feasible:
            risk_adj_return = c.expected_return - risk_aversion * 0.5 * c.expected_risk**2
            scored.append((risk_adj_return, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _score_allocation(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Score an allocation using expected return / risk ratio."""
        ret = self._expected_return(weights, expected_returns)
        risk = self._expected_risk(weights, covariance)
        if risk < 1e-10:
            return 0.0
        return ret / (risk + 1e-10)

    def _expected_return(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        """Compute expected portfolio return."""
        return float(np.dot(weights, expected_returns))

    def _expected_risk(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Compute expected portfolio risk (std of returns)."""
        return float(np.sqrt(weights @ covariance @ weights + 1e-10))

    def _default_trajectory(
        self, current_weights: np.ndarray, horizon: int
    ) -> List[CandidateTrajectory]:
        """Create fallback trajectory (no planning, hold current weights)."""
        weights = current_weights.copy()
        trajectory = np.tile(weights, (horizon, 1))
        candidate = CandidateTrajectory(
            allocations=trajectory,
            expected_return=0.0,
            expected_risk=0.0,
            score=0.0,
            feasible=True,
        )
        return [candidate]

    def _default_trajectory_single(self) -> CandidateTrajectory:
        """Create fallback single trajectory."""
        return CandidateTrajectory(
            allocations=np.tile(self.default_weights, (5, 1)),
            expected_return=0.0,
            expected_risk=0.0,
            score=0.0,
            feasible=True,
        )
