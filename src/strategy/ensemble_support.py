"""Ensemble voter support surface (extracted from src/strategy/ensemble_voter.py,
Item 5 s2 ENSEMBLE-VOTER-SUPPORT-EXTRACT).

Holds the dataclasses, weighting helpers, and the bandit weighter that are
shared by the ensemble voter machinery. The hub module re-exports every name
defined here (F401 re-export hub pattern), so existing import paths and patch
targets on ``src.strategy.ensemble_voter`` keep resolving.
"""
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.paths import DATA_DIR
from src.signals.regime_spec import Regime, SignalReading
from src.utils import safe_get

logger = logging.getLogger(__name__)

_health_tracker = None

_health_tracker = None


def _get_health_tracker():
    global _health_tracker
    if _health_tracker is None:
        try:
            from src.signals.health_tracker import SignalHealthTracker
            _health_tracker = SignalHealthTracker()
        except (ImportError, OSError, KeyError, ValueError, TypeError) as e:
            logger.warning("SignalHealthTracker unavailable: %s", e)
    return _health_tracker


@dataclass
class EnsembleVote:
    """Aggregated ensemble decision."""
    timestamp: str
    regime: Regime
    regime_confidence: float
    
    # Consensus metrics
    num_sources: int
    weighted_consensus: float  # -1 to +1
    agreement_ratio: float     # % of signals agreeing with consensus
    
    # Per-asset recommendations
    equity_bias: float      # SPY direction
    duration_bias: float    # TLT direction
    gold_bias: float        # GLD direction
    
    # Final recommendation
    action: str            # "increase_equity", "decrease_equity", "neutral", "risk_off"
    confidence: float      # 0-1
    reasoning: str

    # Source breakdown
    source_votes: List[SignalReading]

    # Diversity diagnostics
    n_eff: float = 0.0          # Effective number of signals (exp of Shannon entropy)
    weight_entropy: float = 0.0 # Shannon entropy of weight distribution (nats)

    # Regime-conditional diagnostics (v2.60)
    regime_multipliers: Optional[Dict[str, float]] = None

    # Runtime disclosure for adaptive-learning branches.
    adaptive_learning: Dict[str, Any] = field(default_factory=dict)

    # Batch CW: health-gate sleep map (source → reason) for dashboard disclosure
    health_gate_slept: Optional[Dict[str, str]] = None
    health_gate_freeze: bool = False
    # Batch CX: regime-gate map (source → reason) — intentional OFF regimes
    regime_gated: Optional[Dict[str, str]] = None
    # Batch DU: unhealthy/degraded soft-floor arms still contributing (disclosure)
    health_gate_soft_floor: Optional[Dict[str, str]] = None


_REGIME_CONDITIONAL_WEIGHTS_DEFAULTS = {
    "CRISIS": {
        "alternative_data": 1.3,
        "unified_overlay": 0.3,
        "cross_asset_rv": 0.5,
        "cross_asset_regime_arb": 1.2,
        "international_momentum": 0.7,
        "multi_timeframe_fusion": 1.0,
    },
    "HIGH_VOL": {
        "unified_overlay": 1.2,
        "cross_asset_rv": 1.1,
        "cross_asset_regime_arb": 1.1,
        "international_momentum": 0.8,
        "alternative_data": 1.1,
        "multi_timeframe_fusion": 1.0,
    },
    "NORMAL": {
        "multi_timeframe_fusion": 1.0,
    },
    "LOW_VOL": {
        "international_momentum": 1.2,
        "cross_asset_regime_arb": 0.5,
        "unified_overlay": 0.7,
        "alternative_data": 0.8,
        "multi_timeframe_fusion": 1.0,
    },
    "RECOVERY": {
        "international_momentum": 1.3,
        "alternative_data": 1.1,
        "cross_asset_rv": 0.8,
        "multi_timeframe_fusion": 1.0,
    },
}


def _load_regime_conditional_weights(
    weights_file: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Load REGIME_CONDITIONAL_WEIGHTS from JSON config file.

    Supports ENSEMBLE_CONDITIONAL_WEIGHTS_FILE env var override.
    Falls back to hardcoded defaults if the file doesn't exist or is invalid.
    """
    if weights_file is None:
        weights_file = os.environ.get(
            "ENSEMBLE_CONDITIONAL_WEIGHTS_FILE",
            str(DATA_DIR / "regime_conditional_weights.json"),
        )
    weights_path = Path(weights_file)

    if not weights_path.exists():
        logger.info(
            "Regime conditional weights file not found at %s, using hardcoded defaults",
            weights_path,
        )
        return dict(_REGIME_CONDITIONAL_WEIGHTS_DEFAULTS)

    try:
        with open(weights_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Failed to load regime conditional weights from %s: %s, using hardcoded defaults",
            weights_path, e,
        )
        return dict(_REGIME_CONDITIONAL_WEIGHTS_DEFAULTS)

    logger.info(
        "Loaded regime conditional weights from %s (%d regimes)",
        weights_path, len(raw),
    )
    return raw


def _extract_signal_predictions(ic_data: Dict) -> Dict[str, List[float]]:
    """Extract prediction series from IC monitor state, preserving invalid slots."""
    signal_predictions: Dict[str, List[float]] = {}
    for signal_name, observations in ic_data.items():
        if signal_name == "__staged__" or not isinstance(observations, list):
            continue

        preds: List[float] = []
        for obs in observations:
            if not isinstance(obs, (list, tuple)) or len(obs) < 1:
                preds.append(float("nan"))
                continue
            try:
                pred = float(obs[0])
            except (TypeError, ValueError):
                preds.append(float("nan"))
                continue
            preds.append(pred if np.isfinite(pred) else float("nan"))

        if int(np.isfinite(np.asarray(preds, dtype=float)).sum()) >= 10:
            signal_predictions[signal_name] = preds
    return signal_predictions


def _rank_prediction_matrix(signal_predictions: Dict[str, List[float]]) -> Tuple[List[str], np.ndarray]:
    """Build a padded matrix of per-signal ranks, ranking each signal once."""
    signals = sorted(signal_predictions.keys())
    max_len = max(len(signal_predictions[signal]) for signal in signals)
    ranks = np.full((len(signals), max_len), np.nan, dtype=float)

    for row_idx, signal in enumerate(signals):
        values = np.asarray(signal_predictions[signal], dtype=float)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if finite_values.size < 5 or np.ptp(finite_values) < 1e-10:
            continue
        row_ranks = np.argsort(np.argsort(finite_values)).astype(float)
        rank_row = ranks[row_idx, : values.size]
        rank_row[finite_mask] = row_ranks

    return signals, ranks


def _rank_correlation_from_matrix(ranks: np.ndarray, i: int, j: int) -> float:
    """Compute Pearson correlation of two pre-ranked rows."""
    x_rank = ranks[i]
    y_rank = ranks[j]
    mask = np.isfinite(x_rank) & np.isfinite(y_rank)
    if int(mask.sum()) < 5:
        return 0.0

    x = x_rank[mask]
    y = y_rank[mask]
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    denominator = np.sqrt((x_dev ** 2).sum() * (y_dev ** 2).sum())
    if denominator < 1e-10:
        return 0.0
    return float((x_dev * y_dev).sum() / denominator)


def compute_signal_correlation_matrix(
    ic_data: Optional[Dict] = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Compute pairwise signal prediction correlation matrix from IC decay data.

    Reads from the ICMonitor persisted state to get aligned prediction series
    for each signal. Computes pairwise Spearman rank correlation of predictions
    to detect redundant signals — two signals whose predictions are highly
    correlated (>threshold) provide overlapping information and should have
    their ensemble weights penalized.

    Args:
        ic_data: Optional pre-loaded ICMonitor state dict (for testing).
                 If None, reads from DATA_DIR/ic_monitor_state.json.
        threshold: Correlation above this flags a pair as redundant.

    Returns:
        Dict with keys:
        - matrix: nested dict {s1: {s2: corr, ...}, ...}
        - redundant_pairs: list of (s1, s2, correlation) tuples
        - correlation_penalties: per-signal penalty factor 1/(1+mean_abs_corr)
    """
    # Load IC state
    if ic_data is None:
        state_path = DATA_DIR / "ic_monitor_state.json"
        if not state_path.exists():
            return {"matrix": {}, "redundant_pairs": [], "correlation_penalties": {}}
        try:
            with open(state_path) as f:
                ic_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"matrix": {}, "redundant_pairs": [], "correlation_penalties": {}}

    if not ic_data:
        return {"matrix": {}, "redundant_pairs": [], "correlation_penalties": {}}

    signal_predictions = _extract_signal_predictions(ic_data)

    if len(signal_predictions) < 2:
        return {"matrix": {}, "redundant_pairs": [], "correlation_penalties": {}}

    signals, ranks = _rank_prediction_matrix(signal_predictions)
    matrix: Dict[str, Dict[str, float]] = {}
    redundant_pairs: List[Tuple[str, str, float]] = []

    for i, s1 in enumerate(signals):
        matrix[s1] = {}
        for j, s2 in enumerate(signals):
            if i >= j:
                continue
            corr = _rank_correlation_from_matrix(ranks, i, j)
            matrix[s1][s2] = corr
            if abs(corr) > threshold:
                redundant_pairs.append((s1, s2, round(corr, 4)))

    # Compute per-signal correlation penalty factor
    # penalty = 1 / (1 + mean_abs_correlation_with_others)
    # Reduces weight for signals highly correlated with peers
    correlation_penalties: Dict[str, float] = {}
    for s1 in signals:
        correlations: List[float] = []
        for s2 in signals:
            if s1 == s2:
                continue
            if s1 in matrix and s2 in matrix[s1]:
                correlations.append(abs(matrix[s1][s2]))
            elif s2 in matrix and s1 in matrix[s2]:
                correlations.append(abs(matrix[s2][s1]))
        if correlations:
            mean_corr = sum(correlations) / len(correlations)
            correlation_penalties[s1] = round(1.0 / (1.0 + mean_corr), 6)
        else:
            correlation_penalties[s1] = 1.0

    return {
        "matrix": matrix,
        "redundant_pairs": redundant_pairs,
        "correlation_penalties": correlation_penalties,
    }


class BanditWeighter:
    """Thompson Sampling contextual bandit for dynamic signal weight adaptation.

    Tracks per-signal reward distribution using Gaussian-Gamma conjugate priors.
    Thompson Sampling samples from posterior to balance exploration/exploitation,
    converging 2-3x faster than epsilon-greedy in cold-start (<21 observations).

    Falls back to epsilon-greedy when posterior is uninformative (<2 observations).
    Softmax converts sampled Sharpe estimates to weights.

    No external dependencies beyond numpy (already imported).
    """
    def __init__(
        self,
        signals: List[str],
        epsilon: float = 0.1,
        window: int = 252,
        temperature: float = 1.0,
    ):
        self.signals = list(signals)
        self.epsilon = epsilon
        self.window = window
        self.temperature = temperature
        # _history[regime][signal] = list of daily returns (rolling window)
        self._history: dict = {}
        # Thompson Sampling priors: Gaussian-Gamma conjugate
        # mu_0, lambda_0 (prior precision scaling), alpha_0, beta_0
        self._mu_0 = 0.0
        self._lambda_0 = 1.0
        self._alpha_0 = 2.0  # shape — weak prior
        self._beta_0 = 1.0   # rate — weak prior

    def select(self, regime: str) -> str:
        """Select a signal using Thompson Sampling with epsilon-greedy fallback."""
        # Epsilon-greedy exploration (small probability of random)
        if random.random() < self.epsilon:
            return random.choice(self.signals)

        # Thompson Sampling: sample Sharpe from posterior for each signal
        sampled_sharpes = {}
        has_sufficient_data = False
        for sig in self.signals:
            n = len(safe_get(self._history, regime, sig, default=[]))
            if n >= 2:
                has_sufficient_data = True
                sampled_sharpes[sig] = self._sample_sharpe(sig, regime)
            else:
                sampled_sharpes[sig] = random.gauss(0.0, 1.0)  # uninformative prior

        # If no signal has sufficient data, fall back to rolling Sharpe
        if not has_sufficient_data:
            best_signal = self.signals[0]
            best_sharpe = -float("inf")
            for sig in self.signals:
                sh = self._rolling_sharpe(sig, regime)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_signal = sig
            return best_signal

        return max(sampled_sharpes, key=sampled_sharpes.get)

    def _sample_sharpe(self, signal: str, regime: str) -> float:
        """Sample a Sharpe ratio from the Gaussian-Gamma posterior."""
        hist = safe_get(self._history, regime, signal, default=[])
        n = len(hist)
        if n < 2:
            return 0.0

        arr = np.array(hist[-self.window:])
        x_bar = np.mean(arr)

        # Posterior parameters (Gaussian-Gamma conjugate update)
        lambda_n = self._lambda_0 + n
        mu_n = (self._lambda_0 * self._mu_0 + n * x_bar) / lambda_n
        alpha_n = self._alpha_0 + n / 2.0
        beta_n = (self._beta_0
                  + 0.5 * np.sum((arr - x_bar) ** 2)
                  + (self._lambda_0 * n * (x_bar - self._mu_0) ** 2)
                  / (2.0 * lambda_n))

        # Sample precision tau ~ Gamma(alpha_n, beta_n)
        # numpy gamma uses shape/scale, so scale = 1/beta_n
        if beta_n > 1e-10 and alpha_n > 0:
            tau = np.random.gamma(alpha_n, 1.0 / beta_n)
        else:
            tau = 1.0  # fallback

        # Sample mean mu ~ Normal(mu_n, 1/(lambda_n * tau))
        if tau > 1e-10:
            sigma_mu = 1.0 / np.sqrt(lambda_n * tau)
            mu_sample = np.random.normal(mu_n, sigma_mu)
        else:
            mu_sample = mu_n

        # Convert sampled mean to annualized Sharpe
        # Sharpe = mu / sigma * sqrt(252), and sigma = 1/sqrt(tau)
        if tau > 1e-10:
            sigma = 1.0 / np.sqrt(tau)
            return float(mu_sample / sigma * np.sqrt(252))
        return 0.0

    def update(self, signal: str, regime: str, daily_return: float):
        """Record a daily return observation for a signal in a regime."""
        if regime not in self._history:
            self._history[regime] = {}
        if signal not in self._history[regime]:
            self._history[regime][signal] = []
        self._history[regime][signal].append(daily_return)
        # Trim to window
        if len(self._history[regime][signal]) > self.window:
            self._history[regime][signal] = \
                self._history[regime][signal][-self.window:]

    def get_weights(self, regime: str) -> dict | None:
        """Get softmax-normalized weights for all signals in a regime.

        Returns None if no data exists for this regime (cold start).
        Returns dict mapping signal_name -> weight (sums to 1.0).
        """
        if regime not in self._history:
            return None
        sharpes = {}
        for sig in self.signals:
            sharpes[sig] = self._rolling_sharpe(sig, regime)
        return self._softmax(sharpes)

    def get_state(self) -> dict:
        """Serialize bandit history for durable persistence."""
        # Copy nested lists so callers cannot mutate internal state
        history = {
            regime: {sig: list(returns) for sig, returns in signals.items()}
            for regime, signals in self._history.items()
        }
        return {
            "schema_version": "bandit-weighter/v1",
            "signals": list(self.signals),
            "epsilon": self.epsilon,
            "window": self.window,
            "temperature": self.temperature,
            "history": history,
        }

    def load_state(self, state: dict) -> None:
        """Restore bandit history from get_state() payload."""
        if not isinstance(state, dict):
            return
        history = state.get("history")
        if not isinstance(history, dict):
            return
        restored: dict = {}
        for regime, signals in history.items():
            if not isinstance(signals, dict):
                continue
            restored[str(regime)] = {}
            for sig, returns in signals.items():
                if not isinstance(returns, list):
                    continue
                cleaned = []
                for r in returns[-self.window :]:
                    try:
                        cleaned.append(float(r))
                    except (TypeError, ValueError):
                        continue
                if cleaned:
                    restored[str(regime)][str(sig)] = cleaned
        self._history = restored

    def _rolling_sharpe(self, signal: str, regime: str) -> float:
        """Compute rolling Sharpe ratio for a signal in a regime."""
        hist = safe_get(self._history, regime, signal, default=[])
        if len(hist) < 21:  # Need at least 1 month
            return 0.0
        arr = np.array(hist[-self.window:])
        mu = np.mean(arr)
        sigma = np.std(arr)
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    def _softmax(self, sharpes: dict) -> dict:
        """Convert Sharpe estimates to weights via softmax."""
        values = np.array([sharpes[s] for s in self.signals])
        # Subtract max for numerical stability
        values = values - np.max(values)
        if self.temperature > 0:
            values = values / self.temperature
        exp_values = np.exp(values)
        total = np.sum(exp_values)
        if total < 1e-10:
            # All equal if everything is zero
            n = len(self.signals)
            w = 1.0 / n if n > 0 else 0.0
            return {s: w for s in self.signals}
        return {sig: float(exp_values[i] / total)
                for i, sig in enumerate(self.signals)}
