"""
Portfolio-Lab v2.58: Ensemble Signal Voter

Multi-source signal aggregation with regime-dependent weighting and health-adjusted weighting.
Implements soft voting with confidence-based consensus for portfolio decisions.

Active Sources (6):
- Multi-Speed Momentum (v2.56) - Speed-diversified trends
- Cross-Asset Relative Value (v5.71) - Mean-reversion triggers
- International Equity Momentum (v3.13) - EFA/EEM vs SPY
- Alternative Data (v9.00) - SEC EDGAR, NewsAPI, jobs
- Cross-Asset Regime Arbitrage (v8.09) - Divergence detection
- Unified Overlay (v4.90) - Collar + bond + crypto + calendar

Weight Adjustments (applied in order):
1. Static REGIME_WEIGHTS (per-regime allocation)
2. Adaptive ensemble weighting (v6.09, from attribution data)
3. Health-adjusted weighting (v3.12, from signal health scores)
4. Correlation penalty (v2.59, from IC prediction correlations)
5. Regime-conditional weights (v2.60, per-regime signal multipliers)
6. Utility-based reweighting (v2.58, Sharpe contribution + hit rate)
7. Exploration noise (v2.57, Dirichlet sampling)
8. Turnover-aware validation (v8.01, with basis-pursuit + regret-weighted)

# Online IC-based weight learning (new, gated by ENSEMBLE_USE_IC_WEIGHTS)

Consensus threshold: 2/3 weighted signals agree for action

Usage:
    python -m src.strategy.ensemble_voter vote
    python -m src.strategy.ensemble_voter recommend --portfolio 46/38/16
    python -m src.strategy.ensemble_voter explain
"""

import json
import os
import random
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

from src.paths import (
    DATA_DIR, PRICES_JSON, ATTRIBUTION_DIR, BASE_ALLOCATION, sqlite_connect,
    ENSEMBLE_CRISIS_VOL_THRESHOLD, ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD,
    ENSEMBLE_HIGH_VOL_VOL_THRESHOLD, ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD,
    ENSEMBLE_LOW_VOL_VOL_THRESHOLD, ENSEMBLE_LOW_VOL_MOM_THRESHOLD,
    ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD, ENSEMBLE_RECOVERY_MOM_THRESHOLD,
    ENSEMBLE_CONSENSUS_THRESHOLD,
)
from src.data.price_cache import get_prices, get_prices_df
from src.utils import safe_get
from src.utils.computation_cache import get_realized_volatility
from src.strategy.signal_aggregator import SignalAggregator


__all__ = ['Regime', 'SignalSource', 'SignalReading', 'EnsembleVote', 'REGIME_WEIGHTS', 'REGIME_CONDITIONAL_WEIGHTS', 'REGIME_CONSENSUS_THRESHOLDS', 'DEFAULT_DIVERSITY_FLOOR', 'BanditWeighter', 'EnsembleVoter', 'compute_signal_correlation_matrix']

logger = logging.getLogger(__name__)

# Bandit blend parameters — controls static/bandit weight mixing
# Starts 100% static, shifts to (1-BANDIT_MAX_BLEND)/BANDIT_MAX_BLEND after warmup
BANDIT_MAX_BLEND: float = float(os.environ.get("ENSEMBLE_BANDIT_MAX_BLEND", "0.7"))
BANDIT_WARMUP_DAYS: int = int(os.environ.get("ENSEMBLE_BANDIT_WARMUP_DAYS", "252"))
# Skip apply_daily_bandit_rewards when |daily_return| is below this floor so
# flat-NAV micro-noise (~1e-8 from performance.jsonl) does not advance
# observations/reward_days or pollute arm history (sleeping-experts / noise floor).
BANDIT_REWARD_NOISE_FLOOR: float = float(
    os.environ.get("ENSEMBLE_BANDIT_REWARD_NOISE_FLOOR", "1e-6")
)

# Diversity floor — minimum weight for each active signal to prevent
# weight concentration. Improves N_eff (effective signal count) by ensuring
# no signal is completely zeroed out by the weight pipeline.
# Set to 0 to disable. Range: 0.02-0.08 recommended.
DEFAULT_DIVERSITY_FLOOR: float = float(os.environ.get("ENSEMBLE_DIVERSITY_FLOOR", "0.05"))

# Regime-conditional consensus thresholds
# CRISIS: lower threshold (act faster with fewer signals)
# NORMAL: higher threshold (require more consensus when time permits)
# Falls back to ENSEMBLE_CONSENSUS_THRESHOLD env var for unlisted regimes
REGIME_CONSENSUS_THRESHOLDS: dict = {
    "CRISIS": 0.50,
    "HIGH_VOL": 0.55,
    "LOW_VOL": 0.67,
    "NORMAL": 0.75,
    "RECOVERY": 0.60,
}

# Module-level health tracker singleton (lazy initialized)
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


class Regime(Enum):
    """Market regime classifications."""
    LOW_VOL = "low_vol"      # VIX < 15, calm bull market
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


from src.signals.signal_source import SignalSource  # canonical, consolidated May 2026


@dataclass
class SignalReading:
    """Single signal source reading."""
    source: SignalSource
    timestamp: str
    
    # Signal value: -1 (strong short) to +1 (strong long)
    value: float
    
    # Metadata
    confidence: float  # 0-1
    weight: float    # Dynamic regime weight
    regime_fit: str  # Which regime this signal works best in
    
    # Asset-specific signals (optional)
    asset_signals: Optional[Dict[str, float]] = None
    
    # Reasoning
    explanation: str = ""
    # Batch CV: inactive readings are kept for disclosure but vote weight forced 0
    is_active: bool = True
    # Batch DF: provenance for health tracker (pattern, polarity_policy, composite, …)
    metadata: Optional[Dict[str, Any]] = None


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


# Regime-dependent weights (6 active signals, renormalized per regime)
# MSM disabled (net-negative -0.012 Sharpe), weight redistributed to ALT_DATA and INTL_MOM.
# Weights sum=1.0 per regime.
#
# Loaded from JSON file at module init. Override path via ENSEMBLE_WEIGHTS_FILE env var.
# Falls back to hardcoded defaults if file is missing or invalid.


def _build_hardcoded_weights() -> Dict[Regime, Dict[SignalSource, float]]:
    """Return the hardcoded default regime weights (fallback)."""
    return {
        Regime.LOW_VOL: {
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1350,
            SignalSource.ALTERNATIVE_DATA: 0.2650,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2520,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.0000,  # marginal in calm markets
            SignalSource.UNIFIED_OVERLAY: 0.1980,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        },
        Regime.NORMAL: {
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.ALTERNATIVE_DATA: 0.2245,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2205,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1710,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        },
        Regime.HIGH_VOL: {
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.1890,
            SignalSource.ALTERNATIVE_DATA: 0.2470,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1800,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        },
        Regime.CRISIS: {
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.CROSS_ASSET_RV: 0.3285,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1530,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.0000,
            SignalSource.ALTERNATIVE_DATA: 0.1300,
            SignalSource.UNIFIED_OVERLAY: 0.2385,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        },
        Regime.RECOVERY: {
            SignalSource.MULTI_SPEED_MOM: 0.0000,
            SignalSource.ALTERNATIVE_DATA: 0.2245,
            SignalSource.CROSS_ASSET_RV: 0.1170,
            SignalSource.INTERNATIONAL_MOMENTUM: 0.2205,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.1170,
            SignalSource.UNIFIED_OVERLAY: 0.1710,
            SignalSource.MULTI_TIMEFRAME_FUSION: 0.1000,
            SignalSource.GOOGLE_TRENDS: 0.0500,
            SignalSource.VIX_TERM_STRUCTURE: 0.0500,  # v3.23: intraday vol timing
        }
    }


def _load_regime_weights() -> Dict[Regime, Dict[SignalSource, float]]:
    """Load REGIME_WEIGHTS from JSON config file.

    Supports ENSEMBLE_WEIGHTS_FILE env var override (same pattern as
    PAPER_CONFIG in evaluator.py). Falls back to hardcoded defaults
    if the file doesn't exist or contains invalid data.
    """
    weights_file = os.environ.get(
        "ENSEMBLE_WEIGHTS_FILE",
        str(DATA_DIR / "ensemble_weights.json")
    )
    weights_path = Path(weights_file)

    if not weights_path.exists():
        logger.info(
            "Ensemble weights file not found at %s, using hardcoded defaults",
            weights_path
        )
        return _build_hardcoded_weights()

    try:
        with open(weights_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Failed to load ensemble weights from %s: %s, using hardcoded defaults",
            weights_path, e
        )
        return _build_hardcoded_weights()

    regime_weights: Dict[Regime, Dict[SignalSource, float]] = {}
    for regime_name, sources in raw.items():
        # Batch CS: skip _meta / underscore metadata keys without warning noise
        if str(regime_name).startswith("_"):
            continue
        try:
            regime = Regime(regime_name)
        except ValueError:
            logger.warning(
                "Unknown regime '%s' in %s, skipping", regime_name, weights_path
            )
            continue

        regime_dict: Dict[SignalSource, float] = {}
        for source_name, weight in sources.items():
            try:
                source = SignalSource(source_name)
            except ValueError:
                logger.warning(
                    "Unknown signal source '%s' in %s, skipping",
                    source_name, weights_path
                )
                continue
            regime_dict[source] = weight

        regime_weights[regime] = regime_dict

    # Validate: all regimes should be present
    missing = [r.value for r in Regime if r not in regime_weights]
    if missing:
        logger.warning(
            "Missing regimes in %s: %s, falling back to hardcoded defaults",
            weights_path, missing
        )
        return _build_hardcoded_weights()

    logger.info(
        "Loaded ensemble weights from %s (%d regimes)",
        weights_path, len(regime_weights)
    )
    return regime_weights


REGIME_WEIGHTS = _load_regime_weights()


# ── Regime-Conditional Signal Weights ──

# Hardcoded defaults — same as data/regime_conditional_weights.json
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


REGIME_CONDITIONAL_WEIGHTS = _load_regime_conditional_weights()


# ── Signal Correlation Matrix ──


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


# ── Epsilon-Greedy Contextual Bandit for Dynamic Signal Weighting ──

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


class EnsembleVoter:
    """
    Multi-source signal ensemble with regime-adaptive weighting.

    Collects signals from all strategy modules, applies regime-dependent
    weighting, and produces consensus recommendations.
    """

    # Regime detection thresholds (centralized in src/paths.py, env-var configurable)
    CRISIS_VOL_THRESHOLD = ENSEMBLE_CRISIS_VOL_THRESHOLD
    CRISIS_DRAWDOWN_THRESHOLD = ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD
    HIGH_VOL_VOL_THRESHOLD = ENSEMBLE_HIGH_VOL_VOL_THRESHOLD
    HIGH_VOL_DRAWDOWN_THRESHOLD = ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD
    HIGH_VOL_MOM_THRESHOLD = 0.0       # Negative momentum with drawdown → HIGH_VOL
    LOW_VOL_VOL_THRESHOLD = ENSEMBLE_LOW_VOL_VOL_THRESHOLD
    LOW_VOL_MOM_THRESHOLD = ENSEMBLE_LOW_VOL_MOM_THRESHOLD
    RECOVERY_DRAWDOWN_THRESHOLD = ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD
    RECOVERY_MOM_THRESHOLD = ENSEMBLE_RECOVERY_MOM_THRESHOLD
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        regime_detector: Optional[str] = None
    ):
        self.data_path = data_path or DATA_DIR
        self.db_path = self.data_path / "ensemble_signals.db"
        self._init_db()

        # Current readings cache
        self.current_readings: Dict[SignalSource, SignalReading] = {}
        self.current_regime: Regime = Regime.NORMAL
        self.current_regime_confidence: float = 0.5

        # Bandit weighter for dynamic signal weight adaptation
        self.bandit = BanditWeighter(
            signals=[s.value for s in SignalSource],
            epsilon=0.1,
            window=252,
        )
        self.bandit_observations: int = 0
        # Calendar reward steps for warmup blend (not arm×day updates)
        self.bandit_days: int = 0
        self.bandit_state_path = self.data_path / "ensemble_bandit_state.json"
        self._load_bandit_state()

        # Online IC weighter for IC-based ensemble weight learning
        # Gated by ENSEMBLE_USE_IC_WEIGHTS env var (default: off)
        self._use_ic_weights = os.environ.get("ENSEMBLE_USE_IC_WEIGHTS", "0").lower() in ("1", "true")
        self._ic_weighter = None
        if self._use_ic_weights:
            try:
                from src.strategy.online_ic_weighter import OnlineICWeighter
                self._ic_weighter = OnlineICWeighter()
                # Load persisted IC weighter state if available
                ic_weighter_state = self.data_path / "ic_weighter_state.json"
                if ic_weighter_state.exists():
                    with open(ic_weighter_state) as f:
                        self._ic_weighter.load_state(json.load(f))
                    logger.info("OnlineICWeighter state loaded from %s", ic_weighter_state)
            except Exception as e:
                logger.warning("Failed to initialize OnlineICWeighter: %s", e)
                self._ic_weighter = None

        # Regime gate — disables signals in regimes where they are net-negative
        from src.signals.regime_gate import RegimeGate
        self.regime_gate = RegimeGate()

        # Load data-driven gate rules if available (computed by DashboardGenerator)
        try:
            from src.monitor.regime_sharpe_matrix import load_persisted_gate_rules
            persist_path = self.data_path / "regime_gate_persisted.json"
            data_rules = load_persisted_gate_rules(persist_path)
            if data_rules:
                self.regime_gate.gate_rules.update(data_rules)
                logger.info(
                    "Loaded %d data-driven gate rules from persisted file",
                    len(data_rules),
                )
        except (ImportError, Exception) as e:
            logger.debug("Data-driven gate loading skipped: %s", e)

        self._prev_regime: Optional[str] = None
        self._days_in_regime: int = 999  # Start assuming stable regime

        # Signal collection collaborator (extractable / injectable for tests)
        self.signal_aggregator = SignalAggregator(
            # Lambda so instance patches of _load_price_data still apply.
            load_price_data=lambda: self._load_price_data(),
            regime_weights=REGIME_WEIGHTS,
        )

    def _init_db(self):
        """Initialize signal history database."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        with sqlite_connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_votes (
                    timestamp TEXT PRIMARY KEY,
                    regime TEXT,
                    regime_confidence REAL,
                    num_sources INTEGER,
                    consensus REAL,
                    agreement_ratio REAL,
                    equity_bias REAL,
                    duration_bias REAL,
                    gold_bias REAL,
                    action TEXT,
                    confidence REAL,
                    reasoning TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_readings (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    source TEXT,
                    value REAL,
                    confidence REAL,
                    weight REAL,
                    regime_fit TEXT,
                    explanation TEXT
                )
            """)
    
    def detect_regime(self, price_data: Optional[pd.DataFrame] = None) -> Tuple[Regime, float]:
        """
        Detect current market regime from available data.
        
        Uses simple heuristics (can be enhanced with HMM later):
        - Crisis: VIX > 30 or max drawdown > 10% over 20 days
        - High vol: VIX > 20 or vol of vol elevated
        - Recovery: Recent drawdown followed by positive momentum
        - Normal: Otherwise
        """
        if price_data is None:
            price_data = self._load_price_data()
        
        if price_data is None or price_data.empty:
            return Regime.NORMAL, 0.5
        
        # Compute key indicators
        spy = price_data.get('SPY', price_data.iloc[:, 0])

        if len(spy) < 21:
            return Regime.NORMAL, 0.5

        # 20-day realized vol (annualized, TTL-cached)
        vol_20d = get_realized_volatility(spy, window=20)
        if vol_20d is None:
            return Regime.NORMAL, 0.5

        # Returns for momentum and drawdown calculations
        returns = spy.pct_change().dropna()
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1).iloc[-1]
        
        # 20-day momentum
        mom_20d = returns.tail(20).sum()
        
        # Regime detection
        if vol_20d > self.CRISIS_VOL_THRESHOLD or drawdown < self.CRISIS_DRAWDOWN_THRESHOLD:
            regime = Regime.CRISIS
            confidence = min(abs(drawdown) * 5, 0.9) if drawdown < self.HIGH_VOL_DRAWDOWN_THRESHOLD else 0.5
        elif vol_20d > self.HIGH_VOL_VOL_THRESHOLD or (drawdown < self.HIGH_VOL_DRAWDOWN_THRESHOLD and mom_20d < self.HIGH_VOL_MOM_THRESHOLD):
            regime = Regime.HIGH_VOL
            confidence = min(vol_20d * 3, 0.8)
        elif drawdown < self.RECOVERY_DRAWDOWN_THRESHOLD and mom_20d > self.RECOVERY_MOM_THRESHOLD:
            regime = Regime.RECOVERY
            confidence = min(mom_20d * 20, 0.7)
        elif vol_20d < self.LOW_VOL_VOL_THRESHOLD and mom_20d > self.LOW_VOL_MOM_THRESHOLD:
            regime = Regime.LOW_VOL
            confidence = max(0.5, 1.0 - vol_20d * 4)
        else:
            regime = Regime.NORMAL
            confidence = max(0.5, 1.0 - vol_20d * 2)
        
        return regime, confidence
    
    def _load_price_data(self) -> Optional[pd.DataFrame]:
        """Load price data from JSON (TTL-cached)."""
        try:
            df = get_prices_df()
            if df.empty:
                return None
            return df
        except (FileNotFoundError, ValueError):
            return None
    
    def collect_signals(self, date: Optional[str] = None, regime: Optional[Regime] = None) -> Dict[SignalSource, SignalReading]:
        """
        Collect signals from active sources.

        If regime is provided, skip signal sources with zero weight for that
        regime — avoids wasted computation on signals that won't affect the vote.

        Active sources (7 signals):
        - Multi-speed momentum (primary trend signal)
        - Cross-asset relative value (mean-reversion triggers)
        - International equity momentum (EFA/VXUS trend)
        - Alternative data (SEC EDGAR, NewsAPI, jobs)
        - Cross-asset regime arbitrage (divergence detection)
        - Unified overlay (collar + bond + crypto + calendar)
        - Multi-timeframe fusion (v806 redo — timeframe decomposition)

        Collection is delegated to ``self.signal_aggregator`` so the collaborator
        can be injected or stubbed without rewriting vote logic.
        """
        aggregator = self._ensure_signal_aggregator()
        readings = aggregator.collect(date=date, regime=regime)
        self.current_readings = readings
        return readings

    def _ensure_signal_aggregator(self):
        """Return the signal aggregator, creating a default if missing.

        Fixtures that construct via ``EnsembleVoter.__new__`` never run
        ``__init__``; keep collection working for those paths and for
        intentional late injection.
        """
        aggregator = getattr(self, "signal_aggregator", None)
        if aggregator is None:
            aggregator = SignalAggregator(
                load_price_data=lambda: self._load_price_data(),
                regime_weights=REGIME_WEIGHTS,
            )
            self.signal_aggregator = aggregator
        return aggregator

    def _should_skip(self, source: SignalSource, active_sources, regime: Optional[Regime]) -> bool:
        """Check if a signal source should be skipped for the current regime."""
        return self._ensure_signal_aggregator().should_skip(source, active_sources, regime)

    def _collect_msm_signal(self, readings: Dict, active_sources, regime: Optional[Regime], date: Optional[str]) -> None:
        """Collect multi-speed momentum signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_msm_signal(readings, active_sources, regime, date)

    def _collect_cross_asset_rv_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect cross-asset relative value signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_cross_asset_rv_signal(readings, active_sources, regime)

    def _collect_intl_momentum_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect international equity momentum signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_intl_momentum_signal(readings, active_sources, regime)

    def _collect_alt_data_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect alternative data signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_alt_data_signal(readings, active_sources, regime)

    def _collect_regime_arb_signal(self, readings: Dict, active_sources, regime: Optional[Regime]) -> None:
        """Collect cross-asset regime arbitrage signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_regime_arb_signal(readings, active_sources, regime)

    def _collect_unified_overlay_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
    ) -> None:
        """Collect unified overlay signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_unified_overlay_signal(readings, active_sources, regime)

    def _collect_mtf_signal(
        self, readings: Dict, active_sources, regime: Optional[Regime],
        date: Optional[str] = None,
    ) -> None:
        """Collect multi-timeframe fusion signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_mtf_signal(readings, active_sources, regime, date)

    def _collect_google_trends(
        self, readings: dict, active_sources: set, regime, date: str
    ) -> None:
        """Collect Google Trends sentiment signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_google_trends(readings, active_sources, regime)

    def _collect_vix_term_structure_signal(self, readings: dict, active_sources: set, regime) -> None:
        """Collect VIX term structure signal (compat wrapper over aggregator)."""
        self._ensure_signal_aggregator()._collect_vix_term_structure_signal(readings, active_sources, regime)

    @staticmethod
    def _static_zero_baseline_sources(regime_name: str) -> set:
        """SignalSource keys with intentional REGIME_WEIGHTS soft-delete (weight 0).

        Soft-delete is economic/ADR policy — bandit blend, adaptive floors,
        exploration noise, and diversity floors must not reinflate these arms
        until a human promotes non-zero static weights.
        """
        regime_enum = getattr(Regime, str(regime_name).upper(), Regime.NORMAL)
        static = REGIME_WEIGHTS.get(regime_enum, {}) or {}
        zeros: set = set()
        for src, w in static.items():
            try:
                if float(w or 0.0) <= 0.0:
                    zeros.add(src)
            except (TypeError, ValueError):
                zeros.add(src)
        return zeros

    @staticmethod
    def _pin_zero_baseline_weights(weights: Dict, regime_name: str) -> Dict:
        """Force soft-delete arms to 0 and renormalize remaining mass.

        Batch DK: bandit ε-mass + adaptive min_weight + Dirichlet exploration
        previously reinflated multi_speed_momentum (~5–13% vote mass) despite
        REGIME_WEIGHTS soft-delete. Pin after each reinflation-capable stage.
        """
        if not weights:
            return weights
        zeros = EnsembleVoter._static_zero_baseline_sources(regime_name)
        if not zeros:
            return weights
        pinned = dict(weights)
        changed = False
        for src in zeros:
            if src in pinned and float(pinned.get(src) or 0.0) != 0.0:
                pinned[src] = 0.0
                changed = True
            elif src in pinned:
                pinned[src] = 0.0
        if not changed and all(
            float(pinned.get(s) or 0.0) == 0.0 for s in zeros if s in pinned
        ):
            # Still renorm if zeros already 0 but total drifted
            total = sum(float(v or 0.0) for v in pinned.values())
            if total > 0 and abs(total - 1.0) > 1e-9:
                return {k: float(v or 0.0) / total for k, v in pinned.items()}
            return pinned
        total = sum(float(v or 0.0) for v in pinned.values())
        if total > 0:
            pinned = {k: float(v or 0.0) / total for k, v in pinned.items()}
        return pinned

    def get_blended_weights(self, regime_name: str) -> dict:
        """Get regime weights blended between static REGIME_WEIGHTS and bandit.

        Starts 100% static (bandit_blend=0.0), gradually shifts toward
        up to 70% bandit after 252 days of observations.

        Batch DK: static-zero soft-delete arms stay at 0 after blend+renorm
        (bandit posterior must not reintroduce vote mass).
        """
        regime_enum = getattr(Regime, regime_name, Regime.NORMAL)
        static = dict(REGIME_WEIGHTS.get(regime_enum, {}))

        # If bandit not initialized (e.g. test fixtures bypassing __init__), fall back
        if not hasattr(self, 'bandit') or self.bandit is None:
            return static

        bandit = self.bandit.get_weights(regime_name)

        if bandit is None:
            return static  # Cold start: 100% static

        # Blend: starts 100% static, shifts to (1-MAX_BLEND)/MAX_BLEND after warmup
        day_steps = int(getattr(self, "bandit_days", 0) or 0)
        if day_steps <= 0 and int(getattr(self, "bandit_observations", 0) or 0) > 0:
            # Legacy states without bandit_days: approximate days from arm updates
            n_sources = max(1, len(list(SignalSource)))
            day_steps = max(1, int(self.bandit_observations) // n_sources)
        blend = min(BANDIT_MAX_BLEND, day_steps / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND)

        # Convert static keys from SignalSource enum to string values for matching
        static_by_value = {k.value: v for k, v in static.items()}

        blended = {}
        for sig_value in static_by_value:
            bandit_w = bandit.get(sig_value, 0.0)
            static_w = static_by_value[sig_value]
            # Hard-pin soft-delete: never mix bandit mass into static-zero arms
            if float(static_w or 0.0) <= 0.0:
                blended[sig_value] = 0.0
            else:
                blended[sig_value] = static_w * (1 - blend) + bandit_w * blend

        # Normalize to sum=1.0
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        # Convert back to SignalSource keys
        value_to_source = {s.value: s for s in SignalSource}
        out = {value_to_source[k]: v for k, v in blended.items() if k in value_to_source}
        return self._pin_zero_baseline_weights(out, regime_name)

    def get_adaptive_learning_status(self, regime_name: Optional[str] = None) -> Dict[str, Any]:
        """Disclose adaptive-learning branch status without changing weights."""
        if regime_name is None:
            current = getattr(self, "current_regime", Regime.NORMAL)
            regime_name = current.name if hasattr(current, "name") else str(current)

        observations = int(getattr(self, "bandit_observations", 0) or 0)
        reward_days = int(getattr(self, "bandit_days", 0) or 0)
        if reward_days <= 0 and observations > 0:
            n_sources = max(1, len(list(SignalSource)))
            reward_days = max(1, observations // n_sources)
        bandit = getattr(self, "bandit", None)
        bandit_status: Dict[str, Any] = {
            "status": "unavailable",
            "enabled": False,
            "observations": observations,
            "reward_days": reward_days,
            "days": reward_days,
            "warmup_days": BANDIT_WARMUP_DAYS,
            "max_blend": BANDIT_MAX_BLEND,
            "current_blend": 0.0,
            "reason": "bandit_weighter_unavailable",
        }

        if bandit is not None:
            bandit_status.update({
                "enabled": True,
                "status": "non_effective",
                "reason": "cold_start_no_regime_weights",
            })
            try:
                bandit_weights = bandit.get_weights(regime_name)
                if bandit_weights is not None:
                    blend = min(
                        BANDIT_MAX_BLEND,
                        reward_days / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND,
                    )
                    bandit_status["current_blend"] = round(blend, 4)
                    if blend > 0:
                        bandit_status["status"] = "active"
                        bandit_status["reason"] = "blending_with_static_weights"
                    else:
                        bandit_status["reason"] = "cold_start_no_observations"
            except (AttributeError, KeyError, ValueError, TypeError, OSError) as e:
                bandit_status.update({
                    "enabled": False,
                    "status": "unavailable",
                    "reason": f"bandit_status_error:{type(e).__name__}",
                })

        use_ic = bool(getattr(self, "_use_ic_weights", False))
        ic_weighter = getattr(self, "_ic_weighter", None)
        try:
            ic_blend_alpha = float(os.environ.get("ENSEMBLE_IC_WEIGHT_BLEND_ALPHA", "0.3"))
        except ValueError:
            ic_blend_alpha = 0.3

        online_ic_status: Dict[str, Any] = {
            "status": "disabled",
            "enabled": use_ic,
            "state_available": ic_weighter is not None,
            "blend_alpha": ic_blend_alpha,
            "reason": "env_disabled",
        }
        if use_ic and ic_weighter is None:
            online_ic_status.update({
                "status": "unavailable",
                "reason": "initialization_failed_or_unavailable",
            })
        elif use_ic and ic_weighter is not None:
            online_ic_status.update({
                "status": "active",
                "reason": "weighter_initialized",
            })

        last_ic_status = getattr(self, "_last_online_ic_learning_status", None)
        if isinstance(last_ic_status, dict):
            online_ic_status.update(last_ic_status)

        return {
            "bandit": bandit_status,
            "online_ic": online_ic_status,
        }

    def get_rebalance_config(self) -> Dict[str, Any]:
        """
        Return current regime and rebalancing parameters for the
        SmartRebalanceGate/Controller to use regime-adaptive thresholds.

        Returns:
            Dict with 'regime' key (e.g. 'normal', 'crisis', 'high_vol',
            'low_vol', 'recovery') for the rebalancing controller.
        """
        regime_map = {
            Regime.LOW_VOL: 'low_vol',
            Regime.NORMAL: 'normal',
            Regime.HIGH_VOL: 'high_vol',
            Regime.CRISIS: 'crisis',
            Regime.RECOVERY: 'recovery',
        }
        return {
            'regime': regime_map.get(self.current_regime, 'normal'),
            'regime_confidence': self.current_regime_confidence,
        }

    def update_bandit(self, signal_value: str, regime_name: str, daily_return: float):
        """Update bandit with observed return for a signal in a regime."""
        self.bandit.update(signal_value, regime_name, daily_return)
        self.bandit_observations += 1

    def _load_bandit_state(self) -> bool:
        """Load bandit history + observation count from data_path if present."""
        path = getattr(self, "bandit_state_path", None) or (
            self.data_path / "ensemble_bandit_state.json"
        )
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return False
            bandit_state = state.get("bandit") or state
            if hasattr(self.bandit, "load_state"):
                self.bandit.load_state(bandit_state)
            obs = state.get("observations")
            if obs is None:
                # Derive from history length if missing
                hist = getattr(self.bandit, "_history", {}) or {}
                obs = sum(
                    len(returns)
                    for signals in hist.values()
                    if isinstance(signals, dict)
                    for returns in signals.values()
                    if isinstance(returns, list)
                )
            self.bandit_observations = int(obs or 0)
            days = state.get("reward_days", state.get("bandit_days", state.get("days")))
            if days is None:
                if self.bandit_observations > 0:
                    n_sources = max(1, len(list(SignalSource)))
                    days = max(1, self.bandit_observations // n_sources)
                else:
                    days = 0
            self.bandit_days = int(days or 0)
            logger.info(
                "Loaded ensemble bandit state from %s (observations=%s, reward_days=%s)",
                path,
                self.bandit_observations,
                self.bandit_days,
            )
            return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to load ensemble bandit state: %s", exc)
            return False

    def save_bandit_state(self) -> bool:
        """Persist bandit history + observation count atomically."""
        path = getattr(self, "bandit_state_path", None) or (
            self.data_path / "ensemble_bandit_state.json"
        )
        payload = {
            "schema_version": "ensemble-bandit-state/v1",
            "observations": int(self.bandit_observations),
            "reward_days": int(getattr(self, "bandit_days", 0) or 0),
            "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
            "bandit": self.bandit.get_state() if hasattr(self.bandit, "get_state") else {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            tmp_path.replace(path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save ensemble bandit state: %s", exc)
            return False

    @staticmethod
    def contribution_reward_decimal(
        daily_return: float,
        *,
        value: float,
        weight: float,
    ) -> float:
        """Map one signal reading + portfolio daily return → arm reward (decimal).

        Batch BR: same credit formula as ``PerformanceAttribution._compute_source_attribution``
        (directional: ``ret * |value|``; neutral ``|value|<=0.05``: ``ret * weight * 2``),
        returned in decimal return units (not bps) for bandit updates.
        """
        ret = float(daily_return)
        val = float(value)
        w = float(weight)
        if abs(val) > 0.05:
            return ret * abs(val)
        return ret * w * 2.0

    @staticmethod
    def compute_daily_contribution_rewards(
        signals: List[Dict[str, Any]],
        daily_return: float,
        *,
        min_spread: float = 1e-12,
    ) -> Optional[Dict[str, float]]:
        """Build identifying per-source rewards for one calendar day.

        Uses the latest reading per source in ``signals``. Returns None when
        fewer than two sources or zero reward spread (non-identification).
        """
        try:
            ret = float(daily_return)
        except (TypeError, ValueError):
            return None
        by_source: Dict[str, Dict[str, Any]] = {}
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            name = sig.get("source")
            if name is None:
                continue
            src = str(name)
            # Last write wins (callers should pass chronological order)
            by_source[src] = sig
        if len(by_source) < 2:
            return None
        rewards: Dict[str, float] = {}
        for src, sig in by_source.items():
            try:
                value = float(sig.get("value", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                weight = float(sig.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            rewards[src] = EnsembleVoter.contribution_reward_decimal(
                ret, value=value, weight=weight
            )
        if len(rewards) < 2:
            return None
        vals = list(rewards.values())
        if max(vals) - min(vals) < min_spread:
            return None
        return rewards

    @staticmethod
    def load_daily_contribution_source_rewards(
        data_dir: Optional[Path] = None,
        *,
        lookback_days: int = 14,
    ) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        """Load per-source rewards from *one* recent day of signal × PnL credit.

        Batch BR (B1): prefers true daily contribution over windowed
        ``avg_return_bps`` (Batch BQ). Joins ``source_readings`` (latest per
        source/day) with paper daily returns; walks newest dates first until
        an identifying multi-arm map is found.

        Returns ``(rewards, meta)`` or None. Meta includes ``as_of_date``,
        ``reward_mode``, ``live_authoritative: false``. Hermetic when
        ``data_dir`` is an explicit tmp path (no live DATA_DIR leak).
        """
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        lookback = max(int(lookback_days), 2)

        # Lazy import avoids pulling attribution/numpy heavy paths at module load
        try:
            from src.monitor.performance_attribution import PerformanceAttribution
        except ImportError:
            logger.debug("PerformanceAttribution unavailable for daily contribution rewards")
            return None

        try:
            pa = PerformanceAttribution(data_dir=root)
            history = pa._get_signal_history(days=lookback)
            daily_returns = pa._get_paper_trading_returns(days=lookback)
        except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
            logger.debug("Daily contribution load failed: %s", exc)
            return None

        if not history or not daily_returns:
            return None

        # Group source readings by calendar date (latest timestamp per source/day)
        by_day: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in history:
            if not isinstance(row, dict) or row.get("type") == "ensemble_vote":
                continue
            ts = row.get("timestamp")
            src = row.get("source")
            if not ts or not src:
                continue
            day = str(ts)[:10]
            if len(day) < 10:
                continue
            bucket = by_day.setdefault(day, {})
            # history is DESC by timestamp; first seen wins as latest
            if str(src) not in bucket:
                bucket[str(src)] = row

        # Newest return dates first
        for day in sorted(daily_returns.keys(), reverse=True):
            ret_entry = daily_returns.get(day) or {}
            try:
                ret = float(ret_entry.get("daily_return"))
            except (TypeError, ValueError):
                continue
            day_sources = by_day.get(day)
            if not day_sources or len(day_sources) < 2:
                continue
            signals = list(day_sources.values())
            rewards = EnsembleVoter.compute_daily_contribution_rewards(
                signals, daily_return=ret
            )
            if rewards is None:
                continue
            meta: Dict[str, Any] = {
                "reward_mode": "daily_contribution_source_rewards",
                "as_of_date": day,
                "arms": len(rewards),
                "reward_spread": max(rewards.values()) - min(rewards.values()),
                "live_authoritative": False,
                "portfolio_daily_return": ret,
            }
            logger.debug(
                "Loaded daily contribution rewards for %s (%d arms, spread=%.6f)",
                day,
                len(rewards),
                meta["reward_spread"],
            )
            return rewards, meta
        return None

    @staticmethod
    def load_preferred_source_rewards(
        data_dir: Optional[Path] = None,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        """Prefer daily contribution rewards; fall back to windowed attribution.

        Batch BR: ``(rewards, reward_mode)``. Mode is one of
        ``daily_contribution_source_rewards``, ``attribution_source_rewards``,
        or ``none``.
        """
        daily = EnsembleVoter.load_daily_contribution_source_rewards(data_dir)
        if daily is not None:
            rewards, meta = daily
            return rewards, str(meta.get("reward_mode") or "daily_contribution_source_rewards")
        windowed = EnsembleVoter.load_attribution_source_rewards(data_dir)
        if windowed is not None:
            return windowed, "attribution_source_rewards"
        return None, "none"

    @staticmethod
    def load_attribution_source_rewards(
        data_dir: Optional[Path] = None,
        *,
        max_age_days: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Load per-source pseudo-rewards from performance attribution.

        Batch BQ: maps ``avg_return_bps / 1e4`` into decimal return units so
        multi-arm bandit updates can differentiate signals. Windowed attribution
        is a *proxy* for true daily credit assignment (linear/contextual bandit
        ideal); still identifying vs identical portfolio PnL broadcast.

        Batch BR prefers :meth:`load_daily_contribution_source_rewards` via
        :meth:`load_preferred_source_rewards` when a single-day join is available.

        Preference order (when ``data_dir`` is None → default DATA_DIR):
          1. ``{data_dir}/attribution/latest.json``
          2. Newest ``{data_dir}/attribution/attribution_*.json``
          3. Global ``ATTRIBUTION_DIR`` only when ``data_dir`` is default DATA_DIR
             (never leaks live attribution into hermetic tests that pass tmp paths)

        Returns None when missing/empty/unparseable. Never invents zeros for
        unknown sources — callers apply only keys present.
        """
        explicit_dir = data_dir is not None
        root = Path(data_dir) if explicit_dir else Path(DATA_DIR)
        attr_dir = root / "attribution"
        candidates: List[Path] = []
        latest = attr_dir / "latest.json"
        if latest.exists():
            candidates.append(latest)
        if attr_dir.exists():
            dated = sorted(attr_dir.glob("attribution_*.json"), reverse=True)
            for p in dated:
                if p not in candidates:
                    candidates.append(p)
        # Live default only: also search ATTRIBUTION_DIR when distinct
        if not explicit_dir:
            try:
                global_attr = Path(ATTRIBUTION_DIR)
                if global_attr.exists() and global_attr.resolve() != attr_dir.resolve():
                    g_latest = global_attr / "latest.json"
                    if g_latest.exists() and g_latest not in candidates:
                        candidates.insert(0, g_latest)
                    for p in sorted(global_attr.glob("attribution_*.json"), reverse=True)[:3]:
                        if p not in candidates:
                            candidates.append(p)
            except (OSError, TypeError, ValueError):
                pass

        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            sources_block = payload.get("sources")
            if not isinstance(sources_block, dict) or not sources_block:
                continue
            rewards: Dict[str, float] = {}
            for name, meta in sources_block.items():
                if not isinstance(meta, dict):
                    continue
                bps = meta.get("avg_return_bps")
                if bps is None:
                    continue
                try:
                    rewards[str(name)] = float(bps) / 10000.0
                except (TypeError, ValueError):
                    continue
            if len(rewards) >= 2:
                # Distinct values required for identification
                vals = list(rewards.values())
                spread = max(vals) - min(vals)
                if spread < 1e-12:
                    logger.info(
                        "Attribution rewards non-identifying (zero spread) from %s",
                        path,
                    )
                    continue
                logger.debug(
                    "Loaded %d attribution source rewards from %s (spread=%.6f)",
                    len(rewards),
                    path,
                    spread,
                )
                return rewards
        return None

    @staticmethod
    def _soft_delete_source_names(regime_name: str) -> set:
        """String names of REGIME_WEIGHTS soft-delete arms for a regime."""
        zeros = EnsembleVoter._static_zero_baseline_sources(regime_name)
        names: set = set()
        for src in zeros:
            names.add(src.value if hasattr(src, "value") else str(src))
        return names

    def apply_daily_bandit_rewards(
        self,
        daily_return: float,
        regime_name: Optional[str] = None,
        sources: Optional[List[str]] = None,
        *,
        persist: bool = True,
        noise_floor: Optional[float] = None,
        source_rewards: Optional[Dict[str, float]] = None,
        reward_mode: Optional[str] = None,
        include_soft_delete_arms: bool = False,
    ) -> Dict[str, Any]:
        """Apply one day of portfolio return as reward to ensemble bandit sources.

        Production training step: maps paper/portfolio daily return into
        ``update_bandit`` for each active signal source so observations leave
        cold_start. Bandit remains advisory (not live target_allocations).

        Near-zero rewards (|r| < noise_floor, default
        ``ENSEMBLE_BANDIT_REWARD_NOISE_FLOOR`` / 1e-6) are skipped entirely —
        no arm history append, no observation increment, no reward_days step —
        so flat paper NAV / floating-point dust cannot ramp blend.

        Batch BO: multi-arm *identical* portfolio reward broadcast is skipped
        (non-identification). Batch BQ: when ``source_rewards`` maps arms to
        *differentiated* per-source returns (e.g. attribution avg_return_bps),
        multi-arm updates proceed with per-arm credit assignment.
        Batch BR: prefer daily contribution ``source_rewards`` and pass
        ``reward_mode='daily_contribution_source_rewards'`` for honesty tags.

        Batch DL: static soft-delete arms (REGIME_WEIGHTS weight 0) are excluded
        from reward updates by default — sleeping/non-voting experts must not
        train the posterior (Thompson sampling / sleeping-experts hygiene).
        Pass ``include_soft_delete_arms=True`` for explicit shadow learning.

        Returns summary with updates count and observation total.
        """
        try:
            reward = float(daily_return)
        except (TypeError, ValueError):
            return {
                "updates": 0,
                "observations": int(self.bandit_observations),
                "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                "skipped": True,
                "reason": "invalid_daily_return",
            }

        floor = (
            float(noise_floor)
            if noise_floor is not None
            else float(BANDIT_REWARD_NOISE_FLOOR)
        )
        if floor < 0:
            floor = 0.0

        if regime_name is None:
            current = getattr(self, "current_regime", Regime.NORMAL)
            regime_name = current.name if hasattr(current, "name") else str(current)
        regime_name = str(regime_name).upper()

        # Normalize optional per-arm rewards (Batch BQ)
        per_arm: Optional[Dict[str, float]] = None
        if source_rewards:
            cleaned: Dict[str, float] = {}
            for k, v in source_rewards.items():
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                per_arm = cleaned

        if sources is None:
            if per_arm is not None:
                sources = list(per_arm.keys())
            else:
                sources = [s.value for s in SignalSource]
        sources = [str(s) for s in sources]

        # Batch DL: drop soft-delete / non-voting arms from reward training
        soft_delete_excluded: List[str] = []
        if not include_soft_delete_arms:
            soft_names = self._soft_delete_source_names(regime_name)
            if soft_names:
                kept: List[str] = []
                for s in sources:
                    if s in soft_names:
                        soft_delete_excluded.append(s)
                    else:
                        kept.append(s)
                sources = kept
                if per_arm is not None:
                    per_arm = {
                        k: v for k, v in per_arm.items() if k not in soft_names
                    }
                    if not per_arm:
                        per_arm = None
                if not sources:
                    return {
                        "updates": 0,
                        "observations": int(self.bandit_observations),
                        "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                        "days": int(getattr(self, "bandit_days", 0) or 0),
                        "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                        "regime": regime_name,
                        "daily_return": reward,
                        "noise_floor": floor,
                        "skipped": True,
                        "reason": "all_arms_soft_delete_or_empty",
                        "soft_delete_excluded": soft_delete_excluded,
                        "live_authoritative": False,
                    }

        # Multi-arm identical portfolio broadcast guard (Batch BO)
        if len(sources) > 1 and per_arm is None:
            # Still respect noise floor for the scalar portfolio return path
            if abs(reward) < floor:
                logger.info(
                    "Skipping bandit reward update: |daily_return|=%.3e < noise_floor=%.3e",
                    abs(reward),
                    floor,
                )
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "reward_below_noise_floor",
                }
            logger.info(
                "Skipping bandit multi-arm identical reward broadcast: "
                "daily_return=%.6f across %d arms (non-identification guard; "
                "use per-source attribution rewards when available)",
                reward,
                len(sources),
            )
            return {
                "updates": 0,
                "observations": int(self.bandit_observations),
                "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                "days": int(getattr(self, "bandit_days", 0) or 0),
                "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                "regime": regime_name,
                "daily_return": reward,
                "noise_floor": floor,
                "skipped": True,
                "reason": "identical_portfolio_reward_all_arms",
                "arms_considered": len(sources),
            }

        # Build (source, reward) pairs
        pairs: List[Tuple[str, float]] = []
        if per_arm is not None:
            for src in sources:
                if src not in per_arm:
                    continue
                r = float(per_arm[src])
                if abs(r) < floor:
                    continue
                pairs.append((src, r))
            if not pairs:
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "attribution_rewards_below_noise_floor",
                    "arms_considered": len(sources),
                }
            # Non-identification if multi-arm but all remaining rewards equal
            if len(pairs) > 1:
                rs = [r for _, r in pairs]
                if max(rs) - min(rs) < 1e-12:
                    return {
                        "updates": 0,
                        "observations": int(self.bandit_observations),
                        "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                        "days": int(getattr(self, "bandit_days", 0) or 0),
                        "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                        "regime": regime_name,
                        "daily_return": reward,
                        "noise_floor": floor,
                        "skipped": True,
                        "reason": "identical_attribution_rewards_all_arms",
                        "arms_considered": len(pairs),
                    }
        else:
            # Single-arm explicit path (or sources already len==1)
            if abs(reward) < floor:
                logger.info(
                    "Skipping bandit reward update: |daily_return|=%.3e < noise_floor=%.3e",
                    abs(reward),
                    floor,
                )
                return {
                    "updates": 0,
                    "observations": int(self.bandit_observations),
                    "reward_days": int(getattr(self, "bandit_days", 0) or 0),
                    "days": int(getattr(self, "bandit_days", 0) or 0),
                    "bandit_days": int(getattr(self, "bandit_days", 0) or 0),
                    "daily_return": reward,
                    "noise_floor": floor,
                    "skipped": True,
                    "reason": "reward_below_noise_floor",
                }
            pairs = [(src, reward) for src in sources]

        updates = 0
        applied: Dict[str, float] = {}
        for src, r in pairs:
            self.update_bandit(str(src), regime_name, r)
            updates += 1
            applied[src] = r

        # One calendar reward day per apply, independent of arm count
        self.bandit_days = int(getattr(self, "bandit_days", 0) or 0) + 1

        if persist:
            self.save_bandit_state()

        summary: Dict[str, Any] = {
            "updates": updates,
            "observations": int(self.bandit_observations),
            "days": int(self.bandit_days),
            "bandit_days": int(self.bandit_days),
            "regime": regime_name,
            "daily_return": reward,
            "noise_floor": floor,
            "skipped": False,
            "live_authoritative": False,
        }
        if soft_delete_excluded:
            summary["soft_delete_excluded"] = soft_delete_excluded
        if per_arm is not None:
            mode = (
                str(reward_mode)
                if reward_mode
                else "attribution_source_rewards"
            )
            summary["reward_mode"] = mode
            summary["arms_updated"] = list(applied.keys())
            summary["reward_spread"] = (
                max(applied.values()) - min(applied.values()) if applied else 0.0
            )
        else:
            summary["reward_mode"] = (
                str(reward_mode) if reward_mode else "single_arm_or_scalar"
            )
            if applied:
                summary["arms_updated"] = list(applied.keys())
        return summary

    @staticmethod
    def load_latest_daily_return_from_performance(
        performance_path: Optional[Path] = None,
        *,
        max_lines: int = 200,
        prefer_daily_pnl: bool = True,
        data_dir: Optional[Path] = None,
    ) -> Optional[float]:
        """Read newest non-null daily_return for bandit training.

        Prefer ``daily_pnl_latest.json`` (capture_daily_pnl SSOT) when present
        and |return| is finite — avoids replaying flat-NAV micro-noise rows
        that historically polluted performance.jsonl. Falls back to
        performance.jsonl tail.
        """
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        if prefer_daily_pnl:
            pnl_path = root / "daily_pnl_latest.json"
            if pnl_path.exists():
                try:
                    payload = json.loads(pnl_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and "daily_return" in payload:
                        return float(payload["daily_return"])
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.debug("daily_pnl_latest read failed: %s", exc)

        path = Path(performance_path) if performance_path is not None else root / "performance.jsonl"
        if not path.exists():
            return None
        try:
            # Efficient-ish tail for moderate files
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = min(size, 64 * 1024)
                f.seek(max(0, size - block))
                chunk = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in chunk.splitlines() if ln.strip()][-max_lines:]
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "daily_return" not in row:
                    continue
                try:
                    return float(row["daily_return"])
                except (TypeError, ValueError):
                    continue
            return None
        except OSError as exc:
            logger.debug("performance.jsonl read failed: %s", exc)
            return None

    def apply_goal_risk_budget(self, base_allocation: dict) -> dict:
        """Scale allocation weights based on investment goals from goals.json.

        Reads goals.json via src.config.goals, computes risk budget multiplier,
        and shifts allocation toward safer assets proportionally.
        """
        try:
            from src.config.goals import load_goals, get_risk_budget_multiplier
            goals = load_goals()
            risk_mult = get_risk_budget_multiplier(goals)
        except (ImportError, OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("Failed to load goals for risk budget, using risk_mult=1.0: %s", e)
            risk_mult = 1.0

        if risk_mult >= 1.0:
            return base_allocation

        safe_assets = {"SHY", "IEF", "BIL", "TLT"}
        total = sum(base_allocation.values()) if base_allocation else 1.0
        if total == 0:
            return base_allocation

        shifted = {}
        risky_reduction = 0.0
        for asset, weight in base_allocation.items():
            if asset in safe_assets:
                shifted[asset] = weight
            else:
                reduced = weight * risk_mult
                shifted[asset] = reduced
                risky_reduction += weight - reduced

        # Redistribute reduced risk to safe assets proportionally
        safe_total = sum(shifted.get(a, 0) for a in safe_assets if a in shifted)
        if safe_total > 0 and risky_reduction > 0:
            for asset in safe_assets:
                if asset in shifted:
                    shifted[asset] += risky_reduction * (shifted[asset] / safe_total)

        # Renormalize
        new_total = sum(shifted.values())
        if new_total == 0:
            return base_allocation
        return {k: v / new_total * total for k, v in shifted.items()}

    def compute_vote(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]] = None,
        regime: Optional[Regime] = None,
        regime_confidence: Optional[float] = None
    ) -> EnsembleVote:
        """Compute ensemble vote with regime-dependent weighting.

        Delegates to sub-methods for each weighting phase:
        1. _resolve_inputs — resolve readings/regime/confidence defaults
        2. _apply_regime_gating — zero out signals net-negative in this regime
        3. _apply_adaptive_weights — attribution-based weight adjustment
        4. _apply_health_weights — reduce weight for poor health scores
        5. _apply_correlation_penalty — reduce weight for redundant signals
        6. _apply_regime_weights — per-regime signal weight multipliers
        7. _apply_utility_reweighting — boost/reduce by profitability (Sharpe contribution + hit rate)
        8. _apply_exploration_noise — epsilon-greedy Dirichlet exploration for weight discovery
        8a. _apply_diversity_floor — minimum weight floor for active signals (N_eff improvement)
        9. _apply_turnover_validation — turnover + basis-pursuit + regret-weighted
       10. _compute_consensus — weighted consensus, agreement, asset biases, action
       11. _persist_vote — save vote and persist regret state
        """
        caller_supplied_readings = readings is not None
        readings, regime, regime_confidence = self._resolve_inputs(
            readings, regime, regime_confidence
        )

        weights = self.get_blended_weights(regime.name)
        weights = self._apply_regime_gating(weights, regime.name, regime_confidence)
        weights = self._apply_adaptive_weights(weights, regime)
        # Batch DK: adaptive may receive non-zero base if blend leaked; re-pin
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_ic_weights(weights, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_health_weights(weights)
        weights = self._apply_correlation_penalty(weights)
        if os.environ.get("ENSEMBLE_DISABLE_REGIME_WEIGHTS", "").lower() not in ("1", "true"):
            weights = self._apply_regime_weights(weights, regime)
        if os.environ.get("ENSEMBLE_USE_MDP_CONSTRAINT", "").lower() in ("1", "true"):
            weights = self._apply_mdp_constraint(weights)
        weights = self._apply_utility_reweighting(weights, regime)
        weights = self._apply_exploration_noise(weights, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_diversity_floor(weights)
        weights = self._pin_zero_baseline_weights(weights, regime.name)
        weights = self._apply_turnover_validation(weights, readings, regime)
        weights = self._pin_zero_baseline_weights(weights, regime.name)

        # Soft-delete static zeros — exclude from analysis equal-weight floors
        soft_delete = self._static_zero_baseline_sources(regime.name)

        # Safety fallback: when explicit readings are provided for analysis/tests,
        # avoid degenerate outcomes where static regime weights entirely mute
        # one or more provided sources. Soft-delete arms stay excluded.
        active_weights = {
            source: max(0.0, weights.get(source, 0.0))
            for source in readings
            if source not in soft_delete
        }
        active_weight_sum = sum(active_weights.values())
        nonzero_sources = [source for source, weight in active_weights.items() if weight > 0]
        floor_eligible = [s for s in readings if s not in soft_delete]

        if readings and active_weight_sum <= 0:
            if caller_supplied_readings and regime == Regime.NORMAL and floor_eligible:
                fallback_weight = 1.0 / len(floor_eligible)
                weights = {
                    source: (fallback_weight if source in floor_eligible else 0.0)
                    for source in readings
                }
                # Preserve any non-reading keys at 0
                for src in soft_delete:
                    weights[src] = 0.0
                logger.info(
                    "All active ensemble weights were zero after adjustments; "
                    "falling back to equal-weight over %d readings (soft-delete pinned)",
                    len(floor_eligible),
                )
            else:
                logger.info(
                    "All active ensemble weights were zero after adjustments; "
                    "preserving zero-weight gating for regime=%s",
                    regime.value if hasattr(regime, "value") else regime,
                )
        elif (
            caller_supplied_readings
            and floor_eligible
            and regime == Regime.NORMAL
            and 0 < len(nonzero_sources) < len(floor_eligible)
        ):
            floor_weight = 0.05 / len(floor_eligible)  # 5% total floor; soft-delete excluded
            blended = {}
            for source in readings:
                if source in soft_delete:
                    blended[source] = 0.0
                else:
                    blended[source] = max(active_weights.get(source, 0.0), floor_weight)
            total = sum(blended.values())
            if total > 0:
                weights = {source: weight / total for source, weight in blended.items()}
                logger.info(
                    "Applied small analysis floor to %d/%d zero-weight provided signals "
                    "(soft-delete arms pinned at 0)",
                    len(floor_eligible) - len(nonzero_sources),
                    len(floor_eligible),
                )
            weights = self._pin_zero_baseline_weights(weights, regime.name)

        # Apply weights to readings
        weighted_signals = self._apply_weights_to_readings(readings, weights)

        if not weighted_signals:
            return EnsembleVote(
                timestamp=str(datetime.now()),
                regime=regime,
                regime_confidence=regime_confidence,
                num_sources=0,
                weighted_consensus=0.0,
                agreement_ratio=0.0,
                equity_bias=0.0,
                duration_bias=0.0,
                gold_bias=0.0,
                action="neutral",
                confidence=0.0,
                reasoning="No signals available",
                source_votes=[],
                adaptive_learning=self.get_adaptive_learning_status(regime.name),
            )

        consensus_result = self._compute_consensus(weighted_signals, regime, regime_confidence)
        vote = self._build_vote(weighted_signals, consensus_result, regime, regime_confidence)
        self._persist_vote(vote, consensus_result.weighted_consensus)

        return vote

    def _resolve_inputs(
        self,
        readings: Optional[Dict[SignalSource, SignalReading]],
        regime: Optional[Regime],
        regime_confidence: Optional[float],
    ) -> Tuple[Dict[SignalSource, SignalReading], Regime, float]:
        """Resolve default readings, regime, and confidence."""
        if readings is None:
            if regime is None:
                regime, regime_confidence = self.detect_regime()
            readings = self.current_readings or self.collect_signals(regime=regime)

        if regime is None:
            regime, regime_confidence = self.detect_regime()

        if regime_confidence is None:
            regime_confidence = 0.5

        self.current_regime = regime
        self.current_regime_confidence = regime_confidence
        return readings, regime, regime_confidence

    def _apply_regime_gating(
        self, weights: Dict, regime_name: str, regime_confidence: float = 0.5
    ) -> Dict:
        """Apply regime gating — zero out signals that are net-negative in this regime.
        
        Uses confidence-weighted gating (v3.26) to defer gating when regime confidence is low,
        preventing premature switching on uncertain regime classification.
        """
        # Batch CX: reset disclosure map each vote
        self._regime_gated: Dict[str, str] = {}
        if hasattr(self, 'regime_gate') and self.regime_gate is not None:
            # Get active signals based on confidence and hysteresis
            active_signal_names = self.regime_gate.gate_with_confidence(
                regime_name, 
                regime_confidence
            )
            active_signal_set = set(active_signal_names)
            gate_rules = getattr(self.regime_gate, "gate_rules", None)
            explicit_gate_rules = gate_rules if isinstance(gate_rules, dict) else None
            
            # Zero out signals not in the active list
            gated_weights = {}
            for source, weight in weights.items():
                source_name = source.value if hasattr(source, 'value') else str(source)
                has_explicit_gate = (
                    explicit_gate_rules is None or source_name in explicit_gate_rules
                )
                is_active = source_name in active_signal_set or not has_explicit_gate
                if is_active:
                    gated_weights[source] = weight
                else:
                    gated_weights[source] = 0.0
                    if float(weight or 0.0) > 0.0 or has_explicit_gate:
                        off_regimes = set()
                        if isinstance(explicit_gate_rules, dict):
                            off_regimes = set(explicit_gate_rules.get(source_name) or [])
                        self._regime_gated[source_name] = (
                            f"regime_gate_off({regime_name}"
                            f"{'; off=' + ','.join(sorted(off_regimes)) if off_regimes else ''})"
                        )
            
            total = sum(gated_weights.values())
            if total > 0:
                gated_weights = {k: v / total for k, v in gated_weights.items()}
            
            return gated_weights
        return weights

    def _apply_adaptive_weights(
        self, weights: Dict, regime: Regime
    ) -> Dict:
        """Apply adaptive ensemble weighting (v6.09) if attribution data is fresh enough."""
        try:
            from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights

            attribution_dir = ATTRIBUTION_DIR
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)

            if not attribution_files:
                return weights

            with open(attribution_files[0]) as f:
                attribution_data = json.load(f)

            # Check if attribution is stale (>7 days old)
            attr_timestamp = attribution_data.get("timestamp", "")
            if attr_timestamp:
                attr_date = attr_timestamp[:10]
                days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
            else:
                days_stale = 999

            if days_stale > 7:
                return weights

            # Check if we have enough data points
            sources = attribution_data.get("sources", {})
            total_readings = sum(s.get("total_readings", 0) for s in sources.values())
            num_sources = len(sources)
            avg_readings = total_readings / max(num_sources, 1)

            if avg_readings < 5:
                return weights

            # Build base weights in string-keyed format
            base_str = {k.value: v for k, v in weights.items()}

            adaptive = AdaptiveEnsembleWeights(base_weights=base_str)
            adapted = adaptive.update_weights(attribution_data, regime.value)

            # Convert back to enum-keyed dict
            adaptive_weights_enum = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in adapted:
                    adaptive_weights_enum[source_enum] = adapted[source_str]

            if adaptive_weights_enum:
                logger.info("Using adaptive ensemble weights for regime=%s", regime.value)
                return adaptive_weights_enum
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply adaptive ensemble weights: %s", e)
        return weights

    def _apply_ic_weights(self, weights: Dict, regime: Regime) -> Dict:
        """Apply IC-based ensemble weight learning (online IC weighter).

        Uses OnlineICWeighter to compute IC-based weights from the ICMonitor
        persisted state, then blends with the current weights. This is gated
        by ENSEMBLE_USE_IC_WEIGHTS env var (default: off).

        The IC weighter:
        1. Loads IC data from ICMonitor persisted state
        2. Computes rolling IC for each signal
        3. Uses EMA with exponential decay to track IC trends
        4. Converts IC values to weights via temperature-scaled softmax
        5. Blends online weights with current static weights

        Expected impact: +0.005-0.01 Sharpe by dynamically reweighting
        signals based on their recent predictive power.
        """
        if not getattr(self, '_use_ic_weights', False):
            self._last_online_ic_learning_status = {
                "status": "disabled",
                "enabled": False,
                "state_available": False,
                "reason": "env_disabled",
            }
            return weights

        if getattr(self, '_ic_weighter', None) is None:
            self._last_online_ic_learning_status = {
                "status": "unavailable",
                "enabled": True,
                "state_available": False,
                "reason": "initialization_failed_or_unavailable",
            }
            return weights

        try:
            from src.monitor.ic_decay_monitor import ICMonitor

            # Load IC monitor state
            monitor = ICMonitor()
            monitor.load_state()

            # Get IC values and trends for each signal
            ic_values: Dict[str, float] = {}
            ic_trends: Dict[str, str] = {}

            for source_enum in weights:
                source_str = source_enum.value
                ic = monitor.compute_ic(source_str)
                if ic is not None and np.isfinite(ic):
                    ic_values[source_str] = ic
                    trend = monitor.compute_ic_trend(source_str)
                    ic_trends[source_str] = trend

            if not ic_values:
                logger.debug("No IC data available for online weight learning")
                self._last_online_ic_learning_status = {
                    "status": "non_effective",
                    "enabled": True,
                    "state_available": True,
                    "reason": "no_ic_data_available",
                }
                return weights

            # Update the OnlineICWeighter with current IC values and trends
            self._ic_weighter.update(ic_values)
            self._ic_weighter.update_trends(ic_trends)

            # Get IC-based weights (raw)
            ic_weights = self._ic_weighter.get_weights()

            if not ic_weights:
                self._last_online_ic_learning_status = {
                    "status": "non_effective",
                    "enabled": True,
                    "state_available": True,
                    "reason": "no_ic_weights_available",
                }
                return weights

            # Convert weights to string format for blending
            current_weights_str = {k.value: v for k, v in weights.items()}

            # Blend IC-based weights with current weights
            # blend_alpha controls how much we trust IC-based weights (0=static, 1=online)
            # Start conservative: 30% IC-based, 70% current
            blend_alpha = float(os.environ.get("ENSEMBLE_IC_WEIGHT_BLEND_ALPHA", "0.3"))
            blended = {}

            for sig_name in current_weights_str:
                ic_w = ic_weights.get(sig_name, 0.0)
                current_w = current_weights_str[sig_name]
                blended[sig_name] = (1.0 - blend_alpha) * current_w + blend_alpha * ic_w

            # Renormalize
            total = sum(blended.values())
            if total > 0:
                blended = {k: v / total for k, v in blended.items()}

            # Convert back to enum-keyed dict
            ic_adjusted = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in blended:
                    ic_adjusted[source_enum] = blended[source_str]

            if ic_adjusted:
                logger.info(
                    "Online IC weights applied (blend_alpha=%.2f): %s",
                    blend_alpha,
                    ', '.join(
                        f'{k.value}={v:.3f}'
                        for k, v in ic_adjusted.items() if v > 0.01
                    )
                )
                self._last_online_ic_learning_status = {
                    "status": "active",
                    "enabled": True,
                    "state_available": True,
                    "reason": "blending_with_static_weights",
                }
                return ic_adjusted

        except (ImportError, KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply IC-based weights: %s", e)
            self._last_online_ic_learning_status = {
                "status": "unavailable",
                "enabled": True,
                "state_available": getattr(self, "_ic_weighter", None) is not None,
                "reason": f"ic_weight_application_failed:{type(e).__name__}",
            }

        return weights

    def _apply_health_weights(self, weights: Dict) -> Dict:
        """Apply health-adjusted weighting (v3.12) — soft floor + hard-zero gates.

        Batch BH residual honesty:
        - status == unhealthy → multiplier 0 (quality sleep / hard exclude)
        - Batch CN: status == degraded **and** IC < 0 → multiplier 0
          (negative-IC degraded arms are fail-closed; soft floor only for
          degraded/healthy with non-negative IC — hybrid 2025–2026 SRE policy)
        - otherwise soft floor max(0.2, health_score) for graceful degrade
        - if all arms hard-gated: freeze adaptive blend (all-zero mass, do not
          reinflate toxic arms via renorm)
        """
        self._health_gate_freeze = False
        self._health_gate_slept: list[str] = []
        # Batch CW: source → reason (and optional diagnostics) for disclosure
        self._health_gate_sleep_reasons: Dict[str, str] = {}
        try:
            from src.signals.health_tracker import SignalHealthTracker, SignalHealthStatus
            health_tracker = SignalHealthTracker()
            health_scores = health_tracker.calculate_all_health_scores()

            if not health_scores:
                return weights

            adjusted_weights = {}
            slept: list[str] = []
            sleep_reasons: Dict[str, str] = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                if source_str in health_scores:
                    health = health_scores[source_str]
                    status = str(getattr(health, "status", "") or "").lower()
                    hs = float(getattr(health, "health_score", 0.0) or 0.0)
                    ic_raw = getattr(health, "ic", None)
                    try:
                        ic_val = float(ic_raw) if ic_raw is not None else None
                    except (TypeError, ValueError):
                        ic_val = None

                    # Batch CY hybrid (evolves BH/CN):
                    # - hard sleep toxic arms: IC < 0 (any non-healthy status), or
                    #   unhealthy with unknown IC (fail-closed without IC evidence)
                    # - soft floor max(0.2, hs) when quality is poor but IC ≥ 0
                    #   (borderline "unhealthy" with non-neg IC is not toxic drag)
                    hard_zero = False
                    sleep_reason = None
                    if ic_val is not None and ic_val < 0.0:
                        hard_zero = True
                        sleep_reason = (
                            f"negative_ic({ic_val:.3f})"
                            if status != SignalHealthStatus.DEGRADED.value
                            else f"degraded_negative_ic({ic_val:.3f})"
                        )
                    elif status == SignalHealthStatus.UNHEALTHY.value:
                        if ic_val is None:
                            hard_zero = True
                            sleep_reason = "unhealthy_ic_unknown"
                        # else: unhealthy + IC>=0 → soft floor below
                    elif (
                        status == SignalHealthStatus.DEGRADED.value
                        and ic_val is not None
                        and ic_val < 0.0
                    ):
                        # unreachable (neg IC handled above) — keep for clarity
                        hard_zero = True
                        sleep_reason = f"degraded_negative_ic({ic_val:.3f})"

                    if hard_zero:
                        multiplier = 0.0
                        slept.append(source_str)
                        reason = sleep_reason or "hard_zero"
                        sleep_reasons[source_str] = reason
                        logger.info(
                            "Health-gated %s: weight %.2f%% → 0%% (%s, score=%.2f, ic=%s)",
                            source_str,
                            base_weight * 100,
                            reason,
                            hs,
                            ic_val,
                        )
                    else:
                        multiplier = max(0.2, min(1.0, hs))
                        if hs < 0.5 or status == SignalHealthStatus.UNHEALTHY.value:
                            logger.info(
                                "Health-adjusted %s: weight %.2f%% → %.2f%% "
                                "(status=%s, health=%.2f, ic=%s)",
                                source_str,
                                base_weight * 100,
                                base_weight * multiplier * 100,
                                status,
                                hs,
                                ic_val,
                            )
                    adjusted_weights[source_enum] = base_weight * multiplier
                else:
                    adjusted_weights[source_enum] = base_weight

            self._health_gate_slept = slept
            self._health_gate_sleep_reasons = sleep_reasons
            total = sum(adjusted_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in adjusted_weights.items()}
            else:
                # All arms unhealthy / zero — freeze; do not reinflate via renorm
                self._health_gate_freeze = True
                weights = {k: 0.0 for k in weights}
                logger.warning(
                    "Health gate freeze: all ensemble arms hard-zeroed (%s); "
                    "adaptive blend contributes zero mass",
                    ", ".join(slept) if slept else "no sources",
                )
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply health-adjusted weights: %s", e)
        return weights

    def _apply_correlation_penalty(self, weights: Dict) -> Dict:
        """Apply correlation penalty to reduce weight of redundant signals.

        Computes pairwise prediction correlations from IC decay data.
        Signals highly correlated with peers have their weights reduced
        to improve ensemble diversification and prevent double-counting.

        The penalty is conservative: max 30% reduction for perfectly
        correlated signals. The penalty factor is 1/(1+mean_abs_corr),
        so a signal correlated at 0.7 with peers gets ~0.59 penalty.
        """
        try:
            corr_data = compute_signal_correlation_matrix()
            penalties = corr_data.get("correlation_penalties", {})
            if not penalties:
                return weights

            redundant = corr_data.get("redundant_pairs", [])
            if redundant:
                logger.info(
                    "Redundant signal pairs detected: %s",
                    ', '.join(f'{s1}/{s2}(r={c:.2f})' for s1, s2, c in redundant)
                )

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                penalty = penalties.get(source_str, 1.0)
                # Clip to prevent excessive reduction: min penalty = 0.5
                penalty = max(0.5, penalty)
                adjusted[source_enum] = base_weight * penalty
                if abs(penalty - 1.0) > 0.01:
                    logger.info(
                        "Correlation-penalized %s: %.3f -> %.3f (penalty=%.3f)",
                        source_str, base_weight, adjusted[source_enum], penalty
                    )

            # Re-normalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

            return adjusted
        except (KeyError, ValueError, TypeError, OSError, ImportError) as e:
            logger.warning("Could not apply correlation penalty: %s", e)
        return weights

    def _apply_regime_weights(self, weights: Dict, regime: Regime) -> Dict:
        """Apply per-regime signal weight multipliers.

        Varies ensemble signal weights by macro regime using the
        REGIME_CONDITIONAL_WEIGHTS map. Each regime has multipliers
        reflecting which signals perform well in that environment:

        - CRISIS: boost alternative_data, reduce unified_overlay
        - HIGH_VOL: boost unified_overlay (defensive), reduce intl_momentum
        - NORMAL: baseline (no adjustment)
        - LOW_VOL: boost international_momentum, reduce regime_arb (mean-reversion)
        - RECOVERY: boost international_momentum (post-crisis momentum)

        Multipliers are capped at [0.3, 1.5] per signal and weights are
        renormalized to sum=1.0 after adjustment.
        """
        try:
            regime_name = regime.name if hasattr(regime, 'name') else str(regime)
            regime_multipliers = REGIME_CONDITIONAL_WEIGHTS.get(regime_name, {})

            # NORMAL is baseline: all signals at 1.0, no adjustment needed
            if regime_name == "NORMAL" and not any(
                v != 1.0 for v in regime_multipliers.values()
            ):
                return weights

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                multiplier = float(regime_multipliers.get(source_str, 1.0))
                # Conservative caps: min 0.3, max 1.5
                multiplier = max(0.3, min(1.5, multiplier))
                adjusted[source_enum] = base_weight * multiplier

            total = sum(adjusted.values())
            if total <= 0:
                return weights

            # Gate signals below 5% of total weight to zero
            min_threshold = 0.05 * total
            gated = {
                k: (v if v >= min_threshold else 0.0)
                for k, v in adjusted.items()
            }
            gated_total = sum(gated.values())
            if gated_total <= 0:
                return weights

            # Renormalize
            result = {k: v / gated_total for k, v in gated.items()}

            if regime_name != "NORMAL":
                logger.info(
                    "Regime-conditional weights (%s): %s",
                    regime_name,
                    ', '.join(
                        f'{k.value}={result[k]:.3f}'
                        for k in result if result[k] > 0
                    )
                )

            return result

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Could not apply regime-conditional weights: %s", e)
        return weights

    def _apply_utility_reweighting(self, weights: Dict, regime: Regime) -> Dict:
        """Apply utility-based reweighting from signal profitability data.

        Boosts weights for signals with positive Sharpe contribution from
        attribution data, reduces weights for negative contributors.
        This is complementary to health-based adjustment: health measures
        signal reliability (IC, consistency), utility measures profitability.

        The adjustment is conservative: max ±30% weight change, and only
        applied when attribution has enough observations (>=20 readings).
        """
        try:
            attribution_dir = ATTRIBUTION_DIR
            attribution_files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)

            if not attribution_files:
                return weights

            with open(attribution_files[0]) as f:
                attribution_data = json.load(f)

            # Check freshness
            attr_timestamp = attribution_data.get("timestamp", "")
            if attr_timestamp:
                attr_date = attr_timestamp[:10]
                days_stale = (datetime.now() - datetime.strptime(attr_date, "%Y-%m-%d")).days
            else:
                days_stale = 999

            if days_stale > 7:
                return weights

            sources = attribution_data.get("sources", {})
            if not sources:
                return weights

            adjusted = {}
            for source_enum, base_weight in weights.items():
                source_str = source_enum.value
                source_data = sources.get(source_str, {})

                # Need enough observations for meaningful Sharpe
                total_readings = source_data.get("total_readings", 0)
                if total_readings < 20:
                    adjusted[source_enum] = base_weight
                    continue

                sharpe_contrib = source_data.get("sharpe_contribution", 0.0)
                hit_rate = source_data.get("hit_rate", 0.0)

                # Utility score: blend Sharpe contribution (primary) and hit rate (secondary)
                # Sharpe contribution is in annualized units; normalize to [-1, 1] range
                sharpe_signal = np.clip(sharpe_contrib / 2.0, -1.0, 1.0)  # ±2 Sharpe = full signal
                hit_signal = (hit_rate - 0.5) * 2.0 if hit_rate > 0 else 0.0  # 50% hit rate = neutral

                # Weighted blend: 70% Sharpe, 30% hit rate
                utility_score = 0.7 * sharpe_signal + 0.3 * hit_signal

                # Conservative adjustment: max ±30% weight change
                # Positive utility → boost, negative → reduce
                adjustment = 1.0 + np.clip(utility_score * 0.3, -0.3, 0.3)
                adjusted[source_enum] = base_weight * adjustment

                if abs(utility_score) > 0.1:
                    logger.info("Utility-reweighted %s: %.2f%% → %.2f%% (utility=%.3f, sharpe_contrib=%.3f, hit_rate=%.2f)",
                                source_str, base_weight * 100, adjusted[source_enum] * 100,
                                utility_score, sharpe_contrib, hit_rate)

            # Renormalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

            return adjusted

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Could not apply utility-based reweighting: %s", e)
        return weights

    def _apply_exploration_noise(self, weights: Dict, regime: Regime) -> Dict:
        """Apply epsilon-greedy exploration noise to weight allocation.

        With probability exploration_epsilon, draws weights from a Dirichlet
        distribution centered on current weights. This allows the system to
        discover better weight configurations that might otherwise go untested,
        without risking large deviations from the baseline allocation.

        Dirichlet concentration alpha controls how close the samples stay
        to the mean — higher alpha = closer to current weights.
        """
        epsilon = float(os.environ.get("ENSEMBLE_EXPLORATION_EPSILON", "0.05"))
        if random.random() >= epsilon:
            return weights

        # Dirichlet concentration parameter — higher = closer to current weights
        # alpha=10 means samples typically stay within ±10% of current weights
        alpha_base = float(os.environ.get("ENSEMBLE_EXPLORATION_ALPHA", "10.0"))

        weight_values = [weights[k] for k in weights]
        n = len(weight_values)
        if n < 2:
            return weights

        # Dirichlet alpha: concentration * current weight for each component.
        # Soft-delete arms (static zero) get tiny alpha so samples stay ~0, then
        # Batch DK pin zeros them hard after sampling.
        regime_name = regime.name if hasattr(regime, "name") else str(regime)
        soft_delete = self._static_zero_baseline_sources(regime_name)
        alpha = []
        for k, w in zip(weights.keys(), weight_values):
            if k in soft_delete or float(w or 0.0) <= 0.0:
                alpha.append(1e-9)  # near-zero mass; pin finishes the job
            else:
                alpha.append(max(0.1, alpha_base * w))

        # Sample from Dirichlet
        try:
            sampled = np.random.dirichlet(alpha)
            result = {k: float(sampled[i]) for i, k in enumerate(weights)}
            result = self._pin_zero_baseline_weights(result, regime_name)
            logger.info("Exploration noise applied: epsilon=%.2f, regime=%s", epsilon, regime.value)
            return result
        except (ValueError, FloatingPointError) as e:
            logger.warning("Exploration noise failed: %s", e)
            return weights

    def _apply_diversity_floor(
        self,
        weights: Dict,
        floor: Optional[float] = None,
    ) -> Dict:
        """Apply diversity floor — minimum weight for each active signal.

        Prevents weight concentration by ensuring every signal that was
        originally active (weight > 0) retains at least `floor` fraction
        of the total weight. This raises N_eff (effective signal count)
        without overriding the signal quality assessment.

        The floor is applied as a lower bound, not an equalizer: signals
        with higher quality still get proportionally more weight.

        Batch BM: never raise or reinflate arms hard-zeroed by the health
        gate (``_health_gate_slept``). Soft floors must not undo quality sleep.

        Args:
            weights: Current weight dict {SignalSource: weight}.
            floor: Minimum weight fraction per active signal. If None,
                uses DEFAULT_DIVERSITY_FLOOR.

        Returns:
            Adjusted weights dict summing to 1.0 (or all-zero if freeze).
        """
        if floor is None:
            floor = DEFAULT_DIVERSITY_FLOOR
        if floor <= 0:
            return weights

        slept_names = {
            str(s) for s in (getattr(self, "_health_gate_slept", None) or [])
        }

        def _src_name(source) -> str:
            return source.value if hasattr(source, "value") else str(source)

        # Only apply to signals that were active (weight > 0) and not health-slept
        active = {
            k: v
            for k, v in weights.items()
            if v > 0 and _src_name(k) not in slept_names
        }
        if len(active) <= 1:
            # Still force slept arms to zero if somehow positive
            if slept_names:
                cleaned = dict(weights)
                for k in list(cleaned):
                    if _src_name(k) in slept_names:
                        cleaned[k] = 0.0
                total_c = sum(cleaned.values())
                if total_c > 0:
                    return {k: v / total_c for k, v in cleaned.items()}
                return cleaned
            return weights

        total = sum(weights.values())
        if total <= 0:
            return weights

        # Normalize to get fractional weights; keep slept at 0
        frac = {}
        for k, v in weights.items():
            if _src_name(k) in slept_names:
                frac[k] = 0.0
            else:
                frac[k] = v / total

        # Identify signals below the floor (never raise slept)
        adjusted = dict(frac)
        raised_count = 0
        for source in active:
            if adjusted[source] < floor:
                adjusted[source] = floor
                raised_count += 1

        if raised_count == 0 and not slept_names:
            return weights  # No adjustment needed

        # Re-normalize so weights sum to 1.0 (slept stay 0)
        new_total = sum(adjusted.values())
        if new_total > 0:
            adjusted = {k: v / new_total for k, v in adjusted.items()}
            for k in adjusted:
                if _src_name(k) in slept_names:
                    adjusted[k] = 0.0
            # Renorm once more if slept zeros left a hole (shouldn't)
            nt = sum(adjusted.values())
            if nt > 0 and abs(nt - 1.0) > 1e-9:
                adjusted = {k: v / nt for k, v in adjusted.items()}

        if raised_count:
            logger.info(
                "Diversity floor applied: raised %d/%d active signals (floor=%.1f%%); "
                "health-slept excluded=%d",
                raised_count,
                len(active),
                floor * 100,
                len(slept_names),
            )

        return adjusted

    @staticmethod
    def _extract_signal_values(readings: Dict) -> Dict[str, float]:
        """Build signal_values dict from current readings, skipping NaN."""
        signal_values = {}
        for source_enum in readings:
            source_str = source_enum.value
            reading = readings[source_enum]
            if not np.isnan(reading.value):
                signal_values[source_str] = reading.value
        return signal_values

    def _apply_basis_pursuit(
        self, signal_values: Dict[str, float], base_weights_str: Dict[str, float], regime_value: str
    ) -> Dict[str, float]:
        """Apply basis-pursuit signal selection to prune redundant signals."""
        try:
            from src.strategy.basis_pursuit_selector import BasisPursuitSelector
            bp_selector = BasisPursuitSelector()
            bp_result = bp_selector.select_signals(
                signal_values, base_weights_str, regime=regime_value
            )
            sparsity_msg = (
                f" (sparsity={bp_result.sparsity_ratio:.2f}, "
                f"{bp_result.num_pruned} pruned)"
                if bp_result.num_pruned > 0
                else ""
            )
            logger.debug("Basis-pursuit selection applied%s", sparsity_msg)
            return bp_result.active_signals
        except (ImportError, KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError) as bp_e:
            logger.warning("Could not apply basis-pursuit selection: %s", bp_e)
            return base_weights_str

    def _apply_regret_weighting(
        self, signal_values: Dict[str, float], base_weights_str: Dict[str, float], regime_value: str
    ) -> Dict[str, float]:
        """Apply regret-weighted adjustment to penalize signals with high regret."""
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            prev_decision = getattr(rw_selector.state, 'last_ensemble_decision', 0.0)
            rw_result = rw_selector.adjust_weights(
                signal_values, prev_decision, base_weights_str, regime=regime_value
            )
            if rw_result.signals_with_high_regret:
                logger.info(
                    "Regret-adjusted weights: penalized %s (avg_regret=%.3f)",
                    ', '.join(rw_result.signals_with_high_regret),
                    rw_result.avg_regret
                )
            return rw_result.adjusted_weights
        except (ImportError, KeyError, ValueError, TypeError, AttributeError, OSError) as rw_e:
            logger.warning("Could not apply regret-weighted adjustment: %s", rw_e)
            return base_weights_str

    def _apply_turnover_validation(
        self, weights: Dict, readings: Dict, regime: Regime
    ) -> Dict:
        """Apply turnover-aware weight validation (v8.01) with basis-pursuit and regret-weighted."""
        try:
            from src.strategy.turnover_validator import TurnoverValidator
            turnover_validator = TurnoverValidator()

            signal_values = self._extract_signal_values(readings)
            if not signal_values:
                return weights

            base_weights_str = {source_enum.value: w for source_enum, w in weights.items()}

            base_weights_str = self._apply_basis_pursuit(signal_values, base_weights_str, regime.value)
            base_weights_str = self._apply_regret_weighting(signal_values, base_weights_str, regime.value)

            # Apply turnover adjustment
            adjusted_str = turnover_validator.get_adjusted_weights(
                base_weights_str, signal_values
            )

            # Convert back to enum-keyed dict
            turnover_adjusted = {}
            for source_enum in weights:
                source_str = source_enum.value
                if source_str in adjusted_str:
                    turnover_adjusted[source_enum] = adjusted_str[source_str]
                else:
                    turnover_adjusted[source_enum] = weights[source_enum]

            # Re-normalize to sum to 1.0
            total = sum(turnover_adjusted.values())
            if total > 0:
                weights = {k: v / total for k, v in turnover_adjusted.items()}

            logger.debug(
                "Turnover-adjusted %d signals: %s",
                len(signal_values),
                ', '.join(f'{s}={turnover_adjusted.get(enum, 0):.4f}' for enum, s in [(e, e.value) for e in weights])
            )
        except (KeyError, ValueError, TypeError, AttributeError, ZeroDivisionError, OSError) as e:
            logger.warning("Could not apply turnover-aware weights: %s", e)
        return weights

    def _apply_weights_to_readings(
        self,
        readings: Dict[SignalSource, SignalReading],
        weights: Dict,
    ) -> List[SignalReading]:
        """Assign weights to readings and log predictions for health tracking."""
        weighted_signals = []
        for source, reading in readings.items():
            if source in weights:
                # Batch CV: inactive snapshots stay in the vote trail for
                # disclosure (source_breakdown) but must not move consensus.
                if getattr(reading, "is_active", True):
                    reading.weight = weights[source]
                else:
                    reading.weight = 0.0
                weighted_signals.append(reading)

        # Log signal predictions for health tracking (v3.12 / Batch DF provenance)
        try:
            tracker = _get_health_tracker()
            if tracker is not None:
                for reading in weighted_signals:
                    meta = getattr(reading, "metadata", None)
                    if not isinstance(meta, dict):
                        meta = {}
                    else:
                        meta = dict(meta)
                    # Compact provenance always stamped for post-fix IC cohorts
                    meta.setdefault("provenance_batch", "df")
                    if getattr(reading, "explanation", None):
                        meta.setdefault("explanation", str(reading.explanation)[:200])
                    meta.setdefault("is_active", bool(getattr(reading, "is_active", True)))
                    tracker.log_prediction_simple(
                        source=reading.source.value,
                        signal_value=reading.value,
                        confidence=reading.confidence,
                        metadata=meta,
                    )
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
            logger.warning("Health tracking log failed: %s", e)

        return weighted_signals

    @dataclass
    class _ConsensusResult:
        """Internal intermediate result from consensus computation."""
        weighted_consensus: float
        agreement: float
        equity_bias: float
        duration_bias: float
        gold_bias: float
        action: str
        action_confidence: float

    @staticmethod
    def _compute_asset_biases(
        weighted_signals: List[SignalReading], fallback_consensus: float
    ) -> Dict[str, float]:
        """Compute per-asset weighted bias from signal readings."""
        assets = ['SPY', 'TLT', 'GLD']
        asset_biases = {}
        for asset in assets:
            asset_signals = [
                (r.asset_signals.get(asset, 0), r.weight)
                for r in weighted_signals
                if r.asset_signals and asset in r.asset_signals and not np.isnan(r.asset_signals.get(asset, np.nan))
            ]
            if asset_signals:
                total_w = sum(w for _, w in asset_signals) or 1.0
                asset_biases[asset] = sum(v * w for v, w in asset_signals) / total_w
            else:
                asset_biases[asset] = fallback_consensus
        return asset_biases

    @staticmethod
    def _determine_action(
        regime: Regime, regime_confidence: float, equity_bias: float, agreement: float
    ) -> Tuple[str, float]:
        """Determine portfolio action from regime, equity bias, and agreement.

        Uses regime-conditional consensus thresholds:
        CRISIS 0.50, HIGH_VOL 0.55, RECOVERY 0.60, LOW_VOL 0.67, NORMAL 0.75.
        Falls back to ENSEMBLE_CONSENSUS_THRESHOLD env var for unknown regimes.
        """
        if regime == Regime.CRISIS:
            return "risk_off", regime_confidence

        # Regime-specific threshold (falls back to global constant)
        threshold = REGIME_CONSENSUS_THRESHOLDS.get(
            regime.value.upper() if hasattr(regime.value, 'upper') else str(regime.value).upper(),
            ENSEMBLE_CONSENSUS_THRESHOLD,
        )

        if equity_bias > 0.3 and agreement > threshold:
            return "increase_equity", agreement * abs(equity_bias)
        elif equity_bias < -0.3 and agreement > threshold:
            return "decrease_equity", agreement * abs(equity_bias)
        else:
            # Neutral hold conviction tracks agreement × regime confidence —
            # do not hardcode 0.5 (high-agreement hold looked identical to uncertain).
            conf = float(max(0.0, min(1.0, agreement * regime_confidence)))
            return "neutral", conf

    def _compute_consensus(
        self,
        weighted_signals: List[SignalReading],
        regime: Regime,
        regime_confidence: float,
    ) -> '_ConsensusResult':
        """Compute weighted consensus, agreement ratio, and asset biases."""
        # Weighted consensus — handle NaN values
        valid_signals = [
            (r.value, r.weight)
            for r in weighted_signals
            if not np.isnan(r.value)
        ]

        if valid_signals:
            total_weight = sum(w for _, w in valid_signals)
            if total_weight == 0:
                total_weight = 1.0
            weighted_consensus = sum(v * w for v, w in valid_signals) / total_weight
        else:
            weighted_consensus = 0.0
            total_weight = 1.0

        # Agreement ratio: % of weighted signals agreeing with consensus
        agreement = sum(
            r.weight for r in weighted_signals
            if np.sign(r.value) == np.sign(weighted_consensus) or abs(r.value) < 0.1
        ) / total_weight

        # Asset-specific consensus
        asset_biases = self._compute_asset_biases(weighted_signals, weighted_consensus)

        # Determine action
        equity_bias = asset_biases.get('SPY', weighted_consensus)
        duration_bias = asset_biases.get('TLT', 0)
        gold_bias = asset_biases.get('GLD', 0)

        action, action_confidence = self._determine_action(
            regime, regime_confidence, equity_bias, agreement
        )

        return self._ConsensusResult(
            weighted_consensus=weighted_consensus,
            agreement=agreement,
            equity_bias=equity_bias,
            duration_bias=duration_bias,
            gold_bias=gold_bias,
            action=action,
            action_confidence=action_confidence,
        )

    def _build_vote(
        self,
        weighted_signals: List[SignalReading],
        consensus: '_ConsensusResult',
        regime: Regime,
        regime_confidence: float,
    ) -> EnsembleVote:
        """Build EnsembleVote from weighted signals and consensus result."""
        reasons = [
            f"Regime: {regime.value} (confidence: {regime_confidence:.1%})",
            f"Sources: {len(weighted_signals)}, Consensus: {consensus.weighted_consensus:+.3f}",
            f"Agreement: {consensus.agreement:.1%}",
            f"Equity bias: {consensus.equity_bias:+.3f}, Duration: {consensus.duration_bias:+.3f}, Gold: {consensus.gold_bias:+.3f}"
        ]

        for r in weighted_signals[:3]:
            reasons.append(f"  {r.source.value}: {r.value:+.3f} (w={r.weight:.2f}, conf={r.confidence:.1%})")

        # Compute effective signal count (N_eff) and Shannon entropy on
        # *renormalized* positive weights so incomplete collection does not
        # understate diversification (sleeping-experts: active set sums to 1).
        weights_arr = np.array([r.weight for r in weighted_signals], dtype=float)
        weights_arr = weights_arr[np.isfinite(weights_arr) & (weights_arr > 0)]
        active_weight_mass = float(np.sum(weights_arr)) if len(weights_arr) else 0.0
        if len(weights_arr) > 0 and active_weight_mass > 0:
            w_norm = weights_arr / active_weight_mass
            weight_entropy = float(-np.sum(w_norm * np.log(w_norm)))
            n_eff = float(np.exp(weight_entropy))
        else:
            weight_entropy = 0.0
            n_eff = 0.0
            active_weight_mass = 0.0

        sleep_reasons = dict(getattr(self, "_health_gate_sleep_reasons", None) or {})
        regime_gated = dict(getattr(self, "_regime_gated", None) or {})
        return EnsembleVote(
            timestamp=str(datetime.now()),
            regime=regime,
            regime_confidence=regime_confidence,
            num_sources=len(weighted_signals),
            weighted_consensus=consensus.weighted_consensus,
            agreement_ratio=consensus.agreement,
            equity_bias=consensus.equity_bias,
            duration_bias=consensus.duration_bias,
            gold_bias=consensus.gold_bias,
            action=consensus.action,
            confidence=consensus.action_confidence,
            reasoning="\n".join(reasons),
            source_votes=weighted_signals,
            n_eff=round(n_eff, 2),
            weight_entropy=round(weight_entropy, 4),
            adaptive_learning=self.get_adaptive_learning_status(regime.name),
            health_gate_slept=sleep_reasons or None,
            health_gate_freeze=bool(getattr(self, "_health_gate_freeze", False)),
            regime_gated=regime_gated or None,
        )

    def _persist_vote(self, vote: EnsembleVote, weighted_consensus: float) -> None:
        """Persist ensemble decision for regret-weighted cycle and save vote to DB."""
        # Persist ensemble decision for next regret-weighted cycle (v8.03)
        try:
            from src.strategy.regret_weighted_selector import RegretWeightedSelector
            rw_selector = RegretWeightedSelector()
            rw_selector.state.last_ensemble_decision = weighted_consensus
            rw_selector._save_state()
        except (ImportError, OSError, KeyError, ValueError, TypeError, AttributeError) as rw_e:
            logger.warning("Could not persist ensemble decision to regret-weighted state: %s", rw_e)

        # Persist IC weighter state if enabled
        if getattr(self, '_use_ic_weights', False) and getattr(self, '_ic_weighter', None) is not None:
            try:
                ic_state_path = self.data_path / "ic_weighter_state.json"
                ic_state = self._ic_weighter.get_state()
                with open(ic_state_path, "w") as f:
                    json.dump(ic_state, f)
                logger.debug("OnlineICWeighter state saved to %s", ic_state_path)
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to save OnlineICWeighter state: %s", e)

        # Check for IC-based signal decay alerts
        try:
            _tracker = _get_health_tracker()
            if _tracker is not None:
                alerts = _tracker.detect_ic_alerts()
                if alerts:
                    alert_names = [a.source for a in alerts]
                    logger.warning("IC decay alerts detected: %s", alert_names)
        except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as ic_e:
            logger.warning("IC alert check failed: %s", ic_e)

        # Save to DB
        self._save_vote(vote)
    
    def _save_vote(self, vote: EnsembleVote):
        """Save vote to database, including per-source readings (v5.70)."""
        with sqlite_connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ensemble_votes
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vote.timestamp,
                vote.regime.value,
                vote.regime_confidence,
                vote.num_sources,
                vote.weighted_consensus,
                vote.agreement_ratio,
                vote.equity_bias,
                vote.duration_bias,
                vote.gold_bias,
                vote.action,
                vote.confidence,
                vote.reasoning
            ))

            # v5.70: Save individual source readings for attribution
            for reading in vote.source_votes:
                try:
                    conn.execute("""
                        INSERT INTO source_readings
                        (timestamp, source, value, confidence, weight, regime_fit, explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        vote.timestamp,
                        reading.source.value if hasattr(reading.source, 'value') else str(reading.source),
                        float(reading.value),
                        float(reading.confidence),
                        float(reading.weight),
                        reading.regime_fit or "",
                        (reading.explanation or "")[:500],
                    ))
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to save source reading %s: %s", reading.source, e)
    
    def recommend_allocation(
        self,
        base_allocation: Dict[str, float] = None,
        vote: Optional[EnsembleVote] = None,
        max_shift: float = 0.10
    ) -> Dict[str, Dict]:
        """
        Generate allocation recommendation based on ensemble vote.
        
        Returns shifts from base allocation for each asset.
        """
        if base_allocation is None:
            base_allocation = BASE_ALLOCATION
        
        if vote is None:
            vote = self.compute_vote()
        
        # Apply shifts based on biases
        shifts = {
            'SPY': np.clip(vote.equity_bias * max_shift, -max_shift, max_shift),
            'TLT': np.clip(vote.duration_bias * max_shift, -max_shift, max_shift),
            'GLD': np.clip(vote.gold_bias * max_shift, -max_shift, max_shift),
        }
        
        # Risk-off override
        if vote.regime == Regime.CRISIS:
            shifts['SPY'] = -max_shift * 0.5  # Reduce equity
            shifts['GLD'] = max_shift * 0.3   # Increase gold
            shifts['TLT'] = max_shift * 0.2   # Increase bonds
        
        result = {}
        total_shift = 0
        
        for asset, base in base_allocation.items():
            shift = shifts.get(asset, 0)
            new_alloc = base + shift
            
            result[asset] = {
                'base': base,
                'shift': shift,
                'new': np.clip(new_alloc, 0.05, 0.95),  # Bounds
                'bias': shifts.get(asset, 0),
            }
            total_shift += shift
        
        # Normalize to sum to 1
        total = sum(r['new'] for r in result.values())
        for asset in result:
            result[asset]['new'] /= total
            result[asset]['normalized_shift'] = result[asset]['new'] - result[asset]['base']
        
        return {
            'assets': result,
            'regime': vote.regime.value,
            'confidence': vote.confidence,
            'action': vote.action,
            'consensus': vote.weighted_consensus,
            'timestamp': vote.timestamp
        }

    def get_bl_views(
        self,
        vote: Optional[EnsembleVote] = None,
        tau: float = 0.15,
        prior: str = "equal",
    ) -> Dict[str, Any]:
        """Generate Black-Litterman views from ensemble vote.

        Maps equity_bias, duration_bias, and gold_bias from the
        current ensemble consensus to BL absolute views, with view
        confidence derived from signal health scores.

        Args:
            vote: Pre-computed vote (default: compute fresh).
            tau: BL tau parameter (view weight). Default 0.15.
            prior: Prior type — "equal" or "market".

        Returns:
            Dict with 'views' (BLViews), 'tau', 'prior', and
            'health_scores_used' keys.
        """
        from src.strategy.black_litterman_mapper import map_biases_to_views, BLViews

        if vote is None:
            vote = self.compute_vote()

        # Collect health scores from tracker
        health_scores = {}
        tracker = _get_health_tracker()
        if tracker is not None:
            try:
                report = tracker.get_health_report()
                for source_name, data in report.get('sources', {}).items():
                    score = data.get('health_score', 0.5)
                    health_scores[source_name] = score
            except (KeyError, ValueError, TypeError, AttributeError, OSError, sqlite3.Error) as e:
                logger.warning("Could not get health scores for BL views: %s", e)

        views = map_biases_to_views(
            equity_bias=vote.equity_bias,
            duration_bias=vote.duration_bias,
            gold_bias=vote.gold_bias,
            health_scores=health_scores if health_scores else None,
            tau=tau,
            prior=prior,
        )

        return {
            'views': views,
            'tau': tau,
            'prior': prior,
            'health_scores_used': health_scores,
            'equity_bias': vote.equity_bias,
            'duration_bias': vote.duration_bias,
            'gold_bias': vote.gold_bias,
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Signal Voter')
    subparsers = parser.add_subparsers(dest='command')
    
    # Vote command
    vote_parser = subparsers.add_parser('vote', help='Compute ensemble vote')
    vote_parser.add_argument('--date', help='Date for signal (default: latest)')
    
    # Recommend command
    rec_parser = subparsers.add_parser('recommend', help='Generate allocation recommendation')
    rec_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation SPY/GLD/TLT')
    rec_parser.add_argument('--max-shift', type=float, default=0.10, help='Max allocation shift')
    
    # Explain command
    exp_parser = subparsers.add_parser('explain', help='Explain current vote reasoning')
    
    args = parser.parse_args()
    
    voter = EnsembleVoter()
    
    if args.command == 'vote':
        readings = voter.collect_signals(args.date)
        vote = voter.compute_vote(readings)

        logger.info("Ensemble Vote")
        logger.info("Timestamp: %s", vote.timestamp)
        logger.info("Regime: %s (confidence: %.1f%%)", vote.regime.value.upper(), vote.regime_confidence * 100)
        logger.info("Sources: %d", vote.num_sources)
        logger.info("Consensus: %+.3f", vote.weighted_consensus)
        logger.info("Agreement: %.1f%%", vote.agreement_ratio * 100)
        logger.info("Asset Biases:")
        logger.info("  Equity (SPY):   %+.3f", vote.equity_bias)
        logger.info("  Duration (TLT): %+.3f", vote.duration_bias)
        logger.info("  Gold (GLD):     %+.3f", vote.gold_bias)
        logger.info("Recommended Action: %s", vote.action.upper())
        logger.info("Confidence: %.1f%%", vote.confidence * 100)

    elif args.command == 'recommend':
        weights = [float(w) / 100 for w in args.portfolio.split('/')]
        base = {'SPY': weights[0], 'GLD': weights[1], 'TLT': weights[2]}

        vote = voter.compute_vote()
        rec = voter.recommend_allocation(base, vote, args.max_shift)

        logger.info("Allocation Recommendation")
        logger.info("Base: %s", args.portfolio)
        logger.info("Regime: %s (confidence: %.1f%%)", rec['regime'].upper(), rec['confidence'] * 100)
        logger.info("Consensus: %+.3f", rec['consensus'])
        logger.info("Recommended Allocation:")
        for asset, data in rec['assets'].items():
            logger.info("  %s: %.1f%% -> %.1f%% (shift: %+.1f%%)",
                        asset, data['base'] * 100, data['new'] * 100, data['normalized_shift'] * 100)

    elif args.command == 'explain':
        vote = voter.compute_vote()

        logger.info("Ensemble Vote Explanation")
        logger.info(vote.reasoning)
        logger.info("Active Sources (%d):", len(vote.source_votes))
        for src in vote.source_votes:
            logger.info("  %25s | value: %+.3f | weight: %.2f | conf: %.1f%%",
                        src.source.value, src.value, src.weight, src.confidence * 100)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
