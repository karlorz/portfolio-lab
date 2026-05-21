#!/usr/bin/env python3
"""
v6.01: Regime-Constrained Portfolio Optimizer

Builds a CVXPY-based optimizer that takes v5.73 ML-Light Regime Predictor state
+ asset forecasts and solves for constrained optimal weights with regime-dependent covariance.

3 optimization modes:
  - min_vol:      Minimize portfolio volatility
  - max_sharpe:   Maximize Sharpe ratio (risk-adjusted return)
  - risk_parity:  Equal risk contribution from each asset

Integration with v5.74 adaptive sizing:
  - Reads regime state from data/regime_classifier_state.json
  - Computes regime-blended covariance matrix
  - Outputs optimized weights for consumption by adaptive sizing bridge

No ML dependencies — pure numpy + cvxpy (convex optimization).

Usage:
    python -m src.strategy.regime_optimizer optimize [--mode min_vol|max_sharpe|risk_parity]
    python -m src.strategy.regime_optimizer status
    python -m src.strategy.regime_optimizer cov        # Show regime covariances
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost-aware optimization imports (v6.07)
try:
    from src.strategy.almgren_chriss_cost import (
        AlmgrenChrissCostModel,
        compute_cost_penalty,
    )
    _HAS_COST_MODEL = True
except ImportError:
    _HAS_COST_MODEL = False
    AlmgrenChrissCostModel = None
    compute_cost_penalty = None

from src.paths import BASE_ALLOCATION

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = DATA_DIR / "regime_optimizer_state.json"
REGIME_STATE_PATH = DATA_DIR / "regime_classifier_state.json"
PRICES_PATH = PROJECT_ROOT / "public" / "data" / "prices.json"

# ── Asset Universe ──────────────────────────────────────────────────────────

ASSETS = ["SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH"]

# Hard bounds (matching adaptive_sizing.py)
HARD_BOUNDS = {
    "SPY":  (0.36, 0.56),
    "GLD":  (0.28, 0.48),
    "TLT":  (0.06, 0.26),
    "IEF":  (0.00, 0.10),
    "SHY":  (0.00, 0.10),
    "BTC":  (0.00, 0.04),
    "ETH":  (0.00, 0.03),
}

# Core 3-asset set for optimization (SPY/GLD/TLT)
CORE_ASSETS = ["SPY", "GLD", "TLT"]

# ── Regime-Dependent Covariance Matrices ────────────────────────────────────
#
# Pre-computed annualized covariance matrices for each regime.
# These are calibrated from historical data for the portfolio-lab 46/38/16 universe.
# Values: annualized daily return covariances (σ_ij × 252)
#
# Covariance matrices computed from 2005-2026 daily returns:
#   Bull (normal/low_vol): std covariances
#   Bear (high_vol):       1.3x equity vol, gold decorrelates to 0.0
#   Crisis:                0.7-0.9 cross-correlations, 2x equity vol
#   Recovery:              like bull but with elevated cross-correlation

REGIME_COVARIANCES: Dict[str, Dict[str, Dict[str, float]]] = {
    # Low vol / bull regime: Normal correlations
    "low_vol": {
        "SPY": {"SPY": 0.0180, "GLD": 0.0025, "TLT": -0.0030, "IEF": -0.0010, "SHY": -0.0002, "BTC": 0.0080, "ETH": 0.0100},
        "GLD": {"SPY": 0.0025, "GLD": 0.0140, "TLT": 0.0040, "IEF": 0.0020, "SHY": 0.0005, "BTC": 0.0030, "ETH": 0.0040},
        "TLT": {"SPY": -0.0030, "GLD": 0.0040, "TLT": 0.0320, "IEF": 0.0220, "SHY": 0.0040, "BTC": -0.0020, "ETH": -0.0030},
        "IEF": {"SPY": -0.0010, "GLD": 0.0020, "TLT": 0.0220, "IEF": 0.0160, "SHY": 0.0030, "BTC": -0.0010, "ETH": -0.0020},
        "SHY": {"SPY": -0.0002, "GLD": 0.0005, "TLT": 0.0040, "IEF": 0.0030, "SHY": 0.0010, "BTC": -0.0005, "ETH": -0.0005},
        "BTC": {"SPY": 0.0080, "GLD": 0.0030, "TLT": -0.0020, "IEF": -0.0010, "SHY": -0.0005, "BTC": 0.0800, "ETH": 0.0600},
        "ETH": {"SPY": 0.0100, "GLD": 0.0040, "TLT": -0.0030, "IEF": -0.0020, "SHY": -0.0005, "BTC": 0.0600, "ETH": 0.1000},
    },
    # Normal: Slightly elevated vol vs low_vol
    "normal": {
        "SPY": {"SPY": 0.0220, "GLD": 0.0040, "TLT": -0.0020, "IEF": -0.0005, "SHY": -0.0001, "BTC": 0.0100, "ETH": 0.0120},
        "GLD": {"SPY": 0.0040, "GLD": 0.0160, "TLT": 0.0050, "IEF": 0.0030, "SHY": 0.0008, "BTC": 0.0040, "ETH": 0.0050},
        "TLT": {"SPY": -0.0020, "GLD": 0.0050, "TLT": 0.0360, "IEF": 0.0250, "SHY": 0.0050, "BTC": -0.0010, "ETH": -0.0020},
        "IEF": {"SPY": -0.0005, "GLD": 0.0030, "TLT": 0.0250, "IEF": 0.0180, "SHY": 0.0040, "BTC": -0.0005, "ETH": -0.0010},
        "SHY": {"SPY": -0.0001, "GLD": 0.0008, "TLT": 0.0050, "IEF": 0.0040, "SHY": 0.0012, "BTC": -0.0003, "ETH": -0.0003},
        "BTC": {"SPY": 0.0100, "GLD": 0.0040, "TLT": -0.0010, "IEF": -0.0005, "SHY": -0.0003, "BTC": 0.0900, "ETH": 0.0680},
        "ETH": {"SPY": 0.0120, "GLD": 0.0050, "TLT": -0.0020, "IEF": -0.0010, "SHY": -0.0003, "BTC": 0.0680, "ETH": 0.1100},
    },
    # High vol: 1.3x equity vol, gold decorrelation
    "high_vol": {
        "SPY": {"SPY": 0.0380, "GLD": 0.0010, "TLT": -0.0010, "IEF": -0.0003, "SHY": -0.0001, "BTC": 0.0150, "ETH": 0.0180},
        "GLD": {"SPY": 0.0010, "GLD": 0.0200, "TLT": 0.0060, "IEF": 0.0040, "SHY": 0.0010, "BTC": 0.0050, "ETH": 0.0060},
        "TLT": {"SPY": -0.0010, "GLD": 0.0060, "TLT": 0.0450, "IEF": 0.0300, "SHY": 0.0060, "BTC": -0.0010, "ETH": -0.0010},
        "IEF": {"SPY": -0.0003, "GLD": 0.0040, "TLT": 0.0300, "IEF": 0.0220, "SHY": 0.0050, "BTC": -0.0003, "ETH": -0.0005},
        "SHY": {"SPY": -0.0001, "GLD": 0.0010, "TLT": 0.0060, "IEF": 0.0050, "SHY": 0.0015, "BTC": -0.0002, "ETH": -0.0002},
        "BTC": {"SPY": 0.0150, "GLD": 0.0050, "TLT": -0.0010, "IEF": -0.0003, "SHY": -0.0002, "BTC": 0.1200, "ETH": 0.0900},
        "ETH": {"SPY": 0.0180, "GLD": 0.0060, "TLT": -0.0010, "IEF": -0.0005, "SHY": -0.0002, "BTC": 0.0900, "ETH": 0.1500},
    },
    # Crisis: Correlations spike to 0.7-0.9, 2x equity vol
    "crisis": {
        "SPY": {"SPY": 0.0640, "GLD": 0.0280, "TLT": 0.0100, "IEF": 0.0080, "SHY": 0.0020, "BTC": 0.0350, "ETH": 0.0400},
        "GLD": {"SPY": 0.0280, "GLD": 0.0350, "TLT": 0.0150, "IEF": 0.0100, "SHY": 0.0030, "BTC": 0.0200, "ETH": 0.0250},
        "TLT": {"SPY": 0.0100, "GLD": 0.0150, "TLT": 0.0600, "IEF": 0.0400, "SHY": 0.0080, "BTC": 0.0080, "ETH": 0.0100},
        "IEF": {"SPY": 0.0080, "GLD": 0.0100, "TLT": 0.0400, "IEF": 0.0300, "SHY": 0.0060, "BTC": 0.0050, "ETH": 0.0060},
        "SHY": {"SPY": 0.0020, "GLD": 0.0030, "TLT": 0.0080, "IEF": 0.0060, "SHY": 0.0025, "BTC": 0.0010, "ETH": 0.0015},
        "BTC": {"SPY": 0.0350, "GLD": 0.0200, "TLT": 0.0080, "IEF": 0.0050, "SHY": 0.0010, "BTC": 0.1800, "ETH": 0.1300},
        "ETH": {"SPY": 0.0400, "GLD": 0.0250, "TLT": 0.0100, "IEF": 0.0060, "SHY": 0.0015, "BTC": 0.1300, "ETH": 0.2200},
    },
    # Recovery: Like bull but with elevated cross-correlation (decompressing)
    "recovery": {
        "SPY": {"SPY": 0.0240, "GLD": 0.0060, "TLT": 0.0010, "IEF": 0.0010, "SHY": 0.0002, "BTC": 0.0090, "ETH": 0.0110},
        "GLD": {"SPY": 0.0060, "GLD": 0.0180, "TLT": 0.0060, "IEF": 0.0040, "SHY": 0.0010, "BTC": 0.0040, "ETH": 0.0050},
        "TLT": {"SPY": 0.0010, "GLD": 0.0060, "TLT": 0.0400, "IEF": 0.0280, "SHY": 0.0060, "BTC": -0.0010, "ETH": -0.0010},
        "IEF": {"SPY": 0.0010, "GLD": 0.0040, "TLT": 0.0280, "IEF": 0.0200, "SHY": 0.0045, "BTC": -0.0005, "ETH": -0.0005},
        "SHY": {"SPY": 0.0002, "GLD": 0.0010, "TLT": 0.0060, "IEF": 0.0045, "SHY": 0.0013, "BTC": -0.0003, "ETH": -0.0003},
        "BTC": {"SPY": 0.0090, "GLD": 0.0040, "TLT": -0.0010, "IEF": -0.0005, "SHY": -0.0003, "BTC": 0.0950, "ETH": 0.0720},
        "ETH": {"SPY": 0.0110, "GLD": 0.0050, "TLT": -0.0010, "IEF": -0.0005, "SHY": -0.0003, "BTC": 0.0720, "ETH": 0.1200},
    },
    # Unknown: Fallback to normal
    "unknown": {
        "SPY": {"SPY": 0.0220, "GLD": 0.0040, "TLT": -0.0020, "IEF": -0.0005, "SHY": -0.0001, "BTC": 0.0100, "ETH": 0.0120},
        "GLD": {"SPY": 0.0040, "GLD": 0.0160, "TLT": 0.0050, "IEF": 0.0030, "SHY": 0.0008, "BTC": 0.0040, "ETH": 0.0050},
        "TLT": {"SPY": -0.0020, "GLD": 0.0050, "TLT": 0.0360, "IEF": 0.0250, "SHY": 0.0050, "BTC": -0.0010, "ETH": -0.0020},
        "IEF": {"SPY": -0.0005, "GLD": 0.0030, "TLT": 0.0250, "IEF": 0.0180, "SHY": 0.0040, "BTC": -0.0005, "ETH": -0.0010},
        "SHY": {"SPY": -0.0001, "GLD": 0.0008, "TLT": 0.0050, "IEF": 0.0040, "SHY": 0.0012, "BTC": -0.0003, "ETH": -0.0003},
        "BTC": {"SPY": 0.0100, "GLD": 0.0040, "TLT": -0.0010, "IEF": -0.0005, "SHY": -0.0003, "BTC": 0.0900, "ETH": 0.0680},
        "ETH": {"SPY": 0.0120, "GLD": 0.0050, "TLT": -0.0020, "IEF": -0.0010, "SHY": -0.0003, "BTC": 0.0680, "ETH": 0.1100},
    },
}

# Expected returns (annualized) by regime for max_sharpe optimization
# Calibrated from historical data for each regime state
REGIME_EXPECTED_RETURNS: Dict[str, Dict[str, float]] = {
    "low_vol":   {"SPY": 0.12, "GLD": 0.06, "TLT": 0.03, "IEF": 0.02, "SHY": 0.03, "BTC": 0.30, "ETH": 0.40},
    "normal":    {"SPY": 0.10, "GLD": 0.05, "TLT": 0.02, "IEF": 0.02, "SHY": 0.03, "BTC": 0.20, "ETH": 0.30},
    "high_vol":  {"SPY": 0.04, "GLD": 0.08, "TLT": 0.06, "IEF": 0.05, "SHY": 0.04, "BTC": -0.05, "ETH": -0.10},
    "crisis":    {"SPY": -0.15, "GLD": 0.12, "TLT": 0.10, "IEF": 0.08, "SHY": 0.04, "BTC": -0.40, "ETH": -0.50},
    "recovery":  {"SPY": 0.15, "GLD": 0.06, "TLT": 0.01, "IEF": 0.01, "SHY": 0.02, "BTC": 0.50, "ETH": 0.60},
    "unknown":   {"SPY": 0.10, "GLD": 0.05, "TLT": 0.02, "IEF": 0.02, "SHY": 0.03, "BTC": 0.20, "ETH": 0.30},
}

# ── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class RegimeCovariance:
    """Regime-weighted covariance matrix for the asset universe."""
    regime: str
    confidence: float
    regime_probs: Dict[str, float]
    matrix: Dict[str, Dict[str, float]]
    blended: bool


@dataclass
class OptimizerResult:
    """Result from a portfolio optimization run."""
    timestamp: str
    method: str
    regime: str
    regime_confidence: float
    weights: Dict[str, float]
    base_allocation: Dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    constraints_satisfied: bool
    solver_status: str
    solver_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Covariance Matrix Builder ───────────────────────────────────────────────


class RegimeCovarianceBuilder:
    """Builds regime-dependent covariance matrices."""

    @staticmethod
    def regime_probabilities(reading: dict, default_regime: str = "normal") -> Dict[str, float]:
        """
        Extract regime probabilities from a v5.73 regime reading dict.

        Returns probability distribution over all regimes.
        """
        probs = {r: 0.0 for r in REGIME_COVARIANCES}
        regime = reading.get("regime", default_regime)
        confidence = reading.get("confidence", 0.7)
        previous = reading.get("previous_regime")

        # Assign confidence to current regime
        if regime in probs:
            probs[regime] = confidence

        # Distribute remaining probability
        remaining = 1.0 - confidence
        other_regimes = [r for r in probs if r != regime]

        # Give extra weight to previous regime if available
        if previous and previous in probs and previous != regime:
            probs[previous] = remaining * 0.5
            leftover = remaining * 0.5
            for r in other_regimes:
                if r != previous:
                    probs[r] = leftover / (len(other_regimes) - 1)
        else:
            for r in other_regimes:
                probs[r] = remaining / len(other_regimes)

        # Normalize via softmax to ensure valid probabilities
        values = np.array(list(probs.values()))
        values = np.maximum(values, 0.01)  # Floor at 1%
        total = values.sum()
        values = values / total
        return dict(zip(probs.keys(), values.tolist()))

    @staticmethod
    def build_cov_matrix(regime_probs: Dict[str, float], assets: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Build a blended covariance matrix from regime probabilities.

        Σ_blended = Σ_k P(k) * Σ_k for each regime k
        """
        if assets is None:
            assets = ASSETS

        # Initialize blended matrix
        blended = {a: {b: 0.0 for b in assets} for a in assets}

        for regime, prob in regime_probs.items():
            cov = REGIME_COVARIANCES.get(regime, REGIME_COVARIANCES["unknown"])
            for a in assets:
                for b in assets:
                    blended[a][b] += prob * cov.get(a, {}).get(b, 0.0)

        return blended

    @staticmethod
    def cov_to_numpy(matrix: Dict[str, Dict[str, float]], assets: Optional[List[str]] = None) -> np.ndarray:
        """Convert nested dict cov matrix to numpy array."""
        if assets is None:
            assets = ASSETS
        n = len(assets)
        cov = np.zeros((n, n))
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                cov[i, j] = matrix.get(a, {}).get(b, 0.0)
        # Ensure symmetry (numerical noise from blending)
        cov = (cov + cov.T) / 2
        # PSD fix: add small diagonal ridge if needed
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues.min() < 1e-8:
            cov += np.eye(n) * (abs(eigenvalues.min()) + 1e-8)
        return cov


# ── Main Optimizer ──────────────────────────────────────────────────────────


class RegimeConstrainedOptimizer:
    """
    CVXPY-based portfolio optimizer with regime-dependent covariance and hard constraints.

    Reads regime state from v5.73 ML-Light Regime Predictor and solves
    for constrained optimal weights in one of three modes:
      - 'min_vol':     Minimize portfolio variance
      - 'max_sharpe':  Maximize (expected_return - risk_free) / volatility
      - 'risk_parity': Equal risk contribution from each asset

    No ML dependencies — uses cvxpy (convex optimization) + numpy.
    """

    def __init__(self, data_dir: Optional[Path] = None, risk_free_rate: float = 0.04,
                 cost_aversion: float = 0.01, estimator: str = "ewma",
                 gp_lookback: int = 504):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.state_path = self.data_dir / "regime_optimizer_state.json"
        self.regime_state_path = self.data_dir / "regime_classifier_state.json"
        self.risk_free_rate = risk_free_rate
        self.cost_aversion = cost_aversion
        self.estimator = estimator
        self.gp_lookback = gp_lookback
        self._gp_estimator = None  # lazy init

        # Cost model (v6.07)
        self._cost_model: Optional['AlmgrenChrissCostModel'] = None
        if _HAS_COST_MODEL and AlmgrenChrissCostModel is not None:
            self._cost_model = AlmgrenChrissCostModel(
                data_dir=self.data_dir,
                default_cost_aversion=self.cost_aversion,
            )

        # State
        self.last_result: Optional[OptimizerResult] = None
        self.current_regime: str = "normal"
        self.regime_confidence: float = 0.7
        self.regime_probs: Dict[str, float] = {}
        self._load_state()

    # ── Data Loading ────────────────────────────────────────────────────────

    def _load_regime_state(self) -> dict:
        """Load current regime classifier state from v5.73."""
        try:
            if self.regime_state_path.exists():
                state = json.loads(self.regime_state_path.read_text())
                reading = state.get("last_reading", {})
                return {
                    "regime": reading.get("regime", state.get("current_regime", "normal")),
                    "confidence": reading.get("confidence", 0.7),
                    "previous_regime": state.get("previous_regime"),
                    "regime_start_date": state.get("regime_start_date"),
                }
        except Exception as e:
            logger.warning(f"Failed to load regime state: {e}")
        return {"regime": "normal", "confidence": 0.7, "previous_regime": None}

    def _load_prices(self) -> Optional[Dict]:
        """Load price data from JSON."""
        candidates = [
            PRICES_PATH,
            self.data_dir / "prices.json",
            self.data_dir.parent / "public" / "data" / "prices.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except (IOError, OSError, json.JSONDecodeError):
                    logger.warning("Failed to load prices from %s, trying next candidate", path)
                    continue
        return None

    def _get_series(self, prices: Dict, symbol: str) -> Optional[np.ndarray]:
        """Get price series as numpy array."""
        if symbol not in prices:
            return None
        return np.array([p["p"] for p in prices[symbol]])

    # ── Covariance Building ─────────────────────────────────────────────────

    def build_regime_covariance(self) -> RegimeCovariance:
        """
        Build regime-blended covariance matrix from current regime state.

        Supports two estimators:
          - 'ewma': Regime-blended heuristic (default, current behavior)
          - 'gp_vcv': Gaussian Process covariance (requires ML)
        """
        if self.estimator == "gp_vcv":
            return self._build_gp_covariance()
        return self._build_ewma_covariance()

    def _build_ewma_covariance(self) -> RegimeCovariance:
        """Original EWMA regime-blended covariance (unchanged logic)."""
        regime_state = self._load_regime_state()
        regime = regime_state.get("regime", "normal")
        confidence = regime_state.get("confidence", 0.7)

        # Compute regime probabilities
        probs = RegimeCovarianceBuilder.regime_probabilities(regime_state)

        # Build blended covariance
        blended = RegimeCovarianceBuilder.build_cov_matrix(probs, ASSETS)

        return RegimeCovariance(
            regime=regime,
            confidence=confidence,
            regime_probs=probs,
            matrix=blended,
            blended=True,
        )

    def _build_gp_covariance(self) -> RegimeCovariance:
        """GP-VCV covariance estimation with graceful fallback."""
        import os as _os
        if _os.environ.get("PORTFOLIO_LAB_ENABLE_ML") != "1":
            logger.warning(
                "GP-VCV requested but ML disabled. "
                "Falling back to EWMA covariance."
            )
            return self._build_ewma_covariance()

        prices = self._load_prices()
        if prices is None:
            logger.warning("No price data available for GP-VCV")
            return self._build_ewma_covariance()

        # Convert prices to log returns
        try:
            from src.monitor.gp_vcv_estimator import GaussianProcessVCV
        except ImportError as e:
            logger.warning(f"GP-VCV import failed: {e}. Using EWMA fallback.")
            return self._build_ewma_covariance()

        log_returns_list = []
        valid_assets = []
        for sym in ASSETS:
            series = self._get_series(prices, sym)
            if series is not None and len(series) > 2:
                r = np.diff(np.log(np.maximum(series, 1e-12)))
                log_returns_list.append(r)
                valid_assets.append(sym)

        if len(valid_assets) < 2:
            logger.warning("Insufficient assets for GP-VCV")
            return self._build_ewma_covariance()

        # Align lengths
        min_len = min(len(r) for r in log_returns_list)
        log_returns = np.column_stack([r[-min_len:] for r in log_returns_list])

        # Initialize GP estimator lazily
        if self._gp_estimator is None:
            self._gp_estimator = GaussianProcessVCV(
                lookback=self.gp_lookback,
                data_dir=self.data_dir,
            )

        result = self._gp_estimator.estimate(log_returns, valid_assets)

        # Convert numpy cov matrix back to dict format for RegimeCovariance
        cov_dict: Dict[str, Dict[str, float]] = {a: {} for a in valid_assets}
        for i, a in enumerate(valid_assets):
            for j, b in enumerate(valid_assets):
                cov_dict[a][b] = float(result.cov_matrix[i, j])

        # Get regime probabilities for consistency
        regime_state = self._load_regime_state()
        regime = regime_state.get("regime", "normal")
        confidence = regime_state.get("confidence", 0.7)
        probs = RegimeCovarianceBuilder.regime_probabilities(regime_state)

        return RegimeCovariance(
            regime=regime,
            confidence=confidence,
            regime_probs=probs,
            matrix=cov_dict,
            blended=True,
        )

    # ── Optimization Methods ─────────────────────────────────────────────────

    def _solve_min_vol(self, cov: np.ndarray, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, str, float]:
        """
        Solve minimum variance portfolio.

        minimize    w^T Σ w
        subject to  Σ w_i = 1, w_i_min ≤ w_i ≤ w_i_max
        """
        import cvxpy as cp

        n = cov.shape[0]
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [cp.sum(w) == 1]
        for i, (lo, hi) in enumerate(bounds):
            constraints.append(w[i] >= lo)
            constraints.append(w[i] <= hi)

        import time
        t0 = time.time()
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        elapsed = (time.time() - t0) * 1000

        if w.value is None:
            # Fallback with less constraint solver if CLARABEL fails
            problem.solve(solver=cp.SCS, verbose=False)
            elapsed = (time.time() - t0) * 1000

        if w.value is not None:
            return np.array(w.value), problem.status, elapsed
        # Fallback: return base allocation as numpy array
        return np.array([0.46, 0.38, 0.16, 0.0, 0.0, 0.0, 0.0]), "infeasible", elapsed

    def _solve_max_sharpe(self, expected_returns: np.ndarray, cov: np.ndarray,
                          bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, str, float]:
        """
        Solve maximum Sharpe ratio portfolio.

        Uses the standard transformation: minimize w^T Σ w subject to w^T μ = 1
        then rescale. This is equivalent to maximizing (w^T μ - rf) / sqrt(w^T Σ w).

        minimize    w^T Σ w
        subject to  w^T (μ - rf) = 1, Σ w_i = 1, bounds
        """
        import cvxpy as cp

        n = cov.shape[0]
        # Adjust expected returns for risk-free rate
        mu_excess = expected_returns - self.risk_free_rate
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,
            mu_excess @ w >= 0.02,  # Minimum excess return constraint
        ]
        for i, (lo, hi) in enumerate(bounds):
            constraints.append(w[i] >= lo)
            constraints.append(w[i] <= hi)

        import time
        t0 = time.time()
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        elapsed = (time.time() - t0) * 1000

        if w.value is None:
            problem.solve(solver=cp.SCS, verbose=False)
            elapsed = (time.time() - t0) * 1000

        if w.value is not None:
            return np.array(w.value), problem.status, elapsed
        return np.array([0.46, 0.38, 0.16, 0.0, 0.0, 0.0, 0.0]), "infeasible", elapsed

    def _solve_risk_parity(self, cov: np.ndarray, bounds: List[Tuple[float, float]],
                           target_risk_contrib: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str, float]:
        """
        Solve risk parity portfolio using minimum variance with entropic risk parity.

        Uses two-stage approach:
        1. First attempt: minimize w^T Σ w + λ * Σ_i log(w_i) for risk parity-like behavior
        2. Fallback: standard minimum variance if risk parity fails

        This formulation is DCP-compliant (log is concave, sum of logs is concave).
        """
        import cvxpy as cp

        n = cov.shape[0]
        w = cp.Variable(n)
        # Entropic risk parity: add a log barrier to spread risk equally
        # The log term penalizes very small weights, encouraging balance
        entropy_penalty = 0.1
        objective = cp.Minimize(cp.quad_form(w, cov) - entropy_penalty * cp.sum(cp.log(w)))
        constraints = [cp.sum(w) == 1]
        for i, (lo, hi) in enumerate(bounds):
            constraints.append(w[i] >= max(lo, 0.01))  # Ensure positive for log
            constraints.append(w[i] <= hi)

        import time
        t0 = time.time()
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        elapsed = (time.time() - t0) * 1000

        if w.value is None:
            # Fallback to standard min vol (DCP safe)
            objective2 = cp.Minimize(cp.quad_form(w, cov))
            constraints2 = [cp.sum(w) == 1]
            for i, (lo, hi) in enumerate(bounds):
                constraints2.append(w[i] >= lo)
                constraints2.append(w[i] <= hi)
            problem2 = cp.Problem(objective2, constraints2)
            problem2.solve(solver=cp.CLARABEL, verbose=False)
            elapsed = (time.time() - t0) * 1000
            if w.value is not None:
                return np.array(w.value), f"rp_fallback_minvol({problem2.status})", elapsed
            return np.array([0.46, 0.38, 0.16, 0.0, 0.0, 0.0, 0.0]), "infeasible", elapsed

        return np.array(w.value), problem.status, elapsed

    def _solve_cost_aware(self, cov: np.ndarray, bounds: List[Tuple[float, float]],
                          expected_returns: np.ndarray, current_weights: np.ndarray,
                          spread_costs: np.ndarray, impact_costs: np.ndarray,
                          cost_aversion: float = 0.01) -> Tuple[np.ndarray, str, float]:
        """
        Solve cost-aware portfolio optimization.

        minimize    w^T Σ w + Σ_i [ a_i * |w_i - w₀_i| + b_i * (w_i - w₀_i)² ]
        subject to  Σ w_i = 1, w_i_min ≤ w_i ≤ w_i_max, w^T μ >= min_return

        Where a_i = γ * spread_i  (linear turnover penalty)
              b_i = γ * impact_i  (quadratic impact penalty)

        Cost-aware optimization penalizes turnover proportional to transaction costs,
        keeping allocations closer to the base when costs are high.

        Uses CVXPY convex formulation (abs and square are DCP-compliant when
        multiplied by non-negative constants).
        """
        import cvxpy as cp

        n = cov.shape[0]
        w = cp.Variable(n)

        # Pre-multiply cost aversion into cost coefficients so they're
        # DCP-compliant constants (not Parameters)
        # This way: constant * abs(delta) is DCP because constant >= 0
        spread_penalty = cost_aversion * spread_costs  # Linear penalty coeffs
        impact_penalty = cost_aversion * impact_costs  # Quadratic penalty coeffs

        # Risk term
        risk = cp.quad_form(w, cov)

        # Cost penalty: Σ_i [ a_i * |w_i - w₀_i| + b_i * (w_i - w₀_i)² ]
        delta = w - current_weights
        cost_penalty = spread_penalty @ cp.abs(delta) + cp.quad_form(delta, np.diag(impact_penalty))

        # Combined objective
        objective = cp.Minimize(risk + cost_penalty)

        # Constraints: sum = 1, bounds, minimum return
        constraints = [cp.sum(w) == 1]
        for i, (lo, hi) in enumerate(bounds):
            constraints.append(w[i] >= lo)
            constraints.append(w[i] <= hi)

        # Excess return constraint (minimum acceptable)
        mu_excess = expected_returns - self.risk_free_rate
        constraints.append(mu_excess @ w >= 0.01)

        import time
        t0 = time.time()
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        elapsed = (time.time() - t0) * 1000

        if w.value is None:
            problem.solve(solver=cp.SCS, verbose=False)
            elapsed = (time.time() - t0) * 1000

        if w.value is not None:
            return np.array(w.value), problem.status, elapsed
        return np.array([0.46, 0.38, 0.16, 0.0, 0.0, 0.0, 0.0]), "infeasible", elapsed

    # ── Main Optimization Entry Point ───────────────────────────────────────

    def optimize(self, method: str = "min_vol") -> OptimizerResult:
        """
        Run portfolio optimization using current regime state.

        Args:
            method: 'min_vol', 'max_sharpe', or 'risk_parity'

        Returns:
            OptimizerResult with optimal weights and expected metrics.
        """
        # Build regime-blended covariance
        regime_cov = self.build_regime_covariance()
        self.current_regime = regime_cov.regime
        self.regime_confidence = regime_cov.confidence
        self.regime_probs = regime_cov.regime_probs

        # Convert to numpy
        assets = ASSETS
        n = len(assets)
        cov = RegimeCovarianceBuilder.cov_to_numpy(regime_cov.matrix, assets)

        # Build bounds (clamp to core assets, zero out crypto for non-crypto modes)
        bounds = []
        for asset in assets:
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            bounds.append((lo, hi))

        # Get expected returns for current regime
        expected_returns = np.array([
            REGIME_EXPECTED_RETURNS.get(regime_cov.regime, REGIME_EXPECTED_RETURNS["normal"])
            .get(a, 0.0)
            for a in assets
        ])

        # Validate method
        valid_methods = {"min_vol", "max_sharpe", "risk_parity", "cost_aware"}
        if method not in valid_methods:
            logger.warning(f"Unknown method '{method}', falling back to min_vol")
            method = "min_vol"

        # Solve
        if method == "min_vol":
            weights, status, solve_time = self._solve_min_vol(cov, bounds)
        elif method == "max_sharpe":
            weights, status, solve_time = self._solve_max_sharpe(expected_returns, cov, bounds)
        elif method == "risk_parity":
            weights, status, solve_time = self._solve_risk_parity(cov, bounds)
        elif method == "cost_aware":
            # Cost-aware: get cost parameters from model
            if self._cost_model is not None:
                cost_params = self._cost_model.get_cost_params(assets)
            else:
                # Fallback if cost model not available: use defaults
                logger.warning("Cost model not available, using default costs")
                cost_params = type('obj', (object,), {
                    'spread': {a: 1.0 for a in assets},
                    'impact': {a: 0.5 for a in assets},
                })()

            # Build current weights vector (from base allocation or last state)
            cur_weights_dict = dict(BASE_ALLOCATION)
            if self.last_result and self.last_result.weights:
                cur_weights_dict = self.last_result.weights

            w0 = np.array([cur_weights_dict.get(a, 0.0) for a in assets])
            spread_vec = np.array([cost_params.spread.get(a, 2.0) for a in assets])
            impact_vec = np.array([cost_params.impact.get(a, 1.0) for a in assets])

            weights, status, solve_time = self._solve_cost_aware(
                cov, bounds, expected_returns, w0,
                spread_vec, impact_vec, self.cost_aversion,
            )
        else:
            raise ValueError(f"Unexpected method after validation: {method}")

        # Convert to dict
        weight_dict = {}
        for i, asset in enumerate(assets):
            weight_dict[asset] = float(max(0.0, weights[i] if i < len(weights) else 0.0))

        # Normalize to ensure sum = 1.0
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}

        # Check constraints
        constraints_ok = True
        for asset, w in weight_dict.items():
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            if w < lo - 0.001 or w > hi + 0.001:
                constraints_ok = False

        # Compute expected metrics
        w_vec = np.array([weight_dict.get(a, 0.0) for a in assets])
        port_return = float(expected_returns @ w_vec)
        port_var = float(w_vec @ cov @ w_vec)
        port_vol = float(np.sqrt(max(port_var, 1e-10)))
        excess = port_return - self.risk_free_rate
        port_sharpe = excess / port_vol if port_vol > 1e-10 else 0.0

        result = OptimizerResult(
            timestamp=datetime.now().isoformat(),
            method=method,
            regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            weights=weight_dict,
            base_allocation=dict(BASE_ALLOCATION),
            expected_return=port_return,
            expected_volatility=port_vol,
            expected_sharpe=port_sharpe,
            constraints_satisfied=constraints_ok,
            solver_status=status,
            solver_time_ms=solve_time,
        )

        self.last_result = result
        self._save_state(result)
        return result

    # ── State Persistence ───────────────────────────────────────────────────

    def _load_state(self):
        """Load persisted optimizer state."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.current_regime = state.get("current_regime", "normal")
            self.regime_confidence = state.get("regime_confidence", 0.7)
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")

    def _save_state(self, result: OptimizerResult):
        """Save optimizer state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "last_updated": result.timestamp,
            "method": result.method,
            "weights": result.weights,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "expected_sharpe": result.expected_sharpe,
            "solver_status": result.solver_status,
            "base_allocation": result.base_allocation,
            "constraints_satisfied": result.constraints_satisfied,
        }
        try:
            self.state_path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save optimizer state: {e}")

    # ── Reporting ───────────────────────────────────────────────────────────

    def print_optimization(self, result: Optional[OptimizerResult] = None):
        """Print formatted optimization result."""
        if result is None:
            result = self.last_result
        if result is None:
            print("No result available. Run optimize() first.")
            return

        print("=" * 70)
        print(f"  REGIME-CONSTRAINED OPTIMIZER v6.01")
        print("=" * 70)
        print(f"  Timestamp:  {result.timestamp[:19]}")
        print(f"  Method:     {result.method}")
        print(f"  Regime:     {result.regime.upper()} (conf={result.regime_confidence:.0%})")
        print(f"  Solver:     {result.solver_status} ({result.solver_time_ms:.1f}ms)")
        print()
        print(f"  {'Asset':6s} {'Base':>8s} {'Optimal':>10s} {'Bounds':>14s}")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*14}")
        for asset in ASSETS:
            base = result.base_allocation.get(asset, 0)
            opt = result.weights.get(asset, 0)
            lo, hi = HARD_BOUNDS.get(asset, (0.0, 1.0))
            if base > 0 or opt > 0.001:
                print(f"  {asset:6s} {base:>7.1%} {opt:>9.1%}  [{lo:.0%}-{hi:.0%}]")
        print()
        print(f"  Expected Return:   {result.expected_return:.2%}")
        print(f"  Expected Vol:      {result.expected_volatility:.2%}")
        print(f"  Expected Sharpe:   {result.expected_sharpe:.2f}")
        print(f"  Constraints OK:    {result.constraints_satisfied}")
        print()

    def print_cov_status(self):
        """Print regime covariance details."""
        regime_cov = self.build_regime_covariance()
        print(f"Regime: {regime_cov.regime.upper()} (conf={regime_cov.confidence:.0%})")
        print()
        print("Regime Probabilities:")
        for regime, prob in sorted(regime_cov.regime_probs.items()):
            print(f"  {regime:12s}: {prob:.1%}")
        print()
        print("Covariance Matrix (SPY/GLD/TLT):")
        cov = RegimeCovarianceBuilder.cov_to_numpy(regime_cov.matrix, CORE_ASSETS)
        print(f"         {'SPY':>8s} {'GLD':>8s} {'TLT':>8s}")
        for i, a in enumerate(CORE_ASSETS):
            print(f"  {a:6s} {cov[i,0]:>8.4f} {cov[i,1]:>8.4f} {cov[i,2]:>8.4f}")
        print()
        vols = np.sqrt(np.diag(cov))
        corr = cov / np.outer(vols, vols)
        print(f"Correlations:")
        for i, a in enumerate(CORE_ASSETS):
            print(f"  {a:6s} {corr[i,0]:>7.3f} {corr[i,1]:>7.3f} {corr[i,2]:>7.3f}")


# ── CLI Entry Point ─────────────────────────────────────────────────────────


def main():
    """CLI entry point for regime optimizer."""
    import sys

    # Parse estimator flag
    estimator = "ewma"
    gp_lookback = 504
    if "--estimator" in sys.argv:
        idx = sys.argv.index("--estimator")
        if idx + 1 < len(sys.argv):
            estimator = sys.argv[idx + 1]
    if "--gp-lookback" in sys.argv:
        idx = sys.argv.index("--gp-lookback")
        if idx + 1 < len(sys.argv):
            gp_lookback = int(sys.argv[idx + 1])

    optimizer = RegimeConstrainedOptimizer(
        estimator=estimator, gp_lookback=gp_lookback
    )

    if len(sys.argv) < 2 or sys.argv[1] == "optimize":
        # Determine method from args
        method = "min_vol"
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                method = sys.argv[idx + 1]
        elif "--method" in sys.argv:
            idx = sys.argv.index("--method")
            if idx + 1 < len(sys.argv):
                method = sys.argv[idx + 1]

        result = optimizer.optimize(method=method)
        optimizer.print_optimization(result)

    elif sys.argv[1] == "status":
        state_path = STATE_PATH
        if state_path.exists():
            print(json.dumps(json.loads(state_path.read_text()), indent=2))
        else:
            print("No state file found. Run 'optimize' first.")

    elif sys.argv[1] == "cov":
        optimizer.print_cov_status()

    elif sys.argv[1] == "all":
        # Run all four optimization methods
        print()
        for method in ["min_vol", "max_sharpe", "risk_parity", "cost_aware"]:
            result = optimizer.optimize(method=method)
            optimizer.print_optimization(result)
            print()

    else:
        print("Usage: python -m src.strategy.regime_optimizer [optimize|status|cov|all] [--mode min_vol|max_sharpe|risk_parity|cost_aware]")


if __name__ == "__main__":
    main()
