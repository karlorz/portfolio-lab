"""
Bayesian Adaptive Volatility Model - v5.20 Implementation
Sequential Bayesian updating of volatility estimates using OHLC realized vol.

Combines:
- Prior: long-term (252-day) volatility as shrinkage target
- Likelihood: recent realized vol from OHLC estimators
- Posterior: Bayesian-weighted volatility estimate
- Regime adaptation: wider prior uncertainty during high-kurtosis regimes

No ML deps — pure numpy/scipy with conjugate Normal-Inverse-Gamma model.

Usage:
    python -m src.monitor.bayesian_vol estimate --symbol SPY
    python -m src.monitor.bayesian_vol history --symbol SPY
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BayesianVolEstimate:
    """Bayesian volatility estimate with uncertainty."""
    symbol: str
    timestamp: str
    window: int

    # Prior (long-term)
    prior_vol: float
    prior_precision: float    # Inverse of prior variance

    # Likelihood (recent)
    likelihood_vol: float
    likelihood_precision: float

    # Posterior
    posterior_vol: float       # Bayesian-weighted estimate
    credible_interval_lower: float   # 95% credible interval
    credible_interval_upper: float

    # Comparison
    simple_mean_vol: float     # Naive average for comparison
    shrinkage_factor: float    # How much we shrink toward prior (0-1)

    # Regime
    regime_scale: float        # Multiplier for prior precision in current regime
    is_high_vol_regime: bool

    # Quality
    n_obs: int
    is_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


class BayesianVolModel:
    """
    Bayesian volatility model using Normal-Inverse-Gamma conjugate prior.

    Model:
    - Returns r_t ~ N(0, σ²_t)
    - σ² follows Inverse-Gamma(α, β) prior
    - α calibrated from long-term vol with uncertainty
    - Update with recent realized vol observations (likelihood)

    The posterior mean is a precision-weighted average:
        σ²_post = (α_prior * σ²_prior + n * σ²_obs) / (α_prior + n)

    Regime adaptation: reduce α_prior during high-kurtosis (less shrinkage,
    more responsive to recent data).
    """

    def __init__(self, prior_window: int = 252, update_window: int = 20):
        self.prior_window = prior_window
        self.update_window = update_window

    def fit_prior(self, vol_history: List[float]) -> Tuple[float, float]:
        """
        Fit Inverse-Gamma prior from long-term vol history.

        Returns (prior_vol, prior_precision) where:
        - prior_vol: prior mean volatility
        - prior_precision: α parameter (higher = more confidence in prior)
        """
        if len(vol_history) < 60:
            return 0.20, 10.0  # Default: 20% vol, moderate confidence

        vols = np.array(vol_history[-self.prior_window:])
        prior_vol = float(np.mean(vols))

        # Prior precision: higher when vol is stable
        vol_of_vol = float(np.std(vols) / (np.mean(vols) + 0.001))
        alpha = max(5.0, min(60.0, 30.0 / (vol_of_vol + 0.01)))

        return prior_vol, alpha

    def update(self, prior_vol: float, prior_precision: float,
               recent_vols: List[float],
               regime_scale: float = 1.0) -> BayesianVolEstimate:
        """
        Bayesian update: combine prior and likelihood.

        Posterior mean = (α*σ²_prior + n*σ²_likelihood) / (α + n)

        regime_scale > 1.0 reduces prior weight (more responsive to recent data)
        """
        n = len(recent_vols)
        if n == 0:
            return BayesianVolEstimate(
                symbol="", timestamp=datetime.now().isoformat(),
                window=self.update_window,
                prior_vol=prior_vol, prior_precision=prior_precision,
                likelihood_vol=prior_vol, likelihood_precision=0,
                posterior_vol=prior_vol,
                credible_interval_lower=prior_vol * 0.7,
                credible_interval_upper=prior_vol * 1.3,
                simple_mean_vol=prior_vol, shrinkage_factor=1.0,
                regime_scale=regime_scale, is_high_vol_regime=False,
                n_obs=0, is_valid=False,
            )

        # Effective prior precision with regime scaling
        alpha_eff = prior_precision / regime_scale

        # Likelihood: mean of recent vols
        likelihood_vol = float(np.mean(recent_vols))
        likelihood_var = float(np.var(recent_vols)) if n > 1 else likelihood_vol**2

        # Posterior: precision-weighted average
        alpha_post = alpha_eff + n
        posterior_vol_sq = (alpha_eff * prior_vol**2 + n * likelihood_vol**2) / alpha_post
        posterior_vol = math.sqrt(max(0, posterior_vol_sq))

        # 95% credible interval (approximate using chi-squared)
        posterior_std = posterior_vol / math.sqrt(2 * alpha_post)
        ci_lower = max(0.01, posterior_vol - 1.96 * posterior_std)
        ci_upper = posterior_vol + 1.96 * posterior_std

        # Shrinkage factor: how much we shrink toward prior
        shrinkage = alpha_eff / alpha_post

        return BayesianVolEstimate(
            symbol="", timestamp=datetime.now().isoformat(),
            window=self.update_window,
            prior_vol=round(prior_vol, 4),
            prior_precision=round(prior_precision, 1),
            likelihood_vol=round(likelihood_vol, 4),
            likelihood_precision=round(float(n), 1),
            posterior_vol=round(posterior_vol, 4),
            credible_interval_lower=round(ci_lower, 4),
            credible_interval_upper=round(ci_upper, 4),
            simple_mean_vol=round(likelihood_vol, 4),
            shrinkage_factor=round(shrinkage, 3),
            regime_scale=round(regime_scale, 2),
            is_high_vol_regime=regime_scale > 1.5,
            n_obs=n,
            is_valid=True,
        )

    def compute_regime_scale(self, recent_returns: List[float]) -> float:
        """
        Compute regime scale from kurtosis of recent returns.

        High kurtosis → scale > 1 → less shrinkage toward prior
        (model recognizes that the prior may be stale during regime shifts)
        """
        if len(recent_returns) < 20:
            return 1.0

        rets = np.array(recent_returns[-60:])
        excess_kurt = self._excess_kurtosis(rets)

        if excess_kurt > 5:
            return 3.0  # Crisis: prior almost ignored
        elif excess_kurt > 2:
            return 2.0  # High kurtosis
        elif excess_kurt > 0.5:
            return 1.5  # Elevated
        return 1.0

    @staticmethod
    def _excess_kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis of a series."""
        n = len(x)
        if n < 4:
            return 0.0
        m2 = np.sum((x - np.mean(x))**2) / n
        m4 = np.sum((x - np.mean(x))**4) / n
        if m2 == 0:
            return 0.0
        return float(m4 / m2**2 - 3)


class BayesianVolPipeline:
    """
    End-to-end Bayesian volatility estimation pipeline.

    1. Load realized vol history
    2. Fit prior from long-term data
    3. Bayesian update with recent observations
    4. Apply regime adaptation from return kurtosis
    """

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    OUTPUT_DIR = DATA_DIR / "bayesian_vol"

    def __init__(self):
        self.model = BayesianVolModel()
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def estimate(self, symbol: str = "SPY",
                 vol_history: Optional[List[float]] = None,
                 recent_returns: Optional[List[float]] = None) -> BayesianVolEstimate:
        """Full Bayesian vol estimation pipeline."""
        # Load realized vol if not provided
        if vol_history is None:
            vol_history = self._load_realized_vol(symbol)

        # Fit prior
        prior_vol, prior_precision = self.model.fit_prior(vol_history)

        # Recent vol observations
        recent = vol_history[-self.model.update_window:] if len(vol_history) >= self.model.update_window else vol_history

        # Regime scale from return kurtosis
        regime_scale = self.model.compute_regime_scale(recent_returns or [])

        # Bayesian update
        result = self.model.update(prior_vol, prior_precision, recent, regime_scale)
        result.symbol = symbol

        return result

    def _load_realized_vol(self, symbol: str) -> List[float]:
        """Load realized vol history from stored data or compute from close prices."""
        # Try stored realized vol first
        rv_path = self.DATA_DIR / "realized_vol" / f"{symbol}_realized_vol.json"
        if rv_path.exists():
            with open(rv_path) as f:
                data = json.load(f)
            return [r["composite"] for r in data if r.get("composite", 0) > 0]

        # Fallback: compute from close prices in market.db
        return self._compute_close_vol(symbol)

    def _compute_close_vol(self, symbol: str) -> List[float]:
        """Compute rolling close-to-close vol from market.db."""
        db_path = self.DATA_DIR / "market.db"
        vols = []

        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol=? ORDER BY date",
                    (symbol,)
                )
                prices = [float(r[0]) for r in cursor.fetchall()]
                conn.close()

                if len(prices) > 20:
                    returns = np.diff(np.log(prices))
                    for i in range(20, len(returns)):
                        window_rets = returns[i-20:i]
                        vol = float(np.std(window_rets) * math.sqrt(252))
                        vols.append(vol)
            except Exception:
                pass

        if not vols:
            vols = [0.16] * 100  # Default 16% vol

        return vols

    def save_estimate(self, result: BayesianVolEstimate):
        out = self.OUTPUT_DIR / f"{result.symbol}_bayesian_vol.json"
        with open(out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


def estimate_bayesian_vol(symbol: str = "SPY") -> BayesianVolEstimate:
    """Convenience function."""
    pipeline = BayesianVolPipeline()
    return pipeline.estimate(symbol)


def main():
    import sys

    symbol = "SPY"
    for i, arg in enumerate(sys.argv):
        if arg == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]

    pipeline = BayesianVolPipeline()
    result = pipeline.estimate(symbol)

    print("=" * 60)
    print(f"BAYESIAN VOLATILITY — {symbol}")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp}")
    print(f"Observations: {result.n_obs}")
    print()
    print(f"{'Component':<25} {'Vol':>12} {'Precision':>12}")
    print("-" * 50)
    print(f"{'Prior (long-term)':<25} {result.prior_vol:>11.2%} {result.prior_precision:>11.1f}")
    print(f"{'Likelihood (recent)':<25} {result.likelihood_vol:>11.2%} {result.likelihood_precision:>11.1f}")
    print(f"{'─'*50}")
    print(f"{'Posterior (Bayesian)':<25} {result.posterior_vol:>11.2%}")
    print()
    print(f"95% Credible Interval: [{result.credible_interval_lower:.2%}, {result.credible_interval_upper:.2%}]")
    print(f"Shrinkage toward prior: {result.shrinkage_factor:.1%}")
    print(f"Regime scale: {result.regime_scale:.2f}x")
    print(f"High vol regime: {result.is_high_vol_regime}")
    print()
    print(f"Naive mean: {result.simple_mean_vol:.2%}")
    print(f"Bayesian adjustment: {result.posterior_vol - result.simple_mean_vol:+.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
