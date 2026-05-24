"""
Black-Litterman view mapper for ensemble signal integration.

Maps ensemble voter outputs (equity_bias, duration_bias, gold_bias) to
Black-Litterman absolute views, and health_scores to view confidences
(Idzorek method). Runs BL optimization via PyPortfolioOpt to produce
posterior returns and BL-optimized weights.

Architecture: BL sits between grid search (Layer 1: discovery) and
tactical overlays (Layer 3: regime adjustment). It refines baseline
weights using covariance-aware Bayesian updating.

Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014.
PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

# Default tau (view weight vs prior). PyPortfolioOpt defaults to 0.05,
# but for systematically backtested signals, tau=0.15 is more appropriate.
DEFAULT_TAU: float = 0.15

# Default BL symbols aligned with the 3-asset champion portfolio
DEFAULT_SYMBOLS: List[str] = ["SPY", "GLD", "TLT"]

# Market caps (approximate, 2026) — used for market-implied prior
DEFAULT_MARKET_CAPS: Dict[str, float] = {
    "SPY": 400e9,
    "GLD": 60e9,
    "TLT": 35e9,
}

# Bias-to-view scaling: ensemble biases are [-1, +1], BL views are
# expected annual returns. Scale factor converts bias magnitude to
# a reasonable annual return expectation.
BIAS_TO_RETURN_SCALE: float = 0.10  # 10% max shift per unit bias

# Minimum view confidence (avoids zero-confidence degeneracy)
MIN_VIEW_CONFIDENCE: float = 0.10


# ── Dataclasses ────────────────────────────────────────────────────────

@dataclass
class BLViews:
    """Black-Litterman view specification.

    Maps ensemble signal biases to BL absolute views with confidence
    derived from signal health scores.
    """

    absolute_views: Dict[str, float]  # {symbol: expected_return}
    view_confidences: List[float]     # [0-1] per symbol, for Idzorek omega
    tau: float = DEFAULT_TAU
    prior: str = "equal"              # "equal", "market", or custom array
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))


@dataclass
class BLResult:
    """Result from Black-Litterman optimization.

    Contains posterior returns, BL-optimized weights, and diagnostics.
    """

    posterior_returns: Dict[str, float]  # {symbol: posterior_expected_return}
    bl_weights: Dict[str, float]        # {symbol: weight} from EF.max_sharpe
    tau: float
    prior_type: str
    view_confidences: List[float]

    # Performance metrics (from EfficientFrontier)
    expected_sharpe: Optional[float] = None
    expected_cagr: Optional[float] = None
    expected_volatility: Optional[float] = None

    extras: Dict[str, Any] = field(default_factory=dict)


# ── View Mapping ───────────────────────────────────────────────────────

def map_biases_to_views(
    equity_bias: float,
    duration_bias: float,
    gold_bias: float,
    health_scores: Optional[Dict[str, float]] = None,
    tau: float = DEFAULT_TAU,
    prior: str = "equal",
) -> BLViews:
    """Map ensemble signal biases to Black-Litterman views.

    The ensemble voter computes equity_bias (SPY direction), duration_bias
    (TLT direction), and gold_bias (GLD direction) as values in [-1, +1].
    These are scaled to annual return expectations for BL's absolute_views.

    Args:
        equity_bias: SPY directional bias from ensemble (-1 to +1).
        duration_bias: TLT directional bias from ensemble (-1 to +1).
        gold_bias: GLD directional bias from ensemble (-1 to +1).
        health_scores: Per-signal health scores. If provided, the average
            health of signals contributing to each asset's view is used as
            view confidence. If None, defaults to 0.50 (moderate).
        tau: BL tau parameter (view weight). Default 0.15.
        prior: Prior type — "equal" or "market".

    Returns:
        BLViews with absolute_views, view_confidences, and tau.
    """
    # Map biases to expected returns
    absolute_views = {
        "SPY": equity_bias * BIAS_TO_RETURN_SCALE,
        "TLT": duration_bias * BIAS_TO_RETURN_SCALE,
        "GLD": gold_bias * BIAS_TO_RETURN_SCALE,
    }

    # Map health scores to view confidences
    if health_scores is not None:
        # Average health of signals contributing to each asset
        # In practice, all signals contribute to all assets through
        # the ensemble weighting, so use the overall average health
        # scaled by per-asset bias magnitude (stronger views = more confident)
        avg_health = np.mean(list(health_scores.values())) if health_scores else 0.50

        # Per-asset confidence: base from avg_health, modulated by bias magnitude
        # Stronger consensus (larger |bias|) → higher confidence
        confidences = []
        for sym, bias in [("SPY", equity_bias), ("TLT", duration_bias), ("GLD", gold_bias)]:
            bias_magnitude = min(abs(bias), 1.0)
            # Confidence = avg_health * (0.5 + 0.5 * bias_magnitude)
            # This means: even with moderate health, strong bias gives reasonable confidence
            conf = avg_health * (0.5 + 0.5 * bias_magnitude)
            conf = max(conf, MIN_VIEW_CONFIDENCE)
            conf = min(conf, 1.0)
            confidences.append(round(conf, 4))
    else:
        # Default: moderate confidence
        confidences = [0.50, 0.50, 0.50]

    return BLViews(
        absolute_views=absolute_views,
        view_confidences=confidences,
        tau=tau,
        prior=prior,
    )


def run_black_litterman(
    cov_matrix: np.ndarray,
    views: BLViews,
    risk_free_rate: float = 0.02,
    market_caps: Optional[Dict[str, float]] = None,
    pi: Optional[np.ndarray] = None,
) -> BLResult:
    """Run Black-Litterman optimization with PyPortfolioOpt.

    Takes a covariance matrix and BL views, computes posterior returns,
    and optimizes weights via EfficientFrontier.max_sharpe().

    Args:
        cov_matrix: NxN covariance matrix of asset returns.
        views: BLViews from map_biases_to_views().
        risk_free_rate: Annual risk-free rate (default 0.02).
        market_caps: Market cap dict (required if prior="market").
        pi: Custom prior returns array (overrides views.prior).

    Returns:
        BLResult with posterior returns, optimized weights, and metrics.
    """
    from pypfopt import BlackLittermanModel, EfficientFrontier

    symbols = views.symbols
    n_assets = len(symbols)

    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"{n_assets} symbols: {symbols}"
        )

    # Wrap cov_matrix as DataFrame with symbol labels — PyPortfolioOpt
    # requires tickers to match absolute_views keys.
    cov_df = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)

    # Determine prior
    if pi is not None:
        prior_type = "custom"
        prior_returns = pd.Series(pi, index=symbols)
    elif views.prior == "market":
        if market_caps is None:
            market_caps = DEFAULT_MARKET_CAPS
        from pypfopt.black_litterman import market_implied_prior_returns
        mcaps = pd.Series({s: market_caps.get(s, 1e9) for s in symbols})
        delta = 2.5  # Market risk aversion (standard estimate)
        prior_returns = market_implied_prior_returns(mcaps, delta, cov_df)
        prior_type = "market"
    else:
        # Equal-weight prior: each asset has equal expected return
        # This is appropriate for a constrained 3-asset portfolio
        # where market-cap weights would overweight SPY excessively
        prior_returns = pd.Series(np.ones(n_assets) / n_assets, index=symbols)
        prior_type = "equal"

    # Build BL model
    bl = BlackLittermanModel(
        cov_matrix=cov_df,
        pi=prior_returns,
        absolute_views=views.absolute_views,
        omega="idzorek",
        view_confidences=views.view_confidences,
        tau=views.tau,
        risk_aversion=1.0,
    )

    # Posterior returns
    posterior_rets = bl.bl_returns()

    # Posterior covariance
    posterior_cov = bl.bl_cov()

    # Optimize via EfficientFrontier
    ef = EfficientFrontier(posterior_rets, posterior_cov)
    try:
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    except Exception as e:
        logger.warning("BL EfficientFrontier.max_sharpe failed: %s", e)
        # Fallback: BL weights from posterior (no EF optimization)
        raw_weights = bl.bl_weights()
        cleaned = {s: round(w, 4) for s, w in zip(symbols, raw_weights)}
        perf = (None, None, None)

    # Build result
    if isinstance(posterior_rets, pd.Series):
        posterior_dict = {s: round(float(posterior_rets[s]), 6) for s in symbols}
    else:
        posterior_dict = {s: round(float(posterior_rets[i]), 6) for i, s in enumerate(symbols)}
    weights_dict = {k: v for k, v in cleaned.items() if v > 0.001}

    return BLResult(
        posterior_returns=posterior_dict,
        bl_weights=weights_dict,
        tau=views.tau,
        prior_type=prior_type,
        view_confidences=views.view_confidences,
        expected_sharpe=round(perf[2], 4) if perf[2] is not None else None,
        expected_cagr=round(perf[0] * 100, 2) if perf[0] is not None else None,
        expected_volatility=round(perf[1] * 100, 2) if perf[1] is not None else None,
    )


def compute_bl_weights(
    prices_df=None,
    equity_bias: float = 0.0,
    duration_bias: float = 0.0,
    gold_bias: float = 0.0,
    health_scores: Optional[Dict[str, float]] = None,
    tau: float = DEFAULT_TAU,
    prior: str = "equal",
    risk_free_rate: float = 0.02,
) -> BLResult:
    """Convenience function: prices → BL-optimized weights in one call.

    Computes covariance from price DataFrame, maps biases to views,
    and runs BL optimization.

    Args:
        prices_df: DataFrame with columns for each symbol, datetime index.
            If None, loads from PRICES_JSON.
        equity_bias: SPY directional bias (-1 to +1).
        duration_bias: TLT directional bias (-1 to +1).
        gold_bias: GLD directional bias (-1 to +1).
        health_scores: Per-signal health scores for view confidence.
        tau: BL tau parameter.
        prior: Prior type ("equal" or "market").
        risk_free_rate: Annual risk-free rate.

    Returns:
        BLResult with optimized weights and metrics.
    """
    from pypfopt import risk_models

    if prices_df is None:
        from src.paths import PRICES_JSON
        from scripts.optimize_portfolio import load_prices
        prices_df = load_prices(PRICES_JSON, symbols=DEFAULT_SYMBOLS)

    # Filter to default symbols if more are present
    available = [s for s in DEFAULT_SYMBOLS if s in prices_df.columns]
    if len(available) < len(DEFAULT_SYMBOLS):
        missing = set(DEFAULT_SYMBOLS) - set(available)
        logger.warning("Missing symbols for BL: %s", missing)

    prices_subset = prices_df[available]

    # Compute covariance matrix
    S = risk_models.sample_cov(prices_subset)
    cov_matrix = S.values

    # Map biases to views
    views = map_biases_to_views(
        equity_bias=equity_bias,
        duration_bias=duration_bias,
        gold_bias=gold_bias,
        health_scores=health_scores,
        tau=tau,
        prior=prior,
    )
    # Adjust views to only include available symbols
    views.symbols = available
    views.absolute_views = {k: v for k, v in views.absolute_views.items() if k in available}
    # Adjust confidences to match available symbols
    sym_to_idx = {s: i for i, s in enumerate(DEFAULT_SYMBOLS)}
    views.view_confidences = [
        views.view_confidences[sym_to_idx[s]]
        for s in available if s in sym_to_idx
    ]

    return run_black_litterman(
        cov_matrix=cov_matrix,
        views=views,
        risk_free_rate=risk_free_rate,
    )
