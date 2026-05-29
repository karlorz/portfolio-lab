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

from src.paths import RISK_FREE_RATE


__all__ = ['BLViews', 'BLResult', 'map_biases_to_views', 'run_black_litterman', 'compute_bl_weights', 'tau_sensitivity']

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


# ── HRP Fallback ────────────────────────────────────────────────────────

def _run_hrp_fallback(
    cov_matrix: np.ndarray,
    symbols: List[str],
    returns: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Run Hierarchical Risk Parity as fallback when BL EF optimization fails.

    HRP clusters assets by correlation structure and allocates inversely
    proportional to cluster variance. No expected-return input needed —
    purely covariance-driven, making it robust when BL views produce
    degenerate returns.

    Args:
        cov_matrix: NxN covariance matrix.
        symbols: Asset symbol labels.
        returns: Optional returns DataFrame for HRP clustering. If None,
            synthetic returns are generated from the covariance matrix
            with a deterministic seed.

    Returns:
        Dict of {symbol: weight}. Empty dict on failure.
    """
    try:
        from pypfopt import HRPOpt
        import pandas as pd

        if returns is None:
            # Generate deterministic synthetic returns from covariance
            cov_df = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
            # Seed from covariance hash for reproducibility
            seed = int(abs(np.sum(cov_matrix)) * 1e6) % (2**31)
            rng = np.random.RandomState(seed)
            returns = pd.DataFrame(
                rng.multivariate_normal(np.zeros(len(symbols)), cov_df, size=252),
                columns=symbols,
            )

        hrp = HRPOpt(returns)
        hrp.optimize()
        cleaned = hrp.clean_weights()
        weights = {k: round(v, 4) for k, v in cleaned.items() if v > 0.001}
        return weights
    except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError, OverflowError) as e:
        logger.warning("HRP fallback failed: %s", e)
        return {}


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


def _compute_turnover_bps(
    new_weights: Dict[str, float],
    current_weights: Optional[Dict[str, float]],
    symbols: List[str],
) -> float:
    """Compute one-way turnover in basis points."""
    if current_weights is None:
        return 0.0
    total_turnover = 0.0
    for s in symbols:
        curr = current_weights.get(s, 0.0)
        new = new_weights.get(s, 0.0)
        total_turnover += abs(new - curr)
    return round(total_turnover / 2 * 10000, 1)  # One-way, in bps


def run_black_litterman(
    cov_matrix: np.ndarray,
    views: BLViews,
    risk_free_rate: float = RISK_FREE_RATE / 100,
    market_caps: Optional[Dict[str, float]] = None,
    pi: Optional[np.ndarray] = None,
    transaction_costs: bool = True,
    regime: Optional[str] = None,
    turnover_penalty: float = 0.0,
    current_weights: Optional[Dict[str, float]] = None,
) -> BLResult:
    """Run Black-Litterman optimization with PyPortfolioOpt.

    Takes a covariance matrix and BL views, computes posterior returns,
    and optimizes weights via EfficientFrontier.max_sharpe().

    Args:
        cov_matrix: NxN covariance matrix of asset returns.
        views: BLViews from map_biases_to_views().
        risk_free_rate: Annual risk-free rate (default centralized).
        market_caps: Market cap dict (required if prior="market").
        pi: Custom prior returns array (overrides views.prior).
        turnover_penalty: Lambda penalty for turnover (0=off, 0.5=moderate, 2+=heavy).
        current_weights: Current portfolio weights for turnover computation.

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

    # Transaction cost adjustment: subtract annualized round-trip costs
    # from posterior returns to penalize high-cost assets.
    cost_penalties_bps = {}
    if transaction_costs:
        from src.costs.etf_cost_table import estimate_cost_bps
        if isinstance(posterior_rets, pd.Series):
            rets_series = posterior_rets
        else:
            rets_series = pd.Series(posterior_rets, index=symbols)
        for sym in symbols:
            one_way_bps = estimate_cost_bps(sym, regime=regime)
            # Annualize: assume monthly rebalance (12x per year)
            annual_cost = one_way_bps * 2 * 12 / 1e4  # round-trip × 12 months
            rets_series[sym] -= annual_cost
            cost_penalties_bps[sym] = round(one_way_bps * 2, 1)
        posterior_rets = rets_series

    # Posterior covariance
    posterior_cov = bl.bl_cov()

    # Optimize via EfficientFrontier with cascade fallback:
    # BL max_sharpe → HRP → Equal Weight
    optimization_method = "bl_max_sharpe"
    turnover_applied = False
    turnover_lambda = turnover_penalty
    ef = EfficientFrontier(posterior_rets, posterior_cov)

    # Turnover penalty: quadratic penalty on weight changes from current
    if turnover_penalty > 0 and current_weights is not None:
        import cvxpy as cp
        curr_w = np.array([current_weights.get(s, 0.0) for s in symbols])

        def _turnover_penalty(w):
            return turnover_penalty * cp.sum(cp.square(w - curr_w))

        ef.add_objective(_turnover_penalty)
        turnover_applied = True
        logger.info("BL turnover penalty applied: lambda=%.2f", turnover_penalty)

    try:
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as e:
        logger.warning("BL EfficientFrontier.max_sharpe failed: %s", e)

        # Fallback 1: HRP — covariance-driven, no return estimates needed
        hrp_weights = _run_hrp_fallback(cov_matrix, symbols)
        if hrp_weights:
            cleaned = hrp_weights
            perf = (None, None, None)
            optimization_method = "bl_hrp"
            logger.info("BL cascade: using HRP fallback (%d assets)", len(hrp_weights))
        else:
            # Fallback 2: Equal weight — last resort
            n = len(symbols)
            cleaned = {s: round(1.0 / n, 4) for s in symbols}
            perf = (None, None, None)
            optimization_method = "bl_equal_weight"
            logger.warning("BL cascade: HRP failed, using equal-weight fallback")

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
        extras={
            "optimization_method": optimization_method,
            "transaction_costs_applied": transaction_costs,
            "cost_penalties_bps": cost_penalties_bps,
            "turnover_penalty_applied": turnover_applied,
            "turnover_penalty_lambda": turnover_lambda,
            "turnover_bps": _compute_turnover_bps(weights_dict, current_weights, symbols) if turnover_applied else 0,
        },
    )


def compute_bl_weights(
    prices_df=None,
    equity_bias: float = 0.0,
    duration_bias: float = 0.0,
    gold_bias: float = 0.0,
    health_scores: Optional[Dict[str, float]] = None,
    tau: float = DEFAULT_TAU,
    prior: str = "equal",
    risk_free_rate: float = RISK_FREE_RATE / 100,
    transaction_costs: bool = True,
    regime: Optional[str] = None,
    turnover_penalty: float = 0.0,
    current_weights: Optional[Dict[str, float]] = None,
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
        from src.data.price_cache import get_prices_df
        prices_df = get_prices_df(symbols=DEFAULT_SYMBOLS)

    # Filter to default symbols if more are present
    available = [s for s in DEFAULT_SYMBOLS if s in prices_df.columns]
    if len(available) < len(DEFAULT_SYMBOLS):
        missing = set(DEFAULT_SYMBOLS) - set(available)
        logger.warning("Missing symbols for BL: %s", missing)

    prices_subset = prices_df[available]

    # Compute covariance matrix with Ledoit-Wolf shrinkage for stability
    S = risk_models.CovarianceShrinkage(prices_subset).ledoit_wolf()
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
        transaction_costs=transaction_costs,
        regime=regime,
        turnover_penalty=turnover_penalty,
        current_weights=current_weights,
    )


def tau_sensitivity(
    cov_matrix: np.ndarray,
    views: BLViews,
    tau_values: Optional[List[float]] = None,
    risk_free_rate: float = RISK_FREE_RATE / 100,
    market_caps: Optional[Dict[str, float]] = None,
    pi: Optional[np.ndarray] = None,
) -> Dict[float, BLResult]:
    """Run BL optimization across multiple tau values.

    Validates the tau=0.15 default by showing how weights change
    as view confidence varies. At low tau, weights approach the prior.
    At high tau, weights approach pure view-based allocation.

    Args:
        cov_matrix: NxN covariance matrix.
        views: BLViews from map_biases_to_views().
        tau_values: List of tau values to test.
            Default: [0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50].
        risk_free_rate: Annual risk-free rate.
        market_caps: Market caps (for market prior).
        pi: Custom prior returns.

    Returns:
        Dict mapping tau value to BLResult.
    """
    if tau_values is None:
        tau_values = [0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    results = {}
    for tau in tau_values:
        # Create a copy of views with modified tau
        view_copy = BLViews(
            absolute_views=dict(views.absolute_views),
            view_confidences=list(views.view_confidences),
            tau=tau,
            prior=views.prior,
            symbols=list(views.symbols),
        )
        try:
            result = run_black_litterman(
                cov_matrix=cov_matrix,
                views=view_copy,
                risk_free_rate=risk_free_rate,
                market_caps=market_caps,
                pi=pi,
            )
            results[tau] = result
        except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as e:
            logger.warning("BL optimization failed at tau=%.3f: %s", tau, e)

    return results
