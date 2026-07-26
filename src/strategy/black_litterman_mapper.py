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


__all__ = [
    'BLViews', 'BLResult', 'map_biases_to_views', 'run_black_litterman',
    'compute_bl_weights', 'compute_regime_covariances', 'tau_sensitivity',
]

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

# Default tau (view weight vs prior). PyPortfolioOpt defaults to 0.05,
# but for systematically backtested signals, tau=0.15 is more appropriate.
DEFAULT_TAU: float = 0.15

# Default BL symbols aligned with the 3-asset champion portfolio
DEFAULT_SYMBOLS: List[str] = ["SPY", "GLD", "TLT"]

# Champion baseline prior (advisory reference SPY/GLD/TLT 46/38/16)
CHAMPION_PRIOR_WEIGHTS: Dict[str, float] = {
    "SPY": 0.46,
    "GLD": 0.38,
    "TLT": 0.16,
}

# Floor so max_sharpe + clean_weights cannot zero a champion sleeve under
# mild views (corner solutions BL exists to avoid). Relative to champion prior.
CHAMPION_MIN_WEIGHT_SCALE: float = 0.25  # e.g. GLD floor = 0.38 * 0.25 = 0.095
CHAMPION_ABSOLUTE_MIN_WEIGHT: float = 0.05  # never below 5% when in book

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

# ── Regime-Conditional Covariance ──────────────────────────────────────

# Rolling window (trading days) for realized vol regime classification.
REGIME_COV_WINDOW: int = 21

# Minimum observations per regime to compute a regime-specific covariance.
# Falls back to full-sample if any regime has fewer samples.
REGIME_COV_MIN_SAMPLES: int = 60

# Annualized vol thresholds for regime classification from rolling realized vol.
# Aligned with VIX regime thresholds (VIX ~ annualized vol × 100).
REGIME_VOL_THRESHOLDS: Dict[str, float] = {
    "crisis": 0.30,    # annualized vol > 30%
    "high_vol": 0.20,  # annualized vol > 20%
    "normal": 0.12,    # annualized vol > 12%
    "low_vol": 0.0,    # catch-all below 12%
}


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


# ── Regime-Conditional Covariance ──────────────────────────────────────

def _classify_regime_from_vol(annualized_vol: float) -> str:
    """Classify regime from annualized realized volatility.

    Args:
        annualized_vol: Annualized volatility (e.g., 0.20 = 20%).

    Returns:
        Regime name: "crisis", "high_vol", "normal", or "low_vol".
    """
    if annualized_vol >= REGIME_VOL_THRESHOLDS["crisis"]:
        return "crisis"
    if annualized_vol >= REGIME_VOL_THRESHOLDS["high_vol"]:
        return "high_vol"
    if annualized_vol >= REGIME_VOL_THRESHOLDS["normal"]:
        return "normal"
    return "low_vol"


def compute_regime_covariances(
    prices_df: pd.DataFrame,
    symbols: Optional[List[str]] = None,
    window: int = REGIME_COV_WINDOW,
    min_samples: int = REGIME_COV_MIN_SAMPLES,
) -> Dict[str, np.ndarray]:
    """Compute per-regime covariance matrices from historical price data.

    Segments historical returns by realized volatility regime, then computes
    Ledoit-Wolf shrinkage covariance for each regime. This allows the BL
    posterior to use regime-appropriate risk estimates instead of a single
    full-sample covariance.

    Args:
        prices_df: DataFrame with datetime index and symbol columns.
        symbols: Asset symbols. If None, uses all columns in prices_df.
        window: Rolling window in trading days for realized vol computation.
        min_samples: Minimum return observations per regime to compute a
            regime-specific covariance. Regimes with fewer samples are merged
            into "normal" or skipped.

    Returns:
        Dict mapping regime name ("crisis", "high_vol", "normal", "low_vol")
        to an NxN covariance matrix. At minimum, "normal" is always present.
    """
    from pypfopt import risk_models

    if symbols is None:
        symbols = list(prices_df.columns)

    available = [s for s in symbols if s in prices_df.columns]
    if len(available) < 2:
        logger.warning("Need at least 2 symbols for covariance, got %d", len(available))
        return {"normal": np.eye(len(symbols)) * 0.0001}

    prices = prices_df[available].dropna()
    returns = prices.pct_change().dropna()

    if len(returns) < window + 10:
        logger.warning("Insufficient history (%d days) for regime covariance, using full-sample", len(returns))
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf().values
        return {"normal": cov}

    # Compute rolling realized vol (annualized)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    # Use mean vol across assets as regime classifier
    mean_vol = rolling_vol.mean(axis=1).dropna()

    # Classify each window into a regime
    regimes = mean_vol.apply(_classify_regime_from_vol)

    # Segment returns by regime (drop first `window` NaN rows)
    valid_returns = returns.iloc[window:]
    aligned_regimes = regimes.reindex(valid_returns.index)

    regime_returns: Dict[str, pd.DataFrame] = {}
    for regime_name in REGIME_VOL_THRESHOLDS:
        mask = aligned_regimes == regime_name
        subset = valid_returns[mask]
        if len(subset) >= min_samples:
            regime_returns[regime_name] = subset
        elif len(subset) > 0:
            # Too few samples — merge into "normal"
            regime_returns.setdefault("normal", pd.DataFrame())
            if not subset.empty:
                regime_returns["normal"] = pd.concat([regime_returns["normal"], subset])

    # Compute covariance per regime
    result: Dict[str, np.ndarray] = {}
    for regime_name, rets in regime_returns.items():
        if len(rets) < min_samples:
            continue
        try:
            # Reconstruct prices from returns for CovarianceShrinkage
            cum_ret = (1 + rets).cumprod()
            cov_shrunk = risk_models.CovarianceShrinkage(cum_ret).ledoit_wolf()
            cov = cov_shrunk.values
            # Ensure correct dimensions match symbols
            if cov.shape[0] == len(available):
                result[regime_name] = cov
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("Covariance computation failed for regime %s: %s", regime_name, e)

    # Guarantee at least "normal" is present via full-sample fallback
    if "normal" not in result:
        full_cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf().values
        result["normal"] = full_cov

    logger.info("Regime covariances computed: %s (%d regimes)",
                list(result.keys()), len(result))
    return result


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

    # Champion min-weight floor: max_sharpe + clean_weights can zero GLD/TLT
    # under near-flat mild views. Re-introduce a prior-scaled floor for any
    # champion sleeve present in the symbol universe (not explicitly excluded).
    weights_full = {s: float(cleaned.get(s, 0.0) or 0.0) for s in symbols}
    zeroed_before = [s for s, w in weights_full.items() if w <= 0.001]
    floor_applied: list[str] = []
    for sym in symbols:
        if sym not in CHAMPION_PRIOR_WEIGHTS:
            continue
        prior_w = CHAMPION_PRIOR_WEIGHTS[sym]
        floor = max(
            CHAMPION_ABSOLUTE_MIN_WEIGHT,
            prior_w * CHAMPION_MIN_WEIGHT_SCALE,
        )
        if weights_full[sym] < floor - 1e-12:
            weights_full[sym] = floor
            floor_applied.append(sym)
    if floor_applied:
        total = sum(weights_full.values())
        if total > 0:
            weights_full = {k: v / total for k, v in weights_full.items()}
        # If still zero after renorm (shouldn't), fall back to HRP / prior blend
        still_zero = [s for s, w in weights_full.items() if w <= 0.001]
        if still_zero and optimization_method == "bl_max_sharpe":
            hrp_weights = _run_hrp_fallback(cov_matrix, symbols)
            if hrp_weights:
                weights_full = {s: float(hrp_weights.get(s, 0.0)) for s in symbols}
                optimization_method = "bl_hrp_after_zero_weight"
                logger.info(
                    "BL cascade: HRP after champion zero_weight under mild views (%s)",
                    still_zero,
                )
            else:
                # Blend toward champion prior for zeroed sleeves only
                for s in still_zero:
                    weights_full[s] = CHAMPION_PRIOR_WEIGHTS.get(s, 1.0 / len(symbols))
                total = sum(weights_full.values())
                weights_full = {k: v / total for k, v in weights_full.items()}
                optimization_method = "bl_champion_floor"
        logger.info(
            "BL champion min-weight floor applied to %s (pre-floor zeros=%s)",
            floor_applied,
            zeroed_before,
        )

    weights_dict = {
        k: round(v, 6) for k, v in weights_full.items() if v > 0.001
    }
    zero_weight_assets = [s for s in symbols if s not in weights_dict]

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
            "champion_min_weight_applied": floor_applied,
            "zero_weight_assets": zero_weight_assets,
            "champion_prior": dict(CHAMPION_PRIOR_WEIGHTS),
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

    # Compute covariance matrix — regime-specific if regime is provided,
    # full-sample with Ledoit-Wolf shrinkage otherwise.
    regime_cov_used: Optional[str] = None
    if regime is not None:
        regime_covs = compute_regime_covariances(prices_subset, symbols=available)
        regime_key = regime.lower().strip()
        if regime_key in regime_covs:
            cov_matrix = regime_covs[regime_key]
            regime_cov_used = regime_key
            logger.info("Using %s regime covariance for BL", regime_key)
        elif "normal" in regime_covs:
            # Unknown regime — fall back to normal regime covariance
            cov_matrix = regime_covs["normal"]
            regime_cov_used = "normal"
            logger.info("Unknown regime '%s', using normal covariance", regime)
        else:
            S = risk_models.CovarianceShrinkage(prices_subset).ledoit_wolf()
            cov_matrix = S.values
            regime_cov_used = "full_sample"
            logger.warning("Unknown regime '%s', using full-sample covariance", regime)
    else:
        S = risk_models.CovarianceShrinkage(prices_subset).ledoit_wolf()
        cov_matrix = S.values
        regime_cov_used = "full_sample"

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

    result = run_black_litterman(
        cov_matrix=cov_matrix,
        views=views,
        risk_free_rate=risk_free_rate,
        transaction_costs=transaction_costs,
        regime=regime,
        turnover_penalty=turnover_penalty,
        current_weights=current_weights,
    )

    # Record regime covariance usage in extras
    result.extras["regime_covariance"] = regime_cov_used

    return result


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
