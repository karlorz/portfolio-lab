"""Maximum Diversification Portfolio (MDP).

Maximizes the diversification ratio: weighted average volatility / portfolio
volatility. From Choueifaty & Coignard (2008) "Toward Maximum Diversification".

The MDP maximizes: DR(w) = (w' * sigma) / sqrt(w' * Sigma * w)
subject to sum(w) = 1, w >= 0.

Uses scipy.optimize for the nonconvex ratio maximization.
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

__all__ = ["compute_mdp_weights"]


def compute_mdp_weights(
    cov_matrix: np.ndarray,
    symbols: List[str],
) -> Dict:
    """Compute Maximum Diversification Portfolio weights.

    Args:
        cov_matrix: NxN covariance matrix of asset returns.
        symbols: List of asset symbols (length N).

    Returns:
        Dict with:
            - weights: Dict[symbol, float] — optimized portfolio weights
            - diversification_ratio: float — the DR value
            - portfolio_vol: float — annualized portfolio volatility
            - individual_vols: Dict[symbol, float] — individual asset volatilities
            - method: "max_diversification"

    Raises:
        ValueError: If cov_matrix shape doesn't match symbols, or < 2 assets.
    """
    n = len(symbols)
    if n < 2:
        raise ValueError(f"Need at least 2 assets, got {n}")
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"{n} symbols"
        )

    # Individual asset volatilities
    sigma = np.sqrt(np.diag(cov_matrix))

    # Optimize: minimize negative diversification ratio
    def neg_dr(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(w @ sigma) / port_vol

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # Start from equal weight
    w0 = np.ones(n) / n

    result = minimize(
        neg_dr,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning("MDP optimization did not converge: %s", result.message)

    weights = np.clip(result.x, 0, 1)
    weights = weights / weights.sum()  # Re-normalize

    # Compute final metrics
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    dr = float((weights @ sigma) / port_vol) if port_vol > 1e-10 else 1.0

    weight_dict = {symbols[i]: round(float(weights[i]), 6) for i in range(n)}
    vol_dict = {symbols[i]: round(float(sigma[i]) * 100, 2) for i in range(n)}

    return {
        "weights": weight_dict,
        "diversification_ratio": round(dr, 4),
        "portfolio_vol": round(float(port_vol) * 100, 2),
        "individual_vols": vol_dict,
        "method": "max_diversification",
    }
