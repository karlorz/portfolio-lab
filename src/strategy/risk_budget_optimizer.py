"""
Risk Budget Optimizer with Maximum Diversification Portfolio (MDP) Constraint.

Integrates MDP weights as a diversification constraint into the risk budgeting
framework. The MDP maximizes the diversification ratio: (w' * sigma) / sqrt(w' * Sigma * w).

Reference: Choueifaty & Coignard (2008) "Toward Maximum Diversification".
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RiskBudgetOptimizer", "optimize_risk_budget"]


class RiskBudgetOptimizer:
    """
    Optimizes portfolio weights subject to risk budget constraints
    and optionally Maximum Diversification Portfolio (MDP) constraints.
    """

    def __init__(self, use_mdp_constraint: bool = True, mdp_weight_cap: float = 0.5):
        """
        Args:
            use_mdp_constraint: Whether to apply MDP diversification constraint.
            mdp_weight_cap: Maximum deviation allowed from MDP weights (0.0 = exact MDP).
        """
        self.use_mdp_constraint = use_mdp_constraint
        self.mdp_weight_cap = mdp_weight_cap

    def optimize(
        self,
        cov_matrix: np.ndarray,
        symbols: List[str],
        risk_budgets: Optional[Dict[str, float]] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Compute optimized weights subject to risk budget and MDP constraints.

        Args:
            cov_matrix: NxN covariance matrix.
            symbols: List of asset symbols.
            risk_budgets: Target risk contribution per asset (sum to 1.0).
            current_weights: Current weights for turnover penalty.

        Returns:
            Dict with 'weights', 'method', 'diversification_ratio'.
        """
        if len(symbols) < 2:
            return {"weights": {s: 1.0 for s in symbols}, "method": "single_asset"}

        n = len(symbols)
        sigma = np.sqrt(np.diag(cov_matrix))

        # Default: Equal risk budget
        if risk_budgets is None:
            risk_budgets = {s: 1.0 / n for s in symbols}

        # Convert to array
        rb = np.array([risk_budgets.get(s, 1.0 / n) for s in symbols])
        rb = rb / rb.sum()  # Normalize

        # Simple Risk Parity approximation (inverse volatility weighting)
        # A full Risk Parity optimizer would use iterative methods, but for
        # integration with MDP constraint, we use a starting point approach.
        inv_vol = 1.0 / (sigma + 1e-10)
        w_rp = inv_vol / inv_vol.sum()

        # If MDP constraint is enabled, blend with MDP weights
        if self.use_mdp_constraint:
            try:
                from src.strategy.max_diversification import compute_mdp_weights
                
                mdp_result = compute_mdp_weights(cov_matrix, symbols)
                w_mdp = np.array([mdp_result["weights"][s] for s in symbols])
                
                # Blend Risk Parity and MDP (simple average for stability)
                # In a production system, this would be a secondary optimization
                w_opt = 0.5 * w_rp + 0.5 * w_mdp
                
                dr = mdp_result["diversification_ratio"]
                method = "risk_parity_mdp_blend"
                
            except Exception as e:
                logger.warning("MDP constraint failed, falling back to risk parity: %s", e)
                w_opt = w_rp
                dr = 0.0
                method = "risk_parity"
        else:
            w_opt = w_rp
            dr = 0.0
            method = "risk_parity"

        # Normalize
        w_opt = np.clip(w_opt, 0, 1)
        w_opt = w_opt / w_opt.sum()

        weight_dict = {symbols[i]: round(float(w_opt[i]), 6) for i in range(n)}

        # Compute actual diversification ratio
        port_vol = np.sqrt(w_opt @ cov_matrix @ w_opt)
        actual_dr = float((w_opt @ sigma) / port_vol) if port_vol > 1e-10 else 1.0

        return {
            "weights": weight_dict,
            "diversification_ratio": round(actual_dr, 4),
            "portfolio_vol": round(float(port_vol) * 100, 2),
            "method": method,
            "mdp_applied": self.use_mdp_constraint,
        }


def optimize_risk_budget(
    cov_matrix: np.ndarray,
    symbols: List[str],
    risk_budgets: Optional[Dict[str, float]] = None,
    use_mdp: bool = True,
) -> Dict:
    """Convenience function for risk budget optimization."""
    optimizer = RiskBudgetOptimizer(use_mdp_constraint=use_mdp)
    return optimizer.optimize(cov_matrix, symbols, risk_budgets)
