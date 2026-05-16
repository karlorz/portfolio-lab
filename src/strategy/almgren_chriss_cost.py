#!/usr/bin/env python3
"""
Portfolio-Lab v6.07: Almgren-Chriss Cost Model for Cost-Aware Optimization

Provides per-symbol transaction cost estimation calibrated by TCA feedback.
Used by RegimeOptimizer's "cost_aware" mode to embed transaction costs
directly in the CVXPY optimization objective.

Cost Model:
    C(w) = Σ_i [ spread_i * |w_i - w₀_i| + impact_i * (w_i - w₀_i)² ]

Where:
    - spread_i: linear cost per % turnover (half-spread + fees)
    - impact_i: quadratic market impact coefficient
    - w₀_i: current/base weight
    - w_i: target weight

Calibration:
    - Baseline costs from empirical spreads
    - TCA feedback cost_calibration factor adjusts baseline per symbol
    - Higher cost_calibration (from poor execution quality) → higher penalty

References:
    - Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
    - Kissell & Glantz (2003) "Optimal Trading Strategies"
    - Portfolio-Lab v6.00 TCA Engine for empirical calibration

Usage:
    from src.strategy.almgren_chriss_cost import AlmgrenChrissCostModel

    model = AlmgrenChrissCostModel()
    spread_costs, impact_costs = model.get_cost_params(assets=["SPY", "GLD", "TLT"])
    total_cost_bps = model.estimate_turnover_cost({"SPY": 0.46}, {"SPY": 0.50})
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Default spread costs in bps per 1% turnover (half-spread + fees)
# Calibrated from typical ETF spreads
DEFAULT_SPREAD_COST: Dict[str, float] = {
    "SPY": 0.5,
    "GLD": 1.0,
    "TLT": 1.2,
    "IEF": 0.8,
    "SHY": 0.3,
    "BTC": 3.0,
    "ETH": 4.0,
    "QQQ": 0.7,
    "EFA": 2.5,
    "VXUS": 2.0,
    "MTUM": 1.5,
    "VLUE": 1.5,
    "USMV": 1.2,
}

# Default quadratic impact cost coefficients (bps per %² turnover)
# Reflects square-root impact model: cost ~ σ * sqrt(turnover)
DEFAULT_IMPACT_COST: Dict[str, float] = {
    "SPY": 0.3,
    "GLD": 0.6,
    "TLT": 0.8,
    "IEF": 0.5,
    "SHY": 0.2,
    "BTC": 4.0,
    "ETH": 5.0,
    "QQQ": 0.4,
    "EFA": 1.5,
    "VXUS": 1.5,
    "MTUM": 1.0,
    "VLUE": 1.0,
    "USMV": 0.8,
}

# Default cost calibration factor (applied when no TCA feedback available)
DEFAULT_COST_CALIBRATION = 1.0


@dataclass
class TurnoverCostEstimate:
    """Estimated turnover cost for a set of weight changes."""
    total_cost_bps: float
    spread_cost_bps: float
    impact_cost_bps: float
    symbol_costs: Dict[str, Dict[str, float]]
    calibration_source: str  # "tca_feedback" or "default"
    active_turnover_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostParameters:
    """Per-symbol cost parameters for optimization."""
    spread: Dict[str, float]
    impact: Dict[str, float]
    calibration_source: str
    cost_aversion_default: float = 0.01

    def to_dict(self) -> dict:
        return {
            "spread": self.spread,
            "impact": self.impact,
            "calibration_source": self.calibration_source,
            "cost_aversion_default": self.cost_aversion_default,
        }


class AlmgrenChrissCostModel:
    """
    Transaction cost estimation model using Almgren-Chriss framework.

    Provides:
    1. Per-symbol cost parameters for optimization (get_cost_params)
    2. Turnover cost estimates for backtesting (estimate_turnover_cost)
    3. TCA feedback calibration for realistic costs

    The model is designed to be lightweight (no ML deps) and usable
    directly within CVXPY objective functions.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        use_tca_calibration: bool = True,
        default_cost_aversion: float = 0.01,
    ):
        """
        Initialize cost model.

        Args:
            data_dir: Path to data directory (for TCA feedback state)
            use_tca_calibration: If True, calibrate costs using TCA feedback
            default_cost_aversion: Default γ for cost-aware optimization
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.use_tca_calibration = use_tca_calibration
        self.default_cost_aversion = default_cost_aversion

        # TCA feedback state (lazy loaded)
        self._tca_feedback: Optional[dict] = None

    # ------------------------------------------------------------------
    # TCA Feedback Loading
    # ------------------------------------------------------------------

    def _load_tca_feedback(self) -> Optional[dict]:
        """Load TCA feedback state for cost calibration."""
        if self._tca_feedback is not None:
            return self._tca_feedback

        tca_path = self.data_dir / "tca_feedback_state.json"
        if not tca_path.exists():
            return None

        try:
            with open(tca_path) as f:
                data = json.load(f)
            if data.get("status") not in ("no_data",) and data.get("symbols"):
                self._tca_feedback = data
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load TCA feedback: {e}")

        return None

    def _get_symbol_calibration(self, symbol: str) -> float:
        """
        Get cost calibration factor for a symbol from TCA feedback.

        Returns multiplier on baseline costs (1.0 = no adjustment).
        Higher value means actual costs exceed estimates.
        """
        feedback = self._load_tca_feedback()
        if feedback:
            symbols = feedback.get("symbols", {})
            sym_data = symbols.get(symbol, {})
            cal = sym_data.get("cost_calibration", DEFAULT_COST_CALIBRATION)
            return float(cal)
        return DEFAULT_COST_CALIBRATION

    # ------------------------------------------------------------------
    # Cost Parameter Generation
    # ------------------------------------------------------------------

    def get_cost_params(
        self, assets: List[str],
    ) -> CostParameters:
        """
        Get per-symbol cost parameters for optimization.

        Args:
            assets: List of asset symbols (e.g., ["SPY", "GLD", "TLT"])

        Returns:
            CostParameters with spread and impact costs, calibrated by TCA.

        The spread cost is the linear coefficient on |Δw|.
        The impact cost is the quadratic coefficient on (Δw)².
        """
        spread = {}
        impact = {}
        source = "default"

        feedback = self._load_tca_feedback()
        if feedback:
            source = "tca_feedback"

        for asset in assets:
            # Baseline costs
            base_spread = DEFAULT_SPREAD_COST.get(asset, 2.0)
            base_impact = DEFAULT_IMPACT_COST.get(asset, 1.0)

            # Calibrate with TCA feedback (if available and enabled)
            if self.use_tca_calibration and feedback:
                cal = self._get_symbol_calibration(asset)
                spread[asset] = base_spread * cal
                impact[asset] = base_impact * cal
            else:
                spread[asset] = base_spread
                impact[asset] = base_impact

        return CostParameters(
            spread=spread,
            impact=impact,
            calibration_source=source,
            cost_aversion_default=self.default_cost_aversion,
        )

    def get_tca_calibration_summary(self) -> dict:
        """
        Get a summary of TCA calibration factors for all symbols.

        Returns dict with symbol → calibration factor.
        """
        feedback = self._load_tca_feedback()
        if not feedback:
            return {"source": "default", "factors": {}}

        symbols = feedback.get("symbols", {})
        factors = {
            sym: {
                "cost_calibration": data.get("cost_calibration", 1.0),
                "avg_quality": data.get("avg_quality", 75.0),
                "quality_bucket": data.get("quality_bucket", "fair"),
            }
            for sym, data in symbols.items()
        }

        return {
            "source": "tca_feedback",
            "overall_quality": feedback.get("overall_quality", 75.0),
            "factors": factors,
        }

    # ------------------------------------------------------------------
    # Turnover Cost Estimation
    # ------------------------------------------------------------------

    def estimate_turnover_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> TurnoverCostEstimate:
        """
        Estimate total turnover cost in bps for rebalancing.

        Cost = Σ_i [ spread_i * |Δw_i| + impact_i * (Δw_i)² ]

        Where Δw_i = target_i - current_i, expressed in decimal weights.

        Args:
            current_weights: Current portfolio weights (e.g., {"SPY": 0.46})
            target_weights: Target portfolio weights (e.g., {"SPY": 0.50})

        Returns:
            TurnoverCostEstimate with total cost and component breakdown.
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        cost_params = self.get_cost_params(list(all_symbols))

        total_spread_bps = 0.0
        total_impact_bps = 0.0
        total_turnover_pct = 0.0
        symbol_costs = {}

        for symbol in all_symbols:
            cur = current_weights.get(symbol, 0.0)
            tgt = target_weights.get(symbol, 0.0)
            delta = tgt - cur

            # Absolute turnover (in percent)
            turnover_pct = abs(delta) * 100  # Convert decimal to percent
            total_turnover_pct += turnover_pct

            # Spread cost: linear in |delta| (in weight decimal)
            spread_cost = cost_params.spread.get(symbol, 2.0) * abs(delta)

            # Impact cost: quadratic in delta (in weight decimal)
            impact_cost = cost_params.impact.get(symbol, 1.0) * (delta ** 2)

            # Scale: the coefficients are in bps per % turnover for spread
            # For impact, coefficient is in bps per (1% turnover)²
            # Delta is in decimal, so we need to scale
            spread_bps = spread_cost * 100  # Convert back to bps
            impact_bps = impact_cost * 10000  # Scale quadratic term properly

            total_spread_bps += spread_bps
            total_impact_bps += impact_bps

            symbol_costs[symbol] = {
                "current": cur,
                "target": tgt,
                "delta": delta,
                "turnover_pct": turnover_pct,
                "spread_bps": spread_bps,
                "impact_bps": impact_bps,
                "total_bps": spread_bps + impact_bps,
            }

        # Active turnover as fraction of portfolio
        active_turnover_pct = total_turnover_pct / 2  # Double-counted, divide by 2

        # Determine calibration source
        source = "tca_feedback" if self._load_tca_feedback() else "default"

        return TurnoverCostEstimate(
            total_cost_bps=total_spread_bps + total_impact_bps,
            spread_cost_bps=total_spread_bps,
            impact_cost_bps=total_impact_bps,
            symbol_costs=symbol_costs,
            calibration_source=source,
            active_turnover_pct=active_turnover_pct,
        )


# ------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------

def get_default_cost_aversion() -> float:
    """Get the recommended default cost aversion parameter."""
    return 0.01


def compute_cost_penalty(
    weights: Dict[str, float],
    current_weights: Dict[str, float],
    spread_costs: Dict[str, float],
    impact_costs: Dict[str, float],
    cost_aversion: float = 0.01,
) -> float:
    """
    Compute cost penalty term for optimization objective.

    Penalty = γ * Σ_i [ spread_i * |w_i - w₀_i| + impact_i * (w_i - w₀_i)² ]

    This is designed to be called with pre-computed cost parameters.
    The result is in the same scale as the variance term (decimal²).

    Args:
        weights: Target weights dict
        current_weights: Current/base weights dict
        spread_costs: Per-symbol spread cost coefficients
        impact_costs: Per-symbol quadratic impact coefficients
        cost_aversion: Cost aversion parameter γ

    Returns:
        Cost penalty value (scalar, for use in objective function)
    """
    total_penalty = 0.0
    for symbol, w in weights.items():
        w0 = current_weights.get(symbol, 0.0)
        delta = w - w0
        if abs(delta) < 1e-8:
            continue  # Skip negligible changes
        spread = spread_costs.get(symbol, 2.0)
        impact = impact_costs.get(symbol, 1.0)
        # Scale: cost coefficients are in bps; variance is in decimal²
        # Convert cost to same scale as variance
        linear_cost = spread * abs(delta)
        quad_cost = impact * (delta ** 2)
        total_penalty += linear_cost + quad_cost

    return float(cost_aversion * total_penalty)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    """CLI entry point for cost model inspection."""
    import argparse

    parser = argparse.ArgumentParser(description="Almgren-Chriss Cost Model")
    sub = parser.add_subparsers(dest="command", help="Command")

    # Estimate command
    est_cmd = sub.add_parser("estimate", help="Estimate turnover cost")
    est_cmd.add_argument("--from", dest="from_weights", nargs="+", default=["SPY:0.46", "GLD:0.38", "TLT:0.16"])
    est_cmd.add_argument("--to", dest="to_weights", nargs="+", default=["SPY:0.50", "GLD:0.35", "TLT:0.15"])

    # Params command
    sub.add_parser("params", help="Show cost parameters")
    sub.add_parser("calibration", help="Show TCA calibration")

    args = parser.parse_args()
    model = AlmgrenChrissCostModel()

    if args.command == "estimate":
        def parse_weights(parts):
            w = {}
            for part in parts:
                sym, val = part.split(":")
                w[sym] = float(val)
            return w

        cur = parse_weights(args.from_weights)
        tgt = parse_weights(args.to_weights)
        est = model.estimate_turnover_cost(cur, tgt)

        print("=" * 60)
        print("  TURNOVER COST ESTIMATE")
        print("=" * 60)
        print(f"  Calibration: {est.calibration_source}")
        print(f"  Active Turnover: {est.active_turnover_pct:.2f}%")
        print(f"  Total Cost: {est.total_cost_bps:.2f} bps")
        print(f"  ├─ Spread: {est.spread_cost_bps:.2f} bps")
        print(f"  └─ Impact: {est.impact_cost_bps:.2f} bps")
        print()
        print(f"  {'Symbol':6s} {'Δw':>8s} {'Spread':>8s} {'Impact':>8s} {'Total':>8s}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for sym, sc in est.symbol_costs.items():
            print(f"  {sym:6s} {sc['delta']:>+8.3f} {sc['spread_bps']:>8.2f} "
                  f"{sc['impact_bps']:>8.2f} {sc['total_bps']:>8.2f}")

    elif args.command == "params":
        params = model.get_cost_params(
            ["SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH"],
        )
        print("=" * 60)
        print("  COST PARAMETERS")
        print("=" * 60)
        print(f"  Source: {params.calibration_source}")
        print(f"  Default Cost Aversion: {params.cost_aversion_default}")
        print()
        print(f"  {'Symbol':6s} {'Spread(bps/%)':>16s} {'Impact(bps/%)':>16s}")
        print(f"  {'-'*6} {'-'*16} {'-'*16}")
        for sym in sorted(params.spread.keys()):
            s = params.spread.get(sym, 0)
            i = params.impact.get(sym, 0)
            print(f"  {sym:6s} {s:>16.1f} {i:>16.1f}")

    elif args.command == "calibration":
        cal = model.get_tca_calibration_summary()
        print("=" * 60)
        print("  TCA CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"  Source: {cal.get('source', 'none')}")
        print(f"  Overall Quality: {cal.get('overall_quality', 'N/A')}")
        print()
        factors = cal.get("factors", {})
        if factors:
            print(f"  {'Symbol':6s} {'Calibration':>12s} {'Quality':>10s} {'Bucket':>10s}")
            print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}")
            for sym, data in sorted(factors.items()):
                print(f"  {sym:6s} {data['cost_calibration']:>12.1f} "
                      f"{data['avg_quality']:>10.1f} {data['quality_bucket']:>10s}")
        else:
            print("  No TCA feedback data available for calibration.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
