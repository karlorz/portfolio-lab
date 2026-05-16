"""
v7.03: Tax-Aware Rebalancer — v7.03
Takes target allocation weights, compares to current (with tax lots),
prioritizes selling lots with losses (tax-loss harvesting), defers selling
lots with short-term gains, and estimates "tax cost" for any proposed rebalance.

Option: set max tax cost per rebalance as constraint.
Option: tax_aware_mode in config (off / naive / optimal).

Integration with EnsembleVoter via TaxAlphaSignalSource.

Usage:
    rebalancer = TaxAwareRebalancer()
    result = rebalancer.compute_rebalance(
        current_holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
        current_prices={'SPY': 500, 'GLD': 200, 'TLT': 90},
        target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
        total_value=100000,
    )
    if result.needs_rebalance:
        print(f"Tax cost: {result.tax_cost_bps} bps, TLH benefit: {result.tlh_benefit_bps} bps")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .tax_lot_tracker import TaxLotTracker, LotSelectionMethod, HoldingPeriod

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TaxAwareMode:
    """Tax-aware rebalancing modes."""
    OFF = "off"          # No tax awareness, just drifts
    NAIVE = "naive"      # Sell oldest lots (FIFO), no TLH preference
    OPTIMAL = "optimal"  # Tax-loss harvest first, defer ST gains


@dataclass
class TaxCostEstimate:
    """Estimated tax cost of a proposed rebalance."""
    total_cost_bps: float           # Total tax cost in bps
    realized_gains_bps: float       # Gains component
    realized_losses_bps: float      # Losses component that can offset gains
    net_taxable_bps: float          # Net taxable gain (gains - losses)
    st_gains_bps: float             # Short-term gains (taxed at ordinary rate)
    lt_gains_bps: float             # Long-term gains (taxed at lower rate)
    wash_sale_count: int            # Number of wash sales detected
    tlh_opportunity_bps: float      # Tax-loss harvesting opportunity
    st_lt_ratio: float              # Short-term / long-term gain ratio
    
    @property
    def has_tax_cost(self) -> bool:
        return self.total_cost_bps > 0.0


@dataclass
class RebalanceAction:
    """A single rebalance action (buy or sell a symbol)."""
    symbol: str
    action: str                    # "buy" or "sell"
    current_value: float
    target_value: float
    delta_value: float             # Positive = buy, negative = sell
    method: str = "fifo"           # Lot selection method for sells
    estimated_tax_cost_bps: float = 0.0


@dataclass
class RebalancePlan:
    """Complete rebalance plan with actions and tax estimates."""
    actions: List[RebalanceAction]
    tax_estimate: TaxCostEstimate
    needs_rebalance: bool
    mode: str                      # Which TaxAwareMode was used
    total_delta: float
    total_tax_cost_bps: float
    total_tlh_benefit_bps: float
    wash_sale_count: int
    warnings: List[str] = field(default_factory=list)


class TaxAwareRebalancer:
    """
    Tax-aware rebalancing engine.
    Compares current holdings to target allocations, computes optimal lot
    selection to minimize tax impact while tracking target drift.
    """
    
    def __init__(self, tracker: Optional[TaxLotTracker] = None,
                 mode: str = 'optimal',
                 max_tax_cost_bps: float = 50.0,
                 max_st_lt_ratio: float = 2.0,
                 state_path: Optional[str] = None):
        """
        Args:
            tracker: TaxLotTracker instance (creates default if None)
            mode: TaxAwareMode ('off', 'naive', 'optimal')
            max_tax_cost_bps: Maximum acceptable tax cost per rebalance
            max_st_lt_ratio: Maximum acceptable ST/LT gain ratio
        """
        self.tracker = tracker or TaxLotTracker(
            state_path or str(PROJECT_ROOT / 'data' / 'tax_lots_state.json')
        )
        self.mode = mode
        self.max_tax_cost_bps = max_tax_cost_bps
        self.max_st_lt_ratio = max_st_lt_ratio
    
    def _compute_current_allocation(self, holdings: Dict[str, float], total_value: float) -> Dict[str, float]:
        """Compute current allocation percentages from holdings."""
        if total_value <= 0:
            return {}
        return {sym: value / total_value for sym, value in holdings.items()}
    
    def _compute_target_values(self, holdings: Dict[str, float],
                                targets: Dict[str, float],
                                total_value: float) -> Dict[str, float]:
        """Compute target dollar values for each holding."""
        if total_value <= 0:
            return {}
        return {sym: total_value * target for sym, target in targets.items()}
    
    def _estimate_sale_tax_cost(self, symbol: str, shares_to_sell: float,
                                 sale_price: float, method: str = 'fifo') -> TaxCostEstimate:
        """
        Estimate tax cost of selling shares without actually executing.
        Returns a TaxCostEstimate for the proposed sale.
        """
        if symbol not in self.tracker.lots or not self.tracker.lots[symbol]:
            return TaxCostEstimate(
                total_cost_bps=0, realized_gains_bps=0, realized_losses_bps=0,
                net_taxable_bps=0, st_gains_bps=0, lt_gains_bps=0,
                wash_sale_count=0, tlh_opportunity_bps=0, st_lt_ratio=0,
            )
        
        lots = self.tracker.lots[symbol]
        
        # Select lots based on method
        if method == 'lifo':
            selected_lots = sorted(lots, key=lambda l: l.acquisition_date, reverse=True)
        elif method == 'hifo':
            selected_lots = sorted(lots, key=lambda l: l.cost_basis_per_share, reverse=True)
        else:
            selected_lots = sorted(lots, key=lambda l: l.acquisition_date)  # FIFO
        
        remaining = shares_to_sell
        total_gains = 0.0
        total_losses = 0.0
        total_st_gains = 0.0
        total_lt_gains = 0.0
        total_shares_sold = 0.0
        
        for lot in selected_lots:
            if remaining <= 0:
                break
            taken = min(lot.shares, remaining)
            pl = (sale_price - lot.cost_basis_per_share) * taken
            
            if pl >= 0:
                total_gains += pl
                if lot.is_long_term:
                    total_lt_gains += pl
                else:
                    total_st_gains += pl
            else:
                total_losses += abs(pl)
            
            remaining -= taken
            total_shares_sold += taken
        
        if total_shares_sold == 0:
            return TaxCostEstimate(
                total_cost_bps=0, realized_gains_bps=0, realized_losses_bps=0,
                net_taxable_bps=0, st_gains_bps=0, lt_gains_bps=0,
                wash_sale_count=0, tlh_opportunity_bps=0, st_lt_ratio=0,
            )
        
        net_taxable = total_gains - total_losses
        total_value_sold = total_shares_sold * sale_price
        
        # Convert to bps
        total_cost_bps = (net_taxable / total_value_sold * 10000) if total_value_sold > 0 else 0
        gains_bps = (total_gains / total_value_sold * 10000) if total_value_sold > 0 else 0
        losses_bps = (total_losses / total_value_sold * 10000) if total_value_sold > 0 else 0
        st_gains_bps = (total_st_gains / total_value_sold * 10000) if total_value_sold > 0 else 0
        lt_gains_bps = (total_lt_gains / total_value_sold * 10000) if total_value_sold > 0 else 0
        tlh_opportunity = losses_bps  # Losses are TLH opportunity
        st_lt_ratio = st_gains_bps / lt_gains_bps if lt_gains_bps > 0 else float('inf') if st_gains_bps > 0 else 0
        
        return TaxCostEstimate(
            total_cost_bps=round(total_cost_bps, 2),
            realized_gains_bps=round(gains_bps, 2),
            realized_losses_bps=round(losses_bps, 2),
            net_taxable_bps=round(max(0, net_taxable / total_value_sold * 10000), 2),
            st_gains_bps=round(st_gains_bps, 2),
            lt_gains_bps=round(lt_gains_bps, 2),
            wash_sale_count=len(self.tracker.detect_wash_sales(symbol)),
            tlh_opportunity_bps=round(tlh_opportunity, 2),
            st_lt_ratio=st_lt_ratio,
        )
    
    def _select_best_sell_method(self, symbol: str, shares_to_sell: float,
                                  sale_price: float) -> Tuple[str, TaxCostEstimate]:
        """
        Select the best lot selection method (FIFO, HIFO, LIFO) to minimize tax cost.
        
        In 'optimal' mode, prefer HIFO (sell highest-cost-basis first) to maximize
        realized losses and minimize gains. Fall back to FIFO for tiebreak.
        """
        if self.mode == TaxAwareMode.NAIVE:
            method = 'fifo'
            estimate = self._estimate_sale_tax_cost(symbol, shares_to_sell, sale_price, method)
            return method, estimate
        
        if self.mode == TaxAwareMode.OPTIMAL:
            # Compare methods and pick the one with lowest tax cost
            candidates = {}
            for m in ['hifo', 'fifo', 'lifo']:
                est = self._estimate_sale_tax_cost(symbol, shares_to_sell, sale_price, m)
                candidates[m] = est
            
            # HIFO maximizes losses (negative net_taxable is best)
            # Sort by net_taxable_bps (lower is better)
            best_method = min(candidates, key=lambda m: (
                candidates[m].net_taxable_bps,
                -candidates[m].tlh_opportunity_bps,  # More TLH is better
                m != 'hifo',  # Prefer HIFO as tiebreaker
            ))
            return best_method, candidates[best_method]
        
        # OFF mode: just FIFO, no tax optimization
        return 'fifo', TaxCostEstimate(
            total_cost_bps=0, realized_gains_bps=0, realized_losses_bps=0,
            net_taxable_bps=0, st_gains_bps=0, lt_gains_bps=0,
            wash_sale_count=0, tlh_opportunity_bps=0, st_lt_ratio=0,
        )
    
    def compute_rebalance(self, current_holdings: Dict[str, float],
                          current_prices: Dict[str, float],
                          target_allocations: Dict[str, float],
                          total_value: float) -> RebalancePlan:
        """
        Compute the optimal rebalance plan given current state.
        
        Args:
            current_holdings: symbol -> current market value
            current_prices: symbol -> current price per share
            target_allocations: symbol -> target allocation (0-1)
            total_value: Total portfolio value
            
        Returns:
            RebalancePlan with actions and tax estimates
        """
        if total_value <= 0:
            return RebalancePlan(
                actions=[], needs_rebalance=False, mode=self.mode,
                total_delta=0, total_tax_cost_bps=0,
                total_tlh_benefit_bps=0, wash_sale_count=0,
                tax_estimate=TaxCostEstimate(
                    total_cost_bps=0, realized_gains_bps=0, realized_losses_bps=0,
                    net_taxable_bps=0, st_gains_bps=0, lt_gains_bps=0,
                    wash_sale_count=0, tlh_opportunity_bps=0, st_lt_ratio=0,
                ),
            )
        
        current_allocs = self._compute_current_allocation(current_holdings, total_value)
        target_values = self._compute_target_values(current_holdings, target_allocations, total_value)
        
        actions = []
        warnings = []
        total_net_taxable_bps = 0
        total_tlh_benefit = 0
        total_wash_count = 0
        
        for symbol in target_allocations:
            current_value = current_holdings.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            delta = target_value - current_value
            price = current_prices.get(symbol, 0)
            
            if abs(delta) / total_value < 0.001:  # < 0.1% change, skip
                continue
            
            if delta > 0:
                # Buy — track as new lot
                if price > 0:
                    shares_to_buy = delta / price
                    action = RebalanceAction(
                        symbol=symbol, action="buy",
                        current_value=current_value,
                        target_value=target_value,
                        delta_value=delta,
                        method="n/a",
                        estimated_tax_cost_bps=0,
                    )
                    actions.append(action)
            else:
                # Sell — use tax-aware lot selection
                if price > 0:
                    shares_to_sell = abs(delta) / price
                    method, tax_est = self._select_best_sell_method(
                        symbol, shares_to_sell, price
                    )
                    total_net_taxable_bps += tax_est.net_taxable_bps
                    total_tlh_benefit += tax_est.tlh_opportunity_bps
                    total_wash_count += tax_est.wash_sale_count
                    
                    action = RebalanceAction(
                        symbol=symbol, action="sell",
                        current_value=current_value,
                        target_value=target_value,
                        delta_value=delta,
                        method=method,
                        estimated_tax_cost_bps=round(tax_est.total_cost_bps, 2),
                    )
                    actions.append(action)
                    
                    # Check for wash sale warning
                    if tax_est.wash_sale_count > 0:
                        warnings.append(f"Wash sale risk for {symbol}: {tax_est.wash_sale_count} lots within 30-day window")
        
        needs_rebalance = len(actions) > 0
        total_delta = sum(abs(a.delta_value) for a in actions)
        
        # Check constraints
        if self.mode == TaxAwareMode.OPTIMAL:
            if total_net_taxable_bps > self.max_tax_cost_bps:
                warnings.append(
                    f"Tax cost {total_net_taxable_bps:.1f} bps exceeds limit {self.max_tax_cost_bps} bps. "
                    "Consider deferring rebalance or splitting into smaller trades."
                )
        
        plan_tax_estimate = TaxCostEstimate(
            total_cost_bps=round(total_net_taxable_bps, 2),
            realized_gains_bps=0,
            realized_losses_bps=0,
            net_taxable_bps=round(total_net_taxable_bps, 2),
            st_gains_bps=0,
            lt_gains_bps=0,
            wash_sale_count=total_wash_count,
            tlh_opportunity_bps=round(total_tlh_benefit, 2),
            st_lt_ratio=0,
        )
        
        return RebalancePlan(
            actions=actions,
            tax_estimate=plan_tax_estimate,
            needs_rebalance=needs_rebalance,
            mode=self.mode,
            total_delta=round(total_delta, 2),
            total_tax_cost_bps=round(total_net_taxable_bps, 2),
            total_tlh_benefit_bps=round(total_tlh_benefit, 2),
            wash_sale_count=total_wash_count,
            warnings=warnings,
        )
    
    def execute_rebalance(self, plan: RebalancePlan) -> Dict[str, List]:
        """
        Execute a rebalance plan by updating tax lots.
        
        Args:
            plan: RebalancePlan from compute_rebalance
            
        Returns:
            Dict with 'sold' and 'bought' lot records
        """
        result = {'sold': [], 'bought': []}
        today = date.today().isoformat()
        
        for action in plan.actions:
            if action.action == "sell" and action.delta_value < 0:
                price = action.target_value / (abs(action.delta_value)) if action.delta_value else 0
                shares = abs(action.delta_value) / price if price > 0 else 0
                sold = self.tracker.sell_lots(
                    action.symbol, shares, price, today, method=action.method
                )
                result['sold'].extend(sold)
            
            elif action.action == "buy" and action.delta_value > 0:
                price = action.delta_value  # dollars
                shares = action.delta_value / (action.target_value / action.delta_value) if action.delta_value else 0
                # Simpler: compute shares from price
                # For buy, we just record the lot
                avg_price = 1.0  # placeholder — real price would come from order
                result['bought'].append({
                    'symbol': action.symbol,
                    'value': action.delta_value,
                })
        
        self.tracker.save_state()
        return result
    
    def compare_strategies(self, holdings: Dict[str, float],
                           prices: Dict[str, float],
                           targets: Dict[str, float],
                           total_value: float) -> Dict:
        """
        Compare naive vs optimal tax rebalancing for the same scenario.
        Returns: dict with 'naive' and 'optimal' plans and 'tax_alpha_bps'
        """
        # Run with naive mode
        naive_rebalancer = TaxAwareRebalancer(
            tracker=self.tracker, mode=TaxAwareMode.NAIVE,
        )
        naive_plan = naive_rebalancer.compute_rebalance(
            holdings, prices, targets, total_value
        )
        
        # Run with optimal mode
        optimal_plan = self.compute_rebalance(
            holdings, prices, targets, total_value
        )
        
        tax_alpha_bps = naive_plan.total_tax_cost_bps - optimal_plan.total_tax_cost_bps
        
        return {
            'naive': {
                'tax_cost_bps': naive_plan.total_tax_cost_bps,
                'tlh_benefit_bps': naive_plan.total_tlh_benefit_bps,
                'wash_sales': naive_plan.wash_sale_count,
                'actions': len(naive_plan.actions),
            },
            'optimal': {
                'tax_cost_bps': optimal_plan.total_tax_cost_bps,
                'tlh_benefit_bps': optimal_plan.total_tlh_benefit_bps,
                'wash_sales': optimal_plan.wash_sale_count,
                'actions': len(optimal_plan.actions),
            },
            'tax_alpha_bps': round(tax_alpha_bps, 2),
            'improvement_pct': round(
                (tax_alpha_bps / naive_plan.total_tax_cost_bps * 100)
                if naive_plan.total_tax_cost_bps > 0 else 0, 1
            ),
        }
    
    def get_status(self) -> Dict:
        """Get current rebalancer status for dashboard."""
        return {
            'mode': self.mode,
            'max_tax_cost_bps': self.max_tax_cost_bps,
            'tracker_summary': self.tracker.get_summary(),
            'wash_stats': self.tracker.get_wash_sales_stats(),
        }


def create_rebalancer(mode: str = 'optimal') -> TaxAwareRebalancer:
    """Create a default TaxAwareRebalancer instance."""
    return TaxAwareRebalancer(mode=mode)


def demo():
    """Demonstrate the tax-aware rebalancer."""
    print("=== Tax-Aware Rebalancer Demo ===\n")
    
    # Create tracker and add lots
    tracker = TaxLotTracker()
    tracker.reset()
    
    # Add some lots with varying cost bases
    tracker.add_lot('SPY', 10, 475.0, '2025-01-15')   # LT, low cost
    tracker.add_lot('SPY', 8, 500.0, '2025-06-01')    # LT, high cost
    tracker.add_lot('SPY', 5, 482.0, '2026-03-10')    # ST
    
    tracker.add_lot('GLD', 20, 190.0, '2025-04-01')   # LT
    tracker.add_lot('GLD', 10, 180.0, '2026-02-15')   # ST
    
    # Current state
    holdings = {'SPY': 23000 * 60, 'GLD': 19000 * 30, 'TLT': 9000}  # Simpler
    holdings = {'SPY': 46000, 'GLD': 38000, 'TLT': 16000}
    prices = {'SPY': 480.0, 'GLD': 195.0, 'TLT': 92.0}
    targets = {'SPY': 0.50, 'GLD': 0.35, 'TLT': 0.15}
    total_value = 100000
    
    # Compare strategies
    for mode in ['off', 'naive', 'optimal']:
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode=mode)
        plan = rebalancer.compute_rebalance(holdings, prices, targets, total_value)
        print(f"\nMode: {mode.upper()}")
        print(f"  Rebalance needed: {plan.needs_rebalance}")
        print(f"  Tax cost: {plan.total_tax_cost_bps:.2f} bps")
        print(f"  TLH benefit: {plan.total_tlh_benefit_bps:.2f} bps")
        print(f"  Actions: {len(plan.actions)}")
        for a in plan.actions:
            print(f"    {a.action.upper()} {a.symbol}: ${abs(a.delta_value):.0f} ({a.method})")
        print(f"  Wash sales: {plan.wash_sale_count}")
        for w in plan.warnings:
            print(f"  ⚠ {w}")
    
    # Show strategy comparison
    rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
    comparison = rebalancer.compare_strategies(holdings, prices, targets, total_value)
    print(f"\n=== Strategy Comparison ===")
    print(f"Naive tax cost: {comparison['naive']['tax_cost_bps']:.2f} bps")
    print(f"Optimal tax cost: {comparison['optimal']['tax_cost_bps']:.2f} bps")
    print(f"Tax alpha: {comparison['tax_alpha_bps']:.2f} bps ({comparison['improvement_pct']:.1f}%)")


if __name__ == '__main__':
    demo()
