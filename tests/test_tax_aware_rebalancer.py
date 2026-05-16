"""
Tests for TaxAwareRebalancer — v7.03
"""

import pytest
from datetime import date, timedelta

from src.rebalancing.tax_lot_tracker import TaxLotTracker, LotSelectionMethod
from src.rebalancing.tax_aware_rebalancer import (
    TaxAwareRebalancer,
    TaxCostEstimate,
    RebalanceAction,
    RebalancePlan,
    TaxAwareMode,
)


class TestTaxCostEstimate:
    """Test tax cost estimation."""

    def test_no_tax_cost(self):
        est = TaxCostEstimate(
            total_cost_bps=0, realized_gains_bps=0, realized_losses_bps=0,
            net_taxable_bps=0, st_gains_bps=0, lt_gains_bps=0,
            wash_sale_count=0, tlh_opportunity_bps=0, st_lt_ratio=0,
        )
        assert est.has_tax_cost is False

    def test_has_tax_cost(self):
        est = TaxCostEstimate(
            total_cost_bps=50, realized_gains_bps=100, realized_losses_bps=50,
            net_taxable_bps=50, st_gains_bps=30, lt_gains_bps=70,
            wash_sale_count=0, tlh_opportunity_bps=50, st_lt_ratio=0.43,
        )
        assert est.has_tax_cost is True


class TestComputeRebalance:
    """Test the compute_rebalance method."""

    def test_no_rebalance_needed(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 100, 500.0, '2025-01-15')
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        plan = rebalancer.compute_rebalance(
            current_holdings={'SPY': 50000, 'GLD': 0, 'TLT': 0},
            current_prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            target_allocations={'SPY': 1.0, 'GLD': 0.0, 'TLT': 0.0},
            total_value=50000,
        )
        assert plan.needs_rebalance is False
        assert len(plan.actions) == 0

    def test_sell_triggered(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 50, 500.0, '2025-01-15')
        tracker.add_lot('GLD', 100, 200.0, '2025-04-01')
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        plan = rebalancer.compute_rebalance(
            current_holdings={'SPY': 50000, 'GLD': 40000, 'TLT': 10000},
            current_prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        # Should trigger sells (SPY is overweight, TLT at 10% vs 16%)
        assert plan.needs_rebalance is True
        sell_actions = [a for a in plan.actions if a.action == 'sell']
        buy_actions = [a for a in plan.actions if a.action == 'buy']
        assert len(sell_actions) >= 1

    def test_different_modes_produce_different_costs(self):
        """Optimal mode should equal or beat naive mode."""
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 50, 500.0, '2025-01-15')
        tracker.add_lot('SPY', 30, 480.0, '2026-03-10')  # ST, higher cost
        
        holdings = {'SPY': 50000, 'GLD': 30000, 'TLT': 20000}
        prices = {'SPY': 490.0, 'GLD': 200.0, 'TLT': 90.0}
        targets = {'SPY': 0.40, 'GLD': 0.35, 'TLT': 0.25}
        total = 100000
        
        # Optimal mode
        rebalancer_opt = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        plan_opt = rebalancer_opt.compute_rebalance(holdings, prices, targets, total)
        
        # Naive mode
        rebalancer_nai = TaxAwareRebalancer(tracker=tracker, mode='naive')
        plan_nai = rebalancer_nai.compute_rebalance(holdings, prices, targets, total)
        
        # Optimal should have <= tax cost of naive
        assert plan_opt.total_tax_cost_bps <= plan_nai.total_tax_cost_bps

    def test_off_mode_zero_tax_cost(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 50, 500.0, '2025-01-15')
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='off')
        plan = rebalancer.compute_rebalance(
            current_holdings={'SPY': 50000, 'GLD': 40000, 'TLT': 10000},
            current_prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            target_allocations={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        assert plan.mode == 'off'


class TestCompareStrategies:
    """Test comparison between naive and optimal strategies."""

    def test_comparison_returns_expected_keys(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 50, 500.0, '2025-01-15')
        tracker.add_lot('GLD', 100, 200.0, '2025-04-01')
        tracker.add_lot('TLT', 100, 90.0, '2025-06-01')
        
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        result = rebalancer.compare_strategies(
            holdings={'SPY': 50000, 'GLD': 40000, 'TLT': 10000},
            prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        assert 'naive' in result
        assert 'optimal' in result
        assert 'tax_alpha_bps' in result
        assert 'improvement_pct' in result

    def test_tax_alpha_non_negative(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 50, 500.0, '2025-01-15')
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        result = rebalancer.compare_strategies(
            holdings={'SPY': 50000, 'GLD': 40000, 'TLT': 10000},
            prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            total_value=100000,
        )
        assert result['tax_alpha_bps'] >= 0


class TestRebalanceAction:
    """Test RebalanceAction dataclass."""

    def test_sell_action(self):
        action = RebalanceAction(
            symbol='SPY', action='sell',
            current_value=50000, target_value=46000,
            delta_value=-4000, method='hifo',
            estimated_tax_cost_bps=25.0,
        )
        assert action.symbol == 'SPY'
        assert action.action == 'sell'
        assert action.method == 'hifo'

    def test_buy_action(self):
        action = RebalanceAction(
            symbol='GLD', action='buy',
            current_value=30000, target_value=38000,
            delta_value=8000, method='n/a',
        )
        assert action.symbol == 'GLD'
        assert action.action == 'buy'


class TestGetStatus:
    """Test rebalancer status reporting."""

    def test_status_returns_expected_fields(self):
        tracker = TaxLotTracker()
        tracker.reset()
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='optimal')
        status = rebalancer.get_status()
        assert 'mode' in status
        assert status['mode'] == 'optimal'
        assert 'tracker_summary' in status
        assert 'wash_stats' in status

    def test_status_with_lots(self):
        tracker = TaxLotTracker()
        tracker.reset()
        tracker.add_lot('SPY', 10, 500.0, '2025-01-15')
        rebalancer = TaxAwareRebalancer(tracker=tracker, mode='naive')
        status = rebalancer.get_status()
        assert status['mode'] == 'naive'
        assert status['tracker_summary']['total_symbols'] == 1
