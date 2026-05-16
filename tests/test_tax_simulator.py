"""
Tests for TaxSimulator — v7.03
"""

import pytest
from datetime import date, timedelta

from src.rebalancing.tax_simulator import (
    TaxSimulator,
    SimulationResult,
    SimulationIteration,
)


class TestSimulationIteration:
    """Test the iteration dataclass."""

    def test_basic_creation(self):
        it = SimulationIteration(
            year=1, naive_tax_cost_bps=100, optimal_tax_cost_bps=60,
            tax_alpha_bps=40, naive_tlh_benefit_bps=50, optimal_tlh_benefit_bps=80,
            naive_wash_count=0, optimal_wash_count=0,
            naive_rebalances=2, optimal_rebalances=2,
            realized_st_gains_bps=30, realized_lt_gains_bps=70,
        )
        assert it.year == 1
        assert it.tax_alpha_bps == 40
        assert it.naive_rebalances == 2


class TestSimulationResult:
    """Test the simulation result."""

    def test_summary(self):
        result = SimulationResult(
            iterations=[],
            total_years=5,
            annual_tax_alpha_bps=35.5,
            avg_naive_cost_bps=120.0,
            avg_optimal_cost_bps=84.5,
            total_naive_rebalances=10,
            total_optimal_rebalances=8,
            avg_st_lt_ratio=0.85,
            wash_sale_incidents=2,
            recovery_pct=29.6,
        )
        summary = result.summary
        assert summary['total_years'] == 5
        assert summary['annual_tax_alpha_bps'] == 35.5
        assert summary['recovery_pct'] == 29.6


class TestTaxSimulator:
    """Test the main simulator."""

    def test_run_simulation_basic(self):
        simulator = TaxSimulator(seed=42)
        result = simulator.run_simulation(
            holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
            prices={'SPY': 480.0, 'GLD': 195.0, 'TLT': 92.0},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            years=3,
            rebalance_frequency='annual',
        )
        assert result.total_years == 3
        assert len(result.iterations) == 3
        assert result.annual_tax_alpha_bps >= 0  # Alpha should be non-negative

    def test_run_simulation_semi_annual(self):
        simulator = TaxSimulator(seed=42)
        result = simulator.run_simulation(
            holdings={'SPY': 50000, 'GLD': 30000, 'TLT': 20000},
            prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            targets={'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20},
            years=2,
            rebalance_frequency='semi_annual',
        )
        assert result.total_years == 2
        assert len(result.iterations) == 2

    def test_simulation_deterministic(self):
        """Same seed should produce same results."""
        r1 = TaxSimulator(seed=42).run_simulation(
            holdings={'SPY': 50000, 'GLD': 30000, 'TLT': 20000},
            prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            targets={'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20},
            years=2,
        )
        r2 = TaxSimulator(seed=42).run_simulation(
            holdings={'SPY': 50000, 'GLD': 30000, 'TLT': 20000},
            prices={'SPY': 500.0, 'GLD': 200.0, 'TLT': 90.0},
            targets={'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20},
            years=2,
        )
        assert r1.annual_tax_alpha_bps == r2.annual_tax_alpha_bps

    def test_single_year_simulation(self):
        simulator = TaxSimulator(seed=42)
        result = simulator.run_simulation(
            holdings={'SPY': 100000},
            prices={'SPY': 500.0},
            targets={'SPY': 1.0},
            years=1,
        )
        assert result.total_years == 1
        assert len(result.iterations) == 1

    def test_empty_holdings(self):
        simulator = TaxSimulator(seed=42)
        # Should not crash with hold-all configuration
        result = simulator.run_simulation(
            holdings={'SPY': 50000, 'GLD': 30000},
            prices={'SPY': 500.0, 'GLD': 200.0},
            targets={'SPY': 0.50, 'GLD': 0.50},
            years=1,
        )
        assert result.total_years == 1

    def test_return_types(self):
        simulator = TaxSimulator(seed=42)
        result = simulator.run_simulation(
            holdings={'SPY': 46000, 'GLD': 38000, 'TLT': 16000},
            prices={'SPY': 480.0, 'GLD': 195.0, 'TLT': 92.0},
            targets={'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16},
            years=2,
        )
        summary = result.summary
        assert isinstance(summary['total_years'], int)
        assert isinstance(summary['annual_tax_alpha_bps'], float)
        assert isinstance(summary['avg_naive_cost_bps'], float)
        assert isinstance(summary['avg_optimal_cost_bps'], float)
        assert isinstance(summary['total_naive_rebalances'], int)
        assert isinstance(summary['total_optimal_rebalances'], int)
        assert isinstance(summary['recovery_pct'], float)
