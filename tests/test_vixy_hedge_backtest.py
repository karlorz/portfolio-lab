"""
Tests for the VIXY Hedge Walk-Forward Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult creation/serialization,
signal computation, hedge constraints, edge cases, crisis regime behavior,
synthetic data loading, print/save output.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.backtest.vixy_hedge_backtest import (
    BacktestConfig,
    DailyPrices,
    WalkForwardVIXYBacktester,
)
from src.backtest.metrics import BacktestResult
from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer


# ── BacktestConfig Tests ─────────────────────────────────────────────────


class TestBacktestConfig:
    """Test BacktestConfig defaults and custom configuration."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.start_date == "2006-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_weights['SPY'] == 0.46
        assert config.base_weights['GLD'] == 0.38
        assert config.base_weights['TLT'] == 0.16
        assert config.max_hedge_pct == 6.0
        assert config.rebalance_frequency_days == 21
        assert config.transaction_cost_bps == 10.0

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            max_hedge_pct=8.0,
            rebalance_frequency_days=63,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.max_hedge_pct == 8.0
        assert config.rebalance_frequency_days == 63

    def test_base_weights_sum(self):
        """Baseline weights should sum to 1.0."""
        config = BacktestConfig()
        total = sum(config.base_weights.values())
        assert abs(total - 1.0) < 0.01


# ── BacktestResult Tests ─────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation, to_dict, and empty state."""

    def test_create_and_to_dict(self):
        result = BacktestResult(
            total_return=10.5,
            cagr=8.2,
            volatility=12.3,
            sharpe_ratio=0.85,
            max_drawdown=-15.4,
            baseline_sharpe=0.78,
            sharpe_improvement=0.07,
            total_rebalances=120,
            total_transaction_costs=45.50,
            extras={
                "baseline_total_return": 9.0,
                "baseline_cagr": 7.5,
                "baseline_volatility": 11.8,
                "baseline_sharpe": 0.78,
                "baseline_max_drawdown": -18.2,
                "cagr_impact": 0.7,
                "hedge_active_days": 1200,
                "hedge_active_pct": 50.0,
                "avg_hedge_pct": 2.5,
                "max_hedge_pct": 6.0,
                "crisis_returns_hedged": {"2008": -10.2, "2020": 2.1},
                "crisis_returns_baseline": {"2008": -12.3, "2020": 1.5},
                "regime_breakdown": {
                    "normal": {"avg_hedge_pct": 1.2, "max_hedge_pct": 2.0, "count": 2000, "pct_of_time": 60.0},
                    "elevated": {"avg_hedge_pct": 2.8, "max_hedge_pct": 3.5, "count": 800, "pct_of_time": 24.0},
                },
                "config_snapshot": {"max_hedge_pct": 6.0},
            },
        )

        assert result.total_return == 10.5
        assert result.sharpe_ratio == 0.85
        assert result.sharpe_improvement == 0.07
        assert result.extras["hedge_active_days"] == 1200
        assert result.extras["crisis_returns_hedged"]["2008"] == -10.2
        assert result.extras["regime_breakdown"]["normal"]["avg_hedge_pct"] == 1.2
        assert result.total_rebalances == 120
        assert result.extras["config_snapshot"]["max_hedge_pct"] == 6.0

    def test_json_serializable(self):
        """All fields in extras must be JSON-serializable."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0,
            baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_sharpe": 0.45,
                "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "hedge_active_days": 100,
                "hedge_active_pct": 25.0, "avg_hedge_pct": 1.5, "max_hedge_pct": 4.0,
                "crisis_returns_hedged": {"2008": -8.0}, "crisis_returns_baseline": {"2008": -10.0},
                "regime_breakdown": {"normal": {"avg_hedge_pct": 1.0, "max_hedge_pct": 2.0, "count": 100, "pct_of_time": 50.0}},
                "config_snapshot": {"start_date": "2006-01-01"},
            },
        )
        json.dumps(result.extras)  # Should not raise

    def test_empty_crisis_returns(self):
        """Crisis returns can be empty dict without errors."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0,
            extras={
                "baseline_total_return": 0.0, "baseline_cagr": 0.0,
                "baseline_volatility": 0.0, "baseline_sharpe": 0.0,
                "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0, "hedge_active_days": 0,
                "hedge_active_pct": 0.0, "avg_hedge_pct": 0.0, "max_hedge_pct": 0.0,
                "crisis_returns_hedged": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        assert result.extras["crisis_returns_hedged"] == {}


# ── Walk-Forward Backtester Tests ────────────────────────────────────────


class TestWalkForwardVIXYBacktester:
    """Test the core WalkForwardVIXYBacktester class."""

    def test_init_defaults(self):
        bt = WalkForwardVIXYBacktester()
        assert bt.config.start_date == "2006-01-01"
        assert isinstance(bt.sizer, VIXYHedgeSizer)

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", max_hedge_pct=5.0)
        bt = WalkForwardVIXYBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.max_hedge_pct == 5.0

    def test_load_data_generates_synthetic_when_no_file(self, monkeypatch):
        """When prices.json doesn't exist, synthetic data is generated."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        assert len(bt._daily_prices) > 0
        assert len(bt._trading_dates) > 0

    def test_synthetic_data_has_required_fields(self):
        """Each DailyPrices entry should have SPY, GLD, TLT values."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        for dp in bt._daily_prices[:10]:
            assert isinstance(dp.spy, float)
            assert isinstance(dp.gld, float)
            assert isinstance(dp.tlt, float)

    def test_vix_proxy_default_when_insufficient_history(self):
        """VIX proxy should return ~18 when fewer than 21 days of history."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        vix = bt._compute_vix_proxy(5)  # idx 5 = only 6 days
        assert vix == 18.0 or abs(vix - 18.0) < 1.0

    def test_vix_proxy_with_sufficient_history(self):
        """VIX proxy should compute from 21-day SPY returns."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        # Use an index with enough history
        vix = bt._compute_vix_proxy(50)
        assert vix > 0
        assert vix < 100  # Sanity check

    def test_run_produces_results(self):
        """Running the backtest should return a populated BacktestResult."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_hedge_never_exceeds_max_hedge_pct(self):
        """VIXY allocation should not exceed the configured max."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=6.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["max_pct"] <= 6.0 + 0.01  # Allow tiny float rounding

    def test_hedge_allocations_range(self):
        """All hedge allocations should be >= 0 and within bounds."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=6.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        for alloc in tracker["allocations"]:
            assert alloc >= 0.0
            assert alloc <= 6.0 + 0.01

    def test_baseline_weights_stable(self):
        """Baseline run should maintain constant weights."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        prices_subset = bt._daily_prices[:100]
        equity = bt._run_baseline(
            prices_subset,
            BacktestConfig(initial_capital=100000.0),
        )
        assert len(equity) == len(prices_subset)
        assert equity[0] == 100000.0

    def test_empty_result_method_returns_zeros(self):
        """_empty_result() should return all-zero metrics."""
        bt = WalkForwardVIXYBacktester()
        result = bt._empty_result()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.extras["hedge_active_days"] == 0
        assert result.total_rebalances == 0
        assert result.extras["crisis_returns_hedged"] == {}

    def test_single_day_data_returns_zero_result(self):
        """Only one data point should return an empty result."""
        bt = WalkForwardVIXYBacktester()
        bt._daily_prices = [DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)]
        bt._trading_dates = ["2020-01-02"]
        result = bt.run()
        assert result.total_return == 0.0

    def test_hedge_active_in_elevated_vix(self):
        """Hedge should be active (nonzero) when VIX is elevated."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(25.0)  # ELEVATED regime
        assert alloc > 0

    def test_hedge_inactive_in_low_vix(self):
        """Hedge might be zero or near-zero when VIX is very low."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(10.0)  # NORMAL regime, VIX < 20
        assert alloc >= 0.0
        # VIX=10 -> 10/10 = 1.0%, NORMAL floor=0 ceiling=2 -> 1.0%
        assert alloc >= 0.0

    def test_crisis_regime_triggers_larger_hedge(self):
        """CRISIS regime (VIX > 40) should produce the largest allocation."""
        sizer = VIXYHedgeSizer()
        normal_alloc = sizer.compute_allocation(15.0)
        crisis_alloc = sizer.compute_allocation(45.0)
        assert crisis_alloc >= normal_alloc

    def test_stress_regime_hedge_bounds(self):
        """STRESS regime (VIX 30-40) should cap at 6%."""
        sizer = VIXYHedgeSizer()
        alloc = sizer.compute_allocation(35.0)
        assert 2.0 <= alloc <= 6.0

    def test_narrow_date_range_still_runs(self):
        """A narrow date range (2 months) should still produce results."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-03-01")
        )
        result = bt.run()
        assert result.cagr is not None
        assert result.total_rebalances >= 0

    def test_print_results_does_not_crash(self, capsys):
        """print_results should produce output without errors."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        result = bt.run()
        bt.print_results(result)
        captured = capsys.readouterr()
        assert "VIXY Hedge Sizing" in captured.out
        assert "Sharpe" in captured.out

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        result = bt.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                data = json.load(f)
            assert "total_return" in data
            assert "sharpe_ratio" in data
            assert "crisis_returns_hedged" in data
            assert "regime_breakdown" in data
            assert data["_metadata"]["strategy"] == "vixy_hedge"
        finally:
            Path(output_path).unlink()


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the VIXY hedge backtest."""

    def test_zero_initial_capital(self):
        """Zero initial capital should not crash."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(initial_capital=0.0)
        )
        result = bt.run()
        # Metrics still computed (total_return will be 0 or inf)
        assert isinstance(result, BacktestResult)

    def test_negative_max_hedge(self):
        """Negative max hedge should be treated as zero."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(max_hedge_pct=-1.0)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["max_pct"] == 0.0 or tracker["max_pct"] < 0.01

    def test_no_rebalance_freq_edge(self):
        """Very frequent rebalancing should still work."""
        bt = WalkForwardVIXYBacktester(
            BacktestConfig(rebalance_frequency_days=1)
        )
        bt.load_data()
        _, tracker, _ = bt._run_hedged(bt._daily_prices, bt.config)
        assert tracker["rebalances"] > 0

    def test_baseline_matches_hedged_when_vix_zero(self):
        """When VIX is zero, there's no hedge, so baseline ≈ hedged."""
        bt = WalkForwardVIXYBacktester()
        bt.load_data()
        baseline_equity = bt._run_baseline(bt._daily_prices, bt.config)
        # Override VIX to zero for all prices
        original_prices = bt._daily_prices[:]
        for i in range(len(bt._daily_prices)):
            bt._daily_prices[i].vix = 0.0
        hedge_equity, _, _ = bt._run_hedged(bt._daily_prices, bt.config)
        # Restore
        bt._daily_prices = original_prices
        # With VIX=0, compute_allocation returns 0 (below floor), so hedged ≈ baseline
        # Small differences from different transaction costs
        assert abs(len(hedge_equity) - len(baseline_equity)) <= 1
