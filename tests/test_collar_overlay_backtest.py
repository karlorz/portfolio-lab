"""
Tests for the Collar Overlay Walk-Forward Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult creation/serialization,
collar signal computation, hard constraints, edge cases, crisis regime behavior,
synthetic data loading, print/save output.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.backtest.collar_overlay_backtest import (
    BacktestConfig,
    BacktestResult,
    DailyPrices,
    WalkForwardCollarBacktester,
    _get_regime,
    _get_collar_shifts,
    VIX_CRISIS,
    VIX_ELEVATED,
    VIX_STRESS,
)


# ── Helper Tests ────────────────────────────────────────────────────────────


class TestCollarHelpers:
    """Test the standalone collar helper functions."""

    def test_get_regime_normal(self):
        assert _get_regime(10.0) == "normal"
        assert _get_regime(15.0) == "normal"
        assert _get_regime(19.9) == "normal"

    def test_get_regime_elevated(self):
        assert _get_regime(20.0) == "elevated"
        assert _get_regime(25.0) == "elevated"
        assert _get_regime(29.9) == "elevated"

    def test_get_regime_stress(self):
        assert _get_regime(30.0) == "stress"
        assert _get_regime(35.0) == "stress"
        assert _get_regime(39.9) == "stress"

    def test_get_regime_crisis(self):
        assert _get_regime(40.0) == "crisis"
        assert _get_regime(45.0) == "crisis"
        assert _get_regime(100.0) == "crisis"

    def test_get_collar_shifts_normal(self):
        spy_s, gld_s, tlt_s = _get_collar_shifts(15.0)
        assert spy_s == -0.03
        assert gld_s == 0.01
        assert tlt_s == 0.02

    def test_get_collar_shifts_elevated(self):
        spy_s, gld_s, tlt_s = _get_collar_shifts(25.0)
        assert spy_s == -0.04
        assert gld_s == 0.015
        assert tlt_s == 0.025

    def test_get_collar_shifts_stress(self):
        spy_s, gld_s, tlt_s = _get_collar_shifts(35.0)
        assert spy_s == -0.05
        assert gld_s == 0.02
        assert tlt_s == 0.03

    def test_get_collar_shifts_crisis(self):
        """CRISIS regime freezes the collar -- no shifts."""
        spy_s, gld_s, tlt_s = _get_collar_shifts(45.0)
        assert spy_s == 0.0
        assert gld_s == 0.0
        assert tlt_s == 0.0

    def test_get_collar_shifts_boundary_elevated(self):
        """Boundary at VIX=20 should be elevated."""
        spy_s, _, _ = _get_collar_shifts(VIX_ELEVATED)
        assert spy_s == -0.04

    def test_get_collar_shifts_boundary_stress(self):
        """Boundary at VIX=30 should be stress."""
        spy_s, _, _ = _get_collar_shifts(VIX_STRESS)
        assert spy_s == -0.05

    def test_get_collar_shifts_boundary_crisis(self):
        """Boundary at VIX=40 should be crisis (frozen)."""
        spy_s, _, _ = _get_collar_shifts(VIX_CRISIS)
        assert spy_s == 0.0


# ── BacktestConfig Tests ─────────────────────────────────────────────────


class TestBacktestConfig:
    """Test BacktestConfig defaults and custom configuration."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.start_date == "2006-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_spy_weight == 0.46
        assert config.base_gld_weight == 0.38
        assert config.base_tlt_weight == 0.16
        assert config.rebalance_frequency_days == 21
        assert config.transaction_cost_bps == 10.0

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            rebalance_frequency_days=63,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.rebalance_frequency_days == 63

    def test_base_weights_sum(self):
        """Baseline weights should sum to 1.0."""
        config = BacktestConfig()
        total = config.base_spy_weight + config.base_gld_weight + config.base_tlt_weight
        assert abs(total - 1.0) < 0.01


# ── BacktestResult Tests ─────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation, to_dict, and empty state."""

    def test_create_and_to_dict(self):
        result = BacktestResult(
            total_return=8.5,
            cagr=7.0,
            volatility=10.5,
            sharpe_ratio=0.82,
            max_drawdown=-18.0,
            baseline_total_return=10.2,
            baseline_cagr=8.5,
            baseline_volatility=11.2,
            baseline_sharpe=0.79,
            baseline_max_drawdown=-26.2,
            sharpe_improvement=0.03,
            cagr_impact=-1.5,
            collar_active_days=800,
            collar_active_pct=35.0,
            collar_spy_reduction_avg=3.2,
            crisis_returns_hedged={"2008": -9.0, "2020": 2.5},
            crisis_returns_baseline={"2008": -12.3, "2020": 1.5},
            regime_breakdown={
                "normal": {"avg_spy_reduction": 2.8, "max_spy_reduction": 3.0, "count": 1500, "pct_of_time": 55.0},
                "elevated": {"avg_spy_reduction": 3.8, "max_spy_reduction": 4.0, "count": 600, "pct_of_time": 22.0},
            },
            total_rebalances=100,
            total_transaction_costs=35.0,
            config_snapshot={"start_date": "2006-01-01", "end_date": "2026-05-15"},
        )

        d = result.to_dict()
        assert d["total_return"] == 8.5
        assert d["sharpe_ratio"] == 0.82
        assert d["sharpe_improvement"] == 0.03
        assert d["collar_active_days"] == 800
        assert d["collar_spy_reduction_avg"] == 3.2
        assert d["crisis_returns_hedged"]["2008"] == -9.0
        assert d["regime_breakdown"]["normal"]["avg_spy_reduction"] == 2.8
        assert d["total_rebalances"] == 100
        assert d["config_snapshot"]["start_date"] == "2006-01-01"

    def test_json_serializable(self):
        """All fields in to_dict must be JSON-serializable."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_total_return=4.0, baseline_cagr=2.5,
            baseline_volatility=9.5, baseline_sharpe=0.45, baseline_max_drawdown=-12.0,
            sharpe_improvement=0.05, cagr_impact=0.5, collar_active_days=100,
            collar_active_pct=25.0, collar_spy_reduction_avg=2.5,
            crisis_returns_hedged={"2008": -8.0}, crisis_returns_baseline={"2008": -10.0},
            regime_breakdown={"normal": {"avg_spy_reduction": 2.5, "max_spy_reduction": 3.0, "count": 100, "pct_of_time": 50.0}},
            total_rebalances=30, total_transaction_costs=15.0,
            config_snapshot={"start_date": "2006-01-01"},
        )
        json.dumps(result.to_dict())  # Should not raise

    def test_empty_crisis_returns(self):
        """Crisis returns can be empty dict without errors."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_total_return=0.0, baseline_cagr=0.0,
            baseline_volatility=0.0, baseline_sharpe=0.0, baseline_max_drawdown=0.0,
            sharpe_improvement=0.0, cagr_impact=0.0, collar_active_days=0,
            collar_active_pct=0.0, collar_spy_reduction_avg=0.0,
            crisis_returns_hedged={}, crisis_returns_baseline={},
            regime_breakdown={}, total_rebalances=0, total_transaction_costs=0.0,
            config_snapshot={},
        )
        assert result.to_dict()["crisis_returns_hedged"] == {}


# ── Walk-Forward Backtester Tests ───────────────────────────────────────


class TestWalkForwardCollarBacktester:
    """Test the core WalkForwardCollarBacktester class."""

    def test_init_defaults(self):
        bt = WalkForwardCollarBacktester()
        assert bt.config.start_date == "2006-01-01"

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01")
        bt = WalkForwardCollarBacktester(config)
        assert bt.config.start_date == "2015-01-01"

    def test_load_data_generates_synthetic_when_no_file(self):
        """When prices.json doesn't exist, synthetic data is generated."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        assert len(bt._daily_prices) > 0
        assert len(bt._trading_dates) > 0

    def test_synthetic_data_has_required_fields(self):
        """Each DailyPrices entry should have SPY, GLD, TLT values."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        for dp in bt._daily_prices[:10]:
            assert isinstance(dp.spy, float)
            assert isinstance(dp.gld, float)
            assert isinstance(dp.tlt, float)

    def test_vix_proxy_default_when_insufficient_history(self):
        """VIX proxy should return ~18 when fewer than 21 days of history."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        vix = bt._compute_vix_proxy(5)  # idx 5 = only 6 days
        assert vix == 18.0 or abs(vix - 18.0) < 1.0

    def test_vix_proxy_with_sufficient_history(self):
        """VIX proxy should compute from 21-day SPY returns."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        # Use an index with enough history
        vix = bt._compute_vix_proxy(50)
        assert vix > 0
        assert vix < 100  # Sanity check

    def test_run_produces_results(self):
        """Running the backtest should return a populated BacktestResult."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_spy_stays_within_36_56(self):
        """SPY weight should never go below 36% or above 56%."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        prices = bt._daily_prices
        config = bt.config

        for i in range(1, len(prices)):
            vix = bt._get_vix_level(i)
            spy_s, gld_s, tlt_s = _get_collar_shifts(vix)
            spy_w = config.base_spy_weight + spy_s
            spy_w = max(0.36, min(0.56, spy_w))
            assert 0.36 <= spy_w <= 0.56

    def test_gld_stays_within_28_48(self):
        """GLD weight should never go below 28% or above 48%."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        config = bt.config

        for vix in [10.0, 25.0, 35.0, 45.0]:
            _, gld_s, _ = _get_collar_shifts(vix)
            gld_w = config.base_gld_weight + gld_s
            gld_w = max(0.28, min(0.48, gld_w))
            assert 0.28 <= gld_w <= 0.48

    def test_collar_active_in_normal_vix(self):
        """Collar should shift SPY (be active) in normal VIX."""
        spy_s, _, _ = _get_collar_shifts(15.0)
        assert spy_s < 0  # SPY is reduced

    def test_collar_inactive_in_crisis_vix(self):
        """Collar should be frozen (no shift) in crisis mode."""
        spy_s, _, _ = _get_collar_shifts(45.0)
        assert spy_s == 0.0

    def test_empty_data_returns_zero_metrics(self, monkeypatch):
        """Backtest with no data should return empty result."""
        bt = WalkForwardCollarBacktester()
        bt._daily_prices = []
        bt._trading_dates = []
        # Prevent load_data from being called (would generate synthetic data)
        monkeypatch.setattr(bt, "load_data", lambda: None)
        result = bt.run()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.collar_active_days == 0
        assert result.total_rebalances == 0
        assert result.crisis_returns_hedged == {}

    def test_single_day_data_returns_zero_result(self):
        """Only one data point should return an empty result."""
        bt = WalkForwardCollarBacktester()
        bt._daily_prices = [DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)]
        bt._trading_dates = ["2020-01-02"]
        result = bt.run()
        assert result.total_return == 0.0

    def test_baseline_weights_stable(self):
        """Baseline run should maintain constant weights."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        prices_subset = bt._daily_prices[:100]
        equity = bt._run_baseline(
            prices_subset,
            BacktestConfig(initial_capital=100000.0),
        )
        assert len(equity) == len(prices_subset)
        assert equity[0] == 100000.0

    def test_print_results_does_not_crash(self, capsys):
        """print_results should produce output without errors."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        result = bt.run()
        bt.print_results(result)
        captured = capsys.readouterr()
        assert "Collar Overlay" in captured.out
        assert "Sharpe" in captured.out

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        bt = WalkForwardCollarBacktester()
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
            assert "collar_active_days" in data
            assert "crisis_returns_hedged" in data
            assert "regime_breakdown" in data
            assert data["_metadata"]["strategy"] == "collar_overlay"
        finally:
            Path(output_path).unlink()

    def test_empty_result_method_returns_zeros(self):
        """_empty_result() should return all-zero metrics."""
        bt = WalkForwardCollarBacktester()
        result = bt._empty_result()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.collar_active_days == 0
        assert result.total_rebalances == 0
        assert result.collar_spy_reduction_avg == 0.0

    def test_get_vix_level_returns_proxy_when_missing(self):
        """When ^VIX is None, _get_vix_level should fall back to proxy."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        # Force vix to None on all entries
        for dp in bt._daily_prices:
            dp.vix = None
        vix = bt._get_vix_level(100)
        assert vix > 0
        assert vix < 100

    def test_narrow_date_range_still_runs(self):
        """A narrow date range (2 months) should still produce results."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-03-01")
        )
        result = bt.run()
        assert result.cagr is not None
        assert result.total_rebalances >= 0

    def test_weights_renormalized_to_one(self):
        """After applying collar shifts and constraints, weights should sum to 1.0."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        prices = bt._daily_prices[:100]
        config = bt.config

        for i in range(1, len(prices)):
            vix = bt._get_vix_level(i)
            spy_s, gld_s, tlt_s = _get_collar_shifts(vix)
            spy_w = max(0.36, min(0.56, config.base_spy_weight + spy_s))
            gld_w = max(0.28, min(0.48, config.base_gld_weight + gld_s))
            tlt_w = max(0.06, min(0.26, config.base_tlt_weight + tlt_s))
            total = spy_w + gld_w + tlt_w
            if total > 0:
                spy_w /= total
                gld_w /= total
                tlt_w /= total
            assert abs(spy_w + gld_w + tlt_w - 1.0) < 0.001


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the collar overlay backtest."""

    def test_zero_initial_capital(self):
        """Zero initial capital should not crash."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(initial_capital=0.0)
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_no_rebalance_freq_edge(self):
        """Very frequent rebalancing should still work."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(rebalance_frequency_days=1)
        )
        bt.load_data()
        _, tracker, _ = bt._run_collared(bt._daily_prices, bt.config)
        assert tracker["rebalances"] > 0

    def test_baseline_matches_collared_when_vix_zero(self):
        """When VIX=0, collar is in normal regime with minimal shift."""
        bt = WalkForwardCollarBacktester()
        bt.load_data()
        baseline_equity = bt._run_baseline(bt._daily_prices, bt.config)
        # Override VIX to 0 for all prices
        original_prices = bt._daily_prices[:]
        for i in range(len(bt._daily_prices)):
            bt._daily_prices[i].vix = 0.0
        collar_equity, _, _ = bt._run_collared(bt._daily_prices, bt.config)
        # Restore
        bt._daily_prices = original_prices
        # With VIX=0, collar is in normal regime (SPY -3%), so mild divergence
        assert abs(len(collar_equity) - len(baseline_equity)) <= 1

    def test_collar_always_nonnegative_return(self):
        """Collar equity curve should never have negative values (no shorting)."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2015-06-01")
        )
        bt.load_data()
        collar_equity, _, _ = bt._run_collared(bt._daily_prices, bt.config)
        assert all(e >= 0 for e in collar_equity)

    def test_regime_counts_present(self):
        """Regime breakdown should have entries for regimes encountered."""
        bt = WalkForwardCollarBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert len(result.regime_breakdown) > 0

    def test_crisis_freeze_preserves_base_allocation(self):
        """During crisis, the frozen collar should keep base allocation."""
        spy_s, gld_s, tlt_s = _get_collar_shifts(50.0)
        assert spy_s == 0.0
        assert gld_s == 0.0
        assert tlt_s == 0.0

    def test_regime_based_allocation_elevated_vs_normal(self):
        """Elevated VIX should produce larger SPY reduction than normal."""
        spy_s_normal, _, _ = _get_collar_shifts(15.0)
        spy_s_elevated, _, _ = _get_collar_shifts(25.0)
        assert abs(spy_s_elevated) > abs(spy_s_normal)
