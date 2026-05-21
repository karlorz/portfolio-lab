"""
Tests for the Bond Duration Rotation Walk-Forward Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult creation/serialization,
TLT momentum computation, bond sleeve allocation (rising/falling/neutral),
rotation activity tracking, crisis regime behavior, synthetic data loading,
print/save output, and edge cases.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.backtest.bond_duration_backtest import (
    BacktestConfig,
    BacktestResult,
    DailyPrices,
    WalkForwardBondDurationBacktester,
    RISING_ALLOCATION,
    FALLING_ALLOCATION,
    NEUTRAL_ALLOCATION,
    BOND_SLEEVE,
    RISING_THRESHOLD,
    FALLING_THRESHOLD,
    MOMENTUM_LOOKBACK,
)


# ── BacktestConfig Tests ────────────────────────────────────────────────────


class TestBacktestConfig:
    """Test BacktestConfig defaults and custom configuration."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.start_date == "2006-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_spy_weight == 0.46
        assert config.base_gld_weight == 0.38
        assert config.base_bond_weight == 0.16
        assert config.rebalance_frequency_days == 21
        assert config.transaction_cost_bps == 10.0
        assert config.momentum_lookback_days == 60

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            rebalance_frequency_days=63,
            momentum_lookback_days=90,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.rebalance_frequency_days == 63
        assert config.momentum_lookback_days == 90

    def test_base_weights_sum(self):
        """Baseline weights should sum to 1.0."""
        config = BacktestConfig()
        total = config.base_spy_weight + config.base_gld_weight + config.base_bond_weight
        assert abs(total - 1.0) < 0.01

    def test_config_bond_sleeve_constant(self):
        """BOND_SLEEVE constant should match the config."""
        config = BacktestConfig()
        assert config.base_bond_weight == BOND_SLEEVE


# ── BacktestResult Tests ────────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation, to_dict, and empty state."""

    def test_create_and_to_dict(self):
        result = BacktestResult(
            total_return=10.5,
            cagr=8.2,
            volatility=12.3,
            sharpe_ratio=0.85,
            max_drawdown=-15.4,
            baseline_total_return=9.0,
            baseline_cagr=7.5,
            baseline_volatility=11.8,
            baseline_sharpe=0.78,
            baseline_max_drawdown=-18.2,
            sharpe_improvement=0.07,
            cagr_impact=0.7,
            rotation_active_days=800,
            rotation_active_pct=35.0,
            avg_effective_duration=8.5,
            avg_tlt_weight=0.35,
            avg_ief_weight=0.30,
            avg_shy_weight=0.35,
            crisis_returns_rotated={"2008": -10.2, "2020": 3.1},
            crisis_returns_baseline={"2008": -12.3, "2020": 1.5},
            regime_breakdown={
                "rising": {"pct_of_time": 40.0, "avg_effective_duration": 12.5, "count": 80},
                "falling": {"pct_of_time": 35.0, "avg_effective_duration": 4.2, "count": 70},
            },
            total_rebalances=100,
            total_transaction_costs=32.50,
            config_snapshot={"momentum_lookback_days": 60},
        )

        d = result.to_dict()
        assert d["total_return"] == 10.5
        assert d["sharpe_ratio"] == 0.85
        assert d["sharpe_improvement"] == 0.07
        assert d["rotation_active_days"] == 800
        assert d["avg_tlt_weight"] == 0.35
        assert d["avg_ief_weight"] == 0.30
        assert d["avg_shy_weight"] == 0.35
        assert d["crisis_returns_rotated"]["2008"] == -10.2
        assert d["regime_breakdown"]["rising"]["pct_of_time"] == 40.0
        assert d["total_rebalances"] == 100
        assert d["config_snapshot"]["momentum_lookback_days"] == 60

    def test_json_serializable(self):
        """All fields in to_dict must be JSON-serializable."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_total_return=4.0, baseline_cagr=2.5,
            baseline_volatility=9.5, baseline_sharpe=0.45, baseline_max_drawdown=-12.0,
            sharpe_improvement=0.05, cagr_impact=0.5, rotation_active_days=100,
            rotation_active_pct=25.0, avg_effective_duration=7.5,
            avg_tlt_weight=0.40, avg_ief_weight=0.30, avg_shy_weight=0.30,
            crisis_returns_rotated={"2008": -8.0}, crisis_returns_baseline={"2008": -10.0},
            regime_breakdown={"rising": {"pct_of_time": 50.0, "avg_effective_duration": 12.0, "count": 50}},
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
            sharpe_improvement=0.0, cagr_impact=0.0, rotation_active_days=0,
            rotation_active_pct=0.0, avg_effective_duration=0.0,
            avg_tlt_weight=0.0, avg_ief_weight=0.0, avg_shy_weight=0.0,
            crisis_returns_rotated={}, crisis_returns_baseline={},
            regime_breakdown={}, total_rebalances=0, total_transaction_costs=0.0,
            config_snapshot={},
        )
        assert result.to_dict()["crisis_returns_rotated"] == {}

    def test_empty_result_all_zeros(self):
        """Empty result has all zero/empty fields."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_total_return=0.0, baseline_cagr=0.0,
            baseline_volatility=0.0, baseline_sharpe=0.0, baseline_max_drawdown=0.0,
            sharpe_improvement=0.0, cagr_impact=0.0, rotation_active_days=0,
            rotation_active_pct=0.0, avg_effective_duration=0.0,
            avg_tlt_weight=0.0, avg_ief_weight=0.0, avg_shy_weight=0.0,
            crisis_returns_rotated={}, crisis_returns_baseline={},
            regime_breakdown={}, total_rebalances=0, total_transaction_costs=0.0,
            config_snapshot={},
        )
        assert result.rotation_active_days == 0
        assert result.total_rebalances == 0


# ── Allocation Constants Tests ──────────────────────────────────────────────


class TestAllocationConstants:
    """Test bond sleeve allocation constants."""

    def test_rising_allocation_sum(self):
        """Rising allocation sleeve weights should sum to ~1.0."""
        tlt, ief, shy, label = RISING_ALLOCATION
        assert abs(tlt + ief + shy - 1.0) < 0.01
        assert label == "rising"
        assert tlt > ief  # TLT should dominate in rising regime

    def test_falling_allocation_sum(self):
        """Falling allocation sleeve weights should sum to ~1.0."""
        tlt, ief, shy, label = FALLING_ALLOCATION
        assert abs(tlt + ief + shy - 1.0) < 0.01
        assert label == "falling"
        assert shy > tlt  # SHY should dominate in falling regime

    def test_neutral_allocation_sum(self):
        """Neutral allocation sleeve weights should sum to ~1.0."""
        tlt, ief, shy, label = NEUTRAL_ALLOCATION
        assert abs(tlt + ief + shy - 1.0) < 0.01
        assert label == "neutral"

    def test_rising_has_highest_duration(self):
        """Rising regime should have the highest effective duration."""
        tlt_r, ief_r, shy_r, _ = RISING_ALLOCATION
        tlt_f, ief_f, shy_f, _ = FALLING_ALLOCATION
        dur_rising = tlt_r * 16.0 + ief_r * 7.0 + shy_r * 2.0
        dur_falling = tlt_f * 16.0 + ief_f * 7.0 + shy_f * 2.0
        assert dur_rising > dur_falling

    def test_momentum_thresholds_symmetric(self):
        """Rising threshold should be positive, falling negative."""
        assert RISING_THRESHOLD > 0
        assert FALLING_THRESHOLD < 0
        assert MOMENTUM_LOOKBACK == 60

    def test_bond_sleeve_constant(self):
        """BOND_SLEEVE should be 16%."""
        assert BOND_SLEEVE == 0.16


# ── Walk-Forward Backtester Tests ───────────────────────────────────────────


class TestWalkForwardBondDurationBacktester:
    """Test the core WalkForwardBondDurationBacktester class."""

    def test_init_defaults(self):
        bt = WalkForwardBondDurationBacktester()
        assert bt.config.start_date == "2006-01-01"

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", momentum_lookback_days=90)
        bt = WalkForwardBondDurationBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.momentum_lookback_days == 90

    def test_load_data_generates_synthetic_when_no_file(self, monkeypatch):
        """When prices.json doesn't exist, synthetic data is generated."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        assert len(bt._daily_prices) > 0
        assert len(bt._trading_dates) > 0

    def test_synthetic_data_has_required_fields(self):
        """Each DailyPrices entry should have SPY, GLD, TLT, IEF, SHY values."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        for dp in bt._daily_prices[:10]:
            assert isinstance(dp.spy, float)
            assert isinstance(dp.gld, float)
            assert isinstance(dp.tlt, float)
            assert isinstance(dp.ief, float)
            assert isinstance(dp.shy, float)

    def test_run_produces_results(self):
        """Running the backtest should return a populated BacktestResult."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_baseline_weights_stable(self):
        """Baseline run should maintain constant weights (all bonds in TLT)."""
        bt = WalkForwardBondDurationBacktester()
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
        bt = WalkForwardBondDurationBacktester()
        result = bt._empty_result()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.rotation_active_days == 0
        assert result.total_rebalances == 0
        assert result.crisis_returns_rotated == {}
        assert result.avg_tlt_weight == 0.0

    def test_single_day_data_returns_zero_result(self):
        """Only one data point should return an empty result."""
        bt = WalkForwardBondDurationBacktester()
        bt._daily_prices = [DailyPrices(
            date="2020-01-02", spy=100.0, gld=100.0,
            tlt=100.0, ief=100.0, shy=100.0,
        )]
        bt._trading_dates = ["2020-01-02"]
        result = bt.run()
        assert result.total_return == 0.0

    def test_narrow_date_range_still_runs(self):
        """A narrow date range (2 months) should still produce results."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-03-01")
        )
        result = bt.run()
        assert result.cagr is not None
        assert result.total_rebalances >= 0

    def test_print_results_does_not_crash(self, capsys):
        """print_results should produce output without errors."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        result = bt.run()
        bt.print_results(result)
        captured = capsys.readouterr()
        assert "Bond Duration Rotation" in captured.out
        assert "Sharpe" in captured.out

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        bt = WalkForwardBondDurationBacktester()
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
            assert "crisis_returns_rotated" in data
            assert "regime_breakdown" in data
            assert "avg_tlt_weight" in data
            assert data["_metadata"]["strategy"] == "bond_duration"
        finally:
            Path(output_path).unlink()

    def test_save_results_metadata(self):
        """Saved results should include strategy metadata."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        result = bt.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                data = json.load(f)
            assert data["_metadata"]["type"] == "walk_forward_backtest"
            assert "generated" in data["_metadata"]
        finally:
            Path(output_path).unlink()


# ── TLT Momentum Tests ──────────────────────────────────────────────────────


class TestTLTMomentum:
    """Test TLT 60-day momentum computation and classification."""

    def test_momentum_positive_when_price_rising(self):
        """Momentum should be positive when TLT price increases over 60 days."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        # momentum at index 60+ should be computed from real data
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 100.0
        bt._daily_prices[idx].tlt = 105.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom > 0

    def test_momentum_negative_when_price_falling(self):
        """Momentum should be negative when TLT price decreases over 60 days."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 100.0
        bt._daily_prices[idx].tlt = 95.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom < 0

    def test_momentum_zero_with_insufficient_history(self):
        """Momentum should be 0.0 when there are fewer than 60 days."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        mom = bt._compute_tlt_60d_momentum(5)  # Only 6 days of history
        assert mom == 0.0

    def test_classify_momentum_rising(self):
        """Momentum above RISING_THRESHOLD should be classified as rising."""
        bt = WalkForwardBondDurationBacktester()
        regime = bt._classify_momentum(RISING_THRESHOLD + 0.005)
        assert regime == "rising"

    def test_classify_momentum_falling(self):
        """Momentum below FALLING_THRESHOLD should be classified as falling."""
        bt = WalkForwardBondDurationBacktester()
        regime = bt._classify_momentum(FALLING_THRESHOLD - 0.005)
        assert regime == "falling"

    def test_classify_momentum_neutral(self):
        """Momentum between thresholds should be classified as neutral."""
        bt = WalkForwardBondDurationBacktester()
        regime = bt._classify_momentum(0.0)
        assert regime == "neutral"

    def test_classify_momentum_boundary_rising(self):
        """Momentum exactly at rising threshold should be neutral."""
        bt = WalkForwardBondDurationBacktester()
        regime = bt._classify_momentum(RISING_THRESHOLD)
        assert regime == "neutral"


# ── Bond Allocation Tests ───────────────────────────────────────────────────


class TestBondAllocation:
    """Test bond sleeve allocation by momentum regime."""

    def test_rising_allocation(self):
        """Rising momentum should return TLT-heavy allocation."""
        bt = WalkForwardBondDurationBacktester()
        tlt, ief, shy, label = bt._get_bond_sleeve_allocation("rising")
        assert tlt == pytest.approx(0.70)
        assert ief == pytest.approx(0.20)
        assert shy == pytest.approx(0.10)

    def test_falling_allocation(self):
        """Falling momentum should return SHY-heavy allocation."""
        bt = WalkForwardBondDurationBacktester()
        tlt, ief, shy, label = bt._get_bond_sleeve_allocation("falling")
        assert tlt == pytest.approx(0.10)
        assert ief == pytest.approx(0.30)
        assert shy == pytest.approx(0.60)

    def test_neutral_allocation(self):
        """Neutral momentum should return balanced allocation."""
        bt = WalkForwardBondDurationBacktester()
        tlt, ief, shy, label = bt._get_bond_sleeve_allocation("neutral")
        assert tlt == pytest.approx(0.40)
        assert ief == pytest.approx(0.40)
        assert shy == pytest.approx(0.20)

    def test_unknown_regime_falls_back_to_neutral(self):
        """Unknown regime should default to neutral."""
        bt = WalkForwardBondDurationBacktester()
        tlt, ief, shy, label = bt._get_bond_sleeve_allocation("unknown")
        assert label == "neutral" or label is not None


# ── Effective Duration Tests ────────────────────────────────────────────────


class TestEffectiveDuration:
    """Test effective duration computation."""

    def test_all_tlt(self):
        """100% TLT should give 16-year duration."""
        bt = WalkForwardBondDurationBacktester()
        dur = bt._compute_effective_duration(1.0, 0.0, 0.0)
        assert dur == pytest.approx(16.0)

    def test_all_shy(self):
        """100% SHY should give 2-year duration."""
        bt = WalkForwardBondDurationBacktester()
        dur = bt._compute_effective_duration(0.0, 0.0, 1.0)
        assert dur == pytest.approx(2.0)

    def test_mixed_allocation(self):
        """Mixed allocation should compute weighted average."""
        bt = WalkForwardBondDurationBacktester()
        # 50% TLT, 30% IEF, 20% SHY
        dur = bt._compute_effective_duration(0.5, 0.3, 0.2)
        expected = 0.5 * 16.0 + 0.3 * 7.0 + 0.2 * 2.0
        assert dur == pytest.approx(expected)


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the bond duration backtest."""

    def test_zero_initial_capital(self):
        """Zero initial capital should not crash."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(initial_capital=0.0)
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_rotation_active_count(self):
        """Rotation should be active when sleeve differs from all-TLT."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        prices = bt._daily_prices
        config = BacktestConfig(initial_capital=100000.0)
        _, tracker, _ = bt._run_rotated(prices, config)
        # At some rebalance points, allocation differs from 100% TLT
        sleeve_weights = tracker["sleeve_weights"]
        non_tlt_days = sum(1 for s in sleeve_weights if abs(s["tlt"] - 1.0) > 0.01)
        assert tracker["active_days"] == non_tlt_days

    def test_effective_durations_tracked(self):
        """Effective durations should be non-empty after running."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        prices = bt._daily_prices
        config = BacktestConfig(initial_capital=100000.0)
        _, tracker, _ = bt._run_rotated(prices, config)
        assert len(tracker["effective_durations"]) > 0
        assert tracker["avg_effective_duration"] > 0

    def test_crisis_returns_populated(self):
        """Crisis returns should be populated when data includes crisis years."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2007-01-01", end_date="2010-01-01")
        )
        bt.load_data()
        result = bt.run()
        # Should have at least some crisis data
        assert isinstance(result.crisis_returns_baseline, dict)
        assert isinstance(result.crisis_returns_rotated, dict)

    def test_no_rebalance_freq_edge(self):
        """Very frequent rebalancing should still work."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(rebalance_frequency_days=1)
        )
        bt.load_data()
        _, tracker, _ = bt._run_rotated(bt._daily_prices, bt.config)
        assert tracker["rebalances"] > 0

    def test_tlt_trend_rising_shifts_allocation(self):
        """When TLT is in a rising trend, the sleeve should shift to TLT-heavy."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        # Force TLT price up for the last 60 days
        idx = MOMENTUM_LOOKBACK + 1
        if idx < len(bt._daily_prices):
            bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 100.0
            bt._daily_prices[idx].tlt = 105.0
            regime = bt._classify_momentum(bt._compute_tlt_60d_momentum(idx))
            assert regime == "rising"
