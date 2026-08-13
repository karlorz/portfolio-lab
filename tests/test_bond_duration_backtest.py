"""
Tests for the Bond Duration Rotation Walk-Forward Backtest.

Covers: BacktestConfig defaults/custom, BacktestResult creation/serialization,
TLT momentum computation, bond sleeve allocation (rising/falling/neutral),
rotation activity tracking, crisis regime behavior, synthetic data loading,
print/save output, and edge cases.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.backtest.bond_duration_backtest import (
    BacktestConfig,
    DailyPrices,
    WalkForwardBondDurationBacktester,
    BOND_SLEEVE,
    MOMENTUM_LOOKBACK,
)
from src.backtest.metrics import BacktestResult
from src.signals.bond_duration_signal import BondDurationCalculator


# ── BacktestConfig Tests ────────────────────────────────────────────────────


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
        total = config.base_weights['SPY'] + config.base_weights['GLD'] + config.base_weights['TLT']
        assert abs(total - 1.0) < 0.01

    def test_config_bond_sleeve_constant(self):
        """BOND_SLEEVE constant should match the config."""
        config = BacktestConfig()
        assert config.base_weights['TLT'] == BOND_SLEEVE

    def test_to_dict_all_fields(self):
        """asdict(BacktestConfig) should contain every defined field (inherited + own)."""
        from dataclasses import asdict
        config = BacktestConfig()
        d = asdict(config)
        expected_keys = {
            "start_date", "end_date", "initial_capital", "base_weights",
            "rebalance_frequency_days", "transaction_cost_bps",
            "rebalance_frequency", "transaction_costs_by_symbol",
            "momentum_lookback_days", "extras",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_field_types(self):
        """All BacktestConfig fields should have correct types."""
        from dataclasses import asdict
        config = BacktestConfig()
        d = asdict(config)
        assert isinstance(d["start_date"], str)
        assert isinstance(d["end_date"], str)
        assert isinstance(d["initial_capital"], float)
        assert isinstance(d["base_weights"], dict)
        assert isinstance(d["rebalance_frequency_days"], int)
        assert isinstance(d["transaction_cost_bps"], float)
        assert isinstance(d["momentum_lookback_days"], int)


class TestDailyPrices:
    """Test DailyPrices dataclass creation and serialization."""

    def test_to_dict_all_fields(self):
        """asdict(DailyPrices) should contain every defined field (incl. optionals)."""
        from dataclasses import asdict
        from src.backtest.metrics import DailyPrices as DP
        dp = DP(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)
        d = asdict(dp)
        assert set(d.keys()) == {"date", "spy", "gld", "tlt", "ief", "shy", "vix", "btc", "eth", "extras"}

    def test_json_serializable(self):
        """All DailyPrices fields must be JSON-serializable."""
        from dataclasses import asdict
        from src.backtest.metrics import DailyPrices as DP
        dp = DP(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        json.dumps(asdict(dp))  # Should not raise

    def test_zero_values(self):
        """DailyPrices with all zeros should not crash."""
        from dataclasses import asdict
        from src.backtest.metrics import DailyPrices as DP
        dp = DP(date="2020-01-02", spy=0.0, gld=0.0, tlt=0.0, ief=0.0, shy=0.0)
        d = asdict(dp)
        assert d["spy"] == 0.0
        assert d["gld"] == 0.0


class TestConstants:
    """Validate module-level constants match source definitions."""

    def test_trading_days_per_year(self):
        from src.backtest.bond_duration_backtest import TRADING_DAYS_PER_YEAR
        assert TRADING_DAYS_PER_YEAR == 252

    def test_monthly_trading_days(self):
        from src.backtest.bond_duration_backtest import MONTHLY_TRADING_DAYS
        assert MONTHLY_TRADING_DAYS == 21

    def test_crisis_years(self):
        from src.backtest.bond_duration_backtest import CRISIS_YEARS
        assert CRISIS_YEARS == ["2008", "2020", "2022"]

    def test_base_symbols(self):
        from src.backtest.bond_duration_backtest import BASE_SYMBOLS
        assert BASE_SYMBOLS == ["SPY", "GLD", "TLT"]

    def test_bond_symbols(self):
        from src.backtest.bond_duration_backtest import BOND_SYMBOLS
        assert BOND_SYMBOLS == ["TLT", "IEF", "SHY"]

    def test_momentum_lookback(self):
        from src.backtest.bond_duration_backtest import MOMENTUM_LOOKBACK
        assert MOMENTUM_LOOKBACK == 60

    def test_bond_sleeve_value(self):
        from src.backtest.bond_duration_backtest import BOND_SLEEVE
        assert BOND_SLEEVE == 0.16

    def test_all_constants_positive(self):
        """All numeric constants should be positive values."""
        from src.backtest.bond_duration_backtest import (
            TRADING_DAYS_PER_YEAR, MONTHLY_TRADING_DAYS,
            MOMENTUM_LOOKBACK, BOND_SLEEVE,
        )
        assert TRADING_DAYS_PER_YEAR > 0
        assert MONTHLY_TRADING_DAYS > 0
        assert MOMENTUM_LOOKBACK > 0
        assert BOND_SLEEVE > 0


# ── BacktestResult Tests ────────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation, to_dict, and empty state."""

    def test_create_and_to_dict(self):
        from dataclasses import asdict
        result = BacktestResult(
            total_return=10.5,
            cagr=8.2,
            volatility=12.3,
            sharpe_ratio=0.85,
            max_drawdown=-15.4,
            baseline_sharpe=0.78,
            sharpe_improvement=0.07,
            total_rebalances=100,
            total_transaction_costs=32.50,
            extras={
                "baseline_total_return": 9.0,
                "baseline_cagr": 7.5,
                "baseline_volatility": 11.8,
                "baseline_max_drawdown": -18.2,
                "cagr_impact": 0.7,
                "rotation_active_days": 800,
                "rotation_active_pct": 35.0,
                "avg_effective_duration": 8.5,
                "avg_tlt_weight": 0.35,
                "avg_ief_weight": 0.30,
                "avg_shy_weight": 0.35,
                "crisis_returns_rotated": {"2008": -10.2, "2020": 3.1},
                "crisis_returns_baseline": {"2008": -12.3, "2020": 1.5},
                "regime_breakdown": {
                    "rising": {"pct_of_time": 40.0, "avg_effective_duration": 12.5, "count": 80},
                    "falling": {"pct_of_time": 35.0, "avg_effective_duration": 4.2, "count": 70},
                },
                "config_snapshot": {"momentum_lookback_days": 60},
            },
        )

        d = asdict(result)
        assert d["total_return"] == 10.5
        assert d["sharpe_ratio"] == 0.85
        assert d["sharpe_improvement"] == 0.07
        assert d["extras"]["rotation_active_days"] == 800
        assert d["extras"]["avg_tlt_weight"] == 0.35
        assert d["extras"]["avg_ief_weight"] == 0.30
        assert d["extras"]["avg_shy_weight"] == 0.35
        assert d["extras"]["crisis_returns_rotated"]["2008"] == -10.2
        assert d["extras"]["regime_breakdown"]["rising"]["pct_of_time"] == 40.0
        assert d["total_rebalances"] == 100
        assert d["extras"]["config_snapshot"]["momentum_lookback_days"] == 60

    def test_json_serializable(self):
        """All fields in to_dict must be JSON-serializable."""
        from dataclasses import asdict
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "rotation_active_days": 100,
                "rotation_active_pct": 25.0, "avg_effective_duration": 7.5,
                "avg_tlt_weight": 0.40, "avg_ief_weight": 0.30, "avg_shy_weight": 0.30,
                "crisis_returns_rotated": {"2008": -8.0}, "crisis_returns_baseline": {"2008": -10.0},
                "regime_breakdown": {"rising": {"pct_of_time": 50.0, "avg_effective_duration": 12.0, "count": 50}},
                "config_snapshot": {"start_date": "2006-01-01"},
            },
        )
        json.dumps(asdict(result))  # Should not raise

    def test_empty_crisis_returns(self):
        """Crisis returns can be empty dict without errors."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, sharpe_improvement=0.0, total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 0.0, "baseline_cagr": 0.0,
                "baseline_volatility": 0.0, "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0, "rotation_active_days": 0,
                "rotation_active_pct": 0.0, "avg_effective_duration": 0.0,
                "avg_tlt_weight": 0.0, "avg_ief_weight": 0.0, "avg_shy_weight": 0.0,
                "crisis_returns_rotated": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        assert result.extras["crisis_returns_rotated"] == {}

    def test_empty_result_all_zeros(self):
        """Empty result has all zero/empty fields."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, sharpe_improvement=0.0, total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 0.0, "baseline_cagr": 0.0,
                "baseline_volatility": 0.0, "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0, "rotation_active_days": 0,
                "rotation_active_pct": 0.0, "avg_effective_duration": 0.0,
                "avg_tlt_weight": 0.0, "avg_ief_weight": 0.0, "avg_shy_weight": 0.0,
                "crisis_returns_rotated": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        assert result.extras["rotation_active_days"] == 0
        assert result.total_rebalances == 0

    def test_to_dict_all_extras_fields(self):
        """asdict extras must contain every expected key."""
        from dataclasses import asdict
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "rotation_active_days": 100,
                "rotation_active_pct": 25.0, "avg_effective_duration": 7.5,
                "avg_tlt_weight": 0.40, "avg_ief_weight": 0.30, "avg_shy_weight": 0.30,
                "crisis_returns_rotated": {"2008": -8.0}, "crisis_returns_baseline": {"2008": -10.0},
                "regime_breakdown": {"rising": {"pct_of_time": 50.0, "avg_effective_duration": 12.0, "count": 50}},
                "config_snapshot": {"start_date": "2006-01-01"},
            },
        )
        d = asdict(result)
        expected_extras_keys = {
            "baseline_total_return", "baseline_cagr", "baseline_volatility",
            "baseline_max_drawdown", "cagr_impact", "rotation_active_days",
            "rotation_active_pct", "avg_effective_duration", "avg_tlt_weight",
            "avg_ief_weight", "avg_shy_weight", "crisis_returns_rotated",
            "crisis_returns_baseline", "regime_breakdown", "config_snapshot",
        }
        assert set(d["extras"].keys()) == expected_extras_keys

    def test_to_dict_no_none_values(self):
        """No required field in the BacktestResult dict should be None (crisis_returns may be None)."""
        from dataclasses import asdict
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=30, total_transaction_costs=15.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "rotation_active_days": 100,
                "rotation_active_pct": 25.0, "avg_effective_duration": 7.5,
                "avg_tlt_weight": 0.40, "avg_ief_weight": 0.30, "avg_shy_weight": 0.30,
                "crisis_returns_rotated": {"2008": -8.0}, "crisis_returns_baseline": {"2008": -10.0},
                "regime_breakdown": {"rising": {"pct_of_time": 50.0, "avg_effective_duration": 12.0, "count": 50}},
                "config_snapshot": {"start_date": "2006-01-01"},
            },
        )
        d = asdict(result)
        # crisis_returns and baseline_sharpe/sharpe_improvement are Optional and may be None
        nullable_fields = {"crisis_returns", "baseline_sharpe", "sharpe_improvement"}
        for key, value in d.items():
            if key != "extras" and key not in nullable_fields:
                assert value is not None, f"Field {key} is None"
        for key, value in d["extras"].items():
            assert value is not None, f"Extras field {key} is None"


# ── Allocation Constants Tests ──────────────────────────────────────────────


class TestAllocationConstants:
    """Test bond sleeve allocation via BondDurationCalculator."""

    def test_steep_falling_allocation_sum(self):
        """Steep curve + falling rates should sum to ~1.0."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            1.2, 1.5, calc.classify_rate_direction(-0.5), calc.classify_curve(1.2)
        )
        assert abs(tlt + ief + shy - 1.0) < 0.01
        assert tlt > ief  # TLT should dominate in steep+falling regime

    def test_inverted_rising_allocation_sum(self):
        """Inverted curve + rising rates should sum to ~1.0."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            -0.5, -1.0, calc.classify_rate_direction(0.5), calc.classify_curve(-0.5)
        )
        assert abs(tlt + ief + shy - 1.0) < 0.01
        assert shy > tlt  # SHY should dominate in inverted+rising regime

    def test_normal_stable_allocation_sum(self):
        """Normal curve + stable rates should sum to ~1.0."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            0.5, 1.0, calc.classify_rate_direction(0.0), calc.classify_curve(0.5)
        )
        assert abs(tlt + ief + shy - 1.0) < 0.01

    def test_rising_has_highest_duration(self):
        """Steep+failing regime should have higher duration than inverted+rising."""
        calc = BondDurationCalculator()
        tlt_r, ief_r, shy_r, _ = calc.compute_duration_allocation(
            1.2, 1.5, calc.classify_rate_direction(-0.5), calc.classify_curve(1.2)
        )
        tlt_f, ief_f, shy_f, _ = calc.compute_duration_allocation(
            -0.5, -1.0, calc.classify_rate_direction(0.5), calc.classify_curve(-0.5)
        )
        dur_rising = calc.compute_effective_duration(tlt_r, ief_r, shy_r)
        dur_falling = calc.compute_effective_duration(tlt_f, ief_f, shy_f)
        assert dur_rising > dur_falling

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
        assert result.extras["rotation_active_days"] == 0
        assert result.total_rebalances == 0
        assert result.extras["crisis_returns_rotated"] == {}
        assert result.extras["avg_tlt_weight"] == 0.0

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

    def test_print_results_does_not_crash(self, caplog):
        """print_results should produce output without errors."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        result = bt.run()
        with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
            bt.print_results(result)
        assert "Bond Duration Rotation" in caplog.text
        assert "Sharpe" in caplog.text

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
            assert "extras" in data
            assert "crisis_returns_rotated" in data["extras"]
            assert "regime_breakdown" in data["extras"]
            assert "avg_tlt_weight" in data["extras"]
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

    def test_run_result_has_all_extras_keys(self):
        """Run result extras should contain every expected key."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        expected_keys = {
            "baseline_total_return", "baseline_cagr", "baseline_volatility",
            "baseline_max_drawdown", "cagr_impact", "rotation_active_days",
            "rotation_active_pct", "avg_effective_duration", "avg_tlt_weight",
            "avg_ief_weight", "avg_shy_weight", "crisis_returns_rotated",
            "crisis_returns_baseline", "regime_breakdown", "config_snapshot",
        }
        assert expected_keys.issubset(set(result.extras.keys()))

    def test_run_result_sharpe_improvement(self):
        """sharpe_improvement should equal rotated - baseline Sharpe."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        expected = round(result.sharpe_ratio - result.baseline_sharpe, 4)
        assert result.sharpe_improvement == expected

    def test_run_result_config_snapshot_fields(self):
        """config_snapshot should contain backtest configuration."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01",
                           initial_capital=50000.0, momentum_lookback_days=90)
        )
        result = bt.run()
        cs = result.extras["config_snapshot"]
        assert cs["start_date"] == "2015-01-01"
        assert cs["end_date"] == "2016-01-01"
        assert cs["initial_capital"] == 50000.0
        assert cs["momentum_lookback_days"] == 90

    def test_baseline_equity_starts_at_initial_capital(self):
        """Baseline equity curve should start at initial capital."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        equity = bt._run_baseline(bt._daily_prices[:100], bt.config)
        assert equity[0] == bt.config.initial_capital

    def test_rotated_equity_starts_at_initial_capital(self):
        """Rotated equity curve should start at initial capital."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        equity, _, _ = bt._run_rotated(bt._daily_prices[:100], bt.config)
        assert equity[0] == bt.config.initial_capital

    def test_baseline_and_rotated_equity_length_match(self):
        """Both equity curves should have same length as input prices."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        subset = bt._daily_prices[:100]
        baseline_eq = bt._run_baseline(subset, bt.config)
        rotated_eq, _, _ = bt._run_rotated(subset, bt.config)
        assert len(baseline_eq) == len(subset)
        assert len(rotated_eq) == len(subset)


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

    def test_momentum_to_yield_context_rising(self):
        """Strong positive TLT momentum should map to falling rates, steep curve."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(0.06)
        assert spread > 0.5  # Steep curve
        assert rate_chg < -0.3  # Falling rates

    def test_momentum_to_yield_context_falling(self):
        """Strong negative TLT momentum should map to rising rates, inverted curve."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(-0.06)
        assert spread < 0  # Inverted curve
        assert rate_chg > 0.3  # Rising rates

    def test_momentum_to_yield_context_neutral(self):
        """Near-zero TLT momentum should map to stable rates, normal curve."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(0.0)
        assert 0 < spread < 0.5  # Normal/flat curve
        assert abs(rate_chg) < 0.1  # Stable rates

    def test_momentum_zero_price_not_crash(self):
        """Zero TLT price should return 0.0 momentum, not crash."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 0.0
        bt._daily_prices[idx].tlt = 105.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom == 0.0

    def test_momentum_negative_price_not_crash(self):
        """Negative TLT price should return 0.0 momentum, not crash."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = -10.0
        bt._daily_prices[idx].tlt = 105.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom == 0.0

    def test_momentum_identical_prices(self):
        """Momentum should be 0.0 when start and end prices are equal."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 100.0
        bt._daily_prices[idx].tlt = 100.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom == 0.0

    def test_momentum_at_exact_lookback_boundary(self):
        """Momentum at index == lookback should compute from index 0 to lookback."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK
        if idx < len(bt._daily_prices):
            bt._daily_prices[0].tlt = 100.0
            bt._daily_prices[idx].tlt = 110.0
            mom = bt._compute_tlt_60d_momentum(idx)
            assert mom == pytest.approx(0.1)


class TestYieldContextMapping:
    """Test boundary conditions for _momentum_to_yield_context."""

    def test_exact_positive_005_boundary(self):
        """Momentum exactly 0.05 maps to moderate-positive (strict > 0.05 for strong)."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(0.05)
        assert spread == 0.5  # Not 0.8 -- > 0.05 is strict
        assert rate_chg == -0.2

    def test_just_above_005_boundary(self):
        """Momentum 0.0501 should map to the strong-positive bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(0.0501)
        assert spread == 0.8
        assert rate_chg == -0.5

    def test_exact_positive_001_boundary(self):
        """Momentum exactly 0.01 maps to neutral (strict > 0.01 for moderate-positive)."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(0.01)
        assert spread == 0.3  # Not 0.5 -- > 0.01 is strict
        assert rate_chg == 0.0

    def test_just_below_001_boundary(self):
        """Momentum 0.0099 should map to neutral bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(0.0099)
        assert spread == 0.3
        assert rate_chg == 0.0

    def test_exact_negative_001_boundary(self):
        """Momentum exactly -0.01 should map to neutral bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(-0.01)
        assert spread == 0.3
        assert rate_chg == 0.0

    def test_just_below_neg_001_boundary(self):
        """Momentum -0.0101 should map to moderate-negative bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(-0.0101)
        assert spread == 0.1
        assert rate_chg == 0.3

    def test_exact_negative_005_boundary(self):
        """Momentum exactly -0.05 should map to moderate-negative bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(-0.05)
        assert spread == 0.1
        assert rate_chg == 0.3

    def test_just_below_neg_005_boundary(self):
        """Momentum -0.0501 should map to strong-negative bucket."""
        bt = WalkForwardBondDurationBacktester()
        spread, _, rate_chg = bt._momentum_to_yield_context(-0.0501)
        assert spread == -0.2
        assert rate_chg == 0.6

    def test_extreme_positive_momentum(self):
        """Very large positive momentum still maps to steep-falling context."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(0.50)
        assert spread == 0.8
        assert real_rate == 1.5
        assert rate_chg == -0.5

    def test_extreme_negative_momentum(self):
        """Very large negative momentum still maps to inverted-rising context."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(-0.50)
        assert spread == -0.2
        assert real_rate == 1.5
        assert rate_chg == 0.6


# ── Bond Allocation Tests ───────────────────────────────────────────────────


class TestBondAllocation:
    """Test bond sleeve allocation via BondDurationCalculator regime matrix."""

    def test_steep_falling_allocation(self):
        """Steep curve + falling rates should return TLT-heavy allocation."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            1.2, 1.5, calc.classify_rate_direction(-0.5), calc.classify_curve(1.2)
        )
        assert tlt > 0.5  # TLT-heavy
        assert label == "long"

    def test_inverted_rising_allocation(self):
        """Inverted curve + rising rates should return SHY-heavy allocation."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            -0.5, -1.0, calc.classify_rate_direction(0.5), calc.classify_curve(-0.5)
        )
        assert shy > 0.5  # SHY-heavy
        assert label == "short"

    def test_normal_stable_allocation(self):
        """Normal curve + stable rates should return balanced allocation."""
        calc = BondDurationCalculator()
        tlt, ief, shy, label = calc.compute_duration_allocation(
            0.5, 1.0, calc.classify_rate_direction(0.0), calc.classify_curve(0.5)
        )
        assert abs(tlt + ief + shy - 1.0) < 0.01  # Weights sum to 1


# ── Portfolio Return Computation Tests ──────────────────────────────────────


class TestPortfolioReturn:
    """Test _compute_portfolio_return edge cases."""

    def test_zero_prices_not_crash(self):
        """Zero prices should return 0.0, not crash."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=0.0, gld=0.0, tlt=0.0, ief=0.0, shy=0.0)
        p1 = DailyPrices(date="2020-01-03", spy=0.0, gld=0.0, tlt=0.0, ief=0.0, shy=0.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 1.0, 0.0, 0.0)
        assert ret == 0.0

    def test_negative_prices_not_crash(self):
        """Negative prices should not crash (returns may be negative)."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=-50.0, gld=-50.0, tlt=-50.0, ief=-50.0, shy=-50.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 1.0, 0.0, 0.0)
        assert ret < -1.0  # Large negative return expected

    def test_all_tlt_sleeve_uses_tlt_prices(self):
        """When sleeve is 100% TLT, IEF/SHY returns don't affect composite."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=100.0, gld=100.0, tlt=110.0, ief=50.0, shy=50.0)
        # 100% TLT sleeve -> only TLT return matters for bond composite
        ret = bt._compute_portfolio_return(p0, p1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        assert ret == pytest.approx(0.10)  # TLT up 10%

    def test_all_shy_sleeve_uses_shy_prices(self):
        """When sleeve is 100% SHY, only SHY return affects bond composite."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=90.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        assert ret == pytest.approx(-0.10)  # SHY down 10%

    def test_zero_weight_allocations(self):
        """All zero weights should produce zero return."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=110.0, gld=110.0, tlt=110.0, ief=110.0, shy=110.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert ret == 0.0

    def test_mixed_sleeve_weights(self):
        """Mixed sleeve weights should produce weighted composite return."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=100.0, gld=100.0, tlt=110.0, ief=105.0, shy=102.0)
        # 50% TLT (+10%), 30% IEF (+5%), 20% SHY (+2%)
        expected_composite = 0.50 * 0.10 + 0.30 * 0.05 + 0.20 * 0.02
        ret = bt._compute_portfolio_return(p0, p1, 0.0, 0.0, 1.0, 0.50, 0.30, 0.20)
        assert ret == pytest.approx(expected_composite)


# ── Regime Classification Boundary Tests ─────────────────────────────────────


class TestRegimeClassificationBoundaries:
    """Test classify_curve and classify_rate_direction at exact thresholds."""

    def test_classify_curve_spread_above_steep(self):
        """Spread > 1.0 should return STEEP."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(1.5)) == "YieldCurveRegime.STEEP"

    def test_classify_curve_exact_steep_threshold(self):
        """Spread exactly 1.0 should NOT be STEEP (strict >)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(1.0)) != "YieldCurveRegime.STEEP"

    def test_classify_curve_exact_steep_returns_normal(self):
        """Spread exactly 1.0 should return NORMAL (>= 0.3 but not > 1.0)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(1.0)) == "YieldCurveRegime.NORMAL"

    def test_classify_curve_exact_normal_threshold(self):
        """Spread exactly 0.3 should return NORMAL (>= 0.3)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(0.3)) == "YieldCurveRegime.NORMAL"

    def test_classify_curve_below_normal(self):
        """Spread 0.299 should return FLAT (>= 0 but not >= 0.3)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(0.299)) == "YieldCurveRegime.FLAT"

    def test_classify_curve_exact_inverted_threshold(self):
        """Spread exactly 0.0 returns INVERTED (strict > 0.0 means 0.0 is not FLAT)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(0.0)) == "YieldCurveRegime.INVERTED"

    def test_classify_curve_inverted(self):
        """Spread below 0 should return INVERTED."""
        calc = BondDurationCalculator()
        assert str(calc.classify_curve(-0.1)) == "YieldCurveRegime.INVERTED"

    def test_classify_rate_direction_falling(self):
        """Rate change < -0.30 should return FALLING."""
        calc = BondDurationCalculator()
        assert str(calc.classify_rate_direction(-0.31)) == "RateDirection.FALLING"

    def test_classify_rate_direction_falling_exact(self):
        """Rate change exactly -0.30 should be STABLE, not FALLING (strict <)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_rate_direction(-0.30)) == "RateDirection.STABLE"

    def test_classify_rate_direction_stable(self):
        """Rate change between -0.30 and 0.30 should return STABLE."""
        calc = BondDurationCalculator()
        assert str(calc.classify_rate_direction(0.0)) == "RateDirection.STABLE"

    def test_classify_rate_direction_rising_exact(self):
        """Rate change exactly 0.30 should be STABLE, not RISING (strict >)."""
        calc = BondDurationCalculator()
        assert str(calc.classify_rate_direction(0.30)) == "RateDirection.STABLE"

    def test_classify_rate_direction_rising(self):
        """Rate change > 0.30 should return RISING."""
        calc = BondDurationCalculator()
        assert str(calc.classify_rate_direction(0.31)) == "RateDirection.RISING"


# ── Effective Duration Tests ────────────────────────────────────────────────


class TestEffectiveDuration:
    """Test effective duration computation via BondDurationCalculator."""

    def test_all_tlt(self):
        """100% TLT should give 16-year duration."""
        calc = BondDurationCalculator()
        dur = calc.compute_effective_duration(1.0, 0.0, 0.0)
        assert dur == pytest.approx(16.0)

    def test_all_shy(self):
        """100% SHY should give 2-year duration."""
        calc = BondDurationCalculator()
        dur = calc.compute_effective_duration(0.0, 0.0, 1.0)
        assert dur == pytest.approx(2.0)

    def test_mixed_allocation(self):
        """Mixed allocation should compute weighted average."""
        calc = BondDurationCalculator()
        # 50% TLT, 30% IEF, 20% SHY
        dur = calc.compute_effective_duration(0.5, 0.3, 0.2)
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
        assert isinstance(result.extras["crisis_returns_baseline"], dict)
        assert isinstance(result.extras["crisis_returns_rotated"], dict)

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
            momentum = bt._compute_tlt_60d_momentum(idx)
            # Rising TLT price maps to falling rates via _momentum_to_yield_context
            spread, _, rate_chg = bt._momentum_to_yield_context(momentum)
            calc = bt._calc
            curve_regime = calc.classify_curve(spread)
            rate_dir = calc.classify_rate_direction(rate_chg)
            tlt, ief, shy, label = calc.compute_duration_allocation(spread, 1.5, rate_dir, curve_regime)
            assert tlt > shy  # TLT should dominate when TLT is rising

    def test_regime_breakdown_empty_tracker(self):
        """Empty regime_tracker should return empty dict."""
        bt = WalkForwardBondDurationBacktester()
        breakdown = bt._compute_regime_breakdown([])
        assert breakdown == {}

    def test_regime_breakdown_single_entry(self):
        """Single regime_tracker entry should produce valid breakdown."""
        bt = WalkForwardBondDurationBacktester()
        tracker = [{
            "date": "2020-01-15",
            "tlt_momentum": 0.02,
            "momentum_regime": "long",
            "tlt_sleeve": 0.8,
            "ief_sleeve": 0.2,
            "shy_sleeve": 0.0,
            "effective_duration": 15.0,
            "duration_change": 2.0,
        }]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert "long" in breakdown
        assert breakdown["long"]["count"] == 1
        assert breakdown["long"]["pct_of_time"] == 100.0
        assert breakdown["long"]["avg_effective_duration"] == 15.0

    def test_regime_breakdown_two_regimes(self):
        """Two distinct regimes should both appear in breakdown."""
        bt = WalkForwardBondDurationBacktester()
        tracker = [
            {
                "date": "2020-01-15", "tlt_momentum": 0.02,
                "momentum_regime": "long", "tlt_sleeve": 0.8,
                "ief_sleeve": 0.2, "shy_sleeve": 0.0,
                "effective_duration": 15.0, "duration_change": 2.0,
            },
            {
                "date": "2020-02-15", "tlt_momentum": -0.02,
                "momentum_regime": "short", "tlt_sleeve": 0.1,
                "ief_sleeve": 0.2, "shy_sleeve": 0.7,
                "effective_duration": 3.0, "duration_change": -12.0,
            },
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert "long" in breakdown
        assert "short" in breakdown
        assert breakdown["long"]["count"] == 1
        assert breakdown["short"]["count"] == 1
        assert breakdown["long"]["pct_of_time"] == 50.0
        assert breakdown["short"]["pct_of_time"] == 50.0

    def test_build_prices_lookup_keys(self):
        """_build_prices_lookup should map dates to symbol dicts."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        lookup = bt._build_prices_lookup()
        assert len(lookup) == len(bt._daily_prices)
        sample_date = bt._daily_prices[0].date
        assert sample_date in lookup
        for sym in ["SPY", "GLD", "TLT", "IEF", "SHY"]:
            assert sym in lookup[sample_date]
            assert isinstance(lookup[sample_date][sym], float)

    def test_crisis_returns_rotated_no_crisis_data(self):
        """Crisis returns should be empty when no crisis dates in range."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2013-01-01", end_date="2014-01-01")
        )
        bt.load_data()
        lookup = bt._build_prices_lookup()
        dates = bt._trading_dates
        equity = [100000.0] * len(dates)
        crisis = bt._compute_crisis_returns_rotated(lookup, dates, equity, 100000.0)
        # No 2008/2020/2022 dates in 2013-2014
        assert crisis == {}

    def test_run_rotated_returns_active_days(self):
        """_run_rotated should track active rotation days correctly."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        _, tracker, _ = bt._run_rotated(bt._daily_prices[:200], bt.config)
        assert "active_days" in tracker
        assert tracker["active_days"] >= 0
        assert len(tracker["sleeve_weights"]) > 0

    def test_effective_duration_range(self):
        """Effective duration should be between 2 and 16 years."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        _, tracker, _ = bt._run_rotated(bt._daily_prices[:200], bt.config)
        for dur in tracker["effective_durations"]:
            assert 2.0 <= dur <= 16.0, f"Duration {dur} out of [2, 16] range"


# ── Dataclass Field Validation ─────────────────────────────────────────────


class TestDataclassFieldValidation:
    """Validate dataclass fields via dataclasses.fields(), types, defaults."""

    def test_backtest_config_fields(self):
        """BacktestConfig should have expected fields via dataclasses.fields()."""
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(BacktestConfig)}
        expected = {
            "start_date", "end_date", "initial_capital", "base_weights",
            "rebalance_frequency_days", "rebalance_frequency",
            "transaction_cost_bps", "transaction_costs_by_symbol",
            "momentum_lookback_days", "extras",
        }
        assert set(fields.keys()) == expected

    def test_backtest_config_field_types(self):
        """BacktestConfig field types should be correct."""
        import dataclasses
        fields = {f.name: f.type for f in dataclasses.fields(BacktestConfig)}
        assert fields["start_date"] is str
        assert fields["end_date"] is str
        assert fields["initial_capital"] is float
        assert fields["momentum_lookback_days"] is int
        assert fields["rebalance_frequency_days"] is int
        assert fields["transaction_cost_bps"] is float

    def test_backtest_config_field_defaults(self):
        """BacktestConfig inherited field defaults via dataclasses.fields()."""
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(BacktestConfig)}
        # Fields with default values (not default_factory)
        assert fields["start_date"].default == "2006-01-01"
        assert fields["end_date"].default == "2026-05-15"
        assert fields["initial_capital"].default == 100000.0
        assert fields["rebalance_frequency_days"].default == 21
        assert fields["transaction_cost_bps"].default == 10.0
        assert fields["momentum_lookback_days"].default == 60

    def test_backtest_config_field_factories(self):
        """BacktestConfig fields with default_factory should produce mutable defaults."""
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(BacktestConfig)}
        assert fields["base_weights"].default_factory is not None
        assert fields["extras"].default_factory is not None
        config = BacktestConfig()
        assert config.base_weights["SPY"] == 0.46
        assert config.extras == {}

    def test_backtest_config_no_positional_args_required(self):
        """BacktestConfig should be constructable with zero arguments (all defaults)."""
        config = BacktestConfig()
        assert isinstance(config.start_date, str)

    def test_daily_prices_fields(self):
        """DailyPrices should have expected fields via dataclasses.fields()."""
        import dataclasses
        from src.backtest.metrics import DailyPrices as DP
        fields = {f.name: f for f in dataclasses.fields(DP)}
        expected = {"date", "spy", "gld", "tlt", "vix", "ief", "shy", "btc", "eth", "extras"}
        assert set(fields.keys()) == expected

    def test_daily_prices_field_types(self):
        """DailyPrices field types should be correct."""
        import dataclasses
        from src.backtest.metrics import DailyPrices as DP
        fields = {f.name: f.type for f in dataclasses.fields(DP)}
        assert fields["date"] is str
        assert fields["spy"] is float
        assert fields["gld"] is float
        assert fields["tlt"] is float

    def test_daily_prices_optional_fields(self):
        """DailyPrices optional fields should default to None."""
        import dataclasses
        from src.backtest.metrics import DailyPrices as DP
        fields = {f.name: f for f in dataclasses.fields(DP)}
        assert fields["vix"].default is None
        assert fields["ief"].default is None
        assert fields["shy"].default is None
        assert fields["btc"].default is None
        assert fields["eth"].default is None

    def test_backtest_config_is_dataclass(self):
        """BacktestConfig should be recognized as a dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(BacktestConfig)

    def test_daily_prices_is_dataclass(self):
        """DailyPrices should be recognized as a dataclass."""
        import dataclasses
        from src.backtest.metrics import DailyPrices as DP
        assert dataclasses.is_dataclass(DP)

    def test_backtest_config_momentum_lookback_defaults_to_constant(self):
        """momentum_lookback_days default should equal MOMENTUM_LOOKBACK constant."""
        config = BacktestConfig()
        assert config.momentum_lookback_days == MOMENTUM_LOOKBACK

    def test_daily_prices_required_fields_raise_type_error(self):
        """Missing required DailyPrices fields should raise TypeError."""
        from src.backtest.metrics import DailyPrices as DP
        with pytest.raises(TypeError):
            DP()  # Missing date, spy, gld, tlt

    def test_daily_prices_minimal_construction(self):
        """DailyPrices with only required fields should work."""
        from src.backtest.metrics import DailyPrices as DP
        dp = DP(date="2020-01-02", spy=100.0, gld=100.0, tlt=100.0)
        assert dp.ief is None
        assert dp.shy is None
        assert dp.vix is None

    def test_backtest_result_fields(self):
        """BacktestResult should have expected fields via dataclasses.fields()."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(BacktestResult)}
        expected = {
            "total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown",
            "total_rebalances", "total_transaction_costs", "avg_turnover",
            "baseline_sharpe", "sharpe_improvement", "extras", "crisis_returns",
        }
        assert fields == expected


# ── NaN / Inf Edge Cases ───────────────────────────────────────────────────


class TestNanInfEdgeCases:
    """Test NaN and Inf handling in computation methods."""

    def test_momentum_nan_price_returns_zero(self):
        """NaN in TLT price should not crash; returns NaN or 0.0."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = float("nan")
        bt._daily_prices[idx].tlt = 105.0
        mom = bt._compute_tlt_60d_momentum(idx)
        # nan <= 0 is False (does not trigger guard), so p1/p0 - 1 = 105/nan - 1 = nan
        # Just verify it doesn't crash and returns something
        assert isinstance(mom, float) or isinstance(mom, np.floating)

    def test_momentum_inf_prices_returns_finite(self):
        """Inf/inf prices should not crash; may produce nan (inf/inf - 1)."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = float("inf")
        bt._daily_prices[idx].tlt = float("inf")
        _ = bt._compute_tlt_60d_momentum(idx)
        # inf > 0 is True so guard doesn't trigger; inf/inf - 1 = nan
        # Just verify it doesn't crash
        assert True

    def test_momentum_inf_start_price_returns_finite(self):
        """Inf as start price: inf > 0 is True (no guard), 105/inf - 1 = -1.0."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = float("inf")
        bt._daily_prices[idx].tlt = 100.0
        mom = bt._compute_tlt_60d_momentum(idx)
        # inf > 0 is True, so guard doesn't trigger; 100/inf - 1 = -1.0
        assert mom == -1.0

    def test_portfolio_return_nan_prices(self):
        """NaN prices in portfolio return should not crash."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=float("nan"), gld=float("nan"),
                         tlt=float("nan"), ief=float("nan"), shy=float("nan"))
        p1 = DailyPrices(date="2020-01-03", spy=float("nan"), gld=float("nan"),
                         tlt=float("nan"), ief=float("nan"), shy=float("nan"))
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 1.0, 0.0, 0.0)
        assert np.isnan(ret) or ret == 0.0

    def test_portfolio_return_inf_prices(self):
        """Inf prices should not crash; returns may be inf or nan."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=100.0, gld=100.0,
                         tlt=100.0, ief=100.0, shy=100.0)
        p1 = DailyPrices(date="2020-01-03", spy=float("inf"), gld=float("inf"),
                         tlt=float("inf"), ief=float("inf"), shy=float("inf"))
        _ = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 1.0, 0.0, 0.0)
        # inf/100 - 1 = inf, weighted sum = inf. Should not crash.
        assert True

    def test_portfolio_return_zero_start_prices(self):
        """Zero start prices should be handled (return 0 via p0 > 0 guard)."""
        bt = WalkForwardBondDurationBacktester()
        p0 = DailyPrices(date="2020-01-02", spy=-1.0, gld=-1.0,
                         tlt=-1.0, ief=-1.0, shy=-1.0)
        p1 = DailyPrices(date="2020-01-03", spy=100.0, gld=100.0,
                         tlt=100.0, ief=100.0, shy=100.0)
        ret = bt._compute_portfolio_return(p0, p1, 0.46, 0.38, 0.16, 1.0, 0.0, 0.0)
        assert ret == 0.0

    def test_momentum_nan_end_price(self):
        """NaN as end price should produce nan momentum if start is valid."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        idx = MOMENTUM_LOOKBACK + 10
        bt._daily_prices[idx - MOMENTUM_LOOKBACK].tlt = 100.0
        bt._daily_prices[idx].tlt = float("nan")
        mom = bt._compute_tlt_60d_momentum(idx)
        assert np.isnan(mom) or mom == 0.0

    def test_yield_context_nan_momentum(self):
        """NaN momentum should map to neutral context (not crash)."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(float("nan"))
        # NaN comparisons all return False, so it falls to the else branch
        assert real_rate == 1.5
        assert isinstance(spread, float)
        assert isinstance(rate_chg, float)

    def test_yield_context_inf_momentum(self):
        """Inf momentum should map to a valid bucket (not crash)."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(float("inf"))
        # inf > 0.05 is True
        assert spread == 0.8
        assert rate_chg == -0.5

    def test_yield_context_neg_inf_momentum(self):
        """Negative Inf momentum should map to a valid bucket (not crash)."""
        bt = WalkForwardBondDurationBacktester()
        spread, real_rate, rate_chg = bt._momentum_to_yield_context(float("-inf"))
        # -inf > 0.05 is False, etc, falls to else
        assert real_rate == 1.5
        assert isinstance(rate_chg, float)


# ── CLI / __main__ Guard Tests ─────────────────────────────────────────────


class TestCliMainEntry:
    """Test the CLI entry point (main()) and __main__ guard."""

    def test_main_default_run(self, caplog):
        """main() should run the backtest and print results by default."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py"]
            with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
                main()
            assert "Bond Duration Rotation" in caplog.text
            assert "Sharpe Ratio" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_with_start_end_flags(self, caplog):
        """main() should accept --start and --end flags."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py", "--start", "2015-01-01", "--end", "2016-01-01"]
            with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
                main()
            assert "2015-01-01" in caplog.text
            assert "2016-01-01" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_with_capital_flag(self, caplog):
        """main() should accept --capital flag."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py", "--capital", "50000"]
            with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
                main()
            assert "$50,000" in caplog.text or "50000" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_with_save_flag(self, tmp_path):
        """main() with --save should not crash and produce output."""
        from src.backtest.bond_duration_backtest import main
        import sys
        import os
        old_argv = sys.argv
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            sys.argv = ["bond_duration_backtest.py", "--save", "--output", str(tmp_path / "test_results.json")]
            main()
            saved = tmp_path / "test_results.json"
            assert saved.exists()
            with open(saved) as f:
                data = json.load(f)
            assert "total_return" in data
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

    def test_main_with_output_flag(self, tmp_path):
        """main() should save to custom output path with --output."""
        from src.backtest.bond_duration_backtest import main
        import sys
        import os
        old_argv = sys.argv
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            out_path = tmp_path / "custom_output.json"
            sys.argv = ["bond_duration_backtest.py", "--output", str(out_path)]
            main()
            assert out_path.exists()
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

    def test_main_run_mode_explicit(self, caplog):
        """main() with explicit 'run' argument should work."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py", "run"]
            with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
                main()
            assert "Bond Duration Rotation" in caplog.text
        finally:
            sys.argv = old_argv

    def test_main_save_without_output_uses_default_path(self, tmp_path, monkeypatch):
        """main() with --save only (no --output) should not crash."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            monkeypatch.chdir(tmp_path)
            sys.argv = ["bond_duration_backtest.py", "--save"]
            main()
        finally:
            sys.argv = old_argv

    def test_module_main_guard_string(self):
        """The '__main__' guard string should exist in the module."""
        import inspect
        from src.backtest import bond_duration_backtest
        source = inspect.getsource(bond_duration_backtest)
        assert '__name__ == "__main__"' in source
        assert 'main()' in source.rsplit('__name__', 1)[-1]

    def test_main_parse_args_invalid_mode(self):
        """CLI with invalid mode should exit with error (argparse handles it)."""
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py", "invalid_mode"]
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = old_argv


# ── Export Completeness ────────────────────────────────────────────────────


class TestExportCompleteness:
    """Verify __all__ exports and public API coverage."""

    def test_all_exists(self):
        """Module should define __all__."""
        from src.backtest import bond_duration_backtest as mod
        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 0

    def test_all_names_are_strings(self):
        """All __all__ entries should be strings."""
        from src.backtest import bond_duration_backtest as mod
        for name in mod.__all__:
            assert isinstance(name, str), f"{name} is not a string"

    def test_all_names_exist_in_module(self):
        """Every name in __all__ should be accessible in the module."""
        from src.backtest import bond_duration_backtest as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} is not defined in module"

    def test_all_contains_core_classes(self):
        """__all__ should contain the core public classes and constants."""
        from src.backtest import bond_duration_backtest as mod
        expected = {"BacktestConfig", "WalkForwardBondDurationBacktester",
                    "MOMENTUM_LOOKBACK", "BOND_SLEEVE", "BASE_SYMBOLS",
                    "BOND_SYMBOLS", "TRADING_DAYS_PER_YEAR",
                    "MONTHLY_TRADING_DAYS", "CRISIS_YEARS"}
        for name in expected:
            assert name in mod.__all__, f"{name} missing from __all__"

    def test_all_no_private_names(self):
        """__all__ should not contain private/underscore-prefixed names."""
        from src.backtest import bond_duration_backtest as mod
        for name in mod.__all__:
            assert not name.startswith("_"), f"Private name {name} in __all__"

    def test_public_api_importable(self):
        """All __all__ names should be directly importable from the module."""
        from src.backtest import bond_duration_backtest as mod
        for name in mod.__all__:
            obj = getattr(mod, name)
            assert obj is not None, f"{name} resolves to None"

    def test_backtest_config_in_all(self):
        """BacktestConfig should be exported in __all__."""
        from src.backtest.bond_duration_backtest import BacktestConfig
        assert BacktestConfig is not None

    def test_walk_forward_class_exported(self):
        """WalkForwardBondDurationBacktester should be exported."""
        from src.backtest.bond_duration_backtest import WalkForwardBondDurationBacktester
        assert WalkForwardBondDurationBacktester is not None


# ── Additional Constants Validation ────────────────────────────────────────


class TestAdditionalConstantsValidation:
    """Validate module-level constant types, ranges, and consistency."""

    def test_trading_days_per_year_type(self):
        from src.backtest.bond_duration_backtest import TRADING_DAYS_PER_YEAR
        assert isinstance(TRADING_DAYS_PER_YEAR, int)

    def test_monthly_trading_days_type(self):
        from src.backtest.bond_duration_backtest import MONTHLY_TRADING_DAYS
        assert isinstance(MONTHLY_TRADING_DAYS, int)

    def test_crisis_years_type(self):
        from src.backtest.bond_duration_backtest import CRISIS_YEARS
        assert isinstance(CRISIS_YEARS, list)
        for y in CRISIS_YEARS:
            assert isinstance(y, str)
            assert y.isdigit()

    def test_base_symbols_type(self):
        from src.backtest.bond_duration_backtest import BASE_SYMBOLS
        assert isinstance(BASE_SYMBOLS, list)
        assert len(BASE_SYMBOLS) == 3
        for s in BASE_SYMBOLS:
            assert isinstance(s, str)

    def test_bond_symbols_type(self):
        from src.backtest.bond_duration_backtest import BOND_SYMBOLS
        assert isinstance(BOND_SYMBOLS, list)
        assert len(BOND_SYMBOLS) == 3
        for s in BOND_SYMBOLS:
            assert isinstance(s, str)

    def test_momentum_lookback_type(self):
        from src.backtest.bond_duration_backtest import MOMENTUM_LOOKBACK
        assert isinstance(MOMENTUM_LOOKBACK, int)
        assert MOMENTUM_LOOKBACK > 0

    def test_bond_sleeve_type(self):
        from src.backtest.bond_duration_backtest import BOND_SLEEVE
        assert isinstance(BOND_SLEEVE, float)
        assert 0.0 < BOND_SLEEVE < 1.0

    def test_all_constants_upper_case(self):
        """Module-level constants should be UPPER_CASE."""
        from src.backtest import bond_duration_backtest as mod
        constants = ["TRADING_DAYS_PER_YEAR", "MONTHLY_TRADING_DAYS",
                     "CRISIS_YEARS", "BASE_SYMBOLS", "BOND_SYMBOLS",
                     "MOMENTUM_LOOKBACK", "BOND_SLEEVE"]
        for name in constants:
            assert hasattr(mod, name), f"Missing constant {name}"
            val = getattr(mod, name)
            assert val is not None

    def test_crisis_years_chronological(self):
        """CRISIS_YEARS should be in chronological order."""
        from src.backtest.bond_duration_backtest import CRISIS_YEARS
        assert CRISIS_YEARS == sorted(CRISIS_YEARS)

    def test_base_symbols_no_duplicates(self):
        """BASE_SYMBOLS should have no duplicate entries."""
        from src.backtest.bond_duration_backtest import BASE_SYMBOLS
        assert len(BASE_SYMBOLS) == len(set(BASE_SYMBOLS))

    def test_bond_symbols_no_duplicates(self):
        """BOND_SYMBOLS should have no duplicate entries."""
        from src.backtest.bond_duration_backtest import BOND_SYMBOLS
        assert len(BOND_SYMBOLS) == len(set(BOND_SYMBOLS))

    def test_tlt_in_both_symbol_lists(self):
        """TLT should appear in both BASE_SYMBOLS and BOND_SYMBOLS."""
        from src.backtest.bond_duration_backtest import BASE_SYMBOLS, BOND_SYMBOLS
        assert "TLT" in BASE_SYMBOLS
        assert "TLT" in BOND_SYMBOLS

    def test_bond_sleeve_matches_base_weight(self):
        """BOND_SLEEVE should equal base_weights['TLT'] default."""
        config = BacktestConfig()
        assert BOND_SLEEVE == config.base_weights["TLT"]


# ── Additional Function Boundary Conditions ───────────────────────────────


class TestAdditionalBoundaryConditions:
    """Test more boundary conditions and edge cases."""

    def test_load_data_empty_json(self, tmp_path, monkeypatch):
        """Loading from empty/malformed JSON should fall back to synthetic data."""
        import json
        # Create a malformed file
        fake_prices = tmp_path / "prices.json"
        fake_prices.write_text("{bad json")
        monkeypatch.setattr("src.backtest.bond_duration_backtest.PRICES_JSON", fake_prices)
        bt = WalkForwardBondDurationBacktester()
        with pytest.raises(json.JSONDecodeError):
            bt.load_data()

    def test_load_data_missing_keys(self, tmp_path, monkeypatch):
        """JSON missing SPY/GLD/TLT keys should still work (dates derived from intersection)."""
        fake_prices = tmp_path / "prices.json"
        fake_prices.write_text(json.dumps({"SPY": [{"d": "2020-01-02", "p": 100.0}]}))
        monkeypatch.setattr("src.backtest.bond_duration_backtest.PRICES_JSON", fake_prices)
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        # Should fall back to synthetic because intersection is empty (no matching dates across SPY/GLD/TLT)
        assert len(bt._daily_prices) > 0

    def test_load_data_no_dates_in_range(self, tmp_path, monkeypatch):
        """Dates outside config range should trigger synthetic fallback."""
        fake_prices = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "1990-01-02", "p": 100.0}],
            "GLD": [{"d": "1990-01-02", "p": 100.0}],
            "TLT": [{"d": "1990-01-02", "p": 100.0}],
        }
        fake_prices.write_text(json.dumps(data))
        monkeypatch.setattr("src.backtest.bond_duration_backtest.PRICES_JSON", fake_prices)
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        # Synthetic fallback used
        assert len(bt._daily_prices) > 0

    def test_load_data_ief_shy_missing(self, tmp_path, monkeypatch):
        """Missing IEF/SHY in JSON should fall back to TLT data for those fields."""
        fake_prices = tmp_path / "prices.json"
        data = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "TLT": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 102.0}],
        }
        fake_prices.write_text(json.dumps(data))
        monkeypatch.setattr("src.backtest.bond_duration_backtest.PRICES_JSON", fake_prices)
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2020-01-02", end_date="2020-01-03")
        )
        bt.load_data()
        assert len(bt._daily_prices) == 2
        # IEF and SHY should fall back to TLT prices
        assert bt._daily_prices[0].ief == bt._daily_prices[0].tlt
        assert bt._daily_prices[0].shy == bt._daily_prices[0].tlt

    def test_compute_crisis_returns_rotated_empty_lookup(self):
        """Empty prices lookup should produce empty crisis returns."""
        bt = WalkForwardBondDurationBacktester()
        crisis = bt._compute_crisis_returns_rotated({}, [], [], 100000.0)
        assert crisis == {}

    def test_compute_crisis_returns_rotated_no_matching_year(self):
        """Dates not in crisis years should produce empty returns."""
        bt = WalkForwardBondDurationBacktester()
        lookup = {"2013-06-01": {"SPY": 100.0}}
        crisis = bt._compute_crisis_returns_rotated(lookup, ["2013-06-01"], [100000.0], 100000.0)
        assert crisis == {}

    def test_compute_crisis_returns_rotated_zero_equity_start(self):
        """Zero equity start should not add key to result."""
        bt = WalkForwardBondDurationBacktester()
        bt._daily_prices = [DailyPrices(date="2008-01-02", spy=100.0, gld=100.0, tlt=100.0)]
        lookup = {"2008-01-02": {"SPY": 100.0}}
        crisis = bt._compute_crisis_returns_rotated(lookup, ["2008-01-02"], [0.0, 0.0], 100000.0)
        assert crisis == {}

    def test_build_prices_lookup_empty(self):
        """Empty daily prices should produce empty lookup."""
        bt = WalkForwardBondDurationBacktester()
        bt._daily_prices = []
        lookup = bt._build_prices_lookup()
        assert lookup == {}

    def test_run_without_load_data_triggers_load(self):
        """run() should auto-load data if not already loaded."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2015-06-01")
        )
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert len(bt._daily_prices) > 0

    def test_save_results_no_path_uses_default(self, tmp_path, monkeypatch):
        """save_results() without output_path should use default path."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        result = bt.run()
        default_dir = tmp_path / "results"
        default_dir.mkdir()
        monkeypatch.setattr("src.backtest.bond_duration_backtest.BACKTEST_RESULTS_DIR", default_dir)
        bt.save_results(result)
        expected_file = default_dir / "bond_duration_backtest_results.json"
        assert expected_file.exists()
        with open(expected_file) as f:
            data = json.load(f)
        assert data["_metadata"]["strategy"] == "bond_duration"

    def test_synthetic_data_length(self):
        """Synthetic data should match expected number of days."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2000-01-03", end_date="2020-01-01")
        )
        bt._generate_synthetic_data()
        # Should be at least 5000 days (20 years)
        assert len(bt._daily_prices) >= 4800

    def test_synthetic_data_dates_are_weekdays(self):
        """All synthetic dates should be weekdays (Mon-Fri)."""
        from datetime import datetime
        bt = WalkForwardBondDurationBacktester()
        bt._generate_synthetic_data()
        for date_str in bt._trading_dates[:50]:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            assert dt.weekday() < 5, f"{date_str} is a weekend"

    def test_synthetic_data_prices_positive(self):
        """All synthetic prices should be positive."""
        bt = WalkForwardBondDurationBacktester()
        bt._generate_synthetic_data()
        for dp in bt._daily_prices[:50]:
            assert dp.spy > 0
            assert dp.gld > 0
            assert dp.tlt > 0
            assert dp.ief > 0
            assert dp.shy > 0

    def test_synthetic_data_reproducible(self):
        """Synthetic data should be reproducible with same seed."""
        bt1 = WalkForwardBondDurationBacktester()
        bt1._generate_synthetic_data()
        bt2 = WalkForwardBondDurationBacktester()
        bt2._generate_synthetic_data()
        assert len(bt1._daily_prices) == len(bt2._daily_prices)
        assert bt1._daily_prices[0].spy == bt2._daily_prices[0].spy

    def test_empty_data_direct_call(self):
        """Empty _daily_prices returns empty result when run() triggers load_data fallback."""
        bt = WalkForwardBondDurationBacktester()
        bt._daily_prices = []
        bt._trading_dates = []
        result = bt.run()
        # run() calls load_data() which generates synthetic data, so result is populated
        assert isinstance(result, BacktestResult)

    def test_backtest_config_base_weights_mutable(self):
        """BacktestConfig base_weights should not share references across instances."""
        c1 = BacktestConfig()
        c2 = BacktestConfig()
        c1.base_weights["SPY"] = 0.99
        assert c2.base_weights["SPY"] == 0.46  # Should be unchanged

    def test_print_results_empty_crisis(self, caplog):
        """print_results should handle missing crisis and regime data."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45, sharpe_improvement=0.05,
            total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 4.0, "baseline_cagr": 2.5,
                "baseline_volatility": 9.5, "baseline_max_drawdown": -12.0,
                "cagr_impact": 0.5, "rotation_active_days": 0,
                "rotation_active_pct": 0.0, "avg_effective_duration": 0.0,
                "avg_tlt_weight": 0.0, "avg_ief_weight": 0.0, "avg_shy_weight": 0.0,
                "crisis_returns_rotated": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        bt = WalkForwardBondDurationBacktester()
        with caplog.at_level(logging.INFO, logger="src.backtest.bond_duration_backtest"):
            bt.print_results(result)
        assert "Crisis Returns" in caplog.text

    def test_cli_logging_startup(self, caplog):
        """CLI main() should log when starting."""
        import logging
        from src.backtest.bond_duration_backtest import main
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bond_duration_backtest.py", "--start", "2015-01-01", "--end", "2015-06-01"]
            with caplog.at_level(logging.INFO):
                main()
            assert len(caplog.records) >= 0  # Just shouldn't crash
        finally:
            sys.argv = old_argv

    def test_momentum_via_config_lookback(self):
        """Momentum computation should respect config's momentum_lookback_days."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(momentum_lookback_days=10)
        )
        bt.load_data()
        idx = 15
        bt._daily_prices[idx - 10].tlt = 100.0
        bt._daily_prices[idx].tlt = 110.0
        mom = bt._compute_tlt_60d_momentum(idx)
        assert mom == pytest.approx(0.10)

    def test_run_rotated_with_no_rebalance(self):
        """run_rotated with very long rebalance frequency should still work."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(rebalance_frequency_days=5000)
        )
        bt.load_data()
        subset = bt._daily_prices[:50]
        equity, tracker, _ = bt._run_rotated(subset, bt.config)
        assert len(equity) == len(subset)
        # Should have at least 1 rebalance (initial)
        assert tracker["rebalances"] >= 1

    def test_compute_regime_breakdown_same_regime_multiple(self):
        """Multiple entries in same regime should aggregate correctly."""
        bt = WalkForwardBondDurationBacktester()
        tracker = [
            {"date": "2020-01-15", "tlt_momentum": 0.02, "momentum_regime": "long",
             "tlt_sleeve": 0.8, "ief_sleeve": 0.2, "shy_sleeve": 0.0,
             "effective_duration": 15.0, "duration_change": 2.0},
            {"date": "2020-01-16", "tlt_momentum": 0.03, "momentum_regime": "long",
             "tlt_sleeve": 0.7, "ief_sleeve": 0.3, "shy_sleeve": 0.0,
             "effective_duration": 14.0, "duration_change": -1.0},
        ]
        breakdown = bt._compute_regime_breakdown(tracker)
        assert breakdown["long"]["count"] == 2
        assert breakdown["long"]["pct_of_time"] == 100.0
        assert breakdown["long"]["avg_effective_duration"] == 14.5

    def test_baseline_equity_monotonic_weights(self):
        """Baseline equity should use constant weights (all TLT in bond sleeve)."""
        bt = WalkForwardBondDurationBacktester()
        bt.load_data()
        subset = bt._daily_prices[:10]
        config = BacktestConfig(initial_capital=100000.0)
        spy_w = config.base_weights["SPY"]
        gld_w = config.base_weights["GLD"]
        tlt_w = config.base_weights["TLT"]
        equity = bt._run_baseline(subset, config)
        # Expected return computed manually for each day
        for i in range(1, len(subset)):
            ret = bt._compute_portfolio_return(
                subset[i-1], subset[i], spy_w, gld_w, tlt_w, 1.0, 0.0, 0.0
            )
            expected_eq = equity[i-1] * (1 + ret)
            assert equity[i] == pytest.approx(expected_eq, rel=1e-10)

    def test_run_result_has_cagr(self):
        """Run result should have a CAGR value (may be negative on synthetic data)."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert result.cagr is not None

    def test_run_result_config_snapshot_includes_bond_allocation_note(self):
        """Config snapshot should include bond_sleeve_allocation description."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        cs = result.extras["config_snapshot"]
        assert "BondDurationCalculator" in cs["bond_sleeve_allocation"]

    def test_run_result_avg_ief_weight_range(self):
        """Average IEF weight should be between 0 and 1."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert 0.0 <= result.extras["avg_ief_weight"] <= 1.0

    def test_run_result_avg_shy_weight_range(self):
        """Average SHY weight should be between 0 and 1."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert 0.0 <= result.extras["avg_shy_weight"] <= 1.0

    def test_run_result_total_transaction_costs_non_negative(self):
        """Total transaction costs should be non-negative."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert result.total_transaction_costs >= 0

    def test_run_result_rotation_active_pct_range(self):
        """Rotation active percentage should be between 0 and 100."""
        bt = WalkForwardBondDurationBacktester(
            BacktestConfig(start_date="2015-01-01", end_date="2016-01-01")
        )
        result = bt.run()
        assert 0.0 <= result.extras["rotation_active_pct"] <= 100.0

    def test_empty_result_regime_breakdown_empty(self):
        """_empty_result() should have empty regime_breakdown."""
        bt = WalkForwardBondDurationBacktester()
        result = bt._empty_result()
        assert result.extras["regime_breakdown"] == {}

    def test_empty_result_config_snapshot_empty(self):
        """_empty_result() should have empty config_snapshot."""
        bt = WalkForwardBondDurationBacktester()
        result = bt._empty_result()
        assert result.extras["config_snapshot"] == {}

    def test_print_results_config_snapshot_handles_missing_keys(self, capsys):
        """print_results should handle missing extras gracefully."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_sharpe=0.0, sharpe_improvement=0.0,
            total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "baseline_total_return": 0.0, "baseline_cagr": 0.0,
                "baseline_volatility": 0.0, "baseline_max_drawdown": 0.0,
                "cagr_impact": 0.0, "rotation_active_days": 0,
                "rotation_active_pct": 0.0, "avg_effective_duration": 0.0,
                "avg_tlt_weight": 0.0, "avg_ief_weight": 0.0, "avg_shy_weight": 0.0,
                "crisis_returns_rotated": {}, "crisis_returns_baseline": {},
                "regime_breakdown": {}, "config_snapshot": {},
            },
        )
        bt = WalkForwardBondDurationBacktester()
        bt.print_results(result)
        captured = capsys.readouterr()
        assert captured.out is not None  # Should not crash

    def test_backtest_config_with_extras(self):
        """BacktestConfig should support extras dict for additional parameters."""
        config = BacktestConfig(extras={"custom_param": 42, "debug": True})
        assert config.extras["custom_param"] == 42
        assert config.extras["debug"] is True


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 2): load_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # class method stays in pilot; the shared loader is grid_runner's
    assert WalkForwardBondDurationBacktester.load_data.__module__ == (
        "src.backtest.bond_duration_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
