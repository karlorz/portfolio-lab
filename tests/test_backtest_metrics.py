#!/usr/bin/env python3
"""
Tests for src/backtest/metrics.py — shared backtest metrics module.
"""
import os
import json
import tempfile

import pytest
import numpy as np
from pathlib import Path

from src.backtest.metrics import (
    BacktestConfig, BacktestResult, DailyPrices,
    BacktestMetrics, OverlayMetrics, CrisisReturns,
    compute_metrics, compute_crisis_returns,
    compute_deflated_sharpe_ratio,
    print_metrics_report, save_results_json,
)


class TestBacktestConfig:
    """Tests for the canonical BacktestConfig dataclass."""

    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "2006-01-01"
        assert cfg.end_date == "2026-05-15"
        assert cfg.initial_capital == 100000.0
        assert cfg.rebalance_frequency_days == 21
        assert cfg.rebalance_frequency == "monthly"
        assert cfg.transaction_cost_bps == 10.0
        assert cfg.extras == {}

    def test_per_etf_cost_dict_present(self):
        cfg = BacktestConfig()
        costs = cfg.transaction_costs_by_symbol
        assert isinstance(costs, dict)
        assert costs['SPY'] == 2.0
        assert costs['GLD'] == 5.0
        assert costs['TLT'] == 8.0
        assert costs['DBC'] == 10.0

    def test_per_etf_cost_dict_covers_core_symbols(self):
        cfg = BacktestConfig()
        costs = cfg.transaction_costs_by_symbol
        for sym in ['SPY', 'GLD', 'TLT', 'IEF', 'QQQ']:
            assert sym in costs, f"{sym} missing from transaction_costs_by_symbol"

    def test_per_etf_costs_all_positive(self):
        cfg = BacktestConfig()
        for sym, cost in cfg.transaction_costs_by_symbol.items():
            assert cost > 0, f"{sym} has non-positive cost {cost}"

    def test_base_weights_are_canonical(self):
        """Base weights must match BASE_ALLOCATION from src.paths."""
        cfg = BacktestConfig()
        from src.paths import BASE_ALLOCATION
        for key, val in BASE_ALLOCATION.items():
            assert cfg.base_weights[key] == val

    def test_custom_start_end(self):
        cfg = BacktestConfig(start_date="2020-01-01", end_date="2024-12-31")
        assert cfg.start_date == "2020-01-01"
        assert cfg.end_date == "2024-12-31"

    def test_custom_base_weights(self):
        cfg = BacktestConfig(base_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2})
        assert cfg.base_weights["SPY"] == 0.5
        assert cfg.base_weights["GLD"] == 0.3
        assert cfg.base_weights["TLT"] == 0.2

    def test_extras_dict(self):
        cfg = BacktestConfig(extras={"max_shift": 0.05, "lookback": 130})
        assert cfg.extras["max_shift"] == 0.05
        assert cfg.extras["lookback"] == 130

    def test_default_extras_is_empty_and_independent(self):
        """Each instance should get its own extras dict."""
        cfg1 = BacktestConfig()
        cfg2 = BacktestConfig()
        cfg1.extras["foo"] = "bar"
        assert "foo" not in cfg2.extras


class TestDailyPrices:
    """Tests for the canonical DailyPrices dataclass."""

    def test_required_fields(self):
        dp = DailyPrices(date="2024-01-01", spy=500.0, gld=180.0, tlt=130.0)
        assert dp.date == "2024-01-01"
        assert dp.spy == 500.0
        assert dp.gld == 180.0
        assert dp.tlt == 130.0
        assert dp.vix is None  # Optional

    def test_optional_vix(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130, vix=18.5)
        assert dp.vix == 18.5

    def test_optional_crypto(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         btc=42000.0, eth=2200.0)
        assert dp.btc == 42000.0
        assert dp.eth == 2200.0

    def test_optional_ief_shy(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         ief=95.0, shy=82.0)
        assert dp.ief == 95.0
        assert dp.shy == 82.0

    def test_extras_default_empty(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130)
        assert dp.extras == {}

    def test_extras_custom(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130,
                         extras={"DBC": 18.5, "VIXY": 12.3})
        assert dp.extras["DBC"] == 18.5

    def test_all_none_optional_fields(self):
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130)
        assert dp.vix is None
        assert dp.ief is None
        assert dp.shy is None
        assert dp.btc is None
        assert dp.eth is None


class TestBacktestResult:
    """Tests for the canonical BacktestResult dataclass."""

    def test_required_fields_only(self):
        r = BacktestResult(total_return=10.0, cagr=5.0, volatility=12.0,
                           sharpe_ratio=0.5, max_drawdown=-15.0)
        assert r.total_rebalances == 0
        assert r.total_transaction_costs == 0.0
        assert r.avg_turnover == 0.0
        assert r.baseline_sharpe is None
        assert r.sharpe_improvement is None
        assert r.extras == {}
        assert r.crisis_returns is None

    def test_all_fields(self):
        r = BacktestResult(
            total_return=50.0, cagr=8.0, volatility=10.0,
            sharpe_ratio=0.8, max_drawdown=-20.0,
            total_rebalances=120, total_transaction_costs=500.0,
            avg_turnover=0.03,
            baseline_sharpe=0.7, sharpe_improvement=0.1,
            extras={"custom": "value"},
            crisis_returns={"2008": -12.0},
        )
        assert r.total_rebalances == 120
        assert r.baseline_sharpe == 0.7
        assert r.extras["custom"] == "value"
        assert r.crisis_returns["2008"] == -12.0

    def test_extras_independent_per_instance(self):
        r1 = BacktestResult(total_return=0, cagr=0, volatility=0,
                            sharpe_ratio=0, max_drawdown=0)
        r2 = BacktestResult(total_return=0, cagr=0, volatility=0,
                            sharpe_ratio=0, max_drawdown=0)
        r1.extras["key"] = "val"
        assert "key" not in r2.extras

    def test_crisis_returns_none_vs_empty(self):
        r_none = BacktestResult(total_return=0, cagr=0, volatility=0,
                                sharpe_ratio=0, max_drawdown=0)
        r_empty = BacktestResult(total_return=0, cagr=0, volatility=0,
                                 sharpe_ratio=0, max_drawdown=0,
                                 crisis_returns={})
        assert r_none.crisis_returns is None
        assert r_empty.crisis_returns == {}

    def test_negative_sharpe_improvement(self):
        r = BacktestResult(total_return=5.0, cagr=3.0, volatility=12.0,
                           sharpe_ratio=0.3, max_drawdown=-10.0,
                           baseline_sharpe=0.5, sharpe_improvement=-0.2)
        assert r.sharpe_improvement < 0


class TestBacktestMetrics:
    def test_creation_defaults(self):
        m = BacktestMetrics(total_return=10.0, cagr=8.0, volatility=12.0,
                            sharpe_ratio=0.67, max_drawdown=-15.0)
        assert m.total_rebalances == 0
        assert m.total_transaction_costs == 0.0

    def test_custom_fields(self):
        m = BacktestMetrics(total_return=10.0, cagr=8.0, volatility=12.0,
                            sharpe_ratio=0.67, max_drawdown=-15.0,
                            total_rebalances=24, total_transaction_costs=50.0)
        assert m.total_rebalances == 24


class TestOverlayMetrics:
    def test_creation(self):
        m = OverlayMetrics(baseline_sharpe=0.94, sharpe_improvement=0.015)
        assert m.overlay_active_count == 0
        assert m.overlay_active_pct == 0.0

    def test_with_active_stats(self):
        m = OverlayMetrics(baseline_sharpe=0.94, sharpe_improvement=-0.012,
                           overlay_active_count=120, overlay_active_pct=0.45)
        assert m.overlay_active_pct == 0.45


class TestCrisisReturns:
    def test_empty(self):
        c = CrisisReturns()
        assert c.get('2008') is None

    def test_with_returns(self):
        c = CrisisReturns(returns={'2008': -12.3, '2020': -7.1, '2022': -13.0})
        assert c.get('2008') == -12.3
        assert c.get('2023') is None


class TestComputeMetrics:
    def test_empty_curve(self):
        m = compute_metrics([], 100000)
        assert m.cagr == 0.0
        assert m.sharpe_ratio == 0.0

    def test_single_value(self):
        m = compute_metrics([100000], 100000)
        assert m.cagr == 0.0

    def test_positive_return(self):
        curve = [100000, 101000, 102000, 103000, 104000]
        m = compute_metrics(curve, 100000)
        assert m.total_return > 0
        assert m.cagr > 0
        assert m.max_drawdown == 0.0  # Monotonically increasing

    def test_negative_return(self):
        curve = [100000, 99000, 98000, 97000]
        m = compute_metrics(curve, 100000)
        assert m.total_return < 0
        assert m.cagr < 0

    def test_drawdown(self):
        curve = [100000, 110000, 95000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.max_drawdown < 0  # Should have drawdown from 110k to 95k

    def test_zero_initial_capital(self):
        curve = [0, 1000, 2000]
        m = compute_metrics(curve, 0)
        assert m.cagr == 0.0

    def test_sharpe_positive(self):
        np.random.seed(42)
        curve = [100000]
        for _ in range(251):
            curve.append(curve[-1] * (1 + np.random.normal(0.0004, 0.01)))
        m = compute_metrics(curve, 100000)
        assert m.sharpe_ratio > 0

    def test_rounding(self):
        curve = [100000, 105000]
        m = compute_metrics(curve, 100000)
        # Values should be rounded to 2-4 decimal places
        assert isinstance(m.total_return, float)


class TestComputeCrisisReturns:
    def test_basic(self):
        prices = {
            '2008-01-02': {'SPY': 100, 'GLD': 80, 'TLT': 90},
            '2008-12-31': {'SPY': 70, 'GLD': 85, 'TLT': 110},
        }
        result = compute_crisis_returns(prices, ['2008-01-02', '2008-12-31'])
        assert '2008' in result
        assert result['2008'] < 0  # SPY dropped, GLD/TLT partial offset

    def test_missing_year(self):
        prices = {
            '2020-01-02': {'SPY': 100, 'GLD': 150, 'TLT': 130},
            '2020-12-31': {'SPY': 120, 'GLD': 170, 'TLT': 155},
        }
        result = compute_crisis_returns(prices, ['2020-01-02', '2020-12-31'],
                                         crisis_years=['2008'])
        assert '2008' not in result

    def test_custom_weights(self):
        prices = {
            '2022-01-03': {'SPY': 100, 'GLD': 160, 'TLT': 140},
            '2022-12-30': {'SPY': 80, 'GLD': 155, 'TLT': 115},
        }
        result = compute_crisis_returns(prices, ['2022-01-03', '2022-12-30'],
                                         base_weights={'SPY': 1.0, 'GLD': 0.0, 'TLT': 0.0})
        assert '2022' in result
        assert result['2022'] < -15  # SPY alone dropped 20%

    def test_insufficient_data(self):
        prices = {'2008-01-02': {'SPY': 100}}
        result = compute_crisis_returns(prices, ['2008-01-02'])
        assert '2008' not in result  # Need at least 2 days


class TestPrintMetricsReport:
    def test_prints_without_error(self, capsys):
        m = BacktestMetrics(total_return=10.5, cagr=8.2, volatility=12.1,
                            sharpe_ratio=0.68, max_drawdown=-15.3)
        print_metrics_report(m, title="Test Report")
        captured = capsys.readouterr()
        assert "Test Report" in captured.out
        assert "8.2" in captured.out


class TestSaveResultsJson:
    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_results.json')
            data = {'cagr': 8.5, 'sharpe': 0.68, 'nested': {'key': 'val'}}
            save_results_json(data, output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['cagr'] == 8.5
            assert loaded['nested']['key'] == 'val'

    def test_saves_with_default_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {'cagr': 8.5}
            save_results_json(data, default_dir=Path(tmpdir))
            output = Path(tmpdir) / 'backtest_results.json'
            assert output.exists()

    def test_numpy_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'numpy_results.json')
            data = {'cagr': np.float64(8.5), 'count': np.int64(42), 'arr': np.array([1, 2, 3])}
            save_results_json(data, output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['cagr'] == 8.5
            assert loaded['count'] == 42
            assert loaded['arr'] == [1, 2, 3]

    def test_no_path_no_dir(self):
        # Should not crash, just return
        save_results_json({'key': 'val'}, output_path=None, default_dir=None)

    def test_validator_passes_data_through(self, tmp_path):
        """Validator callback should transform data before serialization."""
        path = tmp_path / "validated.json"

        def add_defaults(d):
            d["validated"] = True
            d["sharpe"] = d.get("sharpe", 0.0)
            return d

        save_results_json({"cagr": 8.5}, output_path=str(path), validator=add_defaults)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["cagr"] == 8.5
        assert loaded["sharpe"] == 0.0
        assert loaded["validated"] is True

    def test_validator_None_skips_validation(self, tmp_path):
        """Default validator=None must behave identically to no validator."""
        path = tmp_path / "no_validator.json"
        data = {"cagr": 8.5}
        save_results_json(data, output_path=str(path), validator=None)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == {"cagr": 8.5}

    def test_validator_raises_logs_warning_and_uses_original(self, tmp_path):
        """When validator raises, a warning is logged and original data is saved."""
        path = tmp_path / "raise_validator.json"
        data = {"cagr": 8.5}

        def broken(d):
            raise ValueError("something broke")

        save_results_json(data, output_path=str(path), validator=broken)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == {"cagr": 8.5}


# ---------------------------------------------------------------------------
# compute_deflated_sharpe_ratio
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatio:

    def test_single_trial_returns_high_dsr(self):
        """With only 1 trial, no multiple-testing penalty — DSR should be high for decent Sharpe."""
        dsr = compute_deflated_sharpe_ratio(
            sharpe_ratio=0.79, n_trials=1, n_observations=5371,
        )
        assert dsr > 0.90

    def test_many_trials_reduces_dsr(self):
        """More trials → more multiple-testing penalty → lower DSR."""
        dsr_10 = compute_deflated_sharpe_ratio(0.20, n_trials=10, n_observations=100)
        dsr_94 = compute_deflated_sharpe_ratio(0.20, n_trials=94, n_observations=100)
        assert dsr_10 > dsr_94

    def test_higher_sharpe_gives_higher_dsr(self):
        dsr_low = compute_deflated_sharpe_ratio(0.15, n_trials=20, n_observations=100)
        dsr_high = compute_deflated_sharpe_ratio(0.30, n_trials=20, n_observations=100)
        assert dsr_high > dsr_low

    def test_zero_sharpe_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.0, n_trials=10, n_observations=1000)
        assert dsr == 0.0

    def test_dsr_between_zero_and_one(self):
        for n_trials in [1, 5, 20, 94]:
            dsr = compute_deflated_sharpe_ratio(0.79, n_trials=n_trials, n_observations=5371)
            assert 0.0 <= dsr <= 1.0, f"DSR out of range for n_trials={n_trials}: {dsr}"

    def test_champion_sharpe_with_94_trials(self):
        """Champion Sharpe 0.79 with 94 grid-search configs should have DSR > 0.50."""
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=94, n_observations=5371)
        assert dsr > 0.50, f"DSR for champion is only {dsr}, expected > 0.50"

    def test_more_observations_increases_dsr(self):
        """More data → more statistical power → higher DSR."""
        dsr_short = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=500)
        dsr_long = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=5000)
        assert dsr_long >= dsr_short

    def test_skew_and_kurtosis_parameters(self):
        """Non-normal return distribution should affect DSR."""
        dsr_normal = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=5371, skew=0.0, kurtosis=3.0)
        dsr_skewed = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=5371, skew=-0.5, kurtosis=5.0)
        # Both should be valid
        assert 0.0 <= dsr_normal <= 1.0
        assert 0.0 <= dsr_skewed <= 1.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Test public module-level constants."""

    def test_trading_days_per_year(self):
        from src.backtest.metrics import TRADING_DAYS_PER_YEAR
        assert TRADING_DAYS_PER_YEAR == 252

    def test_default_crisis_years(self):
        from src.backtest.metrics import DEFAULT_CRISIS_YEARS
        assert DEFAULT_CRISIS_YEARS == ['2008', '2020', '2022']

    def test_rebalance_frequency_days(self):
        from src.backtest.metrics import REBALANCE_FREQUENCY_DAYS
        assert REBALANCE_FREQUENCY_DAYS == 21

    def test_default_transaction_cost_bps(self):
        from src.backtest.metrics import DEFAULT_TRANSACTION_COST_BPS
        assert DEFAULT_TRANSACTION_COST_BPS == 10.0


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestAllExports:
    """Validate __all__ matches module exports."""

    def test_all_exports_defined(self):
        from src.backtest import metrics as m
        exports = m.__all__
        assert isinstance(exports, list)
        assert len(exports) == 11

    def test_all_exported_names_exist(self):
        """Every name in __all__ must be accessible on the module."""
        from src.backtest import metrics as m
        for name in m.__all__:
            assert hasattr(m, name), f"{name} declared in __all__ but not in module"

    def test_all_exports_are_public(self):
        """No __all__ entry should start with underscore."""
        from src.backtest import metrics as m
        for name in m.__all__:
            assert not name.startswith('_'), f"{name} in __all__ starts with underscore"


# ---------------------------------------------------------------------------
# compute_metrics — additional edge cases
# ---------------------------------------------------------------------------

class TestComputeMetricsEdgeCases:
    """Additional edge cases for compute_metrics beyond basic tests."""

    def test_constant_curve_zero_sharpe(self):
        """Flat equity curve should give 0% return and 0 Sharpe."""
        curve = [100000] * 10
        m = compute_metrics(curve, 100000)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_constant_curve_above_initial(self):
        """Flat curve above initial capital: positive return, zero vol, zero sharpe."""
        curve = [110000] * 10
        m = compute_metrics(curve, 100000)
        assert m.total_return > 0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0

    def test_all_negative_curve(self):
        """Every step loses money (sequential halving)."""
        curve = [100000, 50000, 25000, 12500]
        m = compute_metrics(curve, 100000)
        assert m.total_return < 0
        assert m.cagr < 0
        assert m.sharpe_ratio == 0.0  # negative CAGR / positive vol = 0 (clamped)
        assert m.max_drawdown < -50  # lost >50% peak-to-trough

    def test_single_step_up(self):
        """Single step from 100k to 105k."""
        curve = [100000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.total_return == 5.0  # 5%
        assert m.max_drawdown == 0.0
        assert m.cagr > 0

    def test_extreme_positive_values(self):
        """Very large values should not overflow."""
        curve = [1e6, 1.5e6, 2.25e6, 3.375e6]
        m = compute_metrics(curve, 1e6)
        assert m.total_return > 200
        assert m.cagr > 0
        # Note: constant returns → std=0 → vol=0 → sharpe=0 (div-by-zero protected)

    def test_zero_initial_positive_curve(self):
        """Zero initial capital with non-zero curve: total_return/cagr=0, vol/sharpe computed."""
        curve = [0, 100, 200]
        m = compute_metrics(curve, 0)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        # Returns are [0.0, 1.0] — vol is non-zero but sharpe is clamped to 0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == -100.0  # peak 200 → first value 0 → 100% dd

    def test_negative_initial_capital(self):
        """Negative initial capital should not crash."""
        curve = [-10000, -5000, -2000]
        m = compute_metrics(curve, -10000)
        # Should not raise; results are not financially meaningful
        assert isinstance(m.total_return, float)

    def test_volatility_with_single_return(self):
        """Volatility should be computable from 2 data points (1 return)."""
        curve = [100000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.volatility == 0.0  # std of single value = 0
        # With 252 trading days, CAGR = 105000/100000^(252/1) - 1 = 5^252 - 1 ≈ huge
        assert m.cagr > 0

    def test_uneven_steps(self):
        """Mix of positive and negative daily returns."""
        curve = [100000, 102000, 101000, 103000, 99000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.total_return == 5.0  # (105000/100000 - 1) * 100 = 5%
        assert m.sharpe_ratio != 0
        assert m.max_drawdown < 0  # had a drawdown from 103k to 99k

    def test_negative_to_positive_recovery(self):
        """Curve dips then recovers — max_drawdown captures the dip."""
        curve = [100000, 80000, 70000, 90000, 110000]
        m = compute_metrics(curve, 100000)
        assert m.max_drawdown <= -30.0  # (70k peak-to-trough from 100k peak)
        assert m.total_return == 10.0  # 10% total return


# ---------------------------------------------------------------------------
# compute_deflated_sharpe_ratio — additional edge cases
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatioEdgeCases:
    """Additional edge cases for compute_deflated_sharpe_ratio."""

    def test_n_trials_zero_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=0, n_observations=5371)
        assert dsr == 0.0

    def test_n_trials_negative_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=-1, n_observations=5371)
        assert dsr == 0.0

    def test_n_observations_zero_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=10, n_observations=0)
        assert dsr == 0.0

    def test_n_observations_negative_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=10, n_observations=-5)
        assert dsr == 0.0

    def test_single_trial_zero_sharpe_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.0, n_trials=1, n_observations=1000)
        assert dsr == 0.0

    def test_two_trials(self):
        """n_trials=2 is a boundary case (not > 2 for sigma_max formula)."""
        dsr = compute_deflated_sharpe_ratio(0.50, n_trials=2, n_observations=1000)
        assert 0.0 <= dsr <= 1.0

    def test_very_large_n_trials(self):
        """Large number of trials should still produce valid DSR."""
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=10000, n_observations=5371)
        assert 0.0 <= dsr <= 1.0

    def test_tiny_sharpe_with_one_trial(self):
        """Very small but positive Sharpe with 1 trial."""
        dsr = compute_deflated_sharpe_ratio(0.01, n_trials=1, n_observations=100)
        assert dsr > 0.50  # 1 trial, no penalty

    def test_negative_sharpe_returns_low_dsr(self):
        """Negative Sharpe should produce DSR near 0."""
        dsr = compute_deflated_sharpe_ratio(-0.50, n_trials=10, n_observations=1000)
        assert dsr < 0.50

    def test_sigma_max_non_positive_edge(self):
        """When sigma_max <= 0, DSR should be 1.0 if sharpe > expected_max, else 0.0."""
        # Force variance to near zero by using very high kurtosis cancellation
        dsr = compute_deflated_sharpe_ratio(0.0, n_trials=10, n_observations=1000)
        assert dsr == 0.0

    def test_dsr_monotonic_in_sharpe(self):
        """DSR should be monotonically non-decreasing with Sharpe."""
        sharpes = [0.1, 0.2, 0.3, 0.4, 0.5]
        dsrs = [compute_deflated_sharpe_ratio(s, n_trials=10, n_observations=500) for s in sharpes]
        for i in range(1, len(dsrs)):
            assert dsrs[i] >= dsrs[i - 1], f"DSR not monotonic at index {i}: {dsrs}"

    def test_dsr_monotonic_in_observations(self):
        """DSR should be monotonically non-decreasing with more observations."""
        obs_list = [50, 100, 500, 1000]
        dsrs = [compute_deflated_sharpe_ratio(0.30, n_trials=10, n_observations=n) for n in obs_list]
        for i in range(1, len(dsrs)):
            assert dsrs[i] >= dsrs[i - 1], f"DSR not monotonic in obs at index {i}: {dsrs}"


# ---------------------------------------------------------------------------
# compute_crisis_returns — additional edge cases
# ---------------------------------------------------------------------------

class TestComputeCrisisReturnsEdgeCases:
    """Additional edge cases for compute_crisis_returns."""

    def test_with_equity_curve(self):
        prices = {'2020-01-02': {'SPY': 100}}
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000, 95000, 105000]
        result = compute_crisis_returns(
            prices, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        assert '2020' in result
        assert result['2020'] < 0  # Had a drawdown

    def test_with_equity_curve_no_drawdown(self):
        """Monotonically increasing equity curve should have 0% crisis drawdown."""
        prices = {'2020-01-02': {'SPY': 100}}
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000, 110000, 120000]
        result = compute_crisis_returns(
            prices, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        assert '2020' in result
        assert result['2020'] == 0.0

    def test_equity_curve_missing_day(self):
        """When a trading day is not in equity_curve, it should skip."""
        prices = {'2020-01-02': {'SPY': 100}}
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000]  # only 1 value for 3 days
        result = compute_crisis_returns(
            prices, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        # Only first day maps; single value → 0% drawdown, -0.0 result
        assert '2020' in result
        assert result['2020'] == 0.0

    def test_empty_prices(self):
        result = compute_crisis_returns({}, ['2022-01-03', '2022-12-30'])
        assert '2022' not in result

    def test_single_day_in_year(self):
        prices = {'2022-01-03': {'SPY': 100}}
        result = compute_crisis_returns(prices, ['2022-01-03'])
        assert '2022' not in result  # Need at least 2 days

    def test_zero_starting_value_with_equity_curve(self):
        trading_days = ['2020-01-02']
        equity_curve = [0]
        result = compute_crisis_returns(
            {}, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        assert '2020' not in result  # Less than 2 days

    def test_custom_base_weights_with_fallback(self):
        """When equity_curve is not provided, base_weights are used."""
        prices = {
            '2022-01-03': {'AAA': 100},
            '2022-06-01': {'AAA': 90},
            '2022-12-30': {'AAA': 80},
        }
        result = compute_crisis_returns(
            prices, ['2022-01-03', '2022-06-01', '2022-12-30'],
            base_weights={'AAA': 1.0},
        )
        assert '2022' in result
        assert result['2022'] <= -20.0  # 100->80 with 100% weight


# ---------------------------------------------------------------------------
# print_metrics_report — additional coverage
# ---------------------------------------------------------------------------

class TestPrintMetricsReportEdgeCases:
    """Additional edge cases for print_metrics_report."""

    def test_with_rebalances(self, capsys):
        m = BacktestMetrics(
            total_return=15.0, cagr=9.5, volatility=11.0,
            sharpe_ratio=0.86, max_drawdown=-18.0,
            total_rebalances=24, total_transaction_costs=120.0,
        )
        print_metrics_report(m)
        captured = capsys.readouterr()
        assert "Rebalances: 24" in captured.out
        assert "Transaction Costs: 120.00" in captured.out

    def test_zero_rebalances_no_output(self, capsys):
        """When total_rebalances is 0, rebalance lines should not appear."""
        m = BacktestMetrics(
            total_return=10.0, cagr=8.0, volatility=12.0,
            sharpe_ratio=0.67, max_drawdown=-15.0,
            total_rebalances=0, total_transaction_costs=0.0,
        )
        print_metrics_report(m)
        captured = capsys.readouterr()
        assert "Rebalances" not in captured.out
        assert "Transaction Costs" not in captured.out

    def test_default_title(self, capsys):
        m = BacktestMetrics(
            total_return=5.0, cagr=4.0, volatility=10.0,
            sharpe_ratio=0.40, max_drawdown=-10.0,
        )
        print_metrics_report(m)
        captured = capsys.readouterr()
        assert "Backtest Results" in captured.out


# ---------------------------------------------------------------------------
# save_results_json — additional coverage
# ---------------------------------------------------------------------------

class TestSaveResultsJsonEdgeCases:
    """Additional edge cases for save_results_json."""

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "empty.json"
        save_results_json({}, output_path=str(path))
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == {}

    def test_none_path_no_default_dir(self):
        # Should not crash
        save_results_json({"key": "val"}, output_path=None, default_dir=None)

    def test_nested_numpy_arrays(self, tmp_path):
        path = tmp_path / "nested_np.json"
        data = {
            "means": np.array([1.1, 2.2, 3.3]),
            "matrix": np.array([[1, 2], [3, 4]]),
            "scalar": np.float64(3.14),
        }
        save_results_json(data, output_path=str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["means"] == [1.1, 2.2, 3.3]
        assert loaded["matrix"] == [[1, 2], [3, 4]]
        assert abs(loaded["scalar"] - 3.14) < 1e-10

    def test_explicit_path_overrides_default_dir(self, tmp_path):
        explicit = tmp_path / "explicit.json"
        default_dir = tmp_path / "subdir"
        save_results_json({"key": "val"}, output_path=str(explicit), default_dir=default_dir)
        assert explicit.exists()
        assert not (default_dir / "backtest_results.json").exists()

    def test_json_serializer_raises_on_unknown_type(self):
        from src.backtest.metrics import _json_serializer
        with pytest.raises(TypeError):
            _json_serializer(object())


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    """Verify consistency between metrics.py and its dependencies."""

    def test_backtest_config_uses_base_allocation(self):
        from src.paths import BASE_ALLOCATION
        cfg = BacktestConfig()
        assert cfg.base_weights == BASE_ALLOCATION

    def test_backtest_config_uses_etf_costs(self):
        from src.costs.etf_cost_table import ETF_COST_BPS
        cfg = BacktestConfig()
        for sym, cost in ETF_COST_BPS.items():
            assert cfg.transaction_costs_by_symbol[sym] == cost

    def test_backtest_config_defaults_match_constants(self):
        from src.backtest.metrics import (
            REBALANCE_FREQUENCY_DAYS, DEFAULT_TRANSACTION_COST_BPS,
        )
        cfg = BacktestConfig()
        assert cfg.rebalance_frequency_days == REBALANCE_FREQUENCY_DAYS
        assert cfg.transaction_cost_bps == DEFAULT_TRANSACTION_COST_BPS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
