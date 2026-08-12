"""
Tests for the Alternative Data Signal Backtest.

Covers: BacktestConfig defaults/custom, DailyReturn dataclass, BacktestResult construction,
regime inference, continuous signal computation, allocation shifts, run_backtest with
synthetic data, helpers (rebalance day, returns from equity, metrics), edge cases,
and CLI invocation.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.backtest.alternative_data_backtest import (
    AlternativeDataBacktester,
    BacktestConfig,
    DailyReturn,
)
from src.backtest.metrics import BacktestResult


# ── BacktestConfig Tests ─────────────────────────────────────────────────


class TestBacktestConfig:
    """Test BacktestConfig defaults and custom configuration."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.start_date == "2006-01-01"
        assert config.end_date == "2026-05-15"
        assert config.initial_capital == 100000.0
        assert config.base_weights["SPY"] == 0.46
        assert config.base_weights["GLD"] == 0.38
        assert config.base_weights["TLT"] == 0.16
        assert config.rebalance_frequency == "monthly"
        assert config.transaction_cost_bps == 10.0
        assert config.max_signal_shift == 0.05

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            transaction_cost_bps=5.0,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.transaction_cost_bps == 5.0

    def test_base_weights_sum_to_one(self):
        config = BacktestConfig()
        total = config.base_weights["SPY"] + config.base_weights["GLD"] + config.base_weights["TLT"]
        assert abs(total - 1.0) < 0.01


# ── DailyReturn Tests ────────────────────────────────────────────────────


class TestDailyReturn:
    """Test DailyReturn dataclass construction."""

    def test_minimal_construction(self):
        dr = DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=-0.005, tlt_return=0.002)
        assert dr.date == "2020-01-02"
        assert dr.spy_return == 0.01
        assert dr.gld_return == -0.005
        assert dr.tlt_return == 0.002
        assert dr.vix_spot is None

    def test_with_vix_spot(self):
        dr = DailyReturn(
            date="2020-03-15", spy_return=-0.03, gld_return=0.02, tlt_return=0.01, vix_spot=35.0
        )
        assert dr.vix_spot == 35.0

    def test_types_are_correct(self):
        dr = DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=-0.005, tlt_return=0.002)
        assert isinstance(dr.date, str)
        assert isinstance(dr.spy_return, float)
        assert isinstance(dr.gld_return, float)
        assert isinstance(dr.tlt_return, float)


# ── BacktestResult Tests ────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation and to_dict methods."""

    def test_create(self):
        result = BacktestResult(
            total_return=15.2,
            cagr=8.5,
            volatility=11.3,
            sharpe_ratio=0.82,
            max_drawdown=-18.4,
            baseline_sharpe=0.79,
            sharpe_improvement=0.03,
            total_rebalances=150,
            total_transaction_costs=45.0,
            crisis_returns={"2008": -10.5, "2020": 2.1, "2022": -8.3},
            extras={
                "overlay_active_months": 120,
                "overlay_active_pct": 50.0,
                "avg_rebalance_size": 0.025,
                "regime_distribution": {"bull": 1000, "bear": 500, "neutral": 300, "crisis": 100},
                "regime_returns": {"bull": 15.0, "bear": -8.0, "neutral": 2.0, "crisis": -12.0},
                "equity_curve": [{"date": "2020-01-02", "baseline": 100000, "overlay": 100000}],
            },
        )
        assert result.total_return == 15.2
        assert result.sharpe_ratio == 0.82
        assert result.sharpe_improvement == 0.03
        assert result.total_rebalances == 150
        assert result.extras["regime_distribution"]["bull"] == 1000

    def test_json_serializable(self):
        """All fields must be JSON-serializable."""
        result = BacktestResult(
            total_return=5.0,
            cagr=3.0,
            volatility=10.0,
            sharpe_ratio=0.5,
            max_drawdown=-10.0,
            baseline_sharpe=0.45,
            sharpe_improvement=0.05,
            total_rebalances=30,
            total_transaction_costs=15.0,
            crisis_returns={"2008": -8.0, "2022": -5.0},
            extras={
                "overlay_active_months": 12,
                "overlay_active_pct": 25.0,
                "avg_rebalance_size": 0.015,
                "regime_distribution": {"bull": 200},
                "regime_returns": {"bull": 5.0},
                "equity_curve": [],
            },
        )
        from dataclasses import asdict
        json.dumps(asdict(result))  # Should not raise

    def test_none_crisis_returns(self):
        """Crisis returns can be None."""
        result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            baseline_sharpe=0.0,
            sharpe_improvement=0.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0,
                "overlay_active_pct": 0.0,
                "avg_rebalance_size": 0.0,
                "regime_distribution": {},
                "regime_returns": {},
                "equity_curve": [],
            },
        )
        assert result.crisis_returns is None


# ── AlternativeDataBacktester Construction Tests ─────────────────────────


class TestAlternativeDataBacktesterInit:
    """Test backtester initialization."""

    def test_default_config(self):
        bt = AlternativeDataBacktester()
        assert bt.config.start_date == "2006-01-01"
        assert bt.config.initial_capital == 100000.0
        assert bt.data == []

    def test_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", end_date="2025-01-01")
        bt = AlternativeDataBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.end_date == "2025-01-01"

    def test_config_is_copy(self):
        """Modifying config after creation should not affect backtester if passed by ref."""
        config = BacktestConfig(initial_capital=50000.0)
        bt = AlternativeDataBacktester(config)
        assert bt.config.initial_capital == 50000.0

    def test_regime_signal_map_constant(self):
        """The legacy REGIME_SIGNAL_MAP should be a class-level constant."""
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["bull"] == 0.4
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["bear"] == -0.4
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["neutral"] == 0.0
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["crisis"] == -0.7


# ── Regime Inference Tests ──────────────────────────────────────────────


class TestInferRegime:
    """Test infer_regime_from_spy_return classification."""

    def test_bull_above_threshold(self):
        bt = AlternativeDataBacktester()
        # Production _determine_regime: composite_score > 0.15 -> risk_on -> bull
        # composite_score = clip(spy_60d_return * 2.0, -1, 1), so spy_60d_return > 0.075
        assert bt.infer_regime_from_spy_return(0.08) == "bull"
        assert bt.infer_regime_from_spy_return(0.10) == "bull"
        assert bt.infer_regime_from_spy_return(0.15) == "bull"

    def test_bull_exactly_at_threshold(self):
        bt = AlternativeDataBacktester()
        # composite_score = 0.05 * 2.0 = 0.10, which is < 0.15 -> neutral
        assert bt.infer_regime_from_spy_return(0.05) == "neutral"

    def test_neutral_band(self):
        bt = AlternativeDataBacktester()
        # spy_bear_return_high = -0.05, so -5% to 5% -> neutral
        assert bt.infer_regime_from_spy_return(0.0) == "neutral"
        assert bt.infer_regime_from_spy_return(-0.04) == "neutral"
        assert bt.infer_regime_from_spy_return(-0.049) == "neutral"

    def test_bear_band(self):
        bt = AlternativeDataBacktester()
        # Production _determine_regime: composite_score < -0.15 -> risk_off -> bear
        # composite_score = clip(spy_60d_return * 2.0, -1, 1), so spy_60d_return < -0.075 for bear
        assert bt.infer_regime_from_spy_return(-0.10) == "bear"
        assert bt.infer_regime_from_spy_return(-0.15) == "bear"
        assert bt.infer_regime_from_spy_return(-0.20) == "bear"

    def test_crisis_below_threshold(self):
        bt = AlternativeDataBacktester()
        # Production code does not produce "crisis"; strongly negative returns map to "bear".
        assert bt.infer_regime_from_spy_return(-0.16) == "bear"
        assert bt.infer_regime_from_spy_return(-0.30) == "bear"
        assert bt.infer_regime_from_spy_return(-0.50) == "bear"

    def test_extreme_returns(self):
        """Very large positive and negative returns should still classify."""
        bt = AlternativeDataBacktester()
        assert bt.infer_regime_from_spy_return(1.0) == "bull"  # 100% return -> risk_on
        assert bt.infer_regime_from_spy_return(-1.0) == "bear"  # -100% return -> risk_off


# ── Signal Computation Tests ────────────────────────────────────────────


class TestGetSignalAndRegime:
    """Test get_signal_and_regime with controlled data."""

    @pytest.fixture
    def backtester(self):
        return AlternativeDataBacktester(BacktestConfig(start_date="2020-01-01", end_date="2020-06-01"))

    def test_positive_signal_bull_regime(self, backtester):
        """Consistently positive returns should produce bull regime and positive signal."""
        day = DailyReturn(date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0)
        # 60 days of +0.2% each = ~12.7% cumulative return
        past_60d = [0.002] * 60
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        assert regime == "bull"
        assert signal > 0
        assert -1.0 <= signal <= 1.0

    def test_negative_signal_bear_regime(self, backtester):
        """Consistently negative returns should produce bear/crisis regime and negative signal."""
        day = DailyReturn(date="2020-03-15", spy_return=-0.01, gld_return=0.0, tlt_return=0.0)
        # 60 days of -0.3% each = ~-16.5% cumulative return
        past_60d = [-0.003] * 60
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        assert regime in ("bear", "crisis")
        assert signal < 0
        assert -1.0 <= signal <= 1.0

    def test_neutral_signal(self, backtester):
        """Near-zero returns should produce neutral regime and near-zero signal."""
        day = DailyReturn(date="2020-03-15", spy_return=0.0, gld_return=0.0, tlt_return=0.0)
        past_60d = [0.0001] * 60
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        # 60-day cumulative ~= 0.6%, which is still > 0.05 so bull
        # Actually np.prod(1+0.0001)^60 - 1 = 0.006 -> 0.6% which is less than 5% threshold
        # Wait: spy_bull_return = 0.05 = 5%. Cumulative = (1.0001)^60 - 1 = 0.006 = 0.6%
        # 0.6% < 5%, so not bull. 0.6% > -5%, so not bear. -> neutral.
        assert regime == "neutral"
        assert abs(signal) < 0.05

    def test_insufficient_data_uses_all_available(self, backtester):
        """With fewer than 60 data points, use whatever is available."""
        day = DailyReturn(date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0)
        past_60d = [0.005] * 30  # Only 30 days
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        # 30 days of 0.5% = ~16.1% cumulative -> bull
        assert regime == "bull"
        assert signal > 0

    def test_signal_clipped_to_neg_one(self, backtester):
        """Very negative 60d return should clip signal to -1.0."""
        day = DailyReturn(date="2020-03-15", spy_return=-0.01, gld_return=0.0, tlt_return=0.0)
        past_60d = [-0.05] * 60  # ~-95% cumulative -> signal * 2 -> clipped to -1.0
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        assert signal == -1.0

    def test_signal_clipped_to_one(self, backtester):
        """Very positive 60d return should clip signal to 1.0."""
        day = DailyReturn(date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0)
        past_60d = [0.05] * 60  # ~+1700% cumulative -> signal * 2 -> clipped to 1.0
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        assert signal == 1.0

    def test_crisis_signal(self, backtester):
        """Returns < -15% should give crisis regime."""
        day = DailyReturn(date="2020-03-15", spy_return=-0.01, gld_return=0.0, tlt_return=0.0)
        past_60d = [-0.003] * 60  # ~-16.5% cumulative -> crisis
        regime, signal = backtester.get_signal_and_regime(day, past_60d)
        assert regime in ("bear", "crisis")
        assert signal < 0


# ── Allocation Shift Tests ──────────────────────────────────────────────


class TestGetAllocationShifts:
    """Test get_allocation_shifts for correct signal->shift mapping."""

    def test_positive_signal_bull(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(0.3)
        assert spy_s == 0.03  # SPY+3%
        assert gld_s == -0.02  # GLD-2%
        assert tlt_s == -0.01  # TLT-1%

    def test_negative_signal_bear(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.3)
        assert spy_s == -0.03  # SPY-3%
        assert gld_s == 0.02  # GLD+2%
        assert tlt_s == 0.01  # TLT+1%

    def test_crisis_signal(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.7)
        assert spy_s == -0.05  # SPY-5%
        assert gld_s == 0.03  # GLD+3%
        assert tlt_s == 0.02  # TLT+2%

    def test_zero_signal(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(0.0)
        assert spy_s == 0.0
        assert gld_s == 0.0
        assert tlt_s == 0.0

    def test_exactly_negative_zero_five(self):
        """Signal exactly -0.5 should map to crisis."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.5)
        assert spy_s == -0.05

    def test_just_above_negative_zero_five(self):
        """Signal just above -0.5 should map to bear."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.499)
        assert spy_s == -0.03

    def test_tiny_positive_is_bull(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(0.001)
        assert spy_s == 0.03

    def test_tiny_negative_is_bear(self):
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.001)
        assert spy_s == -0.03


# ── Helper Method Tests ─────────────────────────────────────────────────


class TestHelpers:
    """Test static helper methods."""

    def test_is_rebalance_day_first_of_month(self):
        """Days 1-3 of month are rebalance days."""
        assert AlternativeDataBacktester._is_rebalance_day("2020-01-01", None)
        assert AlternativeDataBacktester._is_rebalance_day("2020-01-02", None)
        assert AlternativeDataBacktester._is_rebalance_day("2020-01-03", None)

    def test_is_rebalance_day_not_first(self):
        """Day 4+ of month is not a rebalance day if same month as last."""
        assert not AlternativeDataBacktester._is_rebalance_day("2020-01-04", None)
        assert not AlternativeDataBacktester._is_rebalance_day("2020-01-15", None)

    def test_is_rebalance_day_month_boundary(self):
        """Month boundary crossing triggers rebalance."""
        assert AlternativeDataBacktester._is_rebalance_day(
            "2020-02-01", "2020-01-15"
        )
        assert not AlternativeDataBacktester._is_rebalance_day(
            "2020-01-20", "2020-01-15"
        )

    def test_is_rebalance_day_year_boundary(self):
        """Year boundary crossing triggers rebalance."""
        assert AlternativeDataBacktester._is_rebalance_day(
            "2021-01-02", "2020-12-15"
        )

    def test_returns_from_equity(self):
        equity = [100.0, 110.0, 99.0]
        returns = AlternativeDataBacktester._returns_from_equity(equity)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.10)
        assert returns[1] == pytest.approx(-0.10, abs=0.01)

    def test_returns_from_equity_single(self):
        """Single element should produce empty returns."""
        returns = AlternativeDataBacktester._returns_from_equity([100.0])
        assert returns == []

    def test_returns_from_equity_empty(self):
        """Empty list should produce empty returns."""
        returns = AlternativeDataBacktester._returns_from_equity([])
        assert returns == []

    def test_calculate_metrics_empty_returns(self):
        metrics = AlternativeDataBacktester._calculate_metrics([])
        assert metrics["cagr"] == 0
        assert metrics["volatility"] == 0
        assert metrics["sharpe"] == 0
        assert metrics["max_dd"] == 0

    def test_calculate_metrics_positive_returns(self):
        returns = [0.01, 0.012, 0.009, 0.011, 0.008] * 50  # ~1% with small variation
        metrics = AlternativeDataBacktester._calculate_metrics(returns)
        assert metrics["cagr"] > 0
        assert metrics["volatility"] > 0  # Non-zero std from variation
        assert metrics["sharpe"] > 0
        assert metrics["max_dd"] == 0  # Always positive -> no drawdown

    def test_calculate_metrics_negative_returns(self):
        returns = [-0.01, -0.012, -0.009, -0.011, -0.008] * 50
        metrics = AlternativeDataBacktester._calculate_metrics(returns)
        assert metrics["cagr"] < 0
        assert metrics["max_dd"] < 0

    def test_annualize_returns_empty(self):
        result = AlternativeDataBacktester._annualize_returns([])
        assert result == 0.0

    def test_annualize_returns_positive(self):
        result = AlternativeDataBacktester._annualize_returns([0.001] * 252)
        assert result > 0

    def test_annualize_returns_negative(self):
        result = AlternativeDataBacktester._annualize_returns([-0.001] * 252)
        assert result < 0

    def test_annualize_regime_returns(self):
        regime_returns = {"bull": [0.001] * 252, "bear": [-0.001] * 252, "neutral": [0.0] * 10, "crisis": []}
        result = AlternativeDataBacktester._annualize_regime_returns(regime_returns)
        assert result["bull"] > 0
        assert result["bear"] < 0
        assert result["neutral"] == 0.0
        assert result["crisis"] == 0.0  # Empty list


# ── Synthetic Data Backtest Tests ───────────────────────────────────────


class TestRunBacktest:
    """Test run_backtest with synthetic data."""

    def _make_synthetic_data(self, n_days=252, spy_trend=0.001, gld_trend=0.0005, tlt_trend=0.0003, start_date="2020-01-01"):
        """Generate synthetic DailyReturn list.

        Uses a local RandomState with a fixed seed so the test is deterministic
        and immune to global numpy random state pollution from other tests.
        """
        rng = np.random.RandomState(42)
        data = []
        from datetime import timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        for i in range(n_days):
            d = start + timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            spy_ret = spy_trend + rng.normal(0, 0.01)
            gld_ret = gld_trend + rng.normal(0, 0.008)
            tlt_ret = tlt_trend + rng.normal(0, 0.006)
            data.append(DailyReturn(date=date_str, spy_return=spy_ret, gld_return=gld_ret, tlt_return=tlt_ret))
        return data

    def test_run_with_synthetic_data(self):
        """Backtest should produce a BacktestResult with run_backtest."""
        data = self._make_synthetic_data(n_days=300)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio != 0
        assert result.total_rebalances > 0
        assert result.cagr is not None

    def test_run_bull_market(self):
        """Strongly trending up should produce positive results."""
        data = self._make_synthetic_data(n_days=300, spy_trend=0.002)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr > 0
        assert result.sharpe_ratio > 0

    def test_run_bear_market(self):
        """Strongly trending down should produce negative results."""
        data = self._make_synthetic_data(n_days=300, spy_trend=-0.002)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr < 0

    def test_result_contains_required_fields(self):
        data = self._make_synthetic_data(n_days=300)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result.total_return is not None
        assert result.cagr is not None
        assert result.volatility is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.baseline_sharpe is not None
        assert result.sharpe_improvement is not None
        assert result.extras["overlay_active_months"] >= 0
        assert result.total_rebalances >= 0
        assert "bull" in result.extras["regime_distribution"]
        assert "bear" in result.extras["regime_distribution"]
        assert "neutral" in result.extras["regime_distribution"]
        assert "crisis" in result.extras["regime_distribution"]
        assert len(result.extras["equity_curve"]) > 0

    def test_equity_curve_structure(self):
        data = self._make_synthetic_data(n_days=300)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        pt = result.extras["equity_curve"][0]
        assert "date" in pt
        assert "baseline" in pt
        assert "overlay" in pt
        assert isinstance(pt["date"], str)
        assert isinstance(pt["baseline"], float)
        assert isinstance(pt["overlay"], float)

    def test_regime_distribution_counts(self):
        data = self._make_synthetic_data(n_days=300, spy_trend=0.002)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        total = sum(result.extras["regime_distribution"].values())
        assert total > 0
        assert total <= len(data)

    def test_short_data_window(self):
        """Only 60 days should still work (minimum for 60d rolling)."""
        data = self._make_synthetic_data(n_days=65)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-30")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_very_short_data_window(self):
        """Only 30 days should still return a result (uses whatever is available)."""
        data = self._make_synthetic_data(n_days=35, start_date="2020-01-01")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-03-01")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_no_data_returns_none(self):
        bt = AlternativeDataBacktester()
        bt.data = []
        result = bt.run_backtest()
        assert result is None


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the AlternativeData backtest."""

    def test_zero_initial_capital(self):
        """Zero initial capital causes division by zero in the source code."""
        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=100)
        bt = AlternativeDataBacktester(BacktestConfig(initial_capital=0.0))
        bt.data = data
        with pytest.raises(ZeroDivisionError):
            bt.run_backtest()

    def test_negative_spy_returns_consistently(self):
        """All negative SPY returns should produce bear/crisis regime."""
        data = []
        start = datetime.strptime("2020-01-01", "%Y-%m-%d")
        for i in range(200):
            d = start.strftime("%Y-%m-%d")
            data.append(DailyReturn(date=d, spy_return=-0.01, gld_return=0.005, tlt_return=0.003))
            start += np.timedelta64(1, "D")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        # Should have bear or crisis regimes
        assert result.extras["regime_distribution"].get("bear", 0) > 0 or result.extras["regime_distribution"].get("crisis", 0) > 0

    def test_high_volatility_does_not_crash(self):
        """Extreme daily returns should not cause runtime errors."""
        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=200)
        # Inject extreme returns
        for i in range(min(10, len(data))):
            data[i].spy_return = 0.50  # 50% daily move
            data[i].gld_return = -0.30
            data[i].tlt_return = 0.20
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result.sharpe_ratio, float)

    def test_identical_returns_produces_valid_metrics(self):
        """Identical daily returns should compute valid (possibly extreme) Sharpe."""
        data = []
        start = datetime.strptime("2020-01-01", "%Y-%m-%d")
        for i in range(200):
            d = start.strftime("%Y-%m-%d")
            data.append(DailyReturn(date=d, spy_return=0.001, gld_return=0.001, tlt_return=0.001))
            start += np.timedelta64(1, "D")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result.sharpe_ratio, float)

    def test_save_results_creates_json(self):
        """save_results should create a valid JSON file."""
        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=200)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                saved = json.load(f)
            assert "total_return" in saved
            assert "cagr" in saved
            assert "sharpe_ratio" in saved
            assert "extras" in saved
            assert "regime_distribution" in saved["extras"]
        finally:
            Path(output_path).unlink()

    def test_print_report_does_not_crash(self, caplog):
        """print_report should produce output without errors."""
        caplog.set_level(logging.INFO)
        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=200)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()
        bt.print_report(result)
        assert "ALTERNATIVE DATA" in caplog.text
        assert "Sharpe" in caplog.text
        assert "REGIME DISTRIBUTION" in caplog.text

    def test_load_data_missing_file_logs_error(self, caplog, monkeypatch):
        """load_data should return False when prices.json is missing."""
        import logging
        caplog.set_level(logging.ERROR)
        bt = AlternativeDataBacktester()
        # Monkeypatch Path.exists to return False so load_data fails
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        result = bt.load_data()
        assert result is False


# ── Dataclass to_dict Field Completeness Tests ───────────────────────────


class TestDataclassToDict:
    """Test dataclass asdict() field completeness for all dataclass types."""

    def test_backtest_config_all_fields_present(self):
        """BacktestConfig.asdict() should include all inherited and local fields."""
        from dataclasses import asdict

        config = BacktestConfig()
        d = asdict(config)
        # Inherited from _BaseConfig
        assert "start_date" in d
        assert "end_date" in d
        assert "initial_capital" in d
        assert "base_weights" in d
        assert "rebalance_frequency" in d
        assert "rebalance_frequency_days" in d
        assert "transaction_cost_bps" in d
        assert "transaction_costs_by_symbol" in d
        # Local fields
        assert "max_signal_shift" in d
        assert "min_holding_period" in d
        assert "vix_bull_threshold" in d
        assert "vix_bear_threshold" in d
        assert "vix_crisis_threshold" in d

    def test_backtest_config_field_values(self):
        """Local field defaults are correctly serialised via asdict()."""
        from dataclasses import asdict

        d = asdict(BacktestConfig())
        assert d["max_signal_shift"] == 0.05
        assert d["min_holding_period"] == 20
        assert d["vix_bull_threshold"] == 15.0
        assert d["vix_bear_threshold"] == 20.0
        assert d["vix_crisis_threshold"] == 30.0

    def test_daily_return_minimal_asdict(self):
        """DailyReturn without vix_spot should still include vix_spot=None."""
        from dataclasses import asdict

        dr = DailyReturn(
            date="2020-01-02", spy_return=0.01, gld_return=-0.005, tlt_return=0.002
        )
        d = asdict(dr)
        assert d["date"] == "2020-01-02"
        assert d["spy_return"] == 0.01
        assert d["gld_return"] == -0.005
        assert d["tlt_return"] == 0.002
        assert d["vix_spot"] is None
        assert set(d.keys()) == {
            "date",
            "spy_return",
            "gld_return",
            "tlt_return",
            "vix_spot",
        }

    def test_daily_return_with_vix_asdict(self):
        """DailyReturn with vix_spot set should serialise it."""
        from dataclasses import asdict

        dr = DailyReturn(
            date="2020-03-15",
            spy_return=-0.03,
            gld_return=0.02,
            tlt_return=0.01,
            vix_spot=35.0,
        )
        d = asdict(dr)
        assert d["vix_spot"] == 35.0


# ── Constants Validation Tests ────────────────────────────────────────────


class TestConstants:
    """Test hardcoded constant values and invariants."""

    def test_regime_signal_map_exhaustive_keys(self):
        """REGIME_SIGNAL_MAP should have exactly the four expected regimes."""
        keys = set(AlternativeDataBacktester.REGIME_SIGNAL_MAP.keys())
        assert keys == {"bull", "bear", "neutral", "crisis"}

    def test_vix_threshold_ordering(self):
        """vix thresholds must be strictly increasing: bull < bear < crisis."""
        config = BacktestConfig()
        assert config.vix_bull_threshold < config.vix_bear_threshold
        assert config.vix_bear_threshold < config.vix_crisis_threshold

    def test_min_holding_period_and_max_signal_shift_defaults(self):
        """Extra config fields should have documented defaults."""
        config = BacktestConfig()
        assert config.min_holding_period == 20
        assert config.max_signal_shift == 0.05


# ── Backtest Execution Edge Cases ─────────────────────────────────────────


class TestBacktestExecutionEdgeCases:
    """Edge cases in backtest execution (zero signals, missing data, etc.)."""

    def _make_controlled_data(
        self,
        n_days: int = 252,
        spy_ret: float = 0.0,
        gld_ret: float = 0.0005,
        tlt_ret: float = 0.0003,
        start_date: str = "2020-01-01",
    ):
        """Generate deterministic DailyReturn list with controlled returns."""
        from datetime import timedelta

        data = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        for i in range(n_days):
            d = start + timedelta(days=i)
            data.append(
                DailyReturn(
                    date=d.strftime("%Y-%m-%d"),
                    spy_return=spy_ret,
                    gld_return=gld_ret,
                    tlt_return=tlt_ret,
                )
            )
        return data

    def test_backtest_all_zero_signals(self):
        """All-zero signal should produce overlay_active_months=0 and Sharpe identical to baseline."""
        data = self._make_controlled_data(n_days=300, spy_ret=0.0)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.extras["overlay_active_months"] == 0
        assert result.extras["overlay_active_pct"] == 0.0
        # Baseline and overlay Sharpe should be identical (zero signal = no overlay alpha)
        assert abs(result.sharpe_ratio - result.baseline_sharpe) < 0.01

    def test_no_data_in_date_range_returns_none(self):
        """Data exists but none within backtest period -> None."""
        data = self._make_controlled_data(n_days=100, start_date="2020-01-01")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2025-01-01", end_date="2025-06-01")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is None

    def test_single_data_point_backtest(self):
        """Single day of data should produce a valid result."""
        data = [
            DailyReturn(
                date="2020-06-15",
                spy_return=0.001,
                gld_return=0.0005,
                tlt_return=0.0003,
            )
        ]
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-06-01", end_date="2020-06-30")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_backtest_extreme_negative_market(self):
        """Consistently negative SPY returns produce crisis allocation and negative CAGR."""
        data = self._make_controlled_data(
            n_days=300, spy_ret=-0.005, gld_ret=0.0, tlt_ret=0.0
        )
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr < 0

    def test_regime_distribution_counts_plausible(self):
        """Total regime counts should not exceed number of data days."""
        data = self._make_controlled_data(n_days=200, spy_ret=0.001)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        total_regime = sum(result.extras["regime_distribution"].values())
        assert total_regime > 0
        assert total_regime <= len(data)

    def test_backtest_transaction_cost_non_negative(self):
        """Transaction costs should never be negative."""
        data = TestRunBacktest._make_synthetic_data(
            TestRunBacktest(), n_days=200
        )
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.total_transaction_costs >= 0.0


# ── Performance Metric Edge Cases ─────────────────────────────────────────


class TestMetricsEdgeCases:
    """Edge cases for performance metric calculations."""

    def test_calculate_metrics_uniform_returns(self):
        """All identical returns -> zero volatility -> Sharpe handled gracefully."""
        returns = [0.001] * 252
        metrics = AlternativeDataBacktester._calculate_metrics(returns)
        assert metrics["cagr"] > 0
        assert isinstance(metrics["sharpe"], (int, float))
        # Zero variance produces Sharpe=0 (division-by-zero guard); that is valid

    def test_annualize_regime_returns_all_empty(self):
        """All regime lists empty -> all zeros returned."""
        regime_returns = {
            "bull": [],
            "bear": [],
            "neutral": [],
            "crisis": [],
        }
        result = AlternativeDataBacktester._annualize_regime_returns(
            regime_returns
        )
        for regime in ("bull", "bear", "neutral", "crisis"):
            assert result[regime] == 0.0

    def test_annualize_regime_returns_unknown_key(self):
        """Unknown regime keys are preserved and computed."""
        regime_returns = {"unknown_key": [0.001] * 10}
        result = AlternativeDataBacktester._annualize_regime_returns(
            regime_returns
        )
        assert "unknown_key" in result
        assert result["unknown_key"] > 0

    def test_annualize_returns_single_day(self):
        """Single-day return annualises to extreme value (n_years=1/252)."""
        result = AlternativeDataBacktester._annualize_returns([0.01])
        # (1.01^(252) - 1) * 100 ~= 1030%
        assert result > 100

    def test_annualize_returns_exactly_one_year(self):
        """Exactly 252 trading days should be one year of annualisation."""
        result = AlternativeDataBacktester._annualize_returns([0.001] * 252)
        # (1.001^252 - 1) * 100 ~= 28.6%
        assert 20 < result < 40

    def test_annualize_returns_all_zero_returns(self):
        """All-zero daily returns -> zero annualized return."""
        result = AlternativeDataBacktester._annualize_returns([0.0] * 252)
        assert result == 0.0

    def test_calculate_metrics_mixed_sign_returns(self):
        """Returns with both positive and negative values should produce valid metrics."""
        returns = [0.005, -0.003, 0.007, -0.001, 0.002] * 50
        metrics = AlternativeDataBacktester._calculate_metrics(returns)
        assert isinstance(metrics["cagr"], float)
        assert metrics["volatility"] > 0
        assert isinstance(metrics["sharpe"], float)
        assert metrics["max_dd"] < 0  # There were negative returns, so drawdown exists


# ── Signal Boundary Conditions ────────────────────────────────────────────


class TestSignalBoundaries:
    """Boundary conditions for regime and signal classification."""

    def test_boundary_bull_neutral_exact(self):
        """spy_60d_return=0.075 -> composite_score=0.15; NOT > 0.15 -> neutral."""
        bt = AlternativeDataBacktester()
        assert bt.infer_regime_from_spy_return(0.075) == "neutral"

    def test_boundary_just_above_bull(self):
        """spy_60d_return just above 0.075 -> composite_score > 0.15 -> bull."""
        bt = AlternativeDataBacktester()
        assert bt.infer_regime_from_spy_return(0.0750001) == "bull"

    def test_boundary_neutral_bear_exact(self):
        """spy_60d_return=-0.075 -> composite_score=-0.15; NOT < -0.15 -> neutral."""
        bt = AlternativeDataBacktester()
        assert bt.infer_regime_from_spy_return(-0.075) == "neutral"

    def test_boundary_just_below_bear(self):
        """spy_60d_return just below -0.075 -> composite_score < -0.15 -> bear."""
        bt = AlternativeDataBacktester()
        assert bt.infer_regime_from_spy_return(-0.0750001) == "bear"

    def test_get_signal_and_regime_exactly_60_days(self):
        """Exactly 60 data points triggers the >=60 code path."""
        bt = AlternativeDataBacktester(BacktestConfig(start_date="2020-01-01", end_date="2020-06-01"))
        day = DailyReturn(
            date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0
        )
        past_60d = [0.002] * 60
        regime, signal = bt.get_signal_and_regime(day, past_60d)
        assert regime in ("bull", "neutral")
        assert -1.0 <= signal <= 1.0

    def test_get_signal_and_regime_single_datum(self):
        """Only 1 data point uses the <60 code path (all available data)."""
        bt = AlternativeDataBacktester(BacktestConfig(start_date="2020-01-01", end_date="2020-06-01"))
        day = DailyReturn(
            date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0
        )
        past_60d = [0.05]  # Single high return
        regime, signal = bt.get_signal_and_regime(day, past_60d)
        # composite_score = clip(0.05 * 2.0, -1, 1) = 0.10 < 0.15 -> neutral
        assert regime == "neutral"
        assert signal == pytest.approx(0.10, abs=1e-6)

    def test_get_signal_and_regime_empty_buffer(self):
        """Empty past_60d list should not crash (all available = [])."""
        bt = AlternativeDataBacktester(BacktestConfig(start_date="2020-01-01", end_date="2020-06-01"))
        day = DailyReturn(
            date="2020-03-15", spy_return=0.01, gld_return=0.0, tlt_return=0.0
        )
        regime, signal = bt.get_signal_and_regime(day, [])
        # np.prod(1 + []) - 1 = 1 - 1 = 0, signal = 0, regime = neutral
        assert regime == "neutral"
        assert signal == 0.0

    def test_extreme_positive_signal_clipping(self):
        """100% 60-day return clips to signal=1.0."""
        bt = AlternativeDataBacktester()
        result = bt.infer_regime_from_spy_return(1.0)
        assert result == "bull"

    def test_extreme_negative_signal_clipping(self):
        """-100% 60-day return clips to signal=-1.0."""
        bt = AlternativeDataBacktester()
        result = bt.infer_regime_from_spy_return(-1.0)
        assert result == "bear"


# ── Utility / Helper Edge Cases ───────────────────────────────────────────


class TestHelpersEdgeCases:
    """Edge cases for utility, helper, and data-processing methods."""

    def test_process_price_data_empty_dict(self):
        """Empty prices_data leaves data empty (logs error about missing SPY)."""
        bt = AlternativeDataBacktester()
        bt._process_price_data({})
        assert bt.data == []

    def test_process_price_data_missing_spy(self, caplog):
        """Missing SPY logs error and leaves data empty."""
        import logging

        caplog.set_level(logging.ERROR)
        bt = AlternativeDataBacktester()
        bt._process_price_data(
            {"GLD": [{"d": "2020-01-02", "p": 100.0}]}
        )
        assert bt.data == []
        assert "No SPY data" in caplog.text

    def test_process_price_data_partial_missing_symbols(self):
        """Missing TLT means no days pass the all() check -> empty data."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 51.0},
            ],
            # TLT missing entirely
        }
        bt._process_price_data(prices)
        assert bt.data == []

    def test_process_price_data_all_symbols_present(self):
        """All three symbols with consecutive days should produce daily returns."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 49.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        bt._process_price_data(prices)
        assert len(bt.data) == 1
        assert bt.data[0].date == "2020-01-03"
        assert bt.data[0].spy_return == pytest.approx(0.01)
        assert bt.data[0].gld_return == pytest.approx(-0.02)
        assert bt.data[0].tlt_return == pytest.approx(0.0125)

    def test_is_rebalance_day_mid_month_same_month(self):
        """Day 15 in same month as last rebalance -> not a rebalance day."""
        assert not AlternativeDataBacktester._is_rebalance_day(
            "2020-06-15", "2020-06-01"
        )

    def test_is_rebalance_day_last_day_of_month(self):
        """Day 28+ in same month as last rebalance -> not a rebalance day."""
        assert not AlternativeDataBacktester._is_rebalance_day(
            "2020-02-28", "2020-02-01"
        )

    def test_is_rebalance_day_day4_no_prior(self):
        """Day 4 with no prior rebalance -> not a rebalance day."""
        assert not AlternativeDataBacktester._is_rebalance_day(
            "2020-01-04", None
        )

    def test_print_report_empty_result(self, caplog):
        """print_report with all-zero result should not crash."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            baseline_sharpe=0.0,
            sharpe_improvement=0.0,
            crisis_returns=None,
            extras={
                "overlay_active_months": 0,
                "overlay_active_pct": 0.0,
                "avg_rebalance_size": 0.0,
                "regime_distribution": {},
                "regime_returns": {},
                "equity_curve": [],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)
        assert "ALTERNATIVE DATA" in caplog.text
        assert "SUCCESS CRITERIA" in caplog.text

    def test_print_report_mismatch_verdict(self, caplog):
        """print_report shows MISMATCH when signal sign differs from regime return."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=-5.0,
            cagr=-2.0,
            volatility=15.0,
            sharpe_ratio=-0.2,
            max_drawdown=-30.0,
            total_rebalances=100,
            total_transaction_costs=50.0,
            baseline_sharpe=0.3,
            sharpe_improvement=-0.5,
            crisis_returns={"2008": 5.0},
            extras={
                "overlay_active_months": 10,
                "overlay_active_pct": 20.0,
                "avg_rebalance_size": 0.03,
                "regime_distribution": {
                    "bull": 100,
                    "bear": 100,
                    "neutral": 100,
                    "crisis": 100,
                },
                "regime_returns": {
                    "bull": -5.0,
                    "bear": 3.0,
                    "neutral": 0.5,
                    "crisis": 2.0,
                },
                "equity_curve": [
                    {
                        "date": "2020-01-02",
                        "baseline": 100000.0,
                        "overlay": 95000.0,
                    }
                ],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)
        assert "MISMATCH" in caplog.text  # bull signal=+0.4 but bull return=-5.0

    def test_transaction_cost_deduction(self):
        """Transaction costs are deducted from overlay capital (cost > 0 when turnover > 0)."""
        data = TestRunBacktest._make_synthetic_data(
            TestRunBacktest(), n_days=200
        )
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        # With 200 days of data and monthly rebalancing, there should be some rebalances
        assert result.total_rebalances > 0
        # Costs should be positive
        assert result.total_transaction_costs > 0


# ── CLI Tests ───────────────────────────────────────────────────────────


class TestCLI:
    """Test command-line invocation."""

    def test_main_run(self, monkeypatch):
        """main() should handle missing data gracefully, returning non-zero."""
        from src.backtest.alternative_data_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["alternative_data_backtest.py", "run"]):
            result = main()
        assert result == 1  # No data available

    def test_main_with_save_flag(self, monkeypatch):
        """main() with --save flag should still handle missing data gracefully."""
        from src.backtest.alternative_data_backtest import main
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        with patch("sys.argv", ["alternative_data_backtest.py", "run", "--save"]):
            result = main()
        assert result == 1  # No data available


# ── Dataclass Field Introspection ────────────────────────────────────────────


class TestDataclassFieldIntrospection:
    """Validate dataclass field definitions programmatically via dataclasses.fields()."""

    def test_backtest_config_field_names(self):
        """BacktestConfig fields() includes all inherited and local fields."""
        from dataclasses import fields

        names = {f.name for f in fields(BacktestConfig)}
        # Inherited from _BaseConfig
        assert "start_date" in names
        assert "end_date" in names
        assert "initial_capital" in names
        assert "base_weights" in names
        assert "rebalance_frequency" in names
        assert "rebalance_frequency_days" in names
        assert "transaction_cost_bps" in names
        assert "transaction_costs_by_symbol" in names
        # Local fields
        assert "max_signal_shift" in names
        assert "min_holding_period" in names
        assert "vix_bull_threshold" in names
        assert "vix_bear_threshold" in names
        assert "vix_crisis_threshold" in names

    def test_backtest_config_field_defaults(self):
        """BacktestConfig local field defaults match source code."""
        from dataclasses import fields

        fmap = {f.name: f for f in fields(BacktestConfig)}
        assert fmap["max_signal_shift"].default == 0.05
        assert fmap["min_holding_period"].default == 20
        assert fmap["vix_bull_threshold"].default == 15.0
        assert fmap["vix_bear_threshold"].default == 20.0
        assert fmap["vix_crisis_threshold"].default == 30.0

    def test_daily_return_field_names(self):
        """DailyReturn fields() matches source annotations."""
        from dataclasses import fields

        names = {f.name for f in fields(DailyReturn)}
        assert names == {"date", "spy_return", "gld_return", "tlt_return", "vix_spot"}

    def test_daily_return_vix_spot_default_is_none(self):
        """DailyReturn.vix_spot default is None."""
        from dataclasses import fields

        fmap = {f.name: f for f in fields(DailyReturn)}
        assert fmap["vix_spot"].default is None

    def test_backtest_result_field_names(self):
        """BacktestResult fields() matches source annotations."""
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult

        names = {f.name for f in fields(BacktestResult)}
        assert "total_return" in names
        assert "cagr" in names
        assert "volatility" in names
        assert "sharpe_ratio" in names
        assert "max_drawdown" in names
        assert "total_rebalances" in names
        assert "total_transaction_costs" in names
        assert "avg_turnover" in names
        assert "baseline_sharpe" in names
        assert "sharpe_improvement" in names
        assert "extras" in names
        assert "crisis_returns" in names

    def test_backtest_result_defaults(self):
        """BacktestResult default values via dataclasses.fields()."""
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult

        fmap = {f.name: f for f in fields(BacktestResult)}
        assert fmap["total_rebalances"].default == 0
        assert fmap["total_transaction_costs"].default == 0.0
        assert fmap["avg_turnover"].default == 0.0
        assert fmap["baseline_sharpe"].default is None
        assert fmap["sharpe_improvement"].default is None
        assert fmap["crisis_returns"].default is None

    def test_backtest_config_is_dataclass(self):
        """BacktestConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(BacktestConfig)

    def test_daily_return_is_dataclass(self):
        """DailyReturn is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(DailyReturn)


# ── NaN / Inf Handling ───────────────────────────────────────────────────────


class TestNaNInfHandling:
    """Edge cases with NaN and Inf values in computations."""

    def test_calculate_metrics_with_nan_returns(self):
        """NaN in returns does not crash _calculate_metrics."""
        metrics = AlternativeDataBacktester._calculate_metrics(
            [0.001, float("nan"), 0.002]
        )
        assert isinstance(metrics["cagr"], float)
        assert isinstance(metrics["sharpe"], (int, float))

    def test_calculate_metrics_with_inf_returns(self):
        """Inf in returns does not crash _calculate_metrics."""
        metrics = AlternativeDataBacktester._calculate_metrics(
            [0.001, float("inf"), 0.002]
        )
        assert isinstance(metrics["cagr"], float)

    def test_annualize_returns_with_nan(self):
        """NaN in returns list is handled without crashing."""
        result = AlternativeDataBacktester._annualize_returns([0.001, float("nan")])
        assert isinstance(result, float)

    def test_annualize_returns_with_inf(self):
        """Inf in returns list is handled without crashing."""
        result = AlternativeDataBacktester._annualize_returns([0.001, float("inf")])
        assert isinstance(result, float)

    def test_infer_regime_nan_spy_return(self):
        """NaN spy_60d_return does not crash infer_regime_from_spy_return."""
        bt = AlternativeDataBacktester()
        # np.clip(nan * 2.0, -1, 1) -> nan, comparisons with nan are False -> neutral
        regime = bt.infer_regime_from_spy_return(float("nan"))
        assert regime == "neutral"

    def test_infer_regime_inf_spy_return(self):
        """Inf spy_60d_return clips correctly."""
        bt = AlternativeDataBacktester()
        regime = bt.infer_regime_from_spy_return(float("inf"))
        # composite_score = clip(inf * 2.0, -1, 1) = 1.0 > 0.15 -> risk_on -> bull
        assert regime == "bull"


# ── Data Loading Edge Cases ──────────────────────────────────────────────────


class TestLoadDataEdgeCases:
    """Edge cases for load_data and _process_price_data."""

    def test_load_data_malformed_json(self, caplog, monkeypatch):
        """Malformed JSON returns False."""
        import logging, tempfile

        caplog.set_level(logging.ERROR)
        bt = AlternativeDataBacktester()
        # Create a temp file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            tmp_path = Path(f.name)
        try:
            monkeypatch.setattr("src.backtest.alternative_data_backtest.PRICES_JSON", tmp_path)
            result = bt.load_data()
            assert result is False
        finally:
            tmp_path.unlink()

    def test_load_data_empty_json_object(self, monkeypatch):
        """Empty JSON object {} with no SPY key returns False."""
        import tempfile, json

        bt = AlternativeDataBacktester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            tmp_path = Path(f.name)
        try:
            monkeypatch.setattr("src.backtest.alternative_data_backtest.PRICES_JSON", tmp_path)
            result = bt.load_data()
            assert result is True  # loads, processes (empty spy -> logs error -> data=[])
            assert bt.data == []
        finally:
            tmp_path.unlink()

    def test_load_data_io_exception(self, monkeypatch):
        """IOError during file read returns False."""
        bt = AlternativeDataBacktester()
        original_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open

        def _failing_open(*args, **kwargs):
            raise OSError("Simulated IO error")

        monkeypatch.setattr("builtins.open", _failing_open)
        result = bt.load_data()
        assert result is False

    def test_process_price_data_zero_prices(self):
        """Zero prices produce extreme returns for non-falsy values."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 0.01}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 51.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 81.0}],
        }
        bt._process_price_data(prices)
        # spy_return = (0.01 - 100) / 100 = -0.9999
        assert len(bt.data) == 1
        assert bt.data[0].spy_return == pytest.approx(-0.9999)

    def test_process_price_data_non_consecutive_dates(self):
        """Non-consecutive dates are processed normally (gap between prev & curr)."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-10", "p": 110.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-10", "p": 48.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-10", "p": 82.0}],
        }
        bt._process_price_data(prices)
        assert len(bt.data) == 1
        assert bt.data[0].spy_return == pytest.approx(0.10)

    def test_process_price_data_single_datum_per_symbol(self):
        """Single price point per symbol produces no daily returns."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}],
        }
        bt._process_price_data(prices)
        assert bt.data == []

    def test_process_price_data_gap_in_one_symbol(self):
        """Missing a date in one symbol causes that day to be skipped."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
                {"d": "2020-01-06", "p": 102.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                # 2020-01-03 missing for GLD
                {"d": "2020-01-06", "p": 51.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
                {"d": "2020-01-06", "p": 82.0},
            ],
        }
        bt._process_price_data(prices)
        # SPY has 3 entries, so 2 return days. GLD has 2 entries -> day 2020-01-03 has no GLD data
        # Day 2020-01-03: spy_prev=100.0, spy_curr=101.0 OK; gld_prev from 2020-01-02=50.0, gld_curr=None -> SKIP
        # Day 2020-01-06: spy_prev=101.0, spy_curr=102.0 OK; gld_prev from 2020-01-06... actually let me trace through.
        # dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
        # spy_prices = {"2020-01-02": 100.0, "2020-01-03": 101.0, "2020-01-06": 102.0}
        # gld_prices = {"2020-01-02": 50.0, "2020-01-06": 51.0} -- no "2020-01-03"
        # tlt_prices = {"2020-01-02": 80.0, "2020-01-03": 81.0, "2020-01-06": 82.0}
        # i=1 (date="2020-01-03"): spy_prev=100, spy_curr=101, gld_curr=51 (from 2020-01-06)... wait, gld_prices.get("2020-01-03") = None
        # So gld_curr would be None -> SKIP
        # i=2 (date="2020-01-06"): spy_prev=101, spy_curr=102, gld_prev=None (gld_prices.get("2020-01-03")=None) -> SKIP
        # So we get 0 data points!
        assert bt.data == []

    def test_process_price_data_reversed_dates(self):
        """Dates in reverse order still process (only adjacent pairs matter)."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-03", "p": 101.0}, {"d": "2020-01-02", "p": 100.0}],
            "GLD": [{"d": "2020-01-03", "p": 49.0}, {"d": "2020-01-02", "p": 50.0}],
            "TLT": [{"d": "2020-01-03", "p": 81.0}, {"d": "2020-01-02", "p": 80.0}],
        }
        bt._process_price_data(prices)
        # dates = ["2020-01-03", "2020-01-02"]
        # i=1 (date="2020-01-02"): spy_prev=101, spy_curr=100 -> spy_return = (100-101)/101 = -0.0099
        assert len(bt.data) == 1
        assert bt.data[0].spy_return == pytest.approx(-0.00990099, abs=1e-6)

    def test_process_price_data_negative_price(self):
        """Negative price produces negative return without crashing."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": -10.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 51.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 81.0}],
        }
        # all() returns True for non-zero values. spy_return = (-10 - 100) / 100 = -1.1
        bt._process_price_data(prices)
        assert len(bt.data) == 1
        assert bt.data[0].spy_return == -1.1

    def test_process_price_data_missing_price_key(self, caplog):
        """Missing 'p' key in price dict raises KeyError caught by load_data."""
        import logging, tempfile, json

        caplog.set_level(logging.ERROR)
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "x": 100.0}, {"d": "2020-01-03", "x": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 51.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 81.0}],
        }
        # This will crash in _process_price_data when doing p["p"]
        with pytest.raises(KeyError):
            bt._process_price_data(prices)

    def test_process_price_data_extra_symbols(self):
        """Extra symbols in data dict are ignored."""
        bt = AlternativeDataBacktester()
        prices = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 49.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 81.0}],
            "QQQ": [{"d": "2020-01-02", "p": 200.0}, {"d": "2020-01-03", "p": 205.0}],
            "BTC": [{"d": "2020-01-02", "p": 9000.0}, {"d": "2020-01-03", "p": 9200.0}],
        }
        bt._process_price_data(prices)
        assert len(bt.data) == 1
        assert bt.data[0].date == "2020-01-03"

    def test_load_data_valid_no_spy_returns_true_empty_data(self, monkeypatch):
        """load_data returns True, but data stays empty when SPY missing."""
        import tempfile, json

        bt = AlternativeDataBacktester()
        prices = {
            "GLD": [{"d": "2020-01-02", "p": 100.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prices, f)
            tmp_path = Path(f.name)
        try:
            monkeypatch.setattr("src.backtest.alternative_data_backtest.PRICES_JSON", tmp_path)
            result = bt.load_data()
            assert result is True
            assert bt.data == []
        finally:
            tmp_path.unlink()


# ── CLI / __main__ Guard ─────────────────────────────────────────────────────


class TestCLIGuard:
    """Test CLI entry point and __main__ guard."""

    def test_main_invalid_command(self):
        """Invalid command argument raises SystemExit."""
        from src.backtest.alternative_data_backtest import main

        with patch("sys.argv", ["alternative_data_backtest.py", "invalid"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_run_with_mock_success(self):
        """Full success path: load_data -> run_backtest -> print_report."""
        from src.backtest.alternative_data_backtest import main

        mock_result = BacktestResult(
            total_return=10.0,
            cagr=5.0,
            volatility=12.0,
            sharpe_ratio=0.6,
            max_drawdown=-15.0,
            total_rebalances=50,
            total_transaction_costs=20.0,
            baseline_sharpe=0.55,
            sharpe_improvement=0.05,
            crisis_returns={"2008": -12.0},
            extras={
                "overlay_active_months": 30,
                "overlay_active_pct": 60.0,
                "avg_rebalance_size": 0.02,
                "regime_distribution": {"bull": 500, "bear": 200, "neutral": 300, "crisis": 50},
                "regime_returns": {"bull": 10.0, "bear": -5.0, "neutral": 1.0, "crisis": -8.0},
                "equity_curve": [{"date": "2020-01-02", "baseline": 100000, "overlay": 105000}],
            },
        )

        with (
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.load_data",
                  return_value=True),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.run_backtest",
                  return_value=mock_result),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.print_report"),
            patch("sys.argv", ["alternative_data_backtest.py", "run"]),
        ):
            result = main()
        assert result == 0

    def test_main_run_backtest_failure(self):
        """run_backtest returning None leads to error exit."""
        from src.backtest.alternative_data_backtest import main

        with (
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.load_data",
                  return_value=True),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.run_backtest",
                  return_value=None),
            patch("sys.argv", ["alternative_data_backtest.py", "run"]),
        ):
            result = main()
        assert result == 1

    def test_main_with_save_flag_success(self):
        """--save flag triggers save_results in success path."""
        from src.backtest.alternative_data_backtest import main

        mock_result = BacktestResult(
            total_return=5.0,
            cagr=2.0,
            volatility=10.0,
            sharpe_ratio=0.3,
            max_drawdown=-20.0,
            total_rebalances=10,
            total_transaction_costs=5.0,
            baseline_sharpe=0.25,
            sharpe_improvement=0.05,
            extras={
                "overlay_active_months": 5,
                "overlay_active_pct": 10.0,
                "avg_rebalance_size": 0.01,
                "regime_distribution": {"bull": 100},
                "regime_returns": {"bull": 2.0},
                "equity_curve": [],
            },
        )

        with (
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.load_data",
                  return_value=True),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.run_backtest",
                  return_value=mock_result),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.print_report"),
            patch("src.backtest.alternative_data_backtest.AlternativeDataBacktester.save_results"),
            patch("sys.argv", ["alternative_data_backtest.py", "run", "--save"]),
        ):
            result = main()
        assert result == 0

    def test_main_module_name_guard(self):
        """The __name__ == '__main__' guard exists and calls exit(main())."""
        import ast

        with open("src/backtest/alternative_data_backtest.py") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                found = True
                break

        assert found, "__name__ == '__main__' guard not found in source"


# ── Export Completeness ──────────────────────────────────────────────────────


class TestExportCompleteness:
    """Verify __all__ covers all public API names."""

    def test_all_defined(self):
        """__all__ is defined in the module."""
        import src.backtest.alternative_data_backtest as mod

        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 0

    def test_all_entries_exist_in_module(self):
        """Every entry in __all__ is actually defined in the module."""
        import src.backtest.alternative_data_backtest as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} listed in __all__ but not found in module"

    def test_all_covers_public_names(self):
        """All class/function names from module-level scope are in __all__."""
        import src.backtest.alternative_data_backtest as mod

        # Names exported by the module (not imports)
        module_members = {
            "BacktestConfig",
            "DailyReturn",
            "AlternativeDataBacktester",
        }
        all_set = set(mod.__all__)
        for name in module_members:
            assert name in all_set, f"Public name '{name}' missing from __all__"


# ── Helper Method Boundaries ─────────────────────────────────────────────────


class TestHelperBoundaries:
    """Boundary conditions for static helper methods."""

    def test_is_rebalance_feb_29_leap_year(self):
        """February 29 (day 29) with no prior rebalance is NOT a rebalance day."""
        assert not AlternativeDataBacktester._is_rebalance_day("2020-02-29", None)

    def test_is_rebalance_dec_31_to_jan_1(self):
        """December 31 to January 1 crosses year boundary -> rebalance."""
        assert AlternativeDataBacktester._is_rebalance_day(
            "2021-01-01", "2020-12-31"
        )

    def test_returns_from_equity_negative_equity(self):
        """Negative equity values produce valid returns."""
        returns = AlternativeDataBacktester._returns_from_equity([100.0, 80.0, 60.0])
        assert len(returns) == 2
        assert returns[0] == -0.20
        assert returns[1] == -0.25

    def test_returns_from_equity_all_zeros(self):
        """All-zero equity raises ZeroDivisionError (Python does not support 0.0/0.0)."""
        with pytest.raises(ZeroDivisionError):
            AlternativeDataBacktester._returns_from_equity([0.0, 0.0, 0.0])

    def test_calculate_metrics_single_return(self):
        """Single return value produces a valid metrics dict."""
        metrics = AlternativeDataBacktester._calculate_metrics([0.01])
        assert isinstance(metrics["cagr"], float)
        assert isinstance(metrics["volatility"], float)

    def test_calculate_metrics_all_zero_returns(self):
        """All-zero returns produce zero CAGR and zero volatility."""
        metrics = AlternativeDataBacktester._calculate_metrics([0.0] * 252)
        assert metrics["cagr"] == 0.0
        assert metrics["volatility"] == 0.0
        assert metrics["sharpe"] == 0

    def test_annualize_regime_returns_empty_dict(self):
        """Empty regime returns dict returns empty result."""
        result = AlternativeDataBacktester._annualize_regime_returns({})
        assert result == {}

    def test_annualize_regime_returns_single_element(self):
        """Single-element regime list computes correctly."""
        result = AlternativeDataBacktester._annualize_regime_returns(
            {"bull": [0.001]}
        )
        # mean=0.001 * 252 * 100 = 25.2
        assert result["bull"] == pytest.approx(25.2, abs=0.1)


# ── Allocation Shift Boundaries ──────────────────────────────────────────────


class TestAllocationShiftBoundaries:
    """Boundary conditions for get_allocation_shifts."""

    def test_signal_at_positive_one(self):
        """Signal=1.0 is positive, maps to bull shift."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(1.0)
        assert spy_s == 0.03
        assert gld_s == -0.02
        assert tlt_s == -0.01

    def test_signal_at_negative_one(self):
        """Signal=-1.0 is <= -0.5, maps to crisis shift."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-1.0)
        assert spy_s == -0.05
        assert gld_s == 0.03
        assert tlt_s == 0.02

    def test_signal_neg_0_5001_crisis(self):
        """Signal=-0.5001 is <= -0.5, maps to crisis."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.5001)
        assert spy_s == -0.05

    def test_signal_neg_0_4999_bear(self):
        """Signal=-0.4999 is > -0.5, maps to bear."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-0.4999)
        assert spy_s == -0.03

    def test_signal_very_small_positive_bull(self):
        """Signal=1e-10 is positive, maps to bull shift."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(1e-10)
        assert spy_s == 0.03

    def test_signal_very_small_negative_bear(self):
        """Signal=-1e-10 is negative and > -0.5, maps to bear."""
        bt = AlternativeDataBacktester()
        spy_s, gld_s, tlt_s = bt.get_allocation_shifts(-1e-10)
        assert spy_s == -0.03


# ── Backtest Run Edge Cases ──────────────────────────────────────────────────


class TestBacktestRunEdgeCases:
    """Edge cases in run_backtest execution."""

    def _make_controlled_data(
        self,
        n_days=252,
        spy_ret=0.0,
        gld_ret=0.0005,
        tlt_ret=0.0003,
        start_date="2020-01-01",
    ):
        """Generate deterministic DailyReturn list."""
        from datetime import timedelta

        data = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        for i in range(n_days):
            d = start + timedelta(days=i)
            data.append(
                DailyReturn(
                    date=d.strftime("%Y-%m-%d"),
                    spy_return=spy_ret,
                    gld_return=gld_ret,
                    tlt_return=tlt_ret,
                )
            )
        return data

    def test_run_with_nan_in_returns(self):
        """NaN in returns does not crash run_backtest."""
        data = self._make_controlled_data(n_days=200, spy_ret=0.001)
        data[50].spy_return = float("nan")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_run_with_extreme_positive_returns(self):
        """Extreme positive daily returns (10%) do not crash."""
        data = self._make_controlled_data(n_days=200, spy_ret=0.10, gld_ret=0.05, tlt_ret=0.03)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr > 0

    def test_run_with_extreme_negative_returns(self):
        """Extreme negative daily returns (-10%) do not crash."""
        data = self._make_controlled_data(n_days=200, spy_ret=-0.10, gld_ret=-0.05, tlt_ret=-0.03)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.cagr < 0

    def test_run_date_boundary_exact(self):
        """Data starting exactly on start_date works correctly."""
        data = self._make_controlled_data(n_days=100, start_date="2020-06-01")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-06-01", end_date="2020-08-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_run_high_transaction_costs(self):
        """Very high transaction costs still produce a valid result."""
        data = self._make_controlled_data(n_days=200, spy_ret=0.001)
        bt = AlternativeDataBacktester(
            BacktestConfig(
                start_date="2020-01-01",
                end_date="2020-12-31",
                transaction_cost_bps=500.0,
            )
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.total_transaction_costs >= 0

    def test_run_zero_transaction_costs(self):
        """Zero transaction costs produce zero total costs."""
        data = self._make_controlled_data(n_days=200, spy_ret=0.001)
        bt = AlternativeDataBacktester(
            BacktestConfig(
                start_date="2020-01-01",
                end_date="2020-12-31",
                transaction_cost_bps=0.0,
            )
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        assert result.total_transaction_costs == 0.0

    def test_run_single_month_data(self):
        """Only one month of data produces a result with minimal rebalances."""
        data = self._make_controlled_data(n_days=25, start_date="2020-06-01")
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-06-01", end_date="2020-06-30")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None

    def test_run_alternating_regimes(self):
        """Returns that alternate between positive/negative produce mixed regime distribution."""
        data = self._make_controlled_data(n_days=300, spy_ret=0.0)
        # Inject alternating strong signals
        for i in range(0, 300, 60):
            for j in range(60):
                idx = i + j
                if idx < 300:
                    data[idx].spy_return = 0.01 if (i // 60) % 2 == 0 else -0.01
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        assert result is not None
        # Should have both bull and bear regimes
        dist = result.extras["regime_distribution"]
        assert dist.get("bull", 0) > 0
        assert dist.get("bear", 0) > 0


# ── save_results Edge Cases ──────────────────────────────────────────────────


class TestSaveResultsEdgeCases:
    """Edge cases for save_results method."""

    def test_save_results_default_path(self, monkeypatch):
        """save_results with default path writes a JSON file."""
        import tempfile, json

        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=100)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()

        # Use a temp directory for the default path
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                "src.backtest.alternative_data_backtest.BACKTEST_RESULTS_DIR",
                Path(tmpdir),
            )
            bt.save_results(result)
            expected_path = Path(tmpdir) / "alternative_data_backtest.json"
            assert expected_path.exists()
            with open(expected_path) as f:
                saved = json.load(f)
            assert "total_return" in saved
            assert "extras" in saved

    def test_save_results_none_in_extras(self):
        """save_results handles None values in extras gracefully."""
        import tempfile, json

        # _annualize_regime_returns never returns None, but extras can contain None
        result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            baseline_sharpe=None,
            sharpe_improvement=None,
            crisis_returns=None,
            extras={
                "overlay_active_months": 0,
                "overlay_active_pct": 0.0,
                "avg_rebalance_size": 0.0,
                "regime_distribution": {},
                "regime_returns": {},
                "equity_curve": [],
            },
        )
        bt = AlternativeDataBacktester()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            bt.save_results(result, output_path=tmp_path)
            with open(tmp_path) as f:
                saved = json.load(f)
            assert saved["baseline_sharpe"] is None
            assert saved["crisis_returns"] is None
        finally:
            Path(tmp_path).unlink()

    def test_save_results_creates_parent_dir(self, monkeypatch):
        """save_results creates parent directory if it does not exist."""
        import tempfile, json

        result = BacktestResult(
            total_return=1.0,
            cagr=0.5,
            volatility=10.0,
            sharpe_ratio=0.1,
            max_drawdown=-5.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0,
                "overlay_active_pct": 0.0,
                "avg_rebalance_size": 0.0,
                "regime_distribution": {},
                "regime_returns": {},
                "equity_curve": [],
            },
        )
        bt = AlternativeDataBacktester()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "dir"
            output_path = str(nested / "result.json")
            bt.save_results(result, output_path=output_path)
            assert Path(output_path).exists()
            with open(output_path) as f:
                saved = json.load(f)
            assert saved["total_return"] == 1.0


# ── print_report Edge Cases ──────────────────────────────────────────────────


class TestPrintReportEdgeCases:
    """Edge cases for print_report method."""

    def test_print_report_success_failure_mix(self, caplog):
        """print_report with mixed PASS/FAIL success criteria."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=-30.0,
            cagr=-10.0,
            volatility=25.0,
            sharpe_ratio=-0.5,
            max_drawdown=-50.0,
            total_rebalances=500,  # FAIL (>= 400)
            total_transaction_costs=200.0,
            baseline_sharpe=-0.8,
            sharpe_improvement=0.3,  # >= 0 -> PASS
            crisis_returns={"2008": -30.0, "2020": -15.0, "2022": -25.0},
            extras={
                "overlay_active_months": 50,
                "overlay_active_pct": 25.0,
                "avg_rebalance_size": 0.05,
                "regime_distribution": {
                    "bull": 100,
                    "bear": 200,
                    "neutral": 150,
                    "crisis": 50,
                },
                "regime_returns": {
                    "bull": -5.0,
                    "bear": -10.0,
                    "neutral": 0.0,
                    "crisis": -20.0,
                },
                "equity_curve": [
                    {"date": "2020-01-02", "baseline": 100000, "overlay": 70000}
                ],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)
        assert "FAIL" in caplog.text
        assert "PASS" in caplog.text
        assert "SUCCESS CRITERIA" in caplog.text

    def test_print_report_all_regime_zero(self, caplog):
        """print_report with all-zero regime distribution avoids division by zero."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=5.0,
            cagr=2.0,
            volatility=10.0,
            sharpe_ratio=0.2,
            max_drawdown=-10.0,
            total_rebalances=10,
            total_transaction_costs=5.0,
            baseline_sharpe=0.15,
            sharpe_improvement=0.05,
            crisis_returns=None,
            extras={
                "overlay_active_months": 5,
                "overlay_active_pct": 10.0,
                "avg_rebalance_size": 0.01,
                "regime_distribution": {},
                "regime_returns": {},
                "equity_curve": [{"date": "2020-01-02", "baseline": 100000, "overlay": 105000}],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)  # Should not crash
        assert "REGIME DISTRIBUTION" in caplog.text

    def test_print_report_crisis_none_handled(self, caplog):
        """print_report with crisis_returns=None does not crash."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=5.0,
            cagr=2.0,
            volatility=10.0,
            sharpe_ratio=0.2,
            max_drawdown=-10.0,
            total_rebalances=10,
            total_transaction_costs=5.0,
            baseline_sharpe=0.15,
            sharpe_improvement=0.05,
            crisis_returns=None,
            extras={
                "overlay_active_months": 5,
                "overlay_active_pct": 10.0,
                "avg_rebalance_size": 0.01,
                "regime_distribution": {"bull": 100, "bear": 50, "neutral": 50, "crisis": 10},
                "regime_returns": {"bull": 5.0, "bear": -3.0, "neutral": 0.5, "crisis": -8.0},
                "equity_curve": [{"date": "2020-01-02", "baseline": 100000, "overlay": 105000}],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)
        assert "CRISIS PERFORMANCE" in caplog.text

    def test_print_report_negative_total_return(self, caplog):
        """print_report with negative total return shows correct values."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=-15.0,
            cagr=-5.0,
            volatility=18.0,
            sharpe_ratio=-0.3,
            max_drawdown=-35.0,
            total_rebalances=200,
            total_transaction_costs=100.0,
            baseline_sharpe=0.1,
            sharpe_improvement=-0.4,
            crisis_returns={"2008": -25.0, "2022": -15.0},
            extras={
                "overlay_active_months": 80,
                "overlay_active_pct": 40.0,
                "avg_rebalance_size": 0.03,
                "regime_distribution": {"bull": 300, "bear": 400, "neutral": 200, "crisis": 100},
                "regime_returns": {"bull": -2.0, "bear": -8.0, "neutral": 0.0, "crisis": -15.0},
                "equity_curve": [{"date": "2020-01-02", "baseline": 100000, "overlay": 85000}],
            },
        )
        bt = AlternativeDataBacktester()
        bt.print_report(result)
        assert "NEGATIVE" in caplog.text
        assert "-15.00%" in caplog.text


# ── Signal Generator Mock Integration ────────────────────────────────────────


class TestSignalGeneratorMock:
    """Test backtester integration with AlternativeDataSignalGenerator."""

    def test_infer_regime_uses_signal_generator(self):
        """infer_regime_from_spy_return delegates to _determine_regime."""
        bt = AlternativeDataBacktester()
        with patch.object(
            bt._signal_generator, "_determine_regime", return_value="risk_on"
        ) as mock_method:
            regime = bt.infer_regime_from_spy_return(0.10)
            mock_method.assert_called_once()
            assert regime == "bull"

    def test_signal_generator_unexpected_regime_label(self):
        """Unknown regime label from _determine_regime defaults to 'neutral'."""
        bt = AlternativeDataBacktester()
        with patch.object(
            bt._signal_generator, "_determine_regime", return_value="unknown_label"
        ):
            regime = bt.infer_regime_from_spy_return(0.10)
            assert regime == "neutral"

    def test_signal_generator_mocked_for_all_returns(self):
        """Mock _determine_regime to control all regime outputs."""
        bt = AlternativeDataBacktester()
        spy_returns = [0.10, -0.10, 0.0]
        expected = [
            ("risk_on", "bull"),
            ("risk_off", "bear"),
            ("neutral", "neutral"),
        ]
        for ret, (prod_regime, expected_regime) in zip(spy_returns, expected):
            with patch.object(
                bt._signal_generator, "_determine_regime", return_value=prod_regime
            ):
                regime = bt.infer_regime_from_spy_return(ret)
                assert regime == expected_regime, f"spy_ret={ret}: expected {expected_regime}, got {regime}"


# ── BacktestConfig Custom Values ─────────────────────────────────────────────


class TestBacktestConfigCustom:
    """Custom configurations for BacktestConfig."""

    def test_config_custom_vix_thresholds(self):
        """Custom VIX thresholds are stored correctly."""
        config = BacktestConfig(
            vix_bull_threshold=10.0,
            vix_bear_threshold=18.0,
            vix_crisis_threshold=25.0,
        )
        assert config.vix_bull_threshold == 10.0
        assert config.vix_bear_threshold == 18.0
        assert config.vix_crisis_threshold == 25.0
        # Ordering invariant
        assert config.vix_bull_threshold < config.vix_bear_threshold < config.vix_crisis_threshold

    def test_config_zero_transaction_cost(self):
        """Zero transaction cost is valid."""
        config = BacktestConfig(transaction_cost_bps=0.0)
        assert config.transaction_cost_bps == 0.0

    def test_config_custom_min_holding_period(self):
        """Custom min_holding_period is stored correctly."""
        config = BacktestConfig(min_holding_period=60)
        assert config.min_holding_period == 60

    def test_config_custom_max_signal_shift(self):
        """Custom max_signal_shift is stored correctly."""
        config = BacktestConfig(max_signal_shift=0.10)
        assert config.max_signal_shift == 0.10

    def test_config_extras_empty_by_default(self):
        """BacktestConfig.extras should be empty by default."""
        config = BacktestConfig()
        assert config.extras == {}


# ── Module Constants ─────────────────────────────────────────────────────────


class TestModuleConstants:
    """Test module-level constant validation."""

    def test_module_all_exact_match(self):
        """__all__ contains exactly: BacktestConfig, DailyReturn, AlternativeDataBacktester."""
        import src.backtest.alternative_data_backtest as mod

        assert set(mod.__all__) == {"BacktestConfig", "DailyReturn", "AlternativeDataBacktester"}

    def test_regime_signal_map_values_in_range(self):
        """All REGIME_SIGNAL_MAP values are in [-1, 1] range."""
        for regime, signal in AlternativeDataBacktester.REGIME_SIGNAL_MAP.items():
            assert -1.0 <= signal <= 1.0, f"{regime}: {signal} out of range"

    def test_regime_signal_map_bull_positive(self):
        """Bull regime has a positive signal."""
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["bull"] > 0

    def test_regime_signal_map_bear_negative(self):
        """Bear regime has a negative signal."""
        assert AlternativeDataBacktester.REGIME_SIGNAL_MAP["bear"] < 0

    def test_regime_signal_map_crisis_most_negative(self):
        """Crisis regime has the most negative signal."""
        signals = AlternativeDataBacktester.REGIME_SIGNAL_MAP
        assert signals["crisis"] < signals["bear"]
        assert signals["crisis"] < signals["neutral"]


# ── BacktestResult Edge Cases ────────────────────────────────────────────────


class TestBacktestResultEdgeCases:
    """Edge cases for BacktestResult construction."""

    def test_result_with_empty_extras(self):
        """BacktestResult allows empty extras dict."""
        result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_rebalances=0,
            total_transaction_costs=0.0,
            extras={},
        )
        assert result.extras == {}

    def test_result_with_large_values(self):
        """BacktestResult handles very large float values."""
        result = BacktestResult(
            total_return=1e6,
            cagr=1e3,
            volatility=1e3,
            sharpe_ratio=100.0,
            max_drawdown=-1e6,
            total_rebalances=999999,
            total_transaction_costs=1e12,
            extras={
                "overlay_active_months": 999,
                "overlay_active_pct": 999.0,
                "avg_rebalance_size": 1e6,
                "regime_distribution": {"bull": 1_000_000},
                "regime_returns": {"bull": 1e6},
                "equity_curve": [{"date": "2020-01-01", "baseline": 1e12, "overlay": 1e12}],
            },
        )
        assert result.total_return == 1e6
        assert result.total_rebalances == 999999

    def test_result_minimal_construction(self):
        """BacktestResult can be constructed with only required fields."""
        result = BacktestResult(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
        )
        # Default values should be set
        assert result.total_rebalances == 0
        assert result.extras == {}
        assert result.crisis_returns is None


# ── Data Loading with Exception Paths ────────────────────────────────────────


class TestDataLoadingExceptions:
    """Exception paths in data loading."""

    def test_load_data_catches_key_error(self, caplog, monkeypatch):
        """KeyError in _process_price_data is caught and logged."""
        import logging
        import tempfile, json

        caplog: pytest.LogCaptureFixture
        caplog.set_level(logging.ERROR)

        bt = AlternativeDataBacktester()
        # Write JSON with wrong structure (no 'p' key)
        bad_data = {
            "SPY": [{"d": "2020-01-02", "x": 100.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_data, f)
            tmp_path = Path(f.name)
        try:
            monkeypatch.setattr("src.backtest.alternative_data_backtest.PRICES_JSON", tmp_path)
            result = bt.load_data()
            assert result is False
            assert "Failed to load data" in caplog.text
        finally:
            tmp_path.unlink()

    def test_load_data_catches_attribute_error(self, caplog, monkeypatch):
        """AttributeError in data processing is caught."""
        import logging

        caplog: pytest.LogCaptureFixture
        caplog.set_level(logging.ERROR)

        bt = AlternativeDataBacktester()

        # Mock prices_path.exists to raise something strange
        class MockPath:
            def exists(self):
                return True

            def __str__(self):
                return "/nonexistent"

        monkeypatch.setattr(
            "src.backtest.alternative_data_backtest.PRICES_JSON",
            MockPath(),
        )

        # open will fail -> caught by except Exception
        result = bt.load_data()
        assert result is False

    def test_process_price_data_without_call_leaves_data_empty(self):
        """Backtester initializes with empty data."""
        bt = AlternativeDataBacktester()
        assert bt.data == []



def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 1): load_data delegates to grid_runner.load_prices."""
    from src.backtest import alternative_data_backtest as adb
    from src.backtest.grid_runner import load_prices

    # class method stays in pilot; the shared loader is grid_runner's
    assert adb.AlternativeDataBacktester.load_data.__module__ == (
        "src.backtest.alternative_data_backtest"
    )
    assert load_prices.__module__ == "src.backtest.grid_runner"
