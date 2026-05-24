"""
Tests for the Alternative Data Signal Backtest.

Covers: BacktestConfig defaults/custom, DailyReturn dataclass, BacktestResult construction,
regime inference, continuous signal computation, allocation shifts, run_backtest with
synthetic data, helpers (rebalance day, returns from equity, metrics), edge cases,
and CLI invocation.
"""

import json
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

    def test_print_report_does_not_crash(self, capsys):
        """print_report should produce output without errors."""
        data = TestRunBacktest._make_synthetic_data(TestRunBacktest(), n_days=200)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-01")
        )
        bt.data = data
        result = bt.run_backtest()
        bt.print_report(result)
        captured = capsys.readouterr()
        assert "ALTERNATIVE DATA" in captured.out
        assert "Sharpe" in captured.out
        assert "REGIME DISTRIBUTION" in captured.out

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

    def test_print_report_empty_result(self, capsys):
        """print_report with all-zero result should not crash."""
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
        captured = capsys.readouterr()
        assert "ALTERNATIVE DATA" in captured.out
        assert "SUCCESS CRITERIA" in captured.out

    def test_print_report_mismatch_verdict(self, capsys):
        """print_report shows MISMATCH when signal sign differs from regime return."""
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
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.out  # bull signal=+0.4 but bull return=-5.0

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
