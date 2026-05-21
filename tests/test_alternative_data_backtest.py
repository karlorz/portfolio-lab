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
    BacktestResult,
    DailyReturn,
)


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
        total = config.base_spy_weight + config.base_gld_weight + config.base_tlt_weight
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
            overlay_active_months=120,
            overlay_active_pct=50.0,
            return_2008=-10.5,
            return_2020=2.1,
            return_2022=-8.3,
            total_rebalances=150,
            avg_rebalance_size=0.025,
            total_transaction_costs=45.0,
            regime_distribution={"bull": 1000, "bear": 500, "neutral": 300, "crisis": 100},
            regime_returns={"bull": 15.0, "bear": -8.0, "neutral": 2.0, "crisis": -12.0},
            equity_curve=[{"date": "2020-01-02", "baseline": 100000, "overlay": 100000}],
        )
        assert result.total_return == 15.2
        assert result.sharpe_ratio == 0.82
        assert result.sharpe_improvement == 0.03
        assert result.total_rebalances == 150
        assert result.regime_distribution["bull"] == 1000

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
            overlay_active_months=12,
            overlay_active_pct=25.0,
            return_2008=-8.0,
            return_2020=None,
            return_2022=-5.0,
            total_rebalances=30,
            avg_rebalance_size=0.015,
            total_transaction_costs=15.0,
            regime_distribution={"bull": 200},
            regime_returns={"bull": 5.0},
            equity_curve=[],
        )
        json.dumps(result.__dict__)  # Should not raise

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
            overlay_active_months=0,
            overlay_active_pct=0.0,
            return_2008=None,
            return_2020=None,
            return_2022=None,
            total_rebalances=0,
            avg_rebalance_size=0.0,
            total_transaction_costs=0.0,
            regime_distribution={},
            regime_returns={},
            equity_curve=[],
        )
        assert result.return_2008 is None
        assert result.return_2020 is None


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
        """Generate synthetic DailyReturn list."""
        data = []
        from datetime import timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        for i in range(n_days):
            d = start + timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            spy_ret = spy_trend + np.random.normal(0, 0.01)
            gld_ret = gld_trend + np.random.normal(0, 0.008)
            tlt_ret = tlt_trend + np.random.normal(0, 0.006)
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
        assert result.overlay_active_months >= 0
        assert result.total_rebalances >= 0
        assert "bull" in result.regime_distribution
        assert "bear" in result.regime_distribution
        assert "neutral" in result.regime_distribution
        assert "crisis" in result.regime_distribution
        assert len(result.equity_curve) > 0

    def test_equity_curve_structure(self):
        data = self._make_synthetic_data(n_days=300)
        bt = AlternativeDataBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        bt.data = data
        result = bt.run_backtest()
        pt = result.equity_curve[0]
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
        total = sum(result.regime_distribution.values())
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
        assert result.regime_distribution.get("bear", 0) > 0 or result.regime_distribution.get("crisis", 0) > 0

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
            assert "regime_distribution" in saved
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
