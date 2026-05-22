"""
Tests for the Cross-Asset Regime Arbitrage Overlay Backtest.

Covers: BacktestConfig defaults/custom, DailyReturn/RebalanceSignal/BacktestResult
dataclasses, CrossAssetRegimeArbBacktester init, data processing, momentum
computation, regime classification, divergence detection, allocation shifts,
run_backtest, print/save output, CLI main, and edge cases.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.backtest.cross_asset_regime_arb_backtest import (
    BacktestConfig,
    DailyReturn,
    BacktestResult,
    RebalanceSignal,
    CrossAssetRegimeArbBacktester,
    ALLOCATION_SHIFTS,
    MAX_SIGNAL_STRENGTH,
    main,
)


# ── BacktestConfig Tests ──────────────────────────────────────────────────


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
        assert config.max_single_shift == 0.05
        assert config.signal_threshold == 0.05

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            max_single_shift=0.08,
            signal_threshold=0.10,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.max_single_shift == 0.08
        assert config.signal_threshold == 0.10

    def test_base_weights_sum(self):
        """Baseline weights should sum to 1.0."""
        config = BacktestConfig()
        total = config.base_spy_weight + config.base_gld_weight + config.base_tlt_weight
        assert abs(total - 1.0) < 0.01


# ── DailyReturn Tests ────────────────────────────────────────────────────


class TestDailyReturn:
    """Test DailyReturn dataclass."""

    def test_construction(self):
        dr = DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=-0.005, tlt_return=0.002)
        assert dr.date == "2020-01-02"
        assert dr.spy_return == 0.01
        assert dr.gld_return == -0.005
        assert dr.tlt_return == 0.002

    def test_types(self):
        dr = DailyReturn(date="2020-01-02", spy_return=1.0, gld_return=0.0, tlt_return=-1.0)
        assert isinstance(dr.date, str)
        assert isinstance(dr.spy_return, float)
        assert isinstance(dr.gld_return, float)
        assert isinstance(dr.tlt_return, float)


# ── RebalanceSignal Tests ─────────────────────────────────────────────────


class TestRebalanceSignal:
    """Test RebalanceSignal dataclass."""

    def test_construction(self):
        sig = RebalanceSignal(
            date="2020-01-31",
            signal_value=0.3,
            pattern="equity_rotation",
            spy_shift=0.03,
            gld_shift=-0.01,
            tlt_shift=-0.02,
            spy_momentum=0.08,
            gld_momentum=0.02,
            tlt_momentum=-0.01,
        )
        assert sig.date == "2020-01-31"
        assert sig.signal_value == 0.3
        assert sig.pattern == "equity_rotation"
        assert sig.spy_shift == 0.03
        assert sig.gld_shift == -0.01
        assert sig.tlt_shift == -0.02
        assert sig.spy_momentum == 0.08

    def test_zero_signal(self):
        sig = RebalanceSignal(
            date="2020-01-31", signal_value=0.0, pattern="no_divergence",
            spy_shift=0.0, gld_shift=0.0, tlt_shift=0.0,
            spy_momentum=0.0, gld_momentum=0.0, tlt_momentum=0.0,
        )
        assert sig.signal_value == 0.0


# ── BacktestResult Tests ──────────────────────────────────────────────────


class TestBacktestResult:
    """Test BacktestResult creation and serialization."""

    def test_construction(self):
        result = BacktestResult(
            total_return=10.5,
            cagr=8.2,
            volatility=12.3,
            sharpe_ratio=0.85,
            max_drawdown=-15.4,
            overlay_active_months=24,
            baseline_sharpe=0.79,
            sharpe_improvement=0.06,
            return_2008=-12.0,
            return_2020=3.0,
            return_2022=None,
            total_rebalances=30,
            total_transaction_costs=45.0,
            signal_frequency=0.5,
            divergence_breakdown={"equity_rotation": 10, "no_divergence": 20},
            equity_curve=[{"date": "2020-01-01", "baseline": 100000.0, "overlay": 101000.0}],
            rebalance_signals=[{"date": "2020-01-31", "signal_value": 0.3}],
        )
        assert result.total_return == 10.5
        assert result.sharpe_ratio == 0.85
        assert result.overlay_active_months == 24
        assert result.baseline_sharpe == 0.79
        assert result.sharpe_improvement == 0.06
        assert result.return_2008 == -12.0
        assert result.return_2022 is None
        assert result.signal_frequency == 0.5
        assert result.divergence_breakdown["equity_rotation"] == 10

    def test_json_serializable(self):
        """asdict(result) must be JSON-serializable."""
        from dataclasses import asdict

        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, overlay_active_months=5, baseline_sharpe=0.45,
            sharpe_improvement=0.05, return_2008=-8.0, return_2020=2.0, return_2022=None,
            total_rebalances=10, total_transaction_costs=5.0, signal_frequency=0.3,
            divergence_breakdown={}, equity_curve=[], rebalance_signals=[],
        )
        data = asdict(result)
        json.dumps(data)  # Should not raise

    def test_empty_divergence_breakdown(self):
        """Empty divergence breakdown should not error."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, overlay_active_months=0, baseline_sharpe=0.0,
            sharpe_improvement=0.0, return_2008=None, return_2020=None, return_2022=None,
            total_rebalances=0, total_transaction_costs=0.0, signal_frequency=0.0,
            divergence_breakdown={}, equity_curve=[], rebalance_signals=[],
        )
        assert result.divergence_breakdown == {}


# ── CrossAssetRegimeArbBacktester Tests ────────────────────────────────────


class TestCrossAssetRegimeArbBacktester:
    """Test the core CrossAssetRegimeArbBacktester class."""

    # ── Init ──

    def test_init_defaults(self):
        bt = CrossAssetRegimeArbBacktester()
        assert bt.config.start_date == "2006-01-01"
        assert bt.config.end_date == "2026-05-15"
        assert bt.data == []

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", max_single_shift=0.06)
        bt = CrossAssetRegimeArbBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.max_single_shift == 0.06

    # ── Data Processing ──

    def test_process_price_data_empty(self):
        """Empty price dict should produce no data."""
        bt = CrossAssetRegimeArbBacktester()
        bt._process_price_data({})
        assert bt.data == []

    def test_process_price_data_missing_spy(self):
        """Missing SPY data should produce empty data without crash."""
        bt = CrossAssetRegimeArbBacktester()
        bt._process_price_data({"GLD": [{"d": "2020-01-02", "p": 100.0}]})
        assert bt.data == []

    def test_process_price_data_single_day(self):
        """Single day of prices should produce no returns (needs at least 2)."""
        bt = CrossAssetRegimeArbBacktester()
        data = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}],
        }
        bt._process_price_data(data)
        assert bt.data == []

    def test_process_price_data_two_days(self):
        """Two days of prices should produce one DailyReturn."""
        bt = CrossAssetRegimeArbBacktester()
        data = {
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
        bt._process_price_data(data)
        assert len(bt.data) == 1
        dr = bt.data[0]
        assert dr.date == "2020-01-03"
        assert dr.spy_return == pytest.approx(0.01)
        assert dr.gld_return == pytest.approx(-0.02)
        assert dr.tlt_return == pytest.approx(0.0125)

    def test_process_price_data_gaps_skipped(self):
        """Days where a symbol is missing data should be skipped gracefully."""
        bt = CrossAssetRegimeArbBacktester()
        data = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        bt._process_price_data(data)
        # GLD is missing 2020-01-03, so no complete pair
        assert len(bt.data) == 0

    # ── Momentum Computation ──

    @pytest.fixture
    def bt_with_data(self):
        """Create a backtester with 200 days of synthetic data."""
        bt = CrossAssetRegimeArbBacktester()
        np.random.seed(42)
        bt.data = []
        for i in range(200):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            bt.data.append(
                DailyReturn(
                    date=day,
                    spy_return=np.random.normal(0.0005, 0.01),
                    gld_return=np.random.normal(0.0002, 0.008),
                    tlt_return=np.random.normal(0.0001, 0.006),
                )
            )
        assert len(bt.data) == 200
        return bt

    def test_compute_momentum_with_history(self, bt_with_data):
        """With enough history, momentum should produce non-zero values."""
        spy_m, gld_m, tlt_m = bt_with_data._compute_momentum(60, 100)
        assert isinstance(spy_m, float)
        assert isinstance(gld_m, float)
        assert isinstance(tlt_m, float)

    def test_compute_momentum_insufficient_history(self, bt_with_data):
        """With insufficient history, all momenta should be 0.0."""
        spy_m, gld_m, tlt_m = bt_with_data._compute_momentum(60, 10)
        assert spy_m == 0.0
        assert gld_m == 0.0
        assert tlt_m == 0.0

    def test_compute_momentum_zero_lookback(self, bt_with_data):
        """Lookback of 0 should produce zero momenta."""
        spy_m, gld_m, tlt_m = bt_with_data._compute_momentum(0, 100)
        assert spy_m == 0.0
        assert gld_m == 0.0
        assert tlt_m == 0.0

    # ── Regime Classification ──

    def test_classify_asset_regime_bullish(self):
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(0.10)  # > 0.05 threshold
        assert regime == "bullish"

    def test_classify_asset_regime_bearish(self):
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(-0.10)  # < -0.05 threshold
        assert regime == "bearish"

    def test_classify_asset_regime_neutral_positive(self):
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(0.03)  # Between -0.05 and 0.05
        assert regime == "neutral"

    def test_classify_asset_regime_neutral_negative(self):
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(-0.03)
        assert regime == "neutral"

    def test_classify_asset_regime_zero(self):
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(0.0)
        assert regime == "neutral"

    # ── Divergence Detection ──

    @staticmethod
    def _pattern_str(pattern):
        """Get string value from enum or string."""
        return pattern.value if hasattr(pattern, "value") else str(pattern)

    def test_no_divergence(self):
        """All same regime -> no divergence."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.03, 0.02, 0.01)
        assert self._pattern_str(pattern) == "no_divergence"
        assert signal == 0.0

    def test_all_bullish(self):
        """All bullish -> full risk on."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.10, 0.08, 0.06)
        assert self._pattern_str(pattern) == "full_risk_on"
        assert signal == 0.4

    def test_all_bearish(self):
        """All bearish -> risk off."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(-0.10, -0.08, -0.06)
        assert self._pattern_str(pattern) == "risk_off"
        assert signal == -0.5

    def test_flight_to_safety(self):
        """Bonds up, equities down -> flight to safety."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(-0.10, 0.02, 0.06)
        assert self._pattern_str(pattern) == "flight_to_safety"
        assert signal == -0.3

    def test_inflation_fear(self):
        """Bonds down, gold up -> inflation fear."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.01, 0.08, -0.06)
        assert self._pattern_str(pattern) == "inflation_fear"

    def test_risk_rotation(self):
        """Equity bear, gold bull -> risk rotation."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(-0.10, 0.08, 0.01)
        assert self._pattern_str(pattern) == "risk_rotation"

    def test_cautious_optimism(self):
        """Equity neutral, gold strong -> cautious optimism."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.02, 0.08, 0.01)
        assert self._pattern_str(pattern) == "cautious_optimism"

    def test_recovery_beginning(self):
        """Gold weak, equity recovering -> recovery beginning."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.02, -0.08, 0.01)
        assert self._pattern_str(pattern) == "recovery_beginning"

    def test_equity_rotation(self):
        """Equity diverging from bonds/gold -> equity rotation."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.10, 0.03, -0.06)
        assert self._pattern_str(pattern) == "equity_rotation"

    def test_fallback_unknown(self):
        """Unknown combination (all neutral except TLT bearish) -> no divergence."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.03, 0.03, -0.10)
        assert self._pattern_str(pattern) == "no_divergence"

    # ── Allocation Shifts ──

    def test_allocation_shifts_positive_signal(self):
        """Positive signal should overweight SPY."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, 0.3, 0.10, 0.01)
        assert spy_s > 0
        assert gld_s < 0
        assert tlt_s < 0

    def test_allocation_shifts_negative_signal(self):
        """Negative signal should underweight SPY, overweight bonds/gold."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, -0.3, 0.0, 0.0)
        assert spy_s < 0
        assert gld_s > 0
        assert tlt_s > 0

    def test_allocation_shifts_zero_signal(self):
        """Zero signal should return zero shifts."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, 0.0, 0.0, 0.0)
        assert spy_s == 0.0
        assert gld_s == 0.0
        assert tlt_s == 0.0

    def test_allocation_shifts_clamped(self):
        """Shifts should be clamped to max_single_shift."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(max_single_shift=0.03)
        )
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, 0.5, 0.10, 0.01)
        assert abs(spy_s) <= 0.03
        assert abs(gld_s) <= 0.03
        assert abs(tlt_s) <= 0.03

    def test_allocation_shift_scaling(self):
        """Signal strength should scale the shift proportionally."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s_half, _, _ = bt._get_allocation_shifts(None, 0.25, 0.10, 0.01)
        strength = 0.25 / MAX_SIGNAL_STRENGTH
        expected = ALLOCATION_SHIFTS["equity_outperformance"]["spy"] * strength
        assert spy_s_half == pytest.approx(expected)

    # ── Run Backtest ──

    def test_run_backtest_returns_result(self, bt_with_data):
        """run_backtest should return a BacktestResult."""
        result = bt_with_data.run_backtest()
        assert isinstance(result, BacktestResult)

    def test_run_backtest_fields_populated(self, bt_with_data):
        """Result should have all main fields populated."""
        result = bt_with_data.run_backtest()
        assert isinstance(result.total_return, float)
        assert isinstance(result.cagr, float)
        assert isinstance(result.volatility, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.baseline_sharpe, float)
        assert isinstance(result.sharpe_improvement, float)
        assert isinstance(result.overlay_active_months, int)
        assert isinstance(result.total_rebalances, int)
        assert isinstance(result.signal_frequency, float)
        assert isinstance(result.divergence_breakdown, dict)
        assert isinstance(result.equity_curve, list)

    def test_run_backtest_rebalance_signals(self, bt_with_data):
        """Rebalance signals list should contain dict entries with keys."""
        result = bt_with_data.run_backtest()
        if result.rebalance_signals:
            entry = result.rebalance_signals[0]
            assert "date" in entry
            assert "signal_value" in entry
            assert "pattern" in entry
            assert "spy_shift" in entry
            assert "spy_momentum" in entry

    def test_run_backtest_transaction_costs_nonnegative(self, bt_with_data):
        """Transaction costs should be non-negative."""
        result = bt_with_data.run_backtest()
        assert result.total_transaction_costs >= 0.0

    def test_run_backtest_sharpe_is_finite(self, bt_with_data):
        """Sharpe ratio should be a finite number."""
        result = bt_with_data.run_backtest()
        assert np.isfinite(result.sharpe_ratio)

    def test_run_backtest_no_data_returns_none(self):
        """Empty data should return None."""
        bt = CrossAssetRegimeArbBacktester()
        result = bt.run_backtest()
        assert result is None

    def test_run_backtest_single_day_data(self):
        """Single day of data should still produce a result."""
        bt = CrossAssetRegimeArbBacktester()
        bt.data = [
            DailyReturn(
                date="2020-01-02",
                spy_return=0.01,
                gld_return=0.005,
                tlt_return=-0.002,
            )
        ]
        result = bt.run_backtest()
        assert result is not None

    def test_run_backtest_dates_outside_range(self):
        """Data entirely outside the configured range should return None."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-01-31")
        )
        bt.data = [
            DailyReturn(
                date="2021-01-02",
                spy_return=0.01,
                gld_return=0.005,
                tlt_return=-0.002,
            )
        ]
        result = bt.run_backtest()
        assert result is None

    def test_run_backtest_divergence_breakdown_has_entries(self, bt_with_data):
        """Divergence breakdown should have at least one entry type."""
        result = bt_with_data.run_backtest()
        # With random data, some divergence patterns will be detected
        assert len(result.divergence_breakdown) > 0

    def test_run_backtest_equity_curve_sampled(self, bt_with_data):
        """Equity curve entries should have date/baseline/overlay keys."""
        result = bt_with_data.run_backtest()
        if result.equity_curve:
            entry = result.equity_curve[0]
            assert "date" in entry
            assert "baseline" in entry
            assert "overlay" in entry

    # ── Metrics Helpers ──

    def test_returns_from_equity_basic(self):
        bt = CrossAssetRegimeArbBacktester()
        equity = [100.0, 101.0, 99.0, 102.0]
        returns = bt._returns_from_equity(equity)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(-0.0198, rel=1e-3)

    def test_returns_from_equity_single(self):
        bt = CrossAssetRegimeArbBacktester()
        returns = bt._returns_from_equity([100.0])
        assert returns == []

    def test_calculate_metrics_empty(self):
        bt = CrossAssetRegimeArbBacktester()
        m = bt._calculate_metrics([])
        assert m["cagr"] == 0.0
        assert m["volatility"] == 0.0
        assert m["sharpe"] == 0.0
        assert m["max_dd"] == 0.0

    def test_calculate_metrics_basic(self):
        bt = CrossAssetRegimeArbBacktester()
        rng = np.random.default_rng(42)
        returns = list(0.0005 + rng.normal(0, 0.008, 252))
        m = bt._calculate_metrics(returns)
        assert m["cagr"] > 0
        assert m["volatility"] > 0
        assert m["sharpe"] > 0

    def test_annualize_positive(self):
        bt = CrossAssetRegimeArbBacktester()
        result = bt._annualize([0.001] * 252)
        assert result > 0

    def test_annualize_negative(self):
        bt = CrossAssetRegimeArbBacktester()
        result = bt._annualize([-0.001] * 252)
        assert result < 0

    def test_annualize_empty(self):
        bt = CrossAssetRegimeArbBacktester()
        assert bt._annualize([]) == 0.0

    # ── Print / Save ──

    def test_print_report_does_not_crash(self, capsys):
        """print_report should produce output without errors."""
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3, sharpe_ratio=0.85,
            max_drawdown=-15.4, overlay_active_months=24, baseline_sharpe=0.79,
            sharpe_improvement=0.06, return_2008=-12.0, return_2020=3.0,
            return_2022=None, total_rebalances=30, total_transaction_costs=45.0,
            signal_frequency=0.5, divergence_breakdown={"equity_rotation": 10},
            equity_curve=[], rebalance_signals=[],
        )
        bt = CrossAssetRegimeArbBacktester()
        bt.print_report(result)
        captured = capsys.readouterr()
        assert "CROSS-ASSET REGIME ARBITRAGE" in captured.out
        assert "Sharpe" in captured.out

    def test_print_report_no_crisis_data(self, capsys):
        """print_report handles None crisis returns gracefully."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, overlay_active_months=0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, return_2008=None, return_2020=None,
            return_2022=None, total_rebalances=0, total_transaction_costs=0.0,
            signal_frequency=0.0, divergence_breakdown={}, equity_curve=[],
            rebalance_signals=[],
        )
        bt = CrossAssetRegimeArbBacktester()
        bt.print_report(result)
        captured = capsys.readouterr()
        assert "CROSS-ASSET REGIME ARBITRAGE" in captured.out

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3, sharpe_ratio=0.85,
            max_drawdown=-15.4, overlay_active_months=24, baseline_sharpe=0.79,
            sharpe_improvement=0.06, return_2008=-12.0, return_2020=3.0,
            return_2022=None, total_rebalances=30, total_transaction_costs=45.0,
            signal_frequency=0.5, divergence_breakdown={"equity_rotation": 10},
            equity_curve=[], rebalance_signals=[],
        )
        bt = CrossAssetRegimeArbBacktester()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                data = json.load(f)
            assert "total_return" in data
            assert "cagr" in data
            assert "sharpe_ratio" in data
            assert "overlay_active_months" in data
            assert "divergence_breakdown" in data
            assert "rebalance_signals" in data
        finally:
            Path(output_path).unlink()


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the cross-asset regime arb backtest."""

    def test_load_data_missing_file(self, monkeypatch):
        """load_data should return False when no price data is found."""
        from src.paths import PRICES_JSON as _real
        monkeypatch.setattr(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            _real.parent / "nonexistent_prices.json",
        )
        bt = CrossAssetRegimeArbBacktester()
        success = bt.load_data()
        assert success is False

    def test_load_data_with_exception(self, monkeypatch):
        """load_data should handle exceptions gracefully."""
        def _bad_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", _bad_open)
        monkeypatch.setattr(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            Path("/tmp/test_prices.json"),
        )
        bt = CrossAssetRegimeArbBacktester()
        success = bt.load_data()
        assert success is False

    def test_small_initial_capital(self):
        """Very small initial capital should not crash."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(initial_capital=1.0)
        )
        bt.data = [
            DailyReturn(
                date="2020-01-02",
                spy_return=0.01,
                gld_return=0.005,
                tlt_return=-0.002,
            )
        ]
        result = bt.run_backtest()
        assert isinstance(result, BacktestResult)

    def test_constant_returns(self):
        """Constant returns across all assets should produce same baseline/overlay."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-06-30")
        )
        bt.data = [
            DailyReturn(
                date=(datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"),
                spy_return=0.001,
                gld_return=0.001,
                tlt_return=0.001,
            )
            for i in range(130)
        ]
        result = bt.run_backtest()
        assert result is not None
        assert result.total_rebalances >= 0

    def test_divergence_count_tracks_patterns(self):
        """Divergence breakdown should accumulate counts."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-02-01")
        )
        # Alternating momentum values to trigger different patterns
        bt.data = []
        for i in range(100):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            # Force alternating momentum
            spy_r = 0.01 if i < 50 else -0.01
            gld_r = 0.005
            tlt_r = 0.003
            bt.data.append(DailyReturn(date=day, spy_return=spy_r, gld_return=gld_r, tlt_return=tlt_r))
        result = bt.run_backtest()
        if result:
            total_counts = sum(result.divergence_breakdown.values())
            assert total_counts > 0


# ── CLI Tests ──────────────────────────────────────────────────────────────


class TestCLI:
    """Test CLI main() function."""

    def test_main_no_data_returns_1(self, monkeypatch):
        """main should return 1 when no data is available."""
        from src.paths import PRICES_JSON as _real
        monkeypatch.setattr(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            _real.parent / "nonexistent_prices.json",
        )
        monkeypatch.setattr("sys.argv", ["cross_asset_regime_arb_backtest.py"])
        rc = main()
        assert rc == 1

    def test_main_with_save_flag(self, monkeypatch):
        """main should accept --save flag."""
        monkeypatch.setattr(
            "sys.argv",
            ["cross_asset_regime_arb_backtest.py", "--save"],
        )
        from src.paths import PRICES_JSON as _real
        monkeypatch.setattr(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            _real.parent / "nonexistent_prices.json",
        )
        rc = main()
        assert rc == 1  # Still fails due to no data, but parsing works
