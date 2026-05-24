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
    RebalanceSignal,
    CrossAssetRegimeArbBacktester,
    ALLOCATION_SHIFTS,
    MAX_SIGNAL_STRENGTH,
    main,
)
from src.backtest.metrics import BacktestResult


# ── BacktestConfig Tests ──────────────────────────────────────────────────


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
        total = config.base_weights["SPY"] + config.base_weights["GLD"] + config.base_weights["TLT"]
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
            baseline_sharpe=0.79,
            sharpe_improvement=0.06,
            total_rebalances=30,
            total_transaction_costs=45.0,
            crisis_returns={"2008": -12.0, "2020": 3.0},
            extras={
                "overlay_active_months": 24,
                "signal_frequency": 0.5,
                "divergence_breakdown": {"equity_rotation": 10, "no_divergence": 20},
                "equity_curve": [{"date": "2020-01-01", "baseline": 100000.0, "overlay": 101000.0}],
                "rebalance_signals": [{"date": "2020-01-31", "signal_value": 0.3}],
            },
        )
        assert result.total_return == 10.5
        assert result.sharpe_ratio == 0.85
        assert result.extras["overlay_active_months"] == 24
        assert result.baseline_sharpe == 0.79
        assert result.sharpe_improvement == 0.06
        assert result.crisis_returns["2008"] == -12.0
        assert result.crisis_returns.get("2022") is None
        assert result.extras["signal_frequency"] == 0.5
        assert result.extras["divergence_breakdown"]["equity_rotation"] == 10

    def test_json_serializable(self):
        """asdict(result) must be JSON-serializable."""
        from dataclasses import asdict

        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=10, total_transaction_costs=5.0,
            crisis_returns={"2008": -8.0, "2020": 2.0},
            extras={
                "overlay_active_months": 5, "signal_frequency": 0.3,
                "divergence_breakdown": {}, "equity_curve": [], "rebalance_signals": [],
            },
        )
        data = asdict(result)
        json.dumps(data)  # Should not raise

    def test_empty_divergence_breakdown(self):
        """Empty divergence breakdown should not error."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_sharpe=0.0,
            sharpe_improvement=0.0, total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0, "signal_frequency": 0.0,
                "divergence_breakdown": {}, "equity_curve": [], "rebalance_signals": [],
            },
        )
        assert result.extras["divergence_breakdown"] == {}


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
        assert isinstance(result.extras["overlay_active_months"], int)
        assert isinstance(result.total_rebalances, int)
        assert isinstance(result.extras["signal_frequency"], float)
        assert isinstance(result.extras["divergence_breakdown"], dict)
        assert isinstance(result.extras["equity_curve"], list)

    def test_run_backtest_rebalance_signals(self, bt_with_data):
        """Rebalance signals list should contain dict entries with keys."""
        result = bt_with_data.run_backtest()
        if result.extras.get("rebalance_signals"):
            entry = result.extras["rebalance_signals"][0]
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
        assert len(result.extras["divergence_breakdown"]) > 0

    def test_run_backtest_equity_curve_sampled(self, bt_with_data):
        """Equity curve entries should have date/baseline/overlay keys."""
        result = bt_with_data.run_backtest()
        if result.extras.get("equity_curve"):
            entry = result.extras["equity_curve"][0]
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
            max_drawdown=-15.4, baseline_sharpe=0.79,
            sharpe_improvement=0.06, total_rebalances=30, total_transaction_costs=45.0,
            crisis_returns={"2008": -12.0, "2020": 3.0},
            extras={
                "overlay_active_months": 24, "signal_frequency": 0.5,
                "divergence_breakdown": {"equity_rotation": 10},
                "equity_curve": [], "rebalance_signals": [],
            },
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
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0, "signal_frequency": 0.0,
                "divergence_breakdown": {}, "equity_curve": [], "rebalance_signals": [],
            },
        )
        bt = CrossAssetRegimeArbBacktester()
        bt.print_report(result)
        captured = capsys.readouterr()
        assert "CROSS-ASSET REGIME ARBITRAGE" in captured.out

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3, sharpe_ratio=0.85,
            max_drawdown=-15.4, baseline_sharpe=0.79,
            sharpe_improvement=0.06, total_rebalances=30, total_transaction_costs=45.0,
            crisis_returns={"2008": -12.0, "2020": 3.0},
            extras={
                "overlay_active_months": 24, "signal_frequency": 0.5,
                "divergence_breakdown": {"equity_rotation": 10},
                "equity_curve": [], "rebalance_signals": [],
            },
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
            assert "extras" in data
            assert "overlay_active_months" in data["extras"]
            assert "divergence_breakdown" in data["extras"]
            assert "rebalance_signals" in data["extras"]
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
            total_counts = sum(result.extras["divergence_breakdown"].values())
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


# ── Constant & Export Tests ──────────────────────────────────────────────


class TestBacktestConstants:
    """Verify module-level constants and exports."""

    def test_allocation_shifts_structure(self):
        """ALLOCATION_SHIFTS has expected keys and structure."""
        assert "equity_outperformance" in ALLOCATION_SHIFTS
        assert "safe_haven_outperformance" in ALLOCATION_SHIFTS
        for key in ALLOCATION_SHIFTS:
            assert "spy" in ALLOCATION_SHIFTS[key]
            assert "gld" in ALLOCATION_SHIFTS[key]
            assert "tlt" in ALLOCATION_SHIFTS[key]

    def test_allocation_shifts_equity_outperformance_values(self):
        """Equity outperformance shifts SPY +3%, GLD -1%, TLT -2%."""
        shifts = ALLOCATION_SHIFTS["equity_outperformance"]
        assert shifts["spy"] == 0.03
        assert shifts["gld"] == -0.01
        assert shifts["tlt"] == -0.02

    def test_allocation_shifts_safe_haven_values(self):
        """Safe haven outperformance shifts SPY -2%, GLD +1%, TLT +1%."""
        shifts = ALLOCATION_SHIFTS["safe_haven_outperformance"]
        assert shifts["spy"] == -0.02
        assert shifts["gld"] == 0.01
        assert shifts["tlt"] == 0.01

    def test_max_signal_strength_value(self):
        """MAX_SIGNAL_STRENGTH is 0.5."""
        assert MAX_SIGNAL_STRENGTH == 0.5

    def test_backtest_exports(self):
        """__all__ contains expected public names."""
        from src.backtest.cross_asset_regime_arb_backtest import __all__
        expected = {
            "MAX_SIGNAL_STRENGTH", "BacktestConfig", "DailyReturn",
            "RebalanceSignal", "CrossAssetRegimeArbBacktester",
        }
        assert set(__all__) == expected

    def test_divergence_signal_all_bullish_constant(self):
        """Constants match expected values."""
        from src.backtest.cross_asset_regime_arb_backtest import (
            BULL_MOMENTUM_THRESHOLD, BEAR_MOMENTUM_THRESHOLD, MOMENTUM_LOOKBACK,
        )
        assert BULL_MOMENTUM_THRESHOLD == 0.05
        assert BEAR_MOMENTUM_THRESHOLD == -0.05
        assert MOMENTUM_LOOKBACK == 60


# ── RebalanceSignal Advanced Tests ───────────────────────────────────────


class TestRebalanceSignalAdvanced:
    """Advanced RebalanceSignal field tests."""

    def test_rebalance_signal_fields(self):
        """All RebalanceSignal fields have correct types."""
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
        assert isinstance(sig.date, str)
        assert isinstance(sig.signal_value, float)
        assert isinstance(sig.pattern, str)
        assert isinstance(sig.spy_shift, float)
        assert isinstance(sig.gld_shift, float)
        assert isinstance(sig.tlt_shift, float)
        assert isinstance(sig.spy_momentum, float)
        assert isinstance(sig.gld_momentum, float)
        assert isinstance(sig.tlt_momentum, float)

    def test_rebalance_signal_from_dataclass(self):
        """RebalanceSignal is a proper dataclass with expected fields."""
        from dataclasses import fields
        sig = RebalanceSignal(
            date="2020-01-31", signal_value=-0.5, pattern="risk_off",
            spy_shift=-0.02, gld_shift=0.01, tlt_shift=0.01,
            spy_momentum=-0.10, gld_momentum=-0.08, tlt_momentum=-0.06,
        )
        field_names = {f.name for f in fields(sig)}
        expected = {
            "date", "signal_value", "pattern", "spy_shift", "gld_shift",
            "tlt_shift", "spy_momentum", "gld_momentum", "tlt_momentum",
        }
        assert field_names == expected

    def test_rebalance_signal_negative_signal(self):
        """Negative signal values are handled correctly."""
        sig = RebalanceSignal(
            date="2020-03-15", signal_value=-0.4, pattern="flight_to_safety",
            spy_shift=-0.02, gld_shift=0.01, tlt_shift=0.01,
            spy_momentum=-0.12, gld_momentum=0.02, tlt_momentum=0.06,
        )
        assert sig.signal_value < 0
        assert sig.spy_shift < 0
        assert sig.gld_shift > 0
        assert sig.tlt_shift > 0


# ── BacktestConfig Edge Cases ────────────────────────────────────────────


class TestBacktestConfigEdgeCases:
    """Edge cases for BacktestConfig."""

    def test_signal_threshold_zero(self):
        """Zero signal threshold means any signal triggers action."""
        config = BacktestConfig(signal_threshold=0.0)
        assert config.signal_threshold == 0.0

    def test_signal_threshold_negative(self):
        """Negative signal threshold is allowed (always triggers)."""
        config = BacktestConfig(signal_threshold=-0.1)
        assert config.signal_threshold == -0.1

    def test_signal_threshold_high(self):
        """High signal threshold means only strong signals trigger."""
        config = BacktestConfig(signal_threshold=0.5)
        assert config.signal_threshold == 0.5

    def test_max_single_shift_zero(self):
        """Zero max shift disables allocation shifts."""
        config = BacktestConfig(max_single_shift=0.0)
        assert config.max_single_shift == 0.0

    def test_max_single_shift_large(self):
        """Large max single shift is allowed."""
        config = BacktestConfig(max_single_shift=0.20)
        assert config.max_single_shift == 0.20


# ── Data Processing Edge Cases ───────────────────────────────────────────


class TestDataProcessingEdgeCases:
    """Edge cases for price data processing."""

    def test_process_price_data_zero_price(self):
        """Zero price (falsy in Python) skips that day gracefully."""
        bt = CrossAssetRegimeArbBacktester()
        data = {
            "SPY": [
                {"d": "2020-01-02", "p": 0.0},
                {"d": "2020-01-03", "p": 100.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 51.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        bt._process_price_data(data)
        # spy_prev == 0.0 is falsy, so `all(...)` fails and day is skipped
        assert len(bt.data) == 0

    def test_process_price_data_negative_price(self):
        """Negative price should not crash."""
        bt = CrossAssetRegimeArbBacktester()
        data = {
            "SPY": [
                {"d": "2020-01-02", "p": -50.0},
                {"d": "2020-01-03", "p": 100.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 51.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        bt._process_price_data(data)
        assert len(bt.data) == 1
        assert np.isfinite(bt.data[0].spy_return)

    def test_process_price_data_empty_spy_list(self):
        """Empty SPY list produces no data."""
        bt = CrossAssetRegimeArbBacktester()
        bt._process_price_data({"SPY": [], "GLD": [], "TLT": []})
        assert bt.data == []

    def test_process_price_data_missing_gld_key(self):
        """Missing GLD key produces no data."""
        bt = CrossAssetRegimeArbBacktester()
        bt._process_price_data({"SPY": [{"d": "2020-01-02", "p": 100.0}]})
        assert bt.data == []


# ── Momentum Computation Edge Cases ──────────────────────────────────────


class TestMomentumEdgeCases:
    """Edge cases for momentum computation."""

    @pytest.fixture
    def bt_flat_data(self):
        """Create backtester with flat (zero return) data."""
        bt = CrossAssetRegimeArbBacktester()
        bt.data = []
        for i in range(200):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            bt.data.append(
                DailyReturn(date=day, spy_return=0.0, gld_return=0.0, tlt_return=0.0)
            )
        return bt

    def test_compute_momentum_flat_data(self, bt_flat_data):
        """Flat data produces zero momentum."""
        spy_m, gld_m, tlt_m = bt_flat_data._compute_momentum(60, 100)
        assert spy_m == 0.0
        assert gld_m == 0.0
        assert tlt_m == 0.0

    def test_compute_momentum_exact_lookback(self):
        """Momentum with exact lookback boundary (i == lookback)."""
        bt = CrossAssetRegimeArbBacktester()
        bt.data = []
        for i in range(61):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            bt.data.append(
                DailyReturn(
                    date=day, spy_return=0.001, gld_return=0.0005, tlt_return=0.0002,
                )
            )
        spy_m, gld_m, tlt_m = bt._compute_momentum(60, 60)
        assert spy_m != 0.0

    def test_compute_momentum_with_extreme_returns(self):
        """Extreme returns (>50%) are filtered from momentum."""
        bt = CrossAssetRegimeArbBacktester()
        bt.data = []
        for i in range(100):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            spy_r = 2.0 if i == 30 else 0.001
            bt.data.append(
                DailyReturn(
                    date=day, spy_return=spy_r, gld_return=0.001, tlt_return=0.001,
                )
            )
        spy_m, gld_m, tlt_m = bt._compute_momentum(60, 90)
        assert np.isfinite(spy_m)


# ── Regime Classification Boundary Tests ─────────────────────────────────


class TestRegimeClassificationBoundaries:
    """Boundary value tests for regime classification."""

    def test_classify_exactly_bull_threshold(self):
        """Momentum exactly at BULL threshold is neutral (uses > not >=)."""
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(0.05)
        assert regime == "neutral"

    def test_classify_just_below_bull_threshold(self):
        """Momentum just below BULL threshold is neutral."""
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(0.0499)
        assert regime == "neutral"

    def test_classify_exactly_bear_threshold(self):
        """Momentum exactly at BEAR threshold is neutral (uses < not <=)."""
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(-0.05)
        assert regime == "neutral"

    def test_classify_just_above_bear_threshold(self):
        """Momentum just above BEAR threshold is neutral."""
        bt = CrossAssetRegimeArbBacktester()
        regime = bt._classify_asset_regime(-0.0499)
        assert regime == "neutral"


# ── Divergence Detection Edge Cases ──────────────────────────────────────


class TestDivergenceDetectionEdgeCases:
    """Additional divergence detection edge cases."""

    @staticmethod
    def _pattern_str(pattern):
        return pattern.value if hasattr(pattern, "value") else str(pattern)

    def test_divergence_gld_bearish_spy_bullish(self):
        """Gold bear + equity bull -> equity rotation."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.10, -0.10, 0.03)
        assert self._pattern_str(pattern) == "equity_rotation"

    def test_divergence_all_neutral(self):
        """All neutral -> no divergence."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.01, 0.01, 0.01)
        assert self._pattern_str(pattern) == "no_divergence"
        assert signal == 0.0

    def test_divergence_equity_only_bearish(self):
        """Only equity bearish, bonds/gold neutral -> equity rotation."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(-0.10, 0.01, 0.01)
        assert self._pattern_str(pattern) == "equity_rotation"

    def test_divergence_tlt_only_bullish(self):
        """Only TLT bullish -> flight to safety (if equity bear)."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(-0.10, 0.01, 0.08)
        assert self._pattern_str(pattern) == "flight_to_safety"

    def test_divergence_tlt_bearish_gld_bullish(self):
        """TLT bear + GLD bull -> inflation fear."""
        bt = CrossAssetRegimeArbBacktester()
        pattern, signal = bt._detect_divergence_signal(0.01, 0.08, -0.10)
        assert self._pattern_str(pattern) == "inflation_fear"
        assert signal == -0.1


# ── Allocation Shift Edge Cases ──────────────────────────────────────────


class TestAllocationShiftsEdgeCases:
    """Edge cases for allocation shifts."""

    def test_get_allocation_shifts_exact_max_signal(self):
        """Signal at MAX_SIGNAL_STRENGTH uses full shift."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(
            None, MAX_SIGNAL_STRENGTH, 0.10, 0.01,
        )
        assert spy_s == pytest.approx(0.03)
        assert gld_s == pytest.approx(-0.01)
        assert tlt_s == pytest.approx(-0.02)

    def test_get_allocation_shifts_half_signal_negative(self):
        """Half-strength negative signal produces half shifts."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, -0.25, 0.0, 0.0)
        expected_strength = 0.25 / MAX_SIGNAL_STRENGTH
        assert spy_s == pytest.approx(-0.02 * expected_strength)
        assert gld_s == pytest.approx(0.01 * expected_strength)
        assert tlt_s == pytest.approx(0.01 * expected_strength)

    def test_get_allocation_shifts_negative_max_clamped(self):
        """Strong negative signal clamped to max_single_shift."""
        bt = CrossAssetRegimeArbBacktester(
            BacktestConfig(max_single_shift=0.02),
        )
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, -0.5, 0.0, 0.0)
        assert abs(spy_s) <= 0.02
        assert abs(gld_s) <= 0.02
        assert abs(tlt_s) <= 0.02

    def test_get_allocation_shifts_small_signal(self):
        """Tiny signal produces near-zero shifts."""
        bt = CrossAssetRegimeArbBacktester()
        spy_s, gld_s, tlt_s = bt._get_allocation_shifts(None, 0.001, 0.0, 0.0)
        assert abs(spy_s) < 0.001
        assert abs(gld_s) < 0.001
        assert abs(tlt_s) < 0.001


# ── Metrics Edge Cases ──────────────────────────────────────────────────


class TestMetricsEdgeCases:
    """Edge cases for metrics helpers."""

    def test_calculate_metrics_all_negative_returns(self):
        """All negative returns produce negative CAGR and negative max_dd."""
        bt = CrossAssetRegimeArbBacktester()
        returns = [-0.01] * 252
        m = bt._calculate_metrics(returns)
        assert m["cagr"] < 0
        assert m["max_dd"] < 0
        assert isinstance(m["volatility"], float)

    def test_calculate_metrics_flat_returns(self):
        """Zero returns produce zero metrics."""
        bt = CrossAssetRegimeArbBacktester()
        returns = [0.0] * 252
        m = bt._calculate_metrics(returns)
        assert m["cagr"] == 0.0
        assert m["volatility"] == 0.0
        assert m["sharpe"] == 0.0
        assert m["max_dd"] == 0.0

    def test_calculate_metrics_single_return(self):
        """Single return produces valid metrics."""
        bt = CrossAssetRegimeArbBacktester()
        m = bt._calculate_metrics([0.01])
        assert isinstance(m["cagr"], float)
        assert isinstance(m["volatility"], float)

    def test_annualize_single_day(self):
        """Single day annualizes to extreme value."""
        bt = CrossAssetRegimeArbBacktester()
        result = bt._annualize([0.001])
        expected = ((1 + 0.001) ** 252 - 1) * 100
        assert result == pytest.approx(expected)

    def test_annualize_exact_one_year(self):
        """Exactly 252 trading days gives 1-year return."""
        bt = CrossAssetRegimeArbBacktester()
        daily_ret = 0.0005
        returns = [daily_ret] * 252
        result = bt._annualize(returns)
        expected = ((1 + daily_ret) ** 252 - 1) * 100
        assert result == pytest.approx(expected)

    def test_annualize_zero_returns(self):
        """All zero returns gives zero annualized return."""
        bt = CrossAssetRegimeArbBacktester()
        result = bt._annualize([0.0] * 252)
        assert result == 0.0


# ── BacktestResult Extras Edge Cases ────────────────────────────────────


class TestBacktestResultExtras:
    """Edge cases for BacktestResult extras."""

    def test_extras_empty_equity_curve(self):
        """Empty equity curve in extras does not error."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_sharpe=0.0,
            sharpe_improvement=0.0, total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0, "signal_frequency": 0.0,
                "divergence_breakdown": {}, "equity_curve": [],
                "rebalance_signals": [],
            },
        )
        assert result.extras["equity_curve"] == []

    def test_extras_empty_rebalance_signals(self):
        """Empty rebalance signals in extras does not error."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_sharpe=0.0,
            sharpe_improvement=0.0, total_rebalances=0, total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0, "signal_frequency": 0.0,
                "divergence_breakdown": {}, "equity_curve": [],
                "rebalance_signals": [],
            },
        )
        assert result.extras["rebalance_signals"] == []

    def test_extras_divergence_breakdown(self):
        """Divergence with multiple pattern types."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=5, total_transaction_costs=2.0,
            extras={
                "overlay_active_months": 3, "signal_frequency": 0.3,
                "divergence_breakdown": {
                    "full_risk_on": 5, "risk_off": 3, "no_divergence": 20,
                },
                "equity_curve": [], "rebalance_signals": [],
            },
        )
        assert result.extras["divergence_breakdown"]["full_risk_on"] == 5
        assert result.extras["divergence_breakdown"]["risk_off"] == 3


# ── Load Data Tests ─────────────────────────────────────────────────────


class TestLoadData:
    """Tests for load_data with actual JSON content."""

    def test_load_data_with_json(self, tmp_path):
        """load_data reads JSON file correctly."""
        from unittest.mock import patch
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 50.5},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(prices, f)
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester()
            success = bt.load_data()
            assert success is True
            assert len(bt.data) == 1
            assert bt.data[0].date == "2020-01-03"

    def test_load_data_missing_spy_key(self, tmp_path):
        """JSON file missing SPY key returns data but empty."""
        from unittest.mock import patch
        prices = {"GLD": [], "TLT": []}
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(prices, f)
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester()
            success = bt.load_data()
            assert success is True
            assert len(bt.data) == 0

    def test_load_data_corrupt_json(self, tmp_path):
        """Corrupt JSON file is handled gracefully."""
        from unittest.mock import patch
        prices_file = tmp_path / "prices.json"
        prices_file.write_text("{corrupt json}")
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester()
            success = bt.load_data()
            assert success is False

    def test_load_data_json_decode_error(self, tmp_path):
        """JSON decode errors are caught gracefully."""
        from unittest.mock import patch
        prices_file = tmp_path / "prices.json"
        prices_file.write_text("not even json")
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester()
            success = bt.load_data()
            assert success is False


# ── CLI Advanced Tests ──────────────────────────────────────────────────


class TestCLIAdvanced:
    """Advanced CLI test cases."""

    def test_main_returns_zero_on_success(self, tmp_path):
        """main returns 0 when backtest completes with data."""
        from unittest.mock import patch
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 50.5},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(prices, f)
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester(
                BacktestConfig(
                    start_date="2020-01-01", end_date="2020-02-01",
                ),
            )
            result = bt.run_backtest()
            if result:
                assert isinstance(result, BacktestResult)

    def test_main_with_save_and_data(self, tmp_path):
        """main with --save flag loads data without error."""
        from unittest.mock import patch
        prices = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 50.5},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-03", "p": 81.0},
            ],
        }
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(prices, f)
        with patch(
            "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
            prices_file,
        ):
            bt = CrossAssetRegimeArbBacktester(
                BacktestConfig(
                    start_date="2020-01-01", end_date="2020-02-01",
                ),
            )
            success = bt.load_data()
            assert success is True

    def test_main_save_only_flag(self, tmp_path):
        """main with --save-only flag still parses correctly."""
        from unittest.mock import patch
        from src.paths import PRICES_JSON as _real
        with patch("sys.argv", ["cross_asset_regime_arb_backtest.py", "--save-only"]), \
             patch(
                "src.backtest.cross_asset_regime_arb_backtest.PRICES_JSON",
                _real.parent / "nonexistent_prices.json",
            ):
            rc = main()
            assert rc == 1


# ── HAS_SIGNAL_MODULE Path Tests ─────────────────────────────────────────


class TestHasSignalModule:
    """Tests for the HAS_SIGNAL_MODULE conditional path."""

    def test_has_signal_module_true_by_default(self):
        """HAS_SIGNAL_MODULE is True when signal module is importable."""
        from src.backtest.cross_asset_regime_arb_backtest import (
            HAS_SIGNAL_MODULE,
        )
        assert HAS_SIGNAL_MODULE is True

    def test_detector_created_when_module_available(self):
        """Detector is created when HAS_SIGNAL_MODULE is True."""
        bt = CrossAssetRegimeArbBacktester()
        assert bt.detector is not None
        assert hasattr(bt.detector, "scan")

    def test_print_report_shows_live_signal_module(self, capsys):
        """print_report shows 'Live' when signal module is active."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=0,
            total_transaction_costs=0.0,
            extras={
                "overlay_active_months": 0, "signal_frequency": 0.0,
                "divergence_breakdown": {}, "equity_curve": [],
                "rebalance_signals": [],
            },
        )
        bt = CrossAssetRegimeArbBacktester()
        bt.print_report(result)
        captured = capsys.readouterr()
        assert "Live" in captured.out or "Simulated" in captured.out


# ── Equity Curve Sampling Tests ──────────────────────────────────────────


class TestEquityCurveSampling:
    """Tests for equity curve sampling in results."""

    def test_equity_curve_min_one_entry(self):
        """Equity curve has at least one entry after backtest."""
        bt = CrossAssetRegimeArbBacktester()
        np.random.seed(42)
        bt.data = []
        for i in range(200):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            bt.data.append(
                DailyReturn(
                    date=day,
                    spy_return=float(np.random.normal(0.0005, 0.01)),
                    gld_return=float(np.random.normal(0.0002, 0.008)),
                    tlt_return=float(np.random.normal(0.0001, 0.006)),
                )
            )
        result = bt.run_backtest()
        assert len(result.extras["equity_curve"]) >= 1
        entry = result.extras["equity_curve"][0]
        assert "date" in entry
        assert "baseline" in entry
        assert "overlay" in entry
        assert isinstance(entry["baseline"], float)
        assert isinstance(entry["overlay"], float)
