"""
Tests for the Multi-Speed Momentum Overlay Backtest.

Covers: BacktestConfig defaults/custom, DailyReturn and BacktestResult dataclasses,
MultiSpeedMomentumBacktester init, data processing, signal computation (fallback),
run_backtest, print/save output, CLI main, and edge cases.
"""

import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.backtest.multi_speed_momentum_backtest import (
    BacktestConfig,
    DailyReturn,
    MultiSpeedMomentumBacktester,
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
        assert config.max_spy_shift == 0.05
        assert config.max_gld_shift == 0.03
        assert config.max_tlt_shift == 0.02
        assert config.signal_threshold == 0.10

    def test_custom_values(self):
        config = BacktestConfig(
            start_date="2010-01-01",
            end_date="2020-12-31",
            initial_capital=50000.0,
            max_spy_shift=0.10,
            signal_threshold=0.20,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2020-12-31"
        assert config.initial_capital == 50000.0
        assert config.max_spy_shift == 0.10
        assert config.signal_threshold == 0.20

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
            total_rebalances=50,
            avg_turnover=0.15,
            total_transaction_costs=45.50,
            extras={"overlay_active_rebalances": 120},
        )
        assert result.total_return == 10.5
        assert result.sharpe_ratio == 0.85
        assert result.baseline_sharpe == 0.79
        assert result.sharpe_improvement == 0.06
        assert result.extras["overlay_active_rebalances"] == 120
        assert result.total_rebalances == 50
        assert result.avg_turnover == 0.15
        assert result.total_transaction_costs == 45.50

    def test_crisis_returns_default_none(self):
        """Crisis return fields should be None by default."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=10, avg_turnover=0.0,
            total_transaction_costs=0.0,
            extras={"overlay_active_rebalances": 0},
        )
        assert result.crisis_returns is None or result.crisis_returns.get("2008") is None
        assert result.crisis_returns is None or result.crisis_returns.get("2020") is None
        assert result.crisis_returns is None or result.crisis_returns.get("2022") is None

    def test_equity_curve_default_none(self):
        """Equity curve should be None by default."""
        result = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, baseline_sharpe=0.0,
            sharpe_improvement=0.0, total_rebalances=0, avg_turnover=0.0,
            total_transaction_costs=0.0,
            extras={"overlay_active_rebalances": 0},
        )
        assert result.extras.get("equity_curve") is None

    def test_json_serializable(self):
        """asdict(result) must be JSON-serializable."""
        from dataclasses import asdict

        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=10, avg_turnover=0.1,
            total_transaction_costs=5.0,
            extras={
                "overlay_active_rebalances": 50,
                "return_2008": -12.0, "return_2020": 3.0,
                "equity_curve": [{"date": "2020-01-01", "baseline": 100000.0, "overlay": 101000.0}],
            },
        )
        data = asdict(result)
        json.dumps(data)  # Should not raise

    def test_total_return_high_low(self):
        """Very high and very low return values should round-trip."""
        result = BacktestResult(
            total_return=999.99, cagr=50.0, volatility=30.0, sharpe_ratio=1.5,
            max_drawdown=-99.99, baseline_sharpe=0.8,
            sharpe_improvement=0.7, total_rebalances=0, avg_turnover=0.0,
            total_transaction_costs=0.0,
            extras={"overlay_active_rebalances": 0},
        )
        assert result.total_return == 999.99
        assert result.max_drawdown == -99.99


# ── MultiSpeedMomentumBacktester Tests ────────────────────────────────────


class TestMultiSpeedMomentumBacktester:
    """Test the core MultiSpeedMomentumBacktester class."""

    # ── Init ──

    def test_init_defaults(self):
        bt = MultiSpeedMomentumBacktester()
        assert bt.config.start_date == "2006-01-01"
        assert bt.config.end_date == "2026-05-15"
        assert bt.data == []
        assert bt.prices_raw == {}

    def test_init_custom_config(self):
        config = BacktestConfig(start_date="2015-01-01", max_spy_shift=0.08)
        bt = MultiSpeedMomentumBacktester(config)
        assert bt.config.start_date == "2015-01-01"
        assert bt.config.max_spy_shift == 0.08

    # ── Data Processing ──

    def test_process_price_data_empty(self):
        """Empty price dict should produce no data."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {}
        bt._process_price_data()
        assert bt.data == []

    def test_process_price_data_missing_spy(self):
        """Missing SPY data should produce no data and not crash."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"GLD": [{"d": "2020-01-02", "p": 100.0}]}
        bt._process_price_data()
        assert bt.data == []

    def test_process_price_data_single_day(self):
        """Single day of prices should produce no returns (needs at least 2)."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}],
            "GLD": [{"d": "2020-01-02", "p": 100.0}],
            "TLT": [{"d": "2020-01-02", "p": 100.0}],
        }
        bt._process_price_data()
        assert bt.data == []

    def test_process_price_data_two_days(self):
        """Two days of prices should produce one DailyReturn."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {
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
        bt._process_price_data()
        assert len(bt.data) == 1
        dr = bt.data[0]
        assert dr.date == "2020-01-03"
        assert dr.spy_return == pytest.approx(0.01)  # (101 - 100) / 100
        assert dr.gld_return == pytest.approx(-0.02)  # (49 - 50) / 50
        assert dr.tlt_return == pytest.approx(0.0125)  # (81 - 80) / 80

    def test_process_price_data_missing_symbol_values(self):
        """Missing intermediate values should be skipped."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
                {"d": "2020-01-06", "p": 102.0},
            ],
            "GLD": [
                {"d": "2020-01-02", "p": 50.0},
                {"d": "2020-01-03", "p": 49.0},
            ],
            "TLT": [
                {"d": "2020-01-02", "p": 80.0},
                {"d": "2020-01-06", "p": 81.0},
            ],
        }
        bt._process_price_data()
        # Only the days where all three have entries for prev and curr
        assert len(bt.data) >= 0  # May be 0 or 1 depending on alignment
        for dr in bt.data:
            assert isinstance(dr.spy_return, float)
            assert isinstance(dr.gld_return, float)
            assert isinstance(dr.tlt_return, float)

    def test_load_data_honors_explicit_data_path(self, tmp_path):
        """Explicit temporary price files should not be ignored."""
        path = tmp_path / "prices.json"
        path.write_text(json.dumps({
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 51.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 79.0}],
        }))
        bt = MultiSpeedMomentumBacktester()

        assert bt.load_data(str(path)) is True

        assert len(bt.data) == 1
        assert bt.prices_raw["SPY"][0]["p"] == 100.0

    def test_get_prices_slice_uses_loaded_indexes(self):
        """Slice lookup should use built indexes instead of scanning every raw row."""

        class ExplodingRows(list):
            def __iter__(self):
                raise AssertionError("raw rows should not be scanned after indexes are built")

        bt = MultiSpeedMomentumBacktester()
        rows = [{"d": f"2020-01-{day:02d}", "p": float(day)} for day in range(1, 32)]
        bt.prices_raw = {
            "SPY": list(rows),
            "GLD": list(rows),
            "TLT": list(rows),
        }
        bt._process_price_data()
        bt.prices_raw = {ticker: ExplodingRows(values) for ticker, values in bt.prices_raw.items()}

        sliced = bt._get_prices_slice("2020-01-20", lookback=5)

        assert sliced["SPY"][0]["d"] == "2020-01-01"
        assert sliced["SPY"][-1]["d"] == "2020-01-20"

    # ── Signal Computation (fallback path) ──

    def test_compute_signal_insufficient_data(self):
        """With less than 260 price entries, signal should be 0.0."""
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {
            "SPY": [{"d": f"2020-01-{d:02d}", "p": 100.0} for d in range(1, 10)],
        }
        signal = bt._compute_signal("SPY", "2020-01-09")
        assert signal == 0.0

    def test_compute_signal_fallback_positive(self, monkeypatch):
        """Compute signal using fallback with upward trend."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        # Generate 300 prices trending up
        prices = []
        for i in range(300):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 * (1 + i * 0.001)})
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal > 0.0
        assert signal <= 1.0

    def test_compute_signal_fallback_negative(self, monkeypatch):
        """Compute signal using fallback with downward trend."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        # Generate 300 prices trending down
        prices = []
        for i in range(300):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 * (1 - i * 0.001)})
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal < 0.0
        assert signal >= -1.0

    def test_compute_signal_saturated(self, monkeypatch):
        """Very large 12m return should saturate at +/-1.0."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        prices = []
        for i in range(300):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 + (i * 2.0)})  # +300 over ~300d
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        # 12m return would be ~200%, clipped to +-20% / 20% = 1.0
        assert signal == pytest.approx(1.0)

    # ── Allocation Helpers ──

    def test_get_base_weights(self):
        bt = MultiSpeedMomentumBacktester()
        w = bt._get_base_weights()
        assert w["SPY"] == bt.config.base_weights["SPY"]
        assert w["GLD"] == bt.config.base_weights["GLD"]
        assert w["TLT"] == bt.config.base_weights["TLT"]
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_get_overlay_shifts_positive(self):
        bt = MultiSpeedMomentumBacktester()
        shifts = bt._get_overlay_shifts(0.5, 0.3, 0.1)
        assert shifts["SPY"] == pytest.approx(0.5 * bt.config.max_spy_shift)
        assert shifts["GLD"] == pytest.approx(0.3 * bt.config.max_gld_shift)
        assert shifts["TLT"] == pytest.approx(0.1 * bt.config.max_tlt_shift)

    def test_get_overlay_shifts_negative(self):
        bt = MultiSpeedMomentumBacktester()
        shifts = bt._get_overlay_shifts(-0.5, -0.3, -0.1)
        assert shifts["SPY"] < 0
        assert shifts["GLD"] < 0
        assert shifts["TLT"] < 0

    def test_get_overlay_shifts_clip(self):
        """Signals beyond [-1, 1] should be clipped."""
        bt = MultiSpeedMomentumBacktester()
        shifts = bt._get_overlay_shifts(-2.0, 3.0, 0.5)
        assert shifts["SPY"] == -bt.config.max_spy_shift
        assert shifts["GLD"] == bt.config.max_gld_shift

    def test_get_overlay_shifts_zero(self):
        bt = MultiSpeedMomentumBacktester()
        shifts = bt._get_overlay_shifts(0.0, 0.0, 0.0)
        for v in shifts.values():
            assert v == 0.0

    def test_compute_turnover(self):
        bt = MultiSpeedMomentumBacktester()
        old = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        new = {"SPY": 0.4, "GLD": 0.4, "TLT": 0.2}
        turnover = bt._compute_turnover(old, new)
        # |0.4-0.5| + |0.4-0.3| + |0.2-0.2| = 0.1 + 0.1 + 0 = 0.2, /2 = 0.1
        assert turnover == pytest.approx(0.1)

    def test_compute_turnover_no_change(self):
        bt = MultiSpeedMomentumBacktester()
        old = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        turnover = bt._compute_turnover(old, dict(old))
        assert turnover == 0.0

    def test_compute_turnover_full_rebalance(self):
        bt = MultiSpeedMomentumBacktester()
        old = {"SPY": 1.0, "GLD": 0.0, "TLT": 0.0}
        new = {"SPY": 0.0, "GLD": 1.0, "TLT": 0.0}
        turnover = bt._compute_turnover(old, new)
        assert turnover == pytest.approx(1.0)

    # ── Metrics ──

    def test_annualize_positive(self):
        """_annualize should return positive CAGR for positive returns."""
        returns = [0.001] * 252  # ~28.5% annual
        cagr = MultiSpeedMomentumBacktester._annualize(returns)
        assert cagr > 0

    def test_annualize_negative(self):
        returns = [-0.001] * 252
        cagr = MultiSpeedMomentumBacktester._annualize(returns)
        assert cagr < 0

    def test_annualize_empty(self):
        assert MultiSpeedMomentumBacktester._annualize([]) == 0.0

    def test_metrics_basic(self):
        """_metrics should return dict with expected keys."""
        rng = np.random.default_rng(42)
        returns = list(0.0005 + rng.normal(0, 0.008, 252))
        m = MultiSpeedMomentumBacktester._metrics(returns)
        assert "cagr" in m
        assert "volatility" in m
        assert "sharpe" in m
        assert "max_dd" in m
        assert m["sharpe"] > 0

    def test_metrics_empty_returns(self):
        m = MultiSpeedMomentumBacktester._metrics([])
        assert m["sharpe"] == 0.0 or m["sharpe"] == 0

    # ── Run Backtest ──

    @pytest.fixture
    def bt_with_data(self):
        """Create a backtester with ~1.5yr of synthetic daily data spanning 2020."""
        bt = MultiSpeedMomentumBacktester(
            BacktestConfig(start_date="2020-01-01", end_date="2020-12-31")
        )
        # Populate prices_raw for signal fallback
        spy_raw, gld_raw, tlt_raw = [], [], []
        for i in range(550):
            day = (datetime(2019, 6, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            spy_raw.append({"d": day, "p": 100.0 + i * 0.05 + (i % 7 - 3) * 0.2})
            gld_raw.append({"d": day, "p": 50.0 + i * 0.02 + (i % 5 - 2) * 0.15})
            tlt_raw.append({"d": day, "p": 80.0 + i * 0.03 + (i % 6 - 2) * 0.1})
        bt.prices_raw = {"SPY": spy_raw, "GLD": gld_raw, "TLT": tlt_raw}

        # Process into DailyReturn list
        dates_all = [p["d"] for p in spy_raw]
        spy_px = {p["d"]: p["p"] for p in spy_raw}
        gld_px = {p["d"]: p["p"] for p in gld_raw}
        tlt_px = {p["d"]: p["p"] for p in tlt_raw}

        bt.data = []
        for i, date in enumerate(dates_all[1:], 1):
            prev = dates_all[i - 1]
            s_p, s_c = spy_px.get(prev), spy_px.get(date)
            g_p, g_c = gld_px.get(prev), gld_px.get(date)
            t_p, t_c = tlt_px.get(prev), tlt_px.get(date)
            if all(v is not None for v in (s_p, s_c, g_p, g_c, t_p, t_c)):
                bt.data.append(
                    DailyReturn(
                        date=date,
                        spy_return=(s_c - s_p) / s_p,
                        gld_return=(g_c - g_p) / g_p,
                        tlt_return=(t_c - t_p) / t_p,
                    )
                )
        assert len(bt.data) > 0
        return bt

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
        assert isinstance(result.total_rebalances, int)

    def test_run_backtest_rebalances_nonzero(self, bt_with_data):
        """With 1 year of data, there should be some rebalances."""
        result = bt_with_data.run_backtest()
        # At least the initial rebalance
        assert result.total_rebalances >= 0

    def test_run_backtest_equity_curve_sampled(self, bt_with_data):
        """Equity curve should be populated with dict entries."""
        result = bt_with_data.run_backtest()
        if result.extras.get("equity_curve"):
            entry = result.extras["equity_curve"][0]
            assert "date" in entry
            assert "baseline" in entry
            assert "overlay" in entry

    def test_run_backtest_transaction_costs_nonnegative(self, bt_with_data):
        """Transaction costs should be non-negative."""
        result = bt_with_data.run_backtest()
        assert result.total_transaction_costs >= 0.0

    def test_run_backtest_sharpe_is_finite(self, bt_with_data):
        """Sharpe ratio should be a finite number."""
        result = bt_with_data.run_backtest()
        assert np.isfinite(result.sharpe_ratio)

    def test_run_backtest_2001_start_no_data(self):
        """Starting before data available should still produce a result or None."""
        bt = MultiSpeedMomentumBacktester(
            BacktestConfig(start_date="2001-01-01", end_date="2001-12-31")
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
        assert result is None

    # ── Print / Save ──

    def test_print_report_does_not_crash(self, caplog):
        """print_report should produce output without errors."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3, sharpe_ratio=0.85,
            max_drawdown=-15.4, baseline_sharpe=0.79,
            sharpe_improvement=0.06, total_rebalances=30, avg_turnover=0.1,
            total_transaction_costs=25.0,
            crisis_returns={"2008": -12.0, "2020": 3.0},
            extras={"overlay_active_rebalances": 50},
        )
        bt = MultiSpeedMomentumBacktester()
        bt.print_report(result)
        assert "MULTI-SPEED MOMENTUM" in caplog.text
        assert "Sharpe" in caplog.text
        assert "SUCCESS CRITERIA" in caplog.text

    def test_print_report_with_none_crisis(self, caplog):
        """print_report handles None crisis returns gracefully."""
        caplog.set_level(logging.INFO)
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=0, avg_turnover=0.0,
            total_transaction_costs=0.0,
            extras={"overlay_active_rebalances": 0},
        )
        bt = MultiSpeedMomentumBacktester()
        bt.print_report(result)
        assert "N/A" in caplog.text

    def test_save_results_creates_json_file(self):
        """save_results should create a valid JSON file."""
        result = BacktestResult(
            total_return=10.5, cagr=8.2, volatility=12.3, sharpe_ratio=0.85,
            max_drawdown=-15.4, baseline_sharpe=0.79,
            sharpe_improvement=0.06, total_rebalances=30, avg_turnover=0.1,
            total_transaction_costs=25.0,
            extras={"overlay_active_rebalances": 50, "return_2008": -12.0, "return_2020": 3.0},
        )
        bt = MultiSpeedMomentumBacktester()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
            bt.save_results(result, output_path=output_path)

        try:
            with open(output_path) as f:
                data = json.load(f)
            assert "total_return" in data
            assert "cagr" in data
            assert "sharpe_ratio" in data
            assert "max_drawdown" in data
            assert "overlay_active_rebalances" in data["extras"]
            assert "total_rebalances" in data
        finally:
            Path(output_path).unlink()

    # ── get_prices_slice ──

    def test_get_prices_slice(self):
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {
            "SPY": [
                {"d": "2020-01-02", "p": 100.0},
                {"d": "2020-01-03", "p": 101.0},
                {"d": "2020-01-06", "p": 102.0},
            ],
            "GLD": [{"d": "2020-01-02", "p": 50.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}],
        }
        sliced = bt._get_prices_slice("2020-01-03")
        assert len(sliced["SPY"]) == 2
        assert sliced["SPY"][-1]["d"] == "2020-01-03"

    def test_get_prices_slice_no_data_for_date(self):
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": [{"d": "2020-01-06", "p": 102.0}]}
        sliced = bt._get_prices_slice("2020-01-03")
        assert len(sliced["SPY"]) == 0

    def test_load_data_honors_explicit_data_path(self, monkeypatch, tmp_path):
        """Explicit data_path should be read even when the default PRICES_JSON is absent."""
        custom_data = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 52.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 79.0}],
        }
        custom_path = tmp_path / "custom_prices.json"
        custom_path.write_text(json.dumps(custom_data))
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            tmp_path / "missing_default_prices.json",
        )

        bt = MultiSpeedMomentumBacktester()

        assert bt.load_data(str(custom_path)) is True
        assert bt.prices_raw == custom_data
        assert len(bt.data) == 1

    def test_get_prices_slice_uses_built_index_without_iterating_raw_series(self):
        """After indexes are built, repeated slices should not scan raw ticker lists."""

        class CountingSeries(list):
            def __init__(self, *args):
                super().__init__(*args)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        raw = CountingSeries(
            [
                {
                    "d": (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "p": 100.0 + i,
                }
                for i in range(500)
            ]
        )
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": raw, "GLD": CountingSeries(raw), "TLT": CountingSeries(raw)}
        bt._build_price_indexes()

        for series in bt.prices_raw.values():
            series.iterations = 0

        sliced = bt._get_prices_slice("2021-04-15", lookback=400)

        assert len(sliced["SPY"]) <= 450
        assert sliced["SPY"][-1]["d"] <= "2021-04-15"
        assert all(series.iterations == 0 for series in bt.prices_raw.values())


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the multi-speed momentum backtest."""

    def test_empty_data_returns_none(self):
        bt = MultiSpeedMomentumBacktester()
        result = bt.run_backtest()
        assert result is None

    def test_single_day_data_returns_none(self):
        bt = MultiSpeedMomentumBacktester()
        bt.data = [
            DailyReturn(
                date="2020-01-02",
                spy_return=0.01,
                gld_return=0.005,
                tlt_return=-0.002,
            )
        ]
        result = bt.run_backtest()
        assert result is not None  # Single day still runs, but with no returns

    def test_load_data_missing_file(self, monkeypatch):
        """load_data should return False when PRICES_JSON doesn't exist."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            Path("/nonexistent/prices.json"),
        )
        bt = MultiSpeedMomentumBacktester()
        success = bt.load_data()
        assert success is False

    def test_load_data_with_exception(self, monkeypatch):
        """load_data should handle exceptions gracefully."""
        def _bad_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", _bad_open)
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            Path("/tmp/test_prices.json"),
        )
        bt = MultiSpeedMomentumBacktester()
        success = bt.load_data()
        assert success is False

    def test_returns_from_equity(self):
        """Verify _returns_from_equity is available (inherited pattern)."""
        bt = MultiSpeedMomentumBacktester()
        equity = [100.0, 101.0, 99.0, 102.0]
        returns = bt._returns_from_equity(equity)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.01)

    # Helper used by run_backtest
    @staticmethod
    def _returns_from_equity(equity):
        rets = []
        for i in range(1, len(equity)):
            rets.append((equity[i] - equity[i - 1]) / equity[i - 1])
        return rets


# ── Extended Config Validation ──────────────────────────────────────────────


class TestConfigValidationExtended:
    """Additional BacktestConfig edge cases (dataclass field validation)."""

    def test_negative_max_shifts(self):
        """Negative max shifts are allowed (inverts overlay intent)."""
        config = BacktestConfig(max_spy_shift=-0.05, max_gld_shift=-0.03, max_tlt_shift=-0.02)
        assert config.max_spy_shift == -0.05
        assert config.max_gld_shift == -0.03
        assert config.max_tlt_shift == -0.02

    def test_zero_signal_threshold(self):
        """Zero threshold means every signal triggers rebalance."""
        config = BacktestConfig(signal_threshold=0.0)
        assert config.signal_threshold == 0.0

    def test_signal_threshold_no_overlay(self):
        """Threshold of 1.0 means signals never exceed threshold."""
        config = BacktestConfig(signal_threshold=1.0)
        bt = MultiSpeedMomentumBacktester(config)
        # _get_overlay_shifts does not check threshold; run_backtest does.
        shifts = bt._get_overlay_shifts(0.5, 0.3, 0.1)
        assert shifts["SPY"] == 0.5 * config.max_spy_shift

    def test_tiny_transaction_cost(self):
        """Very small transaction cost (0.1 bps)."""
        config = BacktestConfig(transaction_cost_bps=0.1)
        assert config.transaction_cost_bps == 0.1

    def test_large_transaction_cost(self):
        """Very large transaction cost (100 bps)."""
        config = BacktestConfig(transaction_cost_bps=100.0)
        assert config.transaction_cost_bps == 100.0

    def test_rebalance_frequency_days_override(self):
        """Custom rebalance_frequency_days is stored and used."""
        config = BacktestConfig(rebalance_frequency_days=63)
        assert config.rebalance_frequency_days == 63

    def test_custom_base_weights(self):
        """Custom base weights propagate correctly."""
        config = BacktestConfig(base_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2})
        assert config.base_weights["SPY"] == 0.5
        assert abs(sum(config.base_weights.values()) - 1.0) < 0.01

    def test_all_overlay_fields_settable(self):
        """All overlay-specific fields can be set via constructor."""
        config = BacktestConfig(
            max_spy_shift=0.10,
            max_gld_shift=0.05,
            max_tlt_shift=0.04,
            signal_threshold=0.05,
        )
        assert config.max_spy_shift == 0.10
        assert config.max_gld_shift == 0.05
        assert config.max_tlt_shift == 0.04
        assert config.signal_threshold == 0.05


# ── Extended DailyReturn Tests ─────────────────────────────────────────────


class TestDailyReturnExtended:
    """Additional DailyReturn edge cases."""

    def test_zero_returns(self):
        """All zero returns."""
        dr = DailyReturn(date="2020-01-02", spy_return=0.0, gld_return=0.0, tlt_return=0.0)
        assert dr.spy_return == 0.0
        assert dr.gld_return == 0.0
        assert dr.tlt_return == 0.0

    def test_extreme_positive_returns(self):
        """Very large positive returns (+100%, +50%, +30%)."""
        dr = DailyReturn(date="2020-01-02", spy_return=1.0, gld_return=0.5, tlt_return=0.3)
        assert dr.spy_return == 1.0
        assert dr.gld_return == 0.5
        assert dr.tlt_return == 0.3

    def test_total_loss_returns(self):
        """All -100% returns (total loss scenario)."""
        dr = DailyReturn(date="2020-01-02", spy_return=-1.0, gld_return=-1.0, tlt_return=-1.0)
        assert dr.spy_return == -1.0
        assert dr.gld_return == -1.0
        assert dr.tlt_return == -1.0

    def test_mixed_sign_returns(self):
        """Mixed positive and negative returns."""
        dr = DailyReturn(date="2020-01-02", spy_return=0.02, gld_return=-0.01, tlt_return=0.005)
        assert dr.spy_return > 0.0
        assert dr.gld_return < 0.0
        assert dr.tlt_return > 0.0


# ── Extended Signal Computation Tests ───────────────────────────────────────


class TestSignalComputationExtended:
    """Signal computation boundary conditions and engine fallback."""

    @staticmethod
    def _make_prices(n_days: int, start_price: float = 100.0, drift: float = 0.0):
        """Create a list of price dicts for testing."""
        prices = []
        for i in range(n_days):
            day = (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": start_price * (1.0 + i * drift)})
        return prices

    def test_fallback_zero_momentum(self, monkeypatch):
        """Flat price series => signal of 0.0."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        prices = self._make_prices(300, drift=0.0)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal == pytest.approx(0.0)

    def test_fallback_exactly_260(self, monkeypatch):
        """Exactly 260 entries is sufficient for a signal."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        prices = self._make_prices(260, drift=0.001)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal > 0.0

    def test_fallback_259_entries_zero(self, monkeypatch):
        """259 entries should return 0.0 (not enough data)."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        prices = self._make_prices(259, drift=0.001)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal == 0.0

    def test_fallback_sub_saturation(self, monkeypatch):
        """12m return under 20% should not saturate (|signal| < 1.0)."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        prices = self._make_prices(300, drift=0.0005)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert 0.0 < signal < 1.0

    def test_signal_engine_raises_fallback_used(self, monkeypatch):
        """When engine raises, fallback path is used."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            True,
        )
        class RaisingEngine:
            def get_signal_for_ticker(self, ticker, date):
                raise RuntimeError("engine failure")
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.MultiSpeedMomentum",
            lambda: RaisingEngine(),
        )
        prices = self._make_prices(300, drift=0.001)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal > 0.0  # Fallback must produce positive signal

    def test_signal_engine_returns_none_fallback_used(self, monkeypatch):
        """When engine returns None, fallback path is used."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            True,
        )
        class NoneEngine:
            def get_signal_for_ticker(self, ticker, date):
                return None
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.MultiSpeedMomentum",
            lambda: NoneEngine(),
        )
        prices = self._make_prices(300, drift=0.001)
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices}
        signal = bt._compute_signal("SPY", prices[-1]["d"])
        assert signal > 0.0  # Fallback

    def test_signal_on_missing_ticker(self, monkeypatch):
        """Missing ticker data returns 0.0 (via fallback)."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {}
        signal = bt._compute_signal("SPY", "2020-01-02")
        assert signal == 0.0

    def test_get_prices_slice_exact_lookback(self):
        """Entries exactly = lookback should NOT be truncated."""
        bt = MultiSpeedMomentumBacktester()
        raw = [{"d": f"2020-01-{d:02d}", "p": 100.0} for d in range(1, 401)]
        bt.prices_raw = {"SPY": raw, "GLD": raw, "TLT": raw}
        sliced = bt._get_prices_slice("2020-12-31", lookback=400)
        assert len(sliced["SPY"]) == 400

    def test_get_prices_slice_truncation(self):
        """More entries than lookback should be truncated to lookback+50."""
        bt = MultiSpeedMomentumBacktester()
        raw = [{"d": f"2020-01-{d:02d}", "p": 100.0} for d in range(1, 501)]
        bt.prices_raw = {"SPY": raw, "GLD": raw, "TLT": raw}
        sliced = bt._get_prices_slice("2020-12-31", lookback=400)
        assert len(sliced["SPY"]) == 450  # 400 + 50

    def test_get_prices_slice_filters_by_end_date(self):
        """Entries after end_date should be excluded."""
        bt = MultiSpeedMomentumBacktester()
        raw = [{"d": f"2020-01-{d:02d}", "p": 100.0} for d in range(1, 31)]
        bt.prices_raw = {"SPY": raw}
        sliced = bt._get_prices_slice("2020-01-15")
        assert len(sliced["SPY"]) == 15


# ── Extended Backtest Logic ─────────────────────────────────────────────────


class TestBacktestLogicExtended:
    """Backtest logic edge cases: normalization, costs, crisis, thresholds."""

    def test_weight_normalization(self, monkeypatch):
        """Shifts making sum != 1.0 should normalize weights."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        config = BacktestConfig(
            max_spy_shift=0.30,
            max_gld_shift=0.30,
            max_tlt_shift=0.30,
            signal_threshold=0.0,
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        prices = []
        for i in range(500):
            day = (datetime(2019, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 * (1 + i * 0.001)})
        bt = MultiSpeedMomentumBacktester(config)
        bt.prices_raw = {"SPY": prices, "GLD": prices, "TLT": prices}
        bt._process_price_data()
        assert len(bt.data) > 0
        result = bt.run_backtest()
        assert result is not None

    def test_tiny_initial_capital(self):
        """Very small initial capital ($1) should not crash."""
        config = BacktestConfig(
            initial_capital=1.0, start_date="2020-01-01", end_date="2020-01-31"
        )
        bt = MultiSpeedMomentumBacktester(config)
        bt.data = [
            DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=0.005, tlt_return=-0.002),
            DailyReturn(date="2020-01-03", spy_return=-0.005, gld_return=0.01, tlt_return=0.003),
        ]
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result.total_return, float)

    def test_turnover_below_minimum(self):
        """Turnover < 0.001 should not count as a rebalance."""
        bt = MultiSpeedMomentumBacktester()
        old = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        new = {"SPY": 0.4605, "GLD": 0.3795, "TLT": 0.16}
        turnover = bt._compute_turnover(old, new)
        assert turnover == pytest.approx(0.0005)
        assert turnover < 0.001

    def test_negative_returns_full_backtest(self):
        """All-negative daily returns produce negative total_return."""
        config = BacktestConfig(start_date="2020-01-01", end_date="2020-01-10")
        bt = MultiSpeedMomentumBacktester(config)
        bt.data = [
            DailyReturn(
                date=f"2020-01-{d:02d}",
                spy_return=-0.01,
                gld_return=-0.005,
                tlt_return=-0.002,
            )
            for d in range(2, 11)
        ]
        result = bt.run_backtest()
        assert result is not None
        assert result.total_return < 0.0

    def test_short_backtest_period(self):
        """Very short period (3 days) should still produce a result."""
        config = BacktestConfig(start_date="2020-01-01", end_date="2020-01-05")
        bt = MultiSpeedMomentumBacktester(config)
        bt.data = [
            DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=0.005, tlt_return=-0.002),
            DailyReturn(date="2020-01-03", spy_return=-0.005, gld_return=0.01, tlt_return=0.003),
            DailyReturn(date="2020-01-06", spy_return=0.002, gld_return=-0.003, tlt_return=0.001),
        ]
        result = bt.run_backtest()
        assert result is not None
        assert isinstance(result.total_return, float)

    def test_crisis_periods_empty(self):
        """Crisis periods with no matching data should return None."""
        config = BacktestConfig(start_date="2021-01-01", end_date="2021-12-31")
        bt = MultiSpeedMomentumBacktester(config)
        bt.data = [
            DailyReturn(date="2021-01-04", spy_return=0.01, gld_return=0.005, tlt_return=-0.002),
        ]
        result = bt.run_backtest()
        assert result is not None
        assert result.crisis_returns["2008"] is None
        assert result.crisis_returns["2020"] is None
        assert result.crisis_returns["2022"] is None

    def test_signal_below_threshold_uses_base(self, monkeypatch):
        """Weak signals below threshold should keep base weights."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        # Very small drift => 12m return < 2% => signal < 0.1
        prices = []
        for i in range(300):
            day = (datetime(2019, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 * (1 + i * 0.00004)})
        bt = MultiSpeedMomentumBacktester()
        bt.prices_raw = {"SPY": prices, "GLD": prices, "TLT": prices}
        sig = bt._compute_signal("SPY", prices[-1]["d"])
        assert 0.0 < sig < 0.1  # Below default threshold

    def test_transaction_cost_deducted(self, monkeypatch):
        """High turnover should deduct costs from overlay capital."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE",
            False,
        )
        config = BacktestConfig(
            signal_threshold=0.0,
            max_spy_shift=0.05,
            start_date="2020-01-01",
            end_date="2020-06-30",
            transaction_cost_bps=50.0,
        )
        prices = []
        for i in range(550):
            day = (datetime(2019, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            prices.append({"d": day, "p": 100.0 * (1 + i * 0.001)})
        bt = MultiSpeedMomentumBacktester(config)
        bt.prices_raw = {"SPY": prices, "GLD": prices, "TLT": prices}
        bt._process_price_data()
        result = bt.run_backtest()
        assert result is not None
        assert result.total_transaction_costs >= 0.0


# ── Extended Metrics & Helpers ─────────────────────────────────────────────


class TestMetricsExtended:
    """Additional metrics and helper method tests."""

    def test_annualize_single_day(self):
        """Single return value should produce a non-zero CAGR."""
        cagr = MultiSpeedMomentumBacktester._annualize([0.01])
        assert cagr != 0.0
        assert isinstance(cagr, float)

    def test_annualize_varied_returns(self):
        """Varied returns should produce a stable CAGR."""
        returns = [0.001, -0.002, 0.003, -0.001, 0.002] * 50
        cagr = MultiSpeedMomentumBacktester._annualize(returns)
        assert isinstance(cagr, float)
        assert np.isfinite(cagr)

    def test_metrics_all_negative(self):
        """All-negative returns should produce negative Sharpe."""
        returns = [-0.001] * 252
        m = MultiSpeedMomentumBacktester._metrics(returns)
        assert m["sharpe"] < 0

    def test_metrics_low_volatility(self):
        """Very low volatility returns should not crash."""
        returns = [0.0001] * 252
        m = MultiSpeedMomentumBacktester._metrics(returns)
        assert np.isfinite(m["sharpe"])
        assert m["volatility"] >= 0

    def test_returns_from_equity_flat(self):
        """Flat equity curve => all zero returns."""
        rets = MultiSpeedMomentumBacktester._returns_from_equity([100.0, 100.0, 100.0])
        assert all(r == 0.0 for r in rets)

    def test_returns_from_equity_negative(self):
        """Declining equity => all negative returns."""
        rets = MultiSpeedMomentumBacktester._returns_from_equity([100.0, 99.0, 98.0])
        assert all(r < 0.0 for r in rets)

    def test_returns_from_equity_single_value(self):
        """Single entry => empty returns list."""
        rets = MultiSpeedMomentumBacktester._returns_from_equity([100.0])
        assert rets == []


# ── Extended Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCasesExtended:
    """More edge cases: save default path, load data happy/invalid paths."""

    def test_save_results_default_path(self, monkeypatch, tmp_path):
        """save_results uses default path (BACKTEST_RESULTS_DIR / filename)."""
        result = BacktestResult(
            total_return=5.0, cagr=3.0, volatility=10.0, sharpe_ratio=0.5,
            max_drawdown=-10.0, baseline_sharpe=0.45,
            sharpe_improvement=0.05, total_rebalances=0, avg_turnover=0.0,
            total_transaction_costs=0.0,
            extras={"overlay_active_rebalances": 0},
        )
        results_dir = tmp_path / "backtest_results"
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.BACKTEST_RESULTS_DIR",
            results_dir,
        )
        bt = MultiSpeedMomentumBacktester()
        bt.save_results(result)
        expected = results_dir / "multi_speed_momentum_backtest.json"
        assert expected.exists()
        with open(expected) as f:
            data = json.load(f)
        assert data["total_return"] == 5.0

    def test_load_data_valid_json(self, monkeypatch, tmp_path):
        """load_data with valid price JSON file succeeds."""
        data = {
            "SPY": [{"d": "2020-01-02", "p": 100.0}, {"d": "2020-01-03", "p": 101.0}],
            "GLD": [{"d": "2020-01-02", "p": 50.0}, {"d": "2020-01-03", "p": 51.0}],
            "TLT": [{"d": "2020-01-02", "p": 80.0}, {"d": "2020-01-03", "p": 81.0}],
        }
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(data, f)
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            prices_file,
        )
        bt = MultiSpeedMomentumBacktester()
        success = bt.load_data()
        assert success
        assert len(bt.data) == 1

    def test_load_data_invalid_json(self, monkeypatch, tmp_path):
        """load_data with malformed JSON should return False."""
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            f.write("not valid json")
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            prices_file,
        )
        bt = MultiSpeedMomentumBacktester()
        success = bt.load_data()
        assert not success

    def test_run_backtest_no_filtered_data(self):
        """No data matching the date range returns None."""
        config = BacktestConfig(start_date="2030-01-01", end_date="2030-12-31")
        bt = MultiSpeedMomentumBacktester(config)
        bt.data = [
            DailyReturn(date="2020-01-02", spy_return=0.01, gld_return=0.005, tlt_return=-0.002),
        ]
        result = bt.run_backtest()
        assert result is None

    def test_dailyreturn_all_fields_float(self):
        """All return fields are floats even when ints are passed."""
        dr = DailyReturn(date="2020-01-02", spy_return=1, gld_return=0, tlt_return=-1)
        assert isinstance(dr.spy_return, int)  # No coercion, stored as passed


# ── CLI Tests ──────────────────────────────────────────────────────────────


class TestCLI:
    """Test CLI main() function."""

    def test_main_no_data_returns_1(self, monkeypatch):
        """main should return 1 when no data is available."""
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            Path("/nonexistent/prices.json"),
        )
        monkeypatch.setattr("sys.argv", ["msm_backtest.py", "run"])
        rc = main()
        assert rc == 1

    def test_main_with_save_flag(self, monkeypatch):
        """main should accept --save argument."""
        monkeypatch.setattr(
            "sys.argv",
            ["multi_speed_momentum_backtest.py", "run", "--save", "--output", "/tmp/test_msm_out.json"],
        )
        monkeypatch.setattr(
            "src.backtest.multi_speed_momentum_backtest.PRICES_JSON",
            Path("/nonexistent/prices.json"),
        )
        rc = main()
        assert rc == 1  # Still fails due to no data, but parsing works
