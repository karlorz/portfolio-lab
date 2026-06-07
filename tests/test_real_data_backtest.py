"""
Tests for Real Data Combined Backtest (v4.90)
"""

import json
import logging
import math
import sqlite3
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from unittest.mock import patch

import numpy as np
import pytest

from src.backtest.metrics import BacktestResult
from src.backtest.real_data_backtest import (
    RealDataBacktest,
    run_real_data_backtest,
)
from src.backtest import real_data_backtest as real_data_backtest_module


class TestRealDataBacktestResult:
    """Test result dataclass."""

    def test_serializable(self):
        result = BacktestResult(
            total_return=87.1, cagr=15.3, volatility=11.6,
            sharpe_ratio=1.318, max_drawdown=-16.6,
            baseline_sharpe=1.204, sharpe_improvement=0.113,
            extras={
                "timestamp": "2026-05-16",
                "data_start": "2021-05-10",
                "data_end": "2026-05-15",
                "trading_days": 1200,
                "baseline_cagr": 14.8,
                "baseline_vol": 12.3,
                "baseline_max_dd": -19.1,
                "baseline_total_return": 82.2,
                "collar_sharpe": 1.22,
                "collar_dd": -16.5,
                "crypto_sharpe": 1.22,
                "bond_dur_sharpe": 1.22,
                "dd_improvement": 2.5,
                "collar_days_pct": 16.0,
                "crypto_days_pct": 55.0,
                "avg_tlt_sleeve_pct": 16.0,
                "meets_target": True,
                "recommendation": "Test recommendation",
            },
        )
        assert result.baseline_sharpe == 1.204
        assert result.extras["meets_target"]

    def test_asdict_has_core_fields(self):
        """Verify core metric fields survive asdict()."""
        result = BacktestResult(
            total_return=87.1, cagr=15.3, volatility=11.6,
            sharpe_ratio=1.318, max_drawdown=-16.6,
        )
        d = asdict(result)
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown"):
            assert field in d, f"Missing core field: {field}"

    def test_asdict_has_trade_fields(self):
        """Verify trade/cost fields survive asdict()."""
        result = BacktestResult(
            total_return=87.1, cagr=15.3, volatility=11.6,
            sharpe_ratio=1.318, max_drawdown=-16.6,
            total_rebalances=5, total_transaction_costs=0.05, avg_turnover=0.12,
        )
        d = asdict(result)
        assert d["total_rebalances"] == 5
        assert d["total_transaction_costs"] == 0.05
        assert d["avg_turnover"] == 0.12

    def test_asdict_has_optional_fields(self):
        """Verify optional baseline/sharpe fields survive asdict()."""
        result = BacktestResult(
            total_return=87.1, cagr=15.3, volatility=11.6,
            sharpe_ratio=1.318, max_drawdown=-16.6,
            baseline_sharpe=1.204, sharpe_improvement=0.113,
            extras={"key": "val"}, crisis_returns={"2008": -0.30},
        )
        d = asdict(result)
        assert d["baseline_sharpe"] == 1.204
        assert d["sharpe_improvement"] == 0.113
        assert d["extras"] == {"key": "val"}
        assert d["crisis_returns"] == {"2008": -0.30}

    def test_asdict_extras_has_all_keys(self):
        """Verify all extras keys present in the no-data fallback result."""
        bt = RealDataBacktest()
        bt.DATA_DIR = Path("/nonexistent")
        result = bt.run()
        d = asdict(result)
        extras = d["extras"]
        required = [
            "timestamp", "data_start", "data_end", "trading_days",
            "baseline_cagr", "baseline_vol", "baseline_max_dd",
            "baseline_total_return", "collar_sharpe", "collar_dd",
            "crypto_sharpe", "bond_dur_sharpe", "dd_improvement",
            "collar_days_pct", "crypto_days_pct", "avg_tlt_sleeve_pct",
            "meets_target", "recommendation",
        ]
        for key in required:
            assert key in extras, f"Missing extras key: {key}"

    def test_asdict_extras_types(self):
        """Verify extras values have expected types in no-data fallback."""
        bt = RealDataBacktest()
        bt.DATA_DIR = Path("/nonexistent")
        result = bt.run()
        extras = asdict(result)["extras"]
        assert isinstance(extras["trading_days"], int)
        assert isinstance(extras["meets_target"], bool)
        assert isinstance(extras["recommendation"], str)
        assert isinstance(extras["timestamp"], str)
        assert isinstance(extras["collar_days_pct"], float)


class TestRealDataBacktest:
    """Test real data backtest."""

    @pytest.fixture
    def bt(self):
        return RealDataBacktest()

    def test_compute_returns(self, bt):
        rets = bt._compute_returns([100, 110, 105])
        assert len(rets) == 2
        assert abs(rets[0] - 0.10) < 0.01

    def test_compute_returns_empty_list(self, bt):
        """Empty price list yields empty returns list."""
        rets = bt._compute_returns([])
        assert rets == []

    def test_compute_returns_single_element(self, bt):
        """Single price yields empty returns list (no pairs to diff)."""
        rets = bt._compute_returns([100.0])
        assert rets == []

    def test_compute_returns_constant_prices(self, bt):
        """All-identical prices yield all-zero returns."""
        rets = bt._compute_returns([100, 100, 100, 100])
        assert all(r == 0.0 for r in rets)

    def test_compute_returns_negative_returns(self, bt):
        """Decreasing prices yield strictly negative returns."""
        rets = bt._compute_returns([100, 90, 80, 70])
        assert all(r < 0 for r in rets)
        assert abs(rets[0] - (-0.10)) < 0.001
        assert abs(rets[2] - (-0.125)) < 0.001

    def test_compute_returns_large_swing(self, bt):
        """Large price jumps compute correctly."""
        rets = bt._compute_returns([100, 50, 200])
        assert abs(rets[0] - (-0.50)) < 0.001
        assert abs(rets[1] - 3.0) < 0.001

    def test_compute_returns_integer_prices(self, bt):
        """Integer prices produce float returns."""
        rets = bt._compute_returns([100, 110, 105])
        assert all(isinstance(r, float) for r in rets)

    def test_compute_rolling_vol(self, bt):
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0, 0.01, 100))
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == len(rets)

    def test_compute_rolling_vol_empty_returns_empty(self, bt):
        """Empty returns yields empty vol list."""
        vols = bt._compute_rolling_vol([], 30)
        assert vols == []

    def test_compute_rolling_vol_two_elements(self, bt):
        """Two returns returns [0.20, 0.20] due to warmup fallback."""
        vols = bt._compute_rolling_vol([0.01, -0.01], 30)
        assert len(vols) == 2
        assert vols[0] == 0.20  # i=0: warmup fallback
        assert vols[1] == 0.20  # i=1: i > 1 is False, warmup fallback

    def test_compute_rolling_vol_three_elements(self, bt):
        """Three elements: first two use warmup fallback, third computes."""
        vols = bt._compute_rolling_vol([0.01, -0.01, 0.02], 30)
        assert len(vols) == 3
        assert vols[0] == 0.20
        assert vols[1] == 0.20
        # Third element computes np.std([0.01, -0.01, 0.02]) * sqrt(252)
        assert vols[2] > 0

    def test_compute_rolling_vol_constant_returns(self, bt):
        """All-zero returns produce vol=0 (no warmup needed after window)."""
        rets = [0.0] * 50
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == 50
        # After warmup, all zeros
        for v in vols[35:]:
            assert v == 0.0

    def test_compute_rolling_vol_custom_window(self, bt):
        """Custom window size produces same-length output."""
        rng = np.random.RandomState(99)
        rets = list(rng.normal(0, 0.01, 50))
        vols_10 = bt._compute_rolling_vol(rets, 10)
        vols_30 = bt._compute_rolling_vol(rets, 30)
        assert len(vols_10) == len(rets)
        assert len(vols_30) == len(rets)

    def test_compute_rolling_vol_window_edge_exact(self, bt):
        """Exactly window-length returns behave correctly."""
        rets = [0.01] * 30
        vols = bt._compute_rolling_vol(rets, 30)
        assert len(vols) == 30
        # First element is warmup, last element uses full 30-element window
        assert vols[0] == 0.20

    def test_compute_rolling_vol_precomputes_without_daily_np_std(self, bt, monkeypatch):
        """Rolling volatility should preserve legacy values without per-day np.std calls."""
        rets = [((i % 17) - 8) / 1000 for i in range(500)]
        window = 30

        def population_std(values):
            mean = sum(values) / len(values)
            return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        expected = []
        for i in range(len(rets)):
            if i < window:
                expected.append(
                    population_std(rets[: i + 1]) * math.sqrt(252)
                    if i > 1
                    else 0.20
                )
            else:
                expected.append(population_std(rets[i - window : i]) * math.sqrt(252))

        def fail_std(*_args, **_kwargs):
            raise AssertionError("np.std should not be called once per rolling-vol day")

        monkeypatch.setattr("src.backtest.real_data_backtest.np.std", fail_std)

        vols = bt._compute_rolling_vol(rets, window)

        assert vols == pytest.approx(expected)

    def test_compute_rolling_vol_negative_returns(self, bt):
        """Negative-only returns still produce positive vol."""
        rets = [-0.01] * 40
        vols = bt._compute_rolling_vol(rets, 10)
        assert all(v >= 0 for v in vols)
        assert vols[35] == pytest.approx(0.0, abs=1e-10)  # constant returns -> std near zero

    def test_collar_signals(self, bt):
        assert bt._collar_signal(15.0) == 0.0
        assert bt._collar_signal(26.0) == -0.01
        assert bt._collar_signal(35.0) == -0.03
        assert bt._collar_signal(50.0) == -0.05

    def test_collar_signal_boundaries(self, bt):
        """Test exact boundary values."""
        assert bt._collar_signal(25.0) == 0.0     # <= 25
        assert bt._collar_signal(25.001) == -0.01  # > 25
        assert bt._collar_signal(30.0) == -0.01    # > 25, <= 30
        assert bt._collar_signal(30.001) == -0.03  # > 30
        assert bt._collar_signal(40.0) == -0.03    # > 30, <= 40
        assert bt._collar_signal(40.001) == -0.05  # > 40

    def test_collar_signal_negative_vix(self, bt):
        """Negative VIX treated as low vol -> no collar."""
        assert bt._collar_signal(-5.0) == 0.0
        assert bt._collar_signal(-100.0) == 0.0

    def test_collar_signal_zero_vix(self, bt):
        """Zero VIX treated as low vol -> no collar."""
        assert bt._collar_signal(0.0) == 0.0

    def test_main_logs_report_without_blank_logger_type_error(self, monkeypatch, caplog):
        """CLI report should emit blank lines with logger.info("") not bare logger.info()."""
        result = BacktestResult(
            total_return=10.0,
            cagr=5.0,
            volatility=8.0,
            sharpe_ratio=0.7,
            max_drawdown=-12.0,
            baseline_sharpe=0.6,
            sharpe_improvement=0.1,
            extras={
                "data_start": "2022-01-01",
                "data_end": "2026-01-01",
                "trading_days": 100,
                "baseline_cagr": 4.0,
                "baseline_vol": 8.5,
                "baseline_max_dd": -14.0,
                "baseline_total_return": 9.0,
                "dd_improvement": 2.0,
                "collar_days_pct": 10.0,
                "crypto_days_pct": 20.0,
                "avg_tlt_sleeve_pct": 30.0,
                "meets_target": False,
                "recommendation": "test",
            },
        )
        monkeypatch.setattr(
            real_data_backtest_module.RealDataBacktest,
            "run",
            lambda _self: result,
        )
        monkeypatch.setattr("sys.argv", ["real_data_backtest.py", "run"])
        caplog.set_level(logging.INFO)

        real_data_backtest_module.main()

        assert "REAL DATA COMBINED BACKTEST" in caplog.text

    def test_bond_duration_signals(self, bt):
        t, i, s = bt._bond_duration_signal(0.15, 0)
        assert t > i  # Strong TLT rally -> heavy TLT

        t, i, s = bt._bond_duration_signal(-0.15, 0)
        assert s > t  # TLT decline -> heavy SHY

    def test_bond_weights_sum_to_one(self, bt):
        for mom in [-0.20, -0.05, 0.0, 0.05, 0.20]:
            t, i, s = bt._bond_duration_signal(mom, 0)
            assert abs(t + i + s - 1.0) < 0.01

    def test_bond_duration_signal_boundaries(self, bt):
        """Test exact boundary momentum values."""
        # > 0.10: heavy TLT
        t, i, s = bt._bond_duration_signal(0.100001, 0)
        assert t == 0.60 and i == 0.25 and s == 0.15

        # > 0.0 and <= 0.10: balanced
        t, i, s = bt._bond_duration_signal(0.10, 0)
        assert t == 0.30 and i == 0.45 and s == 0.25
        t, i, s = bt._bond_duration_signal(0.000001, 0)
        assert t == 0.30 and i == 0.45 and s == 0.25

        # > -0.10 and <= 0.0: defensive
        t, i, s = bt._bond_duration_signal(0.0, 0)
        assert t == 0.10 and i == 0.40 and s == 0.50
        t, i, s = bt._bond_duration_signal(-0.099999, 0)
        assert t == 0.10 and i == 0.40 and s == 0.50

        # <= -0.10: heavy SHY
        t, i, s = bt._bond_duration_signal(-0.10, 0)
        assert t == 0.0 and i == 0.30 and s == 0.70

    def test_bond_duration_signal_large_positive(self, bt):
        """Very large positive momentum still uses max TLT bucket."""
        t, i, s = bt._bond_duration_signal(100.0, 0)
        assert t == 0.60 and i == 0.25 and s == 0.15

    def test_bond_duration_signal_large_negative(self, bt):
        """Very large negative momentum still uses max SHY bucket."""
        t, i, s = bt._bond_duration_signal(-100.0, 0)
        assert t == 0.0 and i == 0.30 and s == 0.70

    def test_crypto_signal_bull(self, bt):
        w = bt._crypto_signal(0.5, 0.3, 0.6, 0.7)
        assert w > 0

    def test_crypto_signal_extreme_vol(self, bt):
        w = bt._crypto_signal(0.5, 0.3, 1.5, 0.7)
        assert w == 0.0

    def test_crypto_signal_bear(self, bt):
        w = bt._crypto_signal(-0.3, -0.2, 0.6, 0.7)
        assert w == 0.0

    def test_crypto_weight_capped(self, bt):
        w = bt._crypto_signal(3.0, 3.0, 0.3, 0.3)
        assert w <= 0.05

    def test_crypto_signal_vol_exactly_one(self, bt):
        """Vol exactly at 1.0 threshold is allowed (not > 1.0)."""
        w = bt._crypto_signal(0.5, 0.3, 1.0, 0.7)
        assert w > 0  # Not rejected because vol == 1.0, not > 1.0

    def test_crypto_signal_vol_exactly_one_eth(self, bt):
        """ETH vol exactly at 1.0 threshold is allowed."""
        w = bt._crypto_signal(0.5, 0.3, 0.7, 1.0)
        assert w > 0

    def test_crypto_signal_mom_exactly_zero(self, bt):
        """Both momentum exactly zero -> no allocation."""
        w = bt._crypto_signal(0.0, 0.0, 0.6, 0.7)
        assert w == 0.0

    def test_crypto_signal_one_positive_mom(self, bt):
        """One positive, one negative momentum -> uses only the positive."""
        w = bt._crypto_signal(0.5, -0.3, 0.6, 0.7)
        expected = min(0.05, 0.02 + 0.03 * (0.5 / 2))
        assert abs(w - expected) < 0.001
        assert w > 0

    def test_crypto_signal_zero_vol(self, bt):
        """Zero vol is allowed through the vol gate."""
        w = bt._crypto_signal(0.5, 0.3, 0.0, 0.0)
        assert w > 0

    def test_crypto_signal_max_weight(self, bt):
        """Very high momentum caps at 0.05."""
        w = bt._crypto_signal(10.0, 10.0, 0.3, 0.3)
        assert w == 0.05

    def test_baseline_allocation_keys(self, bt):
        """BASELINE has expected ticker keys (lowercase)."""
        for key in ("spy", "gld", "tlt"):
            assert key in bt.BASELINE, f"Missing key: {key}"

    def test_baseline_allocation_sums_to_one(self, bt):
        """Baseline weights sum to 1.0."""
        total = sum(bt.BASELINE.values())
        assert abs(total - 1.0) < 0.001

    def test_baseline_allocation_values_in_range(self, bt):
        """All baseline weights are between 0 and 1."""
        for key, val in bt.BASELINE.items():
            assert 0 < val < 1, f"Weight {key}={val} out of range"

    def test_baseline_allocation_rejects_zero(self, bt):
        """No zero-weight entries in baseline."""
        for val in bt.BASELINE.values():
            assert val > 0

    def test_run_with_real_data(self, bt):
        """Should work when market.db is available."""
        result = bt.run()
        assert isinstance(result, BacktestResult)
        # If data loaded successfully
        if result.extras["trading_days"] > 0:
            assert result.baseline_sharpe != 0
            assert result.sharpe_ratio != 0
            assert result.extras["recommendation"] is not None

    def test_convenience_function(self):
        result = run_real_data_backtest()
        assert isinstance(result, BacktestResult)


class TestEdgeCases:
    """Edge cases."""

    def test_no_data_returns_safe_result(self):
        bt = RealDataBacktest()
        # Mock _load_market_data to return empty dict
        with patch.object(bt, '_load_market_data', return_value={}):
            result = bt.run()
        assert result.extras["trading_days"] == 0
        assert "No data" in result.extras["recommendation"]

    def test_no_data_meets_target_false(self):
        """No-data result must have meets_target=False."""
        bt = RealDataBacktest()
        with patch.object(bt, '_load_market_data', return_value={}):
            result = bt.run()
        assert result.extras["meets_target"] is False

    def test_no_data_returns_zero_metrics(self):
        """No-data result must have all zero metrics."""
        bt = RealDataBacktest()
        with patch.object(bt, '_load_market_data', return_value={}):
            result = bt.run()
        assert result.total_return == 0.0
        assert result.cagr == 0.0
        assert result.volatility == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.baseline_sharpe == 0.0
        assert result.sharpe_improvement == 0.0

    def test_empty_market_db_returns_empty_data(self, tmp_path):
        """Empty SQLite db returns empty data dict."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices ("
            "  symbol TEXT, date TEXT, close REAL,"
            "  PRIMARY KEY (symbol, date)"
            ")"
        )
        conn.commit()
        conn.close()

        bt = RealDataBacktest()
        bt.DATA_DIR = tmp_path
        data = bt._load_market_data()
        assert data == {}, f"Expected empty data dict, got {data}"

    def test_market_db_with_spy_only(self, tmp_path):
        """DB with only SPY data loads SPY but not GLD/TLT."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices ("
            "  symbol TEXT, date TEXT, close REAL,"
            "  PRIMARY KEY (symbol, date)"
            ")"
        )
        for date, close in [("2021-01-04", 370.0), ("2021-01-05", 371.0)]:
            conn.execute(
                "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                ("SPY", date, close),
            )
        conn.commit()
        conn.close()

        bt = RealDataBacktest()
        bt.DATA_DIR = tmp_path
        data = bt._load_market_data()
        assert "SPY" in data
        assert "GLD" not in data
        assert "TLT" not in data
        assert len(data["SPY"]["prices"]) == 2

    def test_insufficient_spy_days_returns_only_spy(self, tmp_path):
        """Very few SPY days in DB returns empty data (GLD/TLT missing)."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices ("
            "  symbol TEXT, date TEXT, close REAL,"
            "  PRIMARY KEY (symbol, date)"
            ")"
        )
        for date in ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]:
            conn.execute(
                "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                ("SPY", date, 370.0),
            )
        conn.commit()
        conn.close()

        bt = RealDataBacktest()
        bt.DATA_DIR = tmp_path
        data = bt._load_market_data()
        assert "SPY" in data
        assert len(data["SPY"]["dates"]) == 4

    def test_run_result_has_all_core_fields(self):
        """run() result populates all core BacktestResult fields."""
        bt = RealDataBacktest()
        result = bt.run()
        d = asdict(result)
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown"):
            assert field in d
            assert d[field] is not None

    def test_class_instantiation_no_args(self):
        """RealDataBacktest() can be instantiated without arguments."""
        bt = RealDataBacktest()
        assert bt.DATA_DIR is not None
        assert isinstance(bt.BASELINE, dict)
