#!/usr/bin/env python3
"""
Tests for cta_backtest.py — BacktestResult dataclass, return/drawdown calculation,
acceptance criteria validation, crisis alpha logic, and backtest engine.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from src.backtest.cta_backtest import (
    BacktestResult,
    CTABacktestEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtest_result(**overrides):
    defaults = dict(
        start_date="2021-01-01",
        end_date="2026-05-13",
        total_return=0.45,
        annualized_return=0.08,
        volatility=0.10,
        sharpe_ratio=0.80,
        max_drawdown=-0.15,
        calmar_ratio=0.53,
        num_trades=50,
        win_rate=0.60,
        avg_trade_return=0.002,
        crisis_alpha_2008=0.0,
        crisis_alpha_2020=0.0,
        crisis_alpha_2022=0.05,
        vs_spy_correlation=0.35,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _setup_test_db(db_path, days=300):
    """Create a minimal prices table with synthetic data."""
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL, volume INTEGER)")
    rng = np.random.RandomState(42)
    symbols = {"SPY": 450, "GLD": 190, "TLT": 95, "QQQ": 380, "IWM": 200}
    today = date(2026, 5, 14)
    for sym, start_price in symbols.items():
        price = float(start_price)
        for i in range(days):
            d = today - timedelta(days=days - i)
            if d.weekday() >= 5:
                continue
            price *= (1 + rng.normal(0.0003, 0.012))
            c.execute("INSERT INTO prices VALUES (?, ?, ?, ?)",
                      (sym, d.isoformat(), price, int(rng.uniform(1e6, 1e8))))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# BacktestResult Tests
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_fields(self):
        r = _make_backtest_result()
        assert r.start_date == "2021-01-01"
        assert r.total_return == 0.45
        assert r.sharpe_ratio == 0.80

    def test_crisis_alpha_fields(self):
        r = _make_backtest_result(crisis_alpha_2022=0.08)
        assert r.crisis_alpha_2022 == 0.08

    def test_correlation(self):
        r = _make_backtest_result(vs_spy_correlation=0.25)
        assert r.vs_spy_correlation == 0.25


# ---------------------------------------------------------------------------
# CTABacktestEngine — constants
# ---------------------------------------------------------------------------

class TestEngineConstants:

    def test_crisis_periods(self):
        assert "2008" in CTABacktestEngine.CRISIS_PERIODS
        assert "2020" in CTABacktestEngine.CRISIS_PERIODS
        assert "2022" in CTABacktestEngine.CRISIS_PERIODS

    def test_sg_proxy_universe(self):
        assert "SPY" in CTABacktestEngine.SG_PROXY_UNIVERSE
        assert "GLD" in CTABacktestEngine.SG_PROXY_UNIVERSE

    def test_init(self, tmp_path):
        db = tmp_path / "test.db"
        engine = CTABacktestEngine(db_path=db)
        assert engine.db_path == db


# ---------------------------------------------------------------------------
# CTABacktestEngine — _fetch_historical_data
# ---------------------------------------------------------------------------

class TestFetchHistoricalData:

    def test_fetch_returns_data(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTABacktestEngine(db_path=db)
        data = engine._fetch_historical_data("SPY", "2025-01-01", "2026-05-14")
        assert len(data) > 0
        assert "close" in data[0]

    def test_fetch_empty(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTABacktestEngine(db_path=db)
        data = engine._fetch_historical_data("FAKE", "2025-01-01", "2026-05-14")
        assert data == []

    def test_fetch_missing_db(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "nonexistent.db")
        data = engine._fetch_historical_data("SPY", "2025-01-01", "2026-05-14")
        assert data == []


# ---------------------------------------------------------------------------
# CTABacktestEngine — _calculate_returns
# ---------------------------------------------------------------------------

class TestCalculateReturns:

    def test_basic_returns(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        prices = [100, 102, 101, 105]
        returns = engine._calculate_returns(prices)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.02)
        assert returns[1] == pytest.approx(-1 / 102)
        assert returns[2] == pytest.approx(4 / 101)

    def test_single_price(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        returns = engine._calculate_returns([100])
        assert len(returns) == 0

    def test_constant_prices(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        returns = engine._calculate_returns([100, 100, 100])
        assert all(r == pytest.approx(0.0) for r in returns)


# ---------------------------------------------------------------------------
# CTABacktestEngine — _calculate_max_drawdown
# ---------------------------------------------------------------------------

class TestCalculateMaxDrawdown:

    def test_no_drawdown(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        equity = np.array([100, 105, 110, 115, 120])
        dd = engine._calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.0)

    def test_simple_drawdown(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        equity = np.array([100, 110, 90, 95, 100])
        dd = engine._calculate_max_drawdown(equity)
        # Peak was 110, trough was 90 → dd = (90-110)/110 = -0.1818
        assert dd == pytest.approx(-20 / 110, abs=0.01)

    def test_monotonic_increase(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        equity = np.array([100, 101, 102, 103, 104, 105])
        dd = engine._calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_single_value(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        equity = np.array([100])
        dd = engine._calculate_max_drawdown(equity)
        assert dd == 0.0


# ---------------------------------------------------------------------------
# CTABacktestEngine — validate_acceptance_criteria
# ---------------------------------------------------------------------------

class TestValidateAcceptanceCriteria:

    def test_all_pass(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(
            num_trades=50,
            volatility=0.10,
            crisis_alpha_2022=0.05,
            vs_spy_correlation=0.30,
            sharpe_ratio=0.80,
        )
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["multi_timeframe_trend"]["status"] == "PASS"
        assert criteria["volatility_targeting"]["status"] == "PASS"
        assert criteria["crisis_alpha_2022"]["status"] == "PASS"
        assert criteria["low_correlation"]["status"] == "PASS"
        assert criteria["positive_sharpe"]["status"] == "PASS"

    def test_2008_not_applicable(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result()
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["crisis_alpha_2008"]["status"] == "NOT_APPLICABLE"

    def test_2020_not_applicable(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result()
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["crisis_alpha_2020"]["status"] == "NOT_APPLICABLE"

    def test_2008_applicable_pass(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(crisis_alpha_2008=0.05)
        criteria = engine.validate_acceptance_criteria(result, "2005-01-01")
        assert criteria["crisis_alpha_2008"]["status"] == "PASS"

    def test_2008_applicable_fail(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(crisis_alpha_2008=-0.10)
        criteria = engine.validate_acceptance_criteria(result, "2005-01-01")
        assert criteria["crisis_alpha_2008"]["status"] == "FAIL"

    def test_low_trades_conditional(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(num_trades=2)
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["multi_timeframe_trend"]["status"] == "CONDITIONAL"

    def test_high_volatility_conditional(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(volatility=0.25)
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["volatility_targeting"]["status"] == "CONDITIONAL"

    def test_high_correlation_conditional(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(vs_spy_correlation=0.70)
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["low_correlation"]["status"] == "CONDITIONAL"

    def test_negative_sharpe_fail(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(sharpe_ratio=-0.10)
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["positive_sharpe"]["status"] == "FAIL"

    def test_2022_alpha_fail(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result(crisis_alpha_2022=-0.05)
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        assert criteria["crisis_alpha_2022"]["status"] == "FAIL"

    def test_criteria_structure(self, tmp_path):
        engine = CTABacktestEngine(db_path=tmp_path / "x.db")
        result = _make_backtest_result()
        criteria = engine.validate_acceptance_criteria(result, "2021-01-01")
        for name, check in criteria.items():
            assert "status" in check
            assert "detail" in check
            assert check["status"] in ("PASS", "FAIL", "CONDITIONAL", "NOT_APPLICABLE")


# ---------------------------------------------------------------------------
# CTABacktestEngine — _calculate_crisis_alpha
# ---------------------------------------------------------------------------

class TestCrisisAlpha:

    def test_returns_dict_keys(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTABacktestEngine(db_path=db)
        # Use dates within our test data range
        dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
        returns = np.random.normal(0.0003, 0.01, len(dates))
        equity = 100000 * np.cumprod(1 + returns)
        alpha = engine._calculate_crisis_alpha(dates, returns, equity)
        assert isinstance(alpha, dict)

    def test_missing_dates_zero(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTABacktestEngine(db_path=db)
        # Dates that don't overlap crisis periods
        dates = ["2030-01-01", "2030-01-02", "2030-01-03"]
        returns = np.array([0.01, -0.005, 0.008])
        equity = np.array([100, 101, 100.49])
        alpha = engine._calculate_crisis_alpha(dates, returns, equity)
        assert all(v == 0 for v in alpha.values())


# ---------------------------------------------------------------------------
# CTABacktestEngine — run_backtest (integration)
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_raises_on_no_data(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, volume INTEGER)")
        conn.commit()
        conn.close()
        engine = CTABacktestEngine(db_path=db)
        with pytest.raises(ValueError, match="No data found"):
            engine.run_backtest("2020-01-01", "2026-01-01")

    def test_returns_backtest_result(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, days=400)
        engine = CTABacktestEngine(db_path=db)
        result = engine.run_backtest("2025-01-01", "2026-05-14")
        assert isinstance(result, BacktestResult)
        assert result.start_date == "2025-01-01"
        assert result.end_date == "2026-05-14"

    def test_result_fields_populated(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, days=400)
        engine = CTABacktestEngine(db_path=db)
        result = engine.run_backtest("2025-01-01", "2026-05-14")
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown <= 0  # Drawdown is always <= 0


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestMain:

    def test_main_no_data(self, tmp_path, capsys):
        """main() should return 1 when no data available."""
        with patch("src.backtest.cta_backtest.CTABacktestEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine._fetch_historical_data.return_value = []
            mock_cls.return_value = mock_engine
            from src.backtest.cta_backtest import main
            result = main()
            assert result == 1
