"""
Tests for Factor Rotation Walk-Forward Backtest — v3.00 Phase 3 Validation
"""

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factor_test_db(db_path: Path, vix_values: list = None):
    """Create a test market.db with synthetic factor ETF and VIX data."""
    if vix_values is None:
        np.random.seed(42)
        vix_values = [18.0 + 5 * math.sin(i / 60) + np.random.normal(0, 2) for i in range(300)]

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date)
        )""")

        base_spy = 400.0
        base_mtum = 180.0
        base_usmv = 75.0
        base_vlue = 100.0

        for i, vix in enumerate(vix_values):
            month = 1 + i // 21
            day = 1 + i % 21
            if month > 12:
                month = 12
                day = min(day, 28)
            date_str = f"2022-{month:02d}-{day:02d}"

            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES ('^VIX', ?, ?)",
                (date_str, vix),
            )

            spy_ret = np.random.normal(0.0003, 0.01)
            base_spy *= (1 + spy_ret)
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES ('SPY', ?, ?)",
                (date_str, round(base_spy, 2)),
            )

            mtum_ret = spy_ret + np.random.normal(0.0000, 0.003)
            base_mtum *= (1 + mtum_ret)
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES ('MTUM', ?, ?)",
                (date_str, round(base_mtum, 2)),
            )

            usmv_ret = spy_ret * 0.7 + np.random.normal(0.0001, 0.005)
            base_usmv *= (1 + usmv_ret)
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES ('USMV', ?, ?)",
                (date_str, round(base_usmv, 2)),
            )

            vlue_ret = spy_ret * 0.9 + np.random.normal(-0.0001, 0.006)
            base_vlue *= (1 + vlue_ret)
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES ('VLUE', ?, ?)",
                (date_str, round(base_vlue, 2)),
            )

        conn.commit()


@pytest.fixture
def factor_test_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_factor.db"
        _make_factor_test_db(db_path)
        yield db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFactorRotationBacktest:
    """Tests for the factor rotation backtest engine."""

    def test_import(self):
        from src.backtest.factor_rotation_backtest import (
            FactorRotationBacktest,
            FactorBacktestResult,
        )
        assert FactorRotationBacktest is not None
        assert FactorBacktestResult is not None

    def test_result_dataclass(self):
        from src.backtest.factor_rotation_backtest import FactorBacktestResult

        r = FactorBacktestResult(
            timestamp="",
            start_date="",
            end_date="",
            trading_days=0,
            baseline_cagr=0.0,
            baseline_vol=0.0,
            baseline_sharpe=0.0,
            baseline_max_dd=0.0,
            baseline_crisis_2022=0.0,
            overlay_cagr=0.0,
            overlay_vol=0.0,
            overlay_sharpe=0.0,
            overlay_max_dd=0.0,
            overlay_crisis_2022=0.0,
            sharpe_delta=0.0,
            dd_improvement=0.0,
            cagr_delta=0.0,
            avg_mtum_weight=0.0,
            avg_qual_weight=0.0,
            avg_usmv_weight=0.0,
            avg_vlue_weight=0.0,
            regime_breakdown={},
            regime_bull_sharpe=0.0,
            regime_neutral_sharpe=0.0,
            regime_elevated_sharpe=0.0,
            regime_high_vol_sharpe=0.0,
            regime_crisis_sharpe=0.0,
            meets_sharpe_target=False,
            meets_dd_target=False,
        )
        assert r.sharpe_delta == 0.0
        assert r.trading_days == 0

    def test_to_dict(self):
        from src.backtest.factor_rotation_backtest import FactorBacktestResult

        r = FactorBacktestResult(
            timestamp="2026-01-01",
            start_date="2021-01-01",
            end_date="2026-01-01",
            trading_days=1260,
            baseline_cagr=12.0,
            baseline_vol=15.0,
            baseline_sharpe=0.80,
            baseline_max_dd=-22.0,
            baseline_crisis_2022=-15.0,
            overlay_cagr=13.0,
            overlay_vol=14.0,
            overlay_sharpe=0.88,
            overlay_max_dd=-19.0,
            overlay_crisis_2022=-12.0,
            sharpe_delta=0.08,
            dd_improvement=3.0,
            cagr_delta=1.0,
            avg_mtum_weight=35.0,
            avg_qual_weight=30.0,
            avg_usmv_weight=25.0,
            avg_vlue_weight=10.0,
            regime_breakdown={"bull": 30.0, "neutral": 40.0},
            regime_bull_sharpe=0.9,
            regime_neutral_sharpe=0.8,
            regime_elevated_sharpe=0.7,
            regime_high_vol_sharpe=0.6,
            regime_crisis_sharpe=0.3,
            meets_sharpe_target=True,
            meets_dd_target=True,
        )

        d = r.to_dict()
        assert d["baseline_sharpe"] == 0.80
        assert d["sharpe_delta"] == 0.08
        assert d["meets_sharpe_target"] is True
        assert d["meets_dd_target"] is True
        assert d["avg_mtum_weight"] == 35.0
        assert "bull" in d["regime_breakdown"]

        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["overlay_sharpe"] == 0.88

    def test_backtest_runs(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        assert result.trading_days > 0
        assert result.start_date is not None

    def test_result_fields_populated(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        assert isinstance(result.baseline_sharpe, float)
        assert isinstance(result.overlay_sharpe, float)
        assert isinstance(result.avg_mtum_weight, float)
        assert isinstance(result.regime_breakdown, dict)
        assert isinstance(result.meets_sharpe_target, bool)
        assert isinstance(result.meets_dd_target, bool)

    def test_allocations_sum_to_one(self):
        from src.backtest.factor_rotation_backtest import REGIME_ALLOCATIONS

        for regime, alloc in REGIME_ALLOCATIONS.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.001, f"{regime} allocation sums to {total}"

    def test_vix_to_regime(self):
        from src.backtest.factor_rotation_backtest import _vix_to_regime

        assert _vix_to_regime(12.0) == "bull"
        assert _vix_to_regime(17.0) == "neutral"
        assert _vix_to_regime(22.0) == "elevated"
        assert _vix_to_regime(27.0) == "high_vol"
        assert _vix_to_regime(35.0) == "crisis"

    def test_empty_result_on_no_data(self):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()

            bt = FactorRotationBacktest(cache_db=db)
            result = bt.run(start_date="1999-01-01", end_date="1999-12-31")
            assert result.trading_days == 0

    def test_price_loading(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        prices = bt._load_prices(["SPY", "MTUM"], "2022-01-01", "2022-12-31")

        assert "SPY" in prices
        assert "MTUM" in prices
        if prices["SPY"]:
            first_date = next(iter(prices["SPY"]))
            assert prices["SPY"][first_date] > 0

    def test_vix_loading(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        vix = bt._load_vix("2022-01-01", "2022-12-31")
        assert len(vix) > 0

    def test_regime_sharpes_all_five(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        dates = [f"2022-{1+i//21:02d}-{1+i%21:02d}" for i in range(100)]
        rets = np.array([0.001] * 100, dtype=np.float64)

        sharpes = bt._all_regime_sharpes(dates, rets)
        assert len(sharpes) == 5
        assert all(isinstance(s, float) for s in sharpes)

    def test_max_drawdown(self):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        rets = np.array([-0.05, -0.05, 0.02, 0.02, 0.02])
        dd = FactorRotationBacktest._max_drawdown(rets)
        assert dd <= -0.049

    def test_year_return(self):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        dates = ["2021-12-31", "2022-01-03", "2022-01-04"]
        rets = np.array([0.01, 0.02, -0.01])
        yr = FactorRotationBacktest._year_return(dates, rets, "2022")
        assert yr > 0.005

    def test_regime_counts_tracked(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import FactorRotationBacktest

        bt = FactorRotationBacktest(cache_db=factor_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        total_pct = sum(result.regime_breakdown.values())
        assert abs(total_pct - 100.0) < 1.0

    def test_meets_targets(self):
        from src.backtest.factor_rotation_backtest import FactorBacktestResult

        def _make(s_delta, dd_imp):
            return FactorBacktestResult(
                timestamp="", start_date="", end_date="", trading_days=0,
                baseline_cagr=0, baseline_vol=0, baseline_sharpe=0,
                baseline_max_dd=0, baseline_crisis_2022=0,
                overlay_cagr=0, overlay_vol=0, overlay_sharpe=0,
                overlay_max_dd=0, overlay_crisis_2022=0,
                sharpe_delta=s_delta, dd_improvement=dd_imp, cagr_delta=0,
                avg_mtum_weight=0, avg_qual_weight=0, avg_usmv_weight=0, avg_vlue_weight=0,
                regime_breakdown={},
                regime_bull_sharpe=0, regime_neutral_sharpe=0,
                regime_elevated_sharpe=0, regime_high_vol_sharpe=0, regime_crisis_sharpe=0,
                meets_sharpe_target=(s_delta >= 0.05),
                meets_dd_target=(dd_imp >= 2.0),
            )

        assert _make(0.07, 3.0).meets_sharpe_target is True
        assert _make(0.03, 3.0).meets_sharpe_target is False
        assert _make(0.07, 1.0).meets_dd_target is False
        assert _make(0.07, 2.0).meets_dd_target is True

    def test_cli_imports(self, factor_test_db):
        from src.backtest.factor_rotation_backtest import (
            FactorRotationBacktest,
            FactorBacktestResult,
            REGIME_ALLOCATIONS,
            _vix_to_regime,
        )
        bt = FactorRotationBacktest(cache_db=factor_test_db)
        assert bt is not None
        assert len(REGIME_ALLOCATIONS) == 5
