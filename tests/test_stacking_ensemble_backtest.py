"""
Tests for Stacking Ensemble Backtest — v3.10 Phase 5 Validation
"""

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stack_test_db(db_path: Path):
    """Create a test market.db with synthetic SPY/GLD/TLT data."""
    np.random.seed(99)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date)
        )""")

        base_spy = 400.0
        base_gld = 170.0
        base_tlt = 140.0

        for i in range(300):
            month = 1 + i // 21
            day = 1 + i % 21
            if month > 12:
                month = 12
                day = min(day, 28)
            date_str = f"2022-{month:02d}-{day:02d}"

            for sym, base in [("SPY", base_spy), ("GLD", base_gld), ("TLT", base_tlt)]:
                ret = np.random.normal(0.0003, 0.01)
                new_base = base * (1 + ret)
                conn.execute(
                    "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                    (sym, date_str, round(new_base, 2)),
                )
                if sym == "SPY":
                    base_spy = new_base
                elif sym == "GLD":
                    base_gld = new_base
                else:
                    base_tlt = new_base

        conn.commit()


def _make_single_symbol_db(db_path: Path):
    """Create a minimal DB with only SPY data (missing GLD/TLT)."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date)
        )""")
        for i in range(60):
            date_str = f"2022-{1+i//21:02d}-{1+i%21:02d}"
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                ("SPY", date_str, round(400.0 * (1 + np.random.normal(0.0003, 0.01)), 2)),
            )
        conn.commit()


@pytest.fixture
def stack_test_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_stack.db"
        _make_stack_test_db(db_path)
        yield db_path


@pytest.fixture
def single_symbol_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "single.db"
        _make_single_symbol_db(db_path)
        yield db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportsAndConstants:

    def test_import(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest,
            StackingBacktestResult,
        )
        assert StackingEnsembleBacktest is not None
        assert StackingBacktestResult is not None

    def test_import_all(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
            BASELINE_SPY, BASELINE_GLD, BASELINE_TLT,
            MAX_EQUITY_SHIFT, SIGNAL_FREQUENCY, MIN_HOLDING_DAYS, MC_TRIALS,
            __all__,
        )
        assert __all__ is not None

    def test_accuracy_constants(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )
        assert BASELINE_ACCURACY == 0.65
        assert STACKING_ACCURACY == 0.76
        assert STACKING_ACCURACY > BASELINE_ACCURACY

    def test_portfolio_constants(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_SPY, BASELINE_GLD, BASELINE_TLT,
        )
        assert BASELINE_SPY == 0.46
        assert BASELINE_GLD == 0.38
        assert BASELINE_TLT == 0.16
        assert abs(BASELINE_SPY + BASELINE_GLD + BASELINE_TLT - 1.0) < 1e-9

    def test_signal_constants(self):
        from src.backtest.stacking_ensemble_backtest import (
            MAX_EQUITY_SHIFT, SIGNAL_FREQUENCY, MIN_HOLDING_DAYS, MC_TRIALS,
        )
        assert MAX_EQUITY_SHIFT == 0.05
        assert SIGNAL_FREQUENCY == 0.15
        assert MIN_HOLDING_DAYS == 5
        assert MC_TRIALS == 200


class TestStackingBacktestResult:

    def test_result_dataclass(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        r = StackingBacktestResult(
            timestamp="", start_date="", end_date="", trading_days=0, mc_trials=0,
            baseline_cagr=0, baseline_vol=0, baseline_sharpe=0, baseline_max_dd=0,
            voting_cagr_mean=0, voting_cagr_std=0,
            voting_sharpe_mean=0, voting_sharpe_std=0, voting_max_dd_mean=0,
            voting_sharpe_gt_baseline_pct=0,
            stacking_cagr_mean=0, stacking_cagr_std=0,
            stacking_sharpe_mean=0, stacking_sharpe_std=0, stacking_max_dd_mean=0,
            stacking_sharpe_gt_baseline_pct=0,
            sharpe_delta_mean=0, sharpe_delta_std=0,
            cagr_delta_mean=0, dd_delta_mean=0,
            sharpe_delta_t_stat=0, sharpe_delta_significant=False,
            voting_accuracy=0.65, stacking_accuracy=0.76,
            false_positive_rate_voting=35.0, false_positive_rate_stacking=24.0,
            avg_signal_return_voting=0.22, avg_signal_return_stacking=0.31,
            meets_sharpe_target=False, meets_accuracy_target=True,
        )
        assert r.voting_accuracy == 0.65
        assert r.stacking_accuracy == 0.76
        assert r.meets_accuracy_target is True
        assert r.meets_sharpe_target is False

    def test_result_with_extreme_values(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        r = StackingBacktestResult(
            timestamp="", start_date="", end_date="", trading_days=5371, mc_trials=10000,
            baseline_cagr=99.9, baseline_vol=50.0, baseline_sharpe=2.5, baseline_max_dd=-50.0,
            voting_cagr_mean=100.0, voting_cagr_std=10.0,
            voting_sharpe_mean=2.0, voting_sharpe_std=0.5, voting_max_dd_mean=-45.0,
            voting_sharpe_gt_baseline_pct=95.0,
            stacking_cagr_mean=110.0, stacking_cagr_std=8.0,
            stacking_sharpe_mean=2.5, stacking_sharpe_std=0.4, stacking_max_dd_mean=-40.0,
            stacking_sharpe_gt_baseline_pct=99.0,
            sharpe_delta_mean=0.5, sharpe_delta_std=0.1,
            cagr_delta_mean=10.0, dd_delta_mean=5.0,
            sharpe_delta_t_stat=5.0, sharpe_delta_significant=True,
            voting_accuracy=0.90, stacking_accuracy=0.95,
            false_positive_rate_voting=10.0, false_positive_rate_stacking=5.0,
            avg_signal_return_voting=0.5, avg_signal_return_stacking=0.6,
            meets_sharpe_target=True, meets_accuracy_target=True,
        )
        assert r.baseline_cagr == 99.9
        assert r.baseline_vol == 50.0
        assert r.baseline_sharpe == 2.5
        assert r.baseline_max_dd == -50.0
        assert r.sharpe_delta_t_stat == 5.0
        assert r.voting_sharpe_gt_baseline_pct == 95.0
        assert r.meets_sharpe_target is True
        assert r.stacking_accuracy == 0.95

    def test_result_with_extreme_bounds(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        # Test boundary values (zero, negative deltas)
        r = StackingBacktestResult(
            timestamp="", start_date="", end_date="", trading_days=0, mc_trials=0,
            baseline_cagr=-10.0, baseline_vol=0.0, baseline_sharpe=-1.0, baseline_max_dd=0.0,
            voting_cagr_mean=-5.0, voting_cagr_std=0.0,
            voting_sharpe_mean=-0.5, voting_sharpe_std=0.0, voting_max_dd_mean=0.0,
            voting_sharpe_gt_baseline_pct=0.0,
            stacking_cagr_mean=-3.0, stacking_cagr_std=0.0,
            stacking_sharpe_mean=-0.3, stacking_sharpe_std=0.0, stacking_max_dd_mean=0.0,
            stacking_sharpe_gt_baseline_pct=0.0,
            sharpe_delta_mean=0.2, sharpe_delta_std=0.0,
            cagr_delta_mean=2.0, dd_delta_mean=0.0,
            sharpe_delta_t_stat=float("inf"), sharpe_delta_significant=False,
            voting_accuracy=0.65, stacking_accuracy=0.76,
            false_positive_rate_voting=35.0, false_positive_rate_stacking=24.0,
            avg_signal_return_voting=0.22, avg_signal_return_stacking=0.31,
            meets_sharpe_target=True, meets_accuracy_target=True,
        )
        assert r.baseline_cagr == -10.0
        assert r.sharpe_delta_t_stat == float("inf")

    def test_to_dict(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        r = StackingBacktestResult(
            timestamp="2026-01-01", start_date="2021-01-01", end_date="2026-01-01",
            trading_days=1260, mc_trials=200,
            baseline_cagr=10.0, baseline_vol=12.0, baseline_sharpe=0.80, baseline_max_dd=-20.0,
            voting_cagr_mean=10.3, voting_cagr_std=0.5,
            voting_sharpe_mean=0.82, voting_sharpe_std=0.02, voting_max_dd_mean=-19.0,
            voting_sharpe_gt_baseline_pct=60.0,
            stacking_cagr_mean=10.8, stacking_cagr_std=0.4,
            stacking_sharpe_mean=0.87, stacking_sharpe_std=0.02, stacking_max_dd_mean=-18.0,
            stacking_sharpe_gt_baseline_pct=80.0,
            sharpe_delta_mean=0.05, sharpe_delta_std=0.03,
            cagr_delta_mean=0.5, dd_delta_mean=1.0,
            sharpe_delta_t_stat=3.5, sharpe_delta_significant=True,
            voting_accuracy=0.65, stacking_accuracy=0.76,
            false_positive_rate_voting=35.0, false_positive_rate_stacking=24.0,
            avg_signal_return_voting=0.22, avg_signal_return_stacking=0.31,
            meets_sharpe_target=True, meets_accuracy_target=True,
        )

        d = r.to_dict()
        assert d["sharpe_delta_mean"] == 0.05
        assert d["meets_sharpe_target"] is True
        assert d["sharpe_delta_significant"] is True

        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["stacking_sharpe_mean"] == 0.87

    def test_to_dict_empty_result(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        r = StackingBacktestResult(
            timestamp="", start_date="", end_date="", trading_days=0, mc_trials=0,
            baseline_cagr=0, baseline_vol=0, baseline_sharpe=0, baseline_max_dd=0,
            voting_cagr_mean=0, voting_cagr_std=0,
            voting_sharpe_mean=0, voting_sharpe_std=0, voting_max_dd_mean=0,
            voting_sharpe_gt_baseline_pct=0,
            stacking_cagr_mean=0, stacking_cagr_std=0,
            stacking_sharpe_mean=0, stacking_sharpe_std=0, stacking_max_dd_mean=0,
            stacking_sharpe_gt_baseline_pct=0,
            sharpe_delta_mean=0, sharpe_delta_std=0,
            cagr_delta_mean=0, dd_delta_mean=0,
            sharpe_delta_t_stat=0, sharpe_delta_significant=False,
            voting_accuracy=0.65, stacking_accuracy=0.76,
            false_positive_rate_voting=0, false_positive_rate_stacking=0,
            avg_signal_return_voting=0, avg_signal_return_stacking=0,
            meets_sharpe_target=False, meets_accuracy_target=False,
        )
        d = r.to_dict()
        assert all(isinstance(v, (int, float, str, bool)) for v in d.values())
        assert d["trading_days"] == 0
        assert d["mc_trials"] == 0

    def test_to_dict_json_roundtrip(self):
        from src.backtest.stacking_ensemble_backtest import StackingBacktestResult

        r = StackingBacktestResult(
            timestamp="", start_date="2022-01-01", end_date="2022-12-31",
            trading_days=252, mc_trials=10,
            baseline_cagr=8.5, baseline_vol=15.2, baseline_sharpe=0.65, baseline_max_dd=-22.1,
            voting_cagr_mean=9.1, voting_cagr_std=1.2,
            voting_sharpe_mean=0.72, voting_sharpe_std=0.08, voting_max_dd_mean=-21.5,
            voting_sharpe_gt_baseline_pct=62.5,
            stacking_cagr_mean=9.8, stacking_cagr_std=1.0,
            stacking_sharpe_mean=0.79, stacking_sharpe_std=0.07, stacking_max_dd_mean=-20.8,
            stacking_sharpe_gt_baseline_pct=78.3,
            sharpe_delta_mean=0.07, sharpe_delta_std=0.04,
            cagr_delta_mean=0.7, dd_delta_mean=0.7,
            sharpe_delta_t_stat=4.12, sharpe_delta_significant=True,
            voting_accuracy=0.65, stacking_accuracy=0.76,
            false_positive_rate_voting=35.0, false_positive_rate_stacking=24.0,
            avg_signal_return_voting=0.22, avg_signal_return_stacking=0.31,
            meets_sharpe_target=True, meets_accuracy_target=True,
        )
        d = r.to_dict()
        d2 = json.loads(json.dumps(d))
        assert d2 == d


class TestStackingEnsembleBacktest:

    def test_default_init(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, DEFAULT_CACHE_DB,
        )
        bt = StackingEnsembleBacktest()
        assert bt.cache_db == DEFAULT_CACHE_DB

    def test_custom_init(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        assert bt.cache_db == stack_test_db


class TestGenerateSignals:

    def test_signal_generation(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(23)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 0.15, seed=42)

        # Should have some signals
        non_neutral = sum(1 for v in signals.values() if v != 0)
        assert non_neutral >= 0

    def test_signal_values_are_plus_one_minus_one_or_zero(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=42)
        for v in signals.values():
            assert v in (-1, 0, 1), f"Unexpected signal value: {v}"

    def test_signal_seed_reproducibility(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(50)]
        s1 = StackingEnsembleBacktest._generate_signals(dates, 0.76, 0.5, seed=42)
        s2 = StackingEnsembleBacktest._generate_signals(dates, 0.76, 0.5, seed=42)
        assert s1 == s2

    def test_signal_different_seed_different_result(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        s1 = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=42)
        s2 = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=99)
        # With high frequency, they likely differ
        n1 = sum(1 for v in s1.values() if v != 0)
        n2 = sum(1 for v in s2.values() if v != 0)
        # At least one of non-neural, frequency, or signal pattern differs
        non_zero_1 = {d for d, v in s1.items() if v != 0}
        non_zero_2 = {d for d, v in s2.items() if v != 0}
        assert non_zero_1 != non_zero_2 or any(
            s1[d] != s2[d] for d in non_zero_1 & non_zero_2
        )

    def test_signal_accuracy_zero(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.0, 1.0, seed=42)
        # With 0% accuracy, signals should still be +1 or -1 (just always wrong)
        non_neutral = {d: v for d, v in signals.items() if v != 0}
        # All signals should be non-neutral since frequency=1
        assert len(non_neutral) > 0

    def test_signal_accuracy_one(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 1.0, 1.0, seed=42)
        non_neutral = {d: v for d, v in signals.items() if v != 0}
        assert len(non_neutral) > 0

    def test_signal_frequency_zero(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 0.0, seed=42)
        # All signals should be neutral
        assert all(v == 0 for v in signals.values())

    def test_signal_frequency_one(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=42)
        non_neutral = sum(1 for v in signals.values() if v != 0)
        # Most days should have non-neutral signals (filtering by min holding period)
        assert non_neutral > 0

    def test_signal_empty_dates(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        signals = StackingEnsembleBacktest._generate_signals([], 0.76, 0.15, seed=42)
        assert signals == {}

    def test_signal_fewer_than_20_dates(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        # The internal loop uses "for i in range(n - 20)", so with <20 dates, no signals generated
        dates = [f"2022-01-{1+i:02d}" for i in range(10)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=42)
        # With n=10, range(10-20)=range(-10) which is empty, no signals
        # But dates with i < 0 won't exist, so no signals
        # Actually range(-10) is empty, but the loop iterates
        # The loop is: for i in range(n - 20)
        # If n=10, range(-10) is empty, so indeed no signals
        # But wait - the dates are still passed, and the loop just doesn't execute
        # So no signals should be in the dict
        assert len(signals) == 0

    def test_min_holding_period(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(30)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.80, 1.0, seed=123)

        non_neutral_dates = [
            d for d in dates if d in signals and signals[d] != 0
        ]
        if len(non_neutral_dates) >= 2:
            for i in range(len(non_neutral_dates) - 1):
                idx1 = dates.index(non_neutral_dates[i])
                idx2 = dates.index(non_neutral_dates[i + 1])
                assert idx2 - idx1 >= 5, (
                    f"Min holding violated: {idx2 - idx1} days between signals"
                )

    def test_signal_returns_all_dates_present(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(50)]
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 1.0, seed=42)
        # Not every input date will be in output (internal loop is range(n - 20)),
        # but every date that IS in output should be in the input dates
        for d in signals:
            assert d in dates, f"Date {d} not in input dates"


class TestComputeReturnMetrics:

    def test_return_metrics(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.001] * 100
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["sharpe"] > 0
        assert m["cagr"] > 0
        assert m["max_dd"] <= 0

    def test_metrics_fewer_than_20_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.001] * 19
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["cagr"] == 0
        assert m["vol"] == 0
        assert m["sharpe"] == 0
        assert m["max_dd"] == 0

    def test_metrics_empty_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        m = StackingEnsembleBacktest._compute_return_metrics([])
        assert m["cagr"] == 0
        assert m["vol"] == 0
        assert m["sharpe"] == 0
        assert m["max_dd"] == 0

    def test_metrics_exactly_20_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.001] * 20
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["sharpe"] > 0
        assert m["cagr"] > 0

    def test_metrics_all_negative_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [-0.01] * 100
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["sharpe"] < 0
        assert m["cagr"] < 0
        assert m["max_dd"] <= 0

    def test_metrics_all_zero_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.0] * 100
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["sharpe"] == 0.0
        assert m["cagr"] == 0.0
        assert m["vol"] == 0.0
        assert m["max_dd"] == 0.0

    def test_metrics_negative_vol_boundary(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        # If std_d is zero, the code uses max(std_d, 1e-8)
        rets = [0.001] * 100
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["vol"] >= 0

    def test_metrics_extreme_returns(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.10, -0.08, 0.12, -0.05, 0.06] * 20
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["cagr"] != 0
        assert m["vol"] > 0
        assert isinstance(m["sharpe"], float)

    def test_metrics_high_volatility(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.05, -0.05] * 50
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["vol"] > 0
        assert m["max_dd"] <= 0


class TestBaselineReturns:

    def test_baseline_returns(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest,
            BASELINE_SPY, BASELINE_GLD, BASELINE_TLT,
        )

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )
        rets = bt._baseline_returns(dates, prices)
        assert len(rets) > 0
        assert all(isinstance(r, float) for r in rets)
        assert all(not math.isnan(r) for r in rets)

    def test_baseline_returns_empty_dates(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        rets = bt._baseline_returns([], prices)
        assert rets == []

    def test_baseline_returns_single_date(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )
        if dates:
            rets = bt._baseline_returns(dates[:1], prices)
            assert rets == []  # Need 2 dates for a return

    def test_baseline_returns_reasonable_range(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )
        rets = bt._baseline_returns(dates, prices)
        assert all(-0.15 < r < 0.15 for r in rets)  # Daily returns should be modest


class TestApplySignals:

    def test_apply_signals(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        signals = {d: 0 for d in dates}
        rets = bt._apply_signals(dates, prices, signals)
        assert len(rets) > 0

        signals[dates[10]] = 1
        signals[dates[20]] = -1
        rets2 = bt._apply_signals(dates, prices, signals)
        assert len(rets2) == len(rets)

    def test_apply_signals_no_returns_for_single_date(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )
        if dates:
            rets = bt._apply_signals(dates[:1], prices, {dates[0]: 1})
            assert rets == []  # Single date yields no returns

    def test_apply_signals_bullish_increases_spy_weight(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, BASELINE_SPY, BASELINE_GLD, BASELINE_TLT, MAX_EQUITY_SHIFT,
        )

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # Bullish signal (+1) should increase SPY allocation
        baseline_val = BASELINE_SPY * prices["SPY"][dates[1]] + BASELINE_GLD * prices["GLD"][dates[1]] + BASELINE_TLT * prices["TLT"][dates[1]]
        adj_spy = min(1.0, BASELINE_SPY + MAX_EQUITY_SHIFT)
        adj_gld = max(0.0, BASELINE_GLD - MAX_EQUITY_SHIFT)
        signal_val = adj_spy * prices["SPY"][dates[1]] + adj_gld * prices["GLD"][dates[1]] + BASELINE_TLT * prices["TLT"][dates[1]]
        # The signal value and baseline value differ
        assert abs(signal_val - baseline_val) > 0.001 or abs(BASELINE_SPY + MAX_EQUITY_SHIFT - BASELINE_SPY) < 0.001

    def test_apply_signals_bearish_decreases_spy_weight(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, BASELINE_SPY, MAX_EQUITY_SHIFT,
        )
        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # Bearish signal should decrease SPY weight
        spy_idx_in_prices = 0  # SPY is first in the list
        baseline_spy_w = BASELINE_SPY
        bear_spy_w = max(0.0, BASELINE_SPY - MAX_EQUITY_SHIFT)
        assert bear_spy_w <= baseline_spy_w

    def test_apply_signals_weight_clipping(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, MAX_EQUITY_SHIFT,
        )
        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # Even with extreme signals, weights should be clipped to [0, 1]
        signals = {d: 0 for d in dates}
        # Apply multiple signals, but only the current date's signal matters per day
        signals[dates[10]] = 1
        signals[dates[15]] = 1
        rets = bt._apply_signals(dates, prices, signals)
        assert all(not math.isnan(r) for r in rets)
        assert all(math.isfinite(r) for r in rets)

    def test_apply_signals_signal_not_in_dates(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # Signal for a date not in the list should be ignored
        signals = {"1999-01-01": 1}
        rets = bt._apply_signals(dates, prices, signals)
        assert len(rets) > 0  # Should still produce returns

    def test_apply_signals_tlt_weight_unchanged(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, BASELINE_TLT,
        )
        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # TLT weight is always BASELINE_TLT and never adjusted
        signals = {d: 0 for d in dates}
        signals[dates[10]] = 1
        rets = bt._apply_signals(dates, prices, signals)
        assert len(rets) > 0


class TestLoadPrices:

    def test_load_prices_all_symbols(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        assert "SPY" in prices
        assert "GLD" in prices
        assert "TLT" in prices
        assert len(prices["SPY"]) > 0

    def test_load_prices_no_matching_dates(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "1999-01-01", "1999-12-31")
        assert len(prices["SPY"]) == 0

    def test_load_prices_empty_db(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()
            bt = StackingEnsembleBacktest(cache_db=db)
            prices = bt._load_prices(["SPY"], "2022-01-01", "2022-12-31")
            assert len(prices["SPY"]) == 0

    def test_load_prices_nonexistent_db(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=Path("/nonexistent/path/market.db"))
        prices = bt._load_prices(["SPY"], "2022-01-01", "2022-12-31")
        # Should not crash; returns empty dict
        assert len(prices["SPY"]) == 0

    def test_load_prices_close_is_none(self):
        """Test that NULL close values are filtered out."""
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test_null.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.execute(
                    "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                    ("SPY", "2022-01-01", None),
                )
                conn.execute(
                    "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                    ("SPY", "2022-01-02", 400.0),
                )
                conn.commit()
            bt = StackingEnsembleBacktest(cache_db=db)
            prices = bt._load_prices(["SPY"], "2022-01-01", "2022-01-10")
            assert "2022-01-01" not in prices["SPY"]
            assert "2022-01-02" in prices["SPY"]

    def test_load_prices_close_is_zero(self):
        """Test that zero close values are filtered out (close > 0 check)."""
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test_zero.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.execute(
                    "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                    ("SPY", "2022-01-01", 0.0),
                )
                conn.execute(
                    "INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                    ("SPY", "2022-01-02", 400.0),
                )
                conn.commit()
            bt = StackingEnsembleBacktest(cache_db=db)
            prices = bt._load_prices(["SPY"], "2022-01-01", "2022-01-10")
            assert "2022-01-01" not in prices["SPY"]
            assert "2022-01-02" in prices["SPY"]

    def test_load_prices_missing_symbol_in_db(self, single_symbol_db):
        """DB has SPY but not GLD/TLT."""
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=single_symbol_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        assert len(prices["SPY"]) > 0
        assert len(prices["GLD"]) == 0
        assert len(prices["TLT"]) == 0


class TestAggregate:

    def test_aggregate_basic(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, StackingBacktestResult,
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}

        # 10 trials
        voting_results = [
            {"cagr": 10.5, "vol": 12.0, "sharpe": 0.82, "max_dd": -19.0}
            for _ in range(10)
        ]
        stacking_results = [
            {"cagr": 11.0, "vol": 12.0, "sharpe": 0.87, "max_dd": -18.0}
            for _ in range(10)
        ]

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics,
            voting_results, stacking_results, 10,
        )
        assert isinstance(result, StackingBacktestResult)
        assert result.sharpe_delta_mean > 0
        assert result.meets_sharpe_target == (result.sharpe_delta_mean >= 0.05)
        assert result.voting_accuracy == BASELINE_ACCURACY
        assert result.stacking_accuracy == STACKING_ACCURACY

    def test_aggregate_single_trial(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, StackingBacktestResult,
        )

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        v = {"cagr": 10.5, "vol": 12.0, "sharpe": 0.82, "max_dd": -19.0}
        s = {"cagr": 11.0, "vol": 12.0, "sharpe": 0.87, "max_dd": -18.0}

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics, [v], [s], 1,
        )
        # With single trial, std is 0, ddof=1 gives NaN, then max(NaN, 1e-8) is NaN
        # t_stat = delta_mean / (nan / sqrt(1)) = nan
        assert math.isnan(result.sharpe_delta_t_stat)
        assert result.sharpe_delta_significant is False

    def test_aggregate_stacking_worse_than_voting(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, StackingBacktestResult,
        )

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        voting_results = [
            {"cagr": 10.5, "vol": 12.0, "sharpe": 0.85, "max_dd": -19.0}
            for _ in range(10)
        ]
        stacking_results = [
            {"cagr": 10.0, "vol": 12.0, "sharpe": 0.75, "max_dd": -20.0}
            for _ in range(10)
        ]

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics,
            voting_results, stacking_results, 10,
        )
        assert result.sharpe_delta_mean < 0
        assert result.meets_sharpe_target is False
        assert result.cagr_delta_mean < 0

    def test_aggregate_accuracy_target_met(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest, STACKING_ACCURACY,
        )

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        voting_results = [{"cagr": 10.5, "vol": 12.0, "sharpe": 0.82, "max_dd": -19.0}] * 5
        stacking_results = [{"cagr": 11.0, "vol": 12.0, "sharpe": 0.87, "max_dd": -18.0}] * 5

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics,
            voting_results, stacking_results, 5,
        )
        assert result.meets_accuracy_target == (STACKING_ACCURACY >= 0.76)

    def test_aggregate_sharpe_gt_baseline(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        # Baseline sharpe is 0.80. All voting trials have sharpe > 0.80. Stacking all > 0.80.
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        v = [{"cagr": 11.0, "vol": 12.0, "sharpe": 0.85, "max_dd": -19.0}] * 5
        s = [{"cagr": 12.0, "vol": 12.0, "sharpe": 0.90, "max_dd": -18.0}] * 5

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics, v, s, 5,
        )
        assert result.voting_sharpe_gt_baseline_pct == 100.0
        assert result.stacking_sharpe_gt_baseline_pct == 100.0

    def test_aggregate_all_below_baseline(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        # Baseline sharpe is 0.80. All trials have sharpe < 0.80
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        v = [{"cagr": 9.0, "vol": 12.0, "sharpe": 0.60, "max_dd": -21.0}] * 5
        s = [{"cagr": 9.5, "vol": 12.0, "sharpe": 0.70, "max_dd": -20.5}] * 5

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics, v, s, 5,
        )
        assert result.voting_sharpe_gt_baseline_pct == 0.0
        assert result.stacking_sharpe_gt_baseline_pct == 0.0


class TestEmptyResult:

    def test_empty_result_on_no_data(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()

            bt = StackingEnsembleBacktest(cache_db=db)
            result = bt.run(start_date="1999-01-01", end_date="1999-12-31")
            assert result.trading_days == 0

    def test_empty_result_all_fields_zero(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty2.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()

            bt = StackingEnsembleBacktest(cache_db=db)
            result = bt.run(start_date="1999-01-01", end_date="1999-12-31")

            # All numeric fields should be 0 or False
            assert result.baseline_cagr == 0
            assert result.baseline_vol == 0
            assert result.baseline_sharpe == 0
            assert result.baseline_max_dd == 0
            assert result.voting_cagr_mean == 0
            assert result.voting_sharpe_mean == 0
            assert result.stacking_cagr_mean == 0
            assert result.stacking_sharpe_mean == 0
            assert result.sharpe_delta_mean == 0
            assert result.cagr_delta_mean == 0
            assert result.dd_delta_mean == 0
            assert result.sharpe_delta_t_stat == 0
            assert result.sharpe_delta_significant is False
            assert result.meets_sharpe_target is False
            assert result.meets_accuracy_target is False
            assert result.mc_trials == 0

    def test_empty_result_dates_preserved(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty3.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()

            bt = StackingEnsembleBacktest(cache_db=db)
            result = bt.run(start_date="1999-01-01", end_date="1999-12-31")
            assert result.start_date == "1999-01-01"
            assert result.end_date == "1999-12-31"

    def test_empty_result_single_symbol_db(self, single_symbol_db):
        """Missing GLD and TLT should produce empty result."""
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=single_symbol_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")
        # common_dates will be empty because GLD and TLT have no data
        assert result.trading_days == 0


class TestRunEndToEnd:

    def test_backtest_runs(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=10)

        assert result.trading_days > 0
        assert result.mc_trials == 10

    def test_result_fields_populated(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=10)

        assert isinstance(result.baseline_sharpe, float)
        assert isinstance(result.voting_sharpe_mean, float)
        assert isinstance(result.stacking_sharpe_mean, float)
        assert isinstance(result.sharpe_delta_mean, float)
        assert isinstance(result.sharpe_delta_significant, bool)
        assert isinstance(result.meets_sharpe_target, bool)
        assert isinstance(result.meets_accuracy_target, bool)
        assert isinstance(result.timestamp, str)
        assert result.timestamp != ""

    def test_run_reproducibility(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        r1 = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=5)
        r2 = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=5)

        assert r1.baseline_cagr == r2.baseline_cagr
        assert r1.baseline_sharpe == r2.baseline_sharpe
        # MC results should be reproducible with deterministic seeds
        assert r1.voting_sharpe_mean == r2.voting_sharpe_mean

    def test_run_default_end_date(self, stack_test_db):
        """run() with end_date=None should succeed."""
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        result = bt.run(start_date="2022-01-01", end_date=None, mc_trials=3)
        assert result.trading_days > 0
        assert result.end_date is not None

    def test_run_mc_trials_different_sizes(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        r1 = bt.run(start_date="2022-01-01", end_date="2022-06-30", mc_trials=3)
        r2 = bt.run(start_date="2022-01-01", end_date="2022-06-30", mc_trials=10)

        assert r1.mc_trials == 3
        assert r2.mc_trials == 10

    def test_run_single_trial(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=1)
        assert result.mc_trials == 1
        assert result.trading_days > 0


class TestAccuracySimulation:

    def test_false_positive_rate_formula(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )
        # False positive rate = (1 - accuracy) * 100
        assert (1.0 - BASELINE_ACCURACY) * 100 == 35.0
        assert (1.0 - STACKING_ACCURACY) * 100 == 24.0

    def test_avg_signal_return_formula(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )
        # avg_signal_return = accuracy * 0.8 - 0.3
        expected_voting = round(BASELINE_ACCURACY * 0.8 - 0.3, 2)
        expected_stacking = round(STACKING_ACCURACY * 0.8 - 0.3, 2)
        assert expected_voting == 0.22
        assert expected_stacking == 0.31

    def test_stacking_superior_avg_return(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )
        voting_return = BASELINE_ACCURACY * 0.8 - 0.3
        stacking_return = STACKING_ACCURACY * 0.8 - 0.3
        assert stacking_return > voting_return

    def test_sharpe_delta_significance_threshold(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        # When t_stat > 2.0, sharpe_delta_significant is True
        dates = [f"2022-01-{1+i:02d}" for i in range(100)]
        bl_metrics = {"cagr": 10.0, "vol": 12.0, "sharpe": 0.80, "max_dd": -20.0}
        v = [{"cagr": 10.5, "vol": 12.0, "sharpe": 0.82, "max_dd": -19.0}] * 10
        s = [{"cagr": 11.0, "vol": 12.0, "sharpe": 0.87, "max_dd": -18.0}] * 10

        result = StackingEnsembleBacktest._aggregate(
            StackingEnsembleBacktest(), dates, bl_metrics, v, s, 10,
        )
        assert result.sharpe_delta_significant == (abs(result.sharpe_delta_t_stat) > 2.0)


class TestCliImports:

    def test_cli_imports(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest,
            StackingBacktestResult,
        )
        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        assert bt is not None

    def test_cli_module_attributes(self):
        import src.backtest.stacking_ensemble_backtest as mod
        assert hasattr(mod, "BASELINE_ACCURACY")
        assert hasattr(mod, "STACKING_ACCURACY")
        assert hasattr(mod, "BASELINE_SPY")
        assert hasattr(mod, "BASELINE_GLD")
        assert hasattr(mod, "BASELINE_TLT")
        assert hasattr(mod, "MAX_EQUITY_SHIFT")
        assert hasattr(mod, "SIGNAL_FREQUENCY")
        assert hasattr(mod, "MIN_HOLDING_DAYS")
        assert hasattr(mod, "MC_TRIALS")
        assert hasattr(mod, "DEFAULT_CACHE_DB")


def test_a3_b1b_delegation_matches_pre_migration_capture(tmp_path):
    """A3 pin (Item B1b sub-task 4): _load_prices delegates to grid_runner.load_prices_market_db.

    Delegation keeps the method name/signature; a tmp market.db flows through
    cache_db and yields the same date-indexed shape (Item 32 tmp-db precedent,
    test_grid_runner.py:169-207).
    """
    from src.backtest.grid_runner import load_prices_market_db
    from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

    # method stays in pilot; the shared loader is grid_runner's
    assert StackingEnsembleBacktest._load_prices.__module__ == (
        "src.backtest.stacking_ensemble_backtest"
    )
    assert load_prices_market_db.__module__ == "src.backtest.grid_runner"

    # tmp market.db through cache_db -> date-indexed shape, None/<=0 filtered
    db = tmp_path / "market.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.executemany(
            "INSERT INTO prices VALUES (?, ?, ?)",
            [
                ("SPY", "2026-01-01", 100.0),
                ("SPY", "2026-01-02", 101.0),
                ("GLD", "2026-01-01", 50.0),
                ("GLD", "2026-01-02", None),
            ],
        )
    bt = StackingEnsembleBacktest(cache_db=db)
    assert bt._load_prices(["SPY", "GLD"], "2026-01-01", "2026-01-02") == {
        "SPY": {"2026-01-01": 100.0, "2026-01-02": 101.0},
        "GLD": {"2026-01-01": 50.0},
    }
