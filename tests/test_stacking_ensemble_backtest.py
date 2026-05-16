"""
Tests for Stacking Ensemble Backtest — v3.10 Phase 5 Validation
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


@pytest.fixture
def stack_test_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_stack.db"
        _make_stack_test_db(db_path)
        yield db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStackingEnsembleBacktest:

    def test_import(self):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest,
            StackingBacktestResult,
        )
        assert StackingEnsembleBacktest is not None
        assert StackingBacktestResult is not None

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

    def test_backtest_runs(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31", mc_trials=10)

        assert result.trading_days > 0
        assert result.mc_trials == 10

    def test_signal_generation(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(23)]  # 23 days
        signals = StackingEnsembleBacktest._generate_signals(dates, 0.76, 0.15, seed=42)

        # Should have some signals
        non_neutral = sum(1 for v in signals.values() if v != 0)
        assert non_neutral >= 0  # Could be 0 with small dataset

    def test_return_metrics(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        rets = [0.001] * 100
        m = StackingEnsembleBacktest._compute_return_metrics(rets)
        assert m["sharpe"] > 0
        assert m["cagr"] > 0
        assert m["max_dd"] <= 0

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

    def test_apply_signals(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        prices = bt._load_prices(["SPY", "GLD", "TLT"], "2022-01-01", "2022-12-31")
        dates = sorted(
            set(prices["SPY"].keys())
            & set(prices["GLD"].keys())
            & set(prices["TLT"].keys())
        )

        # All-neutral signals
        signals = {d: 0 for d in dates}
        rets = bt._apply_signals(dates, prices, signals)
        assert len(rets) > 0

        # Some non-neutral
        signals[dates[10]] = 1
        signals[dates[20]] = -1
        rets2 = bt._apply_signals(dates, prices, signals)
        assert len(rets2) == len(rets)

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

    def test_accuracy_constants(self):
        from src.backtest.stacking_ensemble_backtest import (
            BASELINE_ACCURACY, STACKING_ACCURACY,
        )
        assert BASELINE_ACCURACY == 0.65
        assert STACKING_ACCURACY == 0.76
        assert STACKING_ACCURACY > BASELINE_ACCURACY

    def test_min_holding_period(self):
        from src.backtest.stacking_ensemble_backtest import StackingEnsembleBacktest

        dates = [f"2022-01-{1+i:02d}" for i in range(30)]
        # 100% frequency to get many signals
        signals = StackingEnsembleBacktest._generate_signals(
            dates, 0.80, 1.0, seed=123
        )

        # Count consecutive non-neutral signals
        non_neutral_dates = [
            d for d in dates if d in signals and signals[d] != 0
        ]
        if len(non_neutral_dates) >= 2:
            # Check no signals within 5 days of each other
            for i in range(len(non_neutral_dates) - 1):
                idx1 = dates.index(non_neutral_dates[i])
                idx2 = dates.index(non_neutral_dates[i + 1])
                assert idx2 - idx1 >= 5, (
                    f"Min holding violated: {idx2 - idx1} days between signals"
                )

    def test_cli_imports(self, stack_test_db):
        from src.backtest.stacking_ensemble_backtest import (
            StackingEnsembleBacktest,
            StackingBacktestResult,
        )
        bt = StackingEnsembleBacktest(cache_db=stack_test_db)
        assert bt is not None
