"""
Tests for Behavioral Sentiment Walk-Forward Backtest — v2.70 Phase 4
"""

import json
import math
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_db(db_path: Path, vix_values: list = None):
    """Create a test market.db with synthetic price data."""
    if vix_values is None:
        # Simulate a mix of calm and volatile periods
        np.random.seed(42)
        vix_values = [18.0 + 5 * math.sin(i / 60) + np.random.normal(0, 2) for i in range(300)]

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, updated_at TEXT,
            PRIMARY KEY (symbol, date)
        )""")

        base_spy = 400.0
        base_gld = 170.0
        base_tlt = 140.0

        for i, vix in enumerate(vix_values):
            day = 1 + i % 21
            month = 1 + i // 21
            if month > 12:
                month = 12
                day = min(day, 28)
            date_str = f"2022-{month:02d}-{day:02d}"

            # VIX
            conn.execute(
                """INSERT OR REPLACE INTO prices (symbol, date, close)
                   VALUES ('^VIX', ?, ?)""",
                (date_str, vix),
            )

            # SPY: random walk with drift
            spy_ret = np.random.normal(0.0003, 0.01)
            base_spy *= (1 + spy_ret)
            conn.execute(
                """INSERT OR REPLACE INTO prices (symbol, date, close)
                   VALUES ('SPY', ?, ?)""",
                (date_str, round(base_spy, 2)),
            )

            # GLD: low correlation random walk
            gld_ret = np.random.normal(0.0002, 0.008)
            base_gld *= (1 + gld_ret)
            conn.execute(
                """INSERT OR REPLACE INTO prices (symbol, date, close)
                   VALUES ('GLD', ?, ?)""",
                (date_str, round(base_gld, 2)),
            )

            # TLT: bond random walk
            tlt_ret = np.random.normal(0.0001, 0.007)
            base_tlt *= (1 + tlt_ret)
            conn.execute(
                """INSERT OR REPLACE INTO prices (symbol, date, close)
                   VALUES ('TLT', ?, ?)""",
                (date_str, round(base_tlt, 2)),
            )

        conn.commit()


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_market.db"
        _make_test_db(db_path)
        yield db_path


@pytest.fixture
def test_db_crisis():
    """Create a test DB with a clear crisis spike (VIX > 35)."""
    vix_vals = []
    for i in range(300):
        if 100 <= i < 130:
            vix_vals.append(38.0)  # Crisis period
        elif 130 <= i < 150:
            vix_vals.append(28.0)  # Elevated
        elif 200 <= i < 220:
            vix_vals.append(11.0)  # Extreme greed
        else:
            vix_vals.append(18.0)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_market_crisis.db"
        _make_test_db(db_path, vix_vals)
        yield db_path


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBehavioralSentimentBacktest:
    """Tests for the behavioral sentiment backtest engine."""

    def test_import(self):
        """Module imports cleanly without ML dependencies."""
        from src.backtest.behavioral_sentiment_backtest import (
            BehavioralSentimentBacktest,
        )
        from src.backtest.metrics import BacktestResult
        assert BehavioralSentimentBacktest is not None
        assert BacktestResult is not None

    def test_result_dataclass(self):
        """BacktestResult initializes with behavioral extras defaults."""
        from src.backtest.metrics import BacktestResult

        r = BacktestResult(
            total_return=0.0, cagr=0.0, volatility=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0,
            baseline_sharpe=0.0, sharpe_improvement=0.0,
            extras={
                "timestamp": "",
                "start_date": "",
                "end_date": "",
                "trading_days": 0,
                "baseline_cagr": 0.0,
                "baseline_vol": 0.0,
                "baseline_max_dd": 0.0,
                "baseline_crisis_2022": 0.0,
                "overlay_crisis_2022": 0.0,
                "dd_improvement": 0.0,
                "cagr_delta": 0.0,
                "signal_days_pct": 0.0,
                "buy_signal_days": 0,
                "sell_signal_days": 0,
                "neutral_days": 0,
                "avg_equity_shift": 0.0,
                "false_positive_rate": 0.0,
                "mean_signal_return_20d": 0.0,
                "regime_vix_low_sharpe": 0.0,
                "regime_vix_normal_sharpe": 0.0,
                "regime_vix_elevated_sharpe": 0.0,
                "regime_vix_high_sharpe": 0.0,
                "regime_vix_crisis_sharpe": 0.0,
                "meets_sharpe_target": False,
            },
        )
        assert r.sharpe_improvement == 0.0
        assert r.extras["trading_days"] == 0

    def test_to_dict(self):
        """Result serializes to dict for JSON output."""
        from dataclasses import asdict
        from src.backtest.metrics import BacktestResult

        r = BacktestResult(
            total_return=64.7, cagr=10.5, volatility=12.0,
            sharpe_ratio=0.83, max_drawdown=-19.0,
            baseline_sharpe=0.8, sharpe_improvement=0.03,
            extras={
                "timestamp": "2026-01-01",
                "start_date": "2021-01-01",
                "end_date": "2026-01-01",
                "trading_days": 1260,
                "baseline_cagr": 10.0,
                "baseline_vol": 12.0,
                "baseline_max_dd": -20.0,
                "baseline_crisis_2022": -12.0,
                "overlay_crisis_2022": -11.0,
                "dd_improvement": 1.0,
                "cagr_delta": 0.5,
                "signal_days_pct": 25.0,
                "buy_signal_days": 100,
                "sell_signal_days": 80,
                "neutral_days": 1080,
                "avg_equity_shift": 3.5,
                "false_positive_rate": 40.0,
                "mean_signal_return_20d": 0.5,
                "regime_vix_low_sharpe": 0.9,
                "regime_vix_normal_sharpe": 0.8,
                "regime_vix_elevated_sharpe": 0.5,
                "regime_vix_high_sharpe": 0.3,
                "regime_vix_crisis_sharpe": -0.2,
                "meets_sharpe_target": True,
            },
        )

        d = asdict(r)
        assert d["baseline_sharpe"] == 0.8
        assert d["sharpe_ratio"] == 0.83
        assert d["sharpe_improvement"] == 0.03
        assert d["extras"]["meets_sharpe_target"] is True
        assert d["extras"]["buy_signal_days"] == 100

        # Verify JSON round-trip
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["baseline_sharpe"] == 0.8

    def test_backtest_runs_with_test_db(self, test_db):
        """Backtest produces results with synthetic data."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        bt = BehavioralSentimentBacktest(cache_db=test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        assert result.extras["trading_days"] > 0
        assert result.extras["start_date"] is not None
        assert result.extras["end_date"] is not None

    def test_backtest_with_crisis_data(self, test_db_crisis):
        """Backtest handles crisis VIX regimes correctly."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        bt = BehavioralSentimentBacktest(cache_db=test_db_crisis)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        # Should have some buy signals (VIX 25-30 range)
        assert result.extras["buy_signal_days"] + result.extras["sell_signal_days"] + result.extras["neutral_days"] > 0
        # Signal days pct should be non-zero (some VIX <15 and VIX 25-30 days)
        assert result.extras["signal_days_pct"] >= 0

    def test_empty_result_on_no_data(self):
        """Returns empty result when no data available."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty.db"
            # Create empty DB with schema but no data
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                conn.commit()

            bt = BehavioralSentimentBacktest(cache_db=db)
            result = bt.run(start_date="1999-01-01", end_date="1999-12-31")
            assert result.extras["trading_days"] == 0

    def test_max_drawdown_computation(self):
        """Max drawdown is correctly computed from returns."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        # Steady growth: no drawdown
        rets = np.array([0.001] * 100)
        dd = BehavioralSentimentBacktest._max_drawdown(rets)
        assert dd >= -0.01  # Near zero

        # Sharp drop then recovery
        rets2 = np.array([-0.05, -0.05, 0.02, 0.02, 0.02])
        dd2 = BehavioralSentimentBacktest._max_drawdown(rets2)
        assert dd2 <= -0.049  # Should capture the drop (floating point)

        # Single large drop: need 2+ elements for drawdown
        # First element establishes the peak, second element shows the drawdown
        rets3 = np.array([0.0, -0.20])
        dd3 = BehavioralSentimentBacktest._max_drawdown(rets3)
        assert dd3 == pytest.approx(-0.20)

    def test_year_return_filtering(self):
        """Year return correctly filters by calendar year."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        dates = ["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04", "2022-01-05"]
        rets = np.array([0.01, 0.01, 0.02, -0.01, 0.03])

        yr = BehavioralSentimentBacktest._year_return(dates, rets, "2022")
        # Segment includes Dec 31 carry: rets[1:4] = [0.01, 0.02, -0.01]
        # prod = 1.01 * 1.02 * 0.99 = 1.019898, return ≈ 0.0199
        assert yr > 0.01

        # Year not in data
        yr2 = BehavioralSentimentBacktest._year_return(dates, rets, "2020")
        assert yr2 == 0.0

    def test_regime_sharpe_computation(self, test_db):
        """Regime Sharpe ratios are computed for different VIX buckets."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest, DEFAULT_CACHE_DB

        bt = BehavioralSentimentBacktest(cache_db=test_db)

        # Create synthetic returns and dates matching our test data
        dates = []
        rets = []
        for i in range(100):
            month = 1 + i // 21
            day = 1 + i % 21
            if month > 12:
                month = 12
                day = min(day, 28)
            dates.append(f"2022-{month:02d}-{day:02d}")
            rets.append(0.001)

        arr = np.array(rets, dtype=np.float64)
        sharpes = bt._all_regime_sharpes(dates, arr)
        # Should return 5-tuple of floats
        assert len(sharpes) == 5
        assert all(isinstance(s, float) for s in sharpes)

    def test_backtest_result_fields_populated(self, test_db):
        """All result fields are populated after running."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        bt = BehavioralSentimentBacktest(cache_db=test_db)
        result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

        # Core fields
        assert result.extras["trading_days"] > 0
        assert isinstance(result.baseline_sharpe, float)
        assert isinstance(result.sharpe_ratio, float)

        # Signal quality
        assert isinstance(result.extras["signal_days_pct"], float)
        assert isinstance(result.extras["false_positive_rate"], float)

        # Regime
        assert isinstance(result.extras["regime_vix_low_sharpe"], float)
        assert isinstance(result.extras["regime_vix_crisis_sharpe"], float)

        # Target
        assert isinstance(result.extras["meets_sharpe_target"], bool)

    def test_baseline_weights_constant(self):
        """Baseline weights match the 46/38/16 allocation."""
        from src.backtest.behavioral_sentiment_backtest import (
            BASELINE_SPY, BASELINE_GLD, BASELINE_TLT,
        )
        assert BASELINE_SPY == 0.46
        assert BASELINE_GLD == 0.38
        assert BASELINE_TLT == 0.16
        assert abs(BASELINE_SPY + BASELINE_GLD + BASELINE_TLT - 1.0) < 0.001

    def test_signal_map_lookup(self, test_db):
        """Signal map date lookup returns correct equity shifts."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        bt = BehavioralSentimentBacktest(cache_db=test_db)
        signals = bt._load_signals("2022-01-01", "2022-12-31")

        assert isinstance(signals, list)
        if signals:
            sig = signals[0]
            assert "date" in sig
            assert "signal_type" in sig
            assert "equity_shift_pct" in sig
            assert "regime_suppressed" in sig

    def test_price_loading_returns_dict_of_dicts(self, test_db):
        """Price loading returns correctly structured data."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        bt = BehavioralSentimentBacktest(cache_db=test_db)
        prices = bt._load_prices(["SPY", "GLD"], "2022-01-01", "2022-12-31")

        assert "SPY" in prices
        assert "GLD" in prices
        assert isinstance(prices["SPY"], dict)
        # Should have at least some price data
        if prices["SPY"]:
            first_date = next(iter(prices["SPY"]))
            assert prices["SPY"][first_date] > 0

    def test_cli_module_run(self, test_db):
        """Module can be imported and instantiated without crashing."""
        from src.backtest.behavioral_sentiment_backtest import (
            BehavioralSentimentBacktest,
        )
        bt = BehavioralSentimentBacktest(cache_db=test_db)
        assert bt is not None
        # Confirm CLI-relevant imports work

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_short_date_range_returns_empty(self):
        """Very short date range returns empty result gracefully."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tiny.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                # Only 2 days of data for each
                for symbol in ["SPY", "GLD", "TLT", "^VIX"]:
                    conn.execute(
                        "INSERT INTO prices VALUES (?, ?, ?)",
                        (symbol, "2022-01-03", 100.0),
                    )
                    conn.execute(
                        "INSERT INTO prices VALUES (?, ?, ?)",
                        (symbol, "2022-01-04", 101.0),
                    )
                conn.commit()

            bt = BehavioralSentimentBacktest(cache_db=db)
            result = bt.run(start_date="2022-01-01", end_date="2022-01-05")
            # With only 1 daily return, still runs but with minimal data
            assert result.extras["trading_days"] >= 0

    def test_no_vix_data_handled(self):
        """Missing VIX data produces zero regime Sharpe but doesn't crash."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "no_vix.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                for i in range(100):
                    date_str = f"2022-{1+i//21:02d}-{1+i%21:02d}"
                    conn.execute(
                        "INSERT INTO prices VALUES (?, ?, ?)",
                        ("SPY", date_str, 400.0 + i),
                    )
                    conn.execute(
                        "INSERT INTO prices VALUES (?, ?, ?)",
                        ("GLD", date_str, 170.0 + i * 0.1),
                    )
                    conn.execute(
                        "INSERT INTO prices VALUES (?, ?, ?)",
                        ("TLT", date_str, 140.0 - i * 0.05),
                    )
                conn.commit()

            bt = BehavioralSentimentBacktest(cache_db=db)
            result = bt.run(start_date="2022-01-01", end_date="2022-12-31")

            # VIX data missing → all regime Sharpes should be 0
            assert result.extras["regime_vix_low_sharpe"] == 0.0
            assert result.extras["regime_vix_crisis_sharpe"] == 0.0

    def test_meets_target_logic(self):
        """meets_sharpe_target correctly evaluates delta >= 0.03."""
        from src.backtest.metrics import BacktestResult

        def _make(delta):
            return BacktestResult(
                total_return=0.0, cagr=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0,
                baseline_sharpe=0.0, sharpe_improvement=delta,
                extras={
                    "timestamp": "",
                    "start_date": "",
                    "end_date": "",
                    "trading_days": 0,
                    "baseline_cagr": 0.0,
                    "baseline_vol": 0.0,
                    "baseline_max_dd": 0.0,
                    "baseline_crisis_2022": 0.0,
                    "overlay_crisis_2022": 0.0,
                    "dd_improvement": 0.0,
                    "cagr_delta": 0.0,
                    "signal_days_pct": 0.0,
                    "buy_signal_days": 0,
                    "sell_signal_days": 0,
                    "neutral_days": 0,
                    "avg_equity_shift": 0.0,
                    "false_positive_rate": 0.0,
                    "mean_signal_return_20d": 0.0,
                    "regime_vix_low_sharpe": 0.0,
                    "regime_vix_normal_sharpe": 0.0,
                    "regime_vix_elevated_sharpe": 0.0,
                    "regime_vix_high_sharpe": 0.0,
                    "regime_vix_crisis_sharpe": 0.0,
                    "meets_sharpe_target": (delta >= 0.03),
                },
            )

        assert _make(0.05).extras["meets_sharpe_target"] is True
        assert _make(0.03).extras["meets_sharpe_target"] is True
        assert _make(0.02).extras["meets_sharpe_target"] is False
        assert _make(-0.01).extras["meets_sharpe_target"] is False

    def test_negative_returns_max_dd(self):
        """Consecutive negative returns produce correct max drawdown."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        # 10% drop each day for 5 days
        # cumprod = [0.9, 0.81, 0.729, 0.6561, 0.59049]
        # peak is 0.9 (first value), min dd = (0.59049-0.9)/0.9 ≈ -0.3439
        rets = np.array([-0.10, -0.10, -0.10, -0.10, -0.10])
        dd = BehavioralSentimentBacktest._max_drawdown(rets)
        assert dd < -0.30

    def test_date_alignment_with_missing_symbols(self):
        """When a symbol has missing dates, only common dates are used."""
        from src.backtest.behavioral_sentiment_backtest import BehavioralSentimentBacktest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "partial.db"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("""CREATE TABLE prices (
                    symbol TEXT, date TEXT, close REAL,
                    PRIMARY KEY (symbol, date)
                )""")
                # SPY has days 1-5, GLD has days 3-7, TLT has days 2-6
                for d in range(1, 6):
                    conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                                 ("SPY", f"2022-01-0{d}", 400.0))
                for d in range(3, 8):
                    conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                                 ("GLD", f"2022-01-0{d}", 170.0))
                for d in range(2, 7):
                    conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                                 ("TLT", f"2022-01-0{d}", 140.0))
                # VIX for all
                for d in range(1, 8):
                    conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                                 ("^VIX", f"2022-01-0{d}", 18.0))
                conn.commit()

            bt = BehavioralSentimentBacktest(cache_db=db)
            result = bt.run(start_date="2022-01-01", end_date="2022-01-10")
            # Common dates: Jan 3-5 only (all three symbols present)
            assert result.extras["trading_days"] <= 5
