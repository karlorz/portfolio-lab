"""Tests for Credit Spread Monitor (v3.14) — CreditFetcher, CreditMetrics, CreditSignal."""

import sys
import os
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.credit_fetcher import (
    CreditMetrics, CreditSignal, CreditFetcher,
    SPREAD_WIDENING_THRESHOLD, SPREAD_TIGHTENING_THRESHOLD,
    ZSCORE_ALERT_THRESHOLD,
)


@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "test_credit.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS credit_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, data TEXT, spread_absolute REAL,
            trend_direction TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()
    return db


@pytest.fixture
def fetcher(tmp_db):
    return CreditFetcher(cache_db=tmp_db)


# ── CreditMetrics dataclass ────────────────────────────────────────────

class TestCreditMetrics:
    def test_create_default(self):
        m = CreditMetrics(timestamp="t", lqd_price=108.0, hyg_price=78.0,
            agg_price=98.0, lqd_return_30d=2.0, hyg_return_30d=1.5, agg_return_30d=1.0,
            spread_absolute=-0.5, spread_zscore=-0.25, trend_direction="stable",
            volatility_regime="low")
        assert m.lqd_price == 108.0
        assert m.spread_absolute == -0.5
        assert m.trend_direction == "stable"

    def test_to_dict(self):
        m = CreditMetrics(timestamp="t", lqd_price=108.0, hyg_price=78.0,
            agg_price=98.0, lqd_return_30d=2.1, hyg_return_30d=1.5, agg_return_30d=1.0,
            spread_absolute=-0.6, spread_zscore=-0.3, trend_direction="stable",
            volatility_regime="medium")
        d = m.to_dict()
        assert d["lqd_price"] == 108.0
        assert d["volatility_regime"] == "medium"

    def test_trend_directions(self):
        for td in ["widening", "tightening", "stable"]:
            m = CreditMetrics(timestamp="t", lqd_price=1, hyg_price=1, agg_price=1,
                lqd_return_30d=0, hyg_return_30d=0, agg_return_30d=0,
                spread_absolute=0, spread_zscore=0, trend_direction=td, volatility_regime="low")
            assert m.trend_direction == td

    def test_volatility_regimes(self):
        for vr in ["low", "medium", "high"]:
            m = CreditMetrics(timestamp="t", lqd_price=1, hyg_price=1, agg_price=1,
                lqd_return_30d=0, hyg_return_30d=0, agg_return_30d=0,
                spread_absolute=0, spread_zscore=0, trend_direction="stable", volatility_regime=vr)
            assert m.volatility_regime == vr


# ── CreditSignal dataclass ─────────────────────────────────────────────

class TestCreditSignal:
    def test_create_risk_on(self):
        s = CreditSignal(timestamp="t", spread_absolute=3.0, spread_zscore=2.5,
            trend_direction="tightening", signal="risk_on", confidence=0.8,
            equity_shift_pct=3.0, rationale="Tightening.")
        assert s.signal == "risk_on"
        assert s.equity_shift_pct == 3.0

    def test_create_risk_off(self):
        s = CreditSignal(timestamp="t", spread_absolute=-3.0, spread_zscore=-2.5,
            trend_direction="widening", signal="risk_off", confidence=0.8,
            equity_shift_pct=-3.0, rationale="Widening.")
        assert s.signal == "risk_off"
        assert s.equity_shift_pct == -3.0

    def test_create_neutral(self):
        s = CreditSignal(timestamp="t", spread_absolute=0.5, spread_zscore=0.3,
            trend_direction="stable", signal="neutral", confidence=0.5,
            equity_shift_pct=0.0, rationale="Normal.")
        assert s.signal == "neutral"

    def test_to_dict(self):
        s = CreditSignal(timestamp="t", spread_absolute=2.0, spread_zscore=2.1,
            trend_direction="tightening", signal="risk_on", confidence=0.7,
            equity_shift_pct=3.0, rationale="Test.")
        d = s.to_dict()
        assert d["signal"] == "risk_on"


# ── CreditFetcher init ─────────────────────────────────────────────────

class TestCreditFetcherInit:
    def test_init_creates_cache_table(self, tmp_db):
        CreditFetcher(cache_db=tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='credit_cache'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_stores_symbols(self, fetcher):
        assert "LQD" in fetcher.SYMBOLS
        assert "HYG" in fetcher.SYMBOLS
        assert "AGG" in fetcher.SYMBOLS

    def test_init_default_db_path(self):
        fetcher = CreditFetcher()
        assert fetcher.cache_db is not None


# ── Spread calculation ─────────────────────────────────────────────────

class TestSpreadCalc:
    def test_hyg_outperforms(self, fetcher):
        assert fetcher._compute_spread(1.0, 4.0) == 3.0

    def test_lqd_outperforms(self, fetcher):
        assert fetcher._compute_spread(5.0, 2.0) == -3.0

    def test_equal_returns(self, fetcher):
        assert fetcher._compute_spread(2.0, 2.0) == 0.0

    def test_zscore_no_history(self, fetcher):
        assert fetcher._compute_zscore(2.0) == pytest.approx(1.0, rel=1e-3)

    def test_zscore_with_history(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db))
        now = datetime.now().isoformat()
        for s in [0.0, 0.5, -0.5, 0.2, -0.2] * 10:
            conn.execute(
                "INSERT INTO credit_cache (timestamp, data, spread_absolute, trend_direction, created_at) VALUES (?,?,?,?,?)",
                (now, "{}", s, "stable", now),
            )
        conn.commit(); conn.close()
        fetcher = CreditFetcher(cache_db=tmp_db)
        z = fetcher._compute_zscore(3.0)
        assert z > 2.0


# ── Trend classification ───────────────────────────────────────────────

class TestTrendClassification:
    def test_tightening(self, fetcher): assert fetcher._classify_trend(3.0) == "tightening"
    def test_widening(self, fetcher): assert fetcher._classify_trend(-3.0) == "widening"
    def test_stable(self, fetcher): assert fetcher._classify_trend(0.5) == "stable"
    def test_boundary_tightening(self, fetcher): assert fetcher._classify_trend(2.1) == "tightening"
    def test_boundary_widening(self, fetcher): assert fetcher._classify_trend(-2.1) == "widening"
    def test_boundary_stable(self, fetcher):
        assert fetcher._classify_trend(2.0) == "stable"
        assert fetcher._classify_trend(-2.0) == "stable"


# ── Volatility classification ──────────────────────────────────────────

class TestVolatilityClassification:
    def test_low(self, fetcher): assert fetcher._classify_volatility(3.0, 2.0) == "low"
    def test_medium(self, fetcher): assert fetcher._classify_volatility(5.0, 2.0) == "medium"
    def test_high(self, fetcher): assert fetcher._classify_volatility(10.0, 2.0) == "high"
    def test_boundary(self, fetcher): assert fetcher._classify_volatility(4.0, 2.0) == "low"


# ── Signal generation ──────────────────────────────────────────────────

class TestGetSignal:
    def test_neutral_when_stable(self, fetcher):
        with patch.object(fetcher, '_fetch_price', return_value=100.0), \
             patch.object(fetcher, '_fetch_30d_return', return_value=2.0):
            sig = fetcher.get_signal()
            assert sig.signal == "neutral"
            assert sig.equity_shift_pct == 0.0

    def test_risk_on_strong_signal(self, tmp_db):
        # Insert diverse historical data so z-score computes correctly
        conn = sqlite3.connect(str(tmp_db))
        now = datetime.now().isoformat()
        for s in [0.1, -0.1, 0.2, -0.2, 0.05, -0.05] * 10:
            conn.execute(
                "INSERT INTO credit_cache (timestamp, data, spread_absolute, trend_direction, created_at) VALUES (?,?,?,?,?)",
                (now, "{}", s, "stable", now),
            )
        conn.commit(); conn.close()
        fetcher = CreditFetcher(cache_db=tmp_db)
        with patch.object(fetcher, '_fetch_price', return_value=100.0), \
             patch.object(fetcher, '_fetch_30d_return', side_effect=[1.0, 4.0, 2.0]):
            sig = fetcher.get_signal()
            # spread = 3.0, z > 2 (far from mean ~0), tightening → risk_on, shift=3.0
            assert sig.signal == "risk_on"
            assert sig.equity_shift_pct == 3.0

    def test_risk_off_strong_signal(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db))
        now = datetime.now().isoformat()
        for s in [0.1, -0.1, 0.2, -0.2, 0.05, -0.05] * 10:
            conn.execute(
                "INSERT INTO credit_cache (timestamp, data, spread_absolute, trend_direction, created_at) VALUES (?,?,?,?,?)",
                (now, "{}", s, "stable", now),
            )
        conn.commit(); conn.close()
        fetcher = CreditFetcher(cache_db=tmp_db)
        with patch.object(fetcher, '_fetch_price', return_value=100.0), \
             patch.object(fetcher, '_fetch_30d_return', side_effect=[5.0, 1.0, 3.0]):
            sig = fetcher.get_signal()
            assert sig.signal == "risk_off"
            assert sig.equity_shift_pct == -3.0

    def test_moderate_signal(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db))
        now = datetime.now().isoformat()
        for s in [0.1, -0.1] * 5:
            conn.execute(
                "INSERT INTO credit_cache (timestamp, data, spread_absolute, trend_direction, created_at) VALUES (?,?,?,?,?)",
                (now, "{}", s, "stable", now),
            )
        conn.commit(); conn.close()
        fetcher = CreditFetcher(cache_db=tmp_db)
        with patch.object(fetcher, '_fetch_price', return_value=100.0), \
             patch.object(fetcher, '_fetch_30d_return', side_effect=[2.0, 3.5, 2.5]):
            sig = fetcher.get_signal()
            assert sig.signal in ("risk_on", "risk_off", "neutral")


# ── Cache ──────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_roundtrip(self, fetcher):
        metrics = CreditMetrics(
            timestamp=datetime.now().isoformat(), lqd_price=108.0, hyg_price=78.0,
            agg_price=98.0, lqd_return_30d=2.0, hyg_return_30d=1.5, agg_return_30d=1.0,
            spread_absolute=-0.5, spread_zscore=-0.25, trend_direction="stable",
            volatility_regime="low")
        fetcher._save_cache(metrics)
        cached = fetcher._get_cached()
        assert cached is not None
        assert cached.lqd_price == 108.0

    def test_cache_empty(self, fetcher):
        assert fetcher._get_cached() is None


# ── History ────────────────────────────────────────────────────────────

class TestHistory:
    def test_empty(self, fetcher):
        assert fetcher.get_history(30) == []

    def test_with_data(self, fetcher):
        metrics = CreditMetrics(
            timestamp=datetime.now().isoformat(), lqd_price=108.0, hyg_price=78.0,
            agg_price=98.0, lqd_return_30d=2.0, hyg_return_30d=1.5, agg_return_30d=1.0,
            spread_absolute=-0.5, spread_zscore=-0.25, trend_direction="stable",
            volatility_regime="low")
        fetcher._save_cache(metrics)
        hist = fetcher.get_history(30)
        assert len(hist) == 1
        assert hist[0]["lqd_price"] == 108.0
