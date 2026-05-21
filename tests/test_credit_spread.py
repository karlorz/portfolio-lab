"""Tests for credit spread data fetcher."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.credit_fetcher import (
    CreditFetcher,
    CreditMetrics,
    CreditSignal,
    SPREAD_WIDENING_THRESHOLD,
    SPREAD_TIGHTENING_THRESHOLD,
)


class TestCreditMetrics:
    """Test CreditMetrics dataclass."""

    def test_creation(self):
        """Test CreditMetrics creation."""
        m = CreditMetrics(
            timestamp="2026-05-14T15:00:00",
            lqd_price=120.5,
            hyg_price=85.0,
            agg_price=108.0,
            lqd_return_30d=0.021,
            hyg_return_30d=0.018,
            agg_return_30d=0.015,
            spread_absolute=-0.003,
            spread_zscore=-0.5,
            trend_direction="stable",
            volatility_regime="low",
        )
        assert m.lqd_price == 120.5
        assert m.hyg_price == 85.0
        assert m.spread_absolute == pytest.approx(-0.003)

    def test_immutability(self):
        """Test CreditMetrics is frozen/immutable."""
        m = CreditMetrics(
            timestamp="2026-05-14T15:00:00",
            lqd_price=120.5,
            hyg_price=85.0,
            agg_price=108.0,
            lqd_return_30d=0.021,
            hyg_return_30d=0.018,
            agg_return_30d=0.015,
            spread_absolute=-0.003,
            spread_zscore=-0.5,
            trend_direction="stable",
            volatility_regime="low",
        )
        with pytest.raises(AttributeError):
            m.lqd_price = 121.0


class TestCreditSignal:
    """Test CreditSignal dataclass."""

    def test_creation(self):
        """Test CreditSignal creation."""
        s = CreditSignal(
            timestamp="2026-05-14T15:00:00",
            spread_absolute=-0.003,
            spread_zscore=-0.5,
            trend_direction="stable",
            signal="neutral",
            confidence=0.0,
            equity_shift_pct=0.0,
            rationale="Test signal"
        )
        assert s.spread_absolute == pytest.approx(-0.003)
        assert s.signal == "neutral"

    def test_to_dict(self):
        """Test signal serialization to dict."""
        s = CreditSignal(
            timestamp="2026-05-14T15:00:00",
            spread_absolute=-0.003,
            spread_zscore=-0.5,
            trend_direction="stable",
            signal="neutral",
            confidence=0.5,
            equity_shift_pct=0.0,
            rationale="Test"
        )
        d = s.to_dict()
        assert d["signal"] == "neutral"
        assert d["confidence"] == 0.5


class TestCreditFetcherCache:
    """Test CreditFetcher database operations."""

    def test_init_creates_tables(self):
        """Test database initialization creates tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            fetcher = CreditFetcher(cache_db=db_path)

            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in cursor.fetchall()}
                assert "credit_cache" in tables

    def test_is_fresh_true(self):
        """Test is_fresh returns True for recent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            fetcher = CreditFetcher(cache_db=db_path)

            metrics = CreditMetrics(
                timestamp=datetime.now().isoformat(),
                lqd_price=120.5,
                hyg_price=85.0,
                agg_price=108.0,
                lqd_return_30d=0.021,
                hyg_return_30d=0.018,
                agg_return_30d=0.015,
                spread_absolute=-0.003,
                spread_zscore=-0.5,
                trend_direction="stable",
                volatility_regime="low",
            )

            fetcher._save_cache(metrics)
            assert fetcher._get_cached() is not None

    def test_is_fresh_false(self):
        """Test is_fresh returns False for old data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            fetcher = CreditFetcher(cache_db=db_path)

            old_time = (datetime.now() - timedelta(hours=5)).isoformat()
            metrics = CreditMetrics(
                timestamp=old_time,
                lqd_price=120.5,
                hyg_price=85.0,
                agg_price=108.0,
                lqd_return_30d=0.021,
                hyg_return_30d=0.018,
                agg_return_30d=0.015,
                spread_absolute=-0.003,
                spread_zscore=-0.5,
                trend_direction="stable",
                volatility_regime="low",
            )

            fetcher._save_cache(metrics)
            with sqlite3.connect(db_path) as conn:
                old_cached_at = (datetime.now() - timedelta(hours=5)).isoformat()
                conn.execute(
                    "UPDATE credit_cache SET created_at = ? WHERE id = 1",
                    (old_cached_at,)
                )
                conn.commit()

            assert fetcher._get_cached() is None


class TestCreditFetcherSpreadCalc:
    """Test CreditFetcher spread calculations."""

    def test_compute_spread(self):
        fetcher = CreditFetcher()
        spread = fetcher._compute_spread(0.021, 0.018)
        assert spread == pytest.approx(-0.003)

    def test_classify_trend_widening(self):
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(-3.0)
        assert trend == "widening"

    def test_classify_trend_tightening(self):
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(3.0)
        assert trend == "tightening"

    def test_classify_trend_stable(self):
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(0.5)
        assert trend == "stable"

    def test_classify_volatility_high(self):
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(6.0, 0.0)
        assert vol == "high"

    def test_classify_volatility_medium(self):
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(3.0, 0.0)
        assert vol == "medium"

    def test_classify_volatility_low(self):
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(0.5, 0.0)
        assert vol == "low"


class TestThresholds:
    """Test signal thresholds."""

    def test_spread_widening_threshold(self):
        assert SPREAD_WIDENING_THRESHOLD == 2.0

    def test_spread_tightening_threshold(self):
        assert SPREAD_TIGHTENING_THRESHOLD == -2.0
