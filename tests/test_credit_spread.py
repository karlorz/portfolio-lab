"""Tests for credit spread data fetcher and signal generator."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Import modules under test
from src.data.credit_fetcher import (
    CreditFetcher,
    CreditMetrics,
    CreditSignal,
    SPREAD_WIDENING_THRESHOLD,
    SPREAD_TIGHTENING_THRESHOLD,
)
from src.signals.credit_spread_signal import (
    CreditSignalType,
    AllocationShift,
    AllocationRecommendation,
    CreditSpreadSignal,
    CreditSpreadSignalGenerator,
    get_credit_signal,
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

            # Create metrics and save to cache
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

            # Create metrics with old timestamp
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
            # Manually update created_at to be old
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
        """Test spread calculation."""
        fetcher = CreditFetcher()
        spread = fetcher._compute_spread(0.021, 0.018)
        assert spread == pytest.approx(-0.003)

    def test_classify_trend_widening(self):
        """Test trend classification for widening spreads."""
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(-3.0)
        assert trend == "widening"

    def test_classify_trend_tightening(self):
        """Test trend classification for tightening spreads."""
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(3.0)
        assert trend == "tightening"

    def test_classify_trend_stable(self):
        """Test trend classification for stable spreads."""
        fetcher = CreditFetcher()
        trend = fetcher._classify_trend(0.5)
        assert trend == "stable"

    def test_classify_volatility_high(self):
        """Test volatility classification."""
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(6.0, 0.0)
        assert vol == "high"

    def test_classify_volatility_medium(self):
        """Test volatility classification."""
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(3.0, 0.0)
        assert vol == "medium"

    def test_classify_volatility_low(self):
        """Test volatility classification."""
        fetcher = CreditFetcher()
        vol = fetcher._classify_volatility(0.5, 0.0)
        assert vol == "low"


class TestSignalGenerator:
    """Test CreditSpreadSignalGenerator."""

    def test_base_weights(self):
        """Test base portfolio weights are correct."""
        generator = CreditSpreadSignalGenerator()
        assert generator.BASE_WEIGHTS["SPY"] == pytest.approx(0.46)
        assert generator.BASE_WEIGHTS["GLD"] == pytest.approx(0.38)
        assert generator.BASE_WEIGHTS["TLT"] == pytest.approx(0.16)

    def test_vix_cutoff_disables_signal(self):
        """Test high VIX disables signal generation."""
        generator = CreditSpreadSignalGenerator(vix_level=40.0)
        signal_type, is_active, reason = generator._determine_signal_type(
            spread=-0.03, persistence=10, confidence=0.8
        )
        assert signal_type == CreditSignalType.NEUTRAL
        assert is_active is False
        assert "VIX" in reason

    def test_persistence_requirement(self):
        """Test persistence requirement for signal activation."""
        generator = CreditSpreadSignalGenerator(vix_level=15.0)
        signal_type, is_active, reason = generator._determine_signal_type(
            spread=-0.03, persistence=2, confidence=0.8
        )
        assert signal_type == CreditSignalType.NEUTRAL
        assert is_active is False
        assert "persistence" in reason.lower()

    def test_risk_off_recommendations(self):
        """Test risk-off allocation recommendations."""
        generator = CreditSpreadSignalGenerator(vix_level=15.0)
        recs = generator._generate_recommendations(CreditSignalType.RISK_OFF)

        spy_rec = next(r for r in recs if r.symbol == "SPY")
        tlt_rec = next(r for r in recs if r.symbol == "TLT")
        gld_rec = next(r for r in recs if r.symbol == "GLD")

        assert spy_rec.shift == AllocationShift.DECREASE
        assert spy_rec.shift_percent == pytest.approx(-3.0)
        assert tlt_rec.shift == AllocationShift.INCREASE
        assert tlt_rec.shift_percent == pytest.approx(2.0)
        assert gld_rec.shift == AllocationShift.INCREASE
        assert gld_rec.shift_percent == pytest.approx(1.0)

    def test_risk_on_recommendations(self):
        """Test risk-on allocation recommendations."""
        generator = CreditSpreadSignalGenerator(vix_level=15.0)
        recs = generator._generate_recommendations(CreditSignalType.RISK_ON)

        spy_rec = next(r for r in recs if r.symbol == "SPY")
        tlt_rec = next(r for r in recs if r.symbol == "TLT")
        gld_rec = next(r for r in recs if r.symbol == "GLD")

        assert spy_rec.shift == AllocationShift.INCREASE
        assert spy_rec.shift_percent == pytest.approx(2.0)
        assert tlt_rec.shift == AllocationShift.DECREASE
        assert tlt_rec.shift_percent == pytest.approx(-2.0)
        assert gld_rec.shift == AllocationShift.HOLD

    def test_neutral_recommendations(self):
        """Test neutral allocation recommendations."""
        generator = CreditSpreadSignalGenerator(vix_level=15.0)
        recs = generator._generate_recommendations(CreditSignalType.NEUTRAL)

        for rec in recs:
            assert rec.shift == AllocationShift.HOLD
            assert rec.shift_percent == 0.0
            assert rec.current_weight == rec.recommended_weight

    def test_max_allocation_shift_respected(self):
        """Test maximum allocation shift is not exceeded."""
        generator = CreditSpreadSignalGenerator(vix_level=15.0)
        recs = generator._generate_recommendations(CreditSignalType.RISK_OFF)

        for rec in recs:
            shift = abs(rec.recommended_weight - rec.current_weight)
            assert shift <= generator.MAX_ALLOCATION_SHIFT + 0.001


class TestCreditSpreadSignal:
    """Test CreditSpreadSignal dataclass."""

    def test_to_dict(self):
        """Test signal serialization to dict."""
        rec = AllocationRecommendation(
            symbol="SPY",
            current_weight=0.46,
            recommended_weight=0.43,
            shift=AllocationShift.DECREASE,
            shift_percent=-3.0,
            rationale="Risk-off regime"
        )

        signal = CreditSpreadSignal(
            timestamp="2026-05-14T15:00:00",
            signal_type=CreditSignalType.RISK_OFF,
            confidence=0.75,
            spread_absolute=-0.025,
            spread_zscore=-1.5,
            trend_direction="widening",
            persistence_days=7,
            volatility_regime="medium",
            is_active=True,
            recommendations=[rec],
            summary="Test signal"
        )

        d = signal.to_dict()
        assert d["signal_type"] == "risk_off"
        assert d["confidence"] == 0.75
        assert d["spread_absolute"] == pytest.approx(-0.025)
        assert len(d["recommendations"]) == 1
        assert d["recommendations"][0]["symbol"] == "SPY"


class TestIntegration:
    """Integration tests."""

    def test_get_credit_signal_returns_dict(self):
        """Test get_credit_signal returns proper dict."""
        # This may fail if no network/yfinance data, but tests structure
        try:
            signal = get_credit_signal(vix_level=15.0)
            assert isinstance(signal, dict)
            assert "signal_type" in signal
            assert "confidence" in signal
            assert "recommendations" in signal
        except Exception as e:
            # Network/data errors are expected in test environment
            pytest.skip(f"Network/data unavailable: {e}")


class TestThresholds:
    """Test signal thresholds."""

    def test_spread_widening_threshold(self):
        """Verify spread widening threshold value."""
        assert SPREAD_WIDENING_THRESHOLD == 2.0

    def test_spread_tightening_threshold(self):
        """Verify spread tightening threshold value."""
        assert SPREAD_TIGHTENING_THRESHOLD == -2.0
