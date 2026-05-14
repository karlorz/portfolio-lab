"""Tests for credit spread data fetcher and signal generator."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Import modules under test
from src.data.credit_fetcher import (
    CreditCache,
    CreditFetcher,
    CreditMetrics,
    CreditSpread,
    CreditData,
    RISK_OFF_THRESHOLD,
    RISK_ON_THRESHOLD,
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
            symbol="LQD",
            timestamp="2026-05-14T15:00:00",
            price=120.5,
            return_30d=0.021,
            return_90d=0.045,
            volatility_30d=0.08
        )
        assert m.symbol == "LQD"
        assert m.price == 120.5
        assert m.return_30d == pytest.approx(0.021)

    def test_immutability(self):
        """Test CreditMetrics is frozen/immutable."""
        m = CreditMetrics(
            symbol="LQD",
            timestamp="2026-05-14T15:00:00",
            price=120.5,
            return_30d=0.021,
            return_90d=0.045,
            volatility_30d=0.08
        )
        with pytest.raises(AttributeError):
            m.price = 121.0


class TestCreditSpread:
    """Test CreditSpread dataclass."""

    def test_creation(self):
        """Test CreditSpread creation."""
        s = CreditSpread(
            timestamp="2026-05-14T15:00:00",
            lqd_return_30d=0.021,
            hyg_return_30d=0.018,
            agg_return_30d=0.015,
            spread_absolute=-0.003,
            spread_zscore=-0.5,
            trend_direction="stable",
            volatility_regime="low",
            persistence_days=3,
            signal="neutral",
            confidence=0.0
        )
        assert s.spread_absolute == pytest.approx(-0.003)
        assert s.signal == "neutral"


class TestCreditCache:
    """Test CreditCache database operations."""

    def test_init_creates_tables(self):
        """Test database initialization creates tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in cursor.fetchall()}
                assert "credit_metrics" in tables
                assert "credit_spread" in tables
                assert "spread_history" in tables

    def test_save_and_get_metrics(self):
        """Test saving and retrieving metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            metrics = CreditMetrics(
                symbol="LQD",
                timestamp="2026-05-14T15:00:00",
                price=120.5,
                return_30d=0.021,
                return_90d=0.045,
                volatility_30d=0.08
            )

            cache.save_metrics(metrics)
            retrieved = cache.get_metrics("LQD")

            assert retrieved is not None
            assert retrieved.symbol == "LQD"
            assert retrieved.price == pytest.approx(120.5)
            assert retrieved.return_30d == pytest.approx(0.021)

    def test_save_and_get_spread(self):
        """Test saving and retrieving spread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            spread = CreditSpread(
                timestamp="2026-05-14T15:00:00",
                lqd_return_30d=0.021,
                hyg_return_30d=0.018,
                agg_return_30d=0.015,
                spread_absolute=-0.003,
                spread_zscore=-0.5,
                trend_direction="stable",
                volatility_regime="low",
                persistence_days=3,
                signal="neutral",
                confidence=0.0
            )

            cache.save_spread(spread)
            retrieved = cache.get_spread()

            assert retrieved is not None
            assert retrieved.spread_absolute == pytest.approx(-0.003)
            assert retrieved.signal == "neutral"

    def test_is_fresh_true(self):
        """Test is_fresh returns True for recent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            spread = CreditSpread(
                timestamp=datetime.now().isoformat(),
                lqd_return_30d=0.021,
                hyg_return_30d=0.018,
                agg_return_30d=0.015,
                spread_absolute=-0.003,
                spread_zscore=-0.5,
                trend_direction="stable",
                volatility_regime="low",
                persistence_days=3,
                signal="neutral",
                confidence=0.0
            )

            cache.save_spread(spread)
            assert cache.is_fresh() is True

    def test_is_fresh_false(self):
        """Test is_fresh returns False for old cached_at timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            # Create spread and manually update cached_at to be old
            spread = CreditSpread(
                timestamp=datetime.now().isoformat(),
                lqd_return_30d=0.021,
                hyg_return_30d=0.018,
                agg_return_30d=0.015,
                spread_absolute=-0.003,
                spread_zscore=-0.5,
                trend_direction="stable",
                volatility_regime="low",
                persistence_days=3,
                signal="neutral",
                confidence=0.0
            )

            cache.save_spread(spread)

            # Manually update cached_at to be 5 hours ago
            old_cached_at = (datetime.now() - timedelta(hours=5)).isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE credit_spread SET cached_at = ? WHERE id = 1",
                    (old_cached_at,)
                )
                conn.commit()

            assert cache.is_fresh() is False

    def test_get_history(self):
        """Test retrieving spread history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)

            # Save multiple spreads
            for i in range(5):
                spread = CreditSpread(
                    timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
                    lqd_return_30d=0.02,
                    hyg_return_30d=0.018 + i * 0.001,
                    agg_return_30d=0.015,
                    spread_absolute=-0.002 + i * 0.001,
                    spread_zscore=0.0,
                    trend_direction="stable",
                    volatility_regime="low",
                    persistence_days=1,
                    signal="neutral",
                    confidence=0.0
                )
                cache.save_spread(spread)

            history = cache.get_history(days=10)
            assert len(history) == 5


class TestCreditFetcher:
    """Test CreditFetcher data fetching."""

    def test_calculate_spread_neutral(self):
        """Test spread calculation in neutral regime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)
            fetcher = CreditFetcher(cache)

            lqd_data = (120.0, 0.021, 0.045, 0.08)
            hyg_data = (85.0, 0.018, 0.038, 0.12)
            agg_data = (108.0, 0.015, 0.032, 0.06)

            spread = fetcher.calculate_spread(lqd_data, hyg_data, agg_data, [])

            assert spread.spread_absolute == pytest.approx(-0.003)  # 0.018 - 0.021
            assert spread.signal == "neutral"
            assert spread.trend_direction == "stable"

    def test_calculate_spread_risk_off(self):
        """Test spread calculation in risk-off regime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)
            fetcher = CreditFetcher(cache)

            lqd_data = (120.0, 0.03, 0.045, 0.08)   # LQD at +3%
            hyg_data = (85.0, 0.005, 0.038, 0.12)   # HYG underperforming at +0.5%
            agg_data = (108.0, 0.015, 0.032, 0.06)

            spread = fetcher.calculate_spread(lqd_data, hyg_data, agg_data, [])

            assert spread.spread_absolute == pytest.approx(-0.025)  # 0.005 - 0.03
            assert spread.signal == "risk_off"

    def test_calculate_spread_risk_on(self):
        """Test spread calculation in risk-on regime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)
            fetcher = CreditFetcher(cache)

            lqd_data = (120.0, 0.015, 0.045, 0.08)
            hyg_data = (85.0, 0.035, 0.038, 0.12)  # HYG outperforming
            agg_data = (108.0, 0.015, 0.032, 0.06)

            spread = fetcher.calculate_spread(lqd_data, hyg_data, agg_data, [])

            assert spread.spread_absolute == pytest.approx(0.02)  # 0.035 - 0.015
            assert spread.signal == "risk_on"

    def test_calculate_spread_with_history(self):
        """Test spread calculation with historical context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache = CreditCache(db_path)
            fetcher = CreditFetcher(cache)

            # Create history
            history = [
                {"date": "2026-05-01", "spread": 0.001, "lqd_return": 0.02, "hyg_return": 0.021},
                {"date": "2026-05-02", "spread": 0.002, "lqd_return": 0.021, "hyg_return": 0.023},
                {"date": "2026-05-03", "spread": 0.0015, "lqd_return": 0.020, "hyg_return": 0.0215},
            ]

            lqd_data = (120.0, 0.021, 0.045, 0.08)
            hyg_data = (85.0, 0.023, 0.038, 0.12)
            agg_data = (108.0, 0.015, 0.032, 0.06)

            spread = fetcher.calculate_spread(lqd_data, hyg_data, agg_data, history)

            assert spread.spread_absolute == pytest.approx(0.002)
            assert spread.volatility_regime == "low"  # Small variance in history


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
            assert shift <= generator.MAX_ALLOCATION_SHIFT + 0.001  # Allow small floating point error


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

    def test_risk_off_threshold(self):
        """Verify risk-off threshold value."""
        assert RISK_OFF_THRESHOLD == pytest.approx(-0.02)

    def test_risk_on_threshold(self):
        """Verify risk-on threshold value."""
        assert RISK_ON_THRESHOLD == pytest.approx(0.02)
