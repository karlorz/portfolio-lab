"""
Tests for Behavioral Sentiment Signal Generator — v2.70 Phase 2
"""

import sys
import os
import pytest
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentSnapshot,
    BehavioralSentimentFetcher,
    OptionsSentiment,
    RetailFlow,
    SocialIntensity,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_options(**kwargs):
    defaults = {
        "timestamp": "2026-05-14T10:00:00",
        "skew_index": 102.0,
        "vix": 16.0,
        "vix9d": 14.4,
        "vix9d_ratio": 0.90,
        "put_call_ratio": 0.65,
        "fear_greed_score": -0.08,
    }
    defaults.update(kwargs)
    return OptionsSentiment(**defaults)


def _make_retail(**kwargs):
    defaults = {
        "timestamp": "2026-05-14T10:00:00",
        "retail_call_put_ratio": 1.0,
        "retail_buy_sell_imbalance": 0.0,
        "retail_top_100_correlation": -0.15,
        "small_lot_premium_ratio": 0.8,
    }
    defaults.update(kwargs)
    return RetailFlow(**defaults)


def _make_social(**kwargs):
    defaults = {
        "timestamp": "2026-05-14T10:00:00",
        "mention_velocity_7d": 1.0,
        "sentiment_divergence": 0.0,
        "bot_activity_flag": False,
        "influencer_concentration": 0.15,
    }
    defaults.update(kwargs)
    return SocialIntensity(**defaults)


def _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0, confidence=0.7, **kwargs):
    return BehavioralSentimentSnapshot(
        timestamp="2026-05-14T10:00:00",
        options=_make_options(vix=vix, fear_greed_score=composite_score),
        retail=_make_retail(),
        social=_make_social(),
        composite_score=composite_score,
        signal_type=signal_type,
        confidence=confidence,
        data_fresh=True,
        **kwargs,
    )


@pytest.fixture
def tmp_cache_db(tmp_path):
    """Create a temporary cache database"""
    db = tmp_path / "test_market.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS behavioral_zscore_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            composite_score REAL,
            signal_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    return db


# ── BehavioralSignal dataclass tests ───────────────────────────────────


class TestBehavioralSignalDataclass:
    """Tests for the BehavioralSignal dataclass"""

    def test_create_default_signal(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="neutral",
            confidence=0.7,
            equity_shift_pct=0.0,
            holding_period_days=5,
            z_score=0.0,
            composite_score=0.0,
            vix=16.0,
            regime_suppressed=False,
            rationale="No signal",
            timestamp="2026-05-14T10:00:00",
        )
        assert sig.signal_type == "neutral"
        assert sig.confidence == 0.7
        assert sig.equity_shift_pct == 0.0
        assert sig.holding_period_days == 5
        assert sig.z_score == 0.0
        assert sig.regime_suppressed is False

    def test_signal_types(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        types = ["contrarian_buy", "contrarian_sell", "moderate_buy", "moderate_sell", "neutral"]
        for t in types:
            sig = BehavioralSignal(
                signal_type=t,
                confidence=0.5,
                equity_shift_pct=0.0,
                holding_period_days=5,
                z_score=0.0,
                composite_score=0.0,
                vix=16.0,
                regime_suppressed=False,
                rationale="test",
                timestamp="2026-05-14T10:00:00",
            )
            assert sig.signal_type == t

    def test_to_dict(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy",
            confidence=0.85,
            equity_shift_pct=5.0,
            holding_period_days=5,
            z_score=-2.1,
            composite_score=-2.5,
            vix=29.0,
            regime_suppressed=False,
            rationale="Extreme fear",
            timestamp="2026-05-14T10:00:00",
        )
        d = sig.to_dict()
        assert d["signal_type"] == "contrarian_buy"
        assert d["confidence"] == 0.85
        assert d["equity_shift_pct"] == 5.0
        assert d["z_score"] == -2.1
        assert d["regime_suppressed"] is False

    def test_equity_shift_capped(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy",
            confidence=0.9,
            equity_shift_pct=5.0,
            holding_period_days=5,
            z_score=-3.0,
            composite_score=-3.0,
            vix=20.0,
            regime_suppressed=False,
            rationale="Max fear",
            timestamp="2026-05-14T10:00:00",
        )
        assert abs(sig.equity_shift_pct) <= 5.0

    def test_negative_equity_shift(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_sell",
            confidence=0.8,
            equity_shift_pct=-5.0,
            holding_period_days=5,
            z_score=2.5,
            composite_score=2.8,
            vix=14.0,
            regime_suppressed=False,
            rationale="Extreme greed",
            timestamp="2026-05-14T10:00:00",
        )
        assert sig.equity_shift_pct == -5.0

    def test_regime_suppressed_signal(self):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="neutral",
            confidence=0.35,
            equity_shift_pct=0.0,
            holding_period_days=5,
            z_score=-2.0,
            composite_score=-2.5,
            vix=35.0,
            regime_suppressed=True,
            rationale="Suppressed",
            timestamp="2026-05-14T10:00:00",
        )
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0


# ── BehavioralSentimentSignal initialization tests ─────────────────────


class TestBehavioralSentimentSignalInit:
    """Tests for signal generator initialization"""

    def test_init_creates_zscore_table(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)

        conn = sqlite3.connect(str(tmp_cache_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='behavioral_zscore_history'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_stores_fetcher(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        assert sig_gen.fetcher is not None

    def test_init_default_state(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        assert sig_gen._last_signal_time is None
        assert sig_gen._last_signal_type is None
        assert sig_gen._signal_count_5d == 0
        assert sig_gen._pause_until is None


# ── Z-score computation tests ──────────────────────────────────────────


class TestZScoreComputation:
    """Tests for rolling z-score normalization"""

    def test_zscore_with_no_history(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(-2.5)
        # With no history, falls back to score / 1.5
        assert z == pytest.approx(-1.6667, rel=1e-3)

    def test_zscore_with_history(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        # Populate history with known distribution
        conn = sqlite3.connect(str(tmp_cache_db))
        for score in [0.0, 0.5, -0.5, 0.2, -0.2] * 10:  # 50 samples, mean≈0
            conn.execute(
                "INSERT INTO behavioral_zscore_history (timestamp, composite_score, signal_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), score, "neutral"),
            )
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(2.5)
        # Should be a high positive z-score (far from mean ~0)
        assert z > 2.0

    def test_zscore_with_constant_history(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        for _ in range(20):
            conn.execute(
                "INSERT INTO behavioral_zscore_history (timestamp, composite_score, signal_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), 0.5, "neutral"),
            )
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(0.5)
        assert z == 0.0  # same as mean, zero std

    def test_zscore_with_insufficient_history(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        for score in [0.0, 0.1]:
            conn.execute(
                "INSERT INTO behavioral_zscore_history (timestamp, composite_score, signal_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), score, "neutral"),
            )
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(1.5)
        # Fewer than 10 samples → fallback
        assert z == pytest.approx(1.0, rel=1e-3)


# ── Regime check tests ─────────────────────────────────────────────────


class TestRegimeCheck:
    """Tests for VIX-based regime gating"""

    def test_normal_vix_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(16.0)
        assert suppressed is False
        assert reason == ""

    def test_crisis_vix_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(35.0)
        assert suppressed is True
        assert "crisis" in reason.lower()

    def test_high_vol_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(30.0)
        assert suppressed is True
        assert "high volatility" in reason.lower()

    def test_elevated_vix_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(25.0)
        assert suppressed is False

    def test_above_crisis_threshold(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(45.0)
        assert suppressed is True


# ── Circuit breaker tests ──────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for circuit breaker logic"""

    def test_no_breaker_when_clean(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is False

    def test_pause_blocks_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._pause_until = datetime.now() + timedelta(hours=24)
        blocked, reason = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is True
        assert "paused" in reason.lower()

    def test_expired_pause_allows_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._pause_until = datetime.now() - timedelta(hours=1)
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is False

    def test_duplicate_signal_blocked(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        sig_gen._last_signal_type = "contrarian_buy"
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is True

    def test_neutral_never_blocked(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now()
        sig_gen._last_signal_type = "neutral"
        blocked, _ = sig_gen._circuit_breaker_check("neutral")
        assert blocked is False


# ── Signal generation tests ────────────────────────────────────────────


class TestGetSignal:
    """Tests for get_signal method"""

    def test_neutral_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0

    def test_extreme_fear_contrarian_buy(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "contrarian_buy"
        assert sig.equity_shift_pct == 5.0

    def test_fear_moderate_buy(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-1.5, signal_type="fear", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "moderate_buy"
        assert sig.equity_shift_pct == 3.0

    def test_extreme_greed_contrarian_sell(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=14.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "contrarian_sell"
        assert sig.equity_shift_pct == -5.0

    def test_greed_moderate_sell(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=1.5, signal_type="greed", vix=14.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "moderate_sell"
        assert sig.equity_shift_pct == -3.0

    def test_high_vix_suppresses_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=32.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0

    def test_crisis_vix_suppresses_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=40.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"

    def test_elevated_vix_half_weight(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        # Fear with elevated VIX → half weight
        snapshot = _make_snapshot(composite_score=-2.0, signal_type="extreme_fear", vix=26.0)
        sig = sig_gen.get_signal(snapshot)
        # equity_shift should be halved from 5.0 to 2.5
        assert sig.equity_shift_pct == pytest.approx(2.5, rel=1e-2)
        # confidence should be reduced
        assert sig.confidence < 0.7

    def test_holding_period_always_5(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        for st, vix in [("extreme_fear", 16.0), ("neutral", 16.0), ("extreme_greed", 14.0)]:
            snapshot = _make_snapshot(composite_score=-2.5 if "fear" in st else 2.5, signal_type=st, vix=vix)
            sig = sig_gen.get_signal(snapshot)
            assert sig.holding_period_days == 5

    def test_signal_records_score(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.0, signal_type="fear", vix=16.0)
        sig_gen.get_signal(snapshot)

        conn = sqlite3.connect(str(tmp_cache_db))
        cursor = conn.execute("SELECT COUNT(*) FROM behavioral_zscore_history")
        count = cursor.fetchone()[0]
        conn.close()
        assert count >= 1

    def test_signal_timestamp_set(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.timestamp is not None
        assert len(sig.timestamp) > 0


# ── Historical backfill tests ──────────────────────────────────────────


class TestHistoricalBackfill:
    """Tests for synthetic historical backfill"""

    def test_backfill_with_no_data(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-01-31")
        assert results == []

    def test_backfill_with_synthetic_data(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        # Insert synthetic VIX data
        conn = sqlite3.connect(str(tmp_cache_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        test_data = [
            ("^VIX", "2020-03-16", 82.0),   # COVID crash
            ("^VIX", "2020-03-17", 75.0),
            ("^VIX", "2020-03-18", 70.0),
            ("^VIX", "2020-06-01", 25.0),   # recovery
            ("^VIX", "2020-06-02", 24.0),
            ("^VIX", "2021-01-04", 22.0),   # normal
            ("^VIX", "2021-01-05", 20.0),
            ("^VIX", "2021-07-01", 12.0),   # greed (low VIX)
            ("^VIX", "2021-07-02", 11.0),
        ]
        conn.executemany("INSERT INTO prices (symbol, date, close) VALUES (?, ?, ?)", test_data)
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2022-12-31")
        assert len(results) == 9

    def test_backfill_crisis_suppressed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        conn.execute("INSERT INTO prices VALUES ('^VIX', '2020-03-16', 82.0)")
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert len(results) == 1
        assert results[0]["regime_suppressed"] is True
        assert results[0]["signal_type"] == "neutral"
        assert results[0]["equity_shift_pct"] == 0.0

    def test_backfill_low_vix_greed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        conn.execute("INSERT INTO prices VALUES ('^VIX', '2021-07-01', 11.0)")
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "contrarian_sell"
        assert results[0]["equity_shift_pct"] == -5.0

    def test_backfill_normal_vix_neutral(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        conn.execute("INSERT INTO prices VALUES ('^VIX', '2021-02-15', 18.0)")
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "neutral"

    def test_backfill_keys_present(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = sqlite3.connect(str(tmp_cache_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                close REAL
            )
        """)
        conn.execute("INSERT INTO prices VALUES ('^VIX', '2020-06-01', 22.0)")
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2021-12-31")
        r = results[0]
        for key in ["date", "vix", "composite_score", "z_score", "signal_type", "equity_shift_pct", "regime_suppressed"]:
            assert key in r


# ── Pause control tests ────────────────────────────────────────────────


class TestPauseControl:
    """Tests for manual pause/clear"""

    def test_trigger_pause(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen.trigger_pause(24, "Test pause")
        assert sig_gen._pause_until is not None

    def test_clear_pause(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen.trigger_pause(24, "Test")
        sig_gen.clear_pause()
        assert sig_gen._pause_until is None

    def test_get_status(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        status = sig_gen.get_status()
        assert "paused" in status
        assert "pause_until" in status
        assert "last_signal_time" in status
        assert "last_signal_type" in status
        assert "signal_count_5d" in status
        assert status["paused"] is False
