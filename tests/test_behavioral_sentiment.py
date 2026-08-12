"""
Tests for Behavioral Sentiment Signal Generator — v2.70 Phase 2
"""

import pytest
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch


from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentSnapshot,
    OptionsSentiment,
    RetailFlow,
    SocialIntensity,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_named_regime_from_live_ssot(monkeypatch):
    """Avoid live data/regime_state.json (often NORMAL) suppressing unit tests.

    Production still reads regime_state; tests opt into NORMAL via explicit
    named_regime= or by re-patching _resolve_named_regime.
    """
    from src.signals.behavioral_sentiment import BehavioralSentimentSignal

    monkeypatch.setattr(
        BehavioralSentimentSignal,
        "_resolve_named_regime",
        lambda self: "LOW_VOL",
    )


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
    """Tests for VIX-based regime gating (named_regime=None → pure VIX path)."""

    def test_normal_vix_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(16.0, named_regime=None)
        assert suppressed is False
        assert reason == ""

    def test_crisis_vix_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(35.0, named_regime=None)
        assert suppressed is True
        assert "crisis" in reason.lower()

    def test_high_vol_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(30.0, named_regime=None)
        assert suppressed is True
        assert "high volatility" in reason.lower()

    def test_elevated_vix_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(25.0, named_regime=None)
        assert suppressed is False

    def test_above_crisis_threshold(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(45.0, named_regime=None)
        assert suppressed is True

    def test_named_regime_normal_suppresses_low_vix(self, tmp_cache_db):
        """RegimeGate OFF in NORMAL even when VIX is calm."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(16.0, named_regime="NORMAL")
        assert suppressed is True
        assert "regimegate" in reason.lower() or "normal" in reason.lower()

    def test_named_regime_low_vol_allows_low_vix(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(16.0, named_regime="LOW_VOL")
        assert suppressed is False
        assert reason == ""


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
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0

    def test_extreme_fear_contrarian_buy(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "contrarian_buy"
        assert sig.equity_shift_pct == 5.0

    def test_fear_moderate_buy(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=-1.5, signal_type="fear", vix=16.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "moderate_buy"
        assert sig.equity_shift_pct == 3.0

    def test_extreme_greed_contrarian_sell(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=14.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "contrarian_sell"
        assert sig.equity_shift_pct == -5.0

    def test_greed_moderate_sell(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=1.5, signal_type="greed", vix=14.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "moderate_sell"
        assert sig.equity_shift_pct == -3.0

    def test_high_vix_suppresses_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=32.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0

    def test_crisis_vix_suppresses_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=40.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"

    def test_elevated_vix_half_weight(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
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
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            for st, vix in [("extreme_fear", 16.0), ("neutral", 16.0), ("extreme_greed", 14.0)]:
                snapshot = _make_snapshot(composite_score=-2.5 if "fear" in st else 2.5, signal_type=st, vix=vix)
                sig = sig_gen.get_signal(snapshot)
                assert sig.holding_period_days == 5

    def test_signal_records_score(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="LOW_VOL"):
            snapshot = _make_snapshot(composite_score=-2.0, signal_type="fear", vix=16.0)
            sig_gen.get_signal(snapshot)

        conn = sqlite3.connect(str(tmp_cache_db))
        cursor = conn.execute("SELECT COUNT(*) FROM behavioral_zscore_history")
        count = cursor.fetchone()[0]
        conn.close()
        assert count >= 1

    def test_normal_regime_suppresses_active_signal(self, tmp_cache_db):
        """Producer must not publish active contrarian signals when RegimeGate is OFF."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(sig_gen, "_resolve_named_regime", return_value="NORMAL"):
            snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
            sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"
        snap = sig.to_signal_snapshot()
        assert snap.is_active is False

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


# ── BehavioralSignal to_signal_snapshot tests ──────────────────────────


class TestBehavioralSignalToSnapshot:
    """Tests for to_signal_snapshot() conversion"""

    def test_contrarian_buy_maps_value_0_5(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy", confidence=0.8, equity_shift_pct=5.0,
            holding_period_days=5, z_score=-2.5, composite_score=-2.5,
            vix=20.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.value == 0.5
        assert snap.source == "behavioral_sentiment"
        assert snap.is_active is True

    def test_contrarian_sell_maps_value_minus_0_5(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_sell", confidence=0.8, equity_shift_pct=-5.0,
            holding_period_days=5, z_score=2.5, composite_score=2.8,
            vix=14.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.value == -0.5
        assert snap.is_active is True

    def test_moderate_buy_maps_value_0_3(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="moderate_buy", confidence=0.7, equity_shift_pct=3.0,
            holding_period_days=5, z_score=-1.5, composite_score=-1.5,
            vix=16.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.value == 0.3

    def test_moderate_sell_maps_value_minus_0_3(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="moderate_sell", confidence=0.7, equity_shift_pct=-3.0,
            holding_period_days=5, z_score=1.5, composite_score=1.5,
            vix=14.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.value == -0.3

    def test_neutral_maps_value_0_0(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="neutral", confidence=0.5, equity_shift_pct=0.0,
            holding_period_days=5, z_score=0.0, composite_score=0.0,
            vix=16.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.value == 0.0

    def test_regime_suppressed_sets_inactive(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy", confidence=0.9, equity_shift_pct=5.0,
            holding_period_days=5, z_score=-3.0, composite_score=-3.0,
            vix=40.0, regime_suppressed=True, rationale="suppressed",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.is_active is False
        assert snap.value == 0.5  # value still mapped from signal_type

    def test_low_confidence_sets_inactive(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="moderate_buy", confidence=0.2, equity_shift_pct=3.0,
            holding_period_days=5, z_score=-1.0, composite_score=-1.0,
            vix=16.0, regime_suppressed=False, rationale="low conf",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.is_active is False
        assert snap.asset_signals == {"SPY": 3.0}

    def test_confidence_0_3_boundary_active(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy", confidence=0.3, equity_shift_pct=5.0,
            holding_period_days=5, z_score=-2.0, composite_score=-2.0,
            vix=16.0, regime_suppressed=False, rationale="boundary",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.is_active is True  # 0.3 >= 0.3

    def test_metadata_contains_all_fields(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy", confidence=0.85, equity_shift_pct=5.0,
            holding_period_days=5, z_score=-2.1, composite_score=-2.5,
            vix=29.0, regime_suppressed=False, rationale="test",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert snap.metadata["signal_type"] == "contrarian_buy"
        assert snap.metadata["composite_score"] == -2.5
        assert snap.metadata["z_score"] == -2.1
        assert snap.metadata["vix"] == 29.0
        assert snap.metadata["regime_suppressed"] is False

    def test_explanation_contains_key_fields(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal

        sig = BehavioralSignal(
            signal_type="contrarian_buy", confidence=0.8, equity_shift_pct=5.0,
            holding_period_days=5, z_score=-2.5, composite_score=-2.5,
            vix=20.0, regime_suppressed=True, rationale="suppressed",
            timestamp="2026-05-14T10:00:00",
        )
        snap = sig.to_signal_snapshot()
        assert "Behavioral:" in snap.explanation
        assert "suppressed=True" in snap.explanation


# ── Init edge-case tests ────────────────────────────────────────────────


class TestInitEdgeCases:
    """Tests for initialization edge cases"""

    def test_init_default_cache_db_resolves(self, tmp_path):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal
        from src.paths import MARKET_DB

        sig_gen = BehavioralSentimentSignal()
        assert sig_gen.cache_db == MARKET_DB

    def test_init_with_pathlib_path(self, tmp_path):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        db = tmp_path / "subdir" / "test.db"
        db.parent.mkdir()
        conn = __import__("sqlite3").connect(str(db))
        conn.execute("CREATE TABLE IF NOT EXISTS behavioral_zscore_history (id INTEGER PRIMARY KEY)")
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=db)
        assert sig_gen.cache_db == db

    def test_init_zscore_table_failure_does_not_raise(self, tmp_cache_db):
        """When sqlite_connect raises during _init_zscore_table, catch and continue."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal
        import src.paths as path_mod

        with patch.object(path_mod, "sqlite_connect", side_effect=RuntimeError("DB failure")):
            sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        # Should not raise; warning logged
        assert sig_gen.cache_db == tmp_cache_db


# ── Z-score edge-case tests ─────────────────────────────────────────────


class TestZScoreEdgeCases:
    """Tests for z-score computation edge cases"""

    def test_get_zscore_sqlite_failure_fallback(self, tmp_cache_db):
        """When sqlite_connect fails, _get_zscore returns composite/1.5"""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal
        import src.paths as path_mod

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(path_mod, "sqlite_connect", side_effect=RuntimeError("DB failure")):
            z = sig_gen._get_zscore(-3.0)
        assert z == pytest.approx(-2.0, rel=1e-3)

    def test_record_score_sqlite_failure_does_not_raise(self, tmp_cache_db):
        """When sqlite_connect fails, _record_score catches and logs."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal
        import src.paths as path_mod

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(path_mod, "sqlite_connect", side_effect=RuntimeError("DB failure")):
            # Should not raise any exception
            sig_gen._record_score(-2.5, "extreme_fear")

    def test_get_zscore_exactly_10_samples(self, tmp_cache_db):
        """10 samples is enough for mean/std computation (not < 10)."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = __import__("sqlite3").connect(str(tmp_cache_db))
        for i in range(10):
            conn.execute(
                "INSERT INTO behavioral_zscore_history (timestamp, composite_score, signal_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), float(i), "neutral"),
            )
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(9.5)
        assert z > 1.5  # far from mean of 4.5

    def test_get_zscore_std_is_zero_returns_zero(self, tmp_cache_db):
        """When std < 0.01, z-score is 0.0."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        conn = __import__("sqlite3").connect(str(tmp_cache_db))
        for score in [0.5] * 20:
            conn.execute(
                "INSERT INTO behavioral_zscore_history (timestamp, composite_score, signal_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), score, "neutral"),
            )
        conn.commit()
        conn.close()

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        z = sig_gen._get_zscore(0.5)
        assert z == 0.0


# ── Regime check boundary tests ─────────────────────────────────────────


class TestRegimeCheckBoundaries:
    """Tests for regime check at exact threshold boundaries (VIX-only path)."""

    def test_elevated_exact_25_not_suppressed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal, VIX_ELEVATED_THRESHOLD

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(VIX_ELEVATED_THRESHOLD, named_regime=None)
        assert suppressed is False

    def test_high_exact_30_suppressed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal, VIX_HIGH_THRESHOLD

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(VIX_HIGH_THRESHOLD, named_regime=None)
        assert suppressed is True
        assert "high volatility" in reason.lower()

    def test_crisis_exact_35_suppressed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal, VIX_CRISIS_THRESHOLD

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, reason = sig_gen._regime_check(VIX_CRISIS_THRESHOLD, named_regime=None)
        assert suppressed is True
        assert "crisis" in reason.lower()

    def test_just_below_crisis_still_high(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal, VIX_CRISIS_THRESHOLD

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(VIX_CRISIS_THRESHOLD - 0.1, named_regime=None)
        assert suppressed is True  # still >= VIX_HIGH_THRESHOLD

    def test_just_below_high_not_suppressed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal, VIX_HIGH_THRESHOLD

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(VIX_HIGH_THRESHOLD - 0.1, named_regime=None)
        assert suppressed is False

    def test_zero_vix_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        suppressed, _ = sig_gen._regime_check(0.0, named_regime=None)
        assert suppressed is False


# ── Circuit breaker edge-case tests ─────────────────────────────────────


class TestCircuitBreakerEdgeCases:
    """Tests for circuit breaker edge cases"""

    def test_different_signal_types_not_blocked(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        sig_gen._last_signal_type = "contrarian_buy"
        # Different type — not a duplicate
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_sell")
        assert blocked is False

    def test_duplicate_allowed_after_5_days(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(days=6)
        sig_gen._last_signal_type = "contrarian_buy"
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is False

    def test_two_non_neutral_churn_control(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        sig_gen._signal_count_5d = 2
        blocked, reason = sig_gen._circuit_breaker_check("contrarian_sell")
        assert blocked is True
        assert "churn" in reason.lower()

    def test_churn_control_does_not_block_neutral(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        sig_gen._signal_count_5d = 5
        blocked, _ = sig_gen._circuit_breaker_check("neutral")
        assert blocked is False

    def test_churn_control_below_threshold_allows(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        sig_gen._signal_count_5d = 1
        blocked, _ = sig_gen._circuit_breaker_check("contrarian_buy")
        assert blocked is False


# ── GetSignal edge-case tests ────────────────────────────────────────────


class TestGetSignalEdgeCases:
    """Tests for get_signal edge cases"""

    def test_calls_fetcher_when_no_snapshot(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        with patch.object(sig_gen.fetcher, "fetch_snapshot", return_value=snapshot):
            sig = sig_gen.get_signal()
        assert sig.signal_type == "neutral"

    def test_regime_suppressed_takes_priority_over_elevated(self, tmp_cache_db):
        """When VIX >= 30 (high), regime_suppressed=True and elevated logic is skipped."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        # VIX=30 triggers regime suppression (high vol). Elevated VIX (25-30) is NOT applied.
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=30.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is True
        assert sig.signal_type == "neutral"
        assert sig.equity_shift_pct == 0.0
        # confidence should be halved by regime_suppression
        assert sig.confidence == pytest.approx(0.35, rel=1e-2)  # 0.7 * 0.5

    def test_elevated_vix_reduces_sell_signal_weight(self, tmp_cache_db):
        """Elevated VIX (25-30) halves equity_shift for sell signals too."""
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=26.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.regime_suppressed is False
        # equity_shift: -5.0 * 0.5 = -2.5
        assert sig.equity_shift_pct == pytest.approx(-2.5, rel=1e-2)
        # confidence: 0.7 * 0.8 = 0.56
        assert sig.confidence == pytest.approx(0.56, rel=1e-2)

    def test_circuit_breaker_reduces_confidence(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._signal_count_5d = 2
        sig_gen._last_signal_time = datetime.now() - timedelta(hours=2)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert sig.signal_type == "neutral"
        # confidence: 0.7 * 0.3 = 0.21
        assert sig.confidence == pytest.approx(0.21, rel=1e-2)

    def test_regime_rationale_includes_vix_info(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=40.0)
        sig = sig_gen.get_signal(snapshot)
        # Regime suppression rationale should mention crisis
        assert "crisis" in sig.rationale.lower()

    def test_neutral_rationale_with_no_suppression(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert "No extreme sentiment detected" in sig.rationale

    def test_state_updates_after_non_neutral_signal(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
        sig_gen.get_signal(snapshot)
        assert sig_gen._last_signal_type == "contrarian_buy"
        assert sig_gen._last_signal_time is not None

    def test_signal_count_increments_within_5_days(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        # Trigger first signal: extreme_fear → contrarian_buy
        snap1 = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
        sig_gen.get_signal(snap1)
        assert sig_gen._signal_count_5d == 1

        # Trigger second signal with DIFFERENT type: extreme_greed → contrarian_sell
        # (different signal type avoids duplicate check)
        snap2 = _make_snapshot(composite_score=2.5, signal_type="extreme_greed", vix=14.0)
        sig_gen.get_signal(snap2)
        assert sig_gen._signal_count_5d == 2

    def test_neutral_signal_does_not_update_state(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        sig_gen.get_signal(snapshot)
        assert sig_gen._last_signal_time is None
        assert sig_gen._last_signal_type is None
        assert sig_gen._signal_count_5d == 0

    def test_get_signal_returns_behavioral_signal_type(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSignal, BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=0.0, signal_type="neutral", vix=16.0)
        sig = sig_gen.get_signal(snapshot)
        assert isinstance(sig, BehavioralSignal)


# ── Trigger pause edge-case tests ───────────────────────────────────────


class TestTriggerPauseEdgeCases:
    """Tests for trigger_pause and clear_pause edge cases"""

    def test_trigger_pause_zero_hours(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen.trigger_pause(0, "Immediate resume")
        assert sig_gen._pause_until is not None
        # With 0 hours, pause_until should be very close to now
        remaining = (sig_gen._pause_until - datetime.now()).total_seconds()
        assert remaining < 60  # less than a minute

    def test_clear_pause_when_already_none(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen.clear_pause()  # Should not raise
        assert sig_gen._pause_until is None

    def test_get_status_with_active_pause(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen.trigger_pause(24, "Test")
        status = sig_gen.get_status()
        assert status["paused"] is True
        assert status["pause_until"] is not None

    def test_get_status_with_expired_pause(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        sig_gen._pause_until = datetime.now() - timedelta(hours=1)
        status = sig_gen.get_status()
        assert status["paused"] is False

    def test_get_status_after_signal_state_update(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        snapshot = _make_snapshot(composite_score=-2.5, signal_type="extreme_fear", vix=16.0)
        sig_gen.get_signal(snapshot)
        status = sig_gen.get_status()
        assert status["last_signal_type"] == "contrarian_buy"
        assert status["signal_count_5d"] == 1


# ── Historical backfill boundary tests ──────────────────────────────────


class TestHistoricalBackfillBoundaries:
    """Tests for historical backfill at VIX boundary values"""

    def _insert_vix(self, db_path, date_str, vix_close):
        conn = __import__("sqlite3").connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)", (date_str, vix_close))
        conn.commit()
        conn.close()

    def test_backfill_vix_12_extreme_greed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2021-07-01", 12.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "contrarian_sell"
        assert results[0]["composite_score"] == 2.0

    def test_backfill_vix_14_greed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2021-07-01", 14.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "moderate_sell"
        assert results[0]["composite_score"] == 1.0

    def test_backfill_vix_15_greed(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2021-07-01", 15.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "moderate_sell"
        assert results[0]["composite_score"] == 1.0

    def test_backfill_vix_16_neutral(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2021-06-01", 16.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2021-01-01", "2021-12-31")
        assert results[0]["signal_type"] == "neutral"
        assert results[0]["composite_score"] == 0.0
        assert results[0]["regime_suppressed"] is False

    def test_backfill_vix_25_fear(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2020-06-01", 25.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert results[0]["signal_type"] == "moderate_buy"
        assert results[0]["composite_score"] == -0.5
        assert results[0]["regime_suppressed"] is False

    def test_backfill_vix_26_fear(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2020-06-01", 26.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert results[0]["signal_type"] == "moderate_buy"
        assert results[0]["composite_score"] == -0.5
        assert results[0]["regime_suppressed"] is False

    def test_backfill_vix_30_regime_suppressed_fear(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2020-06-01", 30.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert results[0]["regime_suppressed"] is True
        assert results[0]["signal_type"] == "neutral"
        assert results[0]["equity_shift_pct"] == 0.0
        # composite still -1.5 for fear
        assert results[0]["composite_score"] == -1.5

    def test_backfill_vix_35_crisis_extreme_fear(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2020-03-16", 35.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert results[0]["regime_suppressed"] is True
        assert results[0]["signal_type"] == "neutral"
        assert results[0]["composite_score"] == -2.5

    def test_backfill_empty_date_range(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        self._insert_vix(tmp_cache_db, "2020-03-16", 35.0)
        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        results = sig_gen.historical_backfill("2099-01-01", "2099-12-31")  # no data in range
        assert results == []

    def test_backfill_sqlite_failure_returns_empty(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal
        import src.paths as path_mod

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        with patch.object(path_mod, "sqlite_connect", side_effect=RuntimeError("DB failure")):
            results = sig_gen.historical_backfill("2020-01-01", "2020-12-31")
        assert results == []


# ── Module-level constant tests ─────────────────────────────────────────


class TestConstants:
    """Tests for module-level constants and __all__ exports"""

    def test_min_holding_days_value(self):
        from src.signals.behavioral_sentiment import MIN_HOLDING_DAYS
        assert MIN_HOLDING_DAYS == 5

    def test_max_equity_shift_pct_value(self):
        from src.signals.behavioral_sentiment import MAX_EQUITY_SHIFT_PCT
        assert MAX_EQUITY_SHIFT_PCT == 5.0

    def test_zscore_window_days_value(self):
        from src.signals.behavioral_sentiment import ZSCORE_WINDOW_DAYS
        assert ZSCORE_WINDOW_DAYS == 90

    def test_vix_threshold_values(self):
        from src.signals.behavioral_sentiment import (
            VIX_CRISIS_THRESHOLD, VIX_HIGH_THRESHOLD, VIX_ELEVATED_THRESHOLD,
        )
        assert VIX_CRISIS_THRESHOLD == 35.0
        assert VIX_HIGH_THRESHOLD == 30.0
        assert VIX_ELEVATED_THRESHOLD == 25.0

    def test_all_exports_contain_expected_symbols(self):
        from src.signals.behavioral_sentiment import __all__

        expected = [
            "MIN_HOLDING_DAYS", "MAX_EQUITY_SHIFT_PCT", "ZSCORE_WINDOW_DAYS",
            "VIX_CRISIS_THRESHOLD", "VIX_HIGH_THRESHOLD", "VIX_ELEVATED_THRESHOLD",
            "BehavioralSignal", "BehavioralSentimentSignal",
        ]
        for exp in expected:
            assert exp in __all__, f"{exp} missing from __all__"

    def test_all_exports_importable(self):
        pass

    def test_confidence_range_zero_to_one(self, tmp_cache_db):
        from src.signals.behavioral_sentiment import BehavioralSentimentSignal

        sig_gen = BehavioralSentimentSignal(cache_db=tmp_cache_db)
        for composite, sig_type, vix in [
            (-2.5, "extreme_fear", 16.0),
            (-1.5, "fear", 16.0),
            (0.0, "neutral", 16.0),
            (1.5, "greed", 14.0),
            (2.5, "extreme_greed", 14.0),
        ]:
            snapshot = _make_snapshot(composite_score=composite, signal_type=sig_type, vix=vix)
            sig = sig_gen.get_signal(snapshot)
            assert 0.0 <= sig.confidence <= 1.0, (
                f"confidence {sig.confidence} out of [0,1] for {sig_type}"
            )


def test_live_payload_prefers_research_caveats_not_backtest_finding():
    """Builder contract: live behavioral block must not ship free-text backtest_finding."""

    src = Path("src/dashboard/signal_section_builder.py").read_text(encoding="utf-8")
    # Static contract: research_caveats present; bare backtest_finding key removed from behavioral block
    assert "research_caveats" in src
    # The behavioral section must not assign backtest_finding= (other dashboards may still)
    # Locate the behavioral_sentiment_data dict assignment region
    idx = src.find("behavioral_sentiment_data = {")
    assert idx > 0
    chunk = src[idx : idx + 2500]
    assert "backtest_finding" not in chunk
    assert "research_caveats" in chunk
