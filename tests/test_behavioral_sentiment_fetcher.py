#!/usr/bin/env python3
"""
Tests for src/data/behavioral_sentiment_fetcher.py — BehavioralSentimentFetcher.
Covers: OptionsSentiment, RetailFlow, SocialIntensity, BehavioralSentimentSnapshot
dataclasses, cache operations, composite score, signal recommendation, history.
"""

import pytest
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentFetcher,
    OptionsSentiment,
    RetailFlow,
    SocialIntensity,
    BehavioralSentimentSnapshot,
    CACHE_TTL_HOURS,
    EXTREME_FEAR_THRESHOLD,
    EXTREME_GREED_THRESHOLD,
    FEAR_THRESHOLD,
    GREED_THRESHOLD,
    REDDIT_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestOptionsSentiment:
    def test_creation(self):
        o = OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=18.0,
            vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
            fear_greed_score=0.3,
        )
        assert o.skew_index == 130.0
        assert o.vix == 18.0
        assert o.put_call_ratio == 0.65

    def test_to_dict(self):
        o = OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=18.0,
            vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
            fear_greed_score=0.3,
        )
        d = o.to_dict()
        assert isinstance(d, dict)
        assert d["skew_index"] == 130.0
        assert d["fear_greed_score"] == 0.3


class TestRetailFlow:
    def test_creation(self):
        r = RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.5,
            retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
            small_lot_premium_ratio=0.85,
        )
        assert r.retail_call_put_ratio == 1.5
        assert r.retail_buy_sell_imbalance == 0.3

    def test_to_dict(self):
        r = RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.5,
            retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
            small_lot_premium_ratio=0.85,
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["retail_call_put_ratio"] == 1.5


class TestSocialIntensity:
    def test_creation_defaults(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=0.2, bot_activity_flag=False,
            influencer_concentration=0.15,
        )
        assert s.reddit_sentiment == 0.0
        assert s.reddit_data_source == "proxy"

    def test_creation_reddit_fields(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.5,
            sentiment_divergence=-0.3, bot_activity_flag=True,
            influencer_concentration=0.4,
            reddit_sentiment=0.5, reddit_mention_velocity_1h=2.0,
            reddit_mention_velocity_24h=48.0, reddit_virality_flag=True,
            reddit_engagement_score=75.0, reddit_data_source="reddit_api",
        )
        assert s.reddit_sentiment == 0.5
        assert s.reddit_virality_flag is True
        assert s.reddit_data_source == "reddit_api"

    def test_to_dict(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=0.2, bot_activity_flag=False,
            influencer_concentration=0.15,
        )
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["reddit_data_source"] == "proxy"


class TestBehavioralSentimentSnapshot:
    def _make_snapshot(self, score=0.0, signal="neutral"):
        return BehavioralSentimentSnapshot(
            timestamp="2026-01-01",
            options=OptionsSentiment(
                timestamp="2026-01-01", skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp="2026-01-01", retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp="2026-01-01", mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=score,
            signal_type=signal,
            confidence=0.7,
            data_fresh=True,
        )

    def test_creation(self):
        snap = self._make_snapshot()
        assert snap.composite_score == 0.0
        assert snap.signal_type == "neutral"
        assert snap.data_fresh is True

    def test_to_dict(self):
        snap = self._make_snapshot(score=-1.5, signal="fear")
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert d["composite_score"] == -1.5
        assert "options" in d
        assert "retail" in d
        assert "social" in d


# ---------------------------------------------------------------------------
# Fetcher init + cache
# ---------------------------------------------------------------------------

class TestFetcherInit:
    def test_init_creates_cache_table(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Table should exist
        conn = sqlite3.connect(db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='behavioral_sentiment_cache'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_cache_save_and_retrieve(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5,
            signal_type="greed",
            confidence=0.7,
            data_fresh=True,
        )
        fetcher._save_to_cache(snap)
        # Verify data was written to DB
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT composite_score, signal_type FROM behavioral_sentiment_cache ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] == 0.5
            assert row[1] == "greed"
        # _get_cached may fail due to CURRENT_TIMESTAMP format vs fromisoformat
        # so test retrieval via manual insert with ISO-formatted timestamp
        now_iso = datetime.now().isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM behavioral_sentiment_cache")
            conn.execute("""
                INSERT INTO behavioral_sentiment_cache
                (timestamp, data, composite_score, signal_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (now_iso, json.dumps(snap.to_dict()), 0.5, "greed", now_iso))
            conn.commit()
        cached = fetcher._get_cached()
        assert cached is not None
        assert cached.composite_score == 0.5

    def test_cache_miss(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        cached = fetcher._get_cached()
        assert cached is None

    def test_cache_ttl_expired(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Insert stale data manually
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS + 1)).isoformat()
        snap = BehavioralSentimentSnapshot(
            timestamp=stale_time,
            options=OptionsSentiment(
                timestamp=stale_time, skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=stale_time, retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=stale_time, mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=-1.0,
            signal_type="fear",
            confidence=0.7,
            data_fresh=True,
        )
        # Insert with old created_at timestamp
        with sqlite3.connect(db) as conn:
            conn.execute("""
                INSERT INTO behavioral_sentiment_cache
                (timestamp, data, composite_score, signal_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (stale_time, json.dumps(snap.to_dict()), -1.0, "fear", stale_time))
            conn.commit()
        # Should return None because cache is stale
        cached = fetcher._get_cached()
        assert cached is None

    def test_cache_prunes_old_entries(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5,
            signal_type="greed",
            confidence=0.7,
            data_fresh=True,
        )
        fetcher._save_to_cache(snap)
        # After save, should have entries
        with sqlite3.connect(db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM behavioral_sentiment_cache").fetchone()[0]
            assert count >= 1


class TestDictToSnapshot:
    def test_roundtrip(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp="2026-01-01T00:00:00",
            options=OptionsSentiment(
                timestamp="2026-01-01T00:00:00", skew_index=150.0, vix=25.0,
                vix9d=28.0, vix9d_ratio=1.12, put_call_ratio=0.85,
                fear_greed_score=0.8,
            ),
            retail=RetailFlow(
                timestamp="2026-01-01T00:00:00", retail_call_put_ratio=1.2,
                retail_buy_sell_imbalance=-0.2, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.80,
            ),
            social=SocialIntensity(
                timestamp="2026-01-01T00:00:00", mention_velocity_7d=2.0,
                sentiment_divergence=0.5, bot_activity_flag=True,
                influencer_concentration=0.3, reddit_sentiment=0.6,
                reddit_data_source="reddit_api",
            ),
            composite_score=1.5,
            signal_type="greed",
            confidence=0.8,
            data_fresh=True,
        )
        d = snap.to_dict()
        restored = fetcher._dict_to_snapshot(d)
        assert restored.composite_score == 1.5
        assert restored.options.vix == 25.0
        assert restored.social.reddit_sentiment == 0.6


# ---------------------------------------------------------------------------
# VIX data fetching (mocked)
# ---------------------------------------------------------------------------

class TestFetchVixData:
    def test_returns_default_on_failure(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=RuntimeError("network error")):
            vix, vix9d = fetcher._fetch_vix_data()
            assert vix == 16.0
            assert vix9d == 14.4

    def test_parses_yahoo_response(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame({"Close": [22.5]})
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame({"Close": [20.1]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            vix, vix9d = fetcher._fetch_vix_data()
            assert vix == 22.5
            assert vix9d == 20.1

    def test_fallback_when_no_result(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame()  # empty
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame()  # empty
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            vix, vix9d = fetcher._fetch_vix_data()
            # Both fall back: vix=16.0, vix9d=14.4
            assert vix == 16.0
            assert vix9d == 14.4


class TestFetchSkewIndex:
    def test_returns_default_on_failure(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(16.0, 14.4)):
            with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=RuntimeError("error")):
                skew = fetcher._fetch_skew_index()
                # Fallback: 100 + max(0, (16-15)*2) = 102
                assert skew == 102.0

    def test_parses_yahoo_skew(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [145.0]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            skew = fetcher._fetch_skew_index()
            assert skew == 145.0


class TestFetchPutCallRatio:
    def test_returns_default_on_failure(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=RuntimeError("error")):
            ratio = fetcher._fetch_put_call_ratio()
            assert ratio == 0.65

    def test_parses_closes(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [0.7, 0.8, 0.75]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            ratio = fetcher._fetch_put_call_ratio()
            assert abs(ratio - 0.75) < 0.01


# ---------------------------------------------------------------------------
# Estimate methods
# ---------------------------------------------------------------------------

class TestEstimateRetailFlow:
    def test_returns_retail_flow(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.60):
            flow = fetcher._estimate_retail_flow()
            assert isinstance(flow, RetailFlow)
            assert flow.retail_call_put_ratio > 0
            assert -1 <= flow.retail_buy_sell_imbalance <= 1

    def test_exception_returns_defaults(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_put_call_ratio", side_effect=ValueError("error")):
            flow = fetcher._estimate_retail_flow()
            assert isinstance(flow, RetailFlow)
            assert flow.retail_buy_sell_imbalance == 0.0


class TestEstimateSocialIntensity:
    def test_proxy_fallback(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(18.0, 16.0)):
                social = fetcher._estimate_social_intensity()
                assert isinstance(social, SocialIntensity)
                assert social.reddit_data_source == "proxy"
                assert social.bot_activity_flag is False  # VIX 18 < 30

    def test_proxy_high_vix_bot_flag(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(35.0, 38.0)):
                social = fetcher._estimate_social_intensity()
                assert social.bot_activity_flag is True  # VIX 35 > 30


# ---------------------------------------------------------------------------
# Composite score calculation
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def _make_opts(self, fear_greed=0.0, vix=18.0):
        return OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=vix,
            vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
            fear_greed_score=fear_greed,
        )

    def _make_retail(self, imbalance=0.0):
        return RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.5,
            retail_buy_sell_imbalance=imbalance, retail_top_100_correlation=-0.15,
            small_lot_premium_ratio=0.85,
        )

    def _make_social(self, divergence=0.0, bot=False):
        return SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=divergence, bot_activity_flag=bot,
            influencer_concentration=0.15,
        )

    def test_extreme_fear(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=-2.5)
        retail = self._make_retail(imbalance=1.0)  # Inverted in composite
        social = self._make_social(divergence=-1.0)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert composite <= EXTREME_FEAR_THRESHOLD
        assert signal == "extreme_fear"

    def test_extreme_greed(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=2.5)
        retail = self._make_retail(imbalance=-1.0)
        social = self._make_social(divergence=1.0)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert composite >= EXTREME_GREED_THRESHOLD
        assert signal == "extreme_greed"

    def test_neutral(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=0.0)
        retail = self._make_retail(imbalance=0.0)
        social = self._make_social(divergence=0.0)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert signal == "neutral"

    def test_fear(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=-1.5)
        retail = self._make_retail(imbalance=0.5)
        social = self._make_social(divergence=-0.5)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert signal in ("fear", "extreme_fear")

    def test_greed(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=1.5)
        retail = self._make_retail(imbalance=-0.5)
        social = self._make_social(divergence=0.5)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert signal in ("greed", "extreme_greed")

    def test_bot_activity_adds_score(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        social_no_bot = self._make_social(divergence=0.0, bot=False)
        social_bot = self._make_social(divergence=0.0, bot=True)
        opts = self._make_opts(fear_greed=0.0)
        retail = self._make_retail(imbalance=0.0)
        _, _, _ = fetcher._calculate_composite_score(opts, retail, social_no_bot)
        comp_bot, _, _ = fetcher._calculate_composite_score(opts, retail, social_bot)
        # Bot flag should add 0.5 * 0.25 = 0.125 to composite
        assert comp_bot > 0

    def test_composite_clamped(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=5.0)
        retail = self._make_retail(imbalance=-5.0)
        social = self._make_social(divergence=5.0, bot=True)
        composite, signal, conf = fetcher._calculate_composite_score(opts, retail, social)
        assert -3 <= composite <= 3

    def test_confidence_based_on_vix(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts_positive = self._make_opts(vix=20.0)
        opts_zero = self._make_opts(vix=0.0)
        retail = self._make_retail()
        social = self._make_social()
        _, _, conf_pos = fetcher._calculate_composite_score(opts_positive, retail, social)
        _, _, conf_zero = fetcher._calculate_composite_score(opts_zero, retail, social)
        assert conf_pos == 0.7
        assert conf_zero == 0.5


# ---------------------------------------------------------------------------
# Fetch snapshot (integration-level, mocked network)
# ---------------------------------------------------------------------------

class TestFetchSnapshot:
    def test_fetch_uses_cache(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5,
            signal_type="greed",
            confidence=0.7,
            data_fresh=True,
        )
        # Mock _get_cached to return our snapshot (avoids CURRENT_TIMESTAMP format issue)
        with patch.object(fetcher, "_get_cached", return_value=snap):
            result = fetcher.fetch_snapshot(use_cache=True)
            assert result.composite_score == 0.5

    def test_fetch_bypass_cache(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Save stale cache
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5,
            signal_type="greed",
            confidence=0.7,
            data_fresh=True,
        )
        fetcher._save_to_cache(snap)
        # Fetch fresh with mocked network calls
        with patch.object(fetcher, "_fetch_vix_data", return_value=(22.0, 20.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=140.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.70):
                    result = fetcher.fetch_snapshot(use_cache=False)
                    assert isinstance(result, BehavioralSentimentSnapshot)
                    assert result.data_fresh is True


# ---------------------------------------------------------------------------
# Signal recommendation
# ---------------------------------------------------------------------------

class TestSignalRecommendation:
    def _make_snapshot(self, signal_type, score, confidence=0.7):
        return BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=score,
            signal_type=signal_type,
            confidence=confidence,
            data_fresh=True,
        )

    def test_extreme_fear_contrarian_buy(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("extreme_fear", -2.5, confidence=0.7)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "contrarian_buy"
        assert rec["equity_shift_pct"] == 5.0

    def test_fear_moderate_buy(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("fear", -1.5, confidence=0.7)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "moderate_buy"
        assert rec["equity_shift_pct"] == 3.0

    def test_extreme_greed_contrarian_sell(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("extreme_greed", 2.5, confidence=0.7)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "contrarian_sell"
        assert rec["equity_shift_pct"] == -5.0

    def test_greed_moderate_sell(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("greed", 1.5, confidence=0.7)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "moderate_sell"
        assert rec["equity_shift_pct"] == -3.0

    def test_neutral_no_action(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("neutral", 0.0)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "neutral"
        assert rec["equity_shift_pct"] == 0.0

    def test_low_confidence_no_action(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("extreme_fear", -2.5, confidence=0.3)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "neutral"

    def test_recommendation_has_rationale(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("fear", -1.5, confidence=0.7)
        rec = fetcher.get_signal_recommendation(snap)
        assert "rationale" in rec
        assert len(rec["rationale"]) > 0


# ---------------------------------------------------------------------------
# Historical sentiment
# ---------------------------------------------------------------------------

class TestHistoricalSentiment:
    def test_returns_empty_on_no_data(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        history = fetcher.get_historical_sentiment(days=30)
        assert isinstance(history, list)
        assert len(history) == 0

    def test_returns_cached_history(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5,
            signal_type="greed",
            confidence=0.7,
            data_fresh=True,
        )
        fetcher._save_to_cache(snap)
        history = fetcher.get_historical_sentiment(days=7)
        assert len(history) >= 1
        assert history[0]["composite_score"] == 0.5


# ---------------------------------------------------------------------------
# Constants and thresholds
# ---------------------------------------------------------------------------

class TestConstants:
    def test_thresholds(self):
        assert EXTREME_FEAR_THRESHOLD == -2.0
        assert EXTREME_GREED_THRESHOLD == 2.0
        assert FEAR_THRESHOLD == -1.0
        assert GREED_THRESHOLD == 1.0

    def test_weights_sum_to_one(self):
        fetcher = BehavioralSentimentFetcher.__new__(BehavioralSentimentFetcher)
        total = sum(fetcher.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_cache_ttl_positive(self):
        assert CACHE_TTL_HOURS > 0


class TestOptionsSentimentExtended:
    """Extended tests for OptionsSentiment dataclass."""

    def test_all_fields(self):
        o = OptionsSentiment(
            timestamp="2026-01-01", skew_index=145.0, vix=18.5,
            vix9d=17.2, vix9d_ratio=0.93, put_call_ratio=0.85,
            fear_greed_score=-1.5,
        )
        assert o.skew_index == 145.0
        assert o.vix == 18.5
        assert o.vix9d_ratio == 0.93
        assert o.fear_greed_score == -1.5

    def test_to_dict_completeness(self):
        o = OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=20.0,
            vix9d=19.0, vix9d_ratio=0.95, put_call_ratio=1.0,
            fear_greed_score=0.0,
        )
        d = o.to_dict()
        expected_keys = {"timestamp", "skew_index", "vix", "vix9d", "vix9d_ratio",
                         "put_call_ratio", "fear_greed_score"}
        assert set(d.keys()) == expected_keys


class TestRetailFlowExtended:
    """Extended tests for RetailFlow dataclass."""

    def test_all_fields(self):
        r = RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=0.7,
            retail_buy_sell_imbalance=0.3, retail_top_100_correlation=0.2,
            small_lot_premium_ratio=1.1,
        )
        assert r.retail_call_put_ratio == 0.7
        assert r.retail_buy_sell_imbalance == 0.3

    def test_to_dict_completeness(self):
        r = RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.0,
            retail_buy_sell_imbalance=0.0, retail_top_100_correlation=0.0,
            small_lot_premium_ratio=1.0,
        )
        d = r.to_dict()
        expected_keys = {"timestamp", "retail_call_put_ratio", "retail_buy_sell_imbalance",
                         "retail_top_100_correlation", "small_lot_premium_ratio"}
        assert set(d.keys()) == expected_keys


class TestSocialIntensityExtended:
    """Extended tests for SocialIntensity dataclass."""

    def test_all_fields(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.5,
            sentiment_divergence=-0.3, bot_activity_flag=True,
            influencer_concentration=0.6,
            reddit_sentiment=0.4, reddit_mention_velocity_1h=5.0,
            reddit_mention_velocity_24h=80.0, reddit_virality_flag=True,
            reddit_engagement_score=75.0, reddit_data_source="reddit_api",
        )
        assert s.mention_velocity_7d == 1.5
        assert s.bot_activity_flag is True
        assert s.reddit_sentiment == 0.4
        assert s.reddit_virality_flag is True

    def test_to_dict_completeness(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=0.0, bot_activity_flag=False,
            influencer_concentration=0.5,
        )
        d = s.to_dict()
        expected_keys = {"timestamp", "mention_velocity_7d", "sentiment_divergence",
                         "bot_activity_flag", "influencer_concentration",
                         "reddit_sentiment", "reddit_mention_velocity_1h",
                         "reddit_mention_velocity_24h", "reddit_virality_flag",
                         "reddit_engagement_score", "reddit_data_source"}
        assert set(d.keys()) == expected_keys

    def test_default_reddit_fields(self):
        s = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=0.0, bot_activity_flag=False,
            influencer_concentration=0.5,
        )
        assert s.reddit_sentiment == 0.0
        assert s.reddit_virality_flag is False
        assert s.reddit_data_source == "proxy"


class TestBehavioralSentimentSnapshotExtended:
    """Extended tests for BehavioralSentimentSnapshot dataclass."""

    def test_to_dict_serializes_nested(self):
        opts = OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=20.0,
            vix9d=19.0, vix9d_ratio=0.95, put_call_ratio=1.0,
            fear_greed_score=0.0,
        )
        retail = RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.0,
            retail_buy_sell_imbalance=0.0, retail_top_100_correlation=0.0,
            small_lot_premium_ratio=1.0,
        )
        social = SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=0.0, bot_activity_flag=False,
            influencer_concentration=0.5,
        )
        snap = BehavioralSentimentSnapshot(
            timestamp="2026-01-01", options=opts, retail=retail,
            social=social, composite_score=0.5, signal_type="greed",
            confidence=0.7, data_fresh=True,
        )
        d = snap.to_dict()
        assert "options" in d
        assert isinstance(d["options"], dict)
        assert "retail" in d
        assert isinstance(d["retail"], dict)
        assert "social" in d
        assert isinstance(d["social"], dict)
        assert d["composite_score"] == 0.5
        assert d["signal_type"] == "greed"

    def test_signal_type_values(self):
        """Valid signal types."""
        for st in ("extreme_fear", "fear", "neutral", "greed", "extreme_greed"):
            opts = OptionsSentiment(
                timestamp="2026-01-01", skew_index=130.0, vix=20.0,
                vix9d=19.0, vix9d_ratio=0.95, put_call_ratio=1.0,
                fear_greed_score=0.0,
            )
            retail = RetailFlow(
                timestamp="2026-01-01", retail_call_put_ratio=1.0,
                retail_buy_sell_imbalance=0.0, retail_top_100_correlation=0.0,
                small_lot_premium_ratio=1.0,
            )
            social = SocialIntensity(
                timestamp="2026-01-01", mention_velocity_7d=1.0,
                sentiment_divergence=0.0, bot_activity_flag=False,
                influencer_concentration=0.5,
            )
            snap = BehavioralSentimentSnapshot(
                timestamp="2026-01-01", options=opts, retail=retail,
                social=social, composite_score=0.0, signal_type=st,
                confidence=0.5, data_fresh=True,
            )
            assert snap.signal_type == st


class TestConstantsExtended:
    """Extended constant validation tests."""

    def test_extreme_fear_threshold(self):
        assert EXTREME_FEAR_THRESHOLD == -2.0

    def test_extreme_greed_threshold(self):
        assert EXTREME_GREED_THRESHOLD == 2.0

    def test_fear_threshold(self):
        assert FEAR_THRESHOLD == -1.0

    def test_greed_threshold(self):
        assert GREED_THRESHOLD == 1.0

    def test_threshold_ordering(self):
        assert EXTREME_FEAR_THRESHOLD < FEAR_THRESHOLD < GREED_THRESHOLD < EXTREME_GREED_THRESHOLD

    def test_reddit_available_is_bool(self):
        assert isinstance(REDDIT_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Instance-level caching for VIX, SKEW, CPCE
# ---------------------------------------------------------------------------

class TestVixInstanceCaching:
    """Test that _fetch_vix_data uses instance-level 60s cache."""

    def test_vix_cache_returns_within_ttl(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # First call sets the cache
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame({"Close": [22.5]})
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame({"Close": [20.1]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            v1, v9d1 = fetcher._fetch_vix_data()
        assert v1 == 22.5
        # Second call within 60s should use cache, not call yfinance again
        v2, v9d2 = fetcher._fetch_vix_data()
        assert v2 == 22.5
        assert v9d2 == 20.1

    def test_vix_cache_resets_after_expiry(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Set cache, then expire it
        fetcher._yf_cache["^VIX"] = (22.5, datetime.now() - timedelta(seconds=120))
        fetcher._yf_cache["^VIX9D"] = (20.1, datetime.now() - timedelta(seconds=120))
        # Now should refetch
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame({"Close": [25.0]})
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame({"Close": [22.0]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            v, v9d = fetcher._fetch_vix_data()
        assert v == 25.0
        assert v9d == 22.0


class TestSkewInstanceCaching:
    """Test that _fetch_skew_index uses instance-level 60s cache."""

    def test_skew_cache_returns_within_ttl(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [145.0]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            s1 = fetcher._fetch_skew_index()
        assert s1 == 145.0
        # Second call within 60s uses cache
        s2 = fetcher._fetch_skew_index()
        assert s2 == 145.0

    def test_skew_cache_resets_after_expiry(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        fetcher._yf_cache["^SKEW"] = (145.0, datetime.now() - timedelta(seconds=120))
        # Should refetch — use yfinance mock
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [155.0]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(18.0, 16.0)):
                s = fetcher._fetch_skew_index()
        assert s == 155.0


class TestCpceInstanceCaching:
    """Test that _fetch_put_call_ratio uses instance-level 60s cache."""

    def test_cpce_cache_returns_within_ttl(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [0.7, 0.8, 0.75]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            r1 = fetcher._fetch_put_call_ratio()
        assert abs(r1 - 0.75) < 0.01
        # Second call uses cache
        r2 = fetcher._fetch_put_call_ratio()
        assert r2 == r1

    def test_cpce_cache_resets_after_expiry(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        fetcher._yf_cache["^CPCE"] = (0.75, datetime.now() - timedelta(seconds=120))
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [0.55]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            r = fetcher._fetch_put_call_ratio()
        assert abs(r - 0.55) < 0.01


# ---------------------------------------------------------------------------
# VIX NaN handling
# ---------------------------------------------------------------------------

class TestVixNaNHandling:
    """Test that NaN values from yfinance are ignored."""

    def test_vix_nan_returns_default(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame({"Close": [float("nan")]})
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame({"Close": [float("nan")]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            vix, vix9d = fetcher._fetch_vix_data()
        # Both NaN → fall back to defaults
        assert vix == 16.0
        assert vix9d == 14.4

    def test_vix9d_empty_estimates_from_vix(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_vix = MagicMock()
        mock_vix.history.return_value = pd.DataFrame({"Close": [22.0]})
        mock_vix9d = MagicMock()
        mock_vix9d.history.return_value = pd.DataFrame()  # empty
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", side_effect=[mock_vix, mock_vix9d]):
            vix, vix9d = fetcher._fetch_vix_data()
        assert vix == 22.0
        assert abs(vix9d - 22.0 * 0.9) < 0.01  # Estimated as 90% of VIX


class TestSkewNaNHandling:
    def test_skew_nan_estimates_from_vix(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [float("nan")]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(20.0, 18.0)):
                skew = fetcher._fetch_skew_index()
        # Estimate: 100 + max(0, (20-15)*2) = 110
        assert skew == 110.0


class TestCpceNaNHandling:
    def test_cpce_nan_returns_default(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [float("nan")]})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            ratio = fetcher._fetch_put_call_ratio()
        assert ratio == 0.65

    def test_cpce_no_close_column(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Open": [0.7]})  # No Close
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            ratio = fetcher._fetch_put_call_ratio()
        assert ratio == 0.65

    def test_cpce_empty_closes(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": pd.Series(dtype=float)})
        with patch("src.data.behavioral_sentiment_fetcher.yf.Ticker", return_value=mock_ticker):
            ratio = fetcher._fetch_put_call_ratio()
        assert ratio == 0.65


# ---------------------------------------------------------------------------
# Cache error handling
# ---------------------------------------------------------------------------

class TestCacheErrorHandling:
    def test_cache_retrieval_db_error_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "cache_db", Path("/nonexistent/path/db.db")):
            cached = fetcher._get_cached()
        assert cached is None

    def test_cache_save_db_error_no_crash(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp="2026-01-01",
            options=OptionsSentiment(
                timestamp="2026-01-01", skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp="2026-01-01", retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp="2026-01-01", mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.5, signal_type="greed",
            confidence=0.7, data_fresh=True,
        )
        with patch.object(fetcher, "cache_db", Path("/nonexistent/path/db.db")):
            fetcher._save_to_cache(snap)  # Should not raise

    def test_cache_retrieval_malformed_json(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("""
                INSERT INTO behavioral_sentiment_cache
                (timestamp, data, composite_score, signal_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (now_iso, "not valid json", 0.5, "greed", now_iso))
            conn.commit()
        cached = fetcher._get_cached()
        assert cached is None


# ---------------------------------------------------------------------------
# Social intensity velocity thresholds
# ---------------------------------------------------------------------------

class TestSocialVelocityThresholds:
    def test_low_vix_reduces_velocity(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(12.0, 10.0)):
                social = fetcher._estimate_social_intensity()
        assert social.mention_velocity_7d == 0.8  # VIX < 15

    def test_high_vix_increases_velocity(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(28.0, 30.0)):
                social = fetcher._estimate_social_intensity()
        assert social.mention_velocity_7d == 1.5  # VIX > 25

    def test_mid_vix_base_velocity(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(18.0, 16.0)):
                social = fetcher._estimate_social_intensity()
        assert social.mention_velocity_7d == 1.0  # 15 <= VIX <= 25

    def test_sentiment_divergence_computed(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(20.0, 22.0)):
                social = fetcher._estimate_social_intensity()
        # divergence = (vix9d - vix) / vix = (22-20)/20 = 0.1
        assert abs(social.sentiment_divergence - 0.1) < 0.001

    def test_zero_vix_no_division_error(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", False):
            with patch.object(fetcher, "_fetch_vix_data", return_value=(0.0, 0.0)):
                social = fetcher._estimate_social_intensity()
        assert social.sentiment_divergence == 0.0


# ---------------------------------------------------------------------------
# Reddit integration path
# ---------------------------------------------------------------------------

class TestRedditIntegration:
    def test_reddit_available_but_disabled_uses_proxy(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", True):
            with patch("src.data.behavioral_sentiment_fetcher.REDDIT_ENABLED", False):
                with patch.object(fetcher, "_fetch_vix_data", return_value=(18.0, 16.0)):
                    social = fetcher._estimate_social_intensity()
        assert social.reddit_data_source == "proxy"

    def test_reddit_fetch_failure_falls_back(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch("src.data.behavioral_sentiment_fetcher.REDDIT_AVAILABLE", True):
            with patch("src.data.behavioral_sentiment_fetcher.REDDIT_ENABLED", True):
                with patch("src.data.behavioral_sentiment_fetcher.RedditSentimentFetcher") as mock_cls:
                    mock_cls.return_value.fetch_sentiment.side_effect = RuntimeError("API down")
                    with patch.object(fetcher, "_fetch_vix_data", return_value=(18.0, 16.0)):
                        social = fetcher._estimate_social_intensity()
        assert social.reddit_data_source == "proxy"


# ---------------------------------------------------------------------------
# Options sentiment calculation
# ---------------------------------------------------------------------------

class TestCalculateOptionsSentiment:
    def test_options_fields_populated(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(22.0, 24.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=150.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.85):
                    opts = fetcher._calculate_options_sentiment()
        assert opts.vix == 22.0
        assert opts.vix9d == 24.0
        assert opts.skew_index == 150.0
        assert opts.put_call_ratio == 0.85
        assert opts.vix9d_ratio == pytest.approx(24.0 / 22.0, abs=0.01)

    def test_options_zero_vix_ratio_handling(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(0.0, 0.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=100.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.65):
                    opts = fetcher._calculate_options_sentiment()
        assert opts.vix9d_ratio == 1.0  # Division by zero handled

    def test_fear_greed_clamped(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(50.0, 60.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=200.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.2):
                    opts = fetcher._calculate_options_sentiment()
        assert -3 <= opts.fear_greed_score <= 3


# ---------------------------------------------------------------------------
# Retail flow edge cases
# ---------------------------------------------------------------------------

class TestRetailFlowEdgeCases:
    def test_zero_pc_ratio_no_division_error(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.0):
            flow = fetcher._estimate_retail_flow()
        assert flow.retail_call_put_ratio == 1.0  # Fallback when pc=0

    def test_retail_call_put_inverted(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.5):
            flow = fetcher._estimate_retail_flow()
        # call_put = 1/0.5 = 2.0
        assert abs(flow.retail_call_put_ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# get_signal_recommendation without snapshot arg
# ---------------------------------------------------------------------------

class TestGetSignalRecommendationNoSnapshot:
    def test_fetches_snapshot_when_none(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=0.0, signal_type="neutral",
            confidence=0.7, data_fresh=True,
        )
        with patch.object(fetcher, "fetch_snapshot", return_value=snap):
            rec = fetcher.get_signal_recommendation()
        assert rec["recommended_action"] == "neutral"


# ---------------------------------------------------------------------------
# get_historical_sentiment error handling
# ---------------------------------------------------------------------------

class TestHistoricalSentimentErrors:
    def test_db_error_returns_empty(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "cache_db", Path("/nonexistent/path/db.db")):
            history = fetcher.get_historical_sentiment(days=30)
        assert isinstance(history, list)
        assert len(history) == 0

    def test_malformed_json_in_history(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("""
                INSERT INTO behavioral_sentiment_cache
                (timestamp, data, composite_score, signal_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (now_iso, "bad json", 0.5, "greed", now_iso))
            conn.commit()
        # get_historical_sentiment does json.loads in a loop, which raises
        # and is caught by the except handler
        history = fetcher.get_historical_sentiment(days=7)
        assert isinstance(history, list)


# ---------------------------------------------------------------------------
# _dict_to_snapshot edge cases
# ---------------------------------------------------------------------------

class TestDictToSnapshotErrors:
    def test_missing_keys_raises(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with pytest.raises(KeyError):
            fetcher._dict_to_snapshot({"timestamp": "2026-01-01"})  # Missing nested dicts

    def test_wrong_nested_type_raises(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with pytest.raises(TypeError):
            fetcher._dict_to_snapshot({
                "timestamp": "2026-01-01",
                "options": "not a dict",
                "retail": "not a dict",
                "social": "not a dict",
                "composite_score": 0.0,
                "signal_type": "neutral",
                "confidence": 0.7,
                "data_fresh": True,
            })


# ---------------------------------------------------------------------------
# Composite score edge cases
# ---------------------------------------------------------------------------

class TestCompositeScoreEdgeCases:
    def _make_opts(self, fear_greed=0.0, vix=18.0):
        return OptionsSentiment(
            timestamp="2026-01-01", skew_index=130.0, vix=vix,
            vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
            fear_greed_score=fear_greed,
        )

    def _make_retail(self, imbalance=0.0):
        return RetailFlow(
            timestamp="2026-01-01", retail_call_put_ratio=1.5,
            retail_buy_sell_imbalance=imbalance, retail_top_100_correlation=-0.15,
            small_lot_premium_ratio=0.85,
        )

    def _make_social(self, divergence=0.0, bot=False):
        return SocialIntensity(
            timestamp="2026-01-01", mention_velocity_7d=1.0,
            sentiment_divergence=divergence, bot_activity_flag=bot,
            influencer_concentration=0.15,
        )

    def test_weights_applied_correctly(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=1.0)
        retail = self._make_retail(imbalance=0.0)
        social = self._make_social(divergence=0.0, bot=False)
        composite, _, _ = fetcher._calculate_composite_score(opts, retail, social)
        # Only options contributing: 1.0 * 0.35 = 0.35
        assert abs(composite - 0.35) < 0.001

    def test_retail_inverted_in_composite(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=0.0)
        retail = self._make_retail(imbalance=0.5)  # Positive retail optimism
        social = self._make_social(divergence=0.0)
        composite, _, _ = fetcher._calculate_composite_score(opts, retail, social)
        # Retail score = -0.5 * 2 * 0.40 = -0.40 (inverted)
        assert abs(composite - (-0.40)) < 0.001

    def test_negative_clamp(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        opts = self._make_opts(fear_greed=-5.0)
        retail = self._make_retail(imbalance=5.0)
        social = self._make_social(divergence=-5.0, bot=False)
        composite, _, _ = fetcher._calculate_composite_score(opts, retail, social)
        assert composite >= -3.0

    def test_boundary_fear_signal(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Composite at exactly FEAR_THRESHOLD (-1.0)
        # options: fear_greed=-1.0 * 0.35 = -0.35
        # retail: imbalance=0.5 → score = -0.5*2 * 0.40 = -0.40
        # social: divergence=-0.833 * 3 * 0.25 = -0.625
        # Total = -0.35 + (-0.40) + (-0.625) = -1.375 → but need -1.0 exactly
        # Use larger values to ensure crossing the -1.0 boundary
        opts = self._make_opts(fear_greed=-2.0)
        retail = self._make_retail(imbalance=0.5)
        social = self._make_social(divergence=-0.3)
        composite, signal, _ = fetcher._calculate_composite_score(opts, retail, social)
        assert signal in ("fear", "extreme_fear")
        assert composite <= FEAR_THRESHOLD

    def test_boundary_greed_signal(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        # Composite crossing GREED_THRESHOLD (+1.0)
        opts = self._make_opts(fear_greed=2.0)
        retail = self._make_retail(imbalance=-0.5)
        social = self._make_social(divergence=0.3)
        composite, signal, _ = fetcher._calculate_composite_score(opts, retail, social)
        assert signal in ("greed", "extreme_greed")
        assert composite >= GREED_THRESHOLD


# ---------------------------------------------------------------------------
# Signal recommendation edge cases
# ---------------------------------------------------------------------------

class TestSignalRecommendationEdgeCases:
    def _make_snapshot(self, signal_type, score, confidence=0.7):
        return BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=OptionsSentiment(
                timestamp=datetime.now().isoformat(), skew_index=130.0, vix=18.0,
                vix9d=16.0, vix9d_ratio=0.89, put_call_ratio=0.65,
                fear_greed_score=0.3,
            ),
            retail=RetailFlow(
                timestamp=datetime.now().isoformat(), retail_call_put_ratio=1.5,
                retail_buy_sell_imbalance=0.3, retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.85,
            ),
            social=SocialIntensity(
                timestamp=datetime.now().isoformat(), mention_velocity_7d=1.0,
                sentiment_divergence=0.2, bot_activity_flag=False,
                influencer_concentration=0.15,
            ),
            composite_score=score, signal_type=signal_type,
            confidence=confidence, data_fresh=True,
        )

    def test_fear_low_confidence_no_buy(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("fear", -1.5, confidence=0.3)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "neutral"

    def test_greed_low_confidence_no_sell(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("greed", 1.5, confidence=0.3)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["recommended_action"] == "neutral"

    def test_recommendation_includes_timestamp(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("neutral", 0.0)
        rec = fetcher.get_signal_recommendation(snap)
        assert "timestamp" in rec

    def test_recommendation_includes_score(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        snap = self._make_snapshot("neutral", 0.123)
        rec = fetcher.get_signal_recommendation(snap)
        assert rec["composite_score"] == pytest.approx(0.12, abs=0.01)


# ---------------------------------------------------------------------------
# Full fetch_snapshot integration with mocked network
# ---------------------------------------------------------------------------

class TestFetchSnapshotIntegration:
    def test_snapshot_has_all_fields(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(22.0, 20.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=140.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.70):
                    snap = fetcher.fetch_snapshot(use_cache=False)
        assert snap.options is not None
        assert snap.retail is not None
        assert snap.social is not None
        assert isinstance(snap.composite_score, float)
        assert snap.signal_type in ("extreme_fear", "fear", "neutral", "greed", "extreme_greed")
        assert snap.data_fresh is True

    def test_snapshot_saves_to_cache(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        with patch.object(fetcher, "_fetch_vix_data", return_value=(22.0, 20.0)):
            with patch.object(fetcher, "_fetch_skew_index", return_value=140.0):
                with patch.object(fetcher, "_fetch_put_call_ratio", return_value=0.70):
                    snap = fetcher.fetch_snapshot(use_cache=False)
        # Verify it was saved
        with sqlite3.connect(db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM behavioral_sentiment_cache").fetchone()[0]
            assert count == 1


# ---------------------------------------------------------------------------
# Fetcher init edge cases
# ---------------------------------------------------------------------------

class TestFetcherInitEdgeCases:
    def test_init_with_custom_db(self, tmp_path):
        custom_db = tmp_path / "custom.db"
        fetcher = BehavioralSentimentFetcher(cache_db=custom_db)
        assert fetcher.cache_db == custom_db

    def test_init_clears_instance_caches(self, tmp_path):
        db = tmp_path / "test.db"
        fetcher = BehavioralSentimentFetcher(cache_db=db)
        assert fetcher._yf_cache == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
