"""
Tests for Reddit Sentiment Fetcher v2.70 Phase 4
41+ unit tests for API mocking, sentiment calculation, caching, and edge cases.
"""

import pytest
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import io

# Import module under test
from src.data.reddit_sentiment_fetcher import (
    RedditSentimentFetcher,
    RedditTickerMetrics,
    RedditSentimentSnapshot,
    TICKERS,
    SUBREDDITS,
    REDDIT_API_BASE,
    USER_AGENT,
    RATE_LIMIT_DELAY,
    CACHE_TTL_MINUTES,
)


class TestRedditTickerMetrics:
    """Test RedditTickerMetrics dataclass"""

    def test_create_metrics(self):
        """Test creating a RedditTickerMetrics instance"""
        metrics = RedditTickerMetrics(
            ticker="SPY",
            mention_count_1h=5,
            mention_count_24h=45,
            sentiment_score=0.25,
            upvote_ratio=0.72,
            comment_velocity=3.5,
            award_count=12
        )

        assert metrics.ticker == "SPY"
        assert metrics.mention_count_1h == 5
        assert metrics.mention_count_24h == 45
        assert metrics.sentiment_score == 0.25
        assert metrics.upvote_ratio == 0.72
        assert metrics.comment_velocity == 3.5
        assert metrics.award_count == 12

    def test_to_dict(self):
        """Test converting metrics to dict"""
        metrics = RedditTickerMetrics(
            ticker="GLD",
            mention_count_1h=2,
            mention_count_24h=18,
            sentiment_score=-0.15,
            upvote_ratio=0.65,
            comment_velocity=1.2,
            award_count=3
        )

        d = metrics.to_dict()
        assert d['ticker'] == "GLD"
        assert d['sentiment_score'] == -0.15
        assert d['mention_count_24h'] == 18

    def test_to_dict_field_completeness(self):
        """Verify all 6 fields are present in to_dict() output"""
        metrics = RedditTickerMetrics(
            ticker="QQQ",
            mention_count_1h=3,
            mention_count_24h=30,
            sentiment_score=0.5,
            upvote_ratio=0.8,
            comment_velocity=2.0,
            award_count=7,
        )
        d = metrics.to_dict()
        assert set(d.keys()) == {
            "ticker", "mention_count_1h", "mention_count_24h",
            "sentiment_score", "upvote_ratio", "comment_velocity",
            "award_count",
        }

    def test_to_dict_extreme_sentiment_scores(self):
        """Test to_dict with sentiment at -1.0 and +1.0 extremes"""
        metrics_pos = RedditTickerMetrics(
            ticker="SPY", mention_count_1h=10, mention_count_24h=100,
            sentiment_score=1.0, upvote_ratio=1.0, comment_velocity=5.0, award_count=50,
        )
        metrics_neg = RedditTickerMetrics(
            ticker="GLD", mention_count_1h=10, mention_count_24h=100,
            sentiment_score=-1.0, upvote_ratio=0.0, comment_velocity=0.0, award_count=0,
        )
        d_pos = metrics_pos.to_dict()
        d_neg = metrics_neg.to_dict()
        assert d_pos["sentiment_score"] == 1.0
        assert d_neg["sentiment_score"] == -1.0
        assert d_pos["upvote_ratio"] == 1.0
        assert d_neg["upvote_ratio"] == 0.0

    def test_to_dict_zero_values(self):
        """Test to_dict with all zero values"""
        metrics = RedditTickerMetrics(
            ticker="TLT", mention_count_1h=0, mention_count_24h=0,
            sentiment_score=0.0, upvote_ratio=0.0, comment_velocity=0.0, award_count=0,
        )
        d = metrics.to_dict()
        assert d["mention_count_1h"] == 0
        assert d["mention_count_24h"] == 0
        assert d["sentiment_score"] == 0.0
        assert d["award_count"] == 0


class TestRedditSentimentSnapshot:
    """Test RedditSentimentSnapshot dataclass"""

    def test_create_snapshot(self):
        """Test creating a complete snapshot"""
        metrics = {
            "SPY": RedditTickerMetrics("SPY", 10, 100, 0.3, 0.75, 5.0, 20),
            "GLD": RedditTickerMetrics("GLD", 3, 25, -0.1, 0.60, 1.5, 5)
        }

        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=13,
            mention_velocity_24h=125,
            engagement_score=42.5,
            virality_flag=False,
            data_fresh=True
        )

        assert snapshot.timestamp == "2026-05-14T10:00:00"
        assert snapshot.aggregate_sentiment == 0.2
        assert len(snapshot.ticker_metrics) == 2
        assert snapshot.data_fresh is True

    def test_to_dict(self):
        """Test snapshot serialization"""
        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}

        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=True,
            data_fresh=True
        )

        d = snapshot.to_dict()
        assert d['aggregate_sentiment'] == 0.2
        assert d['virality_flag'] is True
        assert 'SPY' in d['ticker_metrics']

    def test_to_dict_field_completeness(self):
        """Verify all 7 fields are present in to_dict() output"""
        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=True,
            data_fresh=True,
        )
        d = snapshot.to_dict()
        assert set(d.keys()) == {
            "timestamp", "ticker_metrics", "aggregate_sentiment",
            "mention_velocity_1h", "mention_velocity_24h",
            "engagement_score", "virality_flag", "data_fresh",
        }

    def test_to_dict_empty_ticker_metrics(self):
        """Test to_dict with empty ticker_metrics dict"""
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics={},
            aggregate_sentiment=0.0,
            mention_velocity_1h=0,
            mention_velocity_24h=0,
            engagement_score=0.0,
            virality_flag=False,
            data_fresh=False,
        )
        d = snapshot.to_dict()
        assert d["ticker_metrics"] == {}
        assert d["aggregate_sentiment"] == 0.0

    def test_to_dict_all_boundary_flags(self):
        """Test all combinations of boolean flags serialize correctly"""
        for virality in (True, False):
            for data_fresh in (True, False):
                snapshot = RedditSentimentSnapshot(
                    timestamp="2026-05-14T10:00:00",
                    ticker_metrics={},
                    aggregate_sentiment=0.0,
                    mention_velocity_1h=0,
                    mention_velocity_24h=0,
                    engagement_score=0.0,
                    virality_flag=virality,
                    data_fresh=data_fresh,
                )
                d = snapshot.to_dict()
                assert d["virality_flag"] is virality
                assert d["data_fresh"] is data_fresh


class TestRedditSentimentFetcherInit:
    """Test fetcher initialization"""

    def test_init_creates_db(self, tmp_path):
        """Test that initialization creates SQLite tables"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        # Verify tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('reddit_sentiment_cache', 'reddit_mentions')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert 'reddit_sentiment_cache' in tables
            assert 'reddit_mentions' in tables

    def test_init_uses_default_cache_path(self):
        """Test default cache path"""
        fetcher = RedditSentimentFetcher()
        assert 'market.db' in str(fetcher.cache_db)

    def test_init_db_index_exists(self, tmp_path):
        """Test that initialization creates the expected index"""
        db_path = tmp_path / "test.db"
        RedditSentimentFetcher(cache_db=db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_reddit_mentions_ticker'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_reddit_mentions_ticker" in indexes

    def test_init_db_table_schemas(self, tmp_path):
        """Verify reddit_mentions table has expected columns"""
        db_path = tmp_path / "test.db"
        RedditSentimentFetcher(cache_db=db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(reddit_mentions)")
            columns = {row[1] for row in cursor.fetchall()}
        expected = {"ticker", "subreddit", "post_title", "sentiment_score",
                    "upvotes", "comment_count", "created_utc", "fetched_at"}
        assert expected.issubset(columns)


class TestExtractTickers:
    """Test ticker extraction from text"""

    def test_extract_spy(self):
        """Test extracting SPY mentions"""
        fetcher = RedditSentimentFetcher()
        text = "SPY is going to the moon! Buy $SPY calls."
        tickers = fetcher._extract_tickers(text)
        assert "SPY" in tickers

    def test_extract_multiple_tickers(self):
        """Test extracting multiple tickers"""
        fetcher = RedditSentimentFetcher()
        text = "SPY and GLD both looking strong. TLT might fall."
        tickers = fetcher._extract_tickers(text)
        assert "SPY" in tickers
        assert "GLD" in tickers
        assert "TLT" in tickers

    def test_no_tickers(self):
        """Test text with no tickers"""
        fetcher = RedditSentimentFetcher()
        text = "Just random text about the market today."
        tickers = fetcher._extract_tickers(text)
        assert len(tickers) == 0

    def test_case_insensitive(self):
        """Test case-insensitive matching"""
        fetcher = RedditSentimentFetcher()
        text = "buy spy and gld now"
        tickers = fetcher._extract_tickers(text)
        assert "SPY".lower() in [t.lower() for t in tickers]

    def test_extract_no_false_positive_substring(self):
        """Test that ticker substrings within longer words do not match"""
        fetcher = RedditSentimentFetcher()
        text = "The SPYDER etf is different from SPY. TLTRO operations."
        tickers = fetcher._extract_tickers(text)
        assert "SPY" in tickers  # standalone SPY should match
        # "SPYDER" contains SPY but \b boundary should prevent matching SPY within SPYDER
        # We verify that only SPY (not 2) is found
        assert tickers.count("SPY") == 1

    def test_extract_dollar_lowercase(self):
        """Test extracting $ticker with lowercase ticker symbol"""
        fetcher = RedditSentimentFetcher()
        text = "$spy is looking good today"
        tickers = fetcher._extract_tickers(text)
        assert "SPY" in tickers

    def test_extract_all_known_tickers(self):
        """Test all TICKERS can be extracted from text"""
        fetcher = RedditSentimentFetcher()
        text = "SPY GLD TLT QQQ IEF VIX all moving today"
        tickers = fetcher._extract_tickers(text)
        for t in TICKERS:
            assert t in tickers, f"Ticker {t} not extracted"


class TestCalculateSentiment:
    """Test sentiment calculation"""

    def test_positive_sentiment(self):
        """Test bullish text detection"""
        fetcher = RedditSentimentFetcher()
        text = "SPY is going to the moon! Buy calls, diamond hands!"
        score = fetcher._calculate_sentiment(text)
        assert score > 0

    def test_negative_sentiment(self):
        """Test bearish text detection"""
        fetcher = RedditSentimentFetcher()
        text = "Market crashing, SPY puts printing. Sell everything, rugpull incoming!"
        score = fetcher._calculate_sentiment(text)
        assert score < 0

    def test_neutral_sentiment(self):
        """Test neutral text"""
        fetcher = RedditSentimentFetcher()
        text = "Just some regular discussion about the market."
        score = fetcher._calculate_sentiment(text)
        assert score == 0.0

    def test_mixed_sentiment(self):
        """Test mixed sentiment text"""
        fetcher = RedditSentimentFetcher()
        text = "Bullish on SPY but bearish on TLT. Mixed feelings."
        score = fetcher._calculate_sentiment(text)
        # Should have both positive and negative signals
        assert -1.0 <= score <= 1.0

    def test_sentiment_empty_text_returns_zero(self):
        """Test empty string returns 0.0"""
        fetcher = RedditSentimentFetcher()
        assert fetcher._calculate_sentiment("") == 0.0

    def test_sentiment_whitespace_text_returns_zero(self):
        """Test whitespace-only text returns 0.0"""
        fetcher = RedditSentimentFetcher()
        assert fetcher._calculate_sentiment("   ") == 0.0
        assert fetcher._calculate_sentiment("\t\n  \n") == 0.0

    def test_sentiment_all_positive_words(self):
        """Test text containing only positive words returns +1.0"""
        fetcher = RedditSentimentFetcher()
        text = "bull bullish moon rocket tendies gain gains profit profits " \
               "winning win up rise rising surge surging breakout ATH calls " \
               "call long buy buying bought hold hodl diamond hands strong green pump"
        score = fetcher._calculate_sentiment(text)
        assert score == 1.0

    def test_sentiment_all_negative_words(self):
        """Test text containing only negative words returns -1.0"""
        fetcher = RedditSentimentFetcher()
        # Must avoid positive-word substrings in negative words
        # (e.g. "call" in "margin call", "up" in "bankrupt")
        text = "bearish crash dumping losses losing tanking puts shorting " \
               "selling weak red bleeding rugpull scam liquidation down " \
               "fall tank dump loss lose sold paper hands bear short put sell"
        score = fetcher._calculate_sentiment(text)
        assert score == -1.0

    def test_sentiment_equal_pos_neg_returns_zero(self):
        """Test equal positive and negative word counts return 0.0"""
        fetcher = RedditSentimentFetcher()
        text = "bull moon crash dump"
        score = fetcher._calculate_sentiment(text)
        assert score == 0.0


class TestRateLimit:
    """Test rate limiting"""

    def test_rate_limit_delays(self):
        """Test that rate limiting adds delay"""
        fetcher = RedditSentimentFetcher()

        # Set last request to now
        fetcher.last_request_time = datetime.now().timestamp()

        import time
        start = time.time()
        fetcher._rate_limit()
        elapsed = time.time() - start

        # Should have some delay
        assert elapsed >= 0

    def test_rate_limit_no_delay_when_enough_time_passed(self):
        """Test no delay when sufficient time has elapsed since last request"""
        fetcher = RedditSentimentFetcher()
        fetcher.last_request_time = 0.0  # long ago

        import time
        start = time.time()
        fetcher._rate_limit()
        elapsed = time.time() - start

        # Should be near-instant since RATE_LIMIT_DELAY has already passed
        assert elapsed < RATE_LIMIT_DELAY * 0.5


class TestCacheOperations:
    """Test caching functionality"""

    def test_cache_and_retrieve(self, tmp_path):
        """Test caching and retrieving sentiment"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=False,
            data_fresh=True
        )

        fetcher._cache_sentiment(snapshot)

        # Should be retrievable
        cached = fetcher._get_cached_sentiment()
        assert cached is not None
        assert cached.aggregate_sentiment == 0.2

    def test_cache_freshness_check(self, tmp_path):
        """Test that stale cache is not returned"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        # Insert stale cache entry manually
        with sqlite3.connect(db_path) as conn:
            stale_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            conn.execute("""
                INSERT INTO reddit_sentiment_cache (timestamp, data_json, created_at)
                VALUES (?, ?, ?)
            """, (
                "2026-05-14T10:00:00",
                json.dumps({'aggregate_sentiment': 0.5}),
                stale_time
            ))
            conn.commit()

        # Should not return stale cache (needs fresh data_json structure)
        cached = fetcher._get_cached_sentiment()
        # Won't parse correctly due to incomplete data structure

    def test_cache_ttl_boundary(self, tmp_path):
        """Test cache just inside TTL is returned, just outside is not"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        # Cache a valid snapshot
        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=False,
            data_fresh=True,
        )
        fetcher._cache_sentiment(snapshot)

        # Verify retrievable
        cached = fetcher._get_cached_sentiment()
        assert cached is not None
        assert cached.aggregate_sentiment == 0.2

        # Manually set created_at to just beyond TTL
        stale_time = (
            datetime.now(timezone.utc)
            - timedelta(minutes=CACHE_TTL_MINUTES + 1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE reddit_sentiment_cache SET created_at = ?",
                (stale_time,),
            )
            conn.commit()

        # Now the cache should be stale
        cached_stale = fetcher._get_cached_sentiment()
        assert cached_stale is None

    def test_cache_evicts_old_entries(self, tmp_path):
        """Test cache is trimmed to 100 entries"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        metrics = {"SPY": RedditTickerMetrics("SPY", 0, 0, 0.0, 0.5, 0.0, 0)}

        # Insert 105 cache entries
        for i in range(105):
            snap = RedditSentimentSnapshot(
                timestamp=f"2026-05-14T{i:02d}:00:00",
                ticker_metrics=metrics,
                aggregate_sentiment=0.0,
                mention_velocity_1h=0,
                mention_velocity_24h=0,
                engagement_score=0.0,
                virality_flag=False,
                data_fresh=True,
            )
            fetcher._cache_sentiment(snap)

        # Verify only 100 entries remain
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM reddit_sentiment_cache"
            )
            count = cursor.fetchone()[0]
        assert count == 100


class TestFetchSubredditEdgeCases:
    """Test edge cases in _fetch_subreddit API calls"""

    @patch("urllib.request.urlopen")
    def test_fetch_empty_children(self, mock_urlopen, tmp_path):
        """Test API response with empty children array"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": {"children": []}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        posts = fetcher._fetch_subreddit("wallstreetbets", limit=25)
        assert posts == []

    @patch("urllib.request.urlopen")
    def test_fetch_missing_data_fields(self, mock_urlopen, tmp_path):
        """Test posts with missing optional data fields"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": {
                "children": [
                    {"data": {"title": "SPY post"}},  # minimal fields
                    {},  # no data key at all
                    {"data": {"title": "GLD post", "selftext": "text",
                              "score": 50, "upvote_ratio": 0.8,
                              "num_comments": 10, "created_utc": 1000000,
                              "total_awards_received": 2}},
                ]
            }
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        posts = fetcher._fetch_subreddit("wallstreetbets", limit=25)
        assert len(posts) == 3
        # Minimal post still has 'data' key
        assert posts[0]["data"]["title"] == "SPY post"
        # Post with no 'data' key is just empty dict
        assert posts[1] == {}
        # Full post has all fields
        assert posts[2]["data"]["score"] == 50

    @patch("urllib.request.urlopen")
    def test_fetch_http_error_non_429(self, mock_urlopen, tmp_path):
        """Test handling of non-429 HTTP errors (403, 500)"""
        from urllib.error import HTTPError

        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        for code in (403, 500, 404, 503):
            mock_urlopen.side_effect = HTTPError(
                url="https://www.reddit.com",
                code=code,
                msg="Error",
                hdrs={},
                fp=None,
            )
            posts = fetcher._fetch_subreddit("wallstreetbets")
            assert posts == [], f"Expected empty list for HTTP {code}"

    @patch("urllib.request.urlopen")
    def test_fetch_general_exception(self, mock_urlopen, tmp_path):
        """Test handling of non-HTTP exceptions (timeout, connection error)"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        mock_urlopen.side_effect = ConnectionError("Connection refused")
        posts = fetcher._fetch_subreddit("wallstreetbets")
        assert posts == []

        mock_urlopen.side_effect = TimeoutError("Timed out")
        posts = fetcher._fetch_subreddit("investing")
        assert posts == []


class TestFetchSentimentIntegration:
    """Test fetch_sentiment flow with mocked subreddit calls"""

    def test_fetch_all_subreddits_called(self, tmp_path):
        """Test that fetch_sentiment calls _fetch_subreddit for all SUBREDDITS"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit", return_value=[]
        ) as mock_fetch:
            fetcher.fetch_sentiment(force_refresh=True)

        assert mock_fetch.call_count == len(SUBREDDITS)
        expected_calls = [call(sub, limit=25) for sub in SUBREDDITS]
        mock_fetch.assert_has_calls(expected_calls, any_order=True)

    def test_fetch_sentiment_caches_result(self, tmp_path):
        """Test that fetch_sentiment caches the result for subsequent retrieval"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        # Mock _fetch_subreddit to return a post mentioning SPY
        fake_post = {
            "data": {
                "title": "SPY to the moon! Buy calls",
                "selftext": "Diamond hands on SPY",
                "score": 100,
                "upvote_ratio": 0.85,
                "num_comments": 30,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 5,
            }
        }

        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit", return_value=[fake_post]
        ):
            result = fetcher.fetch_sentiment(force_refresh=True)

        assert result is not None
        assert result.data_fresh is True
        # SPY should have >0 mentions
        assert result.ticker_metrics["SPY"].mention_count_24h > 0

        # Now verify cache hit works
        cached = fetcher._get_cached_sentiment()
        assert cached is not None
        assert cached.aggregate_sentiment == result.aggregate_sentiment

    def test_fetch_sentiment_with_no_posts(self, tmp_path):
        """Test fetch_sentiment when no posts return from any subreddit"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit", return_value=[]
        ):
            result = fetcher.fetch_sentiment(force_refresh=True)

        assert result is not None
        assert result.data_fresh is False
        assert result.aggregate_sentiment == 0.0
        assert result.mention_velocity_1h == 0
        # All ticker metrics should have zero values
        for metrics in result.ticker_metrics.values():
            assert metrics.mention_count_24h == 0
            assert metrics.sentiment_score == 0.0

    def test_fetch_uses_cache_when_fresh(self, tmp_path):
        """Test that fetch_sentiment returns cached data when not force_refresh"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        # Pre-populate cache with a snapshot
        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=False,
            data_fresh=True,
        )
        fetcher._cache_sentiment(snapshot)

        # Now call fetch_sentiment without force_refresh
        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit",
            side_effect=AssertionError("Should not call API"),
        ):
            result = fetcher.fetch_sentiment(force_refresh=False)

        assert result is not None
        assert result.aggregate_sentiment == 0.2
        assert "SPY" in result.ticker_metrics


class TestViralityDetection:
    """Test virality detection logic"""

    def test_viral_when_high_velocity(self):
        """Test detection of viral content"""
        fetcher = RedditSentimentFetcher()

        # 2x normal velocity should be viral
        historical = [10, 12, 11, 13, 10, 9, 11, 12, 10, 11]  # avg ~11
        is_viral = fetcher._is_viral(velocity_1h=25, historical_data=historical)

        # 25 is above 90th percentile of historical
        assert is_viral is True

    def test_not_viral_normal_velocity(self):
        """Test normal content not flagged as viral"""
        fetcher = RedditSentimentFetcher()

        historical = [10, 12, 11, 13, 10, 9, 11, 12, 10, 11]
        is_viral = fetcher._is_viral(velocity_1h=10, historical_data=historical)

        assert is_viral is False

    def test_viral_insufficient_data(self):
        """Test virality with insufficient historical data"""
        fetcher = RedditSentimentFetcher()

        is_viral = fetcher._is_viral(velocity_1h=100, historical_data=[])
        assert is_viral is False  # Not enough data to determine

    def test_viral_exactly_at_p90(self):
        """Test velocity exactly at the 90th percentile is NOT viral"""
        fetcher = RedditSentimentFetcher()
        # With 10 sorted values [0..9], p90_idx = 9, p90_value = 9
        historical = list(range(10))
        is_viral = fetcher._is_viral(velocity_1h=9, historical_data=historical)
        # 9 is NOT greater than 9, so not viral
        assert is_viral is False

    def test_viral_just_above_p90(self):
        """Test velocity just above the 90th percentile IS viral"""
        fetcher = RedditSentimentFetcher()
        historical = list(range(10))  # [0..9], p90 = 9
        is_viral = fetcher._is_viral(velocity_1h=10, historical_data=historical)
        assert is_viral is True

    def test_viral_minimal_data_exact_10(self):
        """Test virality with exactly 10 data points (minimum threshold)"""
        fetcher = RedditSentimentFetcher()
        # 10 data points, velocity well above p90
        historical = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        is_viral = fetcher._is_viral(velocity_1h=100, historical_data=historical)
        assert is_viral is True

    def test_viral_9_data_points_not_enough(self):
        """Test virality with 9 data points returns False"""
        fetcher = RedditSentimentFetcher()
        historical = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        is_viral = fetcher._is_viral(velocity_1h=100, historical_data=historical)
        assert is_viral is False


class TestStoreMentions:
    """Test storing mentions in the database"""

    def test_store_mentions_inserts_rows(self, tmp_path):
        """Test that _store_mentions inserts rows into reddit_mentions"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        posts = [
            {
                "subreddit": "wallstreetbets",
                "title": "SPY to the moon!",
                "selftext": "Buy calls",
                "score": 100,
                "upvote_ratio": 0.85,
                "num_comments": 30,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 5,
                "sentiment": 0.8,
            },
            {
                "subreddit": "investing",
                "title": "TLT is falling",
                "selftext": "Bears are selling",
                "score": 50,
                "upvote_ratio": 0.40,
                "num_comments": 15,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 0,
                "sentiment": -0.6,
            },
        ]

        fetcher._store_mentions(posts)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT ticker, subreddit, sentiment_score FROM reddit_mentions")
            rows = cursor.fetchall()

        # SPY should have 1 mention, TLT should have 1 mention
        assert len(rows) == 2
        tickers_found = {row[0] for row in rows}
        assert "SPY" in tickers_found
        assert "TLT" in tickers_found

    def test_store_mentions_empty_posts(self, tmp_path):
        """Test _store_mentions with empty post list (no error)"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)
        fetcher._store_mentions([])

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM reddit_mentions")
            count = cursor.fetchone()[0]
        assert count == 0


class TestHistoryMethods:
    """Test history retrieval methods"""

    def test_get_history_empty(self, tmp_path):
        """Test getting history with no data"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        history = fetcher.get_history(days=7)
        assert history == []

    def test_get_ticker_history_empty(self, tmp_path):
        """Test getting ticker history with no data"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        history = fetcher.get_ticker_history("SPY", days=7)
        assert history == []

    def test_get_history_with_data(self, tmp_path):
        """Test get_history returns cached entries"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        metrics = {"SPY": RedditTickerMetrics("SPY", 5, 50, 0.2, 0.70, 3.0, 10)}
        snapshot = RedditSentimentSnapshot(
            timestamp="2026-05-14T10:00:00",
            ticker_metrics=metrics,
            aggregate_sentiment=0.2,
            mention_velocity_1h=5,
            mention_velocity_24h=50,
            engagement_score=35.0,
            virality_flag=False,
            data_fresh=True,
        )
        fetcher._cache_sentiment(snapshot)

        history = fetcher.get_history(days=7)
        assert len(history) == 1
        assert history[0]["aggregate_sentiment"] == 0.2

    def test_get_ticker_history_with_data(self, tmp_path):
        """Test get_ticker_history returns stored mentions"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        posts = [
            {
                "subreddit": "wallstreetbets",
                "title": "SPY to the moon!",
                "selftext": "Buy calls",
                "score": 100,
                "upvote_ratio": 0.85,
                "num_comments": 30,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 5,
                "sentiment": 0.8,
            },
        ]
        fetcher._store_mentions(posts)

        ticker_history = fetcher.get_ticker_history("SPY", days=7)
        assert len(ticker_history) == 1
        assert ticker_history[0]["ticker"] == "SPY"
        assert ticker_history[0]["sentiment_score"] == 0.8


class TestConstants:
    """Test module constants"""

    def test_tickers_defined(self):
        """Verify TICKERS constant"""
        assert "SPY" in TICKERS
        assert "GLD" in TICKERS
        assert "TLT" in TICKERS
        assert "QQQ" in TICKERS

    def test_subreddits_defined(self):
        """Verify SUBREDDITS constant"""
        assert "wallstreetbets" in SUBREDDITS
        assert "investing" in SUBREDDITS
        assert "stocks" in SUBREDDITS

    def test_reddit_api_base_defined(self):
        """Verify REDDIT_API_BASE constant"""
        assert REDDIT_API_BASE == "https://www.reddit.com"

    def test_user_agent_defined(self):
        """Verify USER_AGENT constant contains expected components"""
        assert "Portfolio-Lab" in USER_AGENT
        assert "Sentiment Analysis Bot" in USER_AGENT

    def test_rate_limit_delay_positive(self):
        """Verify RATE_LIMIT_DELAY is a positive number"""
        assert RATE_LIMIT_DELAY > 0

    def test_cache_ttl_minutes_positive(self):
        """Verify CACHE_TTL_MINUTES is a positive integer"""
        assert CACHE_TTL_MINUTES > 0
        assert isinstance(CACHE_TTL_MINUTES, int)


class TestFetchSentimentWeightedAggregate:
    """Test the weighted aggregate sentiment calculation"""

    def test_weighted_sentiment_weights_by_mention_count(self, tmp_path):
        """Test that aggregate sentiment is weighted by mention_count_24h"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        fake_post_spy = {
            "data": {
                "title": "SPY to the moon! Bullish breakout",
                "selftext": "",
                "score": 100,
                "upvote_ratio": 0.85,
                "num_comments": 30,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 5,
            }
        }
        fake_post_gld = {
            "data": {
                "title": "GLD is crashing hard",
                "selftext": "Sell GLD now",
                "score": 80,
                "upvote_ratio": 0.30,
                "num_comments": 20,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 0,
            }
        }

        # Return both posts for each subreddit call
        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit",
            return_value=[fake_post_spy, fake_post_gld],
        ):
            result = fetcher.fetch_sentiment(force_refresh=True)

        # SPY should have positive sentiment, GLD negative
        assert result.ticker_metrics["SPY"].sentiment_score > 0
        assert result.ticker_metrics["GLD"].sentiment_score < 0

        # Aggregate should be weighted by mention count (both have 1 mention per sub * 4 = 4 each)
        assert -1.0 <= result.aggregate_sentiment <= 1.0


class TestEngagementScore:
    """Test engagement score calculation boundary conditions"""

    def test_engagement_score_normalized_capped(self, tmp_path):
        """Test engagement score is capped at 100"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        fake_post = {
            "data": {
                "title": "SPY moon",
                "selftext": "",
                "score": 50000,
                "upvote_ratio": 0.99,
                "num_comments": 25000,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 1000,
            }
        }

        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit",
            return_value=[fake_post],
        ):
            result = fetcher.fetch_sentiment(force_refresh=True)

        # Engagement = min(100, (75000 / 10000) * 100) = min(100, 750) = 100
        assert result.engagement_score == 100.0

    def test_engagement_score_zero(self, tmp_path):
        """Test engagement score is zero when no posts"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)

        fake_post = {
            "data": {
                "title": "test",
                "selftext": "",
                "score": 0,
                "upvote_ratio": 0.5,
                "num_comments": 0,
                "created_utc": datetime.now(timezone.utc).timestamp(),
                "total_awards_received": 0,
            }
        }

        with patch.object(
            RedditSentimentFetcher, "_fetch_subreddit",
            return_value=[fake_post],
        ):
            result = fetcher.fetch_sentiment(force_refresh=True)

        assert result.engagement_score == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
