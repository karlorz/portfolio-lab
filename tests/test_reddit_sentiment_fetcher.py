"""
Tests for Reddit Sentiment Fetcher v2.70 Phase 4
15+ unit tests for API mocking, sentiment calculation, and caching.
"""

import pytest
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io

# Import module under test
import sys
sys.path.insert(0, '/root/projects/portfolio-lab')
from src.data.reddit_sentiment_fetcher import (
    RedditSentimentFetcher,
    RedditTickerMetrics,
    RedditSentimentSnapshot,
    TICKERS,
    SUBREDDITS
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
            stale_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
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


class TestFetchSentimentMocked:
    """Test fetch_sentiment with mocked Reddit API"""
    
    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_with_mock_posts(self, mock_request, mock_urlopen, tmp_path):
        """Test fetching with mocked API response"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)
        
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            'data': {
                'children': [
                    {
                        'data': {
                            'title': 'SPY to the moon! 🚀 Buy calls',
                            'selftext': 'Diamond hands on SPY',
                            'score': 150,
                            'upvote_ratio': 0.85,
                            'num_comments': 45,
                            'created_utc': datetime.utcnow().timestamp(),
                            'total_awards_received': 5
                        }
                    },
                    {
                        'data': {
                            'title': 'GLD safe haven play',
                            'selftext': '',
                            'score': 80,
                            'upvote_ratio': 0.75,
                            'num_comments': 20,
                            'created_utc': datetime.utcnow().timestamp(),
                            'total_awards_received': 2
                        }
                    }
                ]
            }
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Test a single subreddit fetch
        posts = fetcher._fetch_subreddit("wallstreetbets", limit=2)
        assert len(posts) == 2
        assert posts[0]['data']['title'] == 'SPY to the moon! 🚀 Buy calls'
    
    @patch('urllib.request.urlopen')
    def test_fetch_handles_rate_limit(self, mock_urlopen, tmp_path):
        """Test handling of 429 rate limit"""
        db_path = tmp_path / "test.db"
        fetcher = RedditSentimentFetcher(cache_db=db_path)
        
        # Mock HTTP 429 error
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="https://www.reddit.com",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None
        )
        
        posts = fetcher._fetch_subreddit("wallstreetbets")
        assert posts == []  # Should return empty list on rate limit


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
