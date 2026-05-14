"""
Reddit Sentiment Data Fetcher for Portfolio-Lab v2.70 Phase 4
Fetches retail sentiment from Reddit (r/wallstreetbets, r/investing, r/stocks)
for contrarian sentiment overlay strategy.

This module provides real social media sentiment as the third pillar of the
behavioral sentiment overlay, replacing the volatility regime proxy used in Phases 1-3.
"""

import sqlite3
import json
import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_DB = Path("/root/projects/portfolio-lab/data/market.db")
CACHE_TTL_MINUTES = 15
RATE_LIMIT_DELAY = 1.0  # seconds between API calls

# Reddit API Configuration
REDDIT_API_BASE = "https://www.reddit.com"
SUBREDDITS = ["wallstreetbets", "investing", "stocks", "options"]
TICKERS = ["SPY", "GLD", "TLT", "QQQ", "IEF", "VIX"]

# User-Agent required by Reddit API
USER_AGENT = "Mozilla/5.0 (compatible; Portfolio-Lab/2.70; Sentiment Analysis Bot)"


@dataclass
class RedditTickerMetrics:
    """Per-ticker sentiment metrics from Reddit"""
    ticker: str
    mention_count_1h: int
    mention_count_24h: int
    sentiment_score: float  # -1.0 to +1.0 aggregate
    upvote_ratio: float  # 0.0 to 1.0
    comment_velocity: float  # comments per hour
    award_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RedditSentimentSnapshot:
    """Complete Reddit sentiment snapshot"""
    timestamp: str
    ticker_metrics: Dict[str, RedditTickerMetrics]
    aggregate_sentiment: float  # -1.0 to +1.0
    mention_velocity_1h: float  # total posts per hour
    mention_velocity_24h: float  # total posts per day
    engagement_score: float  # 0-100 composite
    virality_flag: bool  # True if growing fast (>90th percentile)
    data_fresh: bool
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'ticker_metrics': {k: v.to_dict() for k, v in self.ticker_metrics.items()},
            'aggregate_sentiment': self.aggregate_sentiment,
            'mention_velocity_1h': self.mention_velocity_1h,
            'mention_velocity_24h': self.mention_velocity_24h,
            'engagement_score': self.engagement_score,
            'virality_flag': self.virality_flag,
            'data_fresh': self.data_fresh
        }


class RedditSentimentFetcher:
    """
    Reddit sentiment data fetcher with SQLite caching.
    
    Uses Reddit's JSON API (no authentication required for read-only).
    Respects rate limits with 60 requests/minute free tier.
    """
    
    def __init__(self, cache_db: Path = CACHE_DB):
        self.cache_db = cache_db
        self._init_db()
        self.last_request_time = 0.0
    
    def _init_db(self):
        """Initialize SQLite cache tables"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reddit_sentiment_cache (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reddit_mentions (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    subreddit TEXT NOT NULL,
                    post_title TEXT,
                    sentiment_score REAL,
                    upvotes INTEGER,
                    comment_count INTEGER,
                    created_utc TIMESTAMP,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_mentions_ticker 
                ON reddit_mentions(ticker, fetched_at)
            """)
            conn.commit()
    
    def _rate_limit(self):
        """Respect Reddit API rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _fetch_subreddit(self, subreddit: str, limit: int = 25) -> List[Dict]:
        """
        Fetch recent posts from a subreddit.
        
        Returns list of post dictionaries with title, score, etc.
        """
        import urllib.request
        import urllib.error
        
        url = f"{REDDIT_API_BASE}/r/{subreddit}/hot.json?limit={limit}"
        
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        }
        
        try:
            self._rate_limit()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('data', {}).get('children', [])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"Rate limited on r/{subreddit}, using cache")
                return []
            logger.error(f"HTTP error fetching r/{subreddit}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching r/{subreddit}: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text using regex"""
        tickers_found = []
        for ticker in TICKERS:
            # Match $TICKER or standalone TICKER (case insensitive)
            pattern = rf'\$?\b{ticker}\b'
            if re.search(pattern, text, re.IGNORECASE):
                tickers_found.append(ticker)
        return tickers_found
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Simple VADER-like sentiment calculation.
        Returns score from -1.0 (negative) to +1.0 (positive).
        """
        text_lower = text.lower()
        
        # Positive words (financial context)
        positive_words = {
            'bull', 'bullish', 'moon', 'rocket', 'tendies', 'gain', 'gains',
            'profit', 'profits', 'winning', 'win', 'up', 'rise', 'rising',
            'surge', 'surging', 'breakout', ' ATH', 'all time high',
            'calls', 'call', 'long', 'buy', 'buying', 'bought', 'hold',
            'hodl', 'diamond hands', 'strong', 'green', 'pump'
        }
        
        # Negative words (financial context)
        negative_words = {
            'bear', 'bearish', 'crash', 'dump', 'dumping', 'loss', 'losses',
            'losing', 'lose', 'down', 'fall', 'falling', 'tank', 'tanking',
            'puts', 'put', 'short', 'shorting', 'sell', 'selling', 'sold',
            'paper hands', 'weak', 'red', 'bleeding', 'rugpull', 'scam',
            'bankrupt', 'liquidation', 'margin call'
        }
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        # Normalize to -1 to +1 range
        return (pos_count - neg_count) / total
    
    def _is_viral(self, velocity_1h: float, historical_data: List[float]) -> bool:
        """Determine if current velocity is viral (>90th percentile)"""
        if not historical_data or len(historical_data) < 10:
            return False
        
        sorted_hist = sorted(historical_data)
        p90_idx = int(len(sorted_hist) * 0.9)
        p90_value = sorted_hist[min(p90_idx, len(sorted_hist) - 1)]
        
        return velocity_1h > p90_value
    
    def fetch_sentiment(self, force_refresh: bool = False) -> RedditSentimentSnapshot:
        """
        Fetch current Reddit sentiment snapshot.
        
        Uses cache if data is fresh (<15 minutes old).
        """
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_sentiment()
            if cached:
                logger.info("Using cached Reddit sentiment")
                return cached
        
        # Fetch from Reddit API
        logger.info("Fetching fresh Reddit sentiment data...")
        
        all_posts = []
        for subreddit in SUBREDDITS:
            posts = self._fetch_subreddit(subreddit, limit=25)
            for post in posts:
                post_data = post.get('data', {})
                all_posts.append({
                    'subreddit': subreddit,
                    'title': post_data.get('title', ''),
                    'selftext': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0),
                    'total_awards': post_data.get('total_awards_received', 0)
                })
        
        # Process posts
        ticker_mentions: Dict[str, List[Dict]] = {t: [] for t in TICKERS}
        total_sentiment = 0.0
        total_engagement = 0
        
        for post in all_posts:
            text = f"{post['title']} {post['selftext']}"
            tickers_in_post = self._extract_tickers(text)
            
            sentiment = self._calculate_sentiment(text)
            post['sentiment'] = sentiment
            
            for ticker in tickers_in_post:
                ticker_mentions[ticker].append(post)
            
            total_sentiment += sentiment
            total_engagement += post['score'] + post['num_comments']
        
        # Calculate ticker metrics
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        ticker_metrics = {}
        for ticker in TICKERS:
            posts = ticker_mentions[ticker]
            
            # Time-bucketed counts
            mentions_1h = sum(1 for p in posts 
                            if datetime.utcfromtimestamp(p['created_utc']) > one_hour_ago)
            mentions_24h = len(posts)
            
            # Aggregate sentiment
            if posts:
                avg_sentiment = sum(p['sentiment'] for p in posts) / len(posts)
                avg_upvote = sum(p['upvote_ratio'] for p in posts) / len(posts)
                total_comments = sum(p['num_comments'] for p in posts)
                total_awards = sum(p['total_awards'] for p in posts)
                
                # Comment velocity (comments per hour in last 24h)
                comment_velocity = total_comments / 24.0 if mentions_24h > 0 else 0.0
            else:
                avg_sentiment = 0.0
                avg_upvote = 0.5
                comment_velocity = 0.0
                total_awards = 0
            
            ticker_metrics[ticker] = RedditTickerMetrics(
                ticker=ticker,
                mention_count_1h=mentions_1h,
                mention_count_24h=mentions_24h,
                sentiment_score=avg_sentiment,
                upvote_ratio=avg_upvote,
                comment_velocity=comment_velocity,
                award_count=total_awards
            )
        
        # Aggregate metrics
        total_mentions_1h = sum(m.mention_count_1h for m in ticker_metrics.values())
        total_mentions_24h = sum(m.mention_count_24h for m in ticker_metrics.values())
        
        # Weighted aggregate sentiment (weighted by mention count)
        weighted_sentiment = sum(
            m.sentiment_score * m.mention_count_24h 
            for m in ticker_metrics.values()
        ) / max(total_mentions_24h, 1)
        
        # Engagement score (0-100 based on upvotes, comments, awards)
        max_engagement = 10000  # Normalization factor
        engagement_score = min(100, (total_engagement / max_engagement) * 100)
        
        # Virality check (would need historical data, simplified)
        virality = total_mentions_1h > (total_mentions_24h / 24) * 2  # 2x hourly average
        
        snapshot = RedditSentimentSnapshot(
            timestamp=now.isoformat(),
            ticker_metrics=ticker_metrics,
            aggregate_sentiment=weighted_sentiment,
            mention_velocity_1h=total_mentions_1h,
            mention_velocity_24h=total_mentions_24h,
            engagement_score=engagement_score,
            virality_flag=virality,
            data_fresh=len(all_posts) > 0
        )
        
        # Cache the result
        self._cache_sentiment(snapshot)
        
        # Store mentions for historical analysis
        self._store_mentions(all_posts)
        
        return snapshot
    
    def _get_cached_sentiment(self) -> Optional[RedditSentimentSnapshot]:
        """Get cached sentiment if fresh"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("""
                    SELECT data_json, created_at 
                    FROM reddit_sentiment_cache 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    data_json, created_at = row
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00').replace('+00:00', ''))
                    
                    # Check freshness
                    if datetime.utcnow() - created_time < timedelta(minutes=CACHE_TTL_MINUTES):
                        data = json.loads(data_json)
                        
                        # Reconstruct ticker metrics
                        ticker_metrics = {
                            k: RedditTickerMetrics(**v)
                            for k, v in data.get('ticker_metrics', {}).items()
                        }
                        
                        return RedditSentimentSnapshot(
                            timestamp=data['timestamp'],
                            ticker_metrics=ticker_metrics,
                            aggregate_sentiment=data['aggregate_sentiment'],
                            mention_velocity_1h=data['mention_velocity_1h'],
                            mention_velocity_24h=data['mention_velocity_24h'],
                            engagement_score=data['engagement_score'],
                            virality_flag=data['virality_flag'],
                            data_fresh=True
                        )
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
        
        return None
    
    def _cache_sentiment(self, snapshot: RedditSentimentSnapshot):
        """Cache sentiment snapshot"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    INSERT INTO reddit_sentiment_cache (timestamp, data_json)
                    VALUES (?, ?)
                """, (snapshot.timestamp, json.dumps(snapshot.to_dict())))
                
                # Keep only last 100 entries
                conn.execute("""
                    DELETE FROM reddit_sentiment_cache 
                    WHERE id NOT IN (
                        SELECT id FROM reddit_sentiment_cache 
                        ORDER BY created_at DESC 
                        LIMIT 100
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching sentiment: {e}")
    
    def _store_mentions(self, posts: List[Dict]):
        """Store individual mentions for historical analysis"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                for post in posts:
                    text = f"{post['title']} {post['selftext']}"
                    tickers = self._extract_tickers(text)
                    
                    for ticker in tickers:
                        conn.execute("""
                            INSERT INTO reddit_mentions 
                            (ticker, subreddit, post_title, sentiment_score, 
                             upvotes, comment_count, created_utc)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ticker,
                            post['subreddit'],
                            post['title'][:200],  # Truncate
                            post.get('sentiment', 0.0),
                            post['score'],
                            post['num_comments'],
                            datetime.utcfromtimestamp(post['created_utc']).isoformat()
                        ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing mentions: {e}")
    
    def get_history(self, days: int = 7) -> List[Dict]:
        """Get historical sentiment data"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, data_json 
                    FROM reddit_sentiment_cache 
                    WHERE created_at > datetime('now', '-{} days')
                    ORDER BY created_at DESC
                """.format(days))
                
                history = []
                for row in cursor.fetchall():
                    data = json.loads(row[1])
                    data['timestamp'] = row[0]
                    history.append(data)
                
                return history
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []
    
    def get_ticker_history(self, ticker: str, days: int = 7) -> List[Dict]:
        """Get historical mentions for a specific ticker"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("""
                    SELECT ticker, subreddit, post_title, sentiment_score,
                           upvotes, comment_count, created_utc, fetched_at
                    FROM reddit_mentions 
                    WHERE ticker = ? 
                    AND fetched_at > datetime('now', '-{} days')
                    ORDER BY fetched_at DESC
                """.format(days), (ticker,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching ticker history: {e}")
            return []


def main():
    """CLI interface for Reddit sentiment fetcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reddit Sentiment Fetcher')
    parser.add_argument('--fetch', action='store_true', help='Fetch current sentiment')
    parser.add_argument('--history', type=int, metavar='N', help='Show N-day history')
    parser.add_argument('--ticker', type=str, help='Filter by ticker (e.g., SPY)')
    parser.add_argument('--force', action='store_true', help='Force refresh (ignore cache)')
    
    args = parser.parse_args()
    
    fetcher = RedditSentimentFetcher()
    
    if args.fetch or (not args.history and not args.ticker):
        snapshot = fetcher.fetch_sentiment(force_refresh=args.force)
        
        print("\n=== Reddit Sentiment Snapshot ===")
        print(f"Timestamp: {snapshot.timestamp}")
        print(f"Data Fresh: {snapshot.data_fresh}")
        print(f"\nAggregate Sentiment: {snapshot.aggregate_sentiment:+.3f} (-1.0 bearish to +1.0 bullish)")
        print(f"Mention Velocity (1h): {snapshot.mention_velocity_1h} posts/hour")
        print(f"Mention Velocity (24h): {snapshot.mention_velocity_24h} posts/day")
        print(f"Engagement Score: {snapshot.engagement_score:.1f}/100")
        print(f"Virality Flag: {'🔥 VIRAL' if snapshot.virality_flag else 'Normal'}")
        
        print("\n--- Per-Ticker Metrics ---")
        for ticker, metrics in snapshot.ticker_metrics.items():
            if metrics.mention_count_24h > 0:
                print(f"\n{ticker}:")
                print(f"  Mentions (1h/24h): {metrics.mention_count_1h}/{metrics.mention_count_24h}")
                print(f"  Sentiment: {metrics.sentiment_score:+.3f}")
                print(f"  Upvote Ratio: {metrics.upvote_ratio:.2%}")
                print(f"  Comment Velocity: {metrics.comment_velocity:.1f}/hour")
                print(f"  Awards: {metrics.award_count}")
    
    if args.history:
        history = fetcher.get_history(days=args.history)
        print(f"\n=== {args.history}-Day History ({len(history)} snapshots) ===")
        
        for entry in history[:10]:  # Show last 10
            print(f"\n{entry['timestamp'][:19]}:")
            print(f"  Sentiment: {entry['aggregate_sentiment']:+.3f}")
            print(f"  Mentions (24h): {entry['mention_velocity_24h']}")
            print(f"  Virality: {'Yes' if entry['virality_flag'] else 'No'}")
    
    if args.ticker:
        ticker_history = fetcher.get_ticker_history(args.ticker, days=args.history or 7)
        print(f"\n=== {args.ticker} Mention History ({len(ticker_history)} posts) ===")
        
        for post in ticker_history[:10]:
            print(f"\n{post['fetched_at'][:19]} | r/{post['subreddit']}")
            print(f"  Title: {post['post_title'][:80]}...")
            print(f"  Sentiment: {post['sentiment_score']:+.3f} | Upvotes: {post['upvotes']}")


if __name__ == '__main__':
    main()
