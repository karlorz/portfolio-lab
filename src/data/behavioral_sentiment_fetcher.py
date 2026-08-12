"""
Behavioral Sentiment Data Fetcher for Portfolio-Lab v2.70
Fetches CBOE SKEW, VIX9D/VIX ratios, and retail flow indicators
for contrarian sentiment overlay strategy.

v2.70 Phase 4: Integrated Reddit Sentiment for real social data
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import math

import yfinance as yf
from src.utils.rate_limiter import rate_limited
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

from src.paths import MARKET_DB, sqlite_connect

# Import Reddit sentiment fetcher (v2.70 Phase 4)
try:
    from src.data.reddit_sentiment_fetcher import (
        RedditSentimentFetcher,
    )
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Reddit scraping gated off (HTTP 403 as of 2025+).
# Set REDDIT_ENABLED = True after implementing PRAW-based scraping.
REDDIT_ENABLED = False
_reddit_disabled_warned = False

# Setup logging
logger = logging.getLogger(__name__)

# Constants
CACHE_DB = MARKET_DB
CACHE_TTL_HOURS = 4

# CBOE Data URLs
CBOE_SKEW_URL = "https://www.cboe.com/us/indices/dashboard/skew/"
CBOE_VIX_URL = "https://www.cboe.com/tradable_products/vix/"

# Explicit network bound for every yfinance history() call (G6): a stalled
# provider must surface TimeoutError (→ documented fallback) instead of
# hanging the scheduled job past its deadline.
YF_FETCH_TIMEOUT_SECONDS = 10

# Sentiment thresholds
EXTREME_FEAR_THRESHOLD = -2.0
EXTREME_GREED_THRESHOLD = 2.0
FEAR_THRESHOLD = -1.0
GREED_THRESHOLD = 1.0


@dataclass
class OptionsSentiment:
    """Options market sentiment metrics"""
    timestamp: str
    skew_index: float  # CBOE SKEW (100 = normal)
    vix: float
    vix9d: float
    vix9d_ratio: float  # VIX9D/VIX (short-term vs medium-term)
    put_call_ratio: float  # CBOE Equity P/C ratio
    fear_greed_score: float  # Composite -3 to +3
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RetailFlow:
    """Retail positioning and flow metrics"""
    timestamp: str
    retail_call_put_ratio: float  # Small lot (<50 contracts)
    retail_buy_sell_imbalance: float  # -1 to +1 (buy vs sell)
    retail_top_100_correlation: float  # Robinhood top 100 inverse correlation
    small_lot_premium_ratio: float  # Retail premium spend vs institutional
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SocialIntensity:
    """Social media sentiment intensity - now includes Reddit data"""
    timestamp: str
    mention_velocity_7d: float  # 7d vs 30d rolling avg
    sentiment_divergence: float  # Bullish/bearish vs price momentum
    bot_activity_flag: bool  # Coordinated activity detected
    influencer_concentration: float  # % volume from high-follower accounts
    # Reddit-specific metrics (v2.70 Phase 4)
    reddit_sentiment: float = 0.0  # -1.0 to +1.0 aggregate
    reddit_mention_velocity_1h: float = 0.0  # Posts per hour
    reddit_mention_velocity_24h: float = 0.0  # Posts per day
    reddit_virality_flag: bool = False  # True if trending
    reddit_engagement_score: float = 0.0  # 0-100 composite
    reddit_data_source: str = "proxy"  # "reddit_api" or "proxy"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BehavioralSentimentSnapshot:
    """Complete behavioral sentiment snapshot"""
    timestamp: str
    options: OptionsSentiment
    retail: RetailFlow
    social: SocialIntensity
    composite_score: float  # -3 to +3, weighted aggregation
    signal_type: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    confidence: float  # 0-1 based on data quality
    data_fresh: bool
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'options': self.options.to_dict(),
            'retail': self.retail.to_dict(),
            'social': self.social.to_dict(),
            'composite_score': self.composite_score,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'data_fresh': self.data_fresh
        }


class BehavioralSentimentFetcher:
    """Fetches and caches behavioral sentiment data"""
    
    # Sentiment weights for composite score
    WEIGHTS = {
        'options': 0.35,
        'retail': 0.40,
        'social': 0.25
    }
    
    def __init__(self, cache_db: Path = CACHE_DB):
        self.cache_db = cache_db
        # Unified yfinance cache: {ticker: (value, fetch_time)}
        self._yf_cache: Dict[str, Tuple[float, datetime]] = {}
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache table"""
        with sqlite_connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    data TEXT,
                    composite_score REAL,
                    signal_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _get_cached(self) -> Optional[BehavioralSentimentSnapshot]:
        """Retrieve cached sentiment data"""
        try:
            with sqlite_connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT data, created_at FROM behavioral_sentiment_cache 
                       ORDER BY created_at DESC LIMIT 1"""
                )
                row = cursor.fetchone()
                if row:
                    cache_time = datetime.fromisoformat(row[1])
                    if cache_time.tzinfo is None:
                        cache_time = cache_time.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - cache_time
                    if age < timedelta(hours=CACHE_TTL_HOURS):
                        data = json.loads(row[0])
                        return self._dict_to_snapshot(data)
            return None
        except (sqlite3.Error, json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.warning("Cache retrieval failed: %s", e)
            return None
    
    def _save_to_cache(self, snapshot: BehavioralSentimentSnapshot):
        """Save sentiment data to cache"""
        try:
            with sqlite_connect(self.cache_db) as conn:
                conn.execute("""
                    INSERT INTO behavioral_sentiment_cache 
                    (timestamp, data, composite_score, signal_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    snapshot.timestamp,
                    json.dumps(snapshot.to_dict()),
                    snapshot.composite_score,
                    snapshot.signal_type
                ))
                # Keep only last 7 days
                conn.execute("""
                    DELETE FROM behavioral_sentiment_cache 
                    WHERE created_at < date('now', '-7 days')
                """)
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            logger.warning("Cache save failed: %s", e)
    
    def _dict_to_snapshot(self, data: Dict) -> BehavioralSentimentSnapshot:
        """Convert dict back to snapshot object"""
        return BehavioralSentimentSnapshot(
            timestamp=data['timestamp'],
            options=OptionsSentiment(**data['options']),
            retail=RetailFlow(**data['retail']),
            social=SocialIntensity(**data['social']),
            composite_score=data['composite_score'],
            signal_type=data['signal_type'],
            confidence=data['confidence'],
            data_fresh=data['data_fresh']
        )
    
    @rate_limited("yahoo")
    def _fetch_yf(self, ticker: str, period: str = "1d", default: float = 0.0,
                  ttl: float = 60.0) -> float:
        """Fetch a single value from yfinance with unified TTL cache.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. "^VIX", "^SKEW").
            period: yfinance history period (e.g. "1d", "5d").
            default: Fallback value if fetch fails.
            ttl: Cache TTL in seconds (default 60s).

        Returns:
            Most recent Close price for the ticker, or default.
        """
        now = datetime.now()
        cached = self._yf_cache.get(ticker)
        if cached is not None:
            value, fetch_time = cached
            age = (now - fetch_time).total_seconds()
            if age < ttl:
                logger.debug("Using cached yfinance data for %s (age=%.1fs)", ticker, age)
                return value

        try:
            hist = yf.Ticker(ticker).history(period=period, timeout=YF_FETCH_TIMEOUT_SECONDS)
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    raw = float(closes.iloc[-1])
                    if not math.isnan(raw):
                        self._yf_cache[ticker] = (raw, now)
                        return raw
        except (KeyError, ValueError, TypeError, OSError, RuntimeError, IndexError) as e:
            logger.warning("yfinance fetch failed for %s: %s", ticker, e)

        self._yf_cache[ticker] = (default, now)
        return default

    def _fetch_vix_data(self) -> Tuple[float, float]:
        """Fetch VIX and VIX9D from Yahoo Finance (cached via _fetch_yf)."""
        vix = self._fetch_yf("^VIX", default=16.0)
        vix9d = self._fetch_yf("^VIX9D", default=vix * 0.9)
        return (vix, vix9d)

    def _fetch_skew_index(self) -> float:
        """Fetch CBOE SKEW Index (cached via _fetch_yf)."""
        raw = self._fetch_yf("^SKEW", default=0.0)
        if raw > 0:
            return raw
        # Estimate SKEW from VIX if unavailable
        vix, _ = self._fetch_vix_data()
        return 100 + max(0, (vix - 15)) * 2

    def _fetch_put_call_ratio(self) -> float:
        """Fetch CBOE equity put/call ratio (5-day average, cached via _fetch_yf)."""
        now = datetime.now()
        cached = self._yf_cache.get("^CPCE")
        if cached is not None:
            value, fetch_time = cached
            age = (now - fetch_time).total_seconds()
            if age < 60:
                return value

        try:
            hist = yf.Ticker("^CPCE").history(period="5d", timeout=YF_FETCH_TIMEOUT_SECONDS)
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna().tolist()
                if closes:
                    avg = sum(closes) / len(closes)
                    self._yf_cache["^CPCE"] = (avg, now)
                    return avg
        except (KeyError, ValueError, TypeError, OSError, RuntimeError, IndexError) as e:
            logger.warning("yfinance fetch failed for ^CPCE: %s", e)

        self._yf_cache["^CPCE"] = (0.65, now)
        return 0.65  # Historical average
    
    def _estimate_retail_flow(self) -> RetailFlow:
        """Estimate retail positioning from available data"""
        try:
            # Use P/C ratio trends as proxy for retail sentiment
            current_pc = self._fetch_put_call_ratio()
            
            # Retail tends to buy calls more than institutions
            # High call/put ratio = retail optimism (contrarian signal)
            call_put_ratio = 1.0 / current_pc if current_pc > 0 else 1.0
            
            # Normalize to z-score-like metric
            # Historical avg CPCR ~0.65, retail heavy when < 0.60
            retail_call_bias = (0.65 - current_pc) * 10  # Scaled
            
            return RetailFlow(
                timestamp=datetime.now().isoformat(),
                retail_call_put_ratio=call_put_ratio,
                retail_buy_sell_imbalance=retail_call_bias,  # Proxy from options
                retail_top_100_correlation=-0.15,  # Typical inverse correlation
                small_lot_premium_ratio=0.85  # Estimated retail share
            )
        except (KeyError, ValueError, TypeError, ZeroDivisionError, RuntimeError) as e:
            logger.warning("Failed to estimate retail flow: %s", e)
            return RetailFlow(
                timestamp=datetime.now().isoformat(),
                retail_call_put_ratio=1.0,
                retail_buy_sell_imbalance=0.0,
                retail_top_100_correlation=-0.15,
                small_lot_premium_ratio=0.8
            )
    
    def _estimate_social_intensity(self) -> SocialIntensity:
        """
        Estimate social media intensity using Reddit data when available.
        Falls back to VIX-based proxy if Reddit is unavailable.
        """
        # Warn once if Reddit is available but disabled via REDDIT_ENABLED flag
        global _reddit_disabled_warned
        if REDDIT_AVAILABLE and not REDDIT_ENABLED and not _reddit_disabled_warned:
            _reddit_disabled_warned = True
            logger.info("Reddit scraping disabled (HTTP 403) — using VIX proxy for social intensity")

        # Try Reddit data first (v2.70 Phase 4)
        if REDDIT_AVAILABLE and REDDIT_ENABLED:
            try:
                reddit_fetcher = RedditSentimentFetcher(cache_db=self.cache_db)
                reddit_snapshot = reddit_fetcher.fetch_sentiment()
                
                # Convert Reddit sentiment to social intensity metrics
                # Reddit sentiment: -1 to +1, SocialIntensity divergence uses similar scale
                sentiment_divergence = reddit_snapshot.aggregate_sentiment
                
                # Virality flag indicates bot-like activity or coordinated campaigns
                bot_activity = reddit_snapshot.virality_flag
                
                # Engagement score maps to influencer concentration concept
                influencer_proxy = reddit_snapshot.engagement_score / 100.0
                
                return SocialIntensity(
                    timestamp=datetime.now().isoformat(),
                    mention_velocity_7d=reddit_snapshot.mention_velocity_24h / 24.0,  # Scale to hourly
                    sentiment_divergence=sentiment_divergence,
                    bot_activity_flag=bot_activity,
                    influencer_concentration=influencer_proxy,
                    # Reddit-specific fields
                    reddit_sentiment=reddit_snapshot.aggregate_sentiment,
                    reddit_mention_velocity_1h=reddit_snapshot.mention_velocity_1h,
                    reddit_mention_velocity_24h=reddit_snapshot.mention_velocity_24h,
                    reddit_virality_flag=reddit_snapshot.virality_flag,
                    reddit_engagement_score=reddit_snapshot.engagement_score,
                    reddit_data_source="reddit_api"
                )
            except (KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
                logger.warning("Reddit fetch failed, falling back to proxy: %s", e)
        
        # Fallback: VIX-based proxy estimation
        vix, vix9d = self._fetch_vix_data()
        
        # Estimate mention velocity from VIX level
        base_velocity = 1.0
        if vix > 25:
            base_velocity = 1.5
        elif vix < 15:
            base_velocity = 0.8
        
        # Sentiment divergence from price vs volatility
        sentiment_div = (vix9d - vix) / vix if vix > 0 else 0
        
        return SocialIntensity(
            timestamp=datetime.now().isoformat(),
            mention_velocity_7d=base_velocity,
            sentiment_divergence=sentiment_div,
            bot_activity_flag=vix > 30,  # Flag during high vol
            influencer_concentration=0.15,  # Typical
            reddit_data_source="proxy"
        )
    
    def _calculate_options_sentiment(self) -> OptionsSentiment:
        """Calculate options market sentiment"""
        vix, vix9d = self._fetch_vix_data()
        skew = self._fetch_skew_index()
        pc_ratio = self._fetch_put_call_ratio()
        
        # VIX9D/VIX ratio interpretation
        vix9d_ratio = vix9d / vix if vix > 0 else 1.0
        
        # SKEW interpretation: >140 = tail risk bid (fear), <115 = complacency
        skew_fear = (skew - 100) / 40 * 0.3
        
        # VIX9D/VIX interpretation: >1.1 = near-term anxiety
        vix_ratio_anxiety = (vix9d_ratio - 1.0) * 0.4
        
        # Put/call interpretation: >0.8 = fear, <0.5 = greed
        pc_fear = (0.65 - pc_ratio) * 2.0 * 0.3  # Normalized around 0.65
        
        # Composite fear/greed score (-3 to +3)
        fear_greed = skew_fear + vix_ratio_anxiety + pc_fear
        fear_greed = max(-3, min(3, fear_greed))
        
        return OptionsSentiment(
            timestamp=datetime.now().isoformat(),
            skew_index=skew,
            vix=vix,
            vix9d=vix9d,
            vix9d_ratio=vix9d_ratio,
            put_call_ratio=pc_ratio,
            fear_greed_score=fear_greed
        )
    
    def _calculate_composite_score(
        self,
        options: OptionsSentiment,
        retail: RetailFlow,
        social: SocialIntensity
    ) -> Tuple[float, str, float]:
        """Calculate composite sentiment score and signal type"""
        # Options component
        options_score = options.fear_greed_score
        
        # Retail component (invert - retail optimism = contrarian fear)
        retail_score = -retail.retail_buy_sell_imbalance * 2
        
        # Social component
        social_score = social.sentiment_divergence * 3
        if social.bot_activity_flag:
            social_score += 0.5  # Elevated caution
        
        # Weighted composite
        composite = (
            options_score * self.WEIGHTS['options'] +
            retail_score * self.WEIGHTS['retail'] +
            social_score * self.WEIGHTS['social']
        )
        
        # Clamp to valid range
        composite = max(-3, min(3, composite))
        
        # Determine signal type
        if composite <= EXTREME_FEAR_THRESHOLD:
            signal_type = 'extreme_fear'
        elif composite <= FEAR_THRESHOLD:
            signal_type = 'fear'
        elif composite >= EXTREME_GREED_THRESHOLD:
            signal_type = 'extreme_greed'
        elif composite >= GREED_THRESHOLD:
            signal_type = 'greed'
        else:
            signal_type = 'neutral'
        
        # Confidence based on data freshness
        confidence = 0.7 if options.vix > 0 else 0.5
        
        return composite, signal_type, confidence
    
    def fetch_snapshot(self, use_cache: bool = True) -> BehavioralSentimentSnapshot:
        """Fetch complete behavioral sentiment snapshot"""
        if use_cache:
            cached = self._get_cached()
            if cached is not None:
                logger.info("Using cached behavioral sentiment data")
                return cached
        
        logger.info("Fetching fresh behavioral sentiment data...")
        
        # Fetch all components
        options = self._calculate_options_sentiment()
        retail = self._estimate_retail_flow()
        social = self._estimate_social_intensity()
        
        # Calculate composite
        composite, signal_type, confidence = self._calculate_composite_score(
            options, retail, social
        )
        
        snapshot = BehavioralSentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            options=options,
            retail=retail,
            social=social,
            composite_score=composite,
            signal_type=signal_type,
            confidence=confidence,
            data_fresh=True
        )
        
        # Save to cache
        self._save_to_cache(snapshot)
        
        return snapshot
    
    def get_signal_recommendation(self, snapshot: Optional[BehavioralSentimentSnapshot] = None) -> Dict:
        """Get allocation recommendation from sentiment signal"""
        if snapshot is None:
            snapshot = self.fetch_snapshot()
        
        # Contrarian allocation shifts based on sentiment extremes
        recommendation = {
            'timestamp': snapshot.timestamp,
            'signal_type': snapshot.signal_type,
            'composite_score': round(snapshot.composite_score, 2),
            'confidence': round(snapshot.confidence, 2),
            'recommended_action': 'neutral',
            'equity_shift_pct': 0.0,
            'rationale': ''
        }
        
        if snapshot.signal_type == 'extreme_fear' and snapshot.confidence > 0.5:
            recommendation['recommended_action'] = 'contrarian_buy'
            recommendation['equity_shift_pct'] = 5.0
            recommendation['rationale'] = (
                f"Extreme fear detected (score: {snapshot.composite_score:.1f}). "
                f"Retail positioning and options metrics show capitulation. "
                f"Contrarian equity increase recommended."
            )
        elif snapshot.signal_type == 'fear' and snapshot.confidence > 0.5:
            recommendation['recommended_action'] = 'moderate_buy'
            recommendation['equity_shift_pct'] = 3.0
            recommendation['rationale'] = (
                "Elevated fear detected. Moderate contrarian positioning."
            )
        elif snapshot.signal_type == 'extreme_greed' and snapshot.confidence > 0.5:
            recommendation['recommended_action'] = 'contrarian_sell'
            recommendation['equity_shift_pct'] = -5.0
            recommendation['rationale'] = (
                f"Extreme greed detected (score: {snapshot.composite_score:.1f}). "
                f"Crowd euphoria suggests caution. Reduce equity exposure."
            )
        elif snapshot.signal_type == 'greed' and snapshot.confidence > 0.5:
            recommendation['recommended_action'] = 'moderate_sell'
            recommendation['equity_shift_pct'] = -3.0
            recommendation['rationale'] = (
                "Elevated greed detected. Moderate defensive positioning."
            )
        else:
            recommendation['rationale'] = (
                "Neutral sentiment regime. No behavioral overlay recommended."
            )
        
        return recommendation
    
    def get_historical_sentiment(self, days: int = 30) -> List[Dict]:
        """Retrieve historical sentiment data"""
        try:
            with sqlite_connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT data, created_at FROM behavioral_sentiment_cache
                       WHERE created_at >= date('now', ?)
                       ORDER BY created_at DESC""", (f'-{days} days',)
                )
                rows = cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
        except (sqlite3.Error, json.JSONDecodeError, KeyError, ValueError, OSError, RuntimeError) as e:
            logger.warning("Failed to retrieve history: %s", e)
            return []


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral Sentiment Fetcher')
    parser.add_argument('--fetch', action='store_true', help='Fetch current sentiment')
    parser.add_argument('--recommend', action='store_true', help='Get recommendation')
    parser.add_argument('--history', type=int, help='Get N days of history')
    
    args = parser.parse_args()
    
    fetcher = BehavioralSentimentFetcher()
    
    if args.fetch or (not args.recommend and not args.history):
        snapshot = fetcher.fetch_snapshot()
        logger.info("=== Behavioral Sentiment Snapshot ===")
        logger.info("Timestamp: %s", snapshot.timestamp)
        logger.info("Composite Score: %.2f (-3 fear to +3 greed)", snapshot.composite_score)
        logger.info("Signal Type: %s", snapshot.signal_type)
        logger.info("Confidence: %.1f%%", snapshot.confidence * 100)
        logger.info("--- Options Sentiment ---")
        logger.info("  SKEW Index: %.1f", snapshot.options.skew_index)
        logger.info("  VIX: %.2f", snapshot.options.vix)
        logger.info("  VIX9D/VIX: %.2f", snapshot.options.vix9d_ratio)
        logger.info("  P/C Ratio: %.2f", snapshot.options.put_call_ratio)
        logger.info("  Fear/Greed Score: %.2f", snapshot.options.fear_greed_score)
        logger.info("--- Retail Flow ---")
        logger.info("  Call/Put Ratio: %.2f", snapshot.retail.retail_call_put_ratio)
        logger.info("  Buy/Sell Imbalance: %.2f", snapshot.retail.retail_buy_sell_imbalance)
        logger.info("--- Social Intensity ---")
        logger.info("  Mention Velocity: %.2f", snapshot.social.mention_velocity_7d)
        logger.info("  Sentiment Divergence: %.2f", snapshot.social.sentiment_divergence)
        logger.info("  Bot Activity: %s", snapshot.social.bot_activity_flag)
    
    if args.recommend:
        rec = fetcher.get_signal_recommendation()
        logger.info("=== Allocation Recommendation ===")
        logger.info("Action: %s", rec['recommended_action'])
        logger.info("Equity Shift: %.1f%%", rec['equity_shift_pct'])
        logger.info("Rationale: %s", rec['rationale'])
    
    if args.history:
        history = fetcher.get_historical_sentiment(args.history)
        logger.info("=== Last %s Sentiment Records ===", len(history))
        for h in history[:5]:
            logger.info("  %s | Score: %+.2f | %s", h['timestamp'][:19], h['composite_score'], h['signal_type'])
