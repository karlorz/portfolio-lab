"""
Behavioral Sentiment Data Fetcher for Portfolio-Lab v2.70
Fetches CBOE SKEW, VIX9D/VIX ratios, and retail flow indicators
for contrarian sentiment overlay strategy.

v2.70 Phase 4: Integrated Reddit Sentiment for real social data
"""

import sqlite3
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

# Import Reddit sentiment fetcher (v2.70 Phase 4)
try:
    from src.data.reddit_sentiment_fetcher import (
        RedditSentimentFetcher,
        RedditSentimentSnapshot,
    )
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_DB = Path("/root/projects/portfolio-lab/data/market.db")
CACHE_TTL_HOURS = 4

# CBOE Data URLs
CBOE_SKEW_URL = "https://www.cboe.com/us/indices/dashboard/skew/"
CBOE_VIX_URL = "https://www.cboe.com/tradable_products/vix/"

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
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache table"""
        with sqlite3.connect(self.cache_db) as conn:
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
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT data, created_at FROM behavioral_sentiment_cache 
                       ORDER BY created_at DESC LIMIT 1"""
                )
                row = cursor.fetchone()
                if row:
                    cache_time = datetime.fromisoformat(row[1])
                    age = datetime.now() - cache_time
                    if age < timedelta(hours=CACHE_TTL_HOURS):
                        data = json.loads(row[0])
                        return self._dict_to_snapshot(data)
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _save_to_cache(self, snapshot: BehavioralSentimentSnapshot):
        """Save sentiment data to cache"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
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
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
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
    
    def _fetch_vix_data(self) -> Tuple[float, float]:
        """Fetch VIX and VIX9D from Yahoo Finance API"""
        try:
            # Yahoo Finance API for VIX (^VIX)
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX?interval=1d&range=1d"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                vix = result['meta']['regularMarketPrice']
            else:
                vix = 16.0  # Fallback
            
            # VIX9D (^VIX9D) - short-term VIX
            url9d = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX9D?interval=1d&range=1d"
            response9d = requests.get(url9d, timeout=10)
            data9d = response9d.json()
            
            if 'chart' in data9d and 'result' in data9d['chart'] and data9d['chart']['result']:
                result9d = data9d['chart']['result'][0]
                vix9d = result9d['meta']['regularMarketPrice']
            else:
                vix9d = vix * 0.9  # Estimate as 90% of VIX
            
            return float(vix), float(vix9d)
        except Exception as e:
            logger.warning(f"Failed to fetch VIX data: {e}")
            return 16.0, 14.4  # Default values
    
    def _fetch_skew_index(self) -> float:
        """Fetch CBOE SKEW Index (synthetic from options data)"""
        try:
            # Yahoo Finance for SKEW (^SKEW)
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^SKEW?interval=1d&range=1d"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                skew = result['meta']['regularMarketPrice']
                return float(skew)
        except Exception as e:
            logger.warning(f"Failed to fetch SKEW: {e}")
        
        # Estimate SKEW from VIX if unavailable
        vix, _ = self._fetch_vix_data()
        # SKEW approx 100 + (VIX - 15) * 2
        return 100 + max(0, (vix - 15)) * 2
    
    def _fetch_put_call_ratio(self) -> float:
        """Fetch CBOE equity put/call ratio"""
        try:
            # CBOE publishes daily P/C ratio
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^CPCE?interval=1d&range=5d"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                if 'close' in result['indicators']['quote'][0]:
                    closes = result['indicators']['quote'][0]['close']
                    # Filter None values
                    closes = [c for c in closes if c is not None]
                    if closes:
                        return sum(closes) / len(closes)
        except Exception as e:
            logger.warning(f"Failed to fetch P/C ratio: {e}")
        
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
        except Exception as e:
            logger.warning(f"Failed to estimate retail flow: {e}")
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
        # Try Reddit data first (v2.70 Phase 4)
        if REDDIT_AVAILABLE:
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
            except Exception as e:
                logger.warning(f"Reddit fetch failed, falling back to proxy: {e}")
        
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
                f"Elevated fear detected. Moderate contrarian positioning."
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
                f"Elevated greed detected. Moderate defensive positioning."
            )
        else:
            recommendation['rationale'] = (
                f"Neutral sentiment regime. No behavioral overlay recommended."
            )
        
        return recommendation
    
    def get_historical_sentiment(self, days: int = 30) -> List[Dict]:
        """Retrieve historical sentiment data"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """SELECT data, created_at FROM behavioral_sentiment_cache 
                       WHERE created_at >= date('now', '-{} days')
                       ORDER BY created_at DESC""".format(days)
                )
                rows = cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
        except Exception as e:
            logger.warning(f"Failed to retrieve history: {e}")
            return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral Sentiment Fetcher')
    parser.add_argument('--fetch', action='store_true', help='Fetch current sentiment')
    parser.add_argument('--recommend', action='store_true', help='Get recommendation')
    parser.add_argument('--history', type=int, help='Get N days of history')
    
    args = parser.parse_args()
    
    fetcher = BehavioralSentimentFetcher()
    
    if args.fetch or (not args.recommend and not args.history):
        snapshot = fetcher.fetch_snapshot()
        print("\n=== Behavioral Sentiment Snapshot ===")
        print(f"Timestamp: {snapshot.timestamp}")
        print(f"\nComposite Score: {snapshot.composite_score:.2f} (-3 fear to +3 greed)")
        print(f"Signal Type: {snapshot.signal_type}")
        print(f"Confidence: {snapshot.confidence:.1%}")
        print(f"\n--- Options Sentiment ---")
        print(f"  SKEW Index: {snapshot.options.skew_index:.1f}")
        print(f"  VIX: {snapshot.options.vix:.2f}")
        print(f"  VIX9D/VIX: {snapshot.options.vix9d_ratio:.2f}")
        print(f"  P/C Ratio: {snapshot.options.put_call_ratio:.2f}")
        print(f"  Fear/Greed Score: {snapshot.options.fear_greed_score:.2f}")
        print(f"\n--- Retail Flow ---")
        print(f"  Call/Put Ratio: {snapshot.retail.retail_call_put_ratio:.2f}")
        print(f"  Buy/Sell Imbalance: {snapshot.retail.retail_buy_sell_imbalance:.2f}")
        print(f"\n--- Social Intensity ---")
        print(f"  Mention Velocity: {snapshot.social.mention_velocity_7d:.2f}")
        print(f"  Sentiment Divergence: {snapshot.social.sentiment_divergence:.2f}")
        print(f"  Bot Activity: {snapshot.social.bot_activity_flag}")
    
    if args.recommend:
        rec = fetcher.get_signal_recommendation()
        print("\n=== Allocation Recommendation ===")
        print(f"Action: {rec['recommended_action']}")
        print(f"Equity Shift: {rec['equity_shift_pct']:.1f}%")
        print(f"Rationale: {rec['rationale']}")
    
    if args.history:
        history = fetcher.get_historical_sentiment(args.history)
        print(f"\n=== Last {len(history)} Sentiment Records ===")
        for h in history[:5]:
            print(f"  {h['timestamp'][:19]} | Score: {h['composite_score']:+.2f} | {h['signal_type']}")
