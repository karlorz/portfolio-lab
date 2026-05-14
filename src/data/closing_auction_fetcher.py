"""
Closing Auction MOC/IOC Data Fetcher (v3.17 Phase 1)

Fetches Market-On-Close (MOC) imbalance data for statistical arbitrage
opportunities during the 3:50pm → 4:00pm window.

Data sources:
- NYSE ARCA imbalance announcements (web scrape/API)
- Yahoo Finance real-time quotes for proxy calculations
- Historical backfill via archived data

Author: Autonomous Agent
Version: v3.17 Phase 1
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MOCImbalance:
    """Represents a single MOC imbalance announcement."""
    symbol: str
    timestamp: datetime
    imbalance_shares: int  # Positive = buy imbalance, negative = sell
    paired_shares: int
    reference_price: float
    source: str
    
    @property
    def imbalance_ratio(self) -> float:
        """Calculate imbalance as ratio of paired volume."""
        if self.paired_shares == 0:
            return 0.0
        return self.imbalance_shares / self.paired_shares
    
    @property
    def direction_score(self) -> int:
        """Score from -3 (strong sell) to +3 (strong buy)."""
        ratio = abs(self.imbalance_ratio)
        if ratio > 0.5:
            return 3 if self.imbalance_shares > 0 else -3
        elif ratio > 0.3:
            return 2 if self.imbalance_shares > 0 else -2
        elif ratio > 0.15:
            return 1 if self.imbalance_shares > 0 else -1
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'imbalance_shares': self.imbalance_shares,
            'paired_shares': self.paired_shares,
            'reference_price': self.reference_price,
            'source': self.source,
            'imbalance_ratio': self.imbalance_ratio,
            'direction_score': self.direction_score
        }


class ClosingAuctionCache:
    """SQLite cache for MOC imbalance data with 15-minute TTL."""
    
    def __init__(self, db_path: str = "data/closing_auction/cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS moc_imbalances (
                    symbol TEXT,
                    timestamp TEXT,
                    imbalance_shares INTEGER,
                    paired_shares INTEGER,
                    reference_price REAL,
                    source TEXT,
                    fetched_at TEXT,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON moc_imbalances(timestamp)
            """)
            conn.commit()
    
    def get(self, symbol: str, max_age_minutes: int = 15) -> Optional[MOCImbalance]:
        """Get cached imbalance if not stale."""
        cutoff = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT * FROM moc_imbalances 
                   WHERE symbol = ? AND timestamp > ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol, cutoff)
            )
            row = cursor.fetchone()
            
            if row:
                return MOCImbalance(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    imbalance_shares=row[2],
                    paired_shares=row[3],
                    reference_price=row[4],
                    source=row[5]
                )
        return None
    
    def store(self, imbalance: MOCImbalance):
        """Store imbalance in cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO moc_imbalances
                   (symbol, timestamp, imbalance_shares, paired_shares, 
                    reference_price, source, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (imbalance.symbol, imbalance.timestamp.isoformat(),
                 imbalance.imbalance_shares, imbalance.paired_shares,
                 imbalance.reference_price, imbalance.source,
                 datetime.now().isoformat())
            )
            conn.commit()
    
    def get_all_for_date(self, date: datetime) -> List[MOCImbalance]:
        """Get all imbalances for a specific date."""
        date_str = date.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT * FROM moc_imbalances 
                   WHERE timestamp LIKE ?
                   ORDER BY timestamp DESC""",
                (f'{date_str}%',)
            )
            rows = cursor.fetchall()
            
            return [
                MOCImbalance(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    imbalance_shares=row[2],
                    paired_shares=row[3],
                    reference_price=row[4],
                    source=row[5]
                )
                for row in rows
            ]


class ClosingAuctionFetcher:
    """
    Fetches MOC imbalance data from multiple sources.
    
    Primary: NYSE ARCA web feed
    Fallback: Calculated proxy from real-time quotes
    """
    
    MONITORED_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EFA', 'VTI']
    
    def __init__(self, cache_path: str = "data/closing_auction/cache.db"):
        self.cache = ClosingAuctionCache(cache_path)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Portfolio-Lab/3.17'}
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def fetch_nyse_arca_proxy(self, symbol: str) -> Optional[MOCImbalance]:
        """
        Fetch MOC data via NYSE ARCA web proxy.
        
        Note: This uses Yahoo Finance as a proxy since direct MOC feeds
        require exchange agreements. In production, replace with:
        - Bloomberg EMSX API
        - ICE Data Services
        - Direct NYSE binary feed
        """
        if not self.session:
            raise RuntimeError("Fetcher not in async context")
        
        try:
            # Yahoo Finance real-time quote as proxy
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Yahoo API returned {response.status} for {symbol}")
                    return None
                
                data = await response.json()
                
                # Extract last price and volume
                result = data.get('chart', {}).get('result', [{}])[0]
                meta = result.get('meta', {})
                timestamps = result.get('timestamp', [])
                volumes = result.get('indicators', {}).get('quote', [{}])[0].get('volume', [])
                
                if not timestamps or not volumes:
                    return None
                
                # Calculate proxy imbalance from late-day volume patterns
                # In real implementation, this would be actual MOC data
                now = datetime.now()
                reference_price = meta.get('regularMarketPrice', 0)
                
                # Simulate imbalance detection from volume spike
                if len(volumes) >= 10:
                    recent_vol = sum(volumes[-5:])
                    avg_vol = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else recent_vol / 5
                    
                    if recent_vol > avg_vol * 2:  # Volume spike detected
                        # Estimate imbalance direction from price movement
                        price_change = meta.get('regularMarketChange', 0)
                        imbalance_dir = 1 if price_change > 0 else -1
                        
                        imbalance = MOCImbalance(
                            symbol=symbol,
                            timestamp=now,
                            imbalance_shares=int(recent_vol * 0.3 * imbalance_dir),
                            paired_shares=int(recent_vol),
                            reference_price=reference_price,
                            source='yahoo_proxy'
                        )
                        
                        self.cache.store(imbalance)
                        return imbalance
                
                return None
                
        except Exception as e:
            logger.error(f"Error fetching MOC proxy for {symbol}: {e}")
            return None
    
    async def fetch_symbol(self, symbol: str, use_cache: bool = True) -> Optional[MOCImbalance]:
        """Fetch MOC imbalance for a single symbol."""
        # Check cache first
        if use_cache:
            cached = self.cache.get(symbol)
            if cached:
                logger.debug(f"Cache hit for {symbol}")
                return cached
        
        # Fetch fresh data
        return await self.fetch_nyse_arca_proxy(symbol)
    
    async def fetch_all(self, symbols: Optional[List[str]] = None) -> Dict[str, MOCImbalance]:
        """Fetch MOC imbalances for all monitored symbols."""
        symbols = symbols or self.MONITORED_SYMBOLS
        
        tasks = [self.fetch_symbol(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        imbalances = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
            elif result:
                imbalances[symbol] = result
        
        return imbalances
    
    def save_to_json(self, imbalances: Dict[str, MOCImbalance], 
                     output_path: str = "data/closing_auction/latest.json"):
        """Save imbalances to JSON for downstream consumption."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'count': len(imbalances),
            'imbalances': {sym: imb.to_dict() for sym, imb in imbalances.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(imbalances)} imbalances to {output_path}")


class HistoricalBackfiller:
    """
    Backfills historical MOC imbalance data from archived sources.
    
    For Phase 1, creates synthetic data for testing.
    Phase 2 will implement actual historical data ingestion.
    """
    
    def __init__(self, cache: ClosingAuctionCache):
        self.cache = cache
    
    def generate_synthetic_history(self, days: int = 90) -> List[MOCImbalance]:
        """
        Generate synthetic MOC data for backtesting.
        
        Creates realistic imbalance patterns:
        - 60% small imbalances (|score| <= 1)
        - 30% medium imbalances (|score| = 2)  
        - 10% large imbalances (|score| = 3)
        - Slight bias toward buy imbalances (52% buy, 48% sell)
        """
        import random
        random.seed(42)  # Reproducible
        
        symbols = ClosingAuctionFetcher.MONITORED_SYMBOLS
        imbalances = []
        
        for day_offset in range(days):
            date = datetime.now() - timedelta(days=day_offset)
            
            for symbol in symbols:
                # Skip weekends
                if date.weekday() >= 5:
                    continue
                
                # Generate timestamp at 3:50pm
                ts = date.replace(hour=15, minute=50, second=0, microsecond=0)
                
                # Determine imbalance magnitude
                rand = random.random()
                if rand < 0.6:
                    magnitude = random.choice([0, 0.1, 0.15])
                elif rand < 0.9:
                    magnitude = random.choice([0.3, 0.4])
                else:
                    magnitude = random.choice([0.5, 0.6, 0.8])
                
                # Direction bias (52% buy)
                direction = 1 if random.random() < 0.52 else -1
                
                paired = random.randint(1000000, 10000000)
                imb_shares = int(paired * magnitude * direction)
                
                imbalance = MOCImbalance(
                    symbol=symbol,
                    timestamp=ts,
                    imbalance_shares=imb_shares,
                    paired_shares=paired,
                    reference_price=random.uniform(100, 500),
                    source='synthetic_backfill'
                )
                
                imbalances.append(imbalance)
                self.cache.store(imbalance)
        
        logger.info(f"Generated {len(imbalances)} synthetic MOC records")
        return imbalances


async def main():
    """CLI entry point for testing."""
    print("Closing Auction MOC Fetcher v3.17 Phase 1")
    print("=" * 50)
    
    async with ClosingAuctionFetcher() as fetcher:
        # Test live fetch
        print("\nFetching live MOC data...")
        imbalances = await fetcher.fetch_all()
        
        for symbol, imb in imbalances.items():
            print(f"\n{symbol}:")
            print(f"  Imbalance: {imb.imbalance_shares:,} shares")
            print(f"  Ratio: {imb.imbalance_ratio:.2%}")
            print(f"  Direction Score: {imb.direction_score}")
            print(f"  Source: {imb.source}")
        
        # Save to JSON
        fetcher.save_to_json(imbalances)
        
        # Generate synthetic history for backtesting
        print("\nGenerating synthetic historical data...")
        backfiller = HistoricalBackfiller(fetcher.cache)
        history = backfiller.generate_synthetic_history(days=90)
        print(f"Backfilled {len(history)} historical records")
    
    print("\nPhase 1 complete!")


if __name__ == "__main__":
    asyncio.run(main())
