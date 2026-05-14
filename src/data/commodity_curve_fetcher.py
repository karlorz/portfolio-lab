"""
Commodity Curve Data Fetcher v3.20 Phase 1
Fetches futures curve data for contango/backwardation analysis
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import aiohttp
import yfinance as yf


@dataclass
class CurvePoint:
    """Single point on futures curve"""
    contract_month: str
    days_to_expiry: int
    price: float
    implied_yield_annual: Optional[float] = None


@dataclass
class CommodityCurve:
    """Full futures curve for a commodity"""
    symbol: str
    timestamp: str
    spot_price: Optional[float]
    front_month_price: float
    deferred_month_price: float
    roll_yield_annual: float
    curve_shape: str  # 'backwardation', 'contango', 'flat'
    curve_points: List[CurvePoint]
    data_quality: str  # 'high', 'medium', 'low', 'synthetic'


class CommodityCurveCache:
    """SQLite cache for curve history"""
    
    def __init__(self, db_path: str = "data/commodity_curve/cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS commodity_curves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    spot_price REAL,
                    front_month_price REAL NOT NULL,
                    deferred_month_price REAL NOT NULL,
                    roll_yield_annual REAL NOT NULL,
                    curve_shape TEXT NOT NULL,
                    data_quality TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_time 
                ON commodity_curves(symbol, timestamp)
            """)
            conn.commit()
    
    def store(self, curve: CommodityCurve):
        """Store curve data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO commodity_curves 
                   (symbol, timestamp, spot_price, front_month_price, 
                    deferred_month_price, roll_yield_annual, curve_shape, data_quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (curve.symbol, curve.timestamp, curve.spot_price,
                 curve.front_month_price, curve.deferred_month_price,
                 curve.roll_yield_annual, curve.curve_shape, curve.data_quality)
            )
            conn.commit()
    
    def get_latest(self, symbol: str) -> Optional[CommodityCurve]:
        """Get most recent curve for symbol"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT * FROM commodity_curves 
                   WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1""",
                (symbol,)
            ).fetchone()
            
            if row:
                return CommodityCurve(
                    symbol=row[1],
                    timestamp=row[2],
                    spot_price=row[3],
                    front_month_price=row[4],
                    deferred_month_price=row[5],
                    roll_yield_annual=row[6],
                    curve_shape=row[7],
                    curve_points=[],  # Simplified for cache
                    data_quality=row[8]
                )
            return None
    
    def get_history(self, symbol: str, days: int = 90) -> List[CommodityCurve]:
        """Get historical curves"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM commodity_curves 
                   WHERE symbol = ? AND timestamp > ?
                   ORDER BY timestamp DESC""",
                (symbol, cutoff)
            ).fetchall()
            
            return [
                CommodityCurve(
                    symbol=r[1], timestamp=r[2], spot_price=r[3],
                    front_month_price=r[4], deferred_month_price=r[5],
                    roll_yield_annual=r[6], curve_shape=r[7],
                    curve_points=[], data_quality=r[8]
                )
                for r in rows
            ]


class CommodityCurveFetcher:
    """Fetch commodity futures curve data"""
    
    COMMODITY_ETFS = {
        'DBC': {'name': 'Invesco DB Commodity Index', 'category': 'broad'},
        'USO': {'name': 'United States Oil Fund', 'category': 'energy'},
        'UNG': {'name': 'United States Natural Gas', 'category': 'energy'},
        'GLD': {'name': 'SPDR Gold Trust', 'category': 'metals'},
        'SLV': {'name': 'iShares Silver Trust', 'category': 'metals'},
        'DBA': {'name': 'Invesco DB Agriculture', 'category': 'agriculture'},
        'CORN': {'name': 'Teucrium Corn Fund', 'category': 'agriculture'},
        'WEAT': {'name': 'Teucrium Wheat Fund', 'category': 'agriculture'},
    }
    
    def __init__(self, cache: Optional[CommodityCurveCache] = None):
        self.cache = cache or CommodityCurveCache()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _calculate_curve_shape(self, roll_yield: float) -> str:
        """Classify curve shape based on roll yield"""
        if roll_yield > 0.5:
            return 'backwardation'
        elif roll_yield < -1.0:
            return 'contango'
        else:
            return 'flat'
    
    def _calculate_roll_yield(self, front: float, deferred: float, 
                                days_between: int = 90) -> float:
        """Calculate annualized roll yield"""
        if front <= 0 or deferred <= 0:
            return 0.0
        
        price_diff = deferred - front
        roll_yield = (price_diff / front) * (365 / days_between) * 100
        return roll_yield
    
    async def fetch_from_yahoo(self, symbol: str) -> Optional[CommodityCurve]:
        """Fetch implied curve from ETF price patterns"""
        try:
            # Use yfinance to get ETF data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if len(hist) < 2:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # For ETFs, we estimate curve from price momentum vs spot proxy
            # This is a simplified approach - real implementation would use futures data
            price_5d_ago = hist['Close'].iloc[0]
            momentum_5d = (current_price - price_5d_ago) / price_5d_ago * 100
            
            # Estimate roll yield from momentum and historical patterns
            # Backwardation: ETF trades at premium to expected roll
            # Contango: ETF trades at discount due to roll cost
            estimated_roll = -momentum_5d * 0.5  # Simplified heuristic
            
            # Constrain to realistic bounds
            estimated_roll = max(-15, min(15, estimated_roll))
            
            curve_shape = self._calculate_curve_shape(estimated_roll)
            
            return CommodityCurve(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                spot_price=current_price * 0.98,  # Estimated spot
                front_month_price=current_price,
                deferred_month_price=current_price * (1 + estimated_roll/100),
                roll_yield_annual=estimated_roll,
                curve_shape=curve_shape,
                curve_points=[],
                data_quality='medium'  # ETF-derived estimate
            )
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    async def fetch_all(self) -> Dict[str, CommodityCurve]:
        """Fetch curves for all tracked commodities"""
        results = {}
        
        for symbol in self.COMMODITY_ETFS.keys():
            curve = await self.fetch_from_yahoo(symbol)
            if curve:
                self.cache.store(curve)
                results[symbol] = curve
        
        return results
    
    def generate_synthetic(self, symbol: str, 
                          base_roll_yield: float = 0.0) -> CommodityCurve:
        """Generate synthetic curve for backtesting"""
        now = datetime.now()
        
        # Add some random variation
        import random
        variation = random.uniform(-2, 2)
        roll_yield = base_roll_yield + variation
        
        curve_shape = self._calculate_curve_shape(roll_yield)
        base_price = 20.0  # Arbitrary base
        
        return CommodityCurve(
            symbol=symbol,
            timestamp=now.isoformat(),
            spot_price=base_price,
            front_month_price=base_price * 1.02,
            deferred_month_price=base_price * (1 + roll_yield/100),
            roll_yield_annual=roll_yield,
            curve_shape=curve_shape,
            curve_points=[],
            data_quality='synthetic'
        )


class HistoricalBackfiller:
    """Backfill historical curve data for model training"""
    
    def __init__(self, fetcher: CommodityCurveFetcher, cache: CommodityCurveCache):
        self.fetcher = fetcher
        self.cache = cache
    
    def backfill_synthetic(self, symbol: str, days: int = 90,
                          base_regime: str = 'mixed') -> int:
        """Generate synthetic historical data for testing"""
        import random
        random.seed(42)  # Reproducible
        
        count = 0
        now = datetime.now()
        
        # Define regime patterns
        regimes = {
            'backwardation': 5.0,   # Positive roll yield
            'contango': -5.0,       # Negative roll yield
            'flat': 0.0,            # Near zero
            'mixed': None           # Varying
        }
        
        base_yield = regimes.get(base_regime, 0.0)
        
        for i in range(days):
            date = now - timedelta(days=i)
            
            if base_regime == 'mixed':
                # Alternate between regimes
                if i % 30 < 15:
                    daily_base = 5.0  # Backwardation
                else:
                    daily_base = -5.0  # Contango
            else:
                daily_base = base_yield
            
            # Create synthetic curve
            curve = self.fetcher.generate_synthetic(symbol, daily_base)
            curve.timestamp = date.isoformat()
            
            self.cache.store(curve)
            count += 1
        
        return count
    
    async def backfill_from_yahoo(self, symbol: str, days: int = 90) -> int:
        """Attempt to backfill from historical ETF data"""
        # Yahoo Finance historical data for ETFs
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            count = 0
            prev_close = None
            for date, row in hist.iterrows():
                # Estimate roll yield from price action
                if prev_close is not None:
                    price_change = (row['Close'] - prev_close) / prev_close * 100
                    estimated_roll = -price_change * 0.3
                    estimated_roll = max(-15, min(15, estimated_roll))
                else:
                    estimated_roll = 0.0
                
                curve_shape = self.fetcher._calculate_curve_shape(estimated_roll)
                
                curve = CommodityCurve(
                    symbol=symbol,
                    timestamp=date.isoformat(),
                    spot_price=row['Close'] * 0.98,
                    front_month_price=row['Close'],
                    deferred_month_price=row['Close'] * (1 + estimated_roll/100),
                    roll_yield_annual=estimated_roll,
                    curve_shape=curve_shape,
                    curve_points=[],
                    data_quality='historical'
                )
                
                self.cache.store(curve)
                count += 1
                prev_close = row['Close']
            
            return count
            
        except Exception as e:
            print(f"Backfill error for {symbol}: {e}")
            return 0


async def main():
    """CLI entry point"""
    import sys
    
    cache = CommodityCurveCache()
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.commodity_curve_fetcher <command>")
        print("Commands: fetch, backfill, status")
        return
    
    command = sys.argv[1]
    
    if command == "fetch":
        async with CommodityCurveFetcher(cache) as fetcher:
            results = await fetcher.fetch_all()
            print(f"Fetched {len(results)} commodity curves:")
            for sym, curve in results.items():
                print(f"  {sym}: {curve.curve_shape} ({curve.roll_yield_annual:+.2f}%)")
    
    elif command == "backfill":
        symbol = sys.argv[2] if len(sys.argv) > 2 else 'DBC'
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
        
        async with CommodityCurveFetcher(cache) as fetcher:
            backfiller = HistoricalBackfiller(fetcher, cache)
            
            # Try historical first, fall back to synthetic
            count = await backfiller.backfill_from_yahoo(symbol, days)
            if count == 0:
                count = backfiller.backfill_synthetic(symbol, days, 'mixed')
                print(f"Generated {count} synthetic records for {symbol}")
            else:
                print(f"Backfilled {count} historical records for {symbol}")
    
    elif command == "status":
        for symbol in CommodityCurveFetcher.COMMODITY_ETFS.keys():
            latest = cache.get_latest(symbol)
            if latest:
                print(f"{symbol}: {latest.curve_shape} ({latest.roll_yield_annual:+.2f}%) - {latest.data_quality}")
            else:
                print(f"{symbol}: No data")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())
