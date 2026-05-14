"""
International Equity Data Fetcher for Portfolio-Lab
Fetches EFA (MSCI EAFE) and EEM (MSCI Emerging Markets) price data
for momentum overlay strategy (v3.13)
"""

import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import math

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    yf = None
    pd = None
    np = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_DB = Path("/root/projects/portfolio-lab/data/market.db")
CACHE_TTL_HOURS = 6
MOMENTUM_WINDOWS = [21, 63, 126]  # 1M, 3M, 6M trading days
SYMBOLS = {
    'EFA': 'MSCI EAFE Developed Markets',
    'EEM': 'MSCI Emerging Markets',
    'VEA': 'Vanguard Developed Markets (alt)',
    'VWO': 'Vanguard Emerging Markets (alt)',
    'SPY': 'S&P 500 (benchmark)'
}


@dataclass
class MomentumMetrics:
    """Momentum calculation results for a symbol"""
    symbol: str
    price: float
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    volatility_20d: float
    sharpe_6m: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RelativeMomentum:
    """Relative momentum vs SPY benchmark"""
    symbol: str
    efa_momentum_6m: float
    eem_momentum_6m: float
    spy_momentum_6m: float
    efa_vs_spy: float
    eem_vs_spy: float
    signal: str  # 'neutral', 'efa_lead', 'eem_lead'
    confidence: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InternationalData:
    """Complete international equity data snapshot"""
    timestamp: str
    metrics: Dict[str, MomentumMetrics]
    relative: RelativeMomentum
    data_fresh: bool
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'relative': self.relative.to_dict(),
            'data_fresh': self.data_fresh
        }


class InternationalDataFetcher:
    """Fetches and caches international equity data from Yahoo Finance"""
    
    def __init__(self, cache_db: Path = CACHE_DB):
        self.cache_db = cache_db
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache table if not exists"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS international_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp TEXT,
                    price REAL,
                    momentum_6m REAL
                )
            """)
            conn.commit()
    
    def _is_cache_fresh(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT timestamp FROM international_cache WHERE symbol = ?",
                    (symbol,)
                )
                row = cursor.fetchone()
                if not row:
                    return False
                
                cache_time = datetime.fromisoformat(row[0])
                age = datetime.now() - cache_time
                return age < timedelta(hours=CACHE_TTL_HOURS)
        except Exception as e:
            logger.warning(f"Cache check failed for {symbol}: {e}")
            return False
    
    def _get_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retrieve cached price data"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                # Check if we need to fetch from main price cache
                cursor = conn.execute(
                    """SELECT data FROM prices_cache 
                       WHERE symbol = ? AND date(timestamp) >= date('now', '-6 months')""",
                    (symbol,)
                )
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {symbol}: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save price data to cache"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                # Store in main prices cache (shared with other fetchers)
                latest = df.iloc[-1]
                conn.execute("""
                    INSERT OR REPLACE INTO international_cache 
                    (symbol, data, timestamp, price, momentum_6m)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol,
                    df.to_json(),
                    datetime.now().isoformat(),
                    float(latest['close']),
                    float(df['close'].pct_change(126).iloc[-1]) if len(df) >= 126 else 0.0
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache save failed for {symbol}: {e}")
    
    def fetch_symbol(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch price data for a symbol from Yahoo Finance"""
        logger.info(f"Fetching {symbol} from Yahoo Finance...")
        
        # Check cache first
        if self._is_cache_fresh(symbol):
            cached = self._get_cached(symbol)
            if cached is not None and len(cached) >= 126:
                logger.info(f"Using cached data for {symbol}")
                return cached
        
        # Fetch from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            df = pd.DataFrame({
                'date': hist.index,
                'open': hist['Open'].values,
                'high': hist['High'].values,
                'low': hist['Low'].values,
                'close': hist['Close'].values,
                'volume': hist['Volume'].values
            })
            
            # Save to cache
            self._save_to_cache(symbol, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            # Return cached data even if stale as fallback
            cached = self._get_cached(symbol)
            if cached is not None:
                logger.warning(f"Using stale cache for {symbol}")
                return cached
            raise
    
    def calculate_momentum(self, df: pd.DataFrame, symbol: str) -> MomentumMetrics:
        """Calculate momentum metrics for a symbol"""
        if len(df) < max(MOMENTUM_WINDOWS):
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} rows")
        
        current_price = df['close'].iloc[-1]
        
        # Calculate momentum for each window
        momentum_1m = df['close'].pct_change(MOMENTUM_WINDOWS[0]).iloc[-1]
        momentum_3m = df['close'].pct_change(MOMENTUM_WINDOWS[1]).iloc[-1]
        momentum_6m = df['close'].pct_change(MOMENTUM_WINDOWS[2]).iloc[-1]
        
        # Volatility (20-day)
        volatility_20d = df['close'].pct_change().iloc[-20:].std() * np.sqrt(252)
        
        # Sharpe-like score (6m momentum / 6m volatility)
        returns_6m = df['close'].pct_change().iloc[-126:]
        vol_6m = returns_6m.std() * np.sqrt(252)
        sharpe_6m = momentum_6m / vol_6m if vol_6m > 0 else 0.0
        
        return MomentumMetrics(
            symbol=symbol,
            price=round(current_price, 2),
            momentum_1m=round(momentum_1m, 4),
            momentum_3m=round(momentum_3m, 4),
            momentum_6m=round(momentum_6m, 4),
            volatility_20d=round(volatility_20d, 4),
            sharpe_6m=round(sharpe_6m, 4),
            timestamp=datetime.now().isoformat()
        )
    
    def calculate_relative_momentum(
        self, 
        efa_metrics: MomentumMetrics,
        eem_metrics: MomentumMetrics,
        spy_metrics: MomentumMetrics
    ) -> RelativeMomentum:
        """Calculate relative momentum signals"""
        
        efa_vs_spy = efa_metrics.momentum_6m - spy_metrics.momentum_6m
        eem_vs_spy = eem_metrics.momentum_6m - spy_metrics.momentum_6m
        
        # Signal determination
        # Activation threshold: 5% for EFA, 8% for EEM (higher vol)
        EFA_THRESHOLD = 0.05
        EEM_THRESHOLD = 0.08
        
        if efa_vs_spy > EFA_THRESHOLD:
            signal = 'efa_lead'
            confidence = min(efa_vs_spy / 0.10, 1.0)  # Max confidence at 10% outperformance
        elif eem_vs_spy > EEM_THRESHOLD:
            signal = 'eem_lead'
            confidence = min(eem_vs_spy / 0.15, 1.0)  # Max confidence at 15% outperformance
        else:
            signal = 'neutral'
            confidence = 0.0
        
        return RelativeMomentum(
            symbol='relative_momentum',
            efa_momentum_6m=efa_metrics.momentum_6m,
            eem_momentum_6m=eem_metrics.momentum_6m,
            spy_momentum_6m=spy_metrics.momentum_6m,
            efa_vs_spy=round(efa_vs_spy, 4),
            eem_vs_spy=round(eem_vs_spy, 4),
            signal=signal,
            confidence=round(confidence, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def fetch_all(self) -> InternationalData:
        """Fetch and calculate all international equity data"""
        logger.info("Fetching international equity data...")
        
        data_fresh = True
        metrics = {}
        
        # Fetch primary symbols
        for symbol in ['EFA', 'EEM', 'SPY']:
            try:
                df = self.fetch_symbol(symbol)
                metrics[symbol] = self.calculate_momentum(df, symbol)
                
                # Check if data is fresh
                if not self._is_cache_fresh(symbol):
                    data_fresh = False
                    
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                data_fresh = False
                # Use placeholder if fetch fails
                metrics[symbol] = MomentumMetrics(
                    symbol=symbol,
                    price=0.0,
                    momentum_1m=0.0,
                    momentum_3m=0.0,
                    momentum_6m=0.0,
                    volatility_20d=0.0,
                    sharpe_6m=0.0,
                    timestamp=datetime.now().isoformat()
                )
        
        # Calculate relative momentum
        relative = self.calculate_relative_momentum(
            metrics['EFA'],
            metrics['EEM'],
            metrics['SPY']
        )
        
        return InternationalData(
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            relative=relative,
            data_fresh=data_fresh
        )
    
    def save_snapshot(self, data: InternationalData, output_path: Optional[Path] = None):
        """Save data snapshot to JSON"""
        if output_path is None:
            output_path = Path("/root/projects/portfolio-lab/data/international_momentum.json")
        
        with open(output_path, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)
        
        logger.info(f"Saved international momentum snapshot to {output_path}")
    
    def get_signal_summary(self) -> Dict:
        """Get a quick signal summary for dashboard/CLI"""
        data = self.fetch_all()
        
        return {
            'timestamp': data.timestamp,
            'signal': data.relative.signal,
            'confidence': data.relative.confidence,
            'efa_momentum_6m': data.relative.efa_momentum_6m,
            'eem_momentum_6m': data.relative.eem_momentum_6m,
            'spy_momentum_6m': data.relative.spy_momentum_6m,
            'efa_vs_spy': data.relative.efa_vs_spy,
            'eem_vs_spy': data.relative.eem_vs_spy,
            'data_fresh': data.data_fresh,
            'prices': {
                k: v.price for k, v in data.metrics.items()
            }
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='International Equity Data Fetcher')
    parser.add_argument('--fetch', action='store_true', help='Fetch and display current data')
    parser.add_argument('--save', action='store_true', help='Save snapshot to file')
    parser.add_argument('--signal', action='store_true', help='Show signal summary only')
    parser.add_argument('--symbol', type=str, help='Fetch specific symbol (EFA, EEM, SPY)')
    
    args = parser.parse_args()
    
    fetcher = InternationalDataFetcher()
    
    if args.symbol:
        df = fetcher.fetch_symbol(args.symbol)
        metrics = fetcher.calculate_momentum(df, args.symbol)
        print(json.dumps(metrics.to_dict(), indent=2))
    
    elif args.signal:
        summary = fetcher.get_signal_summary()
        print(json.dumps(summary, indent=2))
    
    elif args.fetch or args.save:
        data = fetcher.fetch_all()
        
        if args.save:
            fetcher.save_snapshot(data)
        
        # Always print to stdout
        print(json.dumps(data.to_dict(), indent=2))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
