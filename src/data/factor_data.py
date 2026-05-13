"""
Factor ETF Data Infrastructure for Quality-Momentum Overlay (v3.00)

Fetches and manages factor ETF data for rotation strategies:
- MTUM: Momentum
- QUAL: Quality  
- USMV: Low Volatility
- VLUE: Value

Phase 1 of v3.00 Factor Rotation implementation.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FactorETF:
    """Factor ETF metadata and configuration."""
    symbol: str
    factor: str  # momentum, quality, low_vol, value
    expense_ratio: float
    aum_billions: float  # Approximate AUM for liquidity assessment
    description: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Factor ETF Universe
FACTOR_ETFS = {
    "MTUM": FactorETF("MTUM", "momentum", 0.0015, 18.5, "iShares MSCI USA Momentum Factor"),
    "QUAL": FactorETF("QUAL", "quality", 0.0015, 19.2, "iShares MSCI USA Quality Factor"),
    "USMV": FactorETF("USMV", "low_vol", 0.0015, 34.8, "iShares MSCI USA Min Vol Factor"),
    "VLUE": FactorETF("VLUE", "value", 0.0015, 11.3, "iShares MSCI USA Value Factor"),
}

# Factor scoring weights for quality calculation
QUALITY_WEIGHTS = {
    "roe": 0.30,           # Return on equity
    "debt_equity": 0.25,  # Debt to equity (lower is better)
    "earnings_stability": 0.25,  # Earnings variance
    "profitability": 0.20,  # Gross margin stability
}


class FactorDataManager:
    """Manages factor ETF price data and quality scores."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/factors")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "factor_data.db"
        self.metadata_path = self.data_dir / "factor_metadata.json"
        self._init_database()
        self._init_metadata()
    
    def _init_database(self):
        """Initialize SQLite database for factor data."""
        with sqlite3.connect(self.db_path) as conn:
            # Price data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Quality scores table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    roe REAL,
                    debt_equity REAL,
                    earnings_stability REAL,
                    profitability REAL,
                    composite_score REAL NOT NULL,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Factor performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    return_1d REAL,
                    return_1m REAL,
                    return_3m REAL,
                    return_6m REAL,
                    return_12m REAL,
                    vol_20d REAL,
                    UNIQUE(symbol, date)
                )
            """)
            
            conn.commit()
    
    def _init_metadata(self):
        """Initialize factor metadata file."""
        if not self.metadata_path.exists():
            metadata = {
                "version": "3.00",
                "created": datetime.now().isoformat(),
                "etfs": {sym: etf.to_dict() for sym, etf in FACTOR_ETFS.items()},
                "quality_weights": QUALITY_WEIGHTS,
                "last_updated": None,
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def store_prices(self, symbol: str, prices: List[Dict]) -> int:
        """Store price data for a factor ETF.
        
        Args:
            symbol: ETF symbol (MTUM, QUAL, USMV, VLUE)
            prices: List of price dicts with date, open, high, low, close, volume
            
        Returns:
            Number of records inserted
        """
        if symbol not in FACTOR_ETFS:
            raise ValueError(f"Unknown factor ETF: {symbol}")
        
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for p in prices:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO factor_prices 
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, p["date"], p.get("open"), p.get("high"),
                        p.get("low"), p["close"], p.get("volume")
                    ))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert price for {symbol} {p.get('date')}: {e}")
            conn.commit()
        
        self._update_metadata_timestamp()
        logger.info(f"Stored {count} price records for {symbol}")
        return count
    
    def get_prices(self, symbol: str, days: int = 252) -> List[Dict]:
        """Get recent price data for a factor ETF.
        
        Args:
            symbol: ETF symbol
            days: Number of trading days to retrieve
            
        Returns:
            List of price records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM factor_prices 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT ?
            """, (symbol, days))
            return [dict(row) for row in cursor.fetchall()]
    
    def calculate_quality_score(
        self,
        roe: float,
        debt_equity: float,
        earnings_stability: float,
        profitability: float
    ) -> float:
        """Calculate composite quality score.
        
        Args:
            roe: Return on equity (higher is better)
            debt_equity: Debt to equity ratio (lower is better)
            earnings_stability: Earnings stability score (higher is better)
            profitability: Profitability score (higher is better)
            
        Returns:
            Composite quality score (0-1 scale)
        """
        # Normalize inputs to 0-1 scale (approximate)
        roe_norm = min(max(roe / 0.25, 0), 1)  # Assume 25% ROE is excellent
        de_norm = min(max(1 - (debt_equity / 2.0), 0), 1)  # Lower debt is better
        earn_norm = min(max(earnings_stability, 0), 1)
        prof_norm = min(max(profitability, 0), 1)
        
        weights = QUALITY_WEIGHTS
        score = (
            weights["roe"] * roe_norm +
            weights["debt_equity"] * de_norm +
            weights["earnings_stability"] * earn_norm +
            weights["profitability"] * prof_norm
        )
        
        return round(score, 4)
    
    def store_quality_score(self, symbol: str, date: str, metrics: Dict) -> bool:
        """Store quality score for a factor ETF.
        
        Args:
            symbol: ETF symbol
            date: Date string (YYYY-MM-DD)
            metrics: Dict with roe, debt_equity, earnings_stability, profitability
            
        Returns:
            True if stored successfully
        """
        if symbol not in FACTOR_ETFS:
            raise ValueError(f"Unknown factor ETF: {symbol}")
        
        score = self.calculate_quality_score(
            metrics.get("roe", 0.15),
            metrics.get("debt_equity", 0.5),
            metrics.get("earnings_stability", 0.5),
            metrics.get("profitability", 0.5)
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quality_scores
                (symbol, date, roe, debt_equity, earnings_stability, profitability, composite_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, date,
                metrics.get("roe"),
                metrics.get("debt_equity"),
                metrics.get("earnings_stability"),
                metrics.get("profitability"),
                score
            ))
            conn.commit()
        
        return True
    
    def get_quality_scores(self, symbol: str, days: int = 90) -> List[Dict]:
        """Get quality score history for a factor ETF."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM quality_scores 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT ?
            """, (symbol, days))
            return [dict(row) for row in cursor.fetchall()]
    
    def calculate_returns(self, symbol: str) -> Optional[Dict]:
        """Calculate returns for a factor ETF from stored prices."""
        prices = self.get_prices(symbol, days=300)
        if len(prices) < 20:
            return None
        
        closes = [p["close"] for p in reversed(prices)]
        
        def calc_return(period: int) -> Optional[float]:
            if len(closes) <= period:
                return None
            return round((closes[-1] / closes[-(period+1)]) - 1, 6)
        
        def calc_vol(period: int = 20) -> Optional[float]:
            if len(closes) < period + 1:
                return None
            returns = [(closes[i] / closes[i-1]) - 1 for i in range(1, period+1)]
            if not returns:
                return None
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            return round((variance ** 0.5) * (252 ** 0.5), 6)  # Annualized
        
        return {
            "symbol": symbol,
            "date": prices[0]["date"],
            "return_1d": calc_return(1),
            "return_1m": calc_return(21),
            "return_3m": calc_return(63),
            "return_6m": calc_return(126),
            "return_12m": calc_return(252),
            "vol_20d": calc_vol(20),
        }
    
    def store_returns(self, symbol: str) -> bool:
        """Calculate and store returns for a factor ETF."""
        returns = self.calculate_returns(symbol)
        if not returns:
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO factor_performance
                (symbol, date, return_1d, return_1m, return_3m, return_6m, return_12m, vol_20d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                returns["symbol"], returns["date"],
                returns["return_1d"], returns["return_1m"], returns["return_3m"],
                returns["return_6m"], returns["return_12m"], returns["vol_20d"]
            ))
            conn.commit()
        
        return True
    
    def get_all_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all factor ETFs."""
        performance = {}
        for symbol in FACTOR_ETFS:
            perf = self.calculate_returns(symbol)
            if perf:
                performance[symbol] = perf
        return performance
    
    def get_factor_rankings(self) -> List[Tuple[str, float]]:
        """Rank factor ETFs by 6-month momentum."""
        performance = self.get_all_performance()
        ranked = []
        for symbol, perf in performance.items():
            if perf.get("return_6m") is not None:
                ranked.append((symbol, perf["return_6m"]))
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def _update_metadata_timestamp(self):
        """Update last updated timestamp in metadata."""
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update metadata: {e}")


def fetch_factor_prices_from_yahoo(symbol: str, days: int = 252) -> List[Dict]:
    """Fetch price data from Yahoo Finance (placeholder for integration).
    
    In production, this would use the existing data pipeline or
    call the Yahoo Finance API directly.
    """
    # Placeholder - actual implementation would integrate with existing pipeline
    logger.info(f"Fetching {days} days of data for {symbol} from Yahoo Finance")
    return []


def main():
    """CLI for factor data management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor ETF Data Manager v3.00")
    parser.add_argument("command", choices=["init", "status", "fetch", "rank"])
    parser.add_argument("--symbol", choices=list(FACTOR_ETFS.keys()))
    parser.add_argument("--days", type=int, default=252)
    
    args = parser.parse_args()
    
    manager = FactorDataManager()
    
    if args.command == "init":
        logger.info("Factor data database initialized")
        logger.info(f"Database: {manager.db_path}")
        logger.info(f"Metadata: {manager.metadata_path}")
        
    elif args.command == "status":
        with sqlite3.connect(manager.db_path) as conn:
            # Count records
            counts = {}
            for table in ["factor_prices", "quality_scores", "factor_performance"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            
            print("\nFactor Data Status:")
            print("-" * 40)
            for table, count in counts.items():
                print(f"  {table}: {count} records")
            
            # Show date range
            cursor = conn.execute("""
                SELECT MIN(date), MAX(date) FROM factor_prices
            """)
            min_date, max_date = cursor.fetchone()
            if min_date:
                print(f"\n  Date range: {min_date} to {max_date}")
        
        # Show metadata
        with open(manager.metadata_path, "r") as f:
            meta = json.load(f)
        print(f"\n  Last updated: {meta.get('last_updated', 'Never')}")
        print(f"  Version: {meta.get('version', 'Unknown')}")
        
    elif args.command == "fetch":
        if args.symbol:
            prices = fetch_factor_prices_from_yahoo(args.symbol, args.days)
            if prices:
                manager.store_prices(args.symbol, prices)
            else:
                logger.info("No data fetched (placeholder implementation)")
        else:
            for symbol in FACTOR_ETFS:
                prices = fetch_factor_prices_from_yahoo(symbol, args.days)
                if prices:
                    manager.store_prices(symbol, prices)
                
    elif args.command == "rank":
        rankings = manager.get_factor_rankings()
        print("\nFactor Rankings (6-month momentum):")
        print("-" * 40)
        for i, (symbol, ret) in enumerate(rankings, 1):
            ret_pct = ret * 100
            print(f"  {i}. {symbol}: {ret_pct:+.2f}%")


if __name__ == "__main__":
    main()
