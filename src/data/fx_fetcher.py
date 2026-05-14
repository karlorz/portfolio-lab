"""
FX Currency Carry Data Fetcher
Fetches UUP/UDN data for currency carry signal detection.

Part of v3.15: FX Currency Carry Overlay
"""

import os
import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/market.db")
CACHE_TTL_HOURS = 4


@dataclass
class FXMetrics:
    """Currency carry metrics for UUP/UDN analysis."""
    timestamp: str
    uup_price: float
    udn_price: float
    uup_return_30d: float
    udn_return_30d: float
    usd_strength_score: float  # -1.0 to 1.0
    carry_regime: str  # positive/negative/neutral
    momentum_direction: str  # bullish/bearish/neutral
    volatility_regime: str  # low/medium/high
    data_freshness_hours: float


class FXFetcher:
    """Fetches and caches UUP/UDN data from Yahoo Finance."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fx_cache (
                    symbol TEXT PRIMARY KEY,
                    price REAL,
                    price_30d_ago REAL,
                    volatility_30d REAL,
                    updated_at TIMESTAMP
                )
            """)
            conn.commit()
    
    def _is_cache_fresh(self, symbol: str) -> bool:
        """Check if cache entry is within TTL."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT updated_at FROM fx_cache WHERE symbol = ?",
                (symbol,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            updated = datetime.fromisoformat(row[0])
            age = datetime.now() - updated
            return age < timedelta(hours=CACHE_TTL_HOURS)
    
    def _fetch_yahoo(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance v8 API."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "interval": "1d",
            "range": "60d",
            "events": "div,splits"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PortfolioLab/1.0)"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            prices = result["indicators"]["adjclose"][0]["adjclose"]
            
            # Filter None values
            valid_data = [(ts, p) for ts, p in zip(timestamps, prices) if p is not None]
            
            if len(valid_data) < 30:
                raise ValueError(f"Insufficient data: {len(valid_data)} days")
            
            current_price = valid_data[-1][1]
            price_30d_ago = valid_data[-31][1] if len(valid_data) >= 31 else valid_data[0][1]
            
            # Calculate 30-day volatility
            returns = []
            for i in range(1, min(31, len(valid_data))):
                if valid_data[i-1][1] > 0:
                    ret = (valid_data[i][1] - valid_data[i-1][1]) / valid_data[i-1][1]
                    returns.append(ret)
            
            if len(returns) > 1:
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5)  # Annualized
            else:
                volatility = 0.1  # Default 10%
            
            return {
                "price": current_price,
                "price_30d_ago": price_30d_ago,
                "volatility_30d": volatility,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise
    
    def _get_cached_or_fetch(self, symbol: str) -> Dict[str, Any]:
        """Get data from cache or fetch fresh."""
        if self._is_cache_fresh(symbol):
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT price, price_30d_ago, volatility_30d, updated_at FROM fx_cache WHERE symbol = ?",
                    (symbol,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "price": row[0],
                        "price_30d_ago": row[1],
                        "volatility_30d": row[2],
                        "timestamp": row[3]
                    }
        
        # Fetch fresh data
        data = self._fetch_yahoo(symbol)
        
        # Update cache
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fx_cache 
                   (symbol, price, price_30d_ago, volatility_30d, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol, data["price"], data["price_30d_ago"], 
                 data["volatility_30d"], data["timestamp"])
            )
            conn.commit()
        
        return data
    
    def fetch_metrics(self) -> FXMetrics:
        """Fetch and compute FX carry metrics."""
        uup_data = self._get_cached_or_fetch("UUP")
        udn_data = self._get_cached_or_fetch("UDN")
        
        # Calculate returns
        uup_return = ((uup_data["price"] - uup_data["price_30d_ago"]) 
                      / uup_data["price_30d_ago"]) * 100
        udn_return = ((udn_data["price"] - udn_data["price_30d_ago"]) 
                      / udn_data["price_30d_ago"]) * 100
        
        # USD strength score (-1.0 to 1.0)
        usd_strength = (uup_return - udn_return) / 8.0  # Normalize
        usd_strength = max(-1.0, min(1.0, usd_strength))
        
        # Determine regimes
        USD_BULL_THRESHOLD = 2.0
        USD_BEAR_THRESHOLD = -2.0
        
        if uup_return > USD_BULL_THRESHOLD and udn_return < -1.0:
            carry_regime = "positive"
            momentum_direction = "bullish"
        elif udn_return > USD_BULL_THRESHOLD and uup_return < -1.0:
            carry_regime = "negative"
            momentum_direction = "bearish"
        else:
            carry_regime = "neutral"
            momentum_direction = "neutral"
        
        # Average volatility
        avg_vol = (uup_data["volatility_30d"] + udn_data["volatility_30d"]) / 2
        if avg_vol < 0.08:
            volatility_regime = "low"
        elif avg_vol < 0.15:
            volatility_regime = "medium"
        else:
            volatility_regime = "high"
        
        # Calculate freshness
        updated = datetime.fromisoformat(uup_data["timestamp"])
        freshness_hours = (datetime.now() - updated).total_seconds() / 3600
        
        return FXMetrics(
            timestamp=datetime.now().isoformat(),
            uup_price=uup_data["price"],
            udn_price=udn_data["price"],
            uup_return_30d=round(uup_return, 2),
            udn_return_30d=round(udn_return, 2),
            usd_strength_score=round(usd_strength, 2),
            carry_regime=carry_regime,
            momentum_direction=momentum_direction,
            volatility_regime=volatility_regime,
            data_freshness_hours=round(freshness_hours, 1)
        )
    
    def save_metrics(self, metrics: FXMetrics, output_path: Optional[Path] = None):
        """Save metrics to JSON file."""
        output_path = output_path or Path("data/fx_metrics.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        logger.info(f"Saved FX metrics to {output_path}")
    
    def get_signal(self) -> Dict[str, Any]:
        """Get current carry signal for ensemble integration."""
        metrics = self.fetch_metrics()
        
        # Risk controls
        if metrics.volatility_regime == "high":
            signal_type = "neutral"
            confidence = 0.0
            reason = "high_volatility"
        elif metrics.carry_regime == "neutral":
            signal_type = "neutral"
            confidence = 0.0
            reason = "no_clear_regime"
        else:
            signal_type = "usd_strength" if metrics.momentum_direction == "bullish" else "usd_weakness"
            # Confidence based on strength score
            confidence = min(abs(metrics.usd_strength_score), 1.0)
            reason = "momentum_aligned"
        
        return {
            "signal_type": signal_type,
            "confidence": round(confidence, 2),
            "regime": metrics.carry_regime,
            "direction": metrics.momentum_direction,
            "reason": reason,
            "timestamp": metrics.timestamp
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FX Currency Carry Data Fetcher")
    parser.add_argument("--fetch", action="store_true", help="Fetch current metrics")
    parser.add_argument("--save", action="store_true", help="Save to JSON file")
    parser.add_argument("--signal", action="store_true", help="Get signal for ensemble")
    parser.add_argument("--output", type=str, default="data/fx_metrics.json", 
                       help="Output file path")
    parser.add_argument("--history", action="store_true", 
                       help="Show 30-day momentum history")
    
    args = parser.parse_args()
    
    fetcher = FXFetcher()
    
    if args.fetch or (not args.signal and not args.history):
        metrics = fetcher.fetch_metrics()
        print(json.dumps(asdict(metrics), indent=2))
        
        if args.save:
            fetcher.save_metrics(metrics, Path(args.output))
    
    if args.signal:
        signal = fetcher.get_signal()
        print(json.dumps(signal, indent=2))
    
    if args.history:
        # Show current momentum context
        metrics = fetcher.fetch_metrics()
        print(f"\n30-Day Momentum History Context:")
        print(f"  UUP: {metrics.uup_return_30d:+.2f}%")
        print(f"  UDN: {metrics.udn_return_30d:+.2f}%")
        print(f"  USD Strength: {metrics.usd_strength_score:+.2f}")
        print(f"  Regime: {metrics.carry_regime.upper()}")
        print(f"  Direction: {metrics.momentum_direction.upper()}")
        print(f"  Volatility: {metrics.volatility_regime.upper()}")


if __name__ == "__main__":
    main()
