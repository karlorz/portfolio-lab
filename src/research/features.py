"""
Feature engineering for ML signal detection.
Generates features from price data, VIX, and market microstructure.
"""
import os
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Features:
    """Feature vector for ML prediction."""
    symbol: str
    timestamp: str
    
    # Price momentum features
    return_1d: float
    return_5d: float
    return_20d: float
    volatility_20d: float
    
    # Trend features
    sma_20: float
    sma_50: float
    price_vs_sma20: float
    price_vs_sma50: float
    
    # Volume features
    volume_20d_avg: float
    volume_ratio: float
    
    # VIX features
    vix_level: float
    vix_change_5d: float
    vix_percentile_20d: float
    
    # Correlation features
    spy_correlation_20d: float
    
    # Regime features
    trend_direction: int  # -1, 0, 1
    vol_regime: str  # low, normal, high
    
    # Target (for training)
    future_return_5d: Optional[float] = None
    regime_label: Optional[int] = None  # 0: bear, 1: neutral, 2: bull


class FeaturePipeline:
    """
    Pipeline for generating ML features from market data.
    
    Uses SQLite price data and generates rolling statistics
    for regime classification and signal generation.
    """
    
    def __init__(self, db_path: str = "data/market.db"):
        self.db_path = db_path
        self.features_cache: Dict[str, List[Features]] = {}
        
    def _get_connection(self):
        """Get SQLite connection."""
        return sqlite3.connect(self.db_path)
    
    def _get_price_data(
        self, 
        symbol: str, 
        days: int = 100,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch price history from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if end_date:
            cursor.execute("""
                SELECT date, close, volume 
                FROM prices 
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT ?
            """, (symbol, end_date, days))
        else:
            cursor.execute("""
                SELECT date, close, volume 
                FROM prices 
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT ?
            """, (symbol, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Return oldest first
        data = []
        for row in reversed(rows):
            data.append({
                "date": row[0],
                "close": row[1],
                "volume": row[2] if row[2] else 0,
            })
        return data
    
    def _get_vix_data(self, days: int = 50) -> List[Dict[str, Any]]:
        """Fetch VIX price history."""
        return self._get_price_data("^VIX", days=days)
    
    def _calculate_returns(self, prices: List[float], periods: List[int]) -> Dict[int, float]:
        """Calculate returns for multiple periods."""
        returns = {}
        for period in periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1] if prices[-period-1] != 0 else 0
                returns[period] = ret
            else:
                returns[period] = 0.0
        return returns
    
    def _calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate annualized volatility."""
        if len(prices) < period + 1:
            return 0.0
        
        # Daily returns
        returns = []
        for i in range(1, min(period + 1, len(prices))):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # Annualized std dev
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate simple moving average."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return np.mean(prices[-period:])
    
    def _calculate_correlation(
        self, 
        prices1: List[float], 
        prices2: List[float], 
        period: int = 20
    ) -> float:
        """Calculate correlation between two price series."""
        if len(prices1) < period or len(prices2) < period:
            return 0.0
        
        # Use returns for correlation
        returns1 = []
        returns2 = []
        
        for i in range(1, min(period + 1, len(prices1), len(prices2))):
            if prices1[i-1] != 0 and prices2[i-1] != 0:
                returns1.append((prices1[i] - prices1[i-1]) / prices1[i-1])
                returns2.append((prices2[i] - prices2[i-1]) / prices2[i-1])
        
        if len(returns1) < 2:
            return 0.0
        
        try:
            return np.corrcoef(returns1, returns2)[0, 1]
        except:
            return 0.0
    
    def generate_features(
        self, 
        symbol: str,
        reference_date: Optional[str] = None
    ) -> Optional[Features]:
        """
        Generate feature vector for a symbol at a given date.
        
        Args:
            symbol: Stock/ETF symbol
            reference_date: Date to calculate features for (default: latest)
            
        Returns:
            Features dataclass or None if insufficient data
        """
        # Fetch data
        price_data = self._get_price_data(symbol, days=100, end_date=reference_date)
        vix_data = self._get_vix_data(days=50)
        spy_data = self._get_price_data("SPY", days=100, end_date=reference_date)
        
        if len(price_data) < 50:
            return None
        
        # Extract price series
        prices = [p["close"] for p in price_data]
        volumes = [p["volume"] for p in price_data]
        current_price = prices[-1]
        current_date = price_data[-1]["date"]
        
        # Calculate returns
        returns = self._calculate_returns(prices, [1, 5, 20])
        
        # Volatility
        vol_20d = self._calculate_volatility(prices, 20)
        
        # Moving averages
        sma_20 = self._calculate_sma(prices, 20)
        sma_50 = self._calculate_sma(prices, 50)
        price_vs_sma20 = (current_price - sma_20) / sma_20 if sma_20 != 0 else 0
        price_vs_sma50 = (current_price - sma_50) / sma_50 if sma_50 != 0 else 0
        
        # Volume features
        volume_20d_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        volume_ratio = volumes[-1] / volume_20d_avg if volume_20d_avg > 0 else 1.0
        
        # VIX features
        vix_prices = [v["close"] for v in vix_data]
        vix_level = vix_prices[-1] if vix_prices else 20
        vix_change_5d = 0.0
        if len(vix_prices) >= 6:
            vix_change_5d = (vix_prices[-1] - vix_prices[-6]) / vix_prices[-6] if vix_prices[-6] != 0 else 0
        
        # VIX percentile
        if len(vix_prices) >= 20:
            vix_percentile_20d = sum(1 for v in vix_prices[-20:] if v <= vix_level) / 20
        else:
            vix_percentile_20d = 0.5
        
        # SPY correlation
        spy_prices = [p["close"] for p in spy_data]
        spy_corr = self._calculate_correlation(prices, spy_prices, 20)
        
        # Trend direction
        trend = 0
        if price_vs_sma20 > 0.02:
            trend = 1  # Up
        elif price_vs_sma20 < -0.02:
            trend = -1  # Down
        
        # Vol regime
        vol_regime = "normal"
        if vix_level > 25:
            vol_regime = "high"
        elif vix_level < 15:
            vol_regime = "low"
        
        return Features(
            symbol=symbol,
            timestamp=current_date,
            return_1d=returns.get(1, 0),
            return_5d=returns.get(5, 0),
            return_20d=returns.get(20, 0),
            volatility_20d=vol_20d,
            sma_20=sma_20,
            sma_50=sma_50,
            price_vs_sma20=price_vs_sma20,
            price_vs_sma50=price_vs_sma50,
            volume_20d_avg=volume_20d_avg,
            volume_ratio=volume_ratio,
            vix_level=vix_level,
            vix_change_5d=vix_change_5d,
            vix_percentile_20d=vix_percentile_20d,
            spy_correlation_20d=spy_corr,
            trend_direction=trend,
            vol_regime=vol_regime,
        )
    
    def generate_all_features(
        self, 
        symbols: List[str],
        lookback_days: int = 252
    ) -> Dict[str, List[Features]]:
        """
        Generate historical features for multiple symbols.
        
        Used for training ML models on historical data.
        """
        all_features = {}
        
        for symbol in symbols:
            features_list = []
            
            # Get price data for lookback period
            price_data = self._get_price_data(symbol, days=lookback_days + 50)
            
            if len(price_data) < 50:
                continue
            
            # Generate features for each date with sufficient history
            for i in range(50, len(price_data)):
                date = price_data[i]["date"]
                feats = self.generate_features(symbol, reference_date=date)
                
                if feats:
                    # Add future return as target
                    if i + 5 < len(price_data):
                        future_price = price_data[i + 5]["close"]
                        current_price = price_data[i]["close"]
                        if current_price > 0:
                            feats.future_return_5d = (future_price - current_price) / current_price
                            
                            # Simple regime labeling
                            if feats.future_return_5d > 0.02:
                                feats.regime_label = 2  # Bull
                            elif feats.future_return_5d < -0.02:
                                feats.regime_label = 0  # Bear
                            else:
                                feats.regime_label = 1  # Neutral
                    
                    features_list.append(feats)
            
            all_features[symbol] = features_list
        
        return all_features
    
    def to_dataframe(self, features_list: List[Features]) -> Any:
        """Convert features list to pandas DataFrame."""
        try:
            import pandas as pd
            
            data = []
            for f in features_list:
                data.append({
                    "symbol": f.symbol,
                    "timestamp": f.timestamp,
                    "return_1d": f.return_1d,
                    "return_5d": f.return_5d,
                    "return_20d": f.return_20d,
                    "volatility_20d": f.volatility_20d,
                    "sma_20": f.sma_20,
                    "sma_50": f.sma_50,
                    "price_vs_sma20": f.price_vs_sma20,
                    "price_vs_sma50": f.price_vs_sma50,
                    "volume_ratio": f.volume_ratio,
                    "vix_level": f.vix_level,
                    "vix_change_5d": f.vix_change_5d,
                    "vix_percentile_20d": f.vix_percentile_20d,
                    "spy_correlation_20d": f.spy_correlation_20d,
                    "trend_direction": f.trend_direction,
                    "vol_regime": f.vol_regime,
                    "future_return_5d": f.future_return_5d,
                    "regime_label": f.regime_label,
                })
            
            return pd.DataFrame(data)
        except ImportError:
            print("pandas not installed, returning list of dicts")
            return [vars(f) for f in features_list]


class FeatureStore:
    """
    Persistent storage for generated features.
    Allows features to be cached and reused across pipeline runs.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.features_file = os.path.join(data_dir, "features.jsonl")
        
    def save_features(self, features: Features):
        """Append features to storage."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        record = {
            "symbol": features.symbol,
            "timestamp": features.timestamp,
            "return_1d": features.return_1d,
            "return_5d": features.return_5d,
            "return_20d": features.return_20d,
            "volatility_20d": features.volatility_20d,
            "price_vs_sma20": features.price_vs_sma20,
            "price_vs_sma50": features.price_vs_sma50,
            "volume_ratio": features.volume_ratio,
            "vix_level": features.vix_level,
            "vix_change_5d": features.vix_change_5d,
            "vix_percentile_20d": features.vix_percentile_20d,
            "spy_correlation_20d": features.spy_correlation_20d,
            "trend_direction": features.trend_direction,
            "vol_regime": features.vol_regime,
            "future_return_5d": features.future_return_5d,
            "regime_label": features.regime_label,
        }
        
        with open(self.features_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def load_recent_features(
        self, 
        symbol: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Load recent features for a symbol."""
        if not os.path.exists(self.features_file):
            return []
        
        features = []
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with open(self.features_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("symbol") == symbol and record.get("timestamp", "") > cutoff:
                        features.append(record)
                except:
                    continue
        
        return features


def main():
    """CLI for feature pipeline."""
    import sys
    
    pipeline = FeaturePipeline()
    store = FeatureStore()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "generate":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "SPY"
            features = pipeline.generate_features(symbol)
            if features:
                print(json.dumps(vars(features), indent=2, default=str))
            else:
                print(f"No features generated for {symbol}")
                
        elif cmd == "batch":
            symbols = sys.argv[2:] if len(sys.argv) > 2 else ["SPY", "GLD", "TLT", "IEF"]
            for symbol in symbols:
                features = pipeline.generate_features(symbol)
                if features:
                    store.save_features(features)
                    print(f"Saved features for {symbol}")
                else:
                    print(f"Failed to generate features for {symbol}")
                    
        elif cmd == "historical":
            symbols = sys.argv[2:] if len(sys.argv) > 2 else ["SPY", "GLD", "TLT"]
            all_features = pipeline.generate_all_features(symbols, lookback_days=252)
            
            total = sum(len(f) for f in all_features.values())
            print(f"Generated {total} feature vectors across {len(symbols)} symbols")
            
            # Save to file
            for symbol, feats in all_features.items():
                for f in feats:
                    store.save_features(f)
            print(f"Saved to {store.features_file}")
            
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: generate [SYMBOL], batch [SYMBOLS...], historical [SYMBOLS...]")
    else:
        # Default: generate for SPY
        features = pipeline.generate_features("SPY")
        if features:
            print(json.dumps(vars(features), indent=2, default=str))


if __name__ == "__main__":
    main()
