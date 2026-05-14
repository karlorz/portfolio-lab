"""FX Currency Carry data fetcher for UUP/UDN ETFs.

Fetches UUP (US Dollar Bullish) and UDN (US Dollar Bearish) data
from Yahoo Finance for currency carry regime detection.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import yfinance as yf
import numpy as np

# Data directory (consistent with other modules)
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()

# Constants
CACHE_DB = DATA_DIR / "fx_cache.db"
CACHE_TTL_HOURS = 4
UUP_SYMBOL = "UUP"
UDN_SYMBOL = "UDN"


@dataclass
class FXMetrics:
    """FX carry metrics for UUP/UDN."""
    timestamp: str
    uup_price: float
    udn_price: float
    uup_return_30d: float
    udn_return_30d: float
    usd_strength_score: float  # -1.0 to 1.0
    carry_regime: str  # positive, negative, neutral
    momentum_direction: str  # bullish, bearish, neutral
    volatility_regime: str  # low, medium, high
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def init_cache():
    """Initialize SQLite cache for FX data."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fx_data (
            symbol TEXT PRIMARY KEY,
            price REAL,
            return_30d REAL,
            volatility REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_cached_data(symbol: str) -> Optional[dict]:
    """Get cached FX data if fresh."""
    if not CACHE_DB.exists():
        return None
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT price, return_30d, volatility, timestamp FROM fx_data WHERE symbol = ?",
        (symbol,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    price, ret_30d, vol, timestamp = row
    cached_time = datetime.fromisoformat(timestamp)
    
    if datetime.now() - cached_time > timedelta(hours=CACHE_TTL_HOURS):
        return None
    
    return {
        "price": price,
        "return_30d": ret_30d,
        "volatility": vol,
        "timestamp": timestamp
    }


def save_to_cache(symbol: str, price: float, ret_30d: float, volatility: float):
    """Save FX data to cache."""
    init_cache()
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO fx_data (symbol, price, return_30d, volatility, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (symbol, price, ret_30d, volatility, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def fetch_etf_data(symbol: str) -> Tuple[float, float, float]:
    """Fetch ETF price and calculate 30-day return and volatility.
    
    Returns:
        Tuple of (current_price, return_30d, volatility)
    """
    # Check cache first
    cached = get_cached_data(symbol)
    if cached:
        return cached["price"], cached["return_30d"], cached["volatility"]
    
    # Fetch from Yahoo Finance
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="60d")
    
    if len(hist) < 30:
        raise ValueError(f"Insufficient data for {symbol}: {len(hist)} days")
    
    current_price = hist["Close"].iloc[-1]
    price_30d_ago = hist["Close"].iloc[-30]
    
    return_30d = ((current_price / price_30d_ago) - 1) * 100
    
    # Calculate 30-day volatility (annualized)
    returns_30d = hist["Close"].iloc[-30:].pct_change().dropna()
    volatility = returns_30d.std() * np.sqrt(252) * 100
    
    # Cache the result
    save_to_cache(symbol, current_price, return_30d, volatility)
    
    return current_price, return_30d, volatility


def calculate_usd_strength_score(uup_return: float, udn_return: float) -> float:
    """Calculate USD strength score from -1.0 to 1.0."""
    # Normalize based on typical ranges (±4%)
    raw_score = (uup_return - udn_return) / 8.0
    return max(-1.0, min(1.0, raw_score))


def classify_carry_regime(uup_return: float, udn_return: float) -> str:
    """Classify carry regime based on momentum."""
    if uup_return > 2.0 and udn_return < -1.0:
        return "positive"  # USD carry advantage
    elif udn_return > 2.0 and uup_return < -1.0:
        return "negative"  # USD carry disadvantage
    else:
        return "neutral"


def classify_momentum_direction(uup_return: float, udn_return: float) -> str:
    """Classify momentum direction."""
    if uup_return > 2.0 and udn_return < -1.0:
        return "bullish"
    elif udn_return > 2.0 and uup_return < -1.0:
        return "bearish"
    else:
        return "neutral"


def classify_volatility_regime(volatility: float) -> str:
    """Classify volatility regime."""
    if volatility < 8:
        return "low"
    elif volatility < 15:
        return "medium"
    else:
        return "high"


def fetch_fx_metrics() -> FXMetrics:
    """Fetch complete FX metrics for UUP/UDN."""
    uup_price, uup_return, uup_vol = fetch_etf_data(UUP_SYMBOL)
    udn_price, udn_return, udn_vol = fetch_etf_data(UDN_SYMBOL)
    
    # Use average volatility for regime classification
    avg_volatility = (uup_vol + udn_vol) / 2
    
    usd_strength = calculate_usd_strength_score(uup_return, udn_return)
    carry_regime = classify_carry_regime(uup_return, udn_return)
    momentum = classify_momentum_direction(uup_return, udn_return)
    vol_regime = classify_volatility_regime(avg_volatility)
    
    return FXMetrics(
        timestamp=datetime.now().isoformat(),
        uup_price=uup_price,
        udn_price=udn_price,
        uup_return_30d=uup_return,
        udn_return_30d=udn_return,
        usd_strength_score=usd_strength,
        carry_regime=carry_regime,
        momentum_direction=momentum,
        volatility_regime=vol_regime
    )


def save_metrics(metrics: FXMetrics, filepath: Optional[Path] = None) -> Path:
    """Save metrics to JSON file."""
    if filepath is None:
        filepath = DATA_DIR / "fx_metrics.json"
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(metrics.to_json())
    
    return filepath


def load_latest_metrics(filepath: Optional[Path] = None) -> Optional[FXMetrics]:
    """Load latest metrics from file."""
    if filepath is None:
        filepath = DATA_DIR / "fx_metrics.json"
    
    if not filepath.exists():
        return None
    
    with open(filepath) as f:
        data = json.load(f)
    
    return FXMetrics(**data)


def get_signal_summary() -> dict:
    """Get human-readable signal summary."""
    metrics = fetch_fx_metrics()
    
    return {
        "timestamp": metrics.timestamp,
        "uup_30d_return": f"{metrics.uup_return_30d:.2f}%",
        "udn_30d_return": f"{metrics.udn_return_30d:.2f}%",
        "usd_strength": f"{metrics.usd_strength_score:.2f}",
        "carry_regime": metrics.carry_regime,
        "momentum": metrics.momentum_direction,
        "volatility": metrics.volatility_regime,
        "signal": "USD_STRENGTH" if metrics.carry_regime == "positive" else
                  "USD_WEAKNESS" if metrics.carry_regime == "negative" else
                  "NEUTRAL"
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FX Currency Carry data fetcher")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data")
    parser.add_argument("--save", action="store_true", help="Save to file")
    parser.add_argument("--signal", action="store_true", help="Print signal summary")
    parser.add_argument("--history", action="store_true", help="Show historical context")
    
    args = parser.parse_args()
    
    if args.fetch or args.save or args.signal or not any([args.fetch, args.save, args.signal, args.history]):
        metrics = fetch_fx_metrics()
        
        if args.save:
            path = save_metrics(metrics)
            print(f"Saved to {path}")
        
        if args.signal or not args.save:
            summary = get_signal_summary()
            print(json.dumps(summary, indent=2))
    
    if args.history:
        # Load cached data for historical context
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, price, return_30d, timestamp FROM fx_data ORDER BY timestamp DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        conn.close()
        
        print("Historical FX Data (last cached):")
        for row in rows:
            symbol, price, ret_30d, timestamp = row
            print(f"  {symbol}: ${price:.2f} ({ret_30d:+.2f}%) at {timestamp}")


if __name__ == "__main__":
    main()
