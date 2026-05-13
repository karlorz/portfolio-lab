#!/usr/bin/env python3
"""
Portfolio-Lab Alpha: Data Pipeline
Fetch market data, detect regime changes, trigger research if needed.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp

# Config
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "market.db"
SYMBOLS = {
    "core": ["SPY", "GLD", "TLT", "QQQ", "IEF"],
    "risk_indicators": ["^VIX", "DXY", "HYG", "LQD"],
    "alternatives": ["BTC-USD", "ETH-USD", "DBC"],
    "factors": ["MTUM", "VLUE", "USMV", "EFA", "VXUS"]
}
ALL_SYMBOLS = [s for group in SYMBOLS.values() for s in group]

def init_db():
    """Initialize SQLite with market data schema."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            updated_at TEXT,
            PRIMARY KEY (symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
        CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol);
        
        CREATE TABLE IF NOT EXISTS regime_log (
            id INTEGER PRIMARY KEY,
            date TEXT,
            regime TEXT,  -- 'low_vol', 'high_vol', 'crisis', 'recovery'
            vix_level REAL,
            correlation_spike BOOLEAN,
            trend_strength REAL,
            detected_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS data_quality (
            symbol TEXT PRIMARY KEY,
            last_fetch TEXT,
            records_count INTEGER,
            missing_dates TEXT,  -- JSON array
            staleness_hours INTEGER
        );
    """)
    conn.commit()
    return conn

async def fetch_yahoo(symbol: str, session: aiohttp.ClientSession) -> List[Dict]:
    """Fetch from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": "1d",
        "range": "5y",
        "events": "div,splits"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        async with session.get(url, params=params, headers=headers, timeout=30) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            
            result = data.get("chart", {}).get("result", [{}])[0]
            timestamps = result.get("timestamp", [])
            adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])

            records = []
            for i, ts in enumerate(timestamps):
                # Use adjclose if available, fall back to close
                close = (adjclose[i] if i < len(adjclose) and adjclose[i] is not None
                         else closes[i] if i < len(closes) and closes[i] is not None
                         else None)
                if close is None:
                    continue
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                records.append({
                    "date": dt,
                    "open": opens[i] if i < len(opens) and opens[i] is not None else close,
                    "high": highs[i] if i < len(highs) and highs[i] is not None else close,
                    "low": lows[i] if i < len(lows) and lows[i] is not None else close,
                    "close": close,
                    "volume": volumes[i] if i < len(volumes) and volumes[i] else 0
                })
            return records
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return []

def detect_regime(conn: sqlite3.Connection) -> Optional[str]:
    """Detect market regime from recent data."""
    cursor = conn.cursor()
    
    # Get recent VIX and SPY data
    cursor.execute("""
        SELECT date, close FROM prices 
        WHERE symbol IN ('VIX', 'SPY') 
        AND date >= date('now', '-63 days')
        ORDER BY date
    """)
    rows = cursor.fetchall()
    
    if len(rows) < 20:
        return None
    
    vix_prices = [r[1] for r in rows if r[0] == 'VIX']
    spy_prices = [r[1] for r in rows if r[0] == 'SPY']
    
    if not vix_prices or not spy_prices:
        return None
    
    current_vix = vix_prices[-1]
    vix_ma20 = sum(vix_prices[-20:]) / 20
    
    # Regime detection logic
    if current_vix > 30:
        regime = "crisis"
    elif current_vix > vix_ma20 * 1.5:
        regime = "vol_spike"
    elif current_vix < 15:
        regime = "low_vol"
    else:
        regime = "normal"
    
    # Log regime detection
    cursor.execute("""
        INSERT OR REPLACE INTO regime_log (date, regime, vix_level, detected_at)
        VALUES (date('now'), ?, ?, datetime('now'))
    """, (regime, current_vix))
    conn.commit()
    
    return regime

def check_data_quality(conn: sqlite3.Connection) -> Dict:
    """Check data freshness and completeness."""
    cursor = conn.cursor()
    quality = {}
    
    for symbol in ALL_SYMBOLS:
        cursor.execute("""
            SELECT MAX(date), COUNT(*), MAX(updated_at)
            FROM prices WHERE symbol = ?
        """, (symbol,))
        last_date, count, updated = cursor.fetchone()
        
        if last_date:
            last_dt = datetime.strptime(last_date, "%Y-%m-%d")
            staleness = (datetime.now() - last_dt).total_seconds() / 3600
        else:
            staleness = 9999
        
        quality[symbol] = {
            "last_date": last_date,
            "records": count or 0,
            "staleness_hours": int(staleness),
            "needs_refresh": staleness > 24
        }
    
    return quality

async def main():
    """Main data pipeline."""
    print(f"[{datetime.now()}] Portfolio-Lab Alpha: Data Pipeline Starting")
    
    conn = init_db()
    cursor = conn.cursor()
    
    # Check data quality first
    quality = check_data_quality(conn)
    symbols_to_fetch = [s for s, q in quality.items() if q["needs_refresh"]]
    
    if not symbols_to_fetch:
        print("All data fresh, no fetch needed")
    else:
        print(f"Fetching {len(symbols_to_fetch)} symbols: {symbols_to_fetch}")
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols_to_fetch:
                records = await fetch_yahoo(symbol, session)
                
                for r in records:
                    cursor.execute("""
                        INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """, (symbol, r["date"], r.get("open"), r.get("high"), r.get("low"), r["close"], r["volume"]))
                
                print(f"  {symbol}: {len(records)} records")
                await asyncio.sleep(0.5)  # Rate limit
        
        conn.commit()
    
    # Detect regime
    regime = detect_regime(conn)
    print(f"Current regime: {regime}")
    
    # Export for app consumption
    export_path = DATA_DIR / "prices.json"
    cursor.execute("SELECT symbol, date, close FROM prices ORDER BY symbol, date")
    all_data = {}
    for symbol, date, close in cursor.fetchall():
        if symbol not in all_data:
            all_data[symbol] = []
        all_data[symbol].append({"d": date, "p": close})
    
    with open(export_path, 'w') as f:
        json.dump(all_data, f)
    
    print(f"Exported to {export_path}")
    
    # Check if research trigger needed
    if regime in ["crisis", "vol_spike"]:
        print("REGIME_CHANGE_DETECTED: Signaling research agent")
        # This would queue a task for the research agent
        trigger_path = DATA_DIR / ".regime_trigger"
        trigger_path.write_text(json.dumps({
            "regime": regime,
            "timestamp": datetime.now().isoformat(),
            "vix": quality.get("VIX", {}).get("last_value")
        }))
    
    conn.close()
    print(f"[{datetime.now()}] Pipeline complete")

if __name__ == "__main__":
    asyncio.run(main())
