#!/usr/bin/env python3
"""
Yield Curve Data Ingestion Script
Populates yield_curve_data table from FRED cache for v3.11 Duration Overlay
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
FRED_PATH = DATA_DIR / "fred_data.json"


def init_yield_curve_table():
    """Create yield_curve_data table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS yield_curve_data (
            date TEXT PRIMARY KEY,
            dgs10 REAL,
            dgs2 REAL,
            dgs30 REAL,
            dgs5 REAL,
            spread_2s10s REAL,
            created_at TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ yield_curve_data table initialized")


def ingest_fred_data():
    """Ingest FRED data into yield_curve_data table."""
    if not FRED_PATH.exists():
        print("❌ fred_data.json not found")
        return 0
    
    with open(FRED_PATH) as f:
        fred_data = json.load(f)
    
    if "DGS10" not in fred_data or "DGS2" not in fred_data:
        print("❌ DGS10 or DGS2 data missing from FRED cache")
        return 0
    
    # Build date-indexed yield data
    yields_by_date = {}
    
    for entry in fred_data.get("DGS10", []):
        date = entry["date"][:10]  # Extract YYYY-MM-DD
        if date not in yields_by_date:
            yields_by_date[date] = {}
        yields_by_date[date]["dgs10"] = entry["value"]
    
    for entry in fred_data.get("DGS2", []):
        date = entry["date"][:10]
        if date not in yields_by_date:
            yields_by_date[date] = {}
        yields_by_date[date]["dgs2"] = entry["value"]
    
    # Calculate spreads and prepare records
    records = []
    for date, data in sorted(yields_by_date.items()):
        if "dgs10" in data and "dgs2" in data:
            spread = data["dgs10"] - data["dgs2"]
            records.append((
                date,
                data["dgs10"],
                data["dgs2"],
                data.get("dgs30"),
                data.get("dgs5"),
                spread,
                datetime.now().isoformat()
            ))
    
    # Insert into database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted = 0
    for record in records:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO yield_curve_data 
                (date, dgs10, dgs2, dgs30, dgs5, spread_2s10s, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, record)
            inserted += 1
        except Exception as e:
            print(f"⚠️  Failed to insert {record[0]}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Ingested {inserted} yield curve records")
    
    # Show latest
    if records:
        latest = records[-1]
        print(f"📊 Latest: {latest[0]} | 10Y: {latest[1]:.2f}% | 2Y: {latest[2]:.2f}% | Spread: {latest[5]:.2f}%")
        
        # Classify regime
        spread = latest[5]
        if spread < -0.25:
            regime = "INVERTED ⚠️"
        elif spread > 0.75:
            regime = "STEEP 📈"
        else:
            regime = "FLAT ➡️"
        print(f"🏷️  Regime: {regime}")
    
    return inserted


if __name__ == "__main__":
    print("=" * 60)
    print("v3.11 Yield Curve Data Ingestion")
    print("=" * 60)
    init_yield_curve_table()
    count = ingest_fred_data()
    print("=" * 60)
    print(f"Done! Ingested {count} records into yield_curve_data table")
