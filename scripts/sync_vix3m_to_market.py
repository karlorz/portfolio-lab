#!/usr/bin/env python3
"""
Sync VIX3M data from prices.json to market.db.
This script is part of the VIX3M data acquisition work item.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def sync_vix3m():
    """Sync VIX3M data from prices.json to market.db"""
    project_root = Path(__file__).parent.parent
    prices_path = project_root / "public" / "data" / "prices.json"
    db_path = project_root / "data" / "market.db"
    
    print(f"=== VIX3M Sync Script ===")
    print(f"Source: {prices_path}")
    print(f"Target: {db_path}")
    
    # Load VIX3M data from prices.json
    with open(prices_path) as f:
        prices_data = json.load(f)
    
    vix3m_data = prices_data.get('^VIX3M', [])
    print(f"\nFound {len(vix3m_data)} VIX3M records in prices.json")
    
    if not vix3m_data:
        print("No VIX3M data found in prices.json")
        return False
    
    # Connect to market.db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check current VIX3M count
    cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = '^VIX3M'")
    current_count = cursor.fetchone()[0]
    print(f"Current VIX3M rows in market.db: {current_count}")
    
    # Prepare insert data
    insert_data = []
    now = datetime.now().isoformat()
    for record in vix3m_data:
        insert_data.append((
            '^VIX3M',           # symbol
            record['d'],        # date
            record['p'],        # open (using close price as proxy)
            record['p'],        # high
            record['p'],        # low
            record['p'],        # close
            0,                  # volume (not available for index)
            now                 # updated_at
        ))
    
    # Insert data
    try:
        cursor.executemany(
            "INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            insert_data
        )
        conn.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = '^VIX3M'")
        new_count = cursor.fetchone()[0]
        print(f"\nSync complete!")
        print(f"Added {new_count - current_count} new rows")
        print(f"Total VIX3M rows now: {new_count}")
        
        return True
        
    except Exception as e:
        print(f"Error syncing VIX3M: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = sync_vix3m()
    exit(0 if success else 1)
