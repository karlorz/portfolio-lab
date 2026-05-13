#!/usr/bin/env python3
"""
Backfill factor ETF data from existing pipeline (v3.00 Phase 1)

Populates the factor_data.db with historical prices for MTUM, QUAL, USMV, VLUE
using existing prices.json data.
"""

import json
import sqlite3
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.factor_data import FactorDataManager, FACTOR_ETFS, fetch_factor_prices_from_pipeline

def backfill_factor_data():
    """Populate factor database with historical data from pipeline."""
    print("v3.00 Phase 1: Factor Data Backfill")
    print("=" * 50)
    
    # Initialize manager
    manager = FactorDataManager()
    
    # Load prices.json once
    prices_path = Path("public/data/prices.json")
    if not prices_path.exists():
        print(f"ERROR: prices.json not found at {prices_path}")
        return 1
    
    with open(prices_path, 'r') as f:
        prices_data = json.load(f)
    
    print(f"\nLoaded prices data with {len(prices_data)} symbols")
    
    # Check which factor ETFs are available
    available = [s for s in FACTOR_ETFS.keys() if s in prices_data]
    missing = [s for s in FACTOR_ETFS.keys() if s not in prices_data]
    
    print(f"\nAvailable factor ETFs: {available}")
    if missing:
        print(f"Missing from prices.json: {missing}")
        print("Run 'bun run fetch-data' to update price data with factor ETFs")
    
    # Backfill available ETFs
    total_records = 0
    for symbol in available:
        print(f"\nProcessing {symbol} ({FACTOR_ETFS[symbol].factor})...")
        
        # Fetch from pipeline
        records = fetch_factor_prices_from_pipeline(symbol, prices_data)
        
        if records:
            # Store in database
            manager.store_prices(symbol, records)
            total_records += len(records)
            print(f"  ✓ Stored {len(records)} records")
            
            # Calculate and store returns
            stored = manager.store_returns(symbol)
            if stored:
                print(f"  ✓ Calculated performance metrics")
        else:
            print(f"  ✗ No data available")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Backfill Summary:")
    print("=" * 50)
    
    with sqlite3.connect(manager.db_path) as conn:
        for table in ["factor_prices", "quality_scores", "factor_performance"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} records")
    
    print(f"\nTotal price records: {total_records}")
    print("\nNext steps for Phase 1:")
    print("  1. Add QUAL, USMV, VLUE to fetcher.ts CORE_SYMBOLS")
    print("  2. Run 'bun run fetch-data' to get all factor ETFs")
    print("  3. Re-run backfill for complete coverage")
    
    return 0

if __name__ == "__main__":
    sys.exit(backfill_factor_data())
