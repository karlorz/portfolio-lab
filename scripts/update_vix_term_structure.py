#!/usr/bin/env python3
"""
Update vix_term_structure.json with VIX3M data from market.db.

This script populates the VIX term structure data file with VIX/VIX3M ratios
and regime classifications for use by the VIX term structure signal generator.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_vix_term_structure():
    """Update vix_term_structure.json with VIX/VIX3M data."""
    
    # Paths
    data_dir = Path('data')
    market_db = data_dir / 'market.db'
    vix_ts_path = data_dir / 'vix_term_structure.json'
    
    # Load current data
    if vix_ts_path.exists():
        with open(vix_ts_path) as f:
            current_data = json.load(f)
        logger.info(f"Current vix_term_structure.json has {len(current_data)} entries")
    else:
        current_data = {}
        logger.info("Creating new vix_term_structure.json")
    
    # Connect to market.db
    conn = sqlite3.connect(market_db)
    
    try:
        # Load VIX data
        vix_df = pd.read_sql_query(
            "SELECT date, close as vix FROM prices WHERE symbol = '^VIX' ORDER BY date",
            conn
        )
        
        # Load VIX3M data
        vix3m_df = pd.read_sql_query(
            "SELECT date, close as vix3m FROM prices WHERE symbol = '^VIX3M' ORDER BY date",
            conn
        )
        
        # Merge on date
        merged = pd.merge(vix_df, vix3m_df, on='date', how='inner')
        logger.info(f"Merged VIX/VIX3M data: {len(merged)} rows")
        logger.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        # Calculate VIX/VIX3M ratio
        merged['vix_vix3m_ratio'] = merged['vix'] / merged['vix3m']
        
        # Update entries
        updated_count = 0
        for _, row in merged.iterrows():
            date = row['date']
            vix = row['vix']
            vix3m = row['vix3m']
            ratio = row['vix_vix3m_ratio']
            
            # Determine regime based on ratio
            if ratio < 0.8:
                regime = "extreme_backwardation"
                is_contango = False
            elif ratio < 0.95:
                regime = "backwardation"
                is_contango = False
            elif ratio < 1.0:
                regime = "flat"
                is_contango = True
            elif ratio < 1.15:
                regime = "contango"
                is_contango = True
            else:
                regime = "extreme_contango"
                is_contango = True
            
            # Create entry
            entry = {
                "date": date,
                "vix_spot": vix,
                "front_month": vix3m,  # VIX3M is front month (3-month)
                "third_month": None,  # VIX6M not available
                "vix_vix3m_ratio": ratio,
                "regime": regime,
                "is_contango": is_contango,
                "contango_spot_1m": (vix3m - vix) / vix * 100,  # Percentage
                "contango_1m_2m": 0.0,  # Not available without VIX6M
                "days_to_expiry_front": None
            }
            
            current_data[date] = entry
            updated_count += 1
        
        logger.info(f"Updated {updated_count} entries")
        
        # Save updated data
        with open(vix_ts_path, 'w') as f:
            json.dump(current_data, f, indent=2)
        
        logger.info(f"Saved vix_term_structure.json with {len(current_data)} total entries")
        
        # Verify the data
        with open(vix_ts_path) as f:
            verify_data = json.load(f)
        
        # Check regime distribution
        regimes = {}
        for entry in verify_data.values():
            regime = entry.get('regime', 'unknown')
            regimes[regime] = regimes.get(regime, 0) + 1
        
        logger.info("Regime distribution:")
        for regime, count in sorted(regimes.items()):
            logger.info(f"  {regime}: {count} ({count/len(verify_data)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating vix_term_structure.json: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = update_vix_term_structure()
    if success:
        logger.info("Successfully updated vix_term_structure.json")
    else:
        logger.error("Failed to update vix_term_structure.json")
        exit(1)