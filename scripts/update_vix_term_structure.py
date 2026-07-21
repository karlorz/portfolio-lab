#!/usr/bin/env python3
"""
Update vix_term_structure.json with VIX3M data from market.db.

This script populates the VIX term structure data file with VIX/VIX3M ratios
and regime classifications for use by the VIX term structure signal generator.
"""

import sqlite3
import json
import sys
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow `uv run python scripts/update_vix_term_structure.py` from repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def update_vix_term_structure():
    """Update vix_term_structure.json with VIX/VIX3M data."""
    
    # Paths
    data_dir = _PROJECT_ROOT / "data"
    market_db = data_dir / "market.db"
    vix_ts_path = data_dir / "vix_term_structure.json"
    
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
        # Load VIX3M (always present in this lab); ^VIX may be absent.
        vix3m_df = pd.read_sql_query(
            "SELECT date, close as vix3m FROM prices WHERE symbol = '^VIX3M' ORDER BY date",
            conn,
        )
        vix_df = pd.read_sql_query(
            "SELECT date, close as vix FROM prices WHERE symbol = '^VIX' ORDER BY date",
            conn,
        )

        if vix_df.empty and not vix3m_df.empty:
            # VIX3M-only proxy: use VIX3M as both spot and front; mild contango
            # placeholder for second/third (same as VIXDataManager hydrate path).
            logger.warning(
                "^VIX missing in market.db — building term structure from ^VIX3M only "
                "(%d rows); contango_spot_1m will be 0 when spot==front",
                len(vix3m_df),
            )
            merged = vix3m_df.copy()
            merged["vix"] = merged["vix3m"]
        elif vix3m_df.empty:
            logger.error("No ^VIX3M rows in market.db")
            return False
        else:
            merged = pd.merge(vix_df, vix3m_df, on="date", how="inner")

        logger.info(f"Merged VIX/VIX3M data: {len(merged)} rows")
        logger.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

        # Calculate VIX/VIX3M ratio (1.0 when VIX3M-only proxy)
        merged["vix_vix3m_ratio"] = merged["vix"] / merged["vix3m"]

        # Update entries via VIXTermStructure.from_dict so contango fields always present
        from src.data.vix_futures import VIXTermStructure

        updated_count = 0
        for _, row in merged.iterrows():
            date = row["date"]
            vix = float(row["vix"])
            vix3m = float(row["vix3m"])
            ratio = float(row["vix_vix3m_ratio"])

            if ratio < 0.8:
                regime = "extreme_backwardation"
            elif ratio < 0.95:
                regime = "backwardation"
            elif ratio < 1.0:
                regime = "flat"
            elif ratio < 1.15:
                regime = "contango"
            else:
                regime = "extreme_contango"

            raw = {
                "date": date,
                "vix_spot": vix,
                "front_month": vix3m,
                "third_month": vix3m,  # no VIX6M — proxy
                "days_to_expiry_front": 0,
            }
            try:
                ts = VIXTermStructure.from_dict(raw)
                entry = ts.to_dict()
            except (TypeError, ValueError, KeyError):
                entry = {
                    "date": date,
                    "vix_spot": vix,
                    "front_month": vix3m,
                    "second_month": vix3m,
                    "third_month": vix3m,
                    "contango_1m_2m": 0.0,
                    "contango_spot_1m": (vix3m / vix - 1.0) * 100.0 if vix else 0.0,
                    "is_contango": bool(vix3m >= vix) if vix else True,
                    "days_to_expiry_front": 0,
                }
            entry["vix_vix3m_ratio"] = ratio
            entry["regime"] = regime
            entry["source"] = "market.db"
            entry["as_of"] = date
            current_data[date] = entry
            updated_count += 1

        logger.info(f"Updated {updated_count} entries")

        with open(vix_ts_path, "w") as f:
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