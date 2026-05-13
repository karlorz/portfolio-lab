#!/usr/bin/env python3
"""Backfill actual directions for signal predictions."""
import sqlite3
from pathlib import Path

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
market_conn = sqlite3.connect(DATA_DIR / "market.db")
market_cursor = market_conn.cursor()

# Get all predictions without actual_direction
market_cursor.execute("""
    SELECT p.id, p.timestamp, p.source, p.predicted_direction
    FROM signal_predictions p
    WHERE p.actual_direction IS NULL
    ORDER BY p.timestamp
""")
rows = market_cursor.fetchall()
print(f"Found {len(rows)} predictions without actual_direction")

# Get prices as lookup
market_cursor.execute("SELECT date, close FROM prices WHERE symbol = 'SPY' ORDER BY date")
price_rows = market_cursor.fetchall()
prices = {row[0]: row[1] for row in price_rows}
print(f"Have {len(prices)} price points")

# Update each prediction
updated = 0
for row in rows:
    id_, timestamp, source, predicted = row
    date = timestamp[:10]
    
    # Get current and next day prices
    curr_price = prices.get(date)
    
    # Find next trading day
    dates = sorted(prices.keys())
    if date in dates:
        idx = dates.index(date)
        if idx + 1 < len(dates):
            next_date = dates[idx + 1]
            next_price = prices.get(next_date)
            
            if curr_price and next_price:
                ret = (next_price - curr_price) / curr_price
                actual = 1 if ret > 0 else (-1 if ret < 0 else 0)
                
                market_cursor.execute("""
                    UPDATE signal_predictions 
                    SET actual_direction = ?, accuracy_calculated = 1
                    WHERE id = ?
                """, (actual, id_))
                updated += 1
                if updated <= 10:
                    print(f"  {date} -> {next_date}: pred={predicted}, actual={actual}, ret={ret:.4%}")

market_conn.commit()
print(f"\nUpdated {updated} predictions with actual_direction")
market_conn.close()
