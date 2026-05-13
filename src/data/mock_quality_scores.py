"""
Mock Quality Score Generator for Factor ETFs (v3.00 Phase 1 Testing)

Generates synthetic quality scores based on known ETF characteristics for testing
the factor rotation infrastructure before Alpha Vantage integration.

This is a TEMPORARY testing utility - replace with real fundamentals fetcher
once Alpha Vantage API key is configured.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known approximate fundamental characteristics for factor ETFs (based on underlying baskets)
# Values derived from iShares methodology documents and historical factor research
ETF_KNOWN_CHARACTERISTICS = {
    "QUAL": {
        # Quality factor: High ROE, low debt, stable earnings
        "roe": 0.22,           # ~22% ROE (high quality companies)
        "debt_equity": 0.35,   # Low leverage
        "earnings_stability": 0.75,  # Very stable earnings
        "profitability": 0.70,  # High gross margins
        "description": "High ROE, low leverage, stable earnings"
    },
    "MTUM": {
        # Momentum factor: Mixed quality, higher volatility
        "roe": 0.16,           # Market-average ROE
        "debt_equity": 0.55,   # Moderate leverage
        "earnings_stability": 0.50,  # Moderate stability
        "profitability": 0.55,  # Average profitability
        "description": "Momentum-driven, average fundamentals"
    },
    "USMV": {
        # Low volatility: Stable, defensive characteristics
        "roe": 0.14,           # Slightly below market (utilities, consumer staples)
        "debt_equity": 0.60,   # Moderate-high (utilities have debt)
        "earnings_stability": 0.80,  # Very stable (low vol names)
        "profitability": 0.50,  # Moderate
        "description": "Defensive, stable earnings, moderate leverage"
    },
    "VLUE": {
        # Value factor: Lower margins, higher leverage (cyclicals, financials)
        "roe": 0.12,           # Below average (value trap risk)
        "debt_equity": 0.75,   # Higher leverage (financials, industrials)
        "earnings_stability": 0.40,  # Less stable (cyclical)
        "profitability": 0.45,  # Lower margins
        "description": "Lower margins, higher leverage, cyclical"
    }
}


def calculate_mock_quality_score(roe: float, debt_equity: float, 
                                 earnings_stability: float, profitability: float) -> float:
    """
    Calculate composite quality score using same weights as production system.
    
    Quality Weights (from Asness QMJ methodology):
    - ROE: 30% (profitability)
    - Debt/Equity: 25% (safety - lower is better)
    - Earnings stability: 25% (consistency)
    - Profitability: 20% (margins)
    """
    # Normalize to 0-1 scale
    roe_norm = min(max(roe / 0.25, 0), 1)  # 25% ROE = excellent
    de_norm = min(max(1 - (debt_equity / 1.5), 0), 1)  # Lower debt is better
    earn_norm = min(max(earnings_stability, 0), 1)
    prof_norm = min(max(profitability, 0), 1)
    
    # Calculate weighted score
    score = (
        0.30 * roe_norm +
        0.25 * de_norm +
        0.25 * earn_norm +
        0.20 * prof_norm
    )
    
    return round(score, 4)


def add_noise_to_metrics(base_metrics: Dict, date_str: str, seed_offset: int = 0) -> Dict:
    """
    Add slight random variation to base metrics for time-series realism.
    Uses deterministic pseudo-random based on date for reproducibility.
    """
    from hashlib import md5
    
    # Create deterministic "random" variation based on date
    date_hash = int(md5(f"{date_str}_{seed_offset}".encode()).hexdigest(), 16)
    
    # Small variations (±5% of base value)
    variation = (date_hash % 100 - 50) / 1000  # -0.05 to +0.05
    
    return {
        "roe": max(0.05, min(0.40, base_metrics["roe"] * (1 + variation))),
        "debt_equity": max(0.1, min(1.5, base_metrics["debt_equity"] * (1 + variation * 0.5))),
        "earnings_stability": max(0.2, min(0.95, base_metrics["earnings_stability"] + variation)),
        "profitability": max(0.2, min(0.85, base_metrics["profitability"] + variation))
    }


def populate_mock_quality_scores(db_path: Path, days: int = 252) -> int:
    """
    Populate quality_scores table with mock data for all factor ETFs.
    
    Args:
        db_path: Path to factor_data.db
        days: Number of days of history to generate (default 252 = 1 year)
    
    Returns:
        Number of records inserted
    """
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 0
    
    # Get date range from factor_prices table
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT DISTINCT symbol, MIN(date), MAX(date) 
            FROM factor_prices 
            GROUP BY symbol
        """)
        date_ranges = cursor.fetchall()
    
    if not date_ranges:
        logger.warning("No price data found - cannot populate quality scores")
        return 0
    
    records_inserted = 0
    
    with sqlite3.connect(db_path) as conn:
        for symbol, min_date, max_date in date_ranges:
            if symbol not in ETF_KNOWN_CHARACTERISTICS:
                logger.warning(f"Unknown symbol: {symbol}")
                continue
            
            base = ETF_KNOWN_CHARACTERISTICS[symbol]
            
            # Generate monthly quality scores (fundamentals don't change daily)
            cursor = conn.execute("""
                SELECT DISTINCT date FROM factor_prices 
                WHERE symbol = ? 
                AND strftime('%d', date) = '01'  -- First of each month
                ORDER BY date DESC
                LIMIT 12
            """, (symbol,))
            
            monthly_dates = [row[0] for row in cursor.fetchall()]
            
            for date_str in monthly_dates:
                # Add slight variation for realism
                metrics = add_noise_to_metrics(base, date_str)
                
                score = calculate_mock_quality_score(
                    metrics["roe"],
                    metrics["debt_equity"],
                    metrics["earnings_stability"],
                    metrics["profitability"]
                )
                
                conn.execute("""
                    INSERT OR REPLACE INTO quality_scores
                    (symbol, date, roe, debt_equity, earnings_stability, profitability, composite_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, date_str,
                    round(metrics["roe"], 4),
                    round(metrics["debt_equity"], 4),
                    round(metrics["earnings_stability"], 4),
                    round(metrics["profitability"], 4),
                    score
                ))
                records_inserted += 1
        
        conn.commit()
    
    logger.info(f"Inserted {records_inserted} mock quality score records")
    return records_inserted


def verify_quality_scores(db_path: Path) -> Dict:
    """Verify quality scores were populated correctly."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Count by symbol
        cursor = conn.execute("""
            SELECT symbol, COUNT(*) as count, AVG(composite_score) as avg_score
            FROM quality_scores
            GROUP BY symbol
        """)
        
        results = {}
        for row in cursor.fetchall():
            results[row["symbol"]] = {
                "count": row["count"],
                "avg_score": round(row["avg_score"], 4)
            }
    
    return results


def main():
    """CLI for mock quality score generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mock Quality Score Generator for Factor ETFs v3.00"
    )
    parser.add_argument("command", choices=["populate", "verify", "both"])
    parser.add_argument("--db-path", type=Path, 
                      default=Path("data/factors/factor_data.db"))
    
    args = parser.parse_args()
    
    if args.command in ["populate", "both"]:
        count = populate_mock_quality_scores(args.db_path)
        print(f"\nPopulated {count} quality score records")
    
    if args.command in ["verify", "both"]:
        results = verify_quality_scores(args.db_path)
        print("\nQuality Score Verification:")
        print("-" * 50)
        for symbol, data in sorted(results.items()):
            print(f"  {symbol}: {data['count']} records, avg_score={data['avg_score']:.4f}")
        
        # Show expected ranking
        print("\n  Expected Quality Ranking (highest to lowest):")
        print("    1. QUAL (high ROE, low debt)")
        print("    2. MTUM (average metrics)")
        print("    3. USMV (stable but lower margins)")
        print("    4. VLUE (higher leverage, cyclical)")


if __name__ == "__main__":
    main()
