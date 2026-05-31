#!/usr/bin/env python3
"""Fetch Google Trends data for macro fear indicators.

Saves search volume data to data/google_trends.json for consumption
by src/signals/google_trends_signal.py.

Usage:
    python scripts/fetch_google_trends.py [--days 90] [--output data/google_trends.json]

Requires: pytrends (pip install pytrends)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Macro fear indicators — same as TREND_TERMS in google_trends_signal.py
SEARCH_TERMS = [
    "recession",
    "inflation",
    "stock market crash",
    "interest rates",
]

# Rate limiting: pytrends has aggressive rate limits
RATE_LIMIT_DELAY = 2.0  # seconds between requests


def fetch_trends(terms: list[str], days: int = 90) -> dict[str, dict[str, int]]:
    """Fetch Google Trends data for the given search terms.

    Args:
        terms: List of search terms to query.
        days: Number of days of historical data to fetch.

    Returns:
        Dict of {term: {date_str: search_volume}}.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed. Run: pip install pytrends")
        sys.exit(1)

    pytrends = TrendReq(hl="en-US", tz=480)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    results = {}

    # Batch terms in groups of 5 (pytrends max per request)
    batch_size = 5
    for batch_start in range(0, len(terms), batch_size):
        batch = terms[batch_start : batch_start + batch_size]

        if batch_start > 0:
            time.sleep(RATE_LIMIT_DELAY)

        try:
            logger.info("Fetching trends for %s...", batch)
            pytrends.build_payload(
                batch,
                cat=0,
                timeframe=timeframe,
                geo="",
                gprop="",
            )
            df = pytrends.interest_over_time()

            if df.empty:
                logger.warning("No data returned for %s", batch)
                continue

            for term in batch:
                if term not in df.columns:
                    logger.warning("No column for '%s' in response", term)
                    continue
                term_data = {}
                for date_idx, row in df.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")
                    term_data[date_str] = int(row[term])
                results[term] = term_data
                logger.info("  Got %d data points for '%s'", len(term_data), term)

        except Exception as e:
            logger.warning("Failed to fetch batch %s: %s", batch, e)
            if "429" in str(e) or "Too Many" in str(e):
                logger.info("Rate limited, waiting 60s...")
                time.sleep(60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Google Trends data")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/google_trends.json",
        help="Output file path (default: data/google_trends.json)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        nargs="*",
        default=None,
        help="Custom search terms (default: recession, inflation, stock market crash, interest rates)",
    )
    args = parser.parse_args()

    terms = args.terms if args.terms else SEARCH_TERMS
    output_path = Path(args.output)

    logger.info("Fetching %d days of Google Trends data for %d terms...", args.days, len(terms))

    data = fetch_trends(terms, days=args.days)

    if not data:
        logger.error("No data fetched. Check network connectivity and rate limits.")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info("Saved %d terms to %s", len(data), output_path)

    # Summary
    for term, term_data in data.items():
        values = list(term_data.values())
        avg = sum(values) / len(values) if values else 0
        recent = values[-7:] if len(values) >= 7 else values
        recent_avg = sum(recent) / len(recent) if recent else 0
        z = (recent_avg - avg) / (max(values) - min(values) + 1) if values else 0
        logger.info("  %s: avg=%.0f, recent_7d=%.0f, z=%.2f", term, avg, recent_avg, z)


if __name__ == "__main__":
    main()
