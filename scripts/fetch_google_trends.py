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
import random
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
# Batch CL: jitter between batches (deep-research: avoid thundering herd on 429)
RATE_LIMIT_JITTER_MIN = 8.0
RATE_LIMIT_JITTER_MAX = 20.0
# Batch CG: exponential backoff on HTTP 429 (initial, factor, max, retries)
RATE_LIMIT_BACKOFF_INITIAL = 30.0
RATE_LIMIT_BACKOFF_FACTOR = 2.0
RATE_LIMIT_BACKOFF_MAX = 300.0
RATE_LIMIT_MAX_RETRIES = 3
# Tasker/cron: distinct exit so operators can separate rate-limit from hard fail
EXIT_RATE_LIMITED = 3


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg or "Too Many" in msg or "rate limit" in msg.lower()


def fetch_trends(
    terms: list[str],
    days: int = 90,
    *,
    max_retries: int = RATE_LIMIT_MAX_RETRIES,
) -> tuple[dict[str, dict[str, int]], bool]:
    """Fetch Google Trends data for the given search terms.

    Args:
        terms: List of search terms to query.
        days: Number of days of historical data to fetch.
        max_retries: Retries per batch after HTTP 429 (exponential backoff).

    Returns:
        (results, rate_limited) where results is {term: {date_str: volume}}
        and rate_limited is True if any 429 exhausted retries without data.
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

    results: dict[str, dict[str, int]] = {}
    saw_rate_limit = False

    # Batch terms in groups of 5 (pytrends max per request)
    batch_size = 5
    for batch_start in range(0, len(terms), batch_size):
        batch = terms[batch_start : batch_start + batch_size]

        if batch_start > 0:
            # Fixed delay + jitter (Batch CL / deep-research anti-429)
            time.sleep(RATE_LIMIT_DELAY + random.uniform(
                RATE_LIMIT_JITTER_MIN, RATE_LIMIT_JITTER_MAX
            ))

        attempt = 0
        delay = RATE_LIMIT_BACKOFF_INITIAL
        while True:
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
                    break

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
                break

            except Exception as e:
                logger.warning("Failed to fetch batch %s: %s", batch, e)
                if _is_rate_limit_error(e):
                    saw_rate_limit = True
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            "Rate limited on %s after %d retries — giving up batch",
                            batch,
                            max_retries,
                        )
                        break
                    wait = min(delay, RATE_LIMIT_BACKOFF_MAX)
                    wait += random.uniform(0, min(15.0, wait * 0.25))  # jitter
                    logger.info(
                        "Rate limited (attempt %d/%d), waiting %.0fs...",
                        attempt,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    delay *= RATE_LIMIT_BACKOFF_FACTOR
                    continue
                break

    return results, saw_rate_limit


def merge_trends_cache(
    cached: dict[str, dict[str, int]],
    fresh: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Merge incremental fetch into historical cache (Batch CL).

    Fresh date keys overwrite cache for the same term; terms only in cache
    are retained so a partial 429 still advances recent windows.
    """
    if not cached:
        return dict(fresh)
    if not fresh:
        return dict(cached)
    out: dict[str, dict[str, int]] = {}
    for term in set(cached) | set(fresh):
        merged = dict(cached.get(term) or {})
        merged.update(fresh.get(term) or {})
        # keep chronological key order for consumers
        out[term] = {k: merged[k] for k in sorted(merged)}
    return out


def load_trends_cache(path: Path) -> dict[str, dict[str, int]]:
    """Load existing google_trends.json when shape is term→{date→volume}."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for term, series in raw.items():
        if not isinstance(series, dict):
            continue
        cleaned: dict[str, int] = {}
        for d, v in series.items():
            try:
                cleaned[str(d)] = int(v)
            except (TypeError, ValueError):
                continue
        if cleaned:
            out[str(term)] = cleaned
    return out


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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=RATE_LIMIT_MAX_RETRIES,
        help=f"429 retries per batch (default: {RATE_LIMIT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--no-cache-merge",
        action="store_true",
        help="Replace artifact entirely (default: merge into existing cache)",
    )
    args = parser.parse_args()

    terms = args.terms if args.terms else SEARCH_TERMS
    output_path = Path(args.output)
    cached = {} if args.no_cache_merge else load_trends_cache(output_path)

    logger.info(
        "Fetching %d days of Google Trends data for %d terms (cache_terms=%d)...",
        args.days,
        len(terms),
        len(cached),
    )

    data, rate_limited = fetch_trends(terms, days=args.days, max_retries=args.max_retries)

    if not data:
        if rate_limited:
            logger.error(
                "No data fetched due to Google Trends rate limit (HTTP 429). "
                "Preserving existing artifact if present; exit=%s",
                EXIT_RATE_LIMITED,
            )
            sys.exit(EXIT_RATE_LIMITED)
        logger.error("No data fetched. Check network connectivity and rate limits.")
        sys.exit(1)

    if not args.no_cache_merge:
        data = merge_trends_cache(cached, data)

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
