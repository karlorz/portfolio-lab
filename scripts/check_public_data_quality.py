#!/usr/bin/env python3
"""Offline quality audit for historical public price data artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data import market_db_sync


REPORT_SCHEMA_VERSION = "public-price-data-quality-cli/v1"
DEFAULT_MAX_LATEST_LAG_DAYS = 5
BLOCKING_ISSUES = {
    "duplicate_dates",
    "empty_symbols",
    "extreme_returns",
    "invalid_dates",
    "invalid_prices",
    "missing_required_keys",
    "non_monotonic_rows",
    "non_object_records",
    "stale_latest_dates",
}


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} must be a YYYY-MM-DD date") from exc


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be a positive number")
    return parsed


def _prices_path(app_dir: Path, prices: Path | None) -> Path:
    return prices if prices is not None else app_dir / "public" / "data" / "prices.json"


def _latest_valid_date(records: list[dict[str, Any]]) -> str | None:
    latest: str | None = None
    for record in records:
        date = record.get("d") if isinstance(record, dict) else None
        if not isinstance(date, str):
            continue
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue
        latest = date if latest is None or date > latest else latest
    return latest


def _reference_date(prices: dict[str, list[dict[str, Any]]], explicit: datetime | None) -> datetime | None:
    if explicit is not None:
        return explicit
    latest_dates = [
        latest
        for records in prices.values()
        if (latest := _latest_valid_date(records)) is not None
    ]
    if not latest_dates:
        return None
    return _parse_iso_date(max(latest_dates))


def _ensure_issue_count(report: dict[str, Any], key: str) -> None:
    issue_counts = report.setdefault("issue_counts", {})
    if isinstance(issue_counts, dict):
        issue_counts.setdefault(key, 0)
    for symbol in report.get("symbols", []):
        if isinstance(symbol, dict) and isinstance(symbol.get("issue_counts"), dict):
            symbol["issue_counts"].setdefault(key, 0)


def _symbol_reports_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    symbols = report.get("symbols")
    if not isinstance(symbols, list):
        return {}
    return {
        str(symbol.get("symbol")): symbol
        for symbol in symbols
        if isinstance(symbol, dict) and symbol.get("symbol")
    }


def _apply_stale_latest_date_check(
    report: dict[str, Any],
    prices: dict[str, list[dict[str, Any]]],
    *,
    reference_date: datetime | None,
    max_latest_lag_days: int,
) -> list[dict[str, Any]]:
    _ensure_issue_count(report, "stale_latest_dates")
    if reference_date is None:
        return []

    stale_symbols: list[dict[str, Any]] = []
    symbol_reports = _symbol_reports_by_name(report)
    issue_counts = report["issue_counts"]
    for symbol, records in prices.items():
        latest = _latest_valid_date(records)
        if latest is None:
            continue
        latest_dt = _parse_iso_date(latest)
        lag_days = (reference_date - latest_dt).days
        if lag_days <= max_latest_lag_days:
            continue
        stale_symbols.append({
            "symbol": symbol,
            "latest_date": latest,
            "reference_date": reference_date.strftime("%Y-%m-%d"),
            "latest_lag_days": lag_days,
        })
        issue_counts["stale_latest_dates"] += 1
        issue_counts["total"] += 1
        symbol_report = symbol_reports.get(symbol)
        if symbol_report is not None and isinstance(symbol_report.get("issue_counts"), dict):
            symbol_report["issue_counts"]["stale_latest_dates"] += 1
            symbol_report["issue_counts"]["total"] += 1
            symbol_report["status"] = market_db_sync.QUALITY_STATUS_FAIL

    if stale_symbols:
        report["blocking"] = True
        report["status"] = market_db_sync.QUALITY_STATUS_FAIL
        report["first_blocking_error"] = report.get("first_blocking_error") or (
            f"Stale latest price date for {stale_symbols[0]['symbol']}: "
            f"{stale_symbols[0]['latest_date']} "
            f"({stale_symbols[0]['latest_lag_days']}d lag)"
        )
    return stale_symbols


def _issue_messages(report: dict[str, Any], stale_symbols: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    issue_counts = report.get("issue_counts")
    counts = issue_counts if isinstance(issue_counts, dict) else {}

    stale_by_symbol = ", ".join(
        f"{row['symbol']} latest {row['latest_date']} ({row['latest_lag_days']}d lag)"
        for row in stale_symbols
    )
    for issue, count in counts.items():
        if issue == "total" or not isinstance(count, int) or count <= 0:
            continue
        message = f"{issue}={count}"
        if issue == "stale_latest_dates" and stale_by_symbol:
            message = f"{message}: {stale_by_symbol}"
        if issue in BLOCKING_ISSUES:
            errors.append(message)
        else:
            warnings.append(message)
    return errors, warnings


def audit_public_prices(
    prices_path: str | Path,
    *,
    reference_date: datetime | None = None,
    max_latest_lag_days: int = DEFAULT_MAX_LATEST_LAG_DAYS,
    critical_return_pct: float = market_db_sync.DEFAULT_CRITICAL_RETURN_PCT,
    split_like_return_pct: float = market_db_sync.DEFAULT_SPLIT_LIKE_RETURN_PCT,
) -> dict[str, Any]:
    """Return a machine-readable offline quality report for compact public prices."""
    resolved_prices_path = Path(prices_path)
    prices = market_db_sync._load_prices_payload(resolved_prices_path)
    report = market_db_sync._audit_prices_payload(
        prices,
        critical_return_pct=critical_return_pct,
        split_like_return_pct=split_like_return_pct,
    )
    stale_symbols = _apply_stale_latest_date_check(
        report,
        prices,
        reference_date=_reference_date(prices, reference_date),
        max_latest_lag_days=max_latest_lag_days,
    )
    errors, warnings = _issue_messages(report, stale_symbols)
    row_count = sum(len(records) for records in prices.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices_path": str(resolved_prices_path),
        "status": report["status"],
        "blocking": bool(report["blocking"]),
        "first_blocking_error": report.get("first_blocking_error"),
        "symbols_checked": report["symbols_checked"],
        "rows_checked": row_count,
        "issue_counts": report["issue_counts"],
        "symbols": report["symbols"],
        "stale_symbols": stale_symbols,
        "errors": errors,
        "warnings": warnings,
    }


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path.cwd(), help="Portfolio Lab checkout directory")
    parser.add_argument("--prices", type=Path, help="Explicit public/data/prices.json path")
    parser.add_argument("--json-report", type=Path, help="Write the full machine-readable audit report to this path")
    parser.add_argument("--reference-date", type=_parse_iso_date, help="YYYY-MM-DD date for stale latest-date checks")
    parser.add_argument(
        "--max-latest-lag-days",
        type=_non_negative_int,
        default=DEFAULT_MAX_LATEST_LAG_DAYS,
        help=f"Maximum allowed latest-date lag before a stale symbol blocks the audit (default: {DEFAULT_MAX_LATEST_LAG_DAYS})",
    )
    parser.add_argument(
        "--critical-return-pct",
        type=_positive_float,
        default=market_db_sync.DEFAULT_CRITICAL_RETURN_PCT,
        help=f"Absolute one-day return percentage considered blocking (default: {market_db_sync.DEFAULT_CRITICAL_RETURN_PCT:g})",
    )
    parser.add_argument(
        "--split-like-return-pct",
        type=_positive_float,
        default=market_db_sync.DEFAULT_SPLIT_LIKE_RETURN_PCT,
        help=f"Absolute one-day return percentage considered warning-level (default: {market_db_sync.DEFAULT_SPLIT_LIKE_RETURN_PCT:g})",
    )
    args = parser.parse_args(argv)

    prices_path = _prices_path(args.app_dir, args.prices)
    report = audit_public_prices(
        prices_path,
        reference_date=args.reference_date,
        max_latest_lag_days=args.max_latest_lag_days,
        critical_return_pct=args.critical_return_pct,
        split_like_return_pct=args.split_like_return_pct,
    )
    if args.json_report:
        _write_json_report(args.json_report, report)

    for warning in report["warnings"]:
        print(f"WARN: {warning}", file=sys.stderr)
    if report["blocking"]:
        if report["first_blocking_error"]:
            print(f"ERROR: {report['first_blocking_error']}", file=sys.stderr)
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "price data quality check passed: "
        f"{report['symbols_checked']} symbols, {report['rows_checked']} rows, "
        f"status={report['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
