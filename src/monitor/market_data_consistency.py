"""Read-only broker-vs-local market data consistency checks."""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.market_db_sync import ADJUSTED_CLOSE_PROXY_SEMANTICS
from src.paths import MARKET_DB, sqlite_connect


DEFAULT_PROVIDER_ADJ_CLOSE_TOLERANCE_PCT = float(
    os.getenv("MARKET_DATA_RECON_ADJ_CLOSE_TOLERANCE_PCT", "0.5")
)
DEFAULT_PROVIDER_MAX_LATEST_LAG_DAYS = int(
    os.getenv("MARKET_DATA_RECON_MAX_LATEST_LAG_DAYS", "1")
)
DEFAULT_PROVIDER_MAX_OFFENDERS = int(os.getenv("MARKET_DATA_RECON_MAX_OFFENDERS", "5"))


class MarketDataSemanticsError(ValueError):
    """Raised when a consumer requests true OHLC bars from proxy-only rows."""


def _position_value(position: Any, key: str, default: Any = None) -> Any:
    if isinstance(position, dict):
        return position.get(key, default)
    return getattr(position, key, default)


def _parse_market_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(f"{value}T21:00:00+00:00")
    except ValueError:
        return None


def _parse_price_row_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _price_row_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _row_adjusted_close(row: Mapping[str, Any]) -> float | None:
    value = _price_row_value(row, ("adj_close", "adjusted_close", "adjClose", "close"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping_value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return default


def is_adjusted_close_proxy_price_row(row: Mapping[str, Any]) -> bool:
    """Return true when row metadata says OHLC fields are adjusted-close proxies."""
    proxy_flag = _mapping_value(row, "is_adjusted_close_proxy")
    if isinstance(proxy_flag, bool):
        return proxy_flag
    if isinstance(proxy_flag, int | float):
        return int(proxy_flag) == 1
    if isinstance(proxy_flag, str):
        normalized_flag = proxy_flag.strip().lower()
        if normalized_flag in {"1", "true", "yes", "y"}:
            return True
        if normalized_flag in {"0", "false", "no", "n", ""}:
            return False

    semantics = str(_mapping_value(row, "price_semantics", "") or "").strip().lower()
    return semantics == ADJUSTED_CLOSE_PROXY_SEMANTICS


def require_true_ohlc_price_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    consumer: str = "consumer",
) -> list[Mapping[str, Any]]:
    """Validate that rows are safe for consumers requiring true OHLC bars.

    Rows synced from compact public prices carry adjusted-close proxy metadata:
    open/high/low/close are all the adjusted close, and volume is zero. Latest
    close and freshness consumers may use those rows, but realized-volatility,
    intraday-range, execution-price, or volume-sensitive consumers should call
    this guard before interpreting OHLC fields.
    """
    checked_rows = list(rows)
    proxy_rows = [row for row in checked_rows if is_adjusted_close_proxy_price_row(row)]
    if proxy_rows:
        first = proxy_rows[0]
        symbol = _mapping_value(first, "symbol", "unknown")
        row_date = _mapping_value(first, "date", "unknown")
        raise MarketDataSemanticsError(
            f"{consumer} requires true OHLC bars but received adjusted-close proxy "
            f"rows; first proxy row: {symbol} {row_date}. Use adjusted-close or "
            "close-only data instead."
        )
    return checked_rows


def _latest_rows_by_symbol(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        row_date = _parse_price_row_date(row.get("date"))
        if not symbol or row_date is None:
            continue
        current = latest.get(symbol)
        if current is None or row_date > current["date"]:
            latest[symbol] = {
                "symbol": symbol,
                "date": row_date,
                "date_raw": row.get("date"),
                "adjusted_close": _row_adjusted_close(row),
            }
    return latest


def _append_missing_issue(
    issues: list[dict[str, Any]],
    *,
    provider: str,
    symbol: str,
) -> None:
    issues.append(
        {
            "issue": "missing_symbol",
            "provider": provider,
            "symbol": symbol,
            "message": f"{provider} missing {symbol}",
        }
    )


def _append_stale_issue(
    issues: list[dict[str, Any]],
    *,
    provider: str,
    symbol: str,
    row_date: date,
    expected_latest_date: date,
    lag_days: int,
) -> None:
    issues.append(
        {
            "issue": "stale_latest_date",
            "provider": provider,
            "symbol": symbol,
            "latest_date": row_date.isoformat(),
            "expected_latest_date": expected_latest_date.isoformat(),
            "lag_days": lag_days,
            "message": (
                f"{provider} {symbol} latest date {row_date.isoformat()} "
                f"lags {expected_latest_date.isoformat()} by {lag_days}d"
            ),
        }
    )


def _append_adjusted_close_issue(
    issues: list[dict[str, Any]],
    *,
    symbol: str,
    primary_provider: str,
    secondary_provider: str,
    primary_price: float,
    secondary_price: float,
    difference_pct: float,
    tolerance_pct: float,
) -> None:
    issues.append(
        {
            "issue": "adjusted_close_divergence",
            "symbol": symbol,
            "primary_provider": primary_provider,
            "secondary_provider": secondary_provider,
            "primary_adjusted_close": primary_price,
            "secondary_adjusted_close": secondary_price,
            "difference_pct": round(difference_pct, 4),
            "tolerance_pct": tolerance_pct,
            "message": (
                f"{symbol} adjusted close differs by {difference_pct:.2f}% "
                f"between {primary_provider} and {secondary_provider}"
            ),
        }
    )


def reconcile_price_providers(
    primary_rows: Iterable[Mapping[str, Any]],
    secondary_rows: Iterable[Mapping[str, Any]],
    *,
    primary_provider: str = "primary",
    secondary_provider: str = "secondary",
    required_symbols: Sequence[str] | None = None,
    adjusted_close_tolerance_pct: float = DEFAULT_PROVIDER_ADJ_CLOSE_TOLERANCE_PCT,
    max_latest_lag_days: int = DEFAULT_PROVIDER_MAX_LATEST_LAG_DAYS,
    max_offenders: int = DEFAULT_PROVIDER_MAX_OFFENDERS,
) -> dict[str, Any]:
    """Compare normalized adjusted-close rows from two historical providers."""
    primary_latest = _latest_rows_by_symbol(primary_rows)
    secondary_latest = _latest_rows_by_symbol(secondary_rows)

    if not primary_latest or not secondary_latest:
        outage_provider = primary_provider if not primary_latest else secondary_provider
        return {
            "status": "unavailable",
            "failure_type": "provider_outage",
            "primary_provider": primary_provider,
            "secondary_provider": secondary_provider,
            "outage_provider": outage_provider,
            "symbols_checked": 0,
            "issue_counts": {"provider_outage": 1},
            "top_offenders": [],
            "message": f"{outage_provider} returned no comparable price rows",
        }

    symbols = (
        [str(symbol).strip().upper() for symbol in required_symbols if str(symbol).strip()]
        if required_symbols is not None
        else sorted(set(primary_latest) | set(secondary_latest))
    )
    expected_latest_date = max(
        row["date"]
        for symbol in symbols
        for row in (primary_latest.get(symbol), secondary_latest.get(symbol))
        if row is not None
    )

    issues: list[dict[str, Any]] = []
    for symbol in symbols:
        primary = primary_latest.get(symbol)
        secondary = secondary_latest.get(symbol)

        if primary is None:
            _append_missing_issue(issues, provider=primary_provider, symbol=symbol)
        if secondary is None:
            _append_missing_issue(issues, provider=secondary_provider, symbol=symbol)

        for provider, row in ((primary_provider, primary), (secondary_provider, secondary)):
            if row is None:
                continue
            lag_days = (expected_latest_date - row["date"]).days
            if lag_days > max_latest_lag_days:
                _append_stale_issue(
                    issues,
                    provider=provider,
                    symbol=symbol,
                    row_date=row["date"],
                    expected_latest_date=expected_latest_date,
                    lag_days=lag_days,
                )

        if primary is None or secondary is None:
            continue
        primary_price = primary["adjusted_close"]
        secondary_price = secondary["adjusted_close"]
        if primary_price is None or secondary_price is None or secondary_price <= 0:
            continue
        difference_pct = (primary_price - secondary_price) / secondary_price * 100
        if abs(difference_pct) > adjusted_close_tolerance_pct:
            _append_adjusted_close_issue(
                issues,
                symbol=symbol,
                primary_provider=primary_provider,
                secondary_provider=secondary_provider,
                primary_price=primary_price,
                secondary_price=secondary_price,
                difference_pct=difference_pct,
                tolerance_pct=adjusted_close_tolerance_pct,
            )

    issue_counts = dict(Counter(str(issue["issue"]) for issue in issues))
    message = "providers reconciled"
    if issues:
        message = "; ".join(str(issue["message"]) for issue in issues[:max_offenders])

    return {
        "status": "warning" if issues else "ok",
        "failure_type": "provider_divergence" if issues else None,
        "primary_provider": primary_provider,
        "secondary_provider": secondary_provider,
        "outage_provider": None,
        "symbols_checked": len(symbols),
        "expected_latest_date": expected_latest_date.isoformat(),
        "adjusted_close_tolerance_pct": adjusted_close_tolerance_pct,
        "max_latest_lag_days": max_latest_lag_days,
        "issue_counts": issue_counts,
        "top_offenders": issues[:max_offenders],
        "message": message,
    }


def _latest_local_price(db_path: str | Path, symbol: str) -> dict[str, Any] | None:
    if not os.path.exists(db_path):
        return None
    with sqlite_connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT close, date FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    return {"price": float(row[0]), "date": row[1]}


def broker_market_data_consistency_report(
    positions: Iterable[Any] | None = None,
    *,
    db_path: str | Path = MARKET_DB,
    warn_threshold_pct: float = 2.0,
    max_local_age_days: int = 3,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compare broker position prices against local market data.

    This function never submits orders. When positions are omitted, it uses
    Alpaca credentials if available; otherwise it returns ``unavailable``.
    """
    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    resolved_now = resolved_now.astimezone(timezone.utc)

    if positions is None:
        try:
            from src.broker.alpaca import AlpacaClient

            client = AlpacaClient()
            if not client.is_ready():
                return {
                    "status": "unavailable",
                    "reason": "alpaca_not_configured",
                    "checked_at": resolved_now.isoformat(),
                    "rows": [],
                    "warnings": [],
                }
            positions = client.get_positions()
        except (ImportError, RuntimeError, OSError, ConnectionError, TimeoutError, ValueError, TypeError) as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "checked_at": resolved_now.isoformat(),
                "rows": [],
                "warnings": [],
            }

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        for position in positions:
            symbol = str(_position_value(position, "symbol", ""))
            broker_price = float(_position_value(position, "current_price", 0.0) or 0.0)
            local = _latest_local_price(db_path, symbol)
            if local is None:
                rows.append(
                    {
                        "symbol": symbol,
                        "broker_price": broker_price,
                        "broker_source": "alpaca_position",
                        "local_price": None,
                        "local_source": "market_db",
                        "local_date": None,
                        "difference_pct": None,
                        "status": "missing_local",
                    }
                )
                warnings.append(f"{symbol}: local market data missing")
                continue

            local_price = float(local["price"])
            local_date = str(local["date"])
            local_dt = _parse_market_date(local_date)
            local_age_days = (
                max((resolved_now - local_dt).total_seconds() / 86400, 0.0)
                if local_dt is not None
                else None
            )
            difference_pct = ((broker_price - local_price) / local_price * 100) if local_price > 0 else None
            row_status = "ok"
            if local_age_days is None or local_age_days > max_local_age_days:
                row_status = "stale_local"
                warnings.append(f"{symbol}: local market data stale")
            if difference_pct is not None and abs(difference_pct) >= warn_threshold_pct:
                row_status = "diverged"
                warnings.append(f"{symbol}: broker/local price differs by {difference_pct:.2f}%")

            rows.append(
                {
                    "symbol": symbol,
                    "broker_price": broker_price,
                    "broker_source": "alpaca_position",
                    "local_price": local_price,
                    "local_source": "market_db",
                    "local_date": local_date,
                    "local_age_days": round(local_age_days, 2) if local_age_days is not None else None,
                    "difference_pct": round(difference_pct, 4) if difference_pct is not None else None,
                    "status": row_status,
                }
            )
    except sqlite3.Error as exc:
        return {
            "status": "unavailable",
            "reason": f"sqlite_error: {exc}",
            "checked_at": resolved_now.isoformat(),
            "rows": rows,
            "warnings": warnings,
        }

    return {
        "status": "warning" if warnings else "ok",
        "reason": None,
        "checked_at": resolved_now.isoformat(),
        "warning_threshold_pct": warn_threshold_pct,
        "max_local_age_days": max_local_age_days,
        "rows": rows,
        "warnings": warnings,
    }
