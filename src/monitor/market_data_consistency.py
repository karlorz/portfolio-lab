"""Read-only broker-vs-local market data consistency checks."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.paths import MARKET_DB, sqlite_connect


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
