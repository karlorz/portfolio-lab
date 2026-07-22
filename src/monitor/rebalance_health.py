#!/usr/bin/env python3
"""
Portfolio-Lab v9.00: Rebalance Health Data Exporter

Generates rebalance_health.json for the RebalanceHealthPanel React component.
Reads order history files and SmartRebalanceData to produce execution timeline.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.backtest.metrics import save_results_json


__all__ = ['generate']

logger = logging.getLogger(__name__)

ORDERS_DIR = DATA_DIR / "historical_orders"
OUTPUT_PATH = DATA_DIR / "rebalance_health.json"


def _parse_order_file(path: Path) -> dict[str, Any] | None:
    """Parse an order history file (YAML frontmatter + JSON body)."""
    try:
        text = path.read_text()
        # Split frontmatter from body
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
            else:
                body = text
        else:
            body = text

        orders = json.loads(body)
        if not isinstance(orders, list) or len(orders) == 0:
            return None

        # Extract timestamp from filename: order_history_YYYYMMDD_HHMMSS_*
        stem = path.stem  # order_history_20260511_143008_bf55ba1899f9a2ef
        ts_parts = stem.split("_")
        if len(ts_parts) >= 3:
            date_str = ts_parts[2]
            time_str = ts_parts[3] if len(ts_parts) > 3 else "000000"
            try:
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError as e:
                logger.debug("Failed to parse timestamp from path %s: %s", path.name, e)
                ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        else:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        total_value = sum(o.get("estimated_value", 0) for o in orders)
        buy_count = sum(1 for o in orders if o.get("side") == "buy")
        sell_count = sum(1 for o in orders if o.get("side") == "sell")
        symbols = sorted(set(o.get("symbol", "?") for o in orders))

        return {
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M"),
            "source": "historical_order_file",
            "orders": len(orders),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_value": round(total_value, 2),
            "symbols": symbols,
            "reasons": sorted(set(o.get("reason", "rebalance") for o in orders)),
        }
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        logger.exception("Failed to parse order file: %s", path)
        return None


def _parse_daily_order_summary(path: Path) -> dict[str, Any] | None:
    """Parse root daily order summaries named order-history-YYYY-MM-DD.json.

    Batch DR: live SSOT uses ``recent_orders`` (not bare ``orders``). Prefer
    file ``date`` for schedule freshness — embedded order timestamps can be
    historical backfill stamps (e.g. 2026-05-11 rows rewritten into July files).
    """
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None
        # Live dual-write schema: recent_orders; legacy/tests: orders
        orders = payload.get("recent_orders")
        if not isinstance(orders, list) or len(orders) == 0:
            orders = payload.get("orders")
        if not isinstance(orders, list) or len(orders) == 0:
            return None

        # Prefer payload date (filename day) over embedded order timestamps —
        # daily summaries are the SSOT for "activity on this calendar day".
        if payload.get("date"):
            ts = datetime.fromisoformat(str(payload["date"])).replace(
                tzinfo=timezone.utc
            )
        else:
            timestamp = None
            for order in orders:
                if isinstance(order, dict) and order.get("timestamp"):
                    timestamp = str(order["timestamp"])
                    break
            if timestamp:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        dict_orders = [order for order in orders if isinstance(order, dict)]
        total_value = sum(
            order.get("estimated_value", order.get("value", order.get("fill_value", 0)))
            for order in dict_orders
        )
        buy_count = sum(1 for order in dict_orders if order.get("side") == "buy")
        sell_count = sum(1 for order in dict_orders if order.get("side") == "sell")
        symbols = sorted(set(order.get("symbol", "?") for order in dict_orders))

        return {
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M"),
            "source": "daily_order_summary",
            "orders": int(payload.get("total_orders") or len(dict_orders)),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_value": round(total_value, 2),
            "symbols": symbols,
            "reasons": sorted(set(order.get("reason", "rebalance") for order in dict_orders)),
        }
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, OSError):
        logger.exception("Failed to parse daily order summary: %s", path)
        return None


def _daily_order_summary_dir() -> Path:
    """Return the root directory that should hold daily order summaries."""
    if ORDERS_DIR.name == "historical_orders":
        return ORDERS_DIR.parent
    return ORDERS_DIR


def _canonical_source_label(history: list[dict[str, Any]]) -> str:
    sources = {entry.get("source") for entry in history}
    if {"historical_order_file", "daily_order_summary"}.issubset(sources):
        return "combined_order_history"
    if "daily_order_summary" in sources:
        return "daily_order_summaries"
    if "historical_order_file" in sources:
        return "historical_order_files"
    return "none"


def _dedupe_canonical_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by execution date, preferring daily summaries for that date."""
    source_rank = {"historical_order_file": 0, "daily_order_summary": 1}
    by_date: dict[str, dict[str, Any]] = {}
    for entry in sorted(history, key=lambda e: e["timestamp"]):
        date_key = entry.get("date", entry.get("timestamp", "")[:10])
        current = by_date.get(date_key)
        if current is None:
            by_date[date_key] = entry
            continue
        entry_key = (
            source_rank.get(entry.get("source"), -1),
            entry.get("timestamp", ""),
        )
        current_key = (
            source_rank.get(current.get("source"), -1),
            current.get("timestamp", ""),
        )
        if entry_key >= current_key:
            by_date[date_key] = entry
    return sorted(by_date.values(), key=lambda e: e["timestamp"])


def generate() -> dict[str, Any]:
    """Generate rebalance health data."""
    # Parse order history
    history = []
    if ORDERS_DIR.exists():
        for f in sorted(ORDERS_DIR.glob("order_history_*.json")):
            entry = _parse_order_file(f)
            if entry:
                history.append(entry)
    daily_dir = _daily_order_summary_dir()
    if daily_dir.exists():
        for f in sorted(daily_dir.glob("order-history-*.json")):
            entry = _parse_daily_order_summary(f)
            if entry:
                history.append(entry)

    # Sort by timestamp
    history.sort(key=lambda e: e["timestamp"])
    canonical_history = _dedupe_canonical_history(history)

    # Determine next scheduled rebalance
    # Default: monthly on the 1st, or 30 days after last rebalance
    last_ts = None
    now = datetime.now(timezone.utc)
    if canonical_history:
        last_ts = datetime.fromisoformat(canonical_history[-1]["timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
    else:
        last_ts = now - timedelta(days=30)

    next_rebalance = last_ts + timedelta(days=30)
    days_until = (next_rebalance - now).days
    # Negative days_until means the projected monthly slot is already past —
    # disclose overdue rather than looking like a future schedule.
    if days_until < 0:
        next_status = "overdue"
        next_status_reason = (
            f"projected next rebalance {next_rebalance.strftime('%Y-%m-%d')} "
            f"is {abs(days_until)} day(s) past (last execution + 30d)"
        )
    elif days_until == 0:
        next_status = "due_today"
        next_status_reason = "projected next rebalance is today"
    else:
        next_status = "scheduled"
        next_status_reason = None

    # Schedule compliance
    on_time = 0
    delayed = 0
    for i, entry in enumerate(canonical_history):
        if i == 0:
            continue
        prev_ts = datetime.fromisoformat(canonical_history[i - 1]["timestamp"])
        curr_ts = datetime.fromisoformat(entry["timestamp"])
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        if curr_ts.tzinfo is None:
            curr_ts = curr_ts.replace(tzinfo=timezone.utc)
        delta_days = (curr_ts - prev_ts).days
        if 25 <= delta_days <= 35:
            on_time += 1
        else:
            delayed += 1

    # Recent executions (last 10)
    recent = history[-10:] if len(history) > 10 else history
    execution_times = [
        {"date": e["date"], "time": e["time"], "orders": e["orders"],
         "total_value": e["total_value"], "symbols": e["symbols"],
         "source": e.get("source", "unknown")}
        for e in reversed(recent)  # Most recent first
    ]

    return {
        "generated": now.isoformat(),
        "next_rebalance": {
            "date": next_rebalance.strftime("%Y-%m-%d"),
            "days_until": days_until,
            "frequency": "monthly (~30 days)",
            "status": next_status,
            "status_reason": next_status_reason,
            "overdue": next_status == "overdue",
        },
        "schedule_compliance": {
            "on_time": on_time,
            "delayed": delayed,
            "total": on_time + delayed,
            "compliance_pct": round(
                on_time / max(on_time + delayed, 1) * 100, 1
            ),
        },
        "execution_history": execution_times,
        "total_executions": len(history),
        "canonical_order_history_source": _canonical_source_label(history),
        "market_data_consistency": _generate_market_data_consistency(),
        "alpaca_feed_entitlement": _generate_alpaca_feed_entitlement(),
    }


def _generate_market_data_consistency() -> dict[str, Any]:
    """Generate read-only broker/local data consistency diagnostics."""
    try:
        from src.monitor.market_data_consistency import broker_market_data_consistency_report

        return broker_market_data_consistency_report()
    except (ImportError, RuntimeError, OSError, ValueError, TypeError) as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "rows": [],
            "warnings": [],
        }


def _generate_alpaca_feed_entitlement() -> dict[str, Any]:
    """Generate public-safe Alpaca feed entitlement diagnostics."""
    try:
        from src.broker.alpaca import resolve_alpaca_feed_entitlement

        return resolve_alpaca_feed_entitlement()
    except (ImportError, RuntimeError, OSError, ValueError, TypeError) as exc:
        return {
            "configured_feed": "unknown",
            "effective_feed": "unknown",
            "entitlement": "unknown",
            "delayed": True,
            "acceptable_for_live": False,
            "policy_decision": "reject",
            "reason": str(exc),
        }


def main():
    data = generate()
    try:
        from src.dashboard.generator import _stamp_generator_git_sha

        data = _stamp_generator_git_sha(data)
    except Exception:  # noqa: BLE001 — never block rebalance export
        pass

    private_path = OUTPUT_PATH
    public_path = Path(PUBLIC_DATA_DIR) / "rebalance_health.json"
    paths_identical = False
    try:
        paths_identical = private_path.resolve() == public_path.resolve()
    except OSError:
        paths_identical = False

    try:
        from src.dashboard.generator import _attach_dual_write_provenance

        data = _attach_dual_write_provenance(
            data,
            private_path=private_path,
            public_path=public_path,
            dual_write_attempted=not paths_identical,
            dual_write_ok=None if not paths_identical else True,
            paths_identical=paths_identical,
        )
    except Exception:  # noqa: BLE001
        pass

    private_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_json(data, output_path=str(private_path))

    if not paths_identical:
        try:
            public_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.dashboard.generator import _attach_dual_write_provenance

                data = _attach_dual_write_provenance(
                    data,
                    private_path=private_path,
                    public_path=public_path,
                    dual_write_attempted=True,
                    dual_write_ok=True,
                    paths_identical=False,
                )
                save_results_json(data, output_path=str(private_path))
            except Exception:  # noqa: BLE001
                pass
            save_results_json(data, output_path=str(public_path))
            # Batch CJ: honest lag/hash after both trees exist
            try:
                from src.dashboard.generator import finalize_dual_write_provenance_after_sync

                data = finalize_dual_write_provenance_after_sync(
                    data,
                    private_path=private_path,
                    public_path=public_path,
                    dual_write_ok=True,
                    note="post_sync rebalance_health dual-write (Batch CJ)",
                )
            except Exception:  # noqa: BLE001
                pass
        except OSError as exc:
            logger.warning("Public rebalance_health dual-write failed: %s", exc)
            try:
                from src.dashboard.generator import _attach_dual_write_provenance

                data = _attach_dual_write_provenance(
                    data,
                    private_path=private_path,
                    public_path=public_path,
                    dual_write_attempted=True,
                    dual_write_ok=False,
                    paths_identical=False,
                    note=str(exc),
                )
                save_results_json(data, output_path=str(private_path))
            except Exception:  # noqa: BLE001
                pass

    logger.info("Rebalance health data exported to %s", OUTPUT_PATH)
    logger.info("  Executions: %d", data['total_executions'])
    logger.info("  Next rebalance: %s (%d days)",
                data['next_rebalance']['date'],
                data['next_rebalance']['days_until'])
    logger.info("  Compliance: %s%%", data['schedule_compliance']['compliance_pct'])


if __name__ == "__main__":
    main()
