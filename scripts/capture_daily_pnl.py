#!/usr/bin/env python3
"""Capture daily P&L snapshot from portfolio state.

Reads portfolio_paper.json (or portfolio_live.json), computes daily P&L
metrics, and appends a snapshot to data/daily_pnl.jsonl. Also writes
the latest snapshot to data/daily_pnl_latest.json for dashboard consumption.

Designed as a cron job — idempotent within a single day (won't duplicate
entries for the same date if run multiple times).

Usage:
    python scripts/capture_daily_pnl.py
    python scripts/capture_daily_pnl.py --mode live
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from src.paths import DATA_DIR

try:
    from src.strategy.evaluator import PAPER_CONFIG
    INITIAL_CAPITAL = PAPER_CONFIG.get("initial_capital", 100000)
except ImportError:
    INITIAL_CAPITAL = 100000

# US cash equity session calendar — never host-local midnight.
_ET = ZoneInfo("America/New_York")


def us_cash_session_date(now: Optional[datetime] = None) -> str:
    """America/New_York calendar date for daily_pnl row keys.

    Policy: date keys follow the US cash session calendar, not the host
    local timezone. A host past local midnight in Asia/Hong_Kong must not
    invent a 'tomorrow' US session row while ET is still the prior day.

    Before 16:00 ET this returns the current ET calendar date (intraday /
    pre-close capture). After 16:00 ET the date remains the same ET day
    until ET midnight rolls to the next session date. Explicit
    ``as_of_date`` overrides this default.
    """
    if now is None:
        now = datetime.now(tz=_ET)
    elif now.tzinfo is None:
        # Naive → interpret as UTC then convert (tests often freeze UTC)
        now = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(_ET)
    else:
        now = now.astimezone(_ET)
    return now.strftime("%Y-%m-%d")


def load_portfolio(mode: str = "paper") -> Optional[Dict[str, Any]]:
    """Load portfolio state from JSON file."""
    path = DATA_DIR / f"portfolio_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _prev_total_value_from_jsonl(
    append_path: Path,
    *,
    before_date: str,
) -> Optional[float]:
    """Latest prior-day total_value from daily_pnl.jsonl (excludes ``before_date``)."""
    if not append_path.exists():
        return None
    best_date = ""
    best_value: Optional[float] = None
    try:
        with open(append_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                d = str(entry.get("date") or "")[:10]
                if not d or d >= before_date:
                    continue
                try:
                    tv = float(entry.get("total_value"))
                except (TypeError, ValueError):
                    continue
                if d >= best_date:
                    best_date = d
                    best_value = tv
    except OSError:
        return None
    return best_value


def compute_daily_return(
    total_value: float,
    portfolio: Dict[str, Any],
    *,
    date: str,
    append_path: Optional[Path] = None,
) -> float:
    """Day-over-day NAV return for the capture job.

    Priority:
    1. Prior calendar day ``total_value`` from daily_pnl.jsonl (SSOT across runs)
    2. Prior distinct total_value from portfolio.history
    3. Last history row's ``daily_return`` field (legacy)
    """
    if total_value is None:
        return 0.0
    try:
        tv = float(total_value)
    except (TypeError, ValueError):
        return 0.0

    # 1) JSONL prior day
    if append_path is not None:
        prev = _prev_total_value_from_jsonl(append_path, before_date=date)
        if prev is not None and prev > 0:
            return (tv / prev) - 1.0

    # 2) History NAV chain (last entry with different total_value)
    history = portfolio.get("history") or []
    if isinstance(history, list):
        for h in reversed(history):
            if not isinstance(h, dict):
                continue
            try:
                hv = float(h.get("total_value"))
            except (TypeError, ValueError):
                continue
            if hv > 0 and abs(hv - tv) > 1e-9:
                return (tv / hv) - 1.0
        # 3) Legacy explicit daily_return on last history row
        if history and isinstance(history[-1], dict):
            try:
                return float(history[-1].get("daily_return", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def compute_pnl_snapshot(
    portfolio: Dict[str, Any],
    *,
    append_path: Optional[Path] = None,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute P&L snapshot from portfolio state."""
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash", 0)
    mode = portfolio.get("mode", "paper")

    total_value = cash + sum(p.get("value", 0) for p in positions.values())

    # Position-level P&L
    position_pnl = {}
    for sym, p in positions.items():
        shares = p.get("shares", 0)
        avg_price = p.get("avg_price", 0)
        current_price = p.get("current_price", 0)
        value = p.get("value", 0)
        unrealized = p.get("unrealized_pnl", 0)
        weight = value / total_value if total_value > 0 else 0
        position_pnl[sym] = {
            "shares": round(shares, 4),
            "avg_price": round(avg_price, 2),
            "current_price": round(current_price, 2),
            "value": round(value, 2),
            "weight": round(weight, 4),
            "unrealized_pnl": round(unrealized, 2),
        }

    date = as_of_date or us_cash_session_date()
    history = portfolio.get("history", [])
    daily_return = compute_daily_return(
        total_value,
        portfolio,
        date=date,
        append_path=append_path,
    )

    # Compute cumulative P&L
    initial_capital = INITIAL_CAPITAL
    total_pnl = total_value - initial_capital
    total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0

    # Drawdown calculation
    max_value = initial_capital
    for h in history:
        if not isinstance(h, dict):
            continue
        v = h.get("total_value", 0) or 0
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v > max_value:
            max_value = v
    drawdown = (total_value - max_value) / max_value if max_value > 0 else 0

    return {
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 6),
        "daily_return": round(float(daily_return), 6),
        "drawdown": round(drawdown, 6),
        "positions_count": len(positions),
        "positions": position_pnl,
    }


def save_snapshot(snapshot: Dict[str, Any], append_path: Path, latest_path: Path) -> bool:
    """Save P&L snapshot. Idempotent within same date — replaces if duplicate."""
    date = snapshot["date"]

    # Read existing entries, remove any from same date (idempotent)
    existing = []
    if append_path.exists():
        with open(append_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("date") != date:
                            existing.append(line)
                    except json.JSONDecodeError:
                        existing.append(line)

    # Append new snapshot
    existing.append(json.dumps(snapshot, default=str))

    with open(append_path, 'w') as f:
        f.write('\n'.join(existing) + '\n')

    # Write latest snapshot
    with open(latest_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return True


def backfill_daily_returns_from_nav(
    append_path: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Recompute historical daily_return from consecutive total_value marks.

    Fixes rows where stored daily_return was 0 while NAV moved (pre-NAV-DoD
    capture). Idempotent: only rewrites when |stored - true| > 1e-6.
    """
    if not append_path.exists():
        return {"rewritten": 0, "rows": 0, "dry_run": dry_run}

    rows: list[dict] = []
    with open(append_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                rows.append(entry)

    rewritten = 0
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        try:
            pv = float(prev.get("total_value"))
            tv = float(cur.get("total_value"))
        except (TypeError, ValueError):
            continue
        if pv <= 0:
            continue
        true_ret = (tv / pv) - 1.0
        try:
            stored = float(cur.get("daily_return") or 0.0)
        except (TypeError, ValueError):
            stored = 0.0
        if abs(true_ret - stored) > 1e-6:
            cur["daily_return"] = round(true_ret, 6)
            cur["daily_return_backfilled"] = True
            rewritten += 1

    if not dry_run and rewritten:
        with open(append_path, "w") as f:
            for entry in rows:
                f.write(json.dumps(entry, default=str) + "\n")

    return {"rewritten": rewritten, "rows": len(rows), "dry_run": dry_run}


def backfill_paper_history_returns_from_nav(
    portfolio_path: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Recompute portfolio_paper history daily_return from consecutive NAV marks.

    Batch CG / c344–c365: pre-session-baseline MTM wrote 0.0 while total_value
    moved. Mirrors ``backfill_daily_returns_from_nav`` for the paper history
    array. Idempotent; marks rewritten rows with ``daily_return_backfilled``.
    """
    if not portfolio_path.exists():
        return {"rewritten": 0, "rows": 0, "dry_run": dry_run, "path": str(portfolio_path)}

    try:
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("backfill_paper_history: failed to read %s: %s", portfolio_path, exc)
        return {
            "rewritten": 0,
            "rows": 0,
            "dry_run": dry_run,
            "error": str(exc),
            "path": str(portfolio_path),
        }

    if not isinstance(portfolio, dict):
        return {"rewritten": 0, "rows": 0, "dry_run": dry_run, "error": "not_object"}

    history = portfolio.get("history")
    if not isinstance(history, list) or len(history) < 2:
        return {
            "rewritten": 0,
            "rows": len(history) if isinstance(history, list) else 0,
            "dry_run": dry_run,
            "path": str(portfolio_path),
        }

    rewritten = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        cur = history[i]
        if not isinstance(prev, dict) or not isinstance(cur, dict):
            continue
        try:
            pv = float(prev.get("total_value"))
            tv = float(cur.get("total_value"))
        except (TypeError, ValueError):
            continue
        if pv <= 0:
            continue
        true_ret = (tv / pv) - 1.0
        try:
            stored = float(cur.get("daily_return") or 0.0)
        except (TypeError, ValueError):
            stored = 0.0
        if abs(true_ret - stored) > 1e-6:
            cur["daily_return"] = round(true_ret, 6)
            cur["daily_return_backfilled"] = True
            rewritten += 1

    if not dry_run and rewritten:
        portfolio["history"] = history
        portfolio_path.write_text(
            json.dumps(portfolio, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    return {
        "rewritten": rewritten,
        "rows": len(history),
        "dry_run": dry_run,
        "path": str(portfolio_path),
    }


def append_performance_jsonl(snapshot: Dict[str, Any], performance_path: Optional[Path] = None) -> bool:
    """Append deduped performance.jsonl row for bandit/stats consumers.

    Same calendar day replaces the last same-date entry (idempotent with daily_pnl).
    Does not replace evaluator intra-day rows from other timestamps on other days.
    """
    path = performance_path or (DATA_DIR / "performance.jsonl")
    date = snapshot.get("date") or us_cash_session_date()
    row = {
        "timestamp": snapshot.get("timestamp") or datetime.now().isoformat(),
        "date": date,
        "total_value": snapshot.get("total_value"),
        "cash": snapshot.get("cash"),
        "daily_return": snapshot.get("daily_return"),
        "positions_count": snapshot.get("positions_count"),
        "mode": snapshot.get("mode", "paper"),
        "source": "capture_daily_pnl",
    }

    lines: list[str] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                entry_date = entry.get("date")
                if not entry_date and isinstance(entry.get("timestamp"), str):
                    entry_date = entry["timestamp"][:10]
                if entry_date == date:
                    continue
                lines.append(json.dumps(entry, default=str))

    lines.append(json.dumps(row, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture daily P&L snapshot")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Portfolio mode to capture")
    parser.add_argument(
        "--backfill-returns",
        action="store_true",
        help="Recompute historical daily_return from consecutive total_value marks (jsonl)",
    )
    parser.add_argument(
        "--backfill-paper-history",
        action="store_true",
        help="Recompute portfolio_paper.json history daily_return from consecutive NAV marks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With backfill flags: report rewrite count without writing",
    )
    args = parser.parse_args()

    from src.utils.log_config import configure_logging
    configure_logging()

    append_path = DATA_DIR / "daily_pnl.jsonl"
    latest_path = DATA_DIR / "daily_pnl_latest.json"

    if args.backfill_returns or args.backfill_paper_history:
        if args.backfill_returns:
            summary = backfill_daily_returns_from_nav(append_path, dry_run=args.dry_run)
            logger.info("backfill_daily_returns: %s", summary)
        if args.backfill_paper_history:
            paper_path = DATA_DIR / f"portfolio_{args.mode}.json"
            summary = backfill_paper_history_returns_from_nav(
                paper_path, dry_run=args.dry_run
            )
            logger.info("backfill_paper_history_returns: %s", summary)
        return

    logger.info("Daily P&L Capture — %s", datetime.now().isoformat())

    portfolio = load_portfolio(args.mode)
    if not portfolio:
        logger.error("No portfolio_%s.json found", args.mode)
        sys.exit(1)

    snapshot = compute_pnl_snapshot(portfolio, append_path=append_path)
    # Stamp write-SSOT provenance on every capture row.
    snapshot["return_source"] = "capture_daily_pnl"
    snapshot["write_ssot"] = "daily_pnl.jsonl"

    save_snapshot(snapshot, append_path, latest_path)
    append_performance_jsonl(snapshot)

    # Align the other four surfaces: history rewrite + paper-trading-performance
    # regen so current_value cannot lag after a successful capture (c358).
    try:
        from src.monitor.paper_return_ssot import apply_capture_ssot_side_effects

        side = apply_capture_ssot_side_effects(
            DATA_DIR, snapshot, mode=args.mode
        )
        logger.info(
            "SSOT side-effects: history_updated=%s snapshot=%s agree=%s",
            (side.get("history") or {}).get("updated"),
            side.get("paper_trading_performance"),
            (side.get("comparison") or {}).get("agree"),
        )
    except Exception as exc:  # noqa: BLE001 — capture must not fail on side-effects
        logger.warning("SSOT side-effects failed (non-fatal): %s", exc)

    logger.info("Date: %s | Value: $%.2f | P&L: $%.2f (%.2f%%) | Daily: %.4f%% | DD: %.2f%% | Positions: %d",
                snapshot['date'], snapshot['total_value'], snapshot['total_pnl'],
                snapshot['total_pnl_pct'] * 100, snapshot['daily_return'] * 100,
                snapshot['drawdown'] * 100, snapshot['positions_count'])


if __name__ == "__main__":
    main()
