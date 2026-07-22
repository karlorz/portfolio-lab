"""Paper return / NAV multi-surface SSOT helpers (c358).

Write authority: ``daily_pnl.jsonl`` / ``daily_pnl_latest.json`` produced by
``scripts/capture_daily_pnl.py``. Other operator surfaces must match the session
series within epsilon or stamp ``return_source`` / why-not provenance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Session DoD / NAV agreement tolerance (fractional return, dollars for NAV).
DEFAULT_RETURN_EPS = 1e-5
DEFAULT_NAV_EPS = 0.02  # two cents


def material_return(val: Any, *, floor: float = 1e-6) -> bool:
    """True for real session returns; drops evaluator micro-noise (~1e-8).

    Exact 0.0 is a valid flat session.
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if v == 0.0:
        return True
    return abs(v) >= floor


def load_daily_pnl_sessions(data_dir: Path) -> List[Dict[str, Any]]:
    """Load deduped session rows from daily_pnl.jsonl ordered by date."""
    path = Path(data_dir) / "daily_pnl.jsonl"
    if not path.exists():
        return []
    by_date: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                d = str(row.get("date") or "")[:10]
                if not d:
                    continue
                by_date[d] = row
    except OSError as exc:
        logger.debug("load_daily_pnl_sessions failed: %s", exc)
        return []
    return [by_date[d] for d in sorted(by_date.keys())]


def load_session_ssot(
    data_dir: Path,
    *,
    session_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the write-SSOT session row (latest or named date).

    Prefer ``daily_pnl_latest.json`` when it matches ``session_date`` (or when
    date is omitted); fall back to jsonl.
    """
    root = Path(data_dir)
    latest_path = root / "daily_pnl_latest.json"
    latest: Optional[Dict[str, Any]] = None
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                latest = payload
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            latest = None

    if latest is not None:
        ld = str(latest.get("date") or "")[:10]
        if not session_date or ld == session_date:
            return {
                "date": ld,
                "daily_return": _as_float(latest.get("daily_return")),
                "total_value": _as_float(latest.get("total_value")),
                "return_source": "daily_pnl_latest",
                "raw": latest,
            }

    sessions = load_daily_pnl_sessions(root)
    if not sessions:
        return None
    if session_date:
        for row in reversed(sessions):
            if str(row.get("date") or "")[:10] == session_date:
                return {
                    "date": session_date,
                    "daily_return": _as_float(row.get("daily_return")),
                    "total_value": _as_float(row.get("total_value")),
                    "return_source": "daily_pnl.jsonl",
                    "raw": row,
                }
        return None
    row = sessions[-1]
    d = str(row.get("date") or "")[:10]
    return {
        "date": d,
        "daily_return": _as_float(row.get("daily_return")),
        "total_value": _as_float(row.get("total_value")),
        "return_source": "daily_pnl.jsonl",
        "raw": row,
    }


def values_agree(
    a: Optional[float],
    b: Optional[float],
    *,
    eps: float,
) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= eps


def align_portfolio_history_to_ssot(
    portfolio_path: Path,
    ssot: Dict[str, Any],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Stamp last history row (or append) to match session SSOT DoD/NAV.

    Ensures portfolio_paper history surface does not claim a conflicting
    daily_return for the same session date.
    """
    if not portfolio_path.exists():
        return {"updated": False, "reason": "missing_portfolio"}
    try:
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"updated": False, "reason": f"read_error:{exc}"}
    if not isinstance(portfolio, dict):
        return {"updated": False, "reason": "not_object"}

    history = portfolio.get("history")
    if not isinstance(history, list):
        history = []
        portfolio["history"] = history

    session_date = str(ssot.get("date") or "")[:10]
    daily_return = ssot.get("daily_return")
    total_value = ssot.get("total_value")
    if not session_date or daily_return is None or total_value is None:
        return {"updated": False, "reason": "incomplete_ssot"}

    updated = False
    matched = False
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        row_date = str(
            row.get("session_date") or row.get("date") or row.get("timestamp") or ""
        )[:10]
        if row_date and row_date != session_date:
            continue
        # Match last row without date, or same session date
        matched = True
        try:
            old_ret = float(row.get("daily_return")) if row.get("daily_return") is not None else None
        except (TypeError, ValueError):
            old_ret = None
        try:
            old_tv = float(row.get("total_value")) if row.get("total_value") is not None else None
        except (TypeError, ValueError):
            old_tv = None
        need = (
            old_ret is None
            or old_tv is None
            or not values_agree(old_ret, float(daily_return), eps=DEFAULT_RETURN_EPS)
            or not values_agree(old_tv, float(total_value), eps=DEFAULT_NAV_EPS)
        )
        if need:
            row["daily_return"] = round(float(daily_return), 6)
            row["total_value"] = round(float(total_value), 2)
            row["session_date"] = session_date
            row["return_source"] = "daily_pnl_ssot"
            updated = True
        break

    if not matched:
        history.append(
            {
                "session_date": session_date,
                "date": session_date,
                "total_value": round(float(total_value), 2),
                "daily_return": round(float(daily_return), 6),
                "return_source": "daily_pnl_ssot",
            }
        )
        updated = True

    if updated and not dry_run:
        portfolio_path.write_text(
            json.dumps(portfolio, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return {
        "updated": updated,
        "session_date": session_date,
        "dry_run": dry_run,
        "path": str(portfolio_path),
    }


def write_paper_trading_performance_from_ssot(
    data_dir: Path,
    *,
    session_date: Optional[str] = None,
    current_value: Optional[float] = None,
) -> Optional[Path]:
    """Regenerate paper-trading-performance snapshot from daily_pnl series.

    ``current_value`` defaults to the SSOT session total_value so the snapshot
    cannot lag the live book after a successful capture.
    """
    root = Path(data_dir)
    sessions = load_daily_pnl_sessions(root)
    if len(sessions) < 1:
        return None

    ssot = load_session_ssot(root, session_date=session_date)
    if ssot is None:
        return None

    date_key = str(ssot.get("date") or session_date or "")[:10]
    if not date_key:
        return None

    values: List[float] = []
    returns: List[float] = []
    for row in sessions:
        try:
            tv = float(row.get("total_value"))
        except (TypeError, ValueError):
            continue
        values.append(tv)
        dr = row.get("daily_return")
        if dr is not None and material_return(dr):
            try:
                returns.append(float(dr))
            except (TypeError, ValueError):
                pass

    if not values:
        return None

    cv = float(current_value) if current_value is not None else float(ssot["total_value"] or values[-1])
    start = values[0]
    total_return = (cv - start) / start if start > 0 else 0.0

    if returns and len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        sharpe = (mean_r / (var ** 0.5)) * (252 ** 0.5) if var > 0 else 0.0
    else:
        sharpe = 0.0

    max_dd = 0.0
    peak = values[0]
    series = list(values[:-1]) + [cv] if values else [cv]
    for v in series:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak else 0.0
        if dd > max_dd:
            max_dd = dd

    summary = {
        "date": date_key,
        "performance": {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "days_tracked": len(values),
            "start_value": start,
            "current_value": round(cv, 2),
            "current_value_source": "daily_pnl_ssot",
            "session_daily_return": ssot.get("daily_return"),
        },
        "daily_returns_distribution": {
            "positive_days": sum(1 for r in returns if r > 0),
            "negative_days": sum(1 for r in returns if r < 0),
            "win_rate": (sum(1 for r in returns if r > 0) / len(returns)) if returns else 0.0,
        },
        "return_source": "daily_pnl.jsonl_session",
        "schema_version": "paper-trading-performance/v3-ssot",
        "generated_at": datetime.now().isoformat(),
    }

    out_path = root / f"paper-trading-performance-{date_key}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return out_path


def read_surface_session(
    data_dir: Path,
    surface: str,
    *,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Read one operator surface's session DoD/NAV view for comparison."""
    root = Path(data_dir)
    if surface == "daily_pnl":
        ssot = load_session_ssot(root, session_date=session_date)
        if ssot is None:
            return {
                "surface": surface,
                "available": False,
                "why_not": "missing_daily_pnl",
            }
        return {
            "surface": surface,
            "available": True,
            "date": ssot.get("date"),
            "daily_return": ssot.get("daily_return"),
            "total_value": ssot.get("total_value"),
            "return_source": ssot.get("return_source"),
        }

    if surface == "portfolio_paper_history":
        path = root / "portfolio_paper.json"
        if not path.exists():
            return {
                "surface": surface,
                "available": False,
                "why_not": "missing_portfolio_paper",
            }
        try:
            paper = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {
                "surface": surface,
                "available": False,
                "why_not": "unreadable_portfolio_paper",
            }
        history = paper.get("history") or []
        positions = paper.get("positions") or {}
        cash = float(paper.get("cash") or 0.0)
        mark_tv = cash + sum(
            float(p.get("value") or 0.0)
            for p in positions.values()
            if isinstance(p, dict)
        )
        last: Optional[Dict[str, Any]] = None
        if isinstance(history, list):
            for row in reversed(history):
                if not isinstance(row, dict):
                    continue
                if session_date:
                    rd = str(
                        row.get("session_date")
                        or row.get("date")
                        or row.get("timestamp")
                        or ""
                    )[:10]
                    if rd and rd != session_date:
                        continue
                last = row
                break
        if last is None:
            return {
                "surface": surface,
                "available": True,
                "date": session_date,
                "daily_return": None,
                "total_value": round(mark_tv, 2),
                "return_source": "portfolio_paper.mark_only",
                "why_not": "no_history_row",
            }
        return {
            "surface": surface,
            "available": True,
            "date": str(
                last.get("session_date") or last.get("date") or session_date or ""
            )[:10]
            or None,
            "daily_return": _as_float(last.get("daily_return")),
            "total_value": _as_float(last.get("total_value")) or round(mark_tv, 2),
            "return_source": last.get("return_source") or "portfolio_paper.history",
        }

    if surface == "unified_dashboard_portfolio":
        # Mirror _get_portfolio_section SSOT preference without importing the
        # full dashboard module (keeps unit tests free of DB side effects).
        # Batch EB: when daily_return comes from daily_pnl_latest, also prefer
        # that snapshot's total_value as NAV (mark-to-market SSOT) so empty
        # positions / lagging marks do not invent nav_mismatch.
        paper = _read_json(root / "portfolio_paper.json")
        pnl = _read_json(root / "daily_pnl_latest.json")
        if not paper:
            return {
                "surface": surface,
                "available": False,
                "why_not": "missing_portfolio_paper",
            }
        positions = paper.get("positions") or {}
        mark_value = sum(
            float(p.get("value") or 0.0)
            for p in positions.values()
            if isinstance(p, dict)
        ) + float(paper.get("cash") or 0.0)
        total_value = mark_value
        daily_return = None
        return_source = None
        date = None
        if isinstance(pnl, dict) and (
            pnl.get("daily_return") is not None or pnl.get("total_value") is not None
        ):
            if pnl.get("daily_return") is not None:
                daily_return = _as_float(pnl.get("daily_return"))
            pnl_tv = _as_float(pnl.get("total_value"))
            if pnl_tv is not None and pnl_tv > 0:
                total_value = pnl_tv
            return_source = "daily_pnl_latest"
            date = str(pnl.get("date") or "")[:10] or None
        if daily_return is None:
            history = paper.get("history") or []
            if isinstance(history, list):
                for row in reversed(history):
                    if not isinstance(row, dict) or row.get("daily_return") is None:
                        continue
                    daily_return = _as_float(row.get("daily_return"))
                    return_source = "portfolio_paper.history"
                    date = str(
                        row.get("session_date")
                        or row.get("date")
                        or row.get("timestamp")
                        or ""
                    )[:10] or None
                    row_tv = _as_float(row.get("total_value"))
                    if row_tv is not None and row_tv > 0:
                        total_value = row_tv
                    break
        out: Dict[str, Any] = {
            "surface": surface,
            "available": True,
            "date": date,
            "daily_return": daily_return,
            "total_value": round(float(total_value), 2),
            "return_source": return_source or "none",
        }
        if daily_return is None:
            out["why_not"] = "no_daily_return_available"
        return out

    if surface == "stats_paper_portfolio":
        sessions = load_daily_pnl_sessions(root)
        returns = []
        values = []
        for row in sessions:
            if not material_return(row.get("daily_return")):
                continue
            try:
                returns.append(float(row["daily_return"]))
                values.append(float(row.get("total_value") or 0))
            except (TypeError, ValueError, KeyError):
                continue
        if len(returns) < 1:
            return {
                "surface": surface,
                "available": False,
                "why_not": "insufficient_daily_pnl_sessions",
                "return_source": "none",
            }
        last = sessions[-1] if sessions else {}
        return {
            "surface": surface,
            "available": True,
            "date": str(last.get("date") or "")[:10] or None,
            "daily_return": _as_float(last.get("daily_return")),
            "total_value": _as_float(last.get("total_value")),
            "days_tracked": len(returns),
            "return_source": "daily_pnl.jsonl_session",
        }

    if surface == "paper_trading_performance":
        date_key = session_date
        if not date_key:
            ssot = load_session_ssot(root)
            date_key = str((ssot or {}).get("date") or "")[:10]
        path = None
        if date_key:
            candidate = root / f"paper-trading-performance-{date_key}.json"
            if candidate.exists():
                path = candidate
        if path is None:
            files = sorted(root.glob("paper-trading-performance-*.json"))
            path = files[-1] if files else None
        if path is None:
            return {
                "surface": surface,
                "available": False,
                "why_not": "missing_snapshot",
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {
                "surface": surface,
                "available": False,
                "why_not": "unreadable_snapshot",
            }
        perf = data.get("performance") or {}
        return {
            "surface": surface,
            "available": True,
            "date": str(data.get("date") or "")[:10] or None,
            "daily_return": _as_float(
                perf.get("session_daily_return")
                if perf.get("session_daily_return") is not None
                else data.get("session_daily_return")
            ),
            "total_value": _as_float(perf.get("current_value")),
            "return_source": data.get("return_source")
            or perf.get("current_value_source")
            or "paper-trading-performance",
            "path": str(path),
        }

    return {
        "surface": surface,
        "available": False,
        "why_not": f"unknown_surface:{surface}",
    }


FIVE_SURFACES = (
    "daily_pnl",
    "portfolio_paper_history",
    "unified_dashboard_portfolio",
    "stats_paper_portfolio",
    "paper_trading_performance",
)


def compare_five_surfaces(
    data_dir: Path,
    *,
    session_date: Optional[str] = None,
    return_eps: float = DEFAULT_RETURN_EPS,
    nav_eps: float = DEFAULT_NAV_EPS,
) -> Dict[str, Any]:
    """Compare five operator surfaces against daily_pnl write SSOT."""
    root = Path(data_dir)
    ssot = load_session_ssot(root, session_date=session_date)
    surfaces = [
        read_surface_session(root, name, session_date=session_date or (ssot or {}).get("date"))
        for name in FIVE_SURFACES
    ]

    if ssot is None:
        return {
            "agree": False,
            "ssot": None,
            "surfaces": surfaces,
            "disagreements": [
                {
                    "surface": "daily_pnl",
                    "why_not": "missing_write_ssot",
                }
            ],
        }

    disagreements: List[Dict[str, Any]] = []
    for surf in surfaces:
        if surf.get("surface") == "daily_pnl":
            continue
        if not surf.get("available"):
            disagreements.append(
                {
                    "surface": surf.get("surface"),
                    "why_not": surf.get("why_not") or "unavailable",
                    "return_source": surf.get("return_source"),
                }
            )
            continue
        # Stats may omit single-day DoD disclosure if series is multi-day; still
        # require last session NAV/return when present.
        ret_ok = values_agree(
            surf.get("daily_return"),
            ssot.get("daily_return"),
            eps=return_eps,
        )
        nav_ok = values_agree(
            surf.get("total_value"),
            ssot.get("total_value"),
            eps=nav_eps,
        )
        # paper-trading-performance may only expose current_value (NAV) when
        # session_daily_return is stamped; require NAV always, DoD when present.
        if surf.get("surface") == "paper_trading_performance":
            if not nav_ok:
                disagreements.append(
                    {
                        "surface": surf.get("surface"),
                        "field": "total_value",
                        "ssot": ssot.get("total_value"),
                        "observed": surf.get("total_value"),
                        "return_source": surf.get("return_source"),
                        "why_not": "nav_mismatch",
                    }
                )
            elif surf.get("daily_return") is not None and not ret_ok:
                disagreements.append(
                    {
                        "surface": surf.get("surface"),
                        "field": "daily_return",
                        "ssot": ssot.get("daily_return"),
                        "observed": surf.get("daily_return"),
                        "return_source": surf.get("return_source"),
                        "why_not": "return_mismatch",
                    }
                )
            continue
        if not ret_ok or not nav_ok:
            disagreements.append(
                {
                    "surface": surf.get("surface"),
                    "ssot_return": ssot.get("daily_return"),
                    "observed_return": surf.get("daily_return"),
                    "ssot_nav": ssot.get("total_value"),
                    "observed_nav": surf.get("total_value"),
                    "return_source": surf.get("return_source"),
                    "why_not": (
                        "return_mismatch"
                        if not ret_ok
                        else "nav_mismatch"
                    ),
                }
            )

    return {
        "agree": len(disagreements) == 0,
        "ssot": {
            "date": ssot.get("date"),
            "daily_return": ssot.get("daily_return"),
            "total_value": ssot.get("total_value"),
            "return_source": ssot.get("return_source"),
        },
        "surfaces": surfaces,
        "disagreements": disagreements,
    }


def apply_capture_ssot_side_effects(
    data_dir: Path,
    snapshot: Dict[str, Any],
    *,
    mode: str = "paper",
) -> Dict[str, Any]:
    """After a successful capture: align history + regenerate performance snapshot."""
    root = Path(data_dir)
    session_date = str(snapshot.get("date") or "")[:10]
    ssot_view = {
        "date": session_date,
        "daily_return": snapshot.get("daily_return"),
        "total_value": snapshot.get("total_value"),
    }
    history_result = align_portfolio_history_to_ssot(
        root / f"portfolio_{mode}.json",
        ssot_view,
    )
    snap_path = write_paper_trading_performance_from_ssot(
        root,
        session_date=session_date,
        current_value=_as_float(snapshot.get("total_value")),
    )
    comparison = compare_five_surfaces(root, session_date=session_date)
    return {
        "history": history_result,
        "paper_trading_performance": str(snap_path) if snap_path else None,
        "comparison": comparison,
    }


def _as_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None
