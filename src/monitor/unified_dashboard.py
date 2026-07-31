#!/usr/bin/env python3
"""
v6.08: Unified System Dashboard

Reads ALL state files across the portfolio-lab system and generates a single
comprehensive report aggregating: system health, portfolio state, risk metrics,
TCA execution quality, overlay states, regime/optimizer state, performance
attribution, and cron job status.

Usage:
    python -m src.monitor.unified_dashboard          # Console summary
    python -m src.monitor.unified_dashboard --save   # Save JSON + console
    python -m src.monitor.unified_dashboard --json    # JSON output only
    python -m src.monitor.unified_dashboard --check   # Return exit code 0/1
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.paths import DATA_DIR, PUBLIC_DATA_DIR
from src.utils import safe_get
from src.backtest.metrics import save_results_json


__all__ = ['generate_unified_dashboard', 'print_summary', 'generate_status_text']

logger = logging.getLogger(__name__)



# ─────────────────────────────────────────────
#  Section Readers
# ─────────────────────────────────────────────



def _normalize_overlay_allocation_pct(value: Any) -> Optional[float]:
    """Return allocation in percent points for display/storage.

    Accepts fraction (0.03 → 3.0) or percent points (3.0 → 3.0). Values with
    abs >= 1 are treated as already percent (VIXY hedge state historical shape).
    """
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if abs(x) < 1.0:
        return x * 100.0
    return x


def _read_json(path: str) -> Optional[Any]:
    """Safely read a JSON file, returning None on failure."""
    full_path = DATA_DIR / path
    if not full_path.exists():
        return None
    try:
        with open(full_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _get_health_section() -> Dict[str, Any]:
    """System health for the unified panel.

    Preference order (Batch CD):
    1. Dashboard ``PUBLIC_DATA_DIR/health.json`` when it carries ``system_status``
       and/or ``signal_health`` — so SH 0/N and degraded system_status surface
       (entropy-only GARCH report was hiding the fleet).
    2. Legacy ``.health_report.json`` structured checks.
    3. GARCH flat metrics in ``.health_report.json`` / risk_metrics.

    Components always include signal_health rollup when available.
    """
    components: Dict[str, Any] = {}
    alerts: list[Any] = []
    status = "unknown"
    timestamp = None
    checks_passed = 0
    checks_total = 0
    sources_used: list[str] = []

    # ── 1) Dashboard health SSOT (public tree) ─────────────────────────
    pub_health: Optional[Dict[str, Any]] = None
    pub_path = Path(PUBLIC_DATA_DIR) / "health.json"
    if pub_path.is_file():
        try:
            blob = json.loads(pub_path.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                pub_health = blob
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to read public health.json: %s", exc)

    if isinstance(pub_health, dict) and (
        pub_health.get("system_status") is not None
        or isinstance(pub_health.get("signal_health"), dict)
    ):
        sources_used.append("public_health")
        status = str(
            pub_health.get("system_status")
            or pub_health.get("status")
            or "unknown"
        ).lower()
        timestamp = (
            pub_health.get("generated_at")
            or pub_health.get("content_patched_at")
            or pub_health.get("timestamp")
        )
        sh = pub_health.get("signal_health")
        if isinstance(sh, dict):
            summary = sh.get("summary") if isinstance(sh.get("summary"), dict) else {}
            healthy = int(summary.get("healthy") or 0)
            total = int(summary.get("total_tracked") or summary.get("total") or 0)
            degraded = int(summary.get("degraded") or 0)
            unhealthy = int(summary.get("unhealthy") or 0)
            sh_status = str(sh.get("status") or sh.get("overall_health") or "unknown")
            components["signal_health"] = {
                "status": sh_status,
                "ok": healthy > 0 and unhealthy == 0,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "total_tracked": total,
            }
            checks_passed = healthy
            checks_total = total
            if total > 0 and healthy == 0:
                alerts.append(
                    f"signal_health 0/{total} healthy "
                    f"(degraded={degraded}, unhealthy={unhealthy})"
                )
        # Compact ops fields if present
        if pub_health.get("ops_health_status"):
            components["ops_health"] = {
                "status": pub_health.get("ops_health_status"),
                "ok": str(pub_health.get("ops_health_status")).lower()
                in {"ok", "healthy", "green"},
            }

    # ── 2/3) .health_report.json (legacy checks or GARCH flat) ────────
    report = _read_json(".health_report.json")
    if report and ("checks" in report or "status" in report) and "var_95" not in report:
        sources_used.append("health_report_legacy")
        if status == "unknown":
            status = str(report.get("status", "unknown"))
            timestamp = timestamp or report.get("timestamp")
            checks_passed = safe_get(report, "summary", "passed", default=0) or 0
            checks_total = safe_get(report, "summary", "total_checks", default=0) or 0
            alerts = list(report.get("alerts") or alerts)
        checks = report.get("checks") or {}
        for name, c in checks.items():
            if not isinstance(c, dict):
                continue
            components[name] = {
                "status": c.get("status"),
                "ok": c.get("ok", False),
            }
    elif report and (
        "var_95" in report
        or "tail_severity" in report
        or "cvar_ratio" in report
    ):
        sources_used.append("health_report_garch")
        tail = report.get("tail_severity", "normal")
        cvar_ratio = report.get("cvar_ratio", 1.0)
        if tail in ("extreme", "severe") or (
            isinstance(cvar_ratio, (int, float)) and cvar_ratio > 3.0
        ):
            garch_status = "unhealthy"
        elif tail in ("elevated", "moderate") or (
            isinstance(cvar_ratio, (int, float)) and cvar_ratio > 2.0
        ):
            garch_status = "degraded"
        else:
            garch_status = "healthy"
        components["garch_cvar"] = {
            "status": garch_status,
            "ok": garch_status == "healthy",
            "var_95": report.get("var_95") or report.get("var_95_daily"),
            "cvar_95": report.get("cvar_95") or report.get("cvar_95_daily"),
            "tail_severity": tail,
            "garch_active": report.get("garch_active"),
            "coverage_pass": safe_get(
                report, "coverage_diagnostics", "coverage_pass", default=None
            ),
        }
        if status == "unknown":
            status = garch_status
            timestamp = timestamp or report.get("timestamp")
            checks_passed = 1 if garch_status == "healthy" else 0
            checks_total = 1
            if garch_status != "healthy":
                try:
                    cr = float(cvar_ratio)
                    alerts.append(f"Tail severity: {tail}, CVaR ratio: {cr:.2f}")
                except (TypeError, ValueError):
                    alerts.append(f"Tail severity: {tail}, CVaR ratio: {cvar_ratio}")
        elif garch_status != "healthy":
            alerts.append(f"GARCH: {garch_status} (tail={tail})")

    # Also fold risk_metrics coverage if garch component missing
    if "garch_cvar" not in components:
        metrics = _read_json("risk_metrics.json")
        if isinstance(metrics, dict) and (
            metrics.get("var_95_daily") is not None or metrics.get("tail_severity")
        ):
            sources_used.append("risk_metrics")
            tail = metrics.get("tail_severity", "normal")
            cov = metrics.get("coverage_diagnostics") or {}
            garch_active = metrics.get("garch_active")
            garch_status = "healthy"
            if metrics.get("status") in {"degraded", "unhealthy"}:
                garch_status = str(metrics.get("status"))
            elif garch_active is False or cov.get("coverage_pass") is False:
                garch_status = "degraded"
            components["garch_cvar"] = {
                "status": garch_status,
                "ok": garch_status == "healthy",
                "var_95": metrics.get("var_95_daily"),
                "cvar_95": metrics.get("cvar_95_daily"),
                "tail_severity": tail,
                "garch_active": garch_active,
                "coverage_pass": cov.get("coverage_pass"),
            }
            if status == "unknown":
                status = garch_status
                timestamp = timestamp or metrics.get("timestamp")
                checks_passed = 1 if garch_status == "healthy" else 0
                checks_total = 1

    if not components and status == "unknown":
        return {"available": False, "status": "unknown"}

    # Max-severity: SH 0/N or public degraded wins over garch-only healthy
    if any(
        isinstance(c, dict) and c.get("status") in {"degraded", "unhealthy", "critical"}
        for c in components.values()
    ):
        if status in {"healthy", "ok", "unknown"}:
            status = "degraded"

    return {
        "available": True,
        "timestamp": timestamp,
        "status": status,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "alerts": alerts,
        "components": components,
        "sources": sources_used,
    }


def _get_portfolio_section() -> Dict[str, Any]:
    """Current portfolio state from portfolio_paper.json.

    Batch CF: surface day-over-day return from ``daily_pnl_latest.json`` (capture
    SSOT) with fallback to the last paper history row — so the unified panel no
    longer omits DoD while MTM/history and capture disagree.
    """
    paper = _read_json("portfolio_paper.json")
    if not paper:
        return {"available": False}

    positions = paper.get("positions", {})
    total_value = sum(
        p.get("value", 0) for p in positions.values()
    ) + paper.get("cash", 0)

    pos_list = []
    for sym, p in positions.items():
        pos_list.append(
            {
                "symbol": sym,
                "shares": p.get("shares", 0),
                "value": p.get("value", 0),
                "weight": round(p.get("value", 0) / total_value * 100, 2) if total_value > 0 else 0,
                "unrealized_pnl": p.get("unrealized_pnl", 0),
                "avg_price": p.get("avg_price"),
                "current_price": p.get("current_price"),
            }
        )

    daily_return: Optional[float] = None
    daily_return_source: Optional[str] = None
    daily_return_date: Optional[str] = None
    pnl = _read_json("daily_pnl_latest.json")
    if isinstance(pnl, dict) and pnl.get("daily_return") is not None:
        try:
            daily_return = float(pnl["daily_return"])
            daily_return_source = "daily_pnl_latest"
            daily_return_date = str(pnl.get("date") or "")[:10] or None
        except (TypeError, ValueError):
            daily_return = None
    if daily_return is None:
        history = paper.get("history") or []
        if isinstance(history, list):
            for row in reversed(history):
                if not isinstance(row, dict) or row.get("daily_return") is None:
                    continue
                try:
                    daily_return = float(row["daily_return"])
                    daily_return_source = "portfolio_paper.history"
                    daily_return_date = str(
                        row.get("session_date") or row.get("date") or row.get("timestamp") or ""
                    )[:10] or None
                    break
                except (TypeError, ValueError):
                    continue

    out: Dict[str, Any] = {
        "available": True,
        "total_value": round(total_value, 2),
        "cash": round(paper.get("cash", 0), 2),
        "cash_pct": round(paper.get("cash", 0) / total_value * 100, 2) if total_value > 0 else 0,
        "positions": sorted(pos_list, key=lambda x: x["value"], reverse=True),
        "update_timestamp": paper.get("updated"),
        "mode": paper.get("mode", "paper"),
        "history_count": len(paper.get("history", [])),
    }
    if daily_return is not None:
        out["daily_return"] = round(daily_return, 6)
        out["daily_return_source"] = daily_return_source
        if daily_return_date:
            out["daily_return_date"] = daily_return_date
    return out


def _normalize_drawdown_to_percent(
    value: Any,
    *,
    source_hint: str = "",
) -> tuple[Optional[float], str]:
    """Canonical drawdown unit is **percent** (e.g. −5.85 for a 5.85% DD).

    Producers disagree:
    - evaluator / paper history: fraction (−0.0585)
    - compute_garch_risk / risk_metrics: percent-like (−10.52)

    Heuristic: |v| <= 1 (and v != 0) → fraction → *100; |v| > 1 → already percent.
    Explicit ``*_pct`` twins and ``drawdown_unit`` source hints override when set.
    Returns (normalized_or_None, unit_label) where unit_label is always "percent"
    for finite values.
    """
    if value is None:
        return None, "percent"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, "percent"
    if v == 0.0:
        return 0.0, "percent"
    # Policy-style limits often stored as −15.0 already percent; fractions are
    # peak-to-trough ratios with abs < 1 (e.g. −0.0585).
    if abs(v) <= 1.0:
        # Treat as fraction unless source already labeled percent (rare edge)
        if source_hint in {"percent", "pct", "%"}:
            return round(v, 6), "percent"
        return round(v * 100.0, 6), "percent"
    return round(v, 6), "percent"


def _normalize_risk_payload(metrics: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    """Map risk_metrics.json or GARCH .health_report.json into unified risk section.

    Drawdown honesty: prefer ``measured_max_drawdown`` over policy ``max_drawdown``
    / ``max_drawdown_limit`` so operators never read −15 policy as live NAV DD.
    All published drawdown fields are **percent** with ``drawdown_unit: percent``.
    GARCH active: honor explicit demote (``garch_active=false`` + runtime_role)
    even when ``filter_active`` is still true on the private health report.
    """
    unit_hint = str(metrics.get("drawdown_unit") or metrics.get("dd_unit") or "").lower()
    # health_report uses var_95/cvar_95; risk_metrics uses var_95_daily/cvar_95_daily
    var_daily = metrics.get("var_95_daily", metrics.get("var_95"))
    cvar_daily = metrics.get("cvar_95_daily", metrics.get("cvar_95"))
    garch_active = metrics.get("garch_active")
    if garch_active is None:
        garch_active = bool(metrics.get("filter_active", False))
    # Explicit demote always wins over filter_active residual true
    if metrics.get("runtime_role") == "advisory_degraded":
        garch_active = False
    garch_filtered = metrics.get("garch_filtered")
    if garch_filtered is None:
        garch_filtered = bool(metrics.get("filter_active", garch_active))
    # Prefer measured NAV peak-to-trough; fall back to max_drawdown only when
    # it is not the renamed policy limit slot.
    measured_dd_raw = metrics.get("measured_max_drawdown")
    # Prefer explicit percent twin if present (already percent)
    if metrics.get("measured_max_drawdown_pct") is not None:
        measured_dd, _ = _normalize_drawdown_to_percent(
            metrics.get("measured_max_drawdown_pct"),
            source_hint="percent",
        )
    else:
        measured_dd, _ = _normalize_drawdown_to_percent(
            measured_dd_raw, source_hint=unit_hint
        )
    policy_limit_raw = metrics.get("max_drawdown_limit")
    if metrics.get("max_drawdown_limit_pct") is not None:
        policy_limit, _ = _normalize_drawdown_to_percent(
            metrics.get("max_drawdown_limit_pct"), source_hint="percent"
        )
    else:
        # Same fraction/percent heuristic as measured DD (no forced "percent"
        # default — policy is often stored as −0.15 fraction).
        policy_limit, _ = _normalize_drawdown_to_percent(
            policy_limit_raw, source_hint=unit_hint
        )
    raw_max_dd = metrics.get("max_drawdown")
    if measured_dd is not None:
        max_drawdown = measured_dd
    elif raw_max_dd is not None:
        max_drawdown, _ = _normalize_drawdown_to_percent(
            raw_max_dd, source_hint=unit_hint
        )
    else:
        max_drawdown = None
    measured_cur_raw = metrics.get("measured_current_drawdown")
    current_raw = (
        measured_cur_raw
        if measured_cur_raw is not None
        else metrics.get("current_drawdown")
    )
    current_dd, _ = _normalize_drawdown_to_percent(current_raw, source_hint=unit_hint)
    out: Dict[str, Any] = {
        "available": True,
        "timestamp": metrics.get("timestamp"),
        "var_95_daily": var_daily,
        "cvar_95_daily": cvar_daily,
        "cvar_ratio": metrics.get("cvar_ratio"),
        "tail_severity": metrics.get("tail_severity"),
        "max_drawdown": max_drawdown,
        "current_drawdown": current_dd,
        "drawdown_unit": "percent",
        "volatility_annual": metrics.get("volatility_annual"),
        "garch_active": bool(garch_active),
        "garch_filtered": bool(garch_filtered),
        "source": source,
    }
    if measured_dd is not None:
        out["measured_max_drawdown"] = measured_dd
        out["measured_max_drawdown_pct"] = measured_dd
    if current_dd is not None:
        out["measured_current_drawdown"] = current_dd
    if policy_limit is not None:
        out["max_drawdown_limit"] = policy_limit
        out["max_drawdown_limit_pct"] = (
            abs(policy_limit) if policy_limit is not None else None
        )
    if metrics.get("drawdown_field_semantics"):
        out["drawdown_field_semantics"] = metrics.get("drawdown_field_semantics")
    if metrics.get("garch_active_reason"):
        out["garch_active_reason"] = metrics.get("garch_active_reason")
    if metrics.get("runtime_role"):
        out["runtime_role"] = metrics.get("runtime_role")
    return out


def _payload_has_risk_metrics(
    metrics: Optional[Dict[str, Any]],
    *,
    require_var_fields: bool = False,
) -> bool:
    """True when payload can drive a risk section.

    Legacy risk_metrics.json may only carry flags/timestamp; GARCH health
    reports should have VaR/CVaR fields when preferred over orphans.
    """
    if not isinstance(metrics, dict) or not metrics:
        return False
    if not require_var_fields:
        return True
    return any(
        metrics.get(k) is not None
        for k in ("var_95_daily", "cvar_95_daily", "var_95", "cvar_95", "cvar_ratio")
    )


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _get_risk_section() -> Dict[str, Any]:
    """Risk metrics: prefer fresher dual-write SSOT; merge demote honesty.

    ``risk_metrics.json`` carries coverage demote + measured DD after Batch AC.
    Private ``.health_report.json`` may still echo ``filter_active=true`` without
    demote fields. When timestamps are equal/near, prefer the payload that has
    explicit ``garch_active`` / measured drawdown honesty.
    """
    metrics = _read_json("risk_metrics.json")
    health = _read_json(".health_report.json")

    candidates: list[tuple[str, Dict[str, Any], Optional[datetime]]] = []
    if _payload_has_risk_metrics(health, require_var_fields=True):
        candidates.append(("health_report", health, _parse_ts(health.get("timestamp"))))
    if _payload_has_risk_metrics(metrics, require_var_fields=False):
        candidates.append(("risk_metrics", metrics, _parse_ts(metrics.get("timestamp"))))

    if not candidates:
        return {"available": False}

    def honesty_rank(payload: Dict[str, Any]) -> int:
        score = 0
        if payload.get("measured_max_drawdown") is not None:
            score += 2
        if payload.get("garch_active") is not None:
            score += 2
        if payload.get("runtime_role") or payload.get("garch_active_reason"):
            score += 1
        if payload.get("max_drawdown_limit") is not None:
            score += 1
        return score

    # Prefer newer timestamp; on tie prefer higher honesty rank then risk_metrics
    # (dual-write path carries demote after coverage check).
    best = max(
        candidates,
        key=lambda it: (
            it[2] or datetime.min,
            honesty_rank(it[1]),
            1 if it[0] == "risk_metrics" else 0,
        ),
    )
    source, payload, _ts = best
    # Merge demote/measured fields from the sibling payload when missing
    sibling = metrics if source == "health_report" else health
    if isinstance(sibling, dict) and sibling:
        merged = dict(payload)
        for key in (
            "measured_max_drawdown",
            "measured_max_drawdown_pct",
            "measured_current_drawdown",
            "max_drawdown_limit",
            "max_drawdown_limit_pct",
            "drawdown_field_semantics",
            "garch_active",
            "garch_active_reason",
            "runtime_role",
            "coverage_diagnostics",
        ):
            if merged.get(key) is None and sibling.get(key) is not None:
                merged[key] = sibling[key]
        # Explicit false on sibling must demote even if health filter_active true
        if sibling.get("garch_active") is False:
            merged["garch_active"] = False
            if sibling.get("runtime_role"):
                merged["runtime_role"] = sibling["runtime_role"]
            if sibling.get("garch_active_reason"):
                merged["garch_active_reason"] = sibling["garch_active_reason"]
        payload = merged
    return _normalize_risk_payload(payload, source=source)


def _get_tca_section() -> Dict[str, Any]:
    """TCA execution quality — producers removed v977."""
    return {"available": False}


def _get_overlays_section() -> Dict[str, Any]:
    """All tactical overlay states."""
    overlays: Dict[str, Any] = {}

    # VIX term structure overlay — prefer public diagnostic, fall back to state file
    vixy = _read_json("vixy_hedge.json") or _read_json("vixy_hedge_state.json")
    if vixy:
        # Prefer explicit *_pct keys (public diagnostic schema)
        raw_alloc = vixy.get("current_allocation_pct", vixy.get("current_allocation", 0))
        if isinstance(raw_alloc, dict):
            is_active = any(float(v or 0) > 0 for v in raw_alloc.values())
            alloc_pct = {
                k: _normalize_overlay_allocation_pct(v) for k, v in raw_alloc.items()
            }
        else:
            alloc_pct = _normalize_overlay_allocation_pct(raw_alloc) or 0.0
            is_active = float(alloc_pct or 0) > 0
        overlays["vix_term_structure"] = {
            "active": is_active,
            "allocation": alloc_pct,  # percent points (e.g. 3.0 = 3%)
            "allocation_unit": "percent",
            "last_shift_date": vixy.get("last_signal_date") or vixy.get("last_rebalance"),
            "regime": vixy.get("regime"),
        }
    else:
        overlays["vix_term_structure"] = {"active": False}

    # Collar, crypto, bond duration, mean reversion overlays removed v938-v980

    # Count active overlays
    active_count = sum(1 for o in overlays.values() if o.get("active"))
    total_count = len(overlays)
    overlays["_meta"] = {
        "active_count": active_count,
        "total_count": total_count,
    }

    return overlays


def _get_regime_section() -> Dict[str, Any]:
    """Regime SSOT from regime_state.json (advisory; not order-routing authority)."""
    regime_file = DATA_DIR / "regime_state.json"
    if not regime_file.exists():
        return {
            "available": False,
            "reason": "regime_state.json_missing",
            "note": "Not live order-routing authority (see target_allocations).",
        }
    try:
        data = json.loads(regime_file.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse regime_state.json: %s", e)
        return {
            "available": False,
            "reason": "regime_state_unreadable",
            "error": str(e),
            "note": "Not live order-routing authority (see target_allocations).",
        }
    if not isinstance(data, dict):
        return {"available": False, "reason": "regime_state_invalid"}
    regime = data.get("regime") or data.get("current_regime")
    conf = data.get("confidence", data.get("regime_confidence"))
    if not regime:
        return {
            "available": False,
            "reason": "regime_missing",
            "note": "Not live order-routing authority (see target_allocations).",
        }
    return {
        "available": True,
        "regime": regime,
        "confidence": conf,
        "source": data.get("source"),
        "updated_at": data.get("updated_at") or data.get("generated_at"),
        "previous_regime": data.get("previous_regime"),
        "note": (
            "SSOT from regime_state.json for operator dashboard; "
            "not live order-routing authority (see target_allocations / regime_authority)."
        ),
    }


def _get_attribution_section() -> Dict[str, Any]:
    """Performance attribution from latest daily file."""
    attribution_dir = DATA_DIR / "attribution"
    if not attribution_dir.exists():
        return {"available": False}

    files = sorted(attribution_dir.glob("attribution_*.json"), reverse=True)
    if not files:
        return {"available": False}

    try:
        data = json.loads(files[0].read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to parse dashboard data: %s", e)
        return {"available": False}

    sources = data.get("sources", {})
    source_summary = []
    for sk, sv in sources.items():
        source_summary.append(
            {
                "source": sk,
                "name": sv.get("display_name", sk),
                "category": sv.get("category"),
                "hit_rate": sv.get("hit_rate"),
                "win_rate": sv.get("win_rate"),
                "total_return_bps": sv.get("total_return_bps"),
                "sharpe_contribution": sv.get("sharpe_contribution"),
                "avg_weight": sv.get("avg_weight"),
                "active_days": sv.get("active_days"),
            }
        )

    return {
        "available": True,
        "source": files[0].name,
        "timestamp": data.get("timestamp"),
        "analysis_days": data.get("analysis_days"),
        "sources": sorted(source_summary, key=lambda x: abs(x.get("total_return_bps", 0) or 0), reverse=True),
    }


def _get_adaptive_weights_section() -> Dict[str, Any]:
    """Adaptive ensemble weights from adaptive_weights_state.json."""
    state = _read_json("adaptive_weights_state.json")
    if not state or not state.get("adjusted_weights"):
        return {"available": False}

    adjusted = state.get("adjusted_weights", {})
    multipliers = state.get("multipliers", {})
    baseline = state.get("baseline_weights", {})

    # Compute top changes
    changes = []
    for source, adj_weight in adjusted.items():
        base = baseline.get(source, 0)
        diff = adj_weight - base
        changes.append({
            "source": source,
            "base_weight": round(base, 4),
            "adjusted_weight": round(adj_weight, 4),
            "change": round(diff, 4),
            "multiplier": round(multipliers.get(source, 1.0), 2),
        })
    changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    # Find biggest winners / losers
    top_boosted = [c for c in changes if c["change"] > 0][:3]
    top_reduced = [c for c in changes if c["change"] < 0][:3]

    return {
        "available": True,
        "timestamp": state.get("timestamp"),
        "regime": state.get("regime", "normal"),
        "num_sources": len(adjusted),
        "top_changes": changes[:10],
        "top_boosted": top_boosted,
        "top_reduced": top_reduced,
        "history_count": len(state.get("history", [])),
    }


def _cron_job_is_ok(status: Any) -> bool:
    """True for healthy terminal statuses across tasker/Makefile vocabularies.

    Tasker stamps ``success``; hermes/Makefile sometimes use ``ok``. Treat both
    (and common synonyms) as healthy so unified does not report 0/16 ok with
    16 errors when every job actually succeeded.
    """
    s = str(status or "").strip().lower()
    return s in {
        "ok",
        "success",
        "succeeded",
        "completed",
        "pass",
        "passed",
        "healthy",
    }


def _cron_job_is_pending(status: Any) -> bool:
    s = str(status or "").strip().lower()
    return s in {"pending", "running", "scheduled", "unknown", ""}


def _cron_job_is_disabled(status: Any, *, job: dict | None = None) -> bool:
    s = str(status or "").strip().lower()
    if s in {"disabled", "paused", "manual_only", "skipped"}:
        return True
    if job and (job.get("manual_only") is True or job.get("enabled") is False):
        return True
    return False


def _get_cron_section() -> Dict[str, Any]:
    """Cron job status summary."""
    status = _read_json("cron_status.json")
    if not status:
        return {"available": False, "jobs": []}

    jobs = status.get("jobs", [])
    ok_count = sum(1 for j in jobs if _cron_job_is_ok(j.get("status")))
    pending_count = sum(
        1
        for j in jobs
        if not _cron_job_is_ok(j.get("status"))
        and not _cron_job_is_disabled(j.get("status"), job=j)
        and _cron_job_is_pending(j.get("status"))
    )
    disabled_count = sum(
        1 for j in jobs if _cron_job_is_disabled(j.get("status"), job=j)
    )
    error_count = sum(
        1
        for j in jobs
        if not _cron_job_is_ok(j.get("status"))
        and not _cron_job_is_pending(j.get("status"))
        and not _cron_job_is_disabled(j.get("status"), job=j)
    )

    job_list = []
    for j in jobs:
        raw = j.get("status")
        display = raw
        if _cron_job_is_ok(raw):
            display = "ok" if str(raw).lower() != "ok" else raw
            # keep tasker "success" visible but count as ok; show normalized
            display = "ok"
        elif _cron_job_is_disabled(raw, job=j):
            display = "disabled"
        job_list.append(
            {
                "name": j.get("name"),
                "status": display,
                "raw_status": raw,
                "last_run": j.get("last_run"),
                "duration_seconds": j.get("duration_seconds"),
                "backend": j.get("backend"),
            }
        )

    return {
        "available": True,
        "total": len(jobs),
        "ok": ok_count,
        "pending": pending_count,
        "disabled": disabled_count,
        "errors": error_count,
        "jobs": job_list,
    }


def _get_risk_history_section() -> Dict[str, Any]:
    """Risk metrics trend summary."""
    history = _read_json("risk_metrics_history.json")
    if not history or not isinstance(history, list):
        return {"available": False}

    if len(history) < 2:
        return {"available": False, "data_points": len(history)}

    latest = history[-1]
    earliest = history[0]

    return {
        "available": True,
        "data_points": len(history),
        "date_range": [earliest.get("timestamp"), latest.get("timestamp")],
        "trend": {
            "var_95": {"first": earliest.get("var_95"), "last": latest.get("var_95")},
            "cvar_95": {"first": earliest.get("cvar_95"), "last": latest.get("cvar_95")},
            "cvar_ratio": {"first": earliest.get("cvar_ratio"), "last": latest.get("cvar_ratio")},
            "current_drawdown": {
                "first": earliest.get("current_drawdown"),
                "last": latest.get("current_drawdown"),
            },
            "volatility_annual": {
                "first": earliest.get("volatility_annual"),
                "last": latest.get("volatility_annual"),
            },
        },
    }


# ─────────────────────────────────────────────
#  Main Generator
# ─────────────────────────────────────────────


def generate_unified_dashboard() -> Dict[str, Any]:
    """Generate the complete unified dashboard by reading all state files."""
    payload: Dict[str, Any] = {
        "dashboard_version": "v6.08",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_at_local": datetime.now().isoformat(),
        "health": _get_health_section(),
        "portfolio": _get_portfolio_section(),
        "risk": _get_risk_section(),
        "risk_history": _get_risk_history_section(),
        "tca": _get_tca_section(),
        "overlays": _get_overlays_section(),
        "regime": _get_regime_section(),
        "attribution": _get_attribution_section(),
        "adaptive_weights": _get_adaptive_weights_section(),
        "cron": _get_cron_section(),
    }
    # Operator lag detection (parity with stats/analytics/graduation stamps)
    try:
        from src.dashboard.generator import _stamp_generator_git_sha

        payload = _stamp_generator_git_sha(payload)
    except Exception:  # noqa: BLE001 — never block unified generate on stamp
        pass
    return payload


# ─────────────────────────────────────────────
#  Output / Display
# ─────────────────────────────────────────────


def _fmt(v: Any, suffix: str = "") -> str:
    """Format a value for display, handling None."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def _fmt_pct(v: Any) -> str:
    """Format as percentage."""
    if v is None:
        return "N/A"
    return f"{v:.2f}%"


def _status_badge(ok: bool) -> str:
    return "✅" if ok else "❌"


def print_summary(dashboard: Dict[str, Any]) -> None:
    """Print a human-readable summary of the unified dashboard."""
    gen = dashboard.get("generated_at_local", "unknown")
    logger.info(f"{'=' * 60}")
    ver = dashboard.get("dashboard_version", "?").lstrip("v")
    logger.info(f"  PORTFOLIO-LAB UNIFIED DASHBOARD  v{ver}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Generated: {gen}")
    logger.info("")

    # ── Health ──
    health = dashboard.get("health", {})
    if health.get("available"):
        status = health.get("status", "unknown")
        badge = "✅" if status == "healthy" else "⚠️"
        logger.info(f"  {badge} HEALTH: {status.upper()}")
        logger.info(f"     Checks: {health.get('checks_passed')}/{health.get('checks_total')} passed")
        for comp_name, comp in health.get("components", {}).items():
            logger.info(f"       {_status_badge(comp.get('ok', False))} {comp_name}: {comp.get('status', '?')}")
        alerts = health.get("alerts", [])
        if alerts:
            logger.info(f"     Alerts ({len(alerts)}):")
            for a in alerts:
                logger.info(f"       ⚠ {a}")
    else:
        logger.info("  ❌ HEALTH: not available")

    logger.info("")

    # ── Portfolio ──
    portfolio = dashboard.get("portfolio", {})
    if portfolio.get("available"):
        logger.info(f"  💼 PORTFOLIO ({portfolio.get('mode', 'paper').upper()})")
        logger.info(f"     Total: ${_fmt(portfolio.get('total_value'))}")
        logger.info(f"     Cash: ${_fmt(portfolio.get('cash'))} ({_fmt(portfolio.get('cash_pct'))}%)")
        logger.info(f"     History: {portfolio.get('history_count', 0)} snapshots")
        for pos in portfolio.get("positions", []):
            sym = pos.get("symbol", "?")
            val = pos.get("value", 0)
            wt = pos.get("weight", 0)
            logger.info(f"       {sym:<6} ${val:>8,.0f}  ({wt:.1f}%)")
    else:
        logger.info("  ❌ PORTFOLIO: not available")

    logger.info("")

    # ── Risk ──
    risk = dashboard.get("risk", {})
    if risk.get("available"):
        dd = risk.get("current_drawdown", 0)
        dd_badge = "✅" if dd is not None and abs(dd) < 10 else "⚠️" if dd is not None and abs(dd) < 20 else "🚨"
        logger.info(f"  {dd_badge} RISK METRICS")
        logger.info(f"     VaR (95%): {_fmt(risk.get('var_95_daily'))}%")
        logger.info(f"     CVaR (95%): {_fmt(risk.get('cvar_95_daily'))}%")
        logger.info(f"     CVaR Ratio: {_fmt(risk.get('cvar_ratio'))}")
        logger.info(f"     Tail: {risk.get('tail_severity', '?').upper()}")
        logger.info(f"     Max DD: {_fmt(risk.get('max_drawdown'))}%")
        logger.info(f"     Current DD: {_fmt(risk.get('current_drawdown'))}%")
        logger.info(f"     Vol (ann): {_fmt(risk.get('volatility_annual'))}%")
        garch = "active" if risk.get("garch_active") else "inactive"
        logger.info(f"     GARCH: {garch}")
    else:
        logger.info("  ❌ RISK: not available")

    logger.info("")

    # ── Overlays ──
    overlays = dashboard.get("overlays", {})
    meta = overlays.pop("_meta", {"active_count": 0, "total_count": 0})
    active_o = meta.get("active_count", 0)
    total_o = meta.get("total_count", 0)
    logger.info(f"  🎯 OVERLAYS ({active_o}/{total_o} active)")
    for name, data in sorted(overlays.items()):
        active = data.get("active", False)
        badge = "✅" if active else "⏹"
        status_text = "active" if active else "inactive"
        detail = ""
        if name == "vix_term_structure" and active:
            alloc = data.get("allocation", 0)
            # allocation is already percent points after normalize
            if isinstance(alloc, dict):
                detail = f" SPY={_fmt_pct(alloc.get('SPY', 0))} GLD={_fmt_pct(alloc.get('GLD', 0))}"
            else:
                detail = f" alloc={_fmt_pct(alloc)}"
        logger.info(f"       {badge} {name:<20} {status_text}{detail}")

    logger.info("")

    # ── Regime ──
    regime = dashboard.get("regime", {})
    if regime.get("available"):
        clf = regime.get("classifier", {})
        opt = regime.get("optimizer", {})
        if clf:
            reg = clf.get("current_regime", "?")
            conf = clf.get("confidence")
            logger.info(f"  🌡 REGIME CLASSIFIER: {reg.upper()} (conf={_fmt(conf)})")
        if opt:
            logger.info(f"     Optimizer: method={opt.get('method')}, solver={opt.get('solver_status')}")
            logger.info(f"     Expected Sharpe: {_fmt(opt.get('expected_sharpe'))}")
        rb = regime.get("risk_budget", {})
        if rb:
            logger.info(f"     Risk Budget: {rb.get('regime')}, vol {_fmt(rb.get('portfolio_vol_after', 0) * 100, '%')}")
    else:
        logger.info("  ❌ REGIME: not available")

    logger.info("")

    # ── TCA ──
    tca = dashboard.get("tca", {})
    if tca.get("available"):
        sc = tca.get("scorecard", {})
        fb = tca.get("feedback", {})
        if sc:
            logger.info(f"  📊 TCA SCORECARD")
            logger.info(f"     Orders: {sc.get('total_orders')}, Notional: ${_fmt(sc.get('total_notional'))}")
            logger.info(f"     Avg Slippage: {_fmt(sc.get('avg_slippage_bps'))} bps")
            logger.info(f"     Avg Quality: {_fmt(sc.get('avg_quality_score'))}/100")
        if fb:
            logger.info(f"     Feedback: urgency={_fmt(fb.get('urgency_global_offset'))}, min_trade={fb.get('min_trade_global_multiplier')}x")
            logger.info(f"     Quality: {_fmt(fb.get('overall_quality'))}/100 ({fb.get('quality_label', '?')})")
    else:
        logger.info("  ❌ TCA: not available")

    logger.info("")

    # ── Attribution ──
    attr = dashboard.get("attribution", {})
    if attr.get("available"):
        logger.info(f"  📈 PERFORMANCE ATTRIBUTION ({attr.get('analysis_days', '?')} days)")
        for src in attr.get("sources", [])[:6]:  # Top 6 sources
            hr = src.get("hit_rate")
            ret = src.get("total_return_bps")
            logger.info(
                f"       {src.get('name', '?'):<25}"
                f" hit={_fmt(hr * 100, '%') if hr else 'N/A':>6}"
                f" ret={_fmt(ret):>8} bps"
                f" sharpe={_fmt(src.get('sharpe_contribution'))}"
            )
        remaining = max(0, len(attr.get("sources", [])) - 6)
        if remaining > 0:
            logger.info(f"       ... and {remaining} more sources")
    else:
        logger.info("  ❌ ATTRIBUTION: not available")

    logger.info("")

    # ── Adaptive Weights ──
    aw = dashboard.get("adaptive_weights", {})
    if aw.get("available"):
        logger.info(f"  🔄 ADAPTIVE ENSEMBLE WEIGHTS (v6.09)")
        logger.info(f"     Regime: {aw.get('regime', '?')} | Sources: {aw.get('num_sources')} | History: {aw.get('history_count')} adj")
        for c in aw.get("top_changes", [])[:5]:
            arrow = "⬆" if c["change"] > 0 else "⬇"
            logger.info(f"       {arrow} {c['source']:<25} {c['base_weight']:.4f} → {c['adjusted_weight']:.4f} (×{c['multiplier']})")
    else:
        logger.info(f"  🔄 ADAPTIVE WEIGHTS: not available (run attribution first)")

    logger.info("")

    # ── Cron ──
    cron = dashboard.get("cron", {})
    if cron.get("available"):
        badge = "✅" if cron.get("errors", 0) == 0 else "⚠️"
        disabled = cron.get("disabled", 0)
        disabled_bit = f", {disabled} disabled" if disabled else ""
        logger.info(
            f"  {badge} CRON JOBS: {cron.get('ok')}/{cron.get('total')} ok, "
            f"{cron.get('errors')} errors{disabled_bit}"
        )
        for job in cron.get("jobs", []):
            dur = job.get("duration_seconds", 0)
            status = job.get("status", "?")
            if status == "ok":
                badge = "✅"
            elif status == "pending":
                badge = "⏳"
            elif status == "disabled":
                badge = "⏸"
            else:
                badge = "❌"
            logger.info(f"       {badge} {job.get('name', '?'):<30} {status:<8} {_fmt(dur, 's')}")
    else:
        logger.info("  ❌ CRON: not available")

    logger.info(f"{'=' * 60}")


def generate_status_text() -> str:
    """Generate a concise one-line status for health monitor integration."""
    dashboard = generate_unified_dashboard()

    health = dashboard.get("health", {})
    portfolio = dashboard.get("portfolio", {})
    risk = dashboard.get("risk", {})
    cron = dashboard.get("cron", {})

    health_ok = health.get("status") == "healthy" if health.get("available") else False
    dd = risk.get("current_drawdown", 0) if risk.get("available") else 0
    val = portfolio.get("total_value", 0) if portfolio.get("available") else 0
    cron_ok = cron.get("errors", 99) == 0 if cron.get("available") else False

    status = "✅" if (health_ok and cron_ok) else "⚠️"
    return (
        f"{status} Unified: val=${val:,.0f}, dd={dd:.1f}%, "
        f"health={'ok' if health_ok else 'warn'}, cron={'ok' if cron_ok else 'err'}"
    )


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────


def _unified_payload_section_score(dashboard: Dict[str, Any]) -> int:
    """Count operator-critical sections that report available=True.

    Used to refuse hollow dual-writes (all sections unavailable) that would
    clobber a richer public SSOT when tasker DATA_DIR is empty/mis-pointed.
    """
    score = 0
    for key in ("health", "portfolio", "risk", "regime", "cron", "attribution"):
        sec = dashboard.get(key)
        if isinstance(sec, dict) and sec.get("available") is True:
            score += 1
    # risk_history uses available too
    rh = dashboard.get("risk_history")
    if isinstance(rh, dict) and rh.get("available") is True:
        score += 1
    return score


def _save_unified_dashboard(dashboard: Dict[str, Any]) -> list:
    """Write private DATA_DIR and dual-write PUBLIC_DATA_DIR (operator WWW SSOT).

    Public dual-write is skipped when the payload is hollow (no critical
    sections available) so a misconfigured DATA_DIR cannot replace a richer
    public file with all-unavailable stubs.

    Attaches ``provenance_completeness`` (Batch AT / H10) for split-brain lag
    forensics when private and public trees diverge.
    """
    written: list = []
    private_path = Path(DATA_DIR) / "unified_dashboard.json"
    public_path = Path(PUBLIC_DATA_DIR) / "unified_dashboard.json"

    score = _unified_payload_section_score(dashboard)
    dual_attempted = score >= 1
    dual_ok: bool | None = None
    skip_reason: str | None = None
    paths_identical = False
    try:
        paths_identical = private_path.resolve() == public_path.resolve()
    except OSError:
        paths_identical = False

    if score < 1:
        skip_reason = f"hollow_payload_section_score={score}"
        dual_attempted = False
        dual_ok = False
    elif paths_identical:
        dual_attempted = False
        dual_ok = True
        skip_reason = "paths_identical"

    # Stamp intent before private write so private always carries completeness
    try:
        from src.dashboard.generator import _attach_dual_write_provenance

        dashboard = _attach_dual_write_provenance(
            dashboard,
            private_path=private_path,
            public_path=public_path,
            dual_write_attempted=dual_attempted and not paths_identical,
            dual_write_ok=dual_ok,
            paths_identical=paths_identical,
            note=skip_reason,
        )
        # section score for operators
        pc = dashboard.get("provenance_completeness")
        if isinstance(pc, dict):
            pc["section_score"] = score
    except Exception:  # noqa: BLE001 — never block save on provenance
        pass

    save_results_json(dashboard, output_path=str(private_path))
    written.append(private_path)
    logger.info("Saved unified dashboard to %s", private_path)

    if score < 1:
        logger.warning(
            "Skipping public unified dual-write: hollow payload "
            "(section_score=%s; refuse clobber of operator SSOT)",
            score,
        )
        return written

    if paths_identical:
        return written

    try:
        public_root = Path(PUBLIC_DATA_DIR)
        public_root.mkdir(parents=True, exist_ok=True)
        # Do not clobber a richer public file with a thinner one
        if public_path.exists():
            try:
                prior = json.loads(public_path.read_text(encoding="utf-8"))
                prior_score = _unified_payload_section_score(prior)
                if prior_score > score:
                    logger.warning(
                        "Skipping public unified dual-write: prior score=%s > new=%s",
                        prior_score,
                        score,
                    )
                    try:
                        from src.dashboard.generator import _attach_dual_write_provenance

                        dashboard = _attach_dual_write_provenance(
                            dashboard,
                            private_path=private_path,
                            public_path=public_path,
                            dual_write_attempted=True,
                            dual_write_ok=False,
                            paths_identical=False,
                            note=f"skipped_thinner_payload prior={prior_score} new={score}",
                        )
                        pc = dashboard.get("provenance_completeness")
                        if isinstance(pc, dict):
                            pc["section_score"] = score
                        save_results_json(dashboard, output_path=str(private_path))
                    except Exception:  # noqa: BLE001
                        pass
                    return written
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        # Mark success on payload before public write
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            dashboard = _attach_dual_write_provenance(
                dashboard,
                private_path=private_path,
                public_path=public_path,
                dual_write_attempted=True,
                dual_write_ok=True,
                paths_identical=False,
            )
            pc = dashboard.get("provenance_completeness")
            if isinstance(pc, dict):
                pc["section_score"] = score
            save_results_json(dashboard, output_path=str(private_path))
        except Exception:  # noqa: BLE001
            pass
        # Atomic plane-specific write so private diagnostic paths do not cross
        # into the public unified-dashboard artifact.
        from src.monitor.signal_authority import (
            is_ephemeral_write_path,
            serialize_json_payload,
        )

        tmp_path = public_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            serialize_json_payload(
                dashboard,
                output_path=public_path,
                public=not is_ephemeral_write_path(public_path),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(public_path)
        written.append(public_path)
        # Batch CJ: recompute lag/hash after both trees exist (clears sticky
        # dual_write_lag_stale frozen from pre-write public mtime).
        try:
            from src.dashboard.generator import finalize_dual_write_provenance_after_sync

            dashboard = finalize_dual_write_provenance_after_sync(
                dashboard,
                private_path=private_path,
                public_path=public_path,
                dual_write_ok=True,
                note=(
                    f"post_sync unified dual-write (Batch CJ); section_score={score}"
                ),
            )
            pc = dashboard.get("provenance_completeness")
            if isinstance(pc, dict):
                pc["section_score"] = score
                # Persist score into both trees (finalize already wrote body once)
                from src.monitor.signal_authority import write_json_multi_dest

                write_json_multi_dest(
                    dashboard,
                    private_path=private_path,
                    public_path=public_path,
                    soft_mirror_repo=False,
                )
        except Exception:  # noqa: BLE001 — never block save on post-sync stamp
            pass
        logger.info(
            "Public dual-write unified dashboard to %s (section_score=%s)",
            public_path,
            score,
        )
    except OSError as exc:
        logger.warning("Public unified_dashboard dual-write failed: %s", exc)
        try:
            from src.dashboard.generator import _attach_dual_write_provenance

            dashboard = _attach_dual_write_provenance(
                dashboard,
                private_path=private_path,
                public_path=public_path,
                dual_write_attempted=True,
                dual_write_ok=False,
                paths_identical=False,
                note=str(exc),
            )
            pc = dashboard.get("provenance_completeness")
            if isinstance(pc, dict):
                pc["section_score"] = score
            save_results_json(dashboard, output_path=str(private_path))
        except Exception:  # noqa: BLE001
            pass
    return written


def main():
    dashboard = generate_unified_dashboard()

    if "--json" in sys.argv or "--save" in sys.argv:
        _save_unified_dashboard(dashboard)

    if "--status-text" in sys.argv:
        logger.info(generate_status_text())
        return

    if "--check" in sys.argv:
        health_ok = safe_get(dashboard, "health", "status") == "healthy"
        cron_ok = safe_get(dashboard, "cron", "errors", default=99) == 0
        sys.exit(0 if (health_ok and cron_ok) else 1)

    # Default: print summary
    print_summary(dashboard)


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
