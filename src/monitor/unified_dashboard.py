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
from typing import Any, Dict, Optional

from src.paths import DATA_DIR
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
    """System health from .health_report.json.

    Supports two formats:
    - Legacy: {status, checks, summary, alerts} — structured health report
    - GARCH flat: {var_95, cvar_95, tail_severity, ...} — GARCHCVaRMetrics asdict
    """
    report = _read_json(".health_report.json")
    if not report:
        return {"available": False, "status": "unknown"}

    # Legacy format: has explicit status field and checks
    if "checks" in report or "status" in report:
        checks = report.get("checks", {})
        return {
            "available": True,
            "timestamp": report.get("timestamp"),
            "status": report.get("status", "unknown"),
            "checks_passed": safe_get(report, "summary", "passed", default=0),
            "checks_total": safe_get(report, "summary", "total_checks", default=0),
            "alerts": report.get("alerts", []),
            "components": {
                name: {
                    "status": c.get("status"),
                    "ok": c.get("ok", False),
                }
                for name, c in checks.items()
            },
        }

    # GARCH flat format: derive status from tail_severity and cvar_ratio
    tail = report.get("tail_severity", "normal")
    cvar_ratio = report.get("cvar_ratio", 1.0)
    if tail in ("extreme", "severe") or cvar_ratio > 3.0:
        status = "unhealthy"
    else:
        status = "healthy"

    return {
        "available": True,
        "timestamp": report.get("timestamp"),
        "status": status,
        "checks_passed": 1 if status == "healthy" else 0,
        "checks_total": 1,
        "alerts": [] if status == "healthy" else [
            f"Tail severity: {tail}, CVaR ratio: {cvar_ratio:.2f}"
        ],
        "components": {
            "garch_cvar": {
                "status": status,
                "ok": status == "healthy",
                "var_95": report.get("var_95"),
                "cvar_95": report.get("cvar_95"),
                "tail_severity": tail,
            },
        },
    }


def _get_portfolio_section() -> Dict[str, Any]:
    """Current portfolio state from portfolio_paper.json."""
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

    return {
        "available": True,
        "total_value": round(total_value, 2),
        "cash": round(paper.get("cash", 0), 2),
        "cash_pct": round(paper.get("cash", 0) / total_value * 100, 2) if total_value > 0 else 0,
        "positions": sorted(pos_list, key=lambda x: x["value"], reverse=True),
        "update_timestamp": paper.get("updated"),
        "mode": paper.get("mode", "paper"),
        "history_count": len(paper.get("history", [])),
    }


def _normalize_risk_payload(metrics: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    """Map risk_metrics.json or GARCH .health_report.json into unified risk section."""
    # health_report uses var_95/cvar_95; risk_metrics uses var_95_daily/cvar_95_daily
    var_daily = metrics.get("var_95_daily", metrics.get("var_95"))
    cvar_daily = metrics.get("cvar_95_daily", metrics.get("cvar_95"))
    garch_active = metrics.get("garch_active")
    if garch_active is None:
        garch_active = bool(metrics.get("filter_active", False))
    garch_filtered = metrics.get("garch_filtered")
    if garch_filtered is None:
        garch_filtered = bool(garch_active)
    return {
        "available": True,
        "timestamp": metrics.get("timestamp"),
        "var_95_daily": var_daily,
        "cvar_95_daily": cvar_daily,
        "cvar_ratio": metrics.get("cvar_ratio"),
        "tail_severity": metrics.get("tail_severity"),
        "max_drawdown": metrics.get("max_drawdown"),
        "current_drawdown": metrics.get("current_drawdown"),
        "volatility_annual": metrics.get("volatility_annual"),
        "garch_active": garch_active,
        "garch_filtered": garch_filtered,
        "source": source,
    }


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
    """Risk metrics: prefer fresher GARCH .health_report over orphan risk_metrics.json."""
    metrics = _read_json("risk_metrics.json")
    health = _read_json(".health_report.json")

    candidates: list[tuple[str, Dict[str, Any], Optional[datetime]]] = []
    if _payload_has_risk_metrics(health, require_var_fields=True):
        candidates.append(("health_report", health, _parse_ts(health.get("timestamp"))))
    if _payload_has_risk_metrics(metrics, require_var_fields=False):
        candidates.append(("risk_metrics", metrics, _parse_ts(metrics.get("timestamp"))))

    if not candidates:
        return {"available": False}

    # Prefer newer timestamp; on tie prefer health_report (garch job SSOT)
    def sort_key(item: tuple[str, Dict[str, Any], Optional[datetime]]):
        source, _payload, ts = item
        # missing ts sorts last
        ts_ord = ts or datetime.min
        source_rank = 0 if source == "health_report" else 1
        return (ts_ord, -source_rank)

    # max by timestamp, then health_report
    best = max(
        candidates,
        key=lambda it: (
            it[2] or datetime.min,
            1 if it[0] == "health_report" else 0,
        ),
    )
    source, payload, _ts = best
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


def _get_cron_section() -> Dict[str, Any]:
    """Cron job status summary."""
    status = _read_json("cron_status.json")
    if not status:
        return {"available": False, "jobs": []}

    jobs = status.get("jobs", [])
    ok_count = sum(1 for j in jobs if j.get("status") == "ok")
    pending_count = sum(1 for j in jobs if j.get("status") == "pending")
    error_count = sum(1 for j in jobs if j.get("status") not in ("ok", "pending"))

    job_list = []
    for j in jobs:
        job_list.append(
            {
                "name": j.get("name"),
                "status": j.get("status"),
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
    return {
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
        logger.info(f"  {badge} CRON JOBS: {cron.get('ok')}/{cron.get('total')} ok, {cron.get('errors')} errors")
        for job in cron.get("jobs", []):
            dur = job.get("duration_seconds", 0)
            status = job.get("status", "?")
            badge = "✅" if status == "ok" else "⏳" if status == "pending" else "❌"
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


def main():
    dashboard = generate_unified_dashboard()

    if "--json" in sys.argv or "--save" in sys.argv:
        out_path = DATA_DIR / "unified_dashboard.json"
        save_results_json(dashboard, output_path=str(out_path))
        logger.info("Saved unified dashboard to %s", out_path)

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
