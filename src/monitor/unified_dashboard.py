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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()


# ─────────────────────────────────────────────
#  Section Readers
# ─────────────────────────────────────────────


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
    """System health from .health_report.json."""
    report = _read_json(".health_report.json")
    if not report:
        return {"available": False, "status": "unknown"}

    checks = report.get("checks", {})
    return {
        "available": True,
        "timestamp": report.get("timestamp"),
        "status": report.get("status", "unknown"),
        "checks_passed": report.get("summary", {}).get("passed", 0),
        "checks_total": report.get("summary", {}).get("total_checks", 0),
        "alerts": report.get("alerts", []),
        "components": {
            name: {
                "status": c.get("status"),
                "ok": c.get("ok", False),
            }
            for name, c in checks.items()
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


def _get_risk_section() -> Dict[str, Any]:
    """Risk metrics from risk_metrics.json."""
    metrics = _read_json("risk_metrics.json")
    if not metrics:
        return {"available": False}

    return {
        "available": True,
        "timestamp": metrics.get("timestamp"),
        "var_95_daily": metrics.get("var_95_daily"),
        "cvar_95_daily": metrics.get("cvar_95_daily"),
        "cvar_ratio": metrics.get("cvar_ratio"),
        "tail_severity": metrics.get("tail_severity"),
        "max_drawdown": metrics.get("max_drawdown"),
        "current_drawdown": metrics.get("current_drawdown"),
        "volatility_annual": metrics.get("volatility_annual"),
        "garch_active": metrics.get("garch_active", False),
        "garch_filtered": metrics.get("garch_filtered", False),
    }


def _get_tca_section() -> Dict[str, Any]:
    """TCA execution quality from scorecard + feedback state."""
    scorecard = _read_json("tca_scorecard.json")
    feedback = _read_json("tca_feedback_state.json")

    section: Dict[str, Any] = {"available": False}

    if scorecard:
        section["available"] = True
        section["scorecard"] = {
            "generated": scorecard.get("generated"),
            "total_orders": scorecard.get("total_orders", 0),
            "total_notional": scorecard.get("total_notional", 0),
            "avg_slippage_bps": scorecard.get("avg_slippage_bps"),
            "avg_quality_score": scorecard.get("avg_quality_score"),
            "weighted_slippage_bps": scorecard.get("weighted_slippage_bps"),
            "by_symbol": scorecard.get("by_symbol", {}),
        }

    if feedback:
        section["available"] = True
        section["feedback"] = {
            "overall_quality": feedback.get("overall_quality"),
            "quality_label": (
                "poor"
                if feedback.get("overall_quality", 0) < 50
                else "fair"
                if feedback.get("overall_quality", 0) < 70
                else "good"
            ),
            "urgency_global_offset": feedback.get("urgency_global_offset"),
            "min_trade_global_multiplier": feedback.get("min_trade_global_multiplier"),
            "cost_calibration_global": feedback.get("cost_calibration_global"),
            "symbols": feedback.get("symbols", {}),
        }

    return section


def _get_overlays_section() -> Dict[str, Any]:
    """All tactical overlay states."""
    overlays: Dict[str, Any] = {}

    # Collar overlay
    collar = _read_json("collar_overlay_state.json")
    if collar:
        overlays["collar"] = {
            "active": collar.get("status") == "active",
            "status": collar.get("status"),
            "call_strike": collar.get("current_call_strike"),
            "put_strike": collar.get("current_put_strike"),
            "ytd_premium_collected": collar.get("ytd_premium_collected"),
            "ytd_premium_paid": collar.get("ytd_premium_paid"),
            "total_rolls": collar.get("total_rolls"),
            "last_roll_date": collar.get("last_roll_date"),
        }
    else:
        overlays["collar"] = {"active": False}

    # Crypto allocation
    crypto = _read_json("crypto_allocation_state.json")
    if crypto:
        overlays["crypto"] = {
            "active": crypto.get("status") == "active",
            "status": crypto.get("status"),
            "btc_weight": crypto.get("btc_weight"),
            "eth_weight": crypto.get("eth_weight"),
            "total_crypto": crypto.get("total_crypto"),
            "gld_reduction": crypto.get("gld_reduction"),
            "entry_date": crypto.get("entry_date"),
            "exit_date": crypto.get("exit_date"),
        }
    else:
        overlays["crypto"] = {"active": False}

    # Bond duration rotation
    bond = _read_json("bond_duration_state.json")
    if bond:
        overlays["bond_duration"] = {
            "active": bond.get("status") == "active",
            "status": bond.get("status"),
            "position": bond.get("current_position"),
            "tlt_weight": bond.get("tlt_weight"),
            "ief_weight": bond.get("ief_weight"),
            "shy_weight": bond.get("shy_weight"),
            "last_rotation_date": bond.get("last_rotation_date"),
        }
    else:
        overlays["bond_duration"] = {"active": False}

    # VIX term structure overlay
    vix = _read_json("vix_overlay_state.json")
    if vix:
        overlays["vix_term_structure"] = {
            "active": vix.get("disabled_until") is None,
            "allocation": vix.get("allocation"),
            "last_shift_date": vix.get("last_shift_date"),
            "disabled_until": vix.get("disabled_until"),
            "shift_count": len(vix.get("shift_history", [])),
        }
    else:
        overlays["vix_term_structure"] = {"active": False}

    # Mean reversion
    mr = _read_json("mean_reversion_state.json")
    if mr:
        overlays["mean_reversion"] = {
            "active": mr.get("active", False),
            "allocation_pct": mr.get("allocation_pct"),
            "entry_vix": mr.get("entry_vix"),
            "hold_days": mr.get("hold_days"),
        }
    else:
        overlays["mean_reversion"] = {"active": False}

    # Count active overlays
    active_count = sum(1 for o in overlays.values() if o.get("active"))
    total_count = len(overlays)
    overlays["_meta"] = {
        "active_count": active_count,
        "total_count": total_count,
    }

    return overlays


def _get_regime_section() -> Dict[str, Any]:
    """Regime classifier + optimizer states."""
    section: Dict[str, Any] = {"available": False}

    classifier = _read_json("regime_classifier_state.json")
    optimizer = _read_json("regime_optimizer_state.json")
    risk_budget = _read_json("risk_budget_state.json")

    if classifier:
        section["available"] = True
        section["classifier"] = {
            "current_regime": classifier.get("current_regime"),
            "previous_regime": classifier.get("previous_regime"),
            "confidence": classifier.get("last_reading", {}).get("confidence"),
            "last_updated": classifier.get("last_updated"),
            "history_length": len(classifier.get("history", [])),
        }

    if optimizer:
        section["available"] = True
        section["optimizer"] = {
            "current_regime": optimizer.get("current_regime"),
            "regime_confidence": optimizer.get("regime_confidence"),
            "method": optimizer.get("method"),
            "weights": optimizer.get("weights"),
            "expected_return": optimizer.get("expected_return"),
            "expected_volatility": optimizer.get("expected_volatility"),
            "expected_sharpe": optimizer.get("expected_sharpe"),
            "solver_status": optimizer.get("solver_status"),
            "constraints_satisfied": optimizer.get("constraints_satisfied"),
        }

    if risk_budget:
        section["available"] = True
        section["risk_budget"] = {
            "regime": risk_budget.get("regime"),
            "weights": risk_budget.get("weights"),
            "all_budgets_met": risk_budget.get("all_budgets_met"),
            "portfolio_vol_before": risk_budget.get("portfolio_vol_before"),
            "portfolio_vol_after": risk_budget.get("portfolio_vol_after"),
        }

    return section


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
    except (json.JSONDecodeError, OSError):
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
    print(f"{'=' * 60}")
    ver = dashboard.get("dashboard_version", "?").lstrip("v")
    print(f"  PORTFOLIO-LAB UNIFIED DASHBOARD  v{ver}")
    print(f"{'=' * 60}")
    print(f"  Generated: {gen}")
    print()

    # ── Health ──
    health = dashboard.get("health", {})
    if health.get("available"):
        status = health.get("status", "unknown")
        badge = "✅" if status == "healthy" else "⚠️"
        print(f"  {badge} HEALTH: {status.upper()}")
        print(f"     Checks: {health.get('checks_passed')}/{health.get('checks_total')} passed")
        for comp_name, comp in health.get("components", {}).items():
            print(f"       {_status_badge(comp.get('ok', False))} {comp_name}: {comp.get('status', '?')}")
        alerts = health.get("alerts", [])
        if alerts:
            print(f"     Alerts ({len(alerts)}):")
            for a in alerts:
                print(f"       ⚠ {a}")
    else:
        print("  ❌ HEALTH: not available")

    print()

    # ── Portfolio ──
    portfolio = dashboard.get("portfolio", {})
    if portfolio.get("available"):
        print(f"  💼 PORTFOLIO ({portfolio.get('mode', 'paper').upper()})")
        print(f"     Total: ${_fmt(portfolio.get('total_value'))}")
        print(f"     Cash: ${_fmt(portfolio.get('cash'))} ({_fmt(portfolio.get('cash_pct'))}%)")
        print(f"     History: {portfolio.get('history_count', 0)} snapshots")
        for pos in portfolio.get("positions", []):
            sym = pos.get("symbol", "?")
            val = pos.get("value", 0)
            wt = pos.get("weight", 0)
            print(f"       {sym:<6} ${val:>8,.0f}  ({wt:.1f}%)")
    else:
        print("  ❌ PORTFOLIO: not available")

    print()

    # ── Risk ──
    risk = dashboard.get("risk", {})
    if risk.get("available"):
        dd = risk.get("current_drawdown", 0)
        dd_badge = "✅" if dd is not None and abs(dd) < 10 else "⚠️" if dd is not None and abs(dd) < 20 else "🚨"
        print(f"  {dd_badge} RISK METRICS")
        print(f"     VaR (95%): {_fmt(risk.get('var_95_daily'))}%")
        print(f"     CVaR (95%): {_fmt(risk.get('cvar_95_daily'))}%")
        print(f"     CVaR Ratio: {_fmt(risk.get('cvar_ratio'))}")
        print(f"     Tail: {risk.get('tail_severity', '?').upper()}")
        print(f"     Max DD: {_fmt(risk.get('max_drawdown'))}%")
        print(f"     Current DD: {_fmt(risk.get('current_drawdown'))}%")
        print(f"     Vol (ann): {_fmt(risk.get('volatility_annual'))}%")
        garch = "active" if risk.get("garch_active") else "inactive"
        print(f"     GARCH: {garch}")
    else:
        print("  ❌ RISK: not available")

    print()

    # ── Overlays ──
    overlays = dashboard.get("overlays", {})
    meta = overlays.pop("_meta", {"active_count": 0, "total_count": 0})
    active_o = meta.get("active_count", 0)
    total_o = meta.get("total_count", 0)
    print(f"  🎯 OVERLAYS ({active_o}/{total_o} active)")
    for name, data in sorted(overlays.items()):
        active = data.get("active", False)
        badge = "✅" if active else "⏹"
        status_text = "active" if active else "inactive"
        detail = ""
        if name == "collar" and active:
            detail = f" call={data.get('call_strike')} put={data.get('put_strike')}"
        elif name == "crypto" and active:
            detail = f" {_fmt(data.get('total_crypto', 0) * 100, '%')} portfolio"
        elif name == "bond_duration" and active:
            detail = f" pos={data.get('position')}"
        elif name == "vix_term_structure" and active:
            alloc = data.get("allocation", {})
            detail = f" SPY={_fmt_pct(alloc.get('SPY', 0) * 100)} GLD={_fmt_pct(alloc.get('GLD', 0) * 100)}"
        elif name == "mean_reversion" and active:
            detail = f" {data.get('allocation_pct')}% alloc"
        print(f"       {badge} {name:<20} {status_text}{detail}")

    print()

    # ── Regime ──
    regime = dashboard.get("regime", {})
    if regime.get("available"):
        clf = regime.get("classifier", {})
        opt = regime.get("optimizer", {})
        if clf:
            reg = clf.get("current_regime", "?")
            conf = clf.get("confidence")
            print(f"  🌡 REGIME CLASSIFIER: {reg.upper()} (conf={_fmt(conf)})")
        if opt:
            print(f"     Optimizer: method={opt.get('method')}, solver={opt.get('solver_status')}")
            print(f"     Expected Sharpe: {_fmt(opt.get('expected_sharpe'))}")
        rb = regime.get("risk_budget", {})
        if rb:
            print(f"     Risk Budget: {rb.get('regime')}, vol {_fmt(rb.get('portfolio_vol_after', 0) * 100, '%')}")
    else:
        print("  ❌ REGIME: not available")

    print()

    # ── TCA ──
    tca = dashboard.get("tca", {})
    if tca.get("available"):
        sc = tca.get("scorecard", {})
        fb = tca.get("feedback", {})
        if sc:
            print(f"  📊 TCA SCORECARD")
            print(f"     Orders: {sc.get('total_orders')}, Notional: ${_fmt(sc.get('total_notional'))}")
            print(f"     Avg Slippage: {_fmt(sc.get('avg_slippage_bps'))} bps")
            print(f"     Avg Quality: {_fmt(sc.get('avg_quality_score'))}/100")
        if fb:
            print(f"     Feedback: urgency={_fmt(fb.get('urgency_global_offset'))}, min_trade={fb.get('min_trade_global_multiplier')}x")
            print(f"     Quality: {_fmt(fb.get('overall_quality'))}/100 ({fb.get('quality_label', '?')})")
    else:
        print("  ❌ TCA: not available")

    print()

    # ── Attribution ──
    attr = dashboard.get("attribution", {})
    if attr.get("available"):
        print(f"  📈 PERFORMANCE ATTRIBUTION ({attr.get('analysis_days', '?')} days)")
        for src in attr.get("sources", [])[:6]:  # Top 6 sources
            hr = src.get("hit_rate")
            ret = src.get("total_return_bps")
            print(
                f"       {src.get('name', '?'):<25}"
                f" hit={_fmt(hr * 100, '%') if hr else 'N/A':>6}"
                f" ret={_fmt(ret):>8} bps"
                f" sharpe={_fmt(src.get('sharpe_contribution'))}"
            )
        remaining = max(0, len(attr.get("sources", [])) - 6)
        if remaining > 0:
            print(f"       ... and {remaining} more sources")
    else:
        print("  ❌ ATTRIBUTION: not available")

    print()

    # ── Adaptive Weights ──
    aw = dashboard.get("adaptive_weights", {})
    if aw.get("available"):
        print(f"  🔄 ADAPTIVE ENSEMBLE WEIGHTS (v6.09)")
        print(f"     Regime: {aw.get('regime', '?')} | Sources: {aw.get('num_sources')} | History: {aw.get('history_count')} adj")
        for c in aw.get("top_changes", [])[:5]:
            arrow = "⬆" if c["change"] > 0 else "⬇"
            print(f"       {arrow} {c['source']:<25} {c['base_weight']:.4f} → {c['adjusted_weight']:.4f} (×{c['multiplier']})")
    else:
        print(f"  🔄 ADAPTIVE WEIGHTS: not available (run attribution first)")

    print()

    # ── Cron ──
    cron = dashboard.get("cron", {})
    if cron.get("available"):
        badge = "✅" if cron.get("errors", 0) == 0 else "⚠️"
        print(f"  {badge} CRON JOBS: {cron.get('ok')}/{cron.get('total')} ok, {cron.get('errors')} errors")
        for job in cron.get("jobs", []):
            dur = job.get("duration_seconds", 0)
            status = job.get("status", "?")
            badge = "✅" if status == "ok" else "⏳" if status == "pending" else "❌"
            print(f"       {badge} {job.get('name', '?'):<30} {status:<8} {_fmt(dur, 's')}")
    else:
        print("  ❌ CRON: not available")

    print(f"{'=' * 60}")


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
        with open(out_path, "w") as f:
            # Use default=str for numpy/non-serializable types
            json.dump(dashboard, f, indent=2, default=str)
        print(f"Saved unified dashboard to {out_path}")

    if "--status-text" in sys.argv:
        print(generate_status_text())
        return

    if "--check" in sys.argv:
        health_ok = dashboard.get("health", {}).get("status") == "healthy"
        cron_ok = dashboard.get("cron", {}).get("errors", 99) == 0
        sys.exit(0 if (health_ok and cron_ok) else 1)

    # Default: print summary
    print_summary(dashboard)


if __name__ == "__main__":
    main()
