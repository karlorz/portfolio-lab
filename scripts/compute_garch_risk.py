#!/usr/bin/env python3
"""Compute GARCH-CVaR risk metrics and write .health_report.json.

Standalone script for cron-based GARCH-CVaR pipeline activation.
Reads portfolio returns from market.db, computes GARCH-filtered CVaR,
and writes results to data/.health_report.json for dashboard consumption.

Usage:
    python scripts/compute_garch_risk.py
    python scripts/compute_garch_risk.py --window 504
"""
import json
import sys
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MARKET_DB, DATA_DIR, PUBLIC_DATA_DIR, BASE_ALLOCATION
from src.monitor.garch_cvar import calculate_garch_cvar, ARCH_AVAILABLE


def compute_portfolio_returns(db_path: Path, days: int = 504) -> np.ndarray:
    """Load SPY/GLD/TLT daily returns from market.db."""
    if not db_path.exists():
        return np.array([])

    weights = BASE_ALLOCATION
    conn = sqlite3.connect(str(db_path))

    # Get common trading dates
    symbols = list(weights.keys())
    all_dates = set()
    symbol_prices = {}

    for sym in symbols:
        cursor = conn.execute(
            "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            (sym, days),
        )
        rows = cursor.fetchall()
        if rows:
            symbol_prices[sym] = {r[0]: r[1] for r in rows}
            if not all_dates:
                all_dates = set(symbol_prices[sym].keys())
            else:
                all_dates &= set(symbol_prices[sym].keys())

    conn.close()

    if not all_dates or not symbol_prices:
        return np.array([])

    sorted_dates = sorted(all_dates)

    # Compute weighted portfolio returns
    portfolio_returns = []
    prev_prices = None
    for date in sorted_dates:
        current_prices = {}
        for sym in symbols:
            if sym in symbol_prices and date in symbol_prices[sym]:
                current_prices[sym] = symbol_prices[sym][date]

        if len(current_prices) != len(symbols) or prev_prices is None:
            prev_prices = current_prices
            continue

        daily_return = 0.0
        for sym in symbols:
            if sym in current_prices and sym in prev_prices and prev_prices[sym] > 0:
                daily_return += weights[sym] * (current_prices[sym] / prev_prices[sym] - 1)

        portfolio_returns.append(daily_return)
        prev_prices = current_prices

    return np.array(portfolio_returns)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute GARCH-CVaR risk metrics")
    parser.add_argument("--window", type=int, default=252, help="GARCH lookback window")
    parser.add_argument("--days", type=int, default=504, help="Days of price data to load")
    args = parser.parse_args()

    print(f"GARCH-CVaR Risk Computation — {datetime.now().isoformat()}")
    print(f"  arch library: {'available' if ARCH_AVAILABLE else 'NOT AVAILABLE (will use historical fallback)'}")

    # Load returns
    returns = compute_portfolio_returns(MARKET_DB, days=args.days)
    print(f"  Loaded {len(returns)} portfolio daily returns")

    if len(returns) < 63:
        print("  ERROR: Insufficient data (need at least 63 days)")
        sys.exit(1)

    # Policy drawdown limit (PAPER_CONFIG default) — never publish as measured DD.
    policy_max_dd = -0.15
    try:
        from src.strategy.evaluator import PAPER_CONFIG

        policy_max_dd = -abs(float(PAPER_CONFIG.get("max_drawdown_pct", 0.15)))
    except Exception:  # noqa: BLE001 — keep hard default
        policy_max_dd = -0.15

    # Measured peak-to-trough from the same return series fed to GARCH
    measured_max_dd = None
    measured_current_dd = None
    if len(returns) >= 2:
        nav = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            nav *= 1.0 + float(r)
            peak = max(peak, nav)
            if peak > 0:
                max_dd = min(max_dd, (nav - peak) / peak)
        measured_max_dd = float(max_dd)
        measured_current_dd = float((nav - peak) / peak) if peak > 0 else 0.0

    # Compute GARCH-CVaR (policy limit is a risk-engine input, not measured NAV DD)
    metrics = calculate_garch_cvar(
        returns=returns,
        current_drawdown=measured_current_dd if measured_current_dd is not None else 0.0,
        max_drawdown=policy_max_dd,
        window=args.window,
    )

    # Write health report
    report_path = DATA_DIR / ".health_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    import math
    report = asdict(metrics)

    # Rename policy input out of the measured-metric slot (Batch AC / Z parity)
    policy_limit_pct = round(abs(policy_max_dd) * 100, 2)
    if "max_drawdown" in report:
        report["max_drawdown_limit"] = report.pop("max_drawdown")
        report["max_drawdown_limit_pct"] = policy_limit_pct
    else:
        report["max_drawdown_limit"] = round(policy_max_dd * 100, 2)
        report["max_drawdown_limit_pct"] = policy_limit_pct
    report["measured_max_drawdown"] = (
        round(measured_max_dd * 100, 2) if measured_max_dd is not None else None
    )
    report["measured_max_drawdown_pct"] = (
        round(abs(measured_max_dd) * 100, 2) if measured_max_dd is not None else None
    )
    report["measured_current_drawdown"] = (
        round(measured_current_dd * 100, 2) if measured_current_dd is not None else None
    )
    report["measured_current_drawdown_pct"] = (
        round(abs(measured_current_dd) * 100, 2)
        if measured_current_dd is not None
        else None
    )
    if measured_current_dd is not None:
        report["current_drawdown"] = round(measured_current_dd * 100, 2)
    report["drawdown_field_semantics"] = (
        "max_drawdown_limit=policy input to GARCH-CVaR; "
        "measured_max_drawdown=NAV peak-to-trough on portfolio returns series"
    )

    # Add portfolio entropy metrics (include max_possible for H_max honesty)
    weights = list(BASE_ALLOCATION.values())
    n = len(weights)
    shannon = -sum(w * math.log(w) for w in weights if w > 0)
    effective_n = math.exp(shannon)
    h_max = math.log(n) if n > 1 else 1.0
    normalized_score = (shannon / h_max) * 100.0 if h_max > 0 else 0.0
    hhi = sum(w * w for w in weights)

    report["checks"] = {
        "portfolio_entropy": {
            "name": "portfolio_entropy",
            "status": "good" if normalized_score > 90 else "warning",
            "ok": normalized_score > 70,
            "metrics": {
                "shannon_entropy": round(shannon, 4),
                "effective_n": round(effective_n, 2),
                "max_possible": round(h_max, 4) if n > 1 else None,
                "normalized_score": round(normalized_score, 1),
                "hhi_index": round(hhi, 4),
            },
        },
    }

    # Derive top-level status for unified dashboard compatibility
    tail = report.get("tail_severity", "normal")
    cvar_ratio = report.get("cvar_ratio", 1.0)
    if tail in ("extreme", "severe") or cvar_ratio > 3.0:
        report["status"] = "unhealthy"
    else:
        report["status"] = "healthy"
    report["summary"] = {"passed": 1, "total_checks": 1}

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Dual-write risk_metrics.json for unified / cvar_metrics consumers
    # (field names match historical risk_metrics schema).
    risk_metrics_path = DATA_DIR / "risk_metrics.json"
    garch_active = bool(report.get("filter_active", False))
    risk_payload = {
        "timestamp": report.get("timestamp") or datetime.now().isoformat(),
        "var_95_daily": report.get("var_95"),
        "cvar_95_daily": report.get("cvar_95"),
        "cvar_ratio": report.get("cvar_ratio"),
        "tail_severity": report.get("tail_severity"),
        # Prefer measured NAV DD; keep limit as separate field
        "max_drawdown": report.get("measured_max_drawdown"),
        "max_drawdown_limit": report.get("max_drawdown_limit"),
        "max_drawdown_limit_pct": report.get("max_drawdown_limit_pct"),
        "measured_max_drawdown": report.get("measured_max_drawdown"),
        "measured_max_drawdown_pct": report.get("measured_max_drawdown_pct"),
        "measured_current_drawdown": report.get("measured_current_drawdown"),
        "current_drawdown": report.get("current_drawdown"),
        "drawdown_field_semantics": report.get("drawdown_field_semantics"),
        "volatility_annual": report.get("volatility_annual"),
        "garch_filtered": bool(report.get("garch_filtered", report.get("filter_active", False))),
        "garch_active": garch_active,
        "garch_params": {
            "omega": report.get("garch_omega"),
            "alpha": report.get("garch_alpha"),
            "beta": report.get("garch_beta"),
            "persistence": report.get("garch_persistence"),
        }
        if report.get("filter_active")
        else None,
        "conditional_volatility_current": report.get("conditional_volatility_current"),
        "source": "compute_garch_risk",
    }

    # Conformal coverage cross-check (same honesty contract as dashboard load):
    # coverage_pass=false → demote garch_active (still publish diagnostics).
    coverage_diagnostics = None
    try:
        from src.monitor.conformal_risk import (
            conformal_coverage_diagnostics,
            conformal_var,
        )

        if len(returns) >= 22:
            cvar_thresh = float(conformal_var(returns, alpha=0.05))
            var_thresholds = np.full_like(returns, cvar_thresh, dtype=float)
            coverage_diagnostics = conformal_coverage_diagnostics(
                returns,
                var_thresholds,
                alpha=0.05,
                rolling_window=252,
            )
            risk_payload["coverage_diagnostics"] = coverage_diagnostics
            if (
                isinstance(coverage_diagnostics, dict)
                and coverage_diagnostics.get("coverage_pass") is False
                and garch_active
            ):
                risk_payload["garch_active"] = False
                risk_payload["runtime_role"] = "advisory_degraded"
                risk_payload["garch_active_reason"] = (
                    "coverage_pass=false (Kupiec/coverage diagnostics failed); "
                    "GARCH remains advisory only"
                )
                print("  GARCH demoted: coverage_pass=false → advisory_degraded")
    except Exception as exc:  # noqa: BLE001 — never block dual-write on conformal
        print(f"  WARNING: conformal coverage check skipped: {exc}")

    with open(risk_metrics_path, "w") as f:
        json.dump(risk_payload, f, indent=2, default=str)

    # Public dual-write: non-dotfile GARCH-CVaR for WWW / index consumers
    try:
        public_root = Path(PUBLIC_DATA_DIR)
        public_root.mkdir(parents=True, exist_ok=True)
        public_payload = {
            **risk_payload,
            "schema_version": "garch-cvar/v1",
            "private_health_report": str(report_path),
            "drawdown_field_semantics": report.get(
                "drawdown_field_semantics",
                "max_drawdown_limit=policy; measured_max_drawdown=NAV series",
            ),
        }
        public_path = public_root / "garch_cvar.json"
        # Atomic write: temp + rename (same FS) so readers never see partial JSON
        tmp_path = public_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(public_payload, f, indent=2, default=str)
        tmp_path.replace(public_path)
        print(f"  Public GARCH: {public_path}")
    except OSError as exc:
        print(f"  WARNING: public garch_cvar dual-write failed: {exc}")

    # Append sparse history so operators can see GARCH job cadence (keep last 720).
    history_path = DATA_DIR / "risk_metrics_history.json"
    try:
        history = []
        if history_path.exists():
            try:
                loaded = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    history = loaded
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(
            {
                "timestamp": risk_payload["timestamp"],
                "var_95": risk_payload["var_95_daily"],
                "cvar_95": risk_payload["cvar_95_daily"],
                "cvar_ratio": risk_payload["cvar_ratio"],
                "tail_severity": risk_payload["tail_severity"],
                "garch_active": risk_payload["garch_active"],
                "source": "compute_garch_risk",
            }
        )
        history = history[-720:]
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except OSError as exc:
        print(f"  WARNING: failed to append risk_metrics_history: {exc}")

    print(f"  VaR 95%:      {metrics.var_95:.2f}%")
    print(f"  CVaR 95%:     {metrics.cvar_95:.2f}%")
    print(f"  CVaR Ratio:   {metrics.cvar_ratio:.2f}")
    print(f"  Tail:         {metrics.tail_severity}")
    print(f"  GARCH Active: {metrics.filter_active}")
    if metrics.filter_active:
        print(f"  Persistence:  {metrics.garch_persistence:.4f}")
        print(f"  Cond Vol:     {metrics.conditional_volatility_current:.2f}%")
    print(f"  Report saved: {report_path}")
    print(f"  Risk metrics: {risk_metrics_path}")


if __name__ == "__main__":
    main()
