#!/usr/bin/env python3
"""
Portfolio-Lab v8.05: Implementation Risk Quantification

Formalizes the gap between backtest expectations and live/paper trading results.
Based on Yin, Miki & Lesnichenko (arXiv:2603.20319, Mar 2026).

Core concepts:
1. **Implementation Gap**: The difference between backtest-predicted metrics and actual results
2. **Confidence Bounds**: Bootstrap-derived confidence intervals for backtest metrics
3. **Grading**: A-F grade for implementation quality (gap width vs expected variance)
4. **Alerting**: Trigger warnings when gap exceeds 95th percentile of expected distribution

Usage:
    python -m src.monitor.implementation_risk status     # Current gap overview
    python -m src.monitor.implementation_risk report     # Detailed report
    python -m src.monitor.implementation_risk check      # Alert check only
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.paths import DATA_DIR, BACKTEST_RESULTS_DIR

# Paths
PERFORMANCE_LOG = DATA_DIR / "performance.jsonl"
PORTFOLIO_STATE = DATA_DIR / "portfolio_paper.json"
STATE_PATH = DATA_DIR / "implementation_risk_state.json"

def _get_default_backtest_files() -> List[Path]:
    """Get default backtest result files (computed at call time, not import time)."""
    return [
        DATA_DIR / "combined_backtest_results.json",
        DATA_DIR / "cta_backtest_results.json",
        DATA_DIR / "smart_rebalance_backtest_results.json",
        DATA_DIR / "factor_timing_backtest_results.json",
        DATA_DIR / "ubt_backtest_results.json",
    ]

# Metric definitions with realistic ranges
METRICS_CONFIG = {
    "sharpe": {
        "label": "Sharpe Ratio",
        "benchmark_value": 0.93,
        "realistic_max": 3.0,
        "unit": "",
        "higher_is_better": True,
    },
    "cagr": {
        "label": "CAGR",
        "benchmark_value": 0.107,
        "realistic_max": 0.50,
        "unit": "%",
        "higher_is_better": True,
    },
    "max_drawdown": {
        "label": "Max Drawdown",
        "benchmark_value": 0.257,
        "realistic_max": 0.50,
        "unit": "%",
        "higher_is_better": False,
    },
    "volatility": {
        "label": "Annualized Volatility",
        "benchmark_value": 0.115,
        "realistic_max": 0.50,
        "unit": "%",
        "higher_is_better": False,
    },
}


# Grading thresholds (A=excellent, F=failing)
GRADE_THRESHOLDS = [
    (0.50, "A", "Excellent — implementation closely tracks backtest"),
    (1.00, "B", "Good — minor implementation gap within normal range"),
    (1.50, "C", "Fair — notable gap, investigate root causes"),
    (2.00, "D", "Poor — significant gap, action recommended"),
    (float("inf"), "F", "Failing — implementation diverging from backtest"),
]


@dataclass
class MetricGap:
    """Gap analysis for a single metric."""

    name: str
    backtest_value: float
    actual_value: float
    gap: float  # absolute difference
    gap_pct: float  # relative difference
    bootstrap_std: float  # expected std from bootstrap
    z_score: float  # gap / bootstrap_std
    confidence_95_lower: float
    confidence_95_upper: float
    grade: str
    grade_detail: str
    within_expected: bool


@dataclass
class ImplementationRiskReport:
    """Complete implementation risk assessment."""

    timestamp: str
    trading_days: int
    report_type: str  # "paper" or "live"
    metrics: Dict[str, MetricGap]
    composite_grade: str
    composite_gap_pct: float
    alerts: List[str]
    recommendations: List[str]
    backtest_source: str
    data_quality: Dict[str, Any]


def load_backtest_results(
    files: Optional[List[Path]] = None,
) -> Optional[Dict[str, float]]:
    """Load backtest results from first available file.

    Searches DEFAULT_BACKTEST_FILES in order, returns first match.
    """
    search_files = files or _get_default_backtest_files()
    for path in search_files:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                logger.info(f"Loaded backtest results from {path.name}")
                return _normalize_backtest(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse {path}: {e}")
                continue
    logger.warning("No backtest results found — using default benchmarks")
    return None


def _normalize_backtest(data: dict) -> Dict[str, float]:
    """Normalize backtest data to standard metric names."""
    mapping = {
        "sharpe_ratio": "sharpe",
        "sharpe": "sharpe",
        "cagr": "cagr",
        "max_drawdown": "max_drawdown",
        "max_dd": "max_drawdown",
        "volatility": "volatility",
        "ann_vol": "volatility",
    }
    result = {}
    for key, standard in mapping.items():
        if key in data and data[key] is not None:
            result[standard] = float(data[key])
    return result


def load_paper_trading_data(
    days: int = 30,
) -> Dict[str, float]:
    """Load paper trading data from performance.jsonl and portfolio state.

    Returns tracked metrics computed from actual returns.
    """
    returns = _extract_returns(days)
    _load_portfolio_state()

    if len(returns) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "volatility": 0.0}

    len(returns)
    float(np.mean(returns))
    std_ret = float(max(np.std(returns, ddof=1), 1e-6))

    # Annualize (assuming daily returns)
    trading_days_actual = 252
    ann_return = float(np.mean(returns) * trading_days_actual)
    ann_vol = float(std_ret * np.sqrt(trading_days_actual))

    # Sharpe with realistic cap
    sharpe = min(ann_return / ann_vol, METRICS_CONFIG["sharpe"]["realistic_max"])

    # Max drawdown from cumulative returns
    cum_returns = np.cumprod(1 + np.array(returns, dtype=np.float64))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = float(abs(min(drawdowns))) if len(drawdowns) > 0 else 0.0

    # Annualized volatility
    vol = ann_vol

    return {
        "sharpe": max(-3.0, min(sharpe, 3.0)),
        "cagr": ann_return,
        "max_drawdown": max_dd,
        "volatility": vol,
    }


def _extract_returns(days: int = 30) -> np.ndarray:
    """Extract daily returns from performance log, deduplicating to one per day."""
    if not PERFORMANCE_LOG.exists():
        logger.warning(f"Performance log not found: {PERFORMANCE_LOG}")
        return np.array([])

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    daily_returns: Dict[str, float] = {}
    with open(PERFORMANCE_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts_str = entry.get("timestamp", "")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str)
                # Handle timezone-naive timestamps
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue

                daily_return = entry.get("daily_return")
                if daily_return is None:
                    continue

                # Deduplicate to one entry per trading day (take last entry of the day)
                day_key = ts.strftime("%Y-%m-%d")
                daily_returns[day_key] = float(daily_return)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    vals = np.array(list(daily_returns.values()), dtype=np.float64)
    logger.info(f"Extracted {len(vals)} daily returns from {days}-day window")
    return vals


def _load_portfolio_state() -> dict:
    """Load current portfolio state for context."""
    if PORTFOLIO_STATE.exists():
        try:
            with open(PORTFOLIO_STATE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _count_trading_days(days: int = 30) -> int:
    """Count actual trading days with data in the window."""
    returns = _extract_returns(days)
    return len(returns)


def bootstrap_confidence(
    backtest_value: float,
    n_simulations: int = 5000,
    n_years: float = 5.0,
) -> Tuple[float, float, float]:
    """Bootstrap confidence intervals for a backtest metric.

    Uses a simple Monte Carlo approach: simulates return sequences with
    realistic noise around the backtest value and measures the spread
    of outcomes.

    Args:
        backtest_value: The backtest metric value
        n_simulations: Number of bootstrap samples
        n_years: Assumed backtest length in years

    Returns:
        (lower_95, upper_95, std): 95% CI bounds and standard deviation
    """
    # Annual observations
    n_obs = max(2, int(n_years * 252))

    # For Sharpe-like metrics, simulate with reasonable return variance
    # Use a conservative estimate of annual return vol ~15%
    if abs(backtest_value) < 10:  # Sharpe-like metric
        # Simulate Sharpe values with noise proportional to 1/sqrt(N)
        sharpe_se = 1.0 / np.sqrt(n_obs)
        samples = np.random.normal(backtest_value, sharpe_se, n_simulations)
    elif abs(backtest_value) < 1.0:  # CAGR or return-like
        # CAGR noise depends on volatility and sample length
        vol = 0.15  # typical annual vol
        cagr_se = vol / np.sqrt(n_obs)
        samples = np.random.normal(backtest_value, cagr_se, n_simulations)
    else:  # Drawdown or other
        # DD noise with moderate uncertainty
        dd_se = 0.05
        samples = np.random.normal(backtest_value, dd_se, n_simulations)

    lower = float(np.percentile(samples, 2.5))
    upper = float(np.percentile(samples, 97.5))
    std = float(np.std(samples, ddof=1))
    return lower, upper, std


def compute_gap(
    backtest_value: float,
    actual_value: float,
    bootstrap_std: float,
) -> Tuple[float, float, float, str, str, bool]:
    """Compute implementation gap between backtest and actual.

    Returns:
        (gap, gap_pct, z_score, grade, grade_detail, within_expected)
    """
    gap = actual_value - backtest_value
    if abs(backtest_value) > 1e-10:
        gap_pct = abs(gap) / abs(backtest_value)
    else:
        gap_pct = abs(gap)

    z_score = gap / max(bootstrap_std, 1e-10) if bootstrap_std > 0 else 0.0

    # Grade based on gap relative to expected std
    relative_gap = abs(gap) / max(bootstrap_std, 1e-10) if bootstrap_std > 0 else 0.0

    grade = "N/A"
    grade_detail = "Insufficient data"
    within_expected = False

    for threshold, g, detail in GRADE_THRESHOLDS:
        if relative_gap <= threshold:
            grade = g
            grade_detail = detail
            break

    within_expected = grade in ("A", "B")
    return gap, gap_pct, z_score, grade, grade_detail, within_expected


def assess_implementation_risk(
    backtest_source: str = "auto",
    paper_days: int = 30,
) -> ImplementationRiskReport:
    """Run full implementation risk assessment.

    Args:
        backtest_source: Path to backtest results file, or "auto" for default
        paper_days: Number of recent days to include from paper trading

    Returns:
        ImplementationRiskReport with all findings
    """
    # 1. Load backtest
    if backtest_source and backtest_source != "auto":
        bt_file = Path(backtest_source)
        bt_data = load_backtest_results([bt_file])
    else:
        bt_data = load_backtest_results()

    # 2. Load paper trading
    paper_data = load_paper_trading_data(days=paper_days)
    trading_days = _count_trading_days(days=paper_days)

    # 3. Compute gaps per metric
    metrics = {}
    alerts = []
    recommendations = []
    composite_gaps = []
    grade_scores = []

    for name, config in METRICS_CONFIG.items():
        raw_bt_val = bt_data.get(name) if bt_data else None
        bt_val: float = raw_bt_val if isinstance(raw_bt_val, (int, float)) else config["benchmark_value"]
        actual_val = paper_data.get(name, 0.0)

        lower_ci, upper_ci, std = bootstrap_confidence(bt_val)
        gap, gap_pct, z_score, grade, grade_detail, within = compute_gap(
            bt_val, actual_val, std
        )

        metrics[name] = MetricGap(
            name=name,
            backtest_value=bt_val,
            actual_value=actual_val,
            gap=gap,
            gap_pct=gap_pct,
            bootstrap_std=std,
            z_score=z_score,
            confidence_95_lower=lower_ci,
            confidence_95_upper=upper_ci,
            grade=grade,
            grade_detail=grade_detail,
            within_expected=within,
        )

        if not within:
            alerts.append(
                f"{config['label']}: actual {actual_val:.4f} vs backtest {bt_val:.4f} "
                f"(gap {gap:+.4f}, z={z_score:.1f}) — outside expected bounds"
            )

        composite_gaps.append(gap_pct)

        # Map grade to numeric score
        grade_score = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}.get(grade, 0.0)
        grade_scores.append(grade_score)

    # 4. Composite grade
    avg_gap_pct = float(np.mean(composite_gaps)) if composite_gaps else 0.0
    avg_grade_score = float(np.mean(grade_scores)) if grade_scores else 0.0
    composite_grade = _score_to_grade(avg_grade_score) if avg_grade_score > 0 else "N/A"

    # 5. Recommendations
    if trading_days < 5:
        recommendations.append(
            f"Insufficient data: only {trading_days} trading days. "
            "Need at least 21 days for meaningful analysis."
        )
    if trading_days < 21:
        recommendations.append(
            "Continue accumulating paper trading data. Reassess after 63+ trading days."
        )
    if any(v.z_score > 2.0 for v in metrics.values()):
        recommendations.append(
            "Implementation gap exceeds 2-sigma on one or more metrics. "
            "Investigate root causes: rebalance timing, data freshness, cost assumptions."
        )
    if avg_gap_pct > 0.5:
        recommendations.append(
            "Large implementation gap detected. Review rebalance execution, "
            "slippage assumptions, and data pipeline integrity."
        )
    if composite_grade in ("D", "F"):
        recommendations.append(
            "Critical: implementation risk high. Consider pausing strategy "
            "changes until gap is resolved."
        )
    if composite_grade in ("A", "B") and trading_days >= 21:
        recommendations.append(
            "Implementation quality is good. Continue monitoring — focus on "
            "strategy improvements rather than infrastructure fixes."
        )
    recommendations.append(
        "Next review: after 63 trading days (quarterly check). "
        "Check: rebalance timing, transaction costs, data pipeline."
    )

    # 6. Data quality context
    data_quality = {
        "trading_days_loaded": trading_days,
        "performance_log_lines": _count_log_lines(),
        "backtest_found": bt_data is not None,
        "backtest_source": bt_data is not None,
    }

    # 7. Persist state
    _save_state(
        metrics=metrics,
        composite_grade=composite_grade,
        composite_gap_pct=avg_gap_pct,
        trading_days=trading_days,
        alerts=alerts,
        backtest_source=str(bt_data) if bt_data else "defaults",
    )

    return ImplementationRiskReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        trading_days=trading_days,
        report_type="paper",
        metrics=metrics,
        composite_grade=composite_grade,
        composite_gap_pct=avg_gap_pct,
        alerts=alerts,
        recommendations=recommendations,
        backtest_source=str(
            list(BACKTEST_RESULTS_DIR.iterdir()) if BACKTEST_RESULTS_DIR.exists() else "defaults"
        ),
        data_quality=data_quality,
    )


def _score_to_grade(avg_score: float) -> str:
    """Convert average grade score to letter grade."""
    if avg_score >= 3.5:
        return "A"
    elif avg_score >= 2.5:
        return "B"
    elif avg_score >= 1.5:
        return "C"
    elif avg_score >= 0.5:
        return "D"
    return "F"


def _count_log_lines() -> int:
    """Count lines in performance log."""
    if PERFORMANCE_LOG.exists():
        try:
            with open(PERFORMANCE_LOG) as f:
                return sum(1 for _ in f)
        except OSError:
            pass
    return 0


def _save_state(
    metrics: Dict[str, MetricGap],
    composite_grade: str,
    composite_gap_pct: float,
    trading_days: int,
    alerts: List[str],
    backtest_source: str,
) -> None:
    """Persist implementation risk state to JSON."""
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trading_days": trading_days,
        "composite_grade": composite_grade,
        "composite_gap_pct": round(composite_gap_pct, 4),
        "metrics": {
            name: {
                "backtest_value": round(m.backtest_value, 4),
                "actual_value": round(m.actual_value, 4),
                "gap": round(m.gap, 4),
                "gap_pct": round(m.gap_pct, 4),
                "z_score": round(m.z_score, 2),
                "grade": m.grade,
                "within_expected": m.within_expected,
            }
            for name, m in metrics.items()
        },
        "alerts": alerts,
        "backtest_source": backtest_source,
    }
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved implementation risk state to {STATE_PATH}")
    except OSError as e:
        logger.error(f"Failed to save state: {e}")


def _format_pct(val: float) -> str:
    """Format as percentage string."""
    return f"{val * 100:.2f}%"


def _format_val(val: float, name: str) -> str:
    """Format a metric value for display."""
    if name in ("cagr", "max_drawdown", "volatility"):
        return _format_pct(val)
    return f"{val:.4f}"


def cmd_status() -> None:
    """Print current implementation risk overview."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        print("=" * 60)
        print("  IMPLEMENTATION RISK — Current Status")
        print("=" * 60)
        print(f"  Evaluated:        {state.get('timestamp', 'N/A')}")
        print(f"  Trading days:     {state.get('trading_days', 0)}")
        print(f"  Composite grade:  {state.get('composite_grade', 'N/A')}")
        print(f"  Composite gap:    {_format_pct(state.get('composite_gap_pct', 0))}")
        print(f"  Alerts:           {len(state.get('alerts', []))}")
        print()
        print("  Metrics:")
        for name, m in state.get("metrics", {}).items():
            label = METRICS_CONFIG.get(name, {}).get("label", name)
            flag = "⚠️" if not m.get("within_expected", True) else "✅"
            print(
                f"    {flag} {label:20s}: "
                f"backtest={_format_val(m['backtest_value'], name):>10s} "
                f"actual={_format_val(m['actual_value'], name):>10s} "
                f"z={m['z_score']:+.1f}  [{m['grade']}]"
            )
        if state.get("alerts"):
            print()
            print("  Alerts:")
            for a in state["alerts"]:
                print(f"    ⚠️ {a}")
    else:
        print("No implementation risk assessment available yet.")
        print("Run: python -m src.monitor.implementation_risk report")


def cmd_report() -> None:
    """Generate and display a full implementation risk report."""
    report = assess_implementation_risk()

    print("=" * 70)
    print("  IMPLEMENTATION RISK REPORT")
    print("=" * 70)
    print(f"  Timestamp:        {report.timestamp}")
    print(f"  Type:             {report.report_type}")
    print(f"  Trading days:     {report.trading_days}")
    print(f"  Composite grade:  {report.composite_grade}")
    print(f"  Composite gap:    {_format_pct(report.composite_gap_pct)}")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │ METRIC GAPS                                                 │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    for name, m in report.metrics.items():
        config = METRICS_CONFIG.get(name, {})
        label = config.get("label", name)
        flag = "⚠️" if not m.within_expected else "✅"
        print(f"  │ {flag} {label:20s}                                      │")
        print(f"  │    Backtest:   {_format_val(m.backtest_value, name):>10s}                                   │")
        print(f"  │    Actual:     {_format_val(m.actual_value, name):>10s}                                   │")
        print(f"  │    Gap:        {m.gap:+.4f} ({_format_pct(m.gap_pct)})                         │")
        print(f"  │    95% CI:     [{_format_val(m.confidence_95_lower, name)}, {_format_val(m.confidence_95_upper, name)}]                    │")
        print(f"  │    Z-score:    {m.z_score:+.2f}  │ Grade: {m.grade}                               │")
        print(f"  │    Status:     {m.grade_detail[:45]:45s}│")
        print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print()

    if report.alerts:
        print("  ALERTS:")
        for a in report.alerts:
            print(f"    ⚠️  {a}")
        print()

    print("  RECOMMENDATIONS:")
    for r in report.recommendations:
        print(f"    • {r}")
    print()

    print("  DATA QUALITY:")
    for k, v in report.data_quality.items():
        print(f"    {k}: {v}")
    print("=" * 70)


def cmd_check() -> None:
    """Quick alert check — exit code 1 if critical."""
    report = assess_implementation_risk()

    critical_alerts = []
    for name, m in report.metrics.items():
        if not m.within_expected and abs(m.z_score) > 2.0:
            critical_alerts.append(name)

    if critical_alerts:
        print(f"CRITICAL: {len(critical_alerts)} metrics outside 2-sigma bounds:")
        for name in critical_alerts:
            label = METRICS_CONFIG.get(name, {}).get("label", name)
            m = report.metrics[name]
            print(f"  ⚠️ {label}: z={m.z_score:+.2f}")
        print(f"Overall grade: {report.composite_grade}")
        sys.exit(1)
    elif report.composite_grade in ("D", "F"):
        print(f"WARNING: Composite grade {report.composite_grade}")
        print("Investigate implementation gaps before making strategy changes.")
        sys.exit(0)
    else:
        print(f"OK: Composite grade {report.composite_grade}")
        print("Implementation risk within acceptable bounds.")
        sys.exit(0)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "status":
        cmd_status()
    elif command == "report":
        cmd_report()
    elif command == "check":
        cmd_check()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.monitor.implementation_risk {status|report|check}")
        sys.exit(1)


if __name__ == "__main__":
    main()
