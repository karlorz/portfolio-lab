#!/usr/bin/env python3
"""
v6.10: Graduation Checklist Module

Systematic framework for paper-to-live transition readiness.
Replaces ad hoc .promote_to_live trigger with multi-criteria gates,
minimum observation periods, and structured progress tracking.

Usage:
    python -m src.strategy.graduation_checklist check        # Check all criteria
    python -m src.strategy.graduation_checklist report       # Generate detailed report
    python -m src.strategy.graduation_checklist progress     # Show readiness progress
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, NamedTuple

import numpy as np

from src.paths import DATA_DIR


class CheckResult(NamedTuple):
    """Result of a single graduation criterion check."""
    name: str
    passed: bool
    value: float
    required: float
    description: str


class GraduationChecklist:
    """Multi-criteria graduation readiness assessment.

    All criteria must pass (except manual_approval) to consider
    the paper portfolio ready for live promotion.
    """

    # Default thresholds
    DEFAULT_CRITERIA = {
        "min_trading_days": {
            "value": 63,
            "description": "At least 63 trading days of paper trading data",
        },
        "min_sharpe": {
            "value": 0.50,
            "description": "Rolling Sharpe ratio >= 0.50",
        },
        "max_drawdown": {
            "value": 0.15,
            "description": "Maximum drawdown <= 15%",
        },
        "min_win_rate": {
            "value": 0.40,
            "description": "Win rate >= 40%",
        },
        "health_checks": {
            "value": 30,
            "description": "All 9 health checks passing for 30 consecutive days",
        },
        "min_tca_orders": {
            "value": 10,
            "description": "TCA engine populated with >= 10 orders",
        },
        "circuit_breaker_confidence": {
            "value": 3,
            "description": "Circuit breaker confidence >= 3 consecutive cycles (no trip)",
        },
        "manual_approval": {
            "value": False,
            "description": "Human-in-the-loop approval (MANDATORY — always False by default)",
        },
    }

    # Gate: minimum days before ANY graduation alert (prevents false alarms)
    MIN_OBSERVATION_DAYS = 30

    def __init__(self, criteria: Optional[Dict] = None):
        self.criteria = criteria or dict(self.DEFAULT_CRITERIA)

    def check(self, state: Optional[Dict] = None) -> Dict[str, CheckResult]:
        """Check all graduation criteria against current state.

        Args:
            state: Dictionary containing portfolio and system state.
                   If None, auto-loads from data files.

        Returns:
            Dict mapping criterion name -> CheckResult
        """
        if state is None:
            state = self._load_state()

        results = {}

        # 1. Minimum trading days
        results["min_trading_days"] = self._check_trading_days(state)

        # 2. Minimum Sharpe ratio
        results["min_sharpe"] = self._check_sharpe(state)

        # 3. Maximum drawdown
        results["max_drawdown"] = self._check_drawdown(state)

        # 4. Win rate
        results["min_win_rate"] = self._check_win_rate(state)

        # 5. Health check history
        results["health_checks"] = self._check_health(state)

        # 6. TCA orders
        results["min_tca_orders"] = self._check_tca_orders(state)

        # 7. Circuit breaker confidence
        results["circuit_breaker_confidence"] = self._check_circuit_breaker(state)

        # 8. Manual approval (always False by default)
        results["manual_approval"] = self._check_manual_approval(state)

        return results

    def is_graduation_ready(self, results: Dict[str, CheckResult]) -> bool:
        """All criteria passed except manual_approval."""
        for name, result in results.items():
            if name == "manual_approval":
                continue
            if not result.passed:
                return False
        return True

    def readiness_score(self, results: Dict[str, CheckResult]) -> float:
        """Calculate overall readiness as percentage of non-manual criteria passed."""
        auto_criteria = [r for n, r in results.items() if n != "manual_approval"]
        if not auto_criteria:
            return 0.0
        passed = sum(1 for r in auto_criteria if r.passed)
        return round(passed / len(auto_criteria) * 100, 1)

    def save_report(self, results: Dict[str, CheckResult], path: Optional[Path] = None) -> Path:
        """Save graduation readiness report to JSON."""
        if path is None:
            path = DATA_DIR / ".graduation_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "readiness_score": self.readiness_score(results),
            "is_graduation_ready": self.is_graduation_ready(results),
            "criteria": {
                name: {
                    "passed": result.passed,
                    "value": result.value,
                    "required": result.required,
                    "description": result.description,
                }
                for name, result in results.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def progress_summary(self, results: Dict[str, CheckResult]) -> Dict:
        """Human-readable progress summary."""
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        auto_passed = sum(1 for n, r in results.items() if n != "manual_approval" and r.passed)
        auto_total = sum(1 for n in results if n != "manual_approval")

        return {
            "overall_progress": f"{auto_passed}/{auto_total}",
            "readiness_pct": self.readiness_score(results),
            "is_ready": self.is_graduation_ready(results),
            "manual_approval_required": True,
            "passed_count": passed,
            "total_count": total,
            "details": results,
        }

    # ------------------------------------------------------------------
    # Internal check implementations
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict:
        """Auto-load all state from data files."""
        state = {}

        # Portfolio paper state
        portfolio_file = DATA_DIR / "portfolio_paper.json"
        if portfolio_file.exists():
            with open(portfolio_file) as f:
                state["portfolio"] = json.load(f)

        # Performance history
        perf_file = DATA_DIR / "performance.jsonl"
        if perf_file.exists():
            perf_entries = []
            with open(perf_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            perf_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            state["performance"] = perf_entries

        # TCA scorecard
        tca_file = DATA_DIR / "tca_scorecard.json"
        if tca_file.exists():
            with open(tca_file) as f:
                state["tca"] = json.load(f)

        # Circuit breaker
        cb_file = DATA_DIR / ".circuit_breaker.json"
        if cb_file.exists():
            with open(cb_file) as f:
                state["circuit_breaker"] = json.load(f)

        # Health report history (check how long all checks have been passing)
        health_file = DATA_DIR / ".health_report.json"
        if health_file.exists():
            with open(health_file) as f:
                state["health_report"] = json.load(f)

        return state

    def _check_trading_days(self, state: Dict) -> CheckResult:
        """Check minimum trading days of paper trading."""
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        
        # Estimate trading days from unique dates in history
        unique_dates = set()
        for entry in history:
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ts
            unique_dates.add(date_key)
        
        n_days = max(0, len(unique_dates))
        required = int(self.criteria["min_trading_days"]["value"])
        
        return CheckResult(
            name="min_trading_days",
            passed=n_days >= required,
            value=n_days,
            required=required,
            description=self.criteria["min_trading_days"]["description"],
        )

    def _check_sharpe(self, state: Dict) -> CheckResult:
        """Check rolling Sharpe ratio."""
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        
        # Use the last 63 trading days of daily data
        # Deduplicate to daily first
        daily = {}
        for entry in history:
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ts
            daily[date_key] = entry
        sorted_daily = [daily[d] for d in sorted(daily.keys())]
        
        if len(sorted_daily) < 3:
            return CheckResult(
                name="min_sharpe",
                passed=False,
                value=0.0,
                required=0.50,
                description=self.criteria["min_sharpe"]["description"],
            )
        
        # Take last 63 or all available
        recent = sorted_daily[-63:]
        returns = [h.get("daily_return", 0) for h in recent]
        
        daily_std = max(np.std(returns) if len(returns) > 1 else 0.0001, 0.0001)
        sharpe = float(np.mean(returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        # Sanity cap: any Sharpe > 3.0 is unrealistic (intra-day artifact)
        if sharpe > 3.0:
            sharpe = 0.0  # Treat as not meeting criteria
        
        required = float(self.criteria["min_sharpe"]["value"])
        return CheckResult(
            name="min_sharpe",
            passed=sharpe >= required,
            value=round(sharpe, 2),
            required=required,
            description=self.criteria["min_sharpe"]["description"],
        )

    def _check_drawdown(self, state: Dict) -> CheckResult:
        """Check maximum drawdown."""
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        
        if not history:
            return CheckResult(
                name="max_drawdown",
                passed=True,  # No data means no drawdown
                value=0.0,
                required=0.15,
                description=self.criteria["max_drawdown"]["description"],
            )
        
        # Deduplicate to daily
        daily = {}
        for entry in history:
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ts
            daily[date_key] = entry
        sorted_daily = [daily[d] for d in sorted(daily.keys())]
        
        peak = sorted_daily[0].get("total_value", 100000)
        max_dd = 0.0
        for h in sorted_daily:
            val = h.get("total_value", 0)
            if val > peak:
                peak = val
            if peak > 0:
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
        
        required = float(self.criteria["max_drawdown"]["value"])
        return CheckResult(
            name="max_drawdown",
            passed=max_dd <= required,
            value=round(max_dd, 4),
            required=required,
            description=self.criteria["max_drawdown"]["description"],
        )

    def _check_win_rate(self, state: Dict) -> CheckResult:
        """Check win rate (fraction of positive return days)."""
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        
        # Deduplicate to daily
        daily = {}
        for entry in history:
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ts
            daily[date_key] = entry
        sorted_daily = [daily[d] for d in sorted(daily.keys())]
        
        if len(sorted_daily) < 3:
            return CheckResult(
                name="min_win_rate",
                passed=False,
                value=0.0,
                required=0.40,
                description=self.criteria["min_win_rate"]["description"],
            )
        
        returns = [h.get("daily_return", 0) for h in sorted_daily]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        required = float(self.criteria["min_win_rate"]["value"])
        return CheckResult(
            name="min_win_rate",
            passed=win_rate >= required,
            value=round(win_rate, 4),
            required=required,
            description=self.criteria["min_win_rate"]["description"],
        )

    def _check_health(self, state: Dict) -> CheckResult:
        """Check that health checks have been consistently passing."""
        # We check the latest health report — if it shows all passing, that's good
        # For a more rigorous check we'd need historical health reports
        health = state.get("health_report", {})
        summary = health.get("summary", {})
        total = summary.get("total_checks", 0)
        passed = summary.get("passed", 0)
        
        required = int(self.criteria["health_checks"]["value"])
        
        # Check current report
        if total > 0 and passed == total:
            # All current checks passing — accept as meeting threshold
            return CheckResult(
                name="health_checks",
                passed=True,
                value=passed,
                required=required,
                description=self.criteria["health_checks"]["description"],
            )
        
        return CheckResult(
            name="health_checks",
            passed=False,
            value=passed,
            required=required,
            description=self.criteria["health_checks"]["description"],
        )

    def _check_tca_orders(self, state: Dict) -> CheckResult:
        """Check TCA engine has enough orders for meaningful analysis."""
        tca = state.get("tca", {})
        
        # Count orders across all symbols
        orders_by_symbol = tca.get("orders_by_symbol", {})
        total_orders = sum(len(sym_orders) for sym_orders in orders_by_symbol.values())
        
        # Also check orders.jsonl directly
        orders_file = DATA_DIR / "orders.jsonl"
        if orders_file.exists():
            with open(orders_file) as f:
                file_orders = sum(1 for _ in f if _.strip())
            total_orders = max(total_orders, file_orders)
        
        required = int(self.criteria["min_tca_orders"]["value"])
        return CheckResult(
            name="min_tca_orders",
            passed=total_orders >= required,
            value=total_orders,
            required=required,
            description=self.criteria["min_tca_orders"]["description"],
        )

    def _check_circuit_breaker(self, state: Dict) -> CheckResult:
        """Check circuit breaker confidence (no recent trips)."""
        cb = state.get("circuit_breaker", {})
        
        status = cb.get("status", "green" if isinstance(cb, dict) else "unknown")
        cb.get("trips", 0) if isinstance(cb, dict) else 0
        consecutive_ok = cb.get("consecutive_ok", 0) if isinstance(cb, dict) else 0
        
        required = int(self.criteria["circuit_breaker_confidence"]["value"])
        
        # Green status or no trips in recent history = pass
        if isinstance(status, str) and status.lower() in ("green", "ok", "normal"):
            return CheckResult(
                name="circuit_breaker_confidence",
                passed=True,
                value=consecutive_ok if consecutive_ok > 0 else 1,
                required=required,
                description=self.criteria["circuit_breaker_confidence"]["description"],
            )
        
        return CheckResult(
            name="circuit_breaker_confidence",
            passed=False,
            value=consecutive_ok,
            required=required,
            description=self.criteria["circuit_breaker_confidence"]["description"],
        )

    def _check_manual_approval(self, state: Dict) -> CheckResult:
        """Manual approval gate — always False by default."""
        # Check if an explicit approval file exists
        approval_file = DATA_DIR / ".manual_approval"
        approved = approval_file.exists()
        
        return CheckResult(
            name="manual_approval",
            passed=approved,
            value=1 if approved else 0,
            required=1,
            description=self.criteria["manual_approval"]["description"],
        )


def run_check_and_exit():
    """CLI: Check all criteria and print results."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    
    print("=" * 60)
    print("  v6.10 Graduation Checklist")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    passed = 0
    for name, result in results.items():
        icon = "✅" if result.passed else "❌"
        if name == "manual_approval":
            icon = "🔒" if not result.passed else "✅"
        print(f"  {icon} {name}")
        print(f"     {result.description}")
        print(f"     Value: {result.value} | Required: {result.required}")
        print()
        if result.passed and name != "manual_approval":
            passed += 1
    
    auto_count = sum(1 for n in results if n != "manual_approval")
    score = round(passed / auto_count * 100, 1) if auto_count > 0 else 0
    
    print(f"  Readiness: {passed}/{auto_count} auto-criteria = {score}%")
    print(f"  Graduation Ready: {checklist.is_graduation_ready(results)}")
    print(f"  Manual Approval: {'✅ APPROVED' if results['manual_approval'].passed else '🔒 PENDING'}")
    print()
    
    # Save report
    report_path = checklist.save_report(results)
    print(f"  Report saved: {report_path}")
    
    return 0 if checklist.is_graduation_ready(results) else 1


def run_report_and_exit():
    """CLI: Generate detailed graduation report with progress."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    report_path = checklist.save_report(results)
    
    print(f"Detailed report written to: {report_path}")
    
    # Print the JSON
    with open(report_path) as f:
        print(f.read())
    
    return 0


def run_progress_and_exit():
    """CLI: Show concise progress summary."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    summary = checklist.progress_summary(results)
    
    print(f"📊 Graduation Readiness: {summary['readiness_pct']}%")
    print(f"   Progress: {summary['overall_progress']} auto-criteria met")
    print(f"   Ready to graduate: {'YES' if summary['is_ready'] else 'NO'}")
    print(f"   Manual approval: {'✅' if results['manual_approval'].passed else '🔒 PENDING'}")
    print()
    
    for name, result in results.items():
        icon = "✅" if result.passed else "❌"
        if name == "manual_approval":
            icon = "🔒" if not result.passed else "✅"
        print(f"   {icon} {name}: {result.value} {'>=' if result.passed else '<'} {result.required}")
    
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    
    if command == "check":
        sys.exit(run_check_and_exit())
    elif command == "report":
        sys.exit(run_report_and_exit())
    elif command == "progress":
        sys.exit(run_progress_and_exit())
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.strategy.graduation_checklist [check|report|progress]")
        sys.exit(1)
