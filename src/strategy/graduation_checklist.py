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

import copy
import json
import logging
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

from src.paths import DATA_DIR
from src.backtest.metrics import save_results_json


PERFORMANCE_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_PERFORMANCE_LOG_TAIL_LINES", "500"))
ORDER_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_ORDER_LOG_TAIL_LINES", "1000"))
REGIME_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_REGIME_LOG_TAIL_LINES", "1000"))


__all__ = ['CheckResult', 'GraduationChecklist', 'run_check_and_exit', 'run_report_and_exit', 'run_progress_and_exit']

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

    # Default thresholds (env-var configurable)
    DEFAULT_CRITERIA = {
        "min_trading_days": {
            "value": int(os.environ.get("GRADUATION_MIN_TRADING_DAYS", "63")),
            "description": "At least 63 trading days of paper trading data",
        },
        "min_sharpe": {
            "value": float(os.environ.get("GRADUATION_MIN_SHARPE", "0.50")),
            "description": "Rolling Sharpe ratio >= 0.50",
        },
        "max_drawdown": {
            "value": float(os.environ.get("GRADUATION_MAX_DRAWDOWN", "0.25")),
            "description": "Maximum drawdown <= 25% (adjusted for 46/38/16 equity/commodity/bond portfolio; champion backtest max DD = -26.2%)",
        },
        "min_win_rate": {
            "value": float(os.environ.get("GRADUATION_MIN_WIN_RATE", "0.40")),
            "description": "Win rate >= 40%",
        },
        "health_checks": {
            "value": int(os.environ.get("GRADUATION_HEALTH_CHECKS", "30")),
            "description": "All 9 health checks passing for 30 consecutive days",
        },
        "min_tca_orders": {
            "value": int(os.environ.get("GRADUATION_MIN_TCA_ORDERS", "10")),
            "description": "TCA engine populated with >= 10 orders",
        },
        "circuit_breaker_confidence": {
            "value": int(os.environ.get("GRADUATION_CIRCUIT_BREAKER", "3")),
            "description": "Circuit breaker confidence >= 3 consecutive cycles (no trip)",
        },
        "min_dsr": {
            "value": float(os.environ.get("GRADUATION_MIN_DSR", "0.50")),
            "description": "Deflated Sharpe Ratio >= 0.50 (validates against multiple-testing bias)",
        },
        "regime_coverage": {
            "value": int(os.environ.get("GRADUATION_REGIME_COVERAGE", "2")),
            "description": "Paper trading must span at least 2 distinct volatility regimes",
        },
        "signal_diversity": {
            "value": int(os.environ.get("GRADUATION_SIGNAL_DIVERSITY", "4")),
            "description": "At least 4 of 6 active signals must have contributed to rebalance decisions",
        },
        "sharpe_ci_lower": {
            "value": float(os.environ.get("GRADUATION_SHARPE_CI_LOWER", "0.30")),
            "description": "Lower bound of 75% CI for Sharpe >= 0.30 (prevents false-positive graduation)",
        },
        "manual_approval": {
            "value": False,
            "description": "Human-in-the-loop approval (MANDATORY — always False by default)",
        },
    }

    # Gate: minimum days before ANY graduation alert (prevents false alarms)
    MIN_OBSERVATION_DAYS = 30
    OBSERVATION_WINDOW_CRITERIA = frozenset({
        "health_checks",
        "circuit_breaker_confidence",
    })

    def __init__(self, criteria: Optional[Dict] = None):
        self.criteria = copy.deepcopy(criteria) if criteria is not None else copy.deepcopy(self.DEFAULT_CRITERIA)

    @staticmethod
    def _read_jsonl_tail(path: Path, max_lines: int, label: str) -> list[Dict]:
        """Read and parse a bounded JSONL tail, skipping malformed lines."""
        entries = []
        with open(path) as f:
            for line in deque(f, maxlen=max_lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.debug("Skipping malformed %s line: %s", label, e)
        return entries

    @staticmethod
    def _count_nonempty_tail_lines(path: Path, max_lines: int) -> int:
        """Count non-empty lines in a bounded file tail."""
        with open(path) as f:
            return sum(1 for line in deque(f, maxlen=max_lines) if line.strip())

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

        # 8. DSR validation (multiple-testing correction)
        results["min_dsr"] = self._check_dsr(state)

        # 10. Regime coverage (must span multiple regimes)
        results["regime_coverage"] = self._check_regime_coverage(state)

        # 11. Signal diversity (multiple signals must contribute)
        results["signal_diversity"] = self._check_signal_diversity(state)

        # 12. Sharpe CI lower bound
        results["sharpe_ci_lower"] = self._check_sharpe_ci(state)

        # 9. Manual approval (always False by default)
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
        passed = sum(
            1
            for name, result in results.items()
            if name != "manual_approval"
            and result.passed
            and not (
                name in self.OBSERVATION_WINDOW_CRITERIA
                and result.value < result.required
            )
        )
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
        save_results_json(report, output_path=str(path))
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

        # Pre-computed paper-trading-performance summary (authoritative metrics)
        # Glob for the latest file — contains days_tracked, sharpe, max_drawdown, win_rate
        perf_summary_files = sorted(DATA_DIR.glob("paper-trading-performance-*.json"))
        if perf_summary_files:
            latest = perf_summary_files[-1]
            try:
                with open(latest) as f:
                    raw = json.load(f)
                state["paper_trading_summary"] = {
                    "days_tracked": raw.get("performance", {}).get("days_tracked", 0),
                    "sharpe": raw.get("performance", {}).get("sharpe", 0),
                    "max_drawdown": raw.get("performance", {}).get("max_drawdown", 0),
                    "win_rate": raw.get("daily_returns_distribution", {}).get("win_rate", 0),
                    "date": raw.get("date", ""),
                }
                logger.info(
                    "Loaded paper-trading summary from %s: %d days, Sharpe %.2f",
                    latest.name,
                    state["paper_trading_summary"]["days_tracked"],
                    state["paper_trading_summary"]["sharpe"],
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Failed to parse %s: %s", latest.name, exc)

        # Performance history
        perf_file = DATA_DIR / "performance.jsonl"
        if perf_file.exists():
            state["performance"] = self._read_jsonl_tail(
                perf_file,
                PERFORMANCE_LOG_TAIL_LINES,
                "performance",
            )

        # TCA scorecard (producer removed v977)

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
        # Prefer pre-computed summary over recomputation from portfolio_paper.json
        summary = state.get("paper_trading_summary", {})
        if summary.get("days_tracked", 0) > 0:
            n_days = summary["days_tracked"]
        else:
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
        # Prefer pre-computed summary
        summary = state.get("paper_trading_summary", {})
        if summary.get("sharpe", 0) > 0:
            sharpe = summary["sharpe"]
            # Sanity cap: any Sharpe > 3.0 is unrealistic (intra-day artifact)
            if sharpe > 3.0:
                sharpe = 0.0
        else:
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
                sharpe = 0.0

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
        # Prefer pre-computed summary
        summary = state.get("paper_trading_summary", {})
        if summary.get("max_drawdown", 0) > 0 or summary.get("days_tracked", 0) > 0:
            max_dd = summary.get("max_drawdown", 0)
        else:
            portfolio = state.get("portfolio", {})
            history = portfolio.get("history", [])

            if not history:
                return CheckResult(
                    name="max_drawdown",
                    passed=True,  # No data means no drawdown
                    value=0.0,
                    required=self.criteria["max_drawdown"]["value"],
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
        # Prefer pre-computed summary
        summary = state.get("paper_trading_summary", {})
        if summary.get("days_tracked", 0) > 0 and "win_rate" in summary:
            win_rate = summary["win_rate"]
        else:
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
        health = state.get("health_report", {})
        summary = health.get("summary", {})
        total = summary.get("total_checks", 0)
        passed = summary.get("passed", 0)
        
        required = int(self.criteria["health_checks"]["value"])
        consecutive_days = int(
            summary.get("consecutive_passing_days")
            or summary.get("consecutive_ok_days")
            or summary.get("consecutive_green_days")
            or 0
        )
        
        if total > 0 and passed == total and consecutive_days >= required:
            return CheckResult(
                name="health_checks",
                passed=True,
                value=consecutive_days,
                required=required,
                description=self.criteria["health_checks"]["description"],
            )
        
        return CheckResult(
            name="health_checks",
            passed=False,
            value=consecutive_days,
            required=required,
            description=self.criteria["health_checks"]["description"],
        )

    def _check_tca_orders(self, state: Dict) -> CheckResult:
        """Check TCA engine has enough orders for meaningful analysis."""
        tca = state.get("tca", {})

        # Count orders across all symbols
        orders_by_symbol = tca.get("orders_by_symbol", {})
        total_orders = (
            sum(len(sym_orders) for sym_orders in orders_by_symbol.values())
            if isinstance(orders_by_symbol, dict)
            else 0
        )
        
        # Also check orders.jsonl directly
        orders_file = DATA_DIR / "orders.jsonl"
        if orders_file.exists():
            file_orders = self._count_nonempty_tail_lines(
                orders_file,
                ORDER_LOG_TAIL_LINES,
            )
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

        if not isinstance(cb, dict):
            cb = {}
        status = cb.get("status", "green")
        trips = cb.get("trips", 0)
        consecutive_ok = cb.get("consecutive_ok", 0)
        
        required = int(self.criteria["circuit_breaker_confidence"]["value"])
        
        status_ok = isinstance(status, str) and status.lower() in ("green", "ok", "normal", "yellow")
        if status_ok and trips == 0 and consecutive_ok >= required:
            return CheckResult(
                name="circuit_breaker_confidence",
                passed=True,
                value=consecutive_ok,
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

    def _check_dsr(self, state: Dict) -> CheckResult:
        """Deflated Sharpe Ratio — validates Sharpe against multiple-testing bias.

        With 94 grid-search configurations, DSR quantifies the probability
        that the observed Sharpe ratio is genuinely positive rather than the
        best of many random trials. DSR >= 0.50 indicates the Sharpe is
        more likely than not to be real.
        """
        required = self.criteria["min_dsr"]["value"]

        # Prefer pre-computed summary for Sharpe
        summary = state.get("paper_trading_summary", {})
        if summary.get("sharpe", 0) > 0:
            sharpe = summary["sharpe"]
            n_days = summary.get("days_tracked", 0)
            if sharpe > 3.0:
                sharpe = 0.0
        else:
            # Compute Sharpe from portfolio history (same as _check_sharpe)
            portfolio = state.get("portfolio", {})
            history = portfolio.get("history", [])
            daily = {}
            for entry in history:
                ts = entry.get("timestamp", "")
                date_key = ts[:10] if len(ts) >= 10 else ts
                daily[date_key] = entry
            sorted_daily = [daily[d] for d in sorted(daily.keys())]
            recent = sorted_daily[-63:] if len(sorted_daily) >= 3 else sorted_daily

            if len(recent) < 3:
                return CheckResult(
                    name="min_dsr",
                    passed=False,
                    value=0.0,
                    required=required,
                    description=self.criteria["min_dsr"]["description"],
                )

            returns = [h.get("daily_return", 0) for h in recent]
            daily_std = max(np.std(returns) if len(returns) > 1 else 0.0001, 0.0001)
            sharpe = float(np.mean(returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            n_days = len(recent)
            if sharpe > 3.0:
                sharpe = 0.0

        try:
            from src.backtest.metrics import compute_deflated_sharpe_ratio
            n_obs = n_days * 252  # Scale to annual observations
            dsr = compute_deflated_sharpe_ratio(
                sharpe_ratio=sharpe, n_trials=94, n_observations=n_obs,
            )
        except (ImportError, ValueError, ZeroDivisionError, OverflowError):
            dsr = 0.0

        return CheckResult(
            name="min_dsr",
            passed=dsr >= required,
            value=round(dsr, 4),
            required=required,
            description=self.criteria["min_dsr"]["description"],
        )

    def _check_regime_coverage(self, state: Dict) -> CheckResult:
        """Check that paper trading has spanned multiple volatility regimes.

        A portfolio that only traded during a calm bull market hasn't been
        stress-tested. We require at least 2 distinct regimes to have been
        observed during the paper trading period.
        """
        required = self.criteria["regime_coverage"]["value"]

        # Count distinct regimes from regime state history
        regime_file = DATA_DIR / "regime_state.json"
        regimes_seen = set()

        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    data = json.load(f)
                current = data.get("regime", "")
                if current:
                    regimes_seen.add(current)
            except (OSError, ValueError, KeyError):
                pass

        # Also check portfolio history for regime snapshots
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        for entry in history:
            regime = entry.get("regime", "")
            if regime:
                regimes_seen.add(regime)

        # Check regime log if available
        regime_log = DATA_DIR / "regime_log.json"
        if regime_log.exists():
            try:
                for entry in self._read_jsonl_tail(
                    regime_log,
                    REGIME_LOG_TAIL_LINES,
                    "regime log",
                ):
                    regime = entry.get("regime", "")
                    if regime:
                        regimes_seen.add(regime)
            except (OSError, ValueError):
                pass

        n_regimes = len(regimes_seen)

        return CheckResult(
            name="regime_coverage",
            passed=n_regimes >= required,
            value=n_regimes,
            required=required,
            description=self.criteria["regime_coverage"]["description"],
        )

    def _check_signal_diversity(self, state: Dict) -> CheckResult:
        """Check that multiple ensemble signals have contributed to decisions.

        If only 1-2 signals drive all rebalance decisions, the portfolio
        hasn't validated the full ensemble. We require at least 4 of 6
        active signals to have non-zero weight in rebalance decisions.
        """
        required = self.criteria["signal_diversity"]["value"]

        # Check ensemble voter state for active signals
        signals_contributing = set()
        ensemble_file = DATA_DIR / "ensemble_state.json"
        if ensemble_file.exists():
            try:
                with open(ensemble_file) as f:
                    data = json.load(f)
                weights = data.get("weights", {})
                for signal, weight in weights.items():
                    if weight > 0.01:  # Non-trivial weight
                        signals_contributing.add(signal)
            except (OSError, ValueError, KeyError):
                pass

        # Also check order log for signal attribution
        orders_file = DATA_DIR / "orders.jsonl"
        if orders_file.exists():
            try:
                for entry in self._read_jsonl_tail(
                    orders_file,
                    ORDER_LOG_TAIL_LINES,
                    "orders",
                ):
                    signal = entry.get("signal_source", "")
                    if signal:
                        signals_contributing.add(signal)
            except (OSError, ValueError):
                pass

        n_signals = len(signals_contributing)

        return CheckResult(
            name="signal_diversity",
            passed=n_signals >= required,
            value=n_signals,
            required=required,
            description=self.criteria["signal_diversity"]["description"],
        )

    def _check_sharpe_ci(self, state: Dict) -> CheckResult:
        """Check that the lower bound of the 75% CI for Sharpe is above threshold.

        With few observations (e.g., 63 days), the Sharpe point estimate has
        wide confidence intervals. Using the lower bound prevents false-positive
        graduation when the point estimate is high but imprecise.

        CI formula: Sharpe ± z * SE, where SE ≈ sqrt((1 + 0.5*Sharpe²) / N)
        For 75% CI, z = 1.15.
        """
        required = self.criteria["sharpe_ci_lower"]["value"]

        # Get Sharpe from the same source as _check_sharpe
        summary = state.get("paper_trading_summary", {})
        if summary.get("sharpe", 0) > 0:
            sharpe = summary["sharpe"]
            n_days = summary.get("days_tracked", 0)
            if sharpe > 3.0:
                sharpe = 0.0
        else:
            portfolio = state.get("portfolio", {})
            history = portfolio.get("history", [])
            daily = {}
            for entry in history:
                ts = entry.get("timestamp", "")
                date_key = ts[:10] if len(ts) >= 10 else ts
                daily[date_key] = entry
            sorted_daily = [daily[d] for d in sorted(daily.keys())]
            recent = sorted_daily[-63:] if len(sorted_daily) >= 3 else sorted_daily

            if len(recent) < 3:
                return CheckResult(
                    name="sharpe_ci_lower",
                    passed=False,
                    value=0.0,
                    required=required,
                    description=self.criteria["sharpe_ci_lower"]["description"],
                )

            returns = [h.get("daily_return", 0) for h in recent]
            daily_std = max(np.std(returns) if len(returns) > 1 else 0.0001, 0.0001)
            sharpe = float(np.mean(returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            n_days = len(recent)
            if sharpe > 3.0:
                sharpe = 0.0

        if n_days < 5:
            return CheckResult(
                name="sharpe_ci_lower",
                passed=False,
                value=0.0,
                required=required,
                description=self.criteria["sharpe_ci_lower"]["description"],
            )

        # Compute 75% CI lower bound
        # SE(Sharpe) ≈ sqrt((1 + 0.5*Sharpe²) / N)  (Lo, 2002)
        n_annual = n_days  # Daily observations
        se_sharpe = np.sqrt((1 + 0.5 * sharpe ** 2) / n_annual) if n_annual > 0 else 1.0
        z_75 = 1.15  # 75% CI z-score
        ci_lower = sharpe - z_75 * se_sharpe

        return CheckResult(
            name="sharpe_ci_lower",
            passed=ci_lower >= required,
            value=round(ci_lower, 4),
            required=required,
            description=self.criteria["sharpe_ci_lower"]["description"],
        )


def run_check_and_exit():
    """CLI: Check all criteria and print results."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    
    logger.info("=" * 60)
    logger.info("  v6.10 Graduation Checklist")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    
    passed = 0
    for name, result in results.items():
        icon = "✅" if result.passed else "❌"
        if name == "manual_approval":
            icon = "🔒" if not result.passed else "✅"
        logger.info(f"  {icon} {name}")
        logger.info(f"     {result.description}")
        logger.info(f"     Value: {result.value} | Required: {result.required}")
        logger.info("")
        if result.passed and name != "manual_approval":
            passed += 1
    
    auto_count = sum(1 for n in results if n != "manual_approval")
    score = round(passed / auto_count * 100, 1) if auto_count > 0 else 0
    
    logger.info(f"  Readiness: {passed}/{auto_count} auto-criteria = {score}%")
    logger.info(f"  Graduation Ready: {checklist.is_graduation_ready(results)}")
    logger.info(f"  Manual Approval: {'✅ APPROVED' if results['manual_approval'].passed else '🔒 PENDING'}")
    logger.info("")
    
    # Save report
    report_path = checklist.save_report(results)
    logger.info(f"  Report saved: {report_path}")
    
    return 0 if checklist.is_graduation_ready(results) else 1


def run_report_and_exit():
    """CLI: Generate detailed graduation report with progress."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    report_path = checklist.save_report(results)
    
    logger.info(f"Detailed report written to: {report_path}")
    
    # Print the JSON
    with open(report_path) as f:
        logger.info(f.read())
    
    return 0


def run_progress_and_exit():
    """CLI: Show concise progress summary."""
    checklist = GraduationChecklist()
    state = checklist._load_state()
    results = checklist.check(state)
    summary = checklist.progress_summary(results)
    
    logger.info(f"📊 Graduation Readiness: {summary['readiness_pct']}%")
    logger.info(f"   Progress: {summary['overall_progress']} auto-criteria met")
    logger.info(f"   Ready to graduate: {'YES' if summary['is_ready'] else 'NO'}")
    logger.info(f"   Manual approval: {'✅' if results['manual_approval'].passed else '🔒 PENDING'}")
    logger.info("")
    
    for name, result in results.items():
        icon = "✅" if result.passed else "❌"
        if name == "manual_approval":
            icon = "🔒" if not result.passed else "✅"
        logger.info(f"   {icon} {name}: {result.value} {'>=' if result.passed else '<'} {result.required}")
    
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
        logger.warning("Unknown command: %s", command)
        logger.warning("Usage: python -m src.strategy.graduation_checklist [check|report|progress]")
        sys.exit(1)
