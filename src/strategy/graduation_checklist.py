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
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Optional, NamedTuple

import numpy as np

from src.paths import DATA_DIR, PUBLIC_DATA_DIR, SIGNALS_JSON
from src.backtest.metrics import save_results_json
logger = logging.getLogger(__name__)



PERFORMANCE_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_PERFORMANCE_LOG_TAIL_LINES", "500"))
ORDER_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_ORDER_LOG_TAIL_LINES", "1000"))
REGIME_LOG_TAIL_LINES = int(os.environ.get("GRADUATION_REGIME_LOG_TAIL_LINES", "1000"))


__all__ = [
    'CheckResult',
    'GraduationChecklist',
    'is_ops_health_inventory',
    'run_check_and_exit',
    'run_report_and_exit',
    'run_progress_and_exit',
]


def is_ops_health_inventory(health: dict | None) -> bool:
    """True only for multi-check ops inventories; False for GARCH/risk stubs.

    GARCH writers reuse ``.health_report.json`` and stamp
    ``summary.total_checks=1`` (portfolio_entropy only). That must not
    suppress the graduation CB consecutive_ok multi-day SSOT.
    """
    if not isinstance(health, dict) or not health:
        return False

    summary = health.get("summary") if isinstance(health.get("summary"), dict) else {}
    role = str(
        summary.get("inventory_role")
        or health.get("inventory_role")
        or health.get("schema_role")
        or ""
    ).lower()
    if role in {"garch_risk", "garch", "risk", "risk_only", "non_ops"}:
        return False

    # Explicit GARCH / risk shape markers (producer historical + current)
    garch_markers = (
        "var_95",
        "cvar_95",
        "garch_filtered",
        "garch_omega",
        "garch_alpha",
        "garch_beta",
        "garch_active",
        "conditional_volatility_current",
    )
    if any(k in health for k in garch_markers):
        return False

    total = 0
    try:
        total = int(summary.get("total_checks") or 0)
    except (TypeError, ValueError):
        total = 0

    if total <= 0:
        return False

    # Single entropy-only stub without multi-day fields is not ops inventory
    checks = health.get("checks") if isinstance(health.get("checks"), dict) else {}
    consecutive_keys = (
        "consecutive_passing_days",
        "consecutive_ok_days",
        "consecutive_green_days",
    )
    has_consecutive = any(summary.get(k) is not None for k in consecutive_keys)
    if total == 1 and not has_consecutive:
        if not checks or set(checks.keys()) <= {"portfolio_entropy"}:
            return False

    return True


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
            "description": (
                "Ops health green AND signal_health not 0/N degraded for "
                "N consecutive days (SSOT: health report consecutive counter "
                "and/or graduation circuit_breaker consecutive_ok; not a "
                "fixed '9 checks' inventory)"
            ),
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

        cb = results.get("circuit_breaker_confidence")
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
            # Batch BP: stamp CB SSOT identity for dual-surface equality checks
            "circuit_breaker_ssot": ".circuit_breaker.json",
            "circuit_breaker_consecutive_ok": (
                cb.value if cb is not None else None
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(report, output_path=str(path))
        return path

    def _tombstone_stale_promote(
        self,
        root: Path,
        *,
        action: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
        readiness_score: Optional[float] = None,
    ) -> None:
        """Rewrite candidacy marker so no file claims action promote_to_live.

        Writes the same tombstone to ``.promote_to_live`` (when present or
        always for kill) and ``.graduation_conflict.json`` for operator SSOT.
        """
        tombstone: Dict[str, Any] = {
            "graduation_conflict": True,
            "action": action,
            "reason": reason,
            "is_graduation_ready": False,
            "timestamp": datetime.now().isoformat(),
            "source": "graduation_checklist",
            "requires_approval": True,
        }
        if readiness_score is not None:
            tombstone["readiness_score"] = readiness_score
        if extra:
            tombstone.update(extra)

        promote_path = root / ".promote_to_live"
        # Always rewrite if a candidacy file exists, or under kill block even if
        # missing (no-op create only when prior candidacy existed — kill path
        # only tombstones when file present so we do not invent markers).
        if promote_path.exists():
            try:
                prior = json.loads(promote_path.read_text(encoding="utf-8"))
                if isinstance(prior, dict) and prior.get("metrics") is not None:
                    tombstone.setdefault("prior_metrics", prior.get("metrics"))
            except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
                pass
            save_results_json(tombstone, output_path=str(promote_path))

        conflict_path = root / ".graduation_conflict.json"
        save_results_json(tombstone, output_path=str(conflict_path))

    def clear_kill_gated_promote_markers(
        self,
        *,
        data_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Batch IU DQ1: clear ``promote_blocked_kill`` markers when kill is healed.

        Kill-gated tombstones (``.graduation_conflict.json`` /
        ``.promote_to_live`` with ``action=promote_blocked_kill``) must not
        stick after kill authority is clear. Checklist-not-ready tombstones
        are left alone — only kill-gated actions clear-on-heal.

        Safe to call from health/dashboard paths every cycle (idempotent).
        Does **not** invent promote_to_live candidacy.
        """
        root = Path(data_dir) if data_dir is not None else DATA_DIR
        out: Dict[str, Any] = {
            "cleared": False,
            "kill_clear": False,
            "removed": [],
        }

        kill_blocked = False
        try:
            from src.dashboard.kill_authority import (
                is_kill_execution_blocked,
                load_kill_switch_payload,
            )

            kill_payload = load_kill_switch_payload(root)
            kill_blocked = bool(is_kill_execution_blocked(kill_payload))
        except ImportError:
            kill_file = root / "kill_switch.json"
            if kill_file.exists():
                try:
                    payload = json.loads(kill_file.read_text(encoding="utf-8"))
                    kill_blocked = bool(
                        isinstance(payload, dict) and payload.get("enabled")
                    )
                except (OSError, json.JSONDecodeError, TypeError):
                    kill_blocked = True  # fail-closed: do not clear

        if kill_blocked:
            out["kill_clear"] = False
            out["reason"] = "kill_still_active"
            return out
        out["kill_clear"] = True

        def _is_kill_gated(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
                return False
            if not isinstance(body, dict):
                return False
            action = str(body.get("action") or "")
            reason = str(body.get("reason") or "")
            return action == "promote_blocked_kill" or (
                body.get("graduation_conflict") is True and reason == "kill_authority"
            )

        for rel in (".graduation_conflict.json", ".promote_to_live"):
            path = root / rel
            if not _is_kill_gated(path):
                continue
            try:
                path.unlink()
                out["removed"].append(rel)
                out["cleared"] = True
            except OSError as exc:
                logger.warning("Failed to clear kill-gated marker %s: %s", path, exc)

        if out["cleared"]:
            logger.info(
                "Cleared kill-gated promote markers after kill heal: %s",
                out["removed"],
            )
        return out

    def write_promote_to_live_if_ready(
        self,
        results: Optional[Dict[str, CheckResult]] = None,
        *,
        data_dir: Optional[Path] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """Sole writer for ``.promote_to_live`` candidacy (SSOT).

        Only writes when the multi-criteria checklist is ready. Never writes
        under authority kill halt. Returns path written, or None if skipped.

        Manual approval remains a separate human gate; this marker means
        *checklist-ready candidate*, not auto-live.

        When blocked (kill or checklist fail), any existing ``action:
        promote_to_live`` candidacy is tombstoned so operators never see a
        live promote claim under halt / not-ready.

        Batch IU DQ1: when kill is clear, kill-gated tombstones are cleared
        even if checklist is not ready (no false promote_blocked_kill stickiness).
        """
        root = Path(data_dir) if data_dir is not None else DATA_DIR
        if results is None:
            results = self.check()

        # Kill authority blocks promote writes (same SSOT as evaluator / order_router)
        kill_payload: Optional[Dict[str, Any]] = None
        try:
            from src.dashboard.kill_authority import (
                is_kill_execution_blocked,
                load_kill_switch_payload,
            )

            kill_payload = load_kill_switch_payload(root)
            if is_kill_execution_blocked(kill_payload):
                level = None
                if isinstance(kill_payload, dict):
                    level = kill_payload.get("level")
                self._tombstone_stale_promote(
                    root,
                    action="promote_blocked_kill",
                    reason="kill_authority",
                    extra={
                        "kill_level": level,
                        "kill_reason": (
                            kill_payload.get("reason")
                            if isinstance(kill_payload, dict)
                            else None
                        ),
                        "kill_incident_id": (
                            kill_payload.get("incident_id")
                            if isinstance(kill_payload, dict)
                            else None
                        ),
                    },
                    readiness_score=self.readiness_score(results),
                )
                logger.info(
                    "Promote blocked by kill authority — tombstoned stale candidacy "
                    "(level=%s)",
                    level,
                )
                return None
        except ImportError:
            kill_file = root / "kill_switch.json"
            if kill_file.exists():
                try:
                    payload = json.loads(kill_file.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and payload.get("enabled"):
                        self._tombstone_stale_promote(
                            root,
                            action="promote_blocked_kill",
                            reason="kill_authority",
                            extra={
                                "kill_level": payload.get("level"),
                                "kill_reason": payload.get("reason"),
                            },
                            readiness_score=self.readiness_score(results),
                        )
                        logger.info(
                            "Promote blocked by kill authority — tombstoned stale candidacy"
                        )
                        return None
                except (OSError, json.JSONDecodeError, TypeError):
                    logger.warning("Kill switch unreadable — fail-closed, skip promote write")
                    return None

        # Kill clear-on-heal (DQ1): drop sticky promote_blocked_kill even when
        # checklist is not ready. Do not invent candidacy.
        self.clear_kill_gated_promote_markers(data_dir=root)

        ready = self.is_graduation_ready(results)
        if not ready and not force:
            # Fail-closed: tombstone stale promote markers that disagree with checklist
            stale = root / ".promote_to_live"
            if stale.exists():
                try:
                    prior = json.loads(stale.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    prior = {}
                # Only tombstone true candidacy — not residual kill-gated markers
                # (already cleared above) or already-checklist-blocked markers.
                action = prior.get("action") if isinstance(prior, dict) else None
                if action == "promote_to_live":
                    self._tombstone_stale_promote(
                        root,
                        action="promote_blocked_checklist",
                        reason="checklist_not_ready",
                        readiness_score=self.readiness_score(results),
                    )
                    logger.info(
                        "Checklist not ready — tombstoned stale promote candidacy "
                        "(action=promote_blocked_checklist)"
                    )
            return None

        metrics = {
            "sharpe": float(results["min_sharpe"].value) if "min_sharpe" in results else None,
            "max_drawdown": float(results["max_drawdown"].value) if "max_drawdown" in results else None,
            "win_rate": float(results["min_win_rate"].value) if "min_win_rate" in results else None,
            "dsr": float(results["min_dsr"].value) if "min_dsr" in results else None,
            "readiness_score": self.readiness_score(results),
        }
        trigger = {
            "action": "promote_to_live",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "requires_approval": True,
            "source": "graduation_checklist",
            "graduation_conflict": False,
            "is_graduation_ready": True,
        }
        trigger_path = root / ".promote_to_live"
        save_results_json(trigger, output_path=str(trigger_path))
        # Clear prior conflict flag if any
        conflict_path = root / ".graduation_conflict.json"
        if conflict_path.exists():
            try:
                conflict_path.unlink()
            except OSError:
                pass
        logger.info("Created promotion trigger (checklist SSOT): %s", trigger_path)
        return trigger_path

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

        # Circuit breaker confidence SSOT (Batch BP):
        # Only ``.circuit_breaker.json`` (health_check consecutive_ok producer).
        # Never invent consecutive_ok from legacy ``.circuit_breaker_state.json``
        # (drawdown paper file without consecutive_ok) — that caused private
        # reports to claim CB=required while live SSOT was lower.
        cb_payload = None
        cb_ssot = DATA_DIR / ".circuit_breaker.json"
        if cb_ssot.exists():
            try:
                with open(cb_ssot) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    cb_payload = dict(raw)
                    cb_payload["ssot_path"] = ".circuit_breaker.json"
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to read graduation CB SSOT %s: %s", cb_ssot, exc)
        if cb_payload is not None:
            # Normalize status/severity; never invent consecutive_ok if missing
            status = cb_payload.get("status") or cb_payload.get("severity") or "green"
            if isinstance(status, str):
                status_l = status.lower()
                if status_l in ("ok", "normal", "closed"):
                    status_l = "green"
                cb_payload["status"] = status_l
            if "consecutive_ok" not in cb_payload:
                # Missing streak counter → fail-closed value 0 (do not invent)
                cb_payload["consecutive_ok"] = 0
                cb_payload["consecutive_ok_invented"] = False
                cb_payload["consecutive_ok_missing"] = True
            state["circuit_breaker"] = cb_payload
        else:
            # No SSOT file yet → explicit zero (health producer not run)
            state["circuit_breaker"] = {
                "status": "unknown",
                "trips": 0,
                "consecutive_ok": 0,
                "ssot_path": ".circuit_breaker.json",
                "ssot_missing": True,
            }

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

    @staticmethod
    def _sharpe_plausibility(sharpe: float) -> tuple[bool, Optional[str]]:
        """Sharpe > 3.0 is treated as implausible (short-sample / artifact).

        Honesty contract: keep the **raw** value for operators; fail the gate.
        Never coerce to 0.0 (looked like measured zero performance).
        """
        if sharpe > 3.0:
            return False, (
                f"implausible raw Sharpe {sharpe:.2f} > 3.0 "
                "(likely short-sample or near-zero vol artifact; gate fails)"
            )
        return True, None

    @staticmethod
    def _summary_sharpe(state: Dict) -> Optional[float]:
        """Return the authoritative summary Sharpe when it is finite and real."""
        summary = state.get("paper_trading_summary", {})
        if not isinstance(summary, dict):
            return None

        value = summary.get("sharpe")
        if isinstance(value, bool) or not isinstance(value, Real):
            return None

        sharpe = float(value)
        return sharpe if np.isfinite(sharpe) else None

    def _check_sharpe(self, state: Dict) -> CheckResult:
        """Check rolling Sharpe ratio."""
        # Prefer pre-computed summary
        summary_sharpe = self._summary_sharpe(state)
        if summary_sharpe is not None:
            sharpe = summary_sharpe
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

        required = float(self.criteria["min_sharpe"]["value"])
        plausible, note = self._sharpe_plausibility(sharpe)
        desc = self.criteria["min_sharpe"]["description"]
        if note:
            desc = f"{desc} — {note}"
        return CheckResult(
            name="min_sharpe",
            # Implausible high Sharpe fails even if numerically above threshold
            passed=bool(plausible and sharpe >= required),
            value=round(sharpe, 2),
            required=required,
            description=desc,
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

    def _session_returns_from_daily_pnl(self) -> list[float]:
        """Load session daily returns from daily_pnl.jsonl (SSOT).

        Excludes zero-placeholder / micro-noise rows (|r| < 1e-8) so phantom
        flat sessions do not dilute win rate.
        """
        path = DATA_DIR / "daily_pnl.jsonl"
        if not path.exists():
            return []
        by_date: dict[str, float] = {}
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
                    date_key = str(row.get("date") or "")[:10]
                    if not date_key:
                        continue
                    try:
                        ret = float(row.get("daily_return") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    by_date[date_key] = ret
        except OSError:
            return []
        returns = []
        for d in sorted(by_date.keys()):
            r = by_date[d]
            if abs(r) < 1e-8:
                continue  # exclude zero placeholders
            returns.append(r)
        return returns

    def _check_win_rate(self, state: Dict) -> CheckResult:
        """Check win rate (fraction of positive return days).

        SSOT: daily_pnl.jsonl session returns when present; else paper summary;
        else portfolio history (legacy). Zero-placeholder rows excluded.
        """
        required = float(self.criteria["min_win_rate"]["value"])
        desc = self.criteria["min_win_rate"]["description"]

        # 0) Explicit injected session returns (tests / callers)
        if isinstance(state.get("session_returns"), list) and len(state["session_returns"]) >= 3:
            session_returns = [float(r) for r in state["session_returns"]]
            win_rate = sum(1 for r in session_returns if r > 0) / len(session_returns)
            return CheckResult(
                name="min_win_rate",
                passed=win_rate >= required,
                value=round(win_rate, 4),
                required=required,
                description=f"{desc} (ssot=state.session_returns, n={len(session_returns)})",
            )

        # 1) Session SSOT from daily_pnl.jsonl (isolated via DATA_DIR in tests)
        session_returns = self._session_returns_from_daily_pnl()
        if len(session_returns) >= 3:
            win_rate = sum(1 for r in session_returns if r > 0) / len(session_returns)
            return CheckResult(
                name="min_win_rate",
                passed=win_rate >= required,
                value=round(win_rate, 4),
                required=required,
                description=f"{desc} (ssot=daily_pnl.jsonl, n={len(session_returns)})",
            )

        # 2) Prefer pre-computed summary when it has material days
        summary = state.get("paper_trading_summary", {})
        if summary.get("days_tracked", 0) > 0 and "win_rate" in summary:
            win_rate = float(summary["win_rate"] or 0.0)
            return CheckResult(
                name="min_win_rate",
                passed=win_rate >= required,
                value=round(win_rate, 4),
                required=required,
                description=f"{desc} (ssot=paper_trading_summary)",
            )

        # 3) Legacy portfolio history (exclude micro-noise zeros)
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        daily = {}
        for entry in history:
            ts = entry.get("timestamp", "")
            date_key = ts[:10] if len(ts) >= 10 else ts
            daily[date_key] = entry
        sorted_daily = [daily[d] for d in sorted(daily.keys())]
        returns = []
        for h in sorted_daily:
            try:
                r = float(h.get("daily_return", 0) or 0)
            except (TypeError, ValueError):
                r = 0.0
            if abs(r) < 1e-8:
                continue
            returns.append(r)

        if len(returns) < 3:
            return CheckResult(
                name="min_win_rate",
                passed=False,
                value=0.0,
                required=required,
                description=desc,
            )

        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        return CheckResult(
            name="min_win_rate",
            passed=win_rate >= required,
            value=round(win_rate, 4),
            required=required,
            description=f"{desc} (ssot=portfolio.history)",
        )

    def _check_health(self, state: Dict) -> CheckResult:
        """Check consecutive healthy days with signal_health honesty.

        SSOT alignment (Batch AO):
        - Prefer ``consecutive_*`` counters from health_report summary.
        - Also accept graduation CB ``consecutive_ok`` as the green streak
          producer when health summary lacks multi-day counters.
        - Do **not** pass when public/ops signal_health is 0 healthy of N tracked
          (ops-only greenwash).
        - Description no longer claims a fixed '9 checks' inventory.
        """
        health = state.get("health_report", {})
        summary = health.get("summary", {}) if isinstance(health, dict) else {}
        if not isinstance(summary, dict):
            summary = {}
        try:
            total = int(summary.get("total_checks") or 0) if summary else 0
        except (TypeError, ValueError):
            total = 0
        try:
            passed = int(summary.get("passed") or 0) if summary else 0
        except (TypeError, ValueError):
            passed = 0

        required = int(self.criteria["health_checks"]["value"])
        consecutive_days = int(
            summary.get("consecutive_passing_days")
            or summary.get("consecutive_ok_days")
            or summary.get("consecutive_green_days")
            or 0
        )
        # CB producer is multi-day green streak SSOT when health_report is not a
        # real ops inventory (missing inventory, or GARCH/risk stub that stamps
        # total_checks without multi-day counters). Real ops inventories with
        # total_checks but no consecutive_* stay fail-closed (Batch AO).
        ops_inventory = is_ops_health_inventory(
            health if isinstance(health, dict) else None
        )
        cb = state.get("circuit_breaker", {})
        if (
            isinstance(cb, dict)
            and consecutive_days == 0
            and not ops_inventory
        ):
            cb_ok = int(cb.get("consecutive_ok") or 0)
            if cb_ok > 0:
                consecutive_days = cb_ok

        # Signal-health quality gate: 0/N healthy cannot count as multi-day pass
        sh_blocked = False
        sh = None
        if isinstance(health, dict):
            sh = health.get("signal_health")
        if sh is None:
            sh = state.get("signal_health")
        if isinstance(sh, dict):
            try:
                from src.dashboard.health_report import signal_health_status_contribution

                contrib = signal_health_status_contribution(sh)
                # Batch EL: align with graduation CB producer — only hard
                # quality outages (degraded/critical, e.g. 0/N healthy)
                # block the health_checks criterion. Soft ``warning``
                # (partial healthy sleeves) must not permanently fail
                # graduation while ops green streak climbs.
                if contrib in {"degraded", "critical"}:
                    sh_blocked = True
            except Exception:  # noqa: BLE001
                # Fall back to summary counts when contribution helper unavailable
                sm = sh.get("summary") if isinstance(sh.get("summary"), dict) else sh
                try:
                    healthy_n = int(sm.get("healthy") or 0)
                    total_n = int(sm.get("total_tracked") or sm.get("total") or 0)
                    if total_n > 0 and healthy_n == 0:
                        sh_blocked = True
                except (TypeError, ValueError):
                    pass

        desc = self.criteria["health_checks"]["description"]
        if sh_blocked:
            return CheckResult(
                name="health_checks",
                passed=False,
                value=consecutive_days,
                required=required,
                description=f"{desc} — blocked: signal_health 0/N or degraded",
            )

        # Ops inventory all-pass, or non-ops / empty inventory relying on
        # consecutive/CB multi-day streak SSOT.
        ops_ok = (
            (ops_inventory and total > 0 and passed == total)
            or (not ops_inventory and consecutive_days > 0)
            or (total == 0 and consecutive_days > 0)
        )
        if ops_ok and consecutive_days >= required:
            return CheckResult(
                name="health_checks",
                passed=True,
                value=consecutive_days,
                required=required,
                description=desc,
            )

        return CheckResult(
            name="health_checks",
            passed=False,
            value=consecutive_days,
            required=required,
            description=desc,
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
        summary_sharpe = self._summary_sharpe(state)
        if summary_sharpe is not None:
            sharpe = summary_sharpe
            n_days = int(summary.get("days_tracked", 0) or 0)
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

        plausible, note = self._sharpe_plausibility(sharpe)
        desc = self.criteria["min_dsr"]["description"]
        if not plausible:
            # Do not feed implausible Sharpe into DSR as if it were real skill
            if note:
                desc = f"{desc} — {note}; DSR not computed on implausible input"
            return CheckResult(
                name="min_dsr",
                passed=False,
                value=0.0,
                required=required,
                description=desc,
            )

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
            description=desc,
        )

    def _check_regime_coverage(self, state: Dict) -> CheckResult:
        """Check that paper trading has spanned multiple volatility regimes.

        A portfolio that only traded during a calm bull market hasn't been
        stress-tested. We require at least 2 distinct regimes to have been
        observed during the paper trading period.
        """
        required = self.criteria["regime_coverage"]["value"]

        # Count distinct regimes from regime state history + log SSOT
        regime_file = DATA_DIR / "regime_state.json"
        regimes_seen = set()
        sources_used: list[str] = []

        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    data = json.load(f)
                current = data.get("regime", "")
                if current:
                    regimes_seen.add(current)
                for entry in data.get("history") or []:
                    if isinstance(entry, dict) and entry.get("regime"):
                        regimes_seen.add(entry["regime"])
                sources_used.append("regime_state.json")
            except (OSError, ValueError, KeyError, TypeError):
                pass

        # Also check portfolio history for regime snapshots
        portfolio = state.get("portfolio", {})
        history = portfolio.get("history", [])
        for entry in history:
            regime = entry.get("regime", "")
            if regime:
                regimes_seen.add(regime)
        if history:
            sources_used.append("portfolio.history")

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
                sources_used.append("regime_log.json")
            except (OSError, ValueError):
                pass

        n_regimes = len(regimes_seen)
        desc = self.criteria["regime_coverage"]["description"]
        if n_regimes == 0 and not sources_used:
            desc = (
                f"{desc} — no producer artifacts "
                "(regime_state.json / regime_log.json / portfolio.history empty)"
            )

        return CheckResult(
            name="regime_coverage",
            passed=n_regimes >= required,
            value=n_regimes,
            required=required,
            description=desc,
        )

    def _count_ensemble_voting_contributors(self) -> tuple[int, str]:
        """Primary SSOT: signals.json ensemble_voting contributing sources.

        Fallback order: ensemble_voting → adaptive non-zero weights →
        ensemble_state.json → orders.jsonl tags.
        """
        candidates: list[Path] = []
        try:
            candidates.append(Path(SIGNALS_JSON))
        except Exception:  # noqa: BLE001
            pass
        candidates.extend(
            [
                Path(PUBLIC_DATA_DIR) / "signals.json",
                Path(DATA_DIR) / "signals.json",
            ]
        )
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            ev = data.get("ensemble_voting")
            if not isinstance(ev, dict):
                continue
            # Prefer explicit contributing count
            csc = ev.get("contributing_source_count")
            if csc is not None:
                try:
                    return int(csc), f"ensemble_voting.contributing_source_count@{path.name}"
                except (TypeError, ValueError):
                    pass
            # configured_source_status: count contributing/active flags
            status_list = ev.get("configured_source_status")
            if isinstance(status_list, list) and status_list:
                n = 0
                for row in status_list:
                    if not isinstance(row, dict):
                        continue
                    if row.get("contributing") or (
                        row.get("active") and float(row.get("effective_weight") or 0) > 0.01
                    ):
                        n += 1
                if n > 0:
                    return n, f"ensemble_voting.configured_source_status@{path.name}"
            # active_weights non-zero mass
            aw = ev.get("active_weights") or ev.get("weights")
            if isinstance(aw, dict) and aw:
                n = sum(1 for w in aw.values() if float(w or 0) > 0.01)
                if n > 0:
                    return n, f"ensemble_voting.active_weights@{path.name}"
        return 0, "none"

    def _check_signal_diversity(self, state: Dict) -> CheckResult:
        """Check that multiple ensemble signals have contributed to decisions.

        Primary SSOT: signals.json ``ensemble_voting`` contributing sources.
        Never return silent 0 solely because ensemble_state.json is empty when
        live ensemble_voting shows active arms.
        """
        required = self.criteria["signal_diversity"]["value"]
        desc = self.criteria["signal_diversity"]["description"]

        n_from_ev, ssot = self._count_ensemble_voting_contributors()
        signals_contributing: set[str] = set()

        # Secondary: ensemble voter state for active signals
        ensemble_file = DATA_DIR / "ensemble_state.json"
        if ensemble_file.exists():
            try:
                with open(ensemble_file) as f:
                    data = json.load(f)
                weights = data.get("weights", {})
                for signal, weight in weights.items():
                    if weight > 0.01:  # Non-trivial weight
                        signals_contributing.add(signal)
            except (OSError, ValueError, KeyError, TypeError):
                pass

        # Adaptive weights surface (non-zero adjusted arms)
        adaptive_file = DATA_DIR / "adaptive_weights_state.json"
        if adaptive_file.exists():
            try:
                with open(adaptive_file) as f:
                    data = json.load(f)
                adj = data.get("adjusted_weights") or {}
                for signal, weight in adj.items():
                    if float(weight or 0) > 0.01:
                        signals_contributing.add(signal)
            except (OSError, ValueError, KeyError, TypeError):
                pass

        # Order log for signal attribution
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

        n_legacy = len(signals_contributing)
        n_signals = max(n_from_ev, n_legacy)
        source_note = ssot if n_from_ev >= n_legacy else "ensemble_state|orders|adaptive"

        return CheckResult(
            name="signal_diversity",
            passed=n_signals >= required,
            value=n_signals,
            required=required,
            description=f"{desc} (ssot={source_note})",
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

        # Get Sharpe from the same source as _check_sharpe (raw, no silent zero)
        summary = state.get("paper_trading_summary", {})
        summary_sharpe = self._summary_sharpe(state)
        if summary_sharpe is not None:
            sharpe = summary_sharpe
            n_days = int(summary.get("days_tracked", 0) or 0)
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

        desc = self.criteria["sharpe_ci_lower"]["description"]
        plausible, note = self._sharpe_plausibility(sharpe)
        if not plausible:
            if note:
                desc = f"{desc} — {note}; CI not trusted on implausible point estimate"
            return CheckResult(
                name="sharpe_ci_lower",
                passed=False,
                # Publish raw point estimate (not zero) so operators see the artifact
                value=round(sharpe, 4),
                required=required,
                description=desc,
            )

        if n_days < 5:
            return CheckResult(
                name="sharpe_ci_lower",
                passed=False,
                value=0.0,
                required=required,
                description=desc,
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
            description=desc,
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
    promote_path = checklist.write_promote_to_live_if_ready(results)
    if promote_path:
        logger.info(f"  Promote marker written: {promote_path}")
    
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
    from src.utils.log_config import configure_logging

    configure_logging()
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
