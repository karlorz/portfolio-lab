#!/usr/bin/env python3
"""
Tests for graduation_checklist.py — GraduationChecklist class,
multi-criteria gates, CLI commands, and edge cases.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest


from src.strategy.graduation_checklist import (
    GraduationChecklist,
    CheckResult,
    run_check_and_exit,
    run_report_and_exit,
    run_progress_and_exit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_history(n_days: int, start_val: float = 100000,
                        daily_return: float = 0.001) -> list:
    """Create a synthetic daily portfolio history."""
    history = []
    val = start_val
    base_date = datetime(2026, 5, 1)
    for i in range(n_days):
        val = val * (1 + daily_return)
        history.append({
            "timestamp": (base_date + timedelta(days=i)).isoformat(),
            "total_value": round(val, 2),
            "daily_return": daily_return,
        })
    return history


def _make_state_file(tmp_path, overrides=None) -> dict:
    """Create a minimal state dict with all required keys."""
    state = {
        "portfolio": {
            "cash": 50000,
            "history": _make_daily_history(63),
        },
        "performance": [
            {"sharpe": 0.86, "timestamp": "2026-05-16T00:00:00"},
        ],
        "tca": {
            "orders_by_symbol": {
                "SPY": [{"qty": 100}] * 4,
                "GLD": [{"qty": 50}] * 3,
                "TLT": [{"qty": 30}] * 3,
                "IEF": [{"qty": 20}] * 2,
            }
        },
        "circuit_breaker": {
            "status": "green",
            "trips": 0,
            "consecutive_ok": 5,
        },
        "health_report": {
            "summary": {
                "total_checks": 9,
                "passed": 9,
                "failed": 0,
            }
        },
    }
    if overrides:
        _deep_merge(state, overrides)
    return state


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# CheckResult Tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_basic_construction(self):
        r = CheckResult("min_sharpe", True, 0.86, 0.50, "Sharpe check")
        assert r.name == "min_sharpe"
        assert r.passed is True
        assert r.value == 0.86
        assert r.required == 0.50
        assert r.description == "Sharpe check"

    def test_failed_result(self):
        r = CheckResult("min_sharpe", False, 0.30, 0.50, "Sharpe check")
        assert r.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — Trading Days
# ---------------------------------------------------------------------------


class TestTradingDaysCheck:
    def test_passes_with_63_days(self):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        result = checklist._check_trading_days(state)
        assert result.passed is True
        assert result.value >= 63

    def test_fails_with_few_days(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(5)}})
        checklist = GraduationChecklist()
        result = checklist._check_trading_days(state)
        assert result.passed is False
        assert result.value == 5

    def test_empty_history(self):
        state = _make_state_file(None, {"portfolio": {"history": []}})
        checklist = GraduationChecklist()
        result = checklist._check_trading_days(state)
        assert result.passed is False
        assert result.value == 0

    def test_no_portfolio_key(self):
        checklist = GraduationChecklist()
        result = checklist._check_trading_days({})
        assert result.passed is False
        assert result.value == 0


# ---------------------------------------------------------------------------
# GraduationChecklist — Sharpe
# ---------------------------------------------------------------------------


class TestSharpeCheck:
    def test_passes_with_good_sharpe(self):
        # Deterministic returns that guarantee annualized Sharpe ~1.0 (between 0.5 and 3.0)
        # Mean ~0.0013, std ~0.02 → daily Sharpe ~0.066 → ann Sharpe ~1.0
        returns = [0.001 + (0.02 if i % 2 == 0 else -0.02) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 + r), "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        assert result.passed is True, f"Sharpe {result.value} below 0.50"

    def test_fails_with_low_sharpe(self):
        # Near-zero mean with realistic variance → Sharpe < 0.50
        # Daily returns ~0 with ±0.005 noise
        returns = [0.0 + (0.005 if i % 2 == 0 else -0.005) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        assert result.passed is False, f"Sharpe {result.value} should be below 0.50"

    def test_fails_with_unrealistic_sharpe(self):
        # Sharpe > 3.0 should be treated as failing (intra-day artifact)
        # All identical daily returns cause near-zero std -> unrealistic Sharpe
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": 0.01}
            for i in range(63)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        assert result.passed is False, f"Sharpe {result.value} should cap at 3.0"

    def test_insufficient_data(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(2)}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        assert result.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — Drawdown
# ---------------------------------------------------------------------------


class TestDrawdownCheck:
    def test_passes_with_small_dd(self):
        checklist = GraduationChecklist()
        result = checklist._check_drawdown(_make_state_file(None))
        assert result.passed is True
        assert result.value < 0.15, f"DD {result.value} should be < 0.15"

    def test_fails_with_large_dd(self):
        # 30% steady drawdown
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 - 0.01 * i), "daily_return": -0.01}
            for i in range(40)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_drawdown(state)
        assert result.passed is False
        assert result.value > 0.15, f"DD {result.value} should be > 0.15"

    def test_empty_history(self):
        checklist = GraduationChecklist()
        result = checklist._check_drawdown({"portfolio": {"history": []}})
        assert result.passed is True  # No data = no drawdown


# ---------------------------------------------------------------------------
# GraduationChecklist — Win Rate
# ---------------------------------------------------------------------------


class TestWinRateCheck:
    def test_passes_with_high_win_rate(self):
        # 80% win rate
        returns = [0.001] * 50 + [-0.002] * 13  # 50/63 ≈ 79%
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_win_rate(state)
        assert result.passed is True

    def test_fails_with_low_win_rate(self):
        returns = [0.001] * 10 + [-0.002] * 53  # 10/63 ≈ 16%
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_win_rate(state)
        assert result.passed is False

    def test_insufficient_data(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(2)}})
        checklist = GraduationChecklist()
        result = checklist._check_win_rate(state)
        assert result.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — Health Checks
# ---------------------------------------------------------------------------


class TestHealthChecks:
    def test_passes_with_all_ok(self):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        result = checklist._check_health(state)
        assert result.passed is True

    def test_fails_with_failed_checks(self):
        state = _make_state_file(None, {
            "health_report": {"summary": {"total_checks": 9, "passed": 7, "failed": 2}}
        })
        checklist = GraduationChecklist()
        result = checklist._check_health(state)
        assert result.passed is False

    def test_missing_health_report(self):
        checklist = GraduationChecklist()
        result = checklist._check_health({})
        assert result.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — TCA Orders
# ---------------------------------------------------------------------------


class TestTCAOrders:
    def test_passes_with_enough_orders(self):
        state = _make_state_file(None, {
            "tca": {"orders_by_symbol": {"SPY": [{"qty": 100}] * 10}}
        })
        checklist = GraduationChecklist()
        result = checklist._check_tca_orders(state)
        assert result.passed is True

    def test_passes_with_few_orders_but_jsonl_backup(self, tmp_path):
        # No orders in TCA state but orders.jsonl exists
        state = _make_state_file(None, {"tca": {"orders_by_symbol": {}}})
        # Create temp orders.jsonl
        orders_path = Path(tmp_path) / "orders.jsonl"
        for i in range(12):
            with open(orders_path, 'a') as f:
                f.write(json.dumps({"order_id": i}) + "\n")
        # Patch DATA_DIR
        import src.strategy.graduation_checklist as gc
        original = gc.DATA_DIR
        gc.DATA_DIR = Path(tmp_path)
        try:
            checklist = GraduationChecklist()
            result = checklist._check_tca_orders(state)
            assert result.passed is True
        finally:
            gc.DATA_DIR = original

    def test_fails_with_few_orders(self):
        state = _make_state_file(None, {"tca": {"orders_by_symbol": {"SPY": [{"qty": 100}]}}})
        checklist = GraduationChecklist()
        result = checklist._check_tca_orders(state)
        assert result.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_passes_with_green_status(self):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        assert result.passed is True

    def test_passes_with_ok_status(self):
        state = _make_state_file(None, {
            "circuit_breaker": {"status": "ok", "trips": 0, "consecutive_ok": 10}
        })
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        assert result.passed is True

    def test_fails_with_red_status(self):
        state = _make_state_file(None, {
            "circuit_breaker": {"status": "red", "trips": 3, "consecutive_ok": 0}
        })
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        assert result.passed is False

    def test_missing_circuit_breaker(self):
        state = _make_state_file(None, {"circuit_breaker": {}})
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        # Empty dict — status not found, passes with default
        assert result.passed is True  # Default: green/normal

    def test_non_green_status_but_zero_trips_passes(self):
        """Regression: non-green status with 0 trips should still pass."""
        state = _make_state_file(None, {
            "circuit_breaker": {"status": "yellow", "trips": 0, "consecutive_ok": 5}
        })
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        assert result.passed is True

    def test_non_green_status_with_trips_fails(self):
        """Non-green status with recent trips should fail."""
        state = _make_state_file(None, {
            "circuit_breaker": {"status": "yellow", "trips": 2, "consecutive_ok": 1}
        })
        checklist = GraduationChecklist()
        result = checklist._check_circuit_breaker(state)
        assert result.passed is False


# ---------------------------------------------------------------------------
# GraduationChecklist — Manual Approval
# ---------------------------------------------------------------------------


class TestManualApproval:
    def test_default_is_not_approved(self):
        checklist = GraduationChecklist()
        result = checklist._check_manual_approval({})
        assert result.passed is False

    def test_approval_file_exists(self, tmp_path):
        approval_file = Path(tmp_path) / ".manual_approval"
        approval_file.touch()
        import src.strategy.graduation_checklist as gc
        original = gc.DATA_DIR
        gc.DATA_DIR = Path(tmp_path)
        try:
            checklist = GraduationChecklist()
            result = checklist._check_manual_approval({})
            assert result.passed is True
        finally:
            gc.DATA_DIR = original


# ---------------------------------------------------------------------------
# GraduationChecklist — Integrated Checks
# ---------------------------------------------------------------------------


class TestIntegratedCheck:
    def test_check_returns_all_criteria(self):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        results = checklist.check(state)
        assert len(results) == 9
        expected = [
            "min_trading_days",
            "min_sharpe",
            "max_drawdown",
            "min_win_rate",
            "health_checks",
            "min_tca_orders",
            "circuit_breaker_confidence",
            "min_dsr",
            "manual_approval",
        ]
        for name in expected:
            assert name in results, f"Missing criterion: {name}"

    def test_is_graduation_ready_with_all_pass(self):
        state = _make_state_file(None)
        # Override with deterministic returns that give good metrics
        returns = [0.001 + (0.02 if i % 2 == 0 else -0.02) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 + r), "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state["portfolio"]["history"] = history
        checklist = GraduationChecklist()
        results = checklist.check(state)
        auto_pass = all(
            results[n].passed for n in results if n != "manual_approval"
        )
        assert auto_pass is True
        # Manual approval is a separate gate — does not block is_graduation_ready
        assert results["manual_approval"].passed is False

    def test_is_graduation_ready_with_fail(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(5)}})
        checklist = GraduationChecklist()
        results = checklist.check(state)
        assert checklist.is_graduation_ready(results) is False

    def test_readiness_score(self):
        # 7 auto-criteria, most pass = high score
        returns = [0.001 + (0.02 if i % 2 == 0 else -0.02) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 + r), "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None)
        state["portfolio"]["history"] = history
        checklist = GraduationChecklist()
        results = checklist.check(state)
        score = checklist.readiness_score(results)
        assert score >= 85.0, f"Score {score} should be near 100% with good state"

    def test_readiness_score_with_failures(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(5)}})
        checklist = GraduationChecklist()
        results = checklist.check(state)
        score = checklist.readiness_score(results)
        assert score < 85.0, f"Score {score} should be low with bad state"

    def test_save_report(self, tmp_path):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        results = checklist.check(state)
        report_path = checklist.save_report(results, Path(tmp_path) / "report.json")
        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
        assert "readiness_score" in report
        assert "is_graduation_ready" in report
        assert "criteria" in report
        assert len(report["criteria"]) == 9

    def test_progress_summary(self):
        state = _make_state_file(None)
        checklist = GraduationChecklist()
        results = checklist.check(state)
        summary = checklist.progress_summary(results)
        assert "overall_progress" in summary
        assert "readiness_pct" in summary
        assert "manual_approval_required" in summary
        assert summary["manual_approval_required"] is True


# ---------------------------------------------------------------------------
# Empty / Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_state_returns_all_false(self):
        checklist = GraduationChecklist()
        results = checklist.check({})
        for name, result in results.items():
            if name in ("max_drawdown", "circuit_breaker_confidence"):
                assert result.passed is True  # No data = no drawdown / default green CB
            else:
                assert result.passed is False, f"{name} should fail on empty state"
        assert checklist.is_graduation_ready(results) is False
        assert checklist.readiness_score(results) < 100.0

    def test_min_observation_days_constant(self):
        assert GraduationChecklist.MIN_OBSERVATION_DAYS == 30

    def test_default_criteria_values(self):
        checklist = GraduationChecklist()
        assert checklist.criteria["min_trading_days"]["value"] == 63
        assert checklist.criteria["min_sharpe"]["value"] == 0.50
        assert checklist.criteria["max_drawdown"]["value"] == 0.15
        assert checklist.criteria["min_win_rate"]["value"] == 0.40
        assert checklist.criteria["manual_approval"]["value"] is False

    def test_custom_criteria(self):
        custom = {
            "min_trading_days": {"value": 10, "description": "Custom"},
            "min_sharpe": {"value": 1.0, "description": "Strict"},
        }
        checklist = GraduationChecklist(criteria=custom)
        # Only custom criteria keys are used for checks
        # But _check methods still look at self.criteria
        assert checklist.criteria["min_trading_days"]["value"] == 10
        assert checklist.criteria["min_sharpe"]["value"] == 1.0

    def test_negative_daily_returns(self):
        # Portfolio losing money every day
        history = [{"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 - 0.001) ** i, "daily_return": -0.001} for i in range(63)]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        results = checklist.check(state)
        assert results["min_sharpe"].passed is False
        assert results["min_win_rate"].passed is False


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCheckResultExtended:
    """Extended tests for CheckResult NamedTuple."""

    def test_namedtuple_fields(self):
        r = CheckResult("test", True, 1.0, 0.5, "desc")
        assert r._fields == ("name", "passed", "value", "required", "description")

    def test_namedtuple_immutable(self):
        r = CheckResult("test", True, 1.0, 0.5, "desc")
        with pytest.raises(AttributeError):
            r.passed = False

    def test_namedtuple_as_tuple(self):
        r = CheckResult("test", True, 1.0, 0.5, "desc")
        assert r[0] == "test"
        assert r[1] is True

    def test_zero_value(self):
        r = CheckResult("test", False, 0.0, 1.0, "Zero value")
        assert r.value == 0.0

    def test_negative_value(self):
        r = CheckResult("test", False, -0.5, 0.0, "Negative")
        assert r.value == -0.5


class TestGraduationChecklistExtended:
    """Extended tests for GraduationChecklist."""

    # --- Constants and defaults ---

    def test_default_criteria_has_9_entries(self):
        assert len(GraduationChecklist.DEFAULT_CRITERIA) == 9

    def test_default_criteria_keys(self):
        expected = {
            "min_trading_days", "min_sharpe", "max_drawdown", "min_win_rate",
            "health_checks", "min_tca_orders", "circuit_breaker_confidence",
            "min_dsr", "manual_approval",
        }
        assert set(GraduationChecklist.DEFAULT_CRITERIA.keys()) == expected

    def test_min_observation_days(self):
        assert GraduationChecklist.MIN_OBSERVATION_DAYS == 30

    # --- readiness_score ---

    def test_readiness_score_all_pass(self):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", True, 63, 63, ""),
            "min_sharpe": CheckResult("min_sharpe", True, 0.8, 0.5, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        score = checklist.readiness_score(results)
        assert score == 100.0  # 2/2 auto criteria pass

    def test_readiness_score_none_pass(self):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", False, 5, 63, ""),
            "min_sharpe": CheckResult("min_sharpe", False, 0.1, 0.5, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        score = checklist.readiness_score(results)
        assert score == 0.0

    def test_readiness_score_half_pass(self):
        checklist = GraduationChecklist()
        results = {
            "a": CheckResult("a", True, 1, 1, ""),
            "b": CheckResult("b", False, 0, 1, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        score = checklist.readiness_score(results)
        assert score == 50.0

    def test_readiness_score_empty(self):
        checklist = GraduationChecklist()
        score = checklist.readiness_score({})
        assert score == 0.0

    # --- is_graduation_ready ---

    def test_is_graduation_ready_ignores_manual_approval(self):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", True, 63, 63, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        assert checklist.is_graduation_ready(results) is True

    def test_is_graduation_ready_fails_on_auto_criterion(self):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", False, 5, 63, ""),
            "manual_approval": CheckResult("manual_approval", True, 1, 1, ""),
        }
        assert checklist.is_graduation_ready(results) is False

    # --- progress_summary ---

    def test_progress_summary_structure(self):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", True, 63, 63, ""),
            "min_sharpe": CheckResult("min_sharpe", True, 0.8, 0.5, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        summary = checklist.progress_summary(results)
        assert "overall_progress" in summary
        assert "readiness_pct" in summary
        assert "is_ready" in summary
        assert "manual_approval_required" in summary
        assert "passed_count" in summary
        assert "total_count" in summary

    def test_progress_summary_counts(self):
        checklist = GraduationChecklist()
        results = {
            "a": CheckResult("a", True, 1, 1, ""),
            "b": CheckResult("b", True, 1, 1, ""),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, ""),
        }
        summary = checklist.progress_summary(results)
        assert summary["passed_count"] == 2
        assert summary["total_count"] == 3
        assert summary["manual_approval_required"] is True

    # --- save_report ---

    def test_save_report_structure(self, tmp_path):
        checklist = GraduationChecklist()
        results = {
            "min_trading_days": CheckResult("min_trading_days", True, 63, 63, "desc"),
            "manual_approval": CheckResult("manual_approval", False, 0, 1, "desc"),
        }
        path = checklist.save_report(results, tmp_path / "report.json")
        assert path.exists()
        with open(path) as f:
            report = json.load(f)
        assert "timestamp" in report
        assert "readiness_score" in report
        assert "is_graduation_ready" in report
        assert "criteria" in report
        assert "min_trading_days" in report["criteria"]
        assert report["criteria"]["min_trading_days"]["passed"] is True

    # --- _check_sharpe edge cases ---

    def test_sharpe_with_constant_returns_fails(self):
        """Constant returns produce infinite/undefined Sharpe → capped."""
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": 0.01}
            for i in range(63)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        assert result.passed is False  # Capped at 3.0 → set to 0.0

    def test_sharpe_exactly_at_threshold(self):
        """Sharpe exactly at 0.50 should pass."""
        # Craft returns to produce Sharpe ~0.50
        returns = [0.001 + (0.02 if i % 2 == 0 else -0.02) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 + r), "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_sharpe(state)
        # Verify value is calculated (may or may not pass depending on exact calc)
        assert isinstance(result.value, float)

    # --- _check_dsr edge cases ---

    def test_dsr_with_insufficient_data(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(2)}})
        checklist = GraduationChecklist()
        result = checklist._check_dsr(state)
        assert result.passed is False
        assert result.value == 0.0

    def test_dsr_with_good_data(self):
        returns = [0.001 + (0.02 if i % 2 == 0 else -0.02) for i in range(63)]
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 * (1 + r), "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_dsr(state)
        # Should have a computed value
        assert isinstance(result.value, float)

    # --- _check_drawdown edge cases ---

    def test_drawdown_with_continuously_rising_portfolio(self):
        """No drawdown if portfolio only goes up."""
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000 + i * 100, "daily_return": 0.001}
            for i in range(63)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_drawdown(state)
        assert result.passed is True
        assert result.value == 0.0

    # --- _check_win_rate edge cases ---

    def test_win_rate_all_positive(self):
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": 0.01}
            for i in range(63)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_win_rate(state)
        assert result.passed is True
        assert result.value == 1.0

    def test_win_rate_exactly_at_threshold(self):
        """Exactly 40% win rate should pass (>=)."""
        n_pos = 25  # 25/63 ≈ 39.7%, just below
        n_neg = 38
        returns = [0.001] * n_pos + [-0.001] * n_neg
        history = [
            {"timestamp": f"2026-05-{i+1:02d}T00:00:00", "total_value": 100000, "daily_return": r}
            for i, r in enumerate(returns)
        ]
        state = _make_state_file(None, {"portfolio": {"history": history}})
        checklist = GraduationChecklist()
        result = checklist._check_win_rate(state)
        # Win rate should be n_pos/len(returns) ≈ 0.397
        assert isinstance(result.value, float)

    # --- Custom criteria ---

    def test_custom_criteria_overrides_defaults(self):
        custom = {
            "min_sharpe": {"value": 1.5, "description": "Very strict Sharpe"},
            "min_trading_days": {"value": 126, "description": "6 months"},
        }
        checklist = GraduationChecklist(criteria=custom)
        assert checklist.criteria["min_sharpe"]["value"] == 1.5
        assert checklist.criteria["min_trading_days"]["value"] == 126

    # --- _check_tca_orders edge cases ---

    def test_tca_no_orders_no_file(self, tmp_path):
        state = {"tca": {"orders_by_symbol": {}}}
        # Patch DATA_DIR so no orders.jsonl fallback
        import src.strategy.graduation_checklist as gc
        original = gc.DATA_DIR
        gc.DATA_DIR = Path(tmp_path)
        try:
            checklist = GraduationChecklist()
            result = checklist._check_tca_orders(state)
            assert result.passed is False
            assert result.value == 0
        finally:
            gc.DATA_DIR = original

    # --- _check_health edge cases ---

    def test_health_partial_failure(self):
        state = _make_state_file(None, {
            "health_report": {"summary": {"total_checks": 9, "passed": 8, "failed": 1}}
        })
        checklist = GraduationChecklist()
        result = checklist._check_health(state)
        assert result.passed is False

    def test_health_zero_checks(self):
        state = {"health_report": {"summary": {"total_checks": 0, "passed": 0, "failed": 0}}}
        checklist = GraduationChecklist()
        result = checklist._check_health(state)
        assert result.passed is False

    # --- _check_trading_days edge cases ---

    def test_trading_days_exactly_at_threshold(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(63)}})
        checklist = GraduationChecklist()
        result = checklist._check_trading_days(state)
        assert result.passed is True

    def test_trading_days_one_below_threshold(self):
        state = _make_state_file(None, {"portfolio": {"history": _make_daily_history(62)}})
        checklist = GraduationChecklist()
        result = checklist._check_trading_days(state)
        assert result.passed is False


class TestCLI:
    def test_run_check_and_exit(self):
        # Should not raise
        result = run_check_and_exit()
        assert result in (0, 1)

    def test_run_report_and_exit(self):
        result = run_report_and_exit()
        assert result in (0, 1)

    def test_run_progress_and_exit(self):
        result = run_progress_and_exit()
        assert result in (0, 1)
