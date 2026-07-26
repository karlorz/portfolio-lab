"""GARCH .health_report.json must not false-fail graduation health_checks.

Work item: projects/portfolio-lab/work/2026-07-26-graduation-health-checks-garch-ssot-collision/
"""

from __future__ import annotations


def test_is_ops_health_inventory_false_for_garch_shape() -> None:
    from src.strategy.graduation_checklist import is_ops_health_inventory

    garch = {
        "var_95": -1.46,
        "cvar_95": -2.21,
        "garch_filtered": True,
        "summary": {"passed": 1, "total_checks": 1},
        "checks": {"portfolio_entropy": {"ok": True}},
    }
    assert is_ops_health_inventory(garch) is False

    ops = {"summary": {"total_checks": 9, "passed": 9, "failed": 0}}
    assert is_ops_health_inventory(ops) is True

    tagged = {
        "summary": {
            "passed": 1,
            "total_checks": 1,
            "inventory_role": "garch_risk",
        }
    }
    assert is_ops_health_inventory(tagged) is False


def test_health_checks_uses_cb_when_health_report_is_garch_risk_artifact() -> None:
    """GARCH .health_report.json summary.total_checks=1 must not block CB streak."""
    from src.strategy.graduation_checklist import GraduationChecklist

    state = {
        "circuit_breaker": {
            "status": "green",
            "trips": 0,
            "consecutive_ok": 30,
        },
        "health_report": {
            "timestamp": "2026-07-26T08:57:05.079626+00:00",
            "var_95": -1.46,
            "cvar_95": -2.21,
            "garch_filtered": True,
            "status": "healthy",
            "checks": {
                "portfolio_entropy": {
                    "name": "portfolio_entropy",
                    "status": "good",
                    "ok": True,
                }
            },
            "summary": {"passed": 1, "total_checks": 1},
        },
    }
    r = GraduationChecklist()._check_health(state)
    assert r.passed is True
    assert r.value == 30
    assert r.required == 30


def test_health_checks_ops_inventory_without_consecutive_still_fails_ao() -> None:
    """Batch AO: real ops inventory with total_checks but no consecutive_* stays fail-closed."""
    from src.strategy.graduation_checklist import GraduationChecklist

    state = {
        "circuit_breaker": {"status": "green", "trips": 0, "consecutive_ok": 30},
        "health_report": {
            "summary": {
                "total_checks": 9,
                "passed": 9,
                "failed": 0,
            }
        },
    }
    r = GraduationChecklist()._check_health(state)
    assert r.passed is False
    assert r.value == 0


def test_health_checks_ops_inventory_with_consecutive_passes() -> None:
    from src.strategy.graduation_checklist import GraduationChecklist

    state = {
        "health_report": {
            "summary": {
                "total_checks": 9,
                "passed": 9,
                "failed": 0,
                "consecutive_passing_days": 30,
            }
        }
    }
    r = GraduationChecklist()._check_health(state)
    assert r.passed is True
    assert r.value == 30


def test_health_checks_still_blocks_on_signal_health_zero_of_n() -> None:
    from src.strategy.graduation_checklist import GraduationChecklist

    state = {
        "circuit_breaker": {"status": "green", "trips": 0, "consecutive_ok": 30},
        "health_report": {
            "var_95": -1.0,
            "garch_filtered": True,
            "summary": {"passed": 1, "total_checks": 1},
            "signal_health": {
                "status": "degraded",
                "summary": {
                    "healthy": 0,
                    "degraded": 8,
                    "unhealthy": 1,
                    "total_tracked": 9,
                },
            },
        },
    }
    r = GraduationChecklist()._check_health(state)
    assert r.passed is False
    assert "signal_health" in r.description.lower() or "blocked" in r.description.lower()
