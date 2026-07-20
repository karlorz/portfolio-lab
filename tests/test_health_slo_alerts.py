"""Unit tests for pure critical/warning health → operator alert projection."""

from src.dashboard.health_slo_alerts import (
    HEALTH_SLO_ALERT_TYPE,
    build_health_slo_alerts,
    critical_health_requires_alert,
    warning_health_requires_alert,
)


def test_build_health_slo_alerts_critical_alpaca_fixture():
    as_of = "2026-07-07T12:00:00"
    health = {
        "system_status": "critical",
        "generated_at": as_of,
        "data_pipeline_slo": {
            "status": "critical",
            "top_dimension": "alpaca_feed_entitlement",
            "dimensions": {
                "alpaca_feed_entitlement": {
                    "status": "critical",
                    "policy_decision": "reject",
                    "reason": "missing_entitlement",
                },
            },
            "runbook": {
                "top_cause": {
                    "code": "missing_entitlement",
                    "reason": "missing_entitlement",
                    "action": "Restore Alpaca feed entitlement before live routing.",
                },
            },
        },
    }
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["type"] == HEALTH_SLO_ALERT_TYPE
    assert alert["level"] == "error"
    assert alert["top_dimension"] == "alpaca_feed_entitlement"
    assert alert["reason"] == "missing_entitlement"
    assert alert["policy_decision"] == "reject"
    assert alert["timestamp"] == as_of
    assert "missing_entitlement" in alert["message"]


def test_critical_system_status_alone_emits_alert():
    health = {
        "system_status": "critical",
        "generated_at": "2026-07-07T12:00:00",
        "data_pipeline_slo": {"status": "healthy"},
    }
    assert critical_health_requires_alert(health) is True
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["system_status"] == "critical"


def test_critical_slo_alone_emits_alert():
    health = {
        "system_status": "warning",
        "generated_at": "2026-07-07T12:00:00",
        "data_pipeline_slo": {
            "status": "critical",
            "top_dimension": "data_quality",
            "runbook": {"top_cause": {"code": "stale_prices", "reason": "stale_latest_dates"}},
        },
    }
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["data_pipeline_slo_status"] == "critical"
    assert alerts[0]["reason"] == "stale_latest_dates"


def test_healthy_health_emits_nothing():
    health = {
        "system_status": "healthy",
        "data_pipeline_slo": {"status": "healthy"},
    }
    assert critical_health_requires_alert(health) is False
    assert build_health_slo_alerts(health) == []


def test_none_and_invalid_health_emits_nothing():
    assert build_health_slo_alerts(None) == []
    assert build_health_slo_alerts("not-a-mapping") == []  # type: ignore[arg-type]


def test_warning_system_status_emits_warning_alert():
    health = {
        "system_status": "warning",
        "generated_at": "2026-07-20T12:00:00",
        "scheduler_status": "degraded",
        "signal_health": {"status": "degraded", "healthy": 0, "degraded": 6},
        "data_pipeline_slo": {"status": "ok"},
    }
    assert critical_health_requires_alert(health) is False
    assert warning_health_requires_alert(health) is True
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "warning"
    assert alerts[0]["type"] == HEALTH_SLO_ALERT_TYPE
    assert alerts[0]["system_status"] == "warning"
    assert alerts[0]["scheduler_status"] == "degraded"
    assert "warning" in alerts[0]["message"].lower() or "degraded" in alerts[0]["message"].lower()


def test_degraded_system_status_emits_warning_alert():
    health = {
        "system_status": "degraded",
        "generated_at": "2026-07-20T12:00:00",
        "data_pipeline_slo": {"status": "healthy"},
    }
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "warning"


def test_critical_wins_over_warning_fields():
    """Critical SLO still emits error even if system_status is only warning."""
    health = {
        "system_status": "warning",
        "generated_at": "2026-07-20T12:00:00",
        "data_pipeline_slo": {
            "status": "critical",
            "top_dimension": "data_quality",
            "runbook": {"top_cause": {"code": "stale_prices", "reason": "stale_latest_dates"}},
        },
    }
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
