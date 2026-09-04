"""Unit tests for src.dashboard.kill_authority (Item Q53).

Tests cover:
- load_kill_switch_payload (present, missing, invalid JSON)
- is_kill_execution_blocked (enabled true, false, missing/none)
- load_open_incidents_summary (halt critical, open warning, closed/resolved filtered, missing file)
- project_kill_switch_fields (halt critical, active warning, cleared ok, none/empty)
- project_compact_kill_fields (monitor checks vs public payload)
- elevate_system_status_for_kill (various current statuses, kill levels, open incident statuses)
- _alert_level_for_kill_payload (warning, restrict, halt, fallback)
- build_kill_switch_alert (enabled with human message vs reason, disabled returns None)
- allocation_roles_under_kill (execution blocked modifications, disabled passthrough)
"""

import json
from pathlib import Path

from src.dashboard import kill_authority as ka


def test_load_kill_switch_payload(tmp_path: Path) -> None:
    # Missing file
    assert ka.load_kill_switch_payload(tmp_path) is None

    # Invalid JSON
    (tmp_path / "kill_switch.json").write_text("{bad json", encoding="utf-8")
    assert ka.load_kill_switch_payload(tmp_path) is None

    # Valid payload
    valid = {"enabled": True, "level": "halt", "reason": "test"}
    (tmp_path / "kill_switch.json").write_text(json.dumps(valid), encoding="utf-8")
    assert ka.load_kill_switch_payload(tmp_path) == valid


def test_is_kill_execution_blocked() -> None:
    assert ka.is_kill_execution_blocked({"enabled": True}) is True
    assert ka.is_kill_execution_blocked({"enabled": False}) is False
    assert ka.is_kill_execution_blocked({}) is False
    assert ka.is_kill_execution_blocked(None) is False


def test_load_open_incidents_summary(tmp_path: Path) -> None:
    # Missing file
    summary_missing = ka.load_open_incidents_summary(tmp_path)
    assert summary_missing["status"] == "ok"
    assert summary_missing["open_count"] == 0
    assert summary_missing["incidents"] == []

    # Filter closed / resolved rows
    (tmp_path / "incidents.json").write_text(
        json.dumps({
            "incidents": [
                {"id": "I1", "state": "closed"},
                {"id": "I2", "status": "resolved"},
                {"id": "I3", "state": "open", "message": "open item"},
            ]
        }),
        encoding="utf-8",
    )
    summary_open = ka.load_open_incidents_summary(tmp_path)
    assert summary_open["status"] == "warning"
    assert summary_open["open_count"] == 1
    assert len(summary_open["incidents"]) == 1
    assert summary_open["incidents"][0]["incident_id"] == "I3"

    # Halt level incident elevates status to critical
    (tmp_path / "incidents.json").write_text(
        json.dumps({
            "incidents": [
                {"id": "I4", "state": "open", "kill_switch_level": "halt"},
            ]
        }),
        encoding="utf-8",
    )
    summary_halt = ka.load_open_incidents_summary(tmp_path)
    assert summary_halt["status"] == "critical"
    assert summary_halt["open_count"] == 1


def test_project_kill_switch_fields() -> None:
    # Empty / None
    empty = ka.project_kill_switch_fields(None)
    assert empty["status"] == "ok"
    assert empty["enabled"] is False
    assert empty["level"] is None

    # Enabled halt
    halt = ka.project_kill_switch_fields({"enabled": True, "level": "halt", "reason": "vol"})
    assert halt["status"] == "critical"
    assert halt["enabled"] is True
    assert halt["level"] == "halt"
    assert halt["reason"] == "vol"

    # Enabled warning
    warn = ka.project_kill_switch_fields({"enabled": True, "level": "warn"})
    assert warn["status"] == "warning"
    assert warn["enabled"] is True

    # Disabled
    cleared = ka.project_kill_switch_fields({"enabled": False, "level": None})
    assert cleared["status"] == "ok"
    assert cleared["enabled"] is False


def test_project_compact_kill_fields() -> None:
    # Top-level public shape
    pub_report = {
        "kill_switch": {"enabled": True, "level": "halt", "reason": "manual"},
        "open_incidents": {"status": "critical", "open_count": 1},
    }
    compact_pub = ka.project_compact_kill_fields(pub_report)
    assert compact_pub["kill_switch_enabled"] is True
    assert compact_pub["kill_switch_level"] == "halt"
    assert compact_pub["kill_switch_reason"] == "manual"
    assert compact_pub["open_incidents_status"] == "critical"
    assert compact_pub["open_incidents_count"] == 1

    # Monitor checks shape
    mon_report = {
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        }
    }
    compact_mon = ka.project_compact_kill_fields(mon_report)
    assert compact_mon["kill_switch_enabled"] is False
    assert compact_mon["open_incidents_count"] == 0


def test_elevate_system_status_for_kill() -> None:
    # Kill halt elevates healthy -> critical
    assert ka.elevate_system_status_for_kill(
        "healthy",
        {"enabled": True, "level": "halt"},
    ) == "critical"

    # Kill warning elevates healthy -> warning
    assert ka.elevate_system_status_for_kill(
        "healthy",
        {"enabled": True, "level": "warning"},
    ) == "warning"

    # Open incidents critical elevates warning -> critical
    assert ka.elevate_system_status_for_kill(
        "warning",
        {"enabled": False},
        {"status": "critical"},
    ) == "critical"

    # Critical stays critical when kill is cleared
    assert ka.elevate_system_status_for_kill(
        "critical",
        {"enabled": False},
        {"status": "ok"},
    ) == "critical"


def test_elevate_system_status_for_kill_additional_permutations() -> None:
    # Open incidents warning elevates healthy -> warning
    assert ka.elevate_system_status_for_kill(
        "healthy",
        {"enabled": False},
        {"status": "warning"},
    ) == "warning"

    # Kill critical status elevates degraded -> critical
    assert ka.elevate_system_status_for_kill(
        "degraded",
        {"enabled": True, "status": "critical"},
    ) == "critical"

    # None current status defaults to healthy and elevates
    assert ka.elevate_system_status_for_kill(
        None,  # type: ignore
        {"enabled": True, "level": "halt"},
    ) == "critical"


def test_allocation_roles_under_kill_non_dict_guards() -> None:
    assert ka.allocation_roles_under_kill("not_a_dict", kill_enabled=True) == "not_a_dict"  # type: ignore
    assert ka.allocation_roles_under_kill({}, kill_enabled=True) == {}
    assert ka.allocation_roles_under_kill({"surfaces": "not_dict"}, kill_enabled=True) == {"surfaces": "not_dict"}  # type: ignore
    assert ka.allocation_roles_under_kill({"surfaces": {}}, kill_enabled=True) == {"surfaces": {}}


def test_build_kill_switch_alert_extra_fields() -> None:
    payload = {
        "enabled": True,
        "level": "restrict",
        "source": "cron_detector",
        "timestamp": "2026-08-17T08:00:00Z",
    }
    alert = ka.build_kill_switch_alert(payload)
    assert alert is not None
    assert alert["level"] == "error"
    assert alert["kill_switch_level"] == "restrict"
    assert alert["source"] == "cron_detector"
    assert alert["timestamp"] == "2026-08-17T08:00:00Z"




def test_alert_level_for_kill_payload() -> None:
    assert ka._alert_level_for_kill_payload({"level": "warning"}) == "warning"
    assert ka._alert_level_for_kill_payload({"level": "warn"}) == "warning"
    assert ka._alert_level_for_kill_payload({"level": "restrict"}) == "error"
    assert ka._alert_level_for_kill_payload({"level": "reduce"}) == "error"
    assert ka._alert_level_for_kill_payload({"level": "halt"}) == "critical"
    assert ka._alert_level_for_kill_payload({"level": "liquidate"}) == "critical"
    assert ka._alert_level_for_kill_payload({"level": "unknown"}) == "error"
    assert ka._alert_level_for_kill_payload({}) == "error"


def test_build_kill_switch_alert() -> None:
    # Disabled or empty returns None
    assert ka.build_kill_switch_alert({}) is None
    assert ka.build_kill_switch_alert({"enabled": False}) is None

    # Enabled with human message preferring human message over reason
    payload = {
        "enabled": True,
        "mode": "auto",
        "level": "halt",
        "reason": "ic_decay_p0",
        "message": "Ensemble IC decay exceeded threshold",
        "incident_id": "INC-123",
        "channel": "slack",
    }
    alert = ka.build_kill_switch_alert(payload)
    assert alert is not None
    assert alert["level"] == "critical"
    assert alert["type"] == "kill_switch"
    assert alert["title"] == "AUTO Kill Switch Triggered"
    assert alert["message"] == "Ensemble IC decay exceeded threshold"
    assert alert["reason"] == "ic_decay_p0"
    assert alert["incident_id"] == "INC-123"
    assert alert["requires_action"] is True

    # Enabled with only reason (no message)
    payload_no_msg = {
        "enabled": True,
        "level": "warning",
        "reason": "vol_spike",
    }
    alert2 = ka.build_kill_switch_alert(payload_no_msg)
    assert alert2 is not None
    assert alert2["level"] == "warning"
    assert alert2["message"] == "vol_spike"


def test_allocation_roles_under_kill() -> None:
    roles = {
        "surfaces": {
            "target_allocations": {
                "live_authoritative": True,
                "role": "authoritative",
                "description": "Primary routing weights",
            }
        }
    }

    # Kill disabled -> returns untouched
    untouched = ka.allocation_roles_under_kill(roles, kill_enabled=False)
    assert untouched["surfaces"]["target_allocations"]["live_authoritative"] is True

    # Kill enabled with halt level
    blocked = ka.allocation_roles_under_kill(roles, kill_enabled=True, kill_level="halt")
    assert blocked["execution_blocked"] is True
    assert blocked["kill_switch_enabled"] is True
    assert blocked["kill_switch_level"] == "halt"

    target = blocked["surfaces"]["target_allocations"]
    assert target["live_authoritative"] is False
    assert target["execution_blocked"] is True
    assert target["role"] == "execution_blocked"
    assert "Order routing blocked by active kill switch (level=halt)" in target["description"]
