"""Unit tests for src.monitor.health_rollup (Item Q49).

Tests cover:
- _check_kill_switch (armed halt, armed warning, cleared, missing file)
- _check_open_incidents (halt critical, open warning, none, missing file, closed rows skipped)
- _status_for_system_rollup (pass-through mapping, FRED advisory de-escalation)
- _compute_system_status (worst-wins, degraded/error mapping, warning set, all-ok, unknown)
- attach_shared_freshness_slis_to_ops_report (mirror lag, graduation CB, execution timeline SLIs)
"""

import json
from pathlib import Path

import pytest

from src.monitor import health_rollup as hr


def test_check_kill_switch_missing_file(tmp_path: Path) -> None:
    result = hr._check_kill_switch(tmp_path)
    assert result["status"] == "ok"
    assert result["enabled"] is False
    assert result["level"] is None


def test_check_kill_switch_cleared(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(
        json.dumps({"enabled": False, "status": "ok"}), encoding="utf-8"
    )
    result = hr._check_kill_switch(tmp_path)
    assert result["status"] == "ok"
    assert result["enabled"] is False


def test_check_kill_switch_armed_halt(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {"enabled": True, "level": "halt", "reason": "test", "incident_id": "INC-1"}
        ),
        encoding="utf-8",
    )
    result = hr._check_kill_switch(tmp_path)
    assert result["status"] == "critical"
    assert result["enabled"] is True
    assert result["level"] == "halt"
    assert result["incident_id"] == "INC-1"


def test_check_kill_switch_armed_warning(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(
        json.dumps({"enabled": True, "level": "warning"}), encoding="utf-8"
    )
    result = hr._check_kill_switch(tmp_path)
    assert result["status"] == "warning"
    assert result["enabled"] is True


def test_check_kill_switch_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text("{not json", encoding="utf-8")
    result = hr._check_kill_switch(tmp_path)
    assert result["status"] == "ok"
    assert result["enabled"] is False


def test_check_open_incidents_missing_file(tmp_path: Path) -> None:
    result = hr._check_open_incidents(tmp_path)
    assert result["status"] == "ok"
    assert result["open_count"] == 0
    assert result["incidents"] == []


def test_check_open_incidents_none(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )
    result = hr._check_open_incidents(tmp_path)
    assert result["status"] == "ok"
    assert result["open_count"] == 0


def test_check_open_incidents_open_warning(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 1,
                "incidents": [
                    {"incident_id": "I1", "state": "open", "severity": "p1"}
                ],
            }
        ),
        encoding="utf-8",
    )
    result = hr._check_open_incidents(tmp_path)
    assert result["status"] == "warning"
    assert result["open_count"] == 1
    assert result["incidents"][0]["incident_id"] == "I1"


def test_check_open_incidents_halt_critical(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 1,
                "incidents": [
                    {"incident_id": "H1", "state": "open", "kill_switch_level": "halt"}
                ],
            }
        ),
        encoding="utf-8",
    )
    result = hr._check_open_incidents(tmp_path)
    assert result["status"] == "critical"
    assert result["open_count"] == 1


def test_check_open_incidents_closed_rows_skipped(tmp_path: Path) -> None:
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 0,
                "incidents": [
                    {"incident_id": "C1", "state": "closed"},
                    {"incident_id": "R1", "status": "resolved"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = hr._check_open_incidents(tmp_path)
    assert result["status"] == "ok"
    assert result["incidents"] == []


def test_status_for_system_rollup_pass_through() -> None:
    assert hr._status_for_system_rollup("scheduler_status", {"status": "ok"}) == "ok"
    assert hr._status_for_system_rollup("scheduler_status", {"status": "warning"}) == "warning"
    assert hr._status_for_system_rollup("scheduler_status", {"status": "critical"}) == "critical"
    assert hr._status_for_system_rollup("scheduler_status", {"status": "stale"}) == "stale"
    # Unknown status maps through
    assert hr._status_for_system_rollup("scheduler_status", {"status": "bogus"}) == "bogus"


def test_status_for_system_rollup_fred_advisories() -> None:
    # Non-blocking ready FRED advisory must not force warning
    check = {"status": "warning", "ready": True, "blocking": False}
    assert hr._status_for_system_rollup("fred_readiness", check) == "ok"
    # Blocking or not-ready keeps raw status
    check2 = {"status": "warning", "ready": False, "blocking": True}
    assert hr._status_for_system_rollup("fred_readiness", check2) == "warning"
    # Empty FRED-MD cache without API key is not blocking
    check3 = {"status": "empty", "api_key_configured": False}
    assert hr._status_for_system_rollup("fred_md_cache", check3) == "ok"
    check4 = {"status": "empty", "api_key_configured": True}
    assert hr._status_for_system_rollup("fred_md_cache", check4) == "empty"


def test_compute_system_status_all_ok() -> None:
    checks = {
        "kill_switch": {"status": "ok"},
        "open_incidents": {"status": "ok"},
        "data_freshness": {"status": "ok"},
    }
    # Circuit breaker contributes to the rollup; all-ok requires it to be ok too
    assert hr._compute_system_status(checks, {"status": "ok"}) == "ok"
    # An empty/unknown circuit keeps the rollup at unknown (not falsely ok)
    assert hr._compute_system_status(checks, {}) == "unknown"


def test_compute_system_status_worst_wins_critical() -> None:
    checks = {
        "kill_switch": {"status": "ok"},
        "open_incidents": {"status": "warning"},
        "data_freshness": {"status": "ok"},
    }
    # Circuit breaker carries the critical and must win
    circuit = {"status": "critical"}
    assert hr._compute_system_status(checks, circuit) == "critical"


def test_compute_system_status_critical_in_checks() -> None:
    checks = {
        "kill_switch": {"status": "critical"},
        "data_freshness": {"status": "ok"},
    }
    assert hr._compute_system_status(checks, {}) == "critical"


def test_compute_system_status_degraded_mapping() -> None:
    checks = {
        "kill_switch": {"status": "ok"},
        "open_incidents": {"status": "missing"},
    }
    assert hr._compute_system_status(checks, {}) == "degraded"
    checks2 = {
        "kill_switch": {"status": "ok"},
        "open_incidents": {"status": "error"},
    }
    assert hr._compute_system_status(checks2, {}) == "degraded"


def test_compute_system_status_warning_set() -> None:
    for status in ("stale", "empty", "degraded", "warning", "unavailable"):
        checks = {
            "kill_switch": {"status": "ok"},
            "open_incidents": {"status": status},
        }
        assert hr._compute_system_status(checks, {}) == "warning", status


def test_compute_system_status_unknown() -> None:
    checks = {"kill_switch": {"status": "ok"}, "open_incidents": {"status": "weird"}}
    assert hr._compute_system_status(checks, {}) == "unknown"
    assert hr._compute_system_status({}, {}) == "unknown"


def test_attach_shared_freshness_slis_to_ops_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Isolate the module-level PUBLIC_DATA_DIR fallback so an ambient
    # repo/live rebalance_health.json cannot override the supplied tmp dir.
    monkeypatch.setattr(hr, "PUBLIC_DATA_DIR", data_dir)
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 12,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    report = {"status": "ok", "checks": {}}
    result = hr.attach_shared_freshness_slis_to_ops_report(
        report, data_dir=data_dir
    )

    # Mirror lag SLI stamped
    assert result["repo_public_mirror_lag_status"] == "ok"
    assert result["repo_public_mirror_lag_badge"] == "lagging=0/12"
    assert result["repo_public_mirror_lag"]["status"] == "ok"
    # Graduation CB SSOT missing -> honest source
    assert result["graduation_circuit_breaker"]["graduation_cb_source"] == "missing"
    # Execution timeline panel missing -> unknown + source missing
    assert result["rebalance_execution_timeline_status"] == "unknown"
    assert result["rebalance_execution_timeline"]["source"] == "missing"


def test_attach_shared_freshness_slis_graduation_cb_ssot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps(
            {
                "consecutive_ok": 3,
                "status": "healthy",
                "signal_health_blocked": False,
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 12,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    result = hr.attach_shared_freshness_slis_to_ops_report(
        {"status": "ok"}, data_dir=data_dir
    )
    cb = result["graduation_circuit_breaker"]
    assert cb["graduation_cb_source"] == "disk_ssot"
    assert cb["consecutive_ok"] == 3
    assert cb["status"] == "healthy"


def test_attach_shared_freshness_slis_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("probe down")),
    )

    result = hr.attach_shared_freshness_slis_to_ops_report(
        {"status": "ok"}, data_dir=data_dir
    )
    # Probe failure must not kill the report; lag status is disclosed as unknown
    assert result["repo_public_mirror_lag_status"] == "unknown"
    assert result["status"] == "ok"