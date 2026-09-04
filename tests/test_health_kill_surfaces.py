"""Unit tests for src.monitor.health_kill_surfaces (Item Q46).

Tests cover:
- health_ops_path (default vs HEALTH_OPS_PATH env override)
- _project_public_kill_fields
- _elevate_public_system_status
- enforce_worst_wins_system_status
- _is_monitor_health_report
- load_ops_monitor_report
- _patch_monitor_report_kill_open
- write_health_generation & commit_public_index
- reconcile_monitor_health_with_disk_ssot
- project_disk_kill_open_to_all_surfaces
- _disk_kill_ssot_is_clear & _disk_kill_and_open_incidents
"""

import json
from pathlib import Path

import pytest

from src.monitor import health_kill_surfaces as hks


def test_health_ops_path_default_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEALTH_OPS_PATH", raising=False)
    monkeypatch.setattr(hks, "PUBLIC_DATA_DIR", tmp_path / "public")
    assert hks.health_ops_path() == tmp_path / "public" / "health_ops.json"

    custom = tmp_path / "custom" / "health_ops.json"
    monkeypatch.setenv("HEALTH_OPS_PATH", str(custom))
    assert hks.health_ops_path() == custom


def test_project_public_kill_fields() -> None:
    report = {
        "status": "warning",
        "timestamp": "2026-08-17T08:00:00Z",
        "scope": "custom_scope",
        "checks": {
            "kill_switch": {"enabled": True, "status": "critical"},
            "open_incidents": {"status": "warning", "open_count": 1},
        },
    }
    projected = hks._project_public_kill_fields(report)
    assert projected["kill_switch"] == {"enabled": True, "status": "critical"}
    assert projected["open_incidents"] == {"status": "warning", "open_count": 1}
    assert projected["ops_health_status"] == "warning"
    assert projected["ops_health_timestamp"] == "2026-08-17T08:00:00Z"
    assert projected["ops_health_scope"] == "custom_scope"

    empty_proj = hks._project_public_kill_fields({})
    assert empty_proj["kill_switch"] == {}
    assert empty_proj["open_incidents"] == {}
    assert empty_proj["ops_health_status"] == "ok"
    assert empty_proj["ops_health_scope"] == "operational_readiness"


def test_elevate_public_system_status() -> None:
    assert hks._elevate_public_system_status("healthy", "ok") == "healthy"
    assert hks._elevate_public_system_status("healthy", "warning") == "warning"
    assert hks._elevate_public_system_status("warning", "degraded") == "degraded"
    assert hks._elevate_public_system_status("warning", "critical") == "critical"
    assert hks._elevate_public_system_status("critical", "healthy") == "critical"
    assert hks._elevate_public_system_status(None, "warning") == "warning"
    assert hks._elevate_public_system_status("healthy", "error") == "critical"


def test_enforce_worst_wins_system_status() -> None:
    assert hks.enforce_worst_wins_system_status("not_a_dict") == "healthy"  # type: ignore

    payload = {"system_status": "healthy", "ops_health_status": "warning"}
    assert hks.enforce_worst_wins_system_status(payload) == "warning"
    assert payload["system_status"] == "warning"
    assert payload["system_status_rollup"] == "worst_wins:ops_health_status,kill_switch,open_incidents"

    # When already worst or equal, no modification
    payload2 = {"system_status": "critical", "ops_health_status": "warning"}
    assert hks.enforce_worst_wins_system_status(payload2) == "critical"
    assert "system_status_rollup" not in payload2


def test_is_monitor_health_report() -> None:
    assert not hks._is_monitor_health_report({})
    assert not hks._is_monitor_health_report({"checks": "not_a_dict"})
    assert not hks._is_monitor_health_report({"checks": {}, "system_status": "ok"})
    assert hks._is_monitor_health_report({"checks": {}, "status": "ok"})
    assert hks._is_monitor_health_report({"checks": {}})


def test_load_ops_monitor_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)

    # Empty dirs -> None
    assert hks.load_ops_monitor_report(data_dir=data_dir, public_dir=public_dir) is None

    # Write older in data, newer in public
    data_report = {
        "status": "ok",
        "timestamp": "2026-08-17T01:00:00Z",
        "checks": {},
    }
    public_report = {
        "status": "warning",
        "timestamp": "2026-08-17T02:00:00Z",
        "checks": {},
    }
    (data_dir / "health.json").write_text(json.dumps(data_report))
    (public_dir / "health_ops.json").write_text(json.dumps(public_report))

    best = hks.load_ops_monitor_report(data_dir=data_dir, public_dir=public_dir)
    assert best is not None
    assert best["timestamp"] == "2026-08-17T02:00:00Z"
    assert best["status"] == "warning"


def test_patch_monitor_report_kill_open() -> None:
    report = {
        "status": "ok",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        },
    }
    disk_kill = {"enabled": True, "status": "critical"}
    disk_open = {"status": "critical", "open_count": 2}

    changed = hks._patch_monitor_report_kill_open(report, disk_kill, disk_open)
    assert changed is True
    assert report["checks"]["kill_switch"] == disk_kill
    assert report["checks"]["open_incidents"] == disk_open
    assert report["status"] == "critical"
    assert "ssot_reconciled_at" in report
    assert report["ssot_reconcile_source"] == "disk_incidents_kill"

    # Same parameters without force -> returns False
    assert hks._patch_monitor_report_kill_open(report, disk_kill, disk_open) is False


def test_write_health_generation_and_commit_public_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKER_RUN_ID", "tasker-run-123")
    out_file = tmp_path / "gen_test.json"
    hks.write_health_generation({"val": 42}, path=out_file, producer_sha="abc1234")
    assert out_file.exists()
    content = json.loads(out_file.read_text())
    assert content["val"] == 42
    assert content["producer_run_id"] == "tasker-run-123"
    assert content["producer_git_sha"] == "abc1234"
    assert "generation_id" in content

    index_path = tmp_path / "index.json"
    hks.commit_public_index({"files": ["gen_test.json"]}, index_path=index_path, generation_id="gen-456")
    assert index_path.exists()
    idx_content = json.loads(index_path.read_text())
    assert idx_content["generation_id"] == "gen-456"
    assert "generated_at" in idx_content


def test_reconcile_monitor_health_with_disk_ssot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    health_file = data_dir / "health.json"

    # Missing file returns False
    assert hks.reconcile_monitor_health_with_disk_ssot(data_dir=data_dir) is False

    # Seed health.json with stale cleared state while kill_switch.json is enabled
    health_report = {
        "status": "ok",
        "timestamp": "2026-08-17T00:00:00Z",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        },
    }
    health_file.write_text(json.dumps(health_report))

    # Seed active kill switch on disk
    (data_dir / "kill_switch.json").write_text(
        json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "manual_test",
            "incident_id": "INC-001",
        })
    )

    reconciled = hks.reconcile_monitor_health_with_disk_ssot(data_dir=data_dir)
    assert reconciled is True

    updated = json.loads(health_file.read_text())
    assert updated["checks"]["kill_switch"]["enabled"] is True
    assert updated["checks"]["kill_switch"]["status"] == "critical"
    assert updated["status"] == "critical"


def test_project_disk_kill_open_to_all_surfaces(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)

    # 1. Private monitor health.json
    (data_dir / "health.json").write_text(
        json.dumps({
            "status": "ok",
            "checks": {
                "kill_switch": {"enabled": False, "status": "ok"},
                "open_incidents": {"status": "ok", "open_count": 0},
            },
        })
    )
    # 2. Public health_ops.json
    (public_dir / "health_ops.json").write_text(
        json.dumps({
            "status": "ok",
            "checks": {
                "kill_switch": {"enabled": False, "status": "ok"},
                "open_incidents": {"status": "ok", "open_count": 0},
            },
        })
    )
    # 3. Public health.json (dashboard schema)
    (public_dir / "health.json").write_text(
        json.dumps({
            "system_status": "healthy",
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        })
    )

    # Active kill on disk
    (data_dir / "kill_switch.json").write_text(
        json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "test_halt",
            "incident_id": "INC-TEST",
        })
    )

    res = hks.project_disk_kill_open_to_all_surfaces(data_dir=data_dir, public_dir=public_dir)
    assert res["monitor"] is True
    assert res["ops"] is True
    assert res["public"] is True

    pub_health = json.loads((public_dir / "health.json").read_text())
    assert pub_health["kill_switch"]["enabled"] is True
    assert pub_health["system_status"] in {"critical", "degraded", "warning"}


def test_disk_kill_ssot_is_clear_and_disk_kill_and_open_incidents(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # Empty data dir is clear
    assert hks._disk_kill_ssot_is_clear(data_dir) is True
    kill, open_inc = hks._disk_kill_and_open_incidents(data_dir)
    assert kill["enabled"] is False
    assert open_inc["open_count"] == 0

    # Arm kill switch
    (data_dir / "kill_switch.json").write_text(json.dumps({"enabled": True, "level": "warn"}))
    assert hks._disk_kill_ssot_is_clear(data_dir) is False
    kill, _ = hks._disk_kill_and_open_incidents(data_dir)
    assert kill["enabled"] is True
