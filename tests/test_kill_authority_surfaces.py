"""Multi-surface kill authority honesty (2026-07-12 batch)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator, _compact_health_summary
from src.dashboard.kill_authority import (
    allocation_roles_under_kill,
    build_kill_switch_alert,
    elevate_system_status_for_kill,
    load_kill_switch_payload,
    project_compact_kill_fields,
    project_kill_switch_fields,
)
from scripts.check_public_data_consistency import check_public_data_consistency


INCIDENT_ID = "4d9e4f53-test-kill-authority"


def _kill_payload(**overrides):
    base = {
        "enabled": True,
        "level": "halt",
        "reason": "max_drawdown_-25.0%",
        "message": "Paper trading halted: max drawdown breached",
        "mode": "paper",
        "source": "incident_manager",
        "channel": "risk",
        "incident_id": INCIDENT_ID,
        "timestamp": "2026-07-12T12:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_project_compact_kill_fields_from_monitor_health():
    report = {
        "status": "critical",
        "checks": {
            "kill_switch": {
                "status": "critical",
                "enabled": True,
                "level": "halt",
                "reason": "max_drawdown_-25.0%",
                "incident_id": INCIDENT_ID,
                "message": "Paper trading halted",
            },
            "open_incidents": {"status": "critical", "open_count": 1},
        },
    }
    compact = project_compact_kill_fields(report)
    assert compact["kill_switch_enabled"] is True
    assert compact["kill_switch_level"] == "halt"
    assert compact["kill_switch_incident_id"] == INCIDENT_ID
    assert compact["open_incidents_status"] == "critical"


def test_compact_health_summary_projects_kill_under_halt():
    summary = _compact_health_summary(
        {
            "status": "critical",
            "generated_at": "2026-07-12T12:00:00",
            "checks": {
                "kill_switch": {
                    "status": "critical",
                    "enabled": True,
                    "level": "halt",
                    "reason": "max_drawdown_-25.0%",
                    "incident_id": INCIDENT_ID,
                },
                "open_incidents": {"status": "critical", "open_count": 1},
            },
            "cron_jobs": [],
            "data_freshness": {},
        }
    )
    assert summary["status"] == "critical"
    assert summary["kill_switch_enabled"] is True
    assert summary["kill_switch_level"] == "halt"
    assert summary["kill_switch_incident_id"] == INCIDENT_ID
    assert summary["open_incidents_count"] == 1


def test_elevate_system_status_for_kill_halt():
    assert elevate_system_status_for_kill(
        "healthy",
        {"enabled": True, "level": "halt", "status": "critical"},
    ) == "critical"
    assert elevate_system_status_for_kill(
        "healthy",
        {"enabled": True, "level": "warning", "status": "warning"},
    ) == "warning"
    assert elevate_system_status_for_kill("healthy", {"enabled": False}) == "healthy"


def test_build_kill_switch_alert_prefers_human_message():
    alert = build_kill_switch_alert(_kill_payload())
    assert alert is not None
    assert alert["type"] == "kill_switch"
    assert alert["message"] == "Paper trading halted: max drawdown breached"
    assert alert["reason"] == "max_drawdown_-25.0%"
    assert alert["incident_id"] == INCIDENT_ID
    assert alert["channel"] == "risk"
    assert "PAPER" in alert["title"]


def test_build_kill_switch_alert_falls_back_to_reason():
    alert = build_kill_switch_alert(
        _kill_payload(message=None)
    )
    assert alert is not None
    assert alert["message"] == "max_drawdown_-25.0%"
    assert alert["reason"] == "max_drawdown_-25.0%"


def test_allocation_roles_under_kill_blocks_live_authority():
    base = DashboardGenerator._build_allocation_surface_roles()
    # Without kill file DATA_DIR may or may not have kill; force via helper
    roles = {
        "schema_version": "allocation-surface-roles/v1",
        "routed_surface": "target_allocations",
        "routed_by": "src.broker.order_router",
        "surfaces": {
            "target_allocations": {
                "label": "Target Allocation",
                "role": "execution_routed",
                "routed": True,
                "routed_by": "src.broker.order_router",
                "live_authoritative": True,
                "description": "routed",
            }
        },
    }
    blocked = allocation_roles_under_kill(roles, kill_enabled=True, kill_level="halt")
    target = blocked["surfaces"]["target_allocations"]
    assert target["live_authoritative"] is False
    assert target["execution_blocked"] is True
    assert target["role"] == "execution_blocked"
    assert blocked["execution_blocked"] is True


def test_build_allocation_surface_roles_kill_on(tmp_path, monkeypatch):
    from src.dashboard import generator as gen_mod

    (tmp_path / "kill_switch.json").write_text(json.dumps(_kill_payload()))
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)
    target = roles["surfaces"]["target_allocations"]
    assert target["live_authoritative"] is False
    assert target["execution_blocked"] is True
    assert target["kill_switch_level"] == "halt"


def test_build_allocation_surface_roles_kill_off(tmp_path, monkeypatch):
    from src.dashboard import generator as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)
    target = roles["surfaces"]["target_allocations"]
    assert target["live_authoritative"] is True
    assert target.get("execution_blocked") is not True


def _make_generator(tmp_path: Path):
    import sqlite3

    db = tmp_path / "market.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit()
    # DashboardGenerator expects MARKET_DB; patch paths
    return db


def test_generate_alerts_json_includes_human_message_and_incident(tmp_path, monkeypatch):
    import sqlite3
    from src.dashboard import generator as gen_mod

    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", tmp_path)
    (tmp_path / "kill_switch.json").write_text(json.dumps(_kill_payload()))
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db))
    gen.conn.row_factory = sqlite3.Row
    path = gen.generate_alerts_json()
    data = json.loads(path.read_text())
    kill_alerts = [a for a in data["alerts"] if a["type"] == "kill_switch"]
    assert len(kill_alerts) == 1
    assert kill_alerts[0]["message"] == "Paper trading halted: max drawdown breached"
    assert kill_alerts[0]["reason"] == "max_drawdown_-25.0%"
    assert kill_alerts[0]["incident_id"] == INCIDENT_ID
    gen.conn.close()


def test_generate_health_json_includes_kill_and_elevates(tmp_path, monkeypatch):
    import sqlite3
    from src.dashboard import generator as gen_mod

    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", tmp_path)
    (tmp_path / "kill_switch.json").write_text(json.dumps(_kill_payload()))
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": INCIDENT_ID,
                        "state": "open",
                        "kill_switch_level": "halt",
                        "message": "halted",
                    }
                ],
            }
        )
    )
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db))
    gen.conn.row_factory = sqlite3.Row
    path = gen.generate_health_json()
    health = json.loads(path.read_text())
    assert "kill_switch" in health
    assert health["kill_switch"]["enabled"] is True
    assert health["kill_switch"]["level"] == "halt"
    assert health["kill_switch"]["incident_id"] == INCIDENT_ID
    assert health["system_status"] == "critical"
    assert health["open_incidents"]["status"] == "critical"
    gen.conn.close()


def test_run_health_check_persists_kill_fields(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "HEALTH_PATH", tmp_path / "health.json")
    (tmp_path / "kill_switch.json").write_text(json.dumps(_kill_payload()))
    # Fresh enough market data not required — function still writes
    report = hc.run_health_check()
    assert report["checks"]["kill_switch"]["enabled"] is True
    on_disk = json.loads((tmp_path / "health.json").read_text())
    assert on_disk["checks"]["kill_switch"]["enabled"] is True
    assert on_disk["checks"]["kill_switch"]["level"] == "halt"
    assert on_disk["checks"]["kill_switch"]["incident_id"] == INCIDENT_ID


def _write_consistent_public_data_set(app_dir: Path) -> None:
    import hashlib

    source_generated_at = "2026-06-12T09:05:25.028Z"
    index_generated_at = "2026-06-12T09:06:00+00:00"
    public_data = app_dir / "public" / "data"
    dist_data = app_dir / "dist" / "data"
    public_data.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    write_json(
        public_data / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": source_generated_at,
            "artifacts": [{"artifact": "prices.json", "provider": "Yahoo Finance", "status": "success"}],
        },
    )
    source_hash = hashlib.sha256((public_data / "source_manifest.json").read_bytes()).hexdigest()
    write_json(
        public_data / "index.json",
        {
            "schema_version": "public-data-index/v1",
            "generated_at": index_generated_at,
            "source_manifest": {
                "path": "source_manifest.json",
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": source_generated_at,
                "sha256": source_hash,
            },
            "entries": [
                {
                    "filename": "source_manifest.json",
                    "path": "source_manifest.json",
                    "status": "present",
                    "generated_at": source_generated_at,
                    "sha256": source_hash,
                }
            ],
        },
    )
    write_json(public_data / "health.json", {"status": "ok", "generated_at": index_generated_at})
    dist_data.mkdir(parents=True, exist_ok=True)
    for filename in ("source_manifest.json", "index.json", "health.json"):
        shutil.copyfile(public_data / filename, dist_data / filename)


def test_consistency_requires_kill_switch_alert_when_enabled(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "kill_switch.json").write_text(json.dumps(_kill_payload()))
    public_data = tmp_path / "public" / "data"
    # alerts without kill row
    alerts_path = public_data / "alerts.json"
    alerts_path.write_text(json.dumps({"alerts": [], "count": 0}))
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is False
    assert any("kill_switch" in e for e in result.errors)


def test_consistency_requires_graduation_candidate_when_promote_present(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".promote_to_live").write_text(json.dumps({"metrics": {"sharpe": 0.9}}))
    public_data = tmp_path / "public" / "data"
    alerts_path = public_data / "alerts.json"
    alerts_path.write_text(json.dumps({"alerts": [], "count": 0}))
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is False
    assert any("graduation_candidate" in e for e in result.errors)


def test_consistency_accepts_matching_kill_identity(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    kill = _kill_payload()
    (data_dir / "kill_switch.json").write_text(json.dumps(kill))
    public_data = tmp_path / "public" / "data"
    health = {
        "system_status": "critical",
        "generated_at": "2026-06-12T09:06:00+00:00",
        "kill_switch": project_kill_switch_fields(kill),
        "data_pipeline_slo": {"status": "ok"},
    }
    (public_data / "health.json").write_text(json.dumps(health, sort_keys=True))
    alert = build_kill_switch_alert(kill)
    assert alert is not None
    (public_data / "alerts.json").write_text(
        json.dumps({"alerts": [alert], "count": 1}, sort_keys=True)
    )
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is True, result.errors


def test_consistency_rejects_divergent_kill_incident_id(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    kill = _kill_payload()
    (data_dir / "kill_switch.json").write_text(json.dumps(kill))
    public_data = tmp_path / "public" / "data"
    health = {
        "system_status": "critical",
        "generated_at": "2026-06-12T09:06:00+00:00",
        "kill_switch": project_kill_switch_fields(kill),
    }
    (public_data / "health.json").write_text(json.dumps(health, sort_keys=True))
    bad_alert = build_kill_switch_alert(kill)
    assert bad_alert is not None
    bad_alert["incident_id"] = "stale-fixture-id"
    (public_data / "alerts.json").write_text(
        json.dumps({"alerts": [bad_alert], "count": 1}, sort_keys=True)
    )
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is False
    assert any("incident_id" in e for e in result.errors)


def test_consistency_rejects_missing_alert_identity_fields(tmp_path: Path) -> None:
    """Kill alert present but missing incident_id/level/reason must fail gate."""
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    kill = _kill_payload()
    (data_dir / "kill_switch.json").write_text(json.dumps(kill))
    public_data = tmp_path / "public" / "data"
    health = {
        "system_status": "critical",
        "generated_at": "2026-06-12T09:06:00+00:00",
        "kill_switch": project_kill_switch_fields(kill),
    }
    (public_data / "health.json").write_text(json.dumps(health, sort_keys=True))
    # Stale reason-only alert (pre-fix shape)
    stale_alert = {
        "level": "error",
        "type": "kill_switch",
        "title": "LIVE Kill Switch Triggered",
        "message": "position_limit_exceeded",
        "requires_action": True,
    }
    (public_data / "alerts.json").write_text(
        json.dumps({"alerts": [stale_alert], "count": 1}, sort_keys=True)
    )
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is False
    joined = " ".join(result.errors)
    assert "missing incident_id" in joined
    assert "missing kill_switch_level" in joined or "missing reason" in joined


def test_consistency_rejects_missing_public_health_kill_switch(tmp_path: Path) -> None:
    """When kill enabled, public health must project kill_switch block."""
    _write_consistent_public_data_set(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    kill = _kill_payload()
    (data_dir / "kill_switch.json").write_text(json.dumps(kill))
    public_data = tmp_path / "public" / "data"
    # health without kill_switch (pre-fix shape)
    health = {
        "system_status": "critical",
        "generated_at": "2026-06-12T09:06:00+00:00",
        "data_pipeline_slo": {"status": "ok"},
    }
    (public_data / "health.json").write_text(json.dumps(health, sort_keys=True))
    alert = build_kill_switch_alert(kill)
    assert alert is not None
    (public_data / "alerts.json").write_text(
        json.dumps({"alerts": [alert], "count": 1}, sort_keys=True)
    )
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {"filename": "alerts.json", "path": "alerts.json", "status": "present"}
    )
    index_path.write_text(json.dumps(index, sort_keys=True))
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path)
    assert result.ok is False
    assert any("missing kill_switch" in e for e in result.errors)


def test_generate_alerts_json_rewrites_stale_public_kill_row(tmp_path, monkeypatch):
    """Stale LIVE/position_limit public kill row is replaced by SSOT paper halt identity."""
    import sqlite3
    from src.dashboard import generator as gen_mod

    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()

    public = tmp_path / "public"
    public.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    # Authority: paper halt with incident id
    kill = _kill_payload()
    (data / "kill_switch.json").write_text(json.dumps(kill))
    # Stale public alerts (the skeptic failure shape)
    stale = {
        "alerts": [
            {
                "level": "error",
                "type": "kill_switch",
                "title": "LIVE Kill Switch Triggered",
                "message": "position_limit_exceeded",
                "timestamp": "2026-05-25T07:00:00",
                "requires_action": True,
                "reason": "position_limit_exceeded",
            }
        ],
        "count": 1,
    }
    (public / "alerts.json").write_text(json.dumps(stale))
    monkeypatch.setattr(gen_mod, "DATA_DIR", data)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db))
    gen.conn.row_factory = sqlite3.Row
    out = gen.generate_alerts_json()
    gen.conn.close()

    payload = json.loads(out.read_text())
    kills = [a for a in payload["alerts"] if a.get("type") == "kill_switch"]
    assert len(kills) == 1
    assert kills[0]["incident_id"] == INCIDENT_ID
    assert kills[0]["reason"] == "max_drawdown_-25.0%"
    assert kills[0]["kill_switch_level"] == "halt"
    assert "PAPER" in kills[0]["title"]
    assert "position_limit" not in str(kills[0].get("message", ""))
    assert "LIVE" not in kills[0]["title"]
    # On-disk public path must match (not only return value)
    disk = json.loads((public / "alerts.json").read_text())
    disk_kills = [a for a in disk["alerts"] if a.get("type") == "kill_switch"]
    assert disk_kills[0]["incident_id"] == INCIDENT_ID
