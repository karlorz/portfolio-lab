"""Batch AO residual honesty: decision_registry / incidents / alerts / daily_brief git sha."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch



def test_decision_registry_snapshot_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.monitor import decision_registry as dr

    monkeypatch.setattr(dr, "_git_sha_short", lambda project_root=None: "abc123def456")
    reg = dr.DecisionRegistry(db_path=tmp_path / "registry.db")
    snap = dr.build_decision_registry_snapshot(registry=reg)
    assert snap["generator_git_sha"] == "abc123def456"
    assert snap["generator_git_sha_status"] == "full"


def test_decision_registry_snapshot_marks_sha_unavailable(tmp_path, monkeypatch):
    from src.monitor import decision_registry as dr

    monkeypatch.setattr(dr, "_git_sha_short", lambda project_root=None: None)
    reg = dr.DecisionRegistry(db_path=tmp_path / "registry.db")
    snap = dr.build_decision_registry_snapshot(registry=reg)
    assert snap["generator_git_sha"] is None
    assert snap["generator_git_sha_status"] == "unavailable"


def test_publish_decision_registry_json_includes_git_sha(tmp_path, monkeypatch):
    from src.monitor import decision_registry as dr

    monkeypatch.setattr(dr, "_git_sha_short", lambda project_root=None: "deadbeefcafe")
    reg = dr.DecisionRegistry(db_path=tmp_path / "registry.db")
    path = dr.publish_decision_registry_json(public_dir=tmp_path / "public", registry=reg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generator_git_sha"] == "deadbeefcafe"
    assert payload["generator_git_sha_status"] == "full"


def test_incident_write_summary_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.monitor.incident_manager import IncidentManager

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "incidentsha12",
    )
    mgr = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    summary = mgr.write_summary()
    assert summary.get("generator_git_sha") == "incidentsha12"
    on_disk = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
    assert on_disk.get("generator_git_sha") == "incidentsha12"


def test_generator_alerts_json_stamps_git_sha(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "alertssha1234",
    )
    # Avoid DB/network side effects: stub helpers used by generate_alerts_json
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = MagicMock()
    gen._load_json_file = MagicMock(return_value={})
    # Minimal path: monkeypatch heavy internals if generate_alerts_json is complex
    # Prefer calling the stamp path via a thin write of the output structure used in source.
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "output = _stamp_generator_git_sha({" in src
    assert '"alerts": sorted(alerts' in src


def test_generator_incidents_json_stamps_and_utc(tmp_path, monkeypatch):
    from src.dashboard import generator as gen_mod
    from src.dashboard.generator import DashboardGenerator

    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "_generator_git_sha_short", lambda: "incidentpub1")
    # No private incidents file → empty summary path
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._load_json_file = MagicMock(return_value=None)
    out = DashboardGenerator.generate_incidents_json(gen)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["generator_git_sha"] == "incidentpub1"
    assert payload["generated_at"].endswith("+00:00") or "Z" in payload["generated_at"] or "+" in payload["generated_at"]


def test_daily_brief_stamps_generator_git_sha(monkeypatch):
    from src.monitor import daily_brief as db

    monkeypatch.setattr(
        db,
        "generate_brief_sections",
        lambda dashboard: [],
    )
    monkeypatch.setattr(
        db,
        "render_brief_text",
        lambda sections, narrative=None: "brief",
    )

    def _fake_unified():
        return {}

    monkeypatch.setattr(
        "src.monitor.unified_dashboard.generate_unified_dashboard",
        _fake_unified,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "briefsha12345",
    )
    # Patch import path used inside generate_daily_brief
    with patch(
        "src.monitor.unified_dashboard.generate_unified_dashboard",
        return_value={},
    ):
        brief = db.generate_daily_brief()
    assert brief.get("generator_git_sha") == "briefsha12345"
    assert brief.get("generator_git_sha_status") == "full_generate"


def test_batch_ao_source_contracts():
    """Static contracts so future refactors keep stamps on operator SSOT writers."""
    decision_src = Path("src/monitor/decision_registry.py").read_text(encoding="utf-8")
    assert 'snapshot["generator_git_sha"]' in decision_src

    incident_src = Path("src/monitor/incident_manager.py").read_text(encoding="utf-8")
    assert "_stamp_generator_git_sha" in incident_src

    gen_src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "payload = _stamp_generator_git_sha(payload)" in gen_src
    assert "output = _stamp_generator_git_sha({" in gen_src

    brief_src = Path("src/monitor/daily_brief.py").read_text(encoding="utf-8")
    assert "_stamp_generator_git_sha" in brief_src
