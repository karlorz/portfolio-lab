"""Health job emits alerts.json with health publish."""

from __future__ import annotations

import json


def test_publish_health_alerts_json_writes_public_and_private(tmp_path, monkeypatch):
    from src.monitor.health_check import publish_health_alerts_json

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    report = {
        "status": "ok",
        "generated_at": "2026-07-22T12:00:00+00:00",
        "timestamp": "2026-07-22T12:00:00+00:00",
        "checks": {},
    }
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data)
    monkeypatch.setattr(
        "src.monitor.health_check.HEALTH_PATH", data / "health.json"
    )
    out = publish_health_alerts_json(report)
    assert out is not None
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload.get("generated_at") == "2026-07-22T12:00:00+00:00"
    assert payload.get("source") == "health_check_job"
    assert "alerts" in payload
    assert (data / "alerts.json").exists()


def test_publish_health_alerts_json_reuses_canonical_kill_alert(
    tmp_path, monkeypatch
):
    from src.dashboard.kill_authority import build_kill_switch_alert
    from src.monitor.health_check import publish_health_alerts_json

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    kill_payload = {
        "enabled": True,
        "level": "halt",
        "mode": "paper",
        "reason": "max_drawdown_-25.0%",
        "message": "Paper trading halted: max drawdown breached",
        "incident_id": "INC-20260728-001",
        "channel": "risk",
        "source": "risk_monitor",
        "timestamp": "2026-07-28T00:00:00+00:00",
    }
    (data / "kill_switch.json").write_text(
        json.dumps(kill_payload),
        encoding="utf-8",
    )
    report = {
        "status": "critical",
        "generated_at": "2026-07-28T00:01:00+00:00",
        "timestamp": "2026-07-28T00:01:00+00:00",
        "checks": {},
    }
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data)
    monkeypatch.setattr(
        "src.monitor.health_check.HEALTH_PATH", data / "health.json"
    )

    out = publish_health_alerts_json(report)

    assert out is not None
    payload = json.loads(out.read_text(encoding="utf-8"))
    kill_alerts = [
        alert
        for alert in payload["alerts"]
        if alert.get("type") == "kill_switch"
    ]
    assert kill_alerts == [build_kill_switch_alert(kill_payload)]
    assert json.loads((data / "alerts.json").read_text(encoding="utf-8")) == payload


def test_publish_health_alerts_json_stamps_full_generate_provenance(
    tmp_path, monkeypatch
):
    """F3: a fresh health-job alerts.json carries a real generator_git_sha.

    The health job fully rebuilds the alerts surface from SSOT each run, so
    the stamp must be ``full_generate`` (not ``partial_patch``) with a non-null
    HEAD-derived SHA — otherwise operators cannot attribute the artifact.
    """
    from src.monitor.health_check import publish_health_alerts_json

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    report = {
        "status": "ok",
        "generated_at": "2026-07-22T12:00:00+00:00",
        "timestamp": "2026-07-22T12:00:00+00:00",
        "checks": {},
    }
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data)
    monkeypatch.setattr(
        "src.monitor.health_check.HEALTH_PATH", data / "health.json"
    )

    out = publish_health_alerts_json(report)

    assert out is not None
    payload = json.loads(out.read_text(encoding="utf-8"))
    sha = payload.get("generator_git_sha")
    assert sha, "generator_git_sha must be non-null on a fresh health-job write"
    assert payload.get("generator_git_sha_status") == "full_generate"
    assert payload.get("last_full_generator_git_sha") == sha
    assert len(sha) == 12
    assert json.loads((data / "alerts.json").read_text(encoding="utf-8"))["generator_git_sha"] == sha
