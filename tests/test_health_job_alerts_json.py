"""Health job emits alerts.json with health publish."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


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
