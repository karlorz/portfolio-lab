"""Health check exit codes and public generated_at on ops merge."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_main_warning_exits_zero(monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "run_health_check", lambda: {"status": "warning", "checks": {}})
    assert hc.main() == 0


def test_main_ok_exits_zero(monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "run_health_check", lambda: {"status": "ok", "checks": {}})
    assert hc.main() == 0


def test_main_critical_exits_one(monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "run_health_check", lambda: {"status": "critical", "checks": {}})
    assert hc.main() == 1


def test_publish_ops_merge_stamps_generated_at(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    public = tmp_path / "public"
    public.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (public / "health.json").write_text(
        json.dumps({"generated_at": "2026-07-20T21:15:03", "system_status": "healthy", "cron_jobs": []})
    )
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(hc, "health_ops_path", lambda: data / "health_ops.json")
    monkeypatch.setattr(hc, "refresh_signals_health_kill_fields", lambda *a, **k: None)
    monkeypatch.setattr(
        hc,
        "apply_ops_monitor_to_dashboard_health",
        lambda payload, report, **kw: payload.update({"ops_health_status": report.get("status")}),
    )

    report = {
        "status": "warning",
        "timestamp": "2026-07-20T14:00:05+00:00",
        "scope": "operational_readiness",
        "checks": {},
    }
    hc.publish_ops_health_surfaces(report)
    out = json.loads((public / "health.json").read_text())
    assert out["generated_at"] != "2026-07-20T21:15:03"
    assert out.get("content_patch_source") == "ops_health_merge"
    assert (data / "health_ops.json").exists()
