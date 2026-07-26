"""Batch JL — JH1a/b/c health alerts SSOT + ops vs signal_quality labeling.

Session A plan (JL):
- JH1c: publish_health_alerts_json must use public dashboard health for SH rollup
- JH1a: SH-only demotion → type/title signal_quality, not Health Warning: ops
- JH1b: public ops_health_status tracks live monitor report.status
Does not touch signals.json.target_allocations / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _sh_zero_healthy_payload() -> dict:
    return {
        "system_status": "degraded",
        "generated_at": "2026-07-23T18:40:00+00:00",
        "scheduler_status": "ok",
        "ops_health_status": "ok",
        "data_pipeline_slo": {"status": "ok"},
        "signal_health": {
            "status": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 7,
                "unhealthy": 2,
                "total_tracked": 9,
                "quality_badge": "0/9 healthy sources",
                "zero_healthy_sources": True,
            },
        },
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }


def test_jh1a_sh_only_not_titled_ops() -> None:
    """Case B / JH1a: SH-only degraded must not title Health Warning: ops."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    alerts = build_health_slo_alerts(_sh_zero_healthy_payload())
    assert alerts, "SH 0/9 must still surface a warning alert"
    a = alerts[0]
    title = (a.get("title") or "").lower()
    assert a.get("type") == "signal_quality" or "signal" in title
    assert "health warning: ops" not in title
    assert a.get("requires_action") is False


def test_jh1a_kill_still_ops_path() -> None:
    """Case C: kill-enabled path remains kill / not quality-only."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _sh_zero_healthy_payload()
    health["system_status"] = "critical"
    health["data_pipeline_slo"] = {
        "status": "critical",
        "top_dimension": "kill_switch",
        "dimensions": {
            "kill_switch": {"reason": "unresolved_incident:signal_staleness"},
        },
    }
    alerts = build_health_slo_alerts(health)
    assert alerts
    a = alerts[0]
    assert a.get("level") == "error"
    assert a.get("type") == "health_slo"
    assert "kill" in (a.get("title") or "").lower() or a.get("top_dimension") == "kill_switch"


def test_jh1c_publish_uses_public_dashboard_when_monitor_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case A: monitor report ok + public SH degraded → alerts count≥1 signal_quality."""
    import src.monitor.health_check as hc

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    (public / "health.json").write_text(
        json.dumps(_sh_zero_healthy_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json")

    monitor_report = {
        "status": "ok",
        "timestamp": "2026-07-23T18:40:00+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
        "scope": "operational_readiness",
    }
    out = hc.publish_health_alerts_json(monitor_report)
    assert out is not None
    body = json.loads((public / "alerts.json").read_text(encoding="utf-8"))
    assert body.get("count", 0) >= 1
    titles = " ".join((a.get("title") or "") for a in body.get("alerts") or []).lower()
    types = [a.get("type") for a in body.get("alerts") or []]
    assert "health warning: ops" not in titles
    assert "signal_quality" in types or "signal" in titles


def test_jh1b_ops_health_status_from_live_report_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case D: sticky public ops_health=warning clears when monitor report is ok."""
    import src.monitor.health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    # No kill / empty incidents
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )

    health = _sh_zero_healthy_payload()
    health["ops_health_status"] = "warning"  # sticky lie
    health["ops_health_timestamp"] = "2026-07-23T17:00:00+00:00"

    report = {
        "status": "ok",
        "timestamp": "2026-07-23T18:40:00+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
        "scope": "operational_readiness",
    }
    monkeypatch.setattr(hc, "DATA_DIR", data)
    out = hc.apply_ops_monitor_to_dashboard_health(
        health, report, data_dir=data, public_dir=public
    )
    assert out.get("ops_health_status") in {"ok", "healthy"}
