"""Health alerts must not title ops when live lag is 0 and only SH is thin.

Session B 2026-07-26 — work item:
projects/portfolio-lab/work/2026-07-26-health-alerts-publish-after-lag-heal/
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _sh_one_of_nine() -> dict:
    return {
        "system_status": "warning",  # sticky mid-job
        "generated_at": "2026-07-26T10:30:17.872281+00:00",
        "scheduler_status": {"status": "ok"},
        "ops_health_status": "ok",
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 2,  # sticky stamp
        "repo_public_mirror_lag_status": "lagging",
        "repo_public_mirror_lagging_paths": ["health.json", "index.json"],
        "signal_health": {
            "status": "degraded",
            "summary": {
                "healthy": 1,
                "degraded": 6,
                "unhealthy": 2,
                "total_tracked": 9,
                "quality_badge": "1/9 healthy sources",
                "zero_healthy_sources": False,
            },
        },
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }


# --- N1 (2026-08-11 session B): dedupe G1-derived critical health_slo alerts ---


def _g1_halt_health() -> dict:
    """Live-shaped payload: system_status critical solely from the G1 halt."""
    return {
        "system_status": "critical",
        "generated_at": "2026-08-11T10:00:17+00:00",
        "ops_health_status": "critical",
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "kill_switch": {
            "enabled": True,
            "status": "critical",
            "level": "halt",
            "reason": "unresolved_incident:ic_decay",
            "incident_id": "8115a9c1-a167-4da7-9832-673617dc7de3",
        },
        "open_incidents": {"open_count": 1, "status": "critical"},
    }


def test_g1_halt_critical_suppresses_duplicate_health_slo_error() -> None:
    """Kill-switch halt alone must not emit a second requires_action alert.

    Regression: recurring critical_health_slo error every :00/:30 since
    09:15:13Z duplicating the dedicated kill_switch critical alert for
    incident 8115a9c1 (halt by design, operator decision pending).
    """
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    alerts = build_health_slo_alerts(_g1_halt_health())
    assert alerts == [], (
        "kill_switch halt with SLO+scheduler ok must not page a duplicate "
        "health_slo critical"
    )


def test_g1_halt_dedupe_keeps_real_slo_critical() -> None:
    """A genuinely critical data_pipeline_slo still alerts despite the halt."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _g1_halt_health()
    health["data_pipeline_slo"] = {"status": "critical", "top_dimension": "prices"}
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
    assert alerts[0]["type"] == "health_slo"


def test_g1_halt_dedupe_keeps_scheduler_critical() -> None:
    """A degraded scheduler still alerts despite the halt."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _g1_halt_health()
    health["scheduler_status"] = {"status": "error", "backends": {"tasker": "error"}}
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
    assert alerts[0]["type"] == "health_slo"


def test_slo_critical_without_halt_still_alerts() -> None:
    """Dedupe only applies to the halt complex — not bare SLO criticals."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _g1_halt_health()
    health["kill_switch"] = {"enabled": False, "status": "ok"}
    health["open_incidents"] = {"open_count": 0, "status": "ok"}
    health["data_pipeline_slo"] = {"status": "critical", "top_dimension": "prices"}
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
    assert alerts[0]["type"] == "health_slo"


def test_open_incident_without_kill_still_alerts() -> None:
    """Open-incident-only criticals keep their page (no kill alert exists)."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _g1_halt_health()
    health["kill_switch"] = {"enabled": False, "status": "ok"}
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
    assert alerts[0]["type"] == "health_slo"


def test_restrict_kill_does_not_suppress_critical_health_slo() -> None:
    """Non-halt kills must not mask a critical health/SLO page."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    health = _g1_halt_health()
    health["kill_switch"] = {
        "enabled": True,
        "status": "warning",
        "level": "restrict",
    }
    health["ops_health_status"] = "warning"
    health["data_pipeline_slo"] = {"status": "ok"}
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "error"
    assert alerts[0]["type"] == "health_slo"


def test_publish_heals_sticky_lag_stamp_when_live_lag_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sticky lag stamp + system_status=warning must not force Health Warning: ops
    when live probe reports lagging_count=0."""
    import src.monitor.health_check as hc

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    sticky = _sh_one_of_nine()
    (public / "health.json").write_text(json.dumps(sticky), encoding="utf-8")

    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json")

    def _live_lag_zero(**kwargs):
        return {
            "lagging_count": 0,
            "total": 35,
            "lagging_paths": [],
            "status": "ok",
            "ok": True,
            "source": str(public),
            "dest": str(tmp_path / "repo_public"),
        }

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        _live_lag_zero,
    )

    monitor = {
        "status": "ok",
        "timestamp": sticky["generated_at"],
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
        "scope": "operational_readiness",
    }
    out = hc.publish_health_alerts_json(monitor)
    assert out is not None
    body = json.loads((public / "alerts.json").read_text(encoding="utf-8"))
    assert body.get("count", 0) >= 1
    a = body["alerts"][0]
    title = (a.get("title") or "").lower()
    assert a.get("type") == "signal_quality"
    assert "health warning: ops" not in title
    assert "signal quality" in title


def test_publish_keeps_ops_title_when_live_lag_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real live lag must still allow ops-path labeling (not silent quality)."""
    import src.monitor.health_check as hc

    public = tmp_path / "public"
    data = tmp_path / "data"
    public.mkdir()
    data.mkdir()
    sticky = _sh_one_of_nine()
    (public / "health.json").write_text(json.dumps(sticky), encoding="utf-8")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json")

    def _live_lag_two(**kwargs):
        return {
            "lagging_count": 2,
            "total": 35,
            "lagging_paths": ["health.json", "index.json"],
            "status": "lagging",
            "ok": False,
        }

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        _live_lag_two,
    )

    out = hc.publish_health_alerts_json({"status": "ok", "checks": {}})
    assert out is not None
    body = json.loads((public / "alerts.json").read_text(encoding="utf-8"))
    a = body["alerts"][0]
    # ops path OK when live lag > 0
    assert a.get("type") == "health_slo" or "ops" in (a.get("title") or "").lower()


def test_build_health_slo_alerts_sticky_lag_stamp_alone_titles_ops() -> None:
    """Control: without live heal, sticky lag stamp forces ops path (root cause)."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts

    alerts = build_health_slo_alerts(_sh_one_of_nine())
    assert alerts
    a = alerts[0]
    assert a.get("type") == "health_slo"
    assert "ops" in (a.get("title") or "").lower()
