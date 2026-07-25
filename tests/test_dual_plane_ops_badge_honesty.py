"""Regression contract for honest ops and signal-quality health planes.

The operator-facing ``system_status`` / ``ops_health_status`` fields describe
serving and data-plumbing readiness. Signal predictive breadth remains visible
through ``signal_health`` and ``signal_quality`` alerts without making the ops
badge warning or critical by itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dashboard.health_report import derive_system_status
from src.dashboard.health_slo_alerts import build_health_slo_alerts
from src.dashboard.signal_health_section import attach_signal_quality_disclosure
from src.dashboard.cron_scheduler_section import _elevate_compact_health_status


def _thin_signal_health(
    data_dir: Path,
    *,
    healthy: int = 1,
    degraded: int = 6,
    unhealthy: int = 2,
) -> dict:
    total = healthy + degraded + unhealthy
    report = {
        "status": "degraded",
        "overall_health": "degraded",
        "summary": {
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "total_tracked": total,
        },
        "scores": {},
    }
    return attach_signal_quality_disclosure(report, data_dir=data_dir)


def _green_ops_health(signal_health: dict) -> dict:
    return {
        "system_status": derive_system_status(
            current="healthy",
            scheduler_status="ok",
            slo_status="ok",
            failed_jobs=0,
            stale_count=0,
        ),
        "generated_at": "2026-07-25T07:16:41+00:00",
        "ops_health_status": "ok",
        "scheduler_status": {"status": "ok", "backends": {}},
        "data_pipeline_slo": {"status": "ok"},
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "ok",
        "signal_health": signal_health,
    }


def test_green_ops_with_one_of_n_signal_health_keeps_ops_badges_green(
    tmp_path: Path,
) -> None:
    """Live-shaped 1/9 quality breadth must not bleed into the ops badge."""
    health = _green_ops_health(_thin_signal_health(tmp_path))

    assert health["ops_health_status"] == "ok"
    assert health["system_status"] == "healthy"

    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["type"] == "signal_quality"
    assert alert["level"] == "warning"
    assert alert["requires_action"] is False
    assert alert["signal_quality_badge"] == "1/9 healthy sources"
    assert "ops" not in (alert.get("title") or "").lower()


def test_quality_only_all_unhealthy_never_creates_ops_critical_status(
    tmp_path: Path,
) -> None:
    """Even 0/N all-unhealthy is quality-critical, not an ops outage."""
    health = _green_ops_health(
        _thin_signal_health(tmp_path, healthy=0, degraded=0, unhealthy=9)
    )

    assert health["system_status"] == "healthy"
    alerts = build_health_slo_alerts(health)
    assert len(alerts) == 1
    assert alerts[0]["type"] == "signal_quality"
    assert alerts[0]["level"] == "warning"
    assert alerts[0]["zero_healthy_sources"] is True


def test_ops_monitor_merge_does_not_refold_signal_quality_into_system_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial ops restamps must preserve the same two-plane projection."""
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(
        hc,
        "_project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )

    health = _green_ops_health(_thin_signal_health(tmp_path))
    report = {
        "status": "ok",
        "timestamp": "2026-07-25T07:16:41+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
        "scope": "operational_readiness",
    }

    result = hc.apply_ops_monitor_to_dashboard_health(
        health, report, data_dir=data, public_dir=public
    )

    assert result["ops_health_status"] == "ok"
    assert result["system_status"] == "healthy"


def test_compact_signals_health_status_keeps_quality_in_compact_fields() -> None:
    """signals.json health.status is ops-plane; SH remains separately compact."""
    compact = {
        "status": "healthy",
        "scheduler_status": "ok",
        "failed_cron_jobs": 0,
        "signal_health_status": "degraded",
        "signal_health_healthy": 1,
        "signal_health_degraded": 6,
        "signal_health_unhealthy": 2,
        "signal_health_total_tracked": 9,
        "signal_quality_badge": "1/9 healthy sources",
    }

    result = _elevate_compact_health_status(compact)

    assert result["status"] == "healthy"
    assert result["signal_health_status"] == "degraded"
    assert result["signal_quality_badge"] == "1/9 healthy sources"


def test_health_job_partial_refresh_clears_sticky_quality_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ops-ok restamp clears old SH-derived compact status, not SH fields."""
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {
            "status": "degraded",
            "signal_health_status": "degraded",
            "signal_health_healthy": 1,
            "signal_health_total_tracked": 9,
            "signal_quality_badge": "1/9 healthy sources",
        },
    }
    for root in (data, public):
        (root / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)

    report = {
        "status": "ok",
        "timestamp": "2026-07-25T07:16:41+00:00",
        "signal_health": _thin_signal_health(tmp_path),
    }
    with patch.object(
        hc,
        "_disk_kill_and_open_incidents",
        return_value=(
            {"enabled": False, "status": "ok", "level": None},
            {"open_count": 0, "status": "ok"},
        ),
    ), patch(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        return_value={
            "ok": True,
            "lagging_count": 0,
            "total": 36,
            "lagging_paths": [],
            "source": str(public),
            "dest": str(public),
        },
    ), patch(
        "src.dashboard.generator.project_paper_return_ssot_onto_health",
        side_effect=lambda health, comparison: health,
    ):
        hc.refresh_signals_health_kill_fields(
            report, public_dir=public, data_dir=data
        )

    result = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    health = result["health"]
    assert health["status"] == "healthy"
    assert health["signal_health_status"] == "degraded"
    assert health["signal_quality_badge"] == "1/9 healthy sources"
    assert result["target_allocations"] == {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    }
