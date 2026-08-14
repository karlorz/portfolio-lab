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


# --- G7 (2026-08-11 session B): worst-wins system_status rollup assertion ---


def test_enforce_worst_wins_kill_halt_never_masked_by_healthy_system_status() -> None:
    """Public system_status must equal worst(ops, kill_switch, open_incidents).

    Regression for 09:05:41Z: public payload served system_status=healthy
    while kill halt + open incident were critical.
    """
    from src.monitor import health_check as hc

    payload = {
        "system_status": "healthy",
        "ops_health_status": "ok",
        "kill_switch": {"enabled": True, "level": "halt", "status": "critical"},
        "open_incidents": {"open_count": 1, "status": "critical"},
    }
    result = hc.enforce_worst_wins_system_status(payload)
    assert result == "critical"
    assert payload["system_status"] == "critical"
    assert "worst_wins" in payload.get("system_status_rollup", "")


def test_enforce_worst_wins_ops_critical_with_clear_disk_ssot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ops critical must surface even when disk kill SSOT is clear.

    Prior behavior: with ssot_clear the elevation branch was skipped, so a
    critical monitor report left public system_status=healthy — the G7
    split-brain. The worst-wins assertion is the final word after every
    demotion/elevation branch in the ops merge.
    """
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _write_clear_kill_ssot(data)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(
        hc,
        "_project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(hc, "load_graduation_cb_ssot", lambda *args, **kwargs: None)

    health = _green_ops_health(_thin_signal_health(tmp_path))
    report = {
        "status": "critical",
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

    assert result["ops_health_status"] == "critical"
    assert result["system_status"] == "critical", (
        "worst-wins assertion must elevate system_status for critical ops "
        "even when disk kill SSOT is clear"
    )


def test_enforce_worst_wins_warning_kill_elevates_to_warning() -> None:
    """A non-halt (restrict/warning) kill elevates system_status to warning."""
    from src.monitor import health_check as hc

    payload = {
        "system_status": "healthy",
        "ops_health_status": "ok",
        "kill_switch": {"enabled": True, "level": "restrict", "status": "warning"},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }
    assert hc.enforce_worst_wins_system_status(payload) == "warning"
    assert payload["system_status"] == "warning"


def test_enforce_worst_wins_never_demotes() -> None:
    """The assertion elevates only; a critical badge stays critical on heal."""
    from src.monitor import health_check as hc

    payload = {
        "system_status": "critical",
        "ops_health_status": "ok",
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }
    assert hc.enforce_worst_wins_system_status(payload) == "critical"
    assert "system_status_rollup" not in payload



def _write_clear_kill_ssot(data_dir: Path) -> None:
    """Seed disk kill_switch.json + incidents.json in clear (off / 0) state."""
    (data_dir / "kill_switch.json").write_text(
        json.dumps({"enabled": False, "status": "ok"}), encoding="utf-8"
    )
    (data_dir / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )


def _ok_monitor_report() -> dict:
    """Monitor report with status=ok and green ops checks."""
    return {
        "status": "ok",
        "timestamp": "2026-07-25T07:16:41+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
        "scope": "operational_readiness",
    }


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
    # apply_ops_monitor_to_dashboard_health receives data_dir/public_dir
    # kwargs and calls the mirror-lag projector from its own module (post
    # HEALTH-CHECK-SPLIT); the hub hc.DATA_DIR binding is not read here.
    monkeypatch.setattr(
        "src.monitor.health_dashboard_apply._project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )

    health = _green_ops_health(_thin_signal_health(tmp_path))
    report = _ok_monitor_report()

    result = hc.apply_ops_monitor_to_dashboard_health(
        health, report, data_dir=data, public_dir=public
    )

    assert result["ops_health_status"] == "ok"
    assert result["system_status"] == "healthy"


def test_partial_ops_restamp_clears_sticky_quality_only_system_status_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sticky public system_status=warning must clear when ops plane is green.

    Live residual (office-hours 2026-07-25): after a partial health_job /
    ops_health_merge, public health carried sticky ``system_status=warning``
    and ``ops_health_status=warning`` while the monitor report was
    ``status=ok`` and all ops SLIs were green (kill off, incidents 0,
    scheduler ok, data_pipeline_slo ok, mirror lag 0). The prior
    ``_elevate_public_system_status`` is max-severity (only raises), and the
    ``ssot_clear`` demote branch only fired when sticky kill/open was present
    - leaving a quality-only warning stuck on the ops badge. The alert
    builder then classified the demotion as ``health_slo`` titled
    "Health Warning: ops" because ``ops_health_status`` was non-green.
    """
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _write_clear_kill_ssot(data)
    # Retarget to health_dashboard_apply module globals: the ops merge calls
    # its own bindings, so hub re-export patches would be silent no-ops.
    monkeypatch.setattr(
        "src.monitor.health_dashboard_apply._project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )
    # load_graduation_cb_ssot is a function-local lazy import inside
    # apply_ops_monitor_to_dashboard_health; patch its owner module.
    monkeypatch.setattr(
        "src.monitor.health_freshness_cb.load_graduation_cb_ssot",
        lambda *args, **kwargs: None,
    )

    # Public dashboard health with sticky warning from a prior partial patch
    # and thin SH 1/9 (quality plane). Ops SLIs all green.
    health = {
        "system_status": "warning",
        "ops_health_status": "warning",
        "ops_health_timestamp": "2026-07-25T06:00:00+00:00",
        "generated_at": "2026-07-25T06:00:00+00:00",
        "scheduler_status": {"status": "ok", "backends": {}},
        "data_pipeline_slo": {"status": "ok"},
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "ok",
        "signal_health": _thin_signal_health(tmp_path),
    }
    report = _ok_monitor_report()

    result = hc.apply_ops_monitor_to_dashboard_health(
        health, report, data_dir=data, public_dir=public
    )

    # Ops plane must be green: monitor ok + all ops SLIs green.
    assert result["system_status"] == "healthy", (
        "sticky quality-only system_status=warning must clear on ops-ok restamp"
    )
    assert result["ops_health_status"] == "ok"

    # Alert builder must classify the thin SH as signal_quality, never
    # health_slo titled "Health Warning: ops".
    alerts = build_health_slo_alerts(result)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["type"] == "signal_quality"
    assert "ops" not in (alert.get("title") or "").lower()
    assert alert["signal_quality_badge"] == "1/9 healthy sources"


def test_partial_ops_restamp_keeps_real_ops_failure_demoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real ops failures must still demote the ops badge after partial restamp.

    A sticky warning is not always quality-only: when the monitor report or
    ops SLIs are genuinely degraded (scheduler fail, SLO warn, kill on,
    mirror lag), the partial restamp must preserve or elevate the demotion.
    """
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _write_clear_kill_ssot(data)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(
        hc,
        "_project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hc,
        "load_graduation_cb_ssot",
        lambda *args, **kwargs: None,
    )

    # Monitor report is warning (real ops issue, e.g. scheduler degraded)
    health = {
        "system_status": "warning",
        "ops_health_status": "warning",
        "generated_at": "2026-07-25T06:00:00+00:00",
        "scheduler_status": {"status": "degraded", "backends": {}},
        "data_pipeline_slo": {"status": "ok"},
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_lag_status": "ok",
        "signal_health": _thin_signal_health(tmp_path),
    }
    report = {
        "status": "warning",
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

    # Real ops failure must keep system_status demoted.
    assert result["system_status"] in {"warning", "degraded", "critical"}
    assert result["ops_health_status"] == "warning"


def test_partial_ops_restamp_demotes_when_monitor_report_ok_but_kill_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill switch on disk must elevate ops badge even if monitor report is ok.

    Disk kill SSOT wins over a lagging monitor report. A sticky healthy
    system_status must elevate to warning/critical when kill_switch.json is
    enabled, even if the monitor report status was ok.
    """
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    # Kill enabled on disk
    (data / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "status": "critical",
                "level": "halt",
                "reason": "manual_halt",
                "incident_id": "INC-TEST-1",
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr(
        hc,
        "_project_mirror_lag_onto_dashboard_health",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hc,
        "load_graduation_cb_ssot",
        lambda *args, **kwargs: None,
    )

    health = {
        "system_status": "healthy",
        "ops_health_status": "ok",
        "generated_at": "2026-07-25T06:00:00+00:00",
        "scheduler_status": {"status": "ok", "backends": {}},
        "data_pipeline_slo": {"status": "ok"},
        "kill_switch": {"enabled": False, "status": "ok"},
        "open_incidents": {"open_count": 0, "status": "ok"},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_lag_status": "ok",
        "signal_health": _thin_signal_health(tmp_path),
    }
    report = _ok_monitor_report()

    result = hc.apply_ops_monitor_to_dashboard_health(
        health, report, data_dir=data, public_dir=public
    )

    # Kill on disk must elevate system_status.
    assert result["system_status"] in {"warning", "degraded", "critical"}
    assert result["kill_switch"]["enabled"] is True


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
