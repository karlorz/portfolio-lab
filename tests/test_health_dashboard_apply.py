"""Unit tests for src.monitor.health_dashboard_apply (Item Q48).

Tests cover:
- apply_ops_monitor_to_dashboard_health (clean ops merge, kill-status elevation,
  degraded status retention, worst-wins enforcement)
- _project_mirror_lag_onto_dashboard_health (lagging/critical badges, empty
  catalog handling, stamp fallback)
- refresh_signals_health_kill_fields (live kill switch / incident authority
  synchronization onto signals.json health)
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.monitor import health_dashboard_apply as hda


def _write_signals_public(
    public: Path,
    private: Path,
) -> None:
    """Write a champion signals.json twin into public + private tmp dirs."""
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "generated_at": "2026-08-17T00:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "health": {"status": "ok", "signal_health_healthy": 1},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")


def test_apply_clean_ops_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monitor report ok merges ops_health_* stamps and keeps system_status."""
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 12,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    report = {
        "status": "ok",
        "timestamp": "2026-08-17T08:00:00Z",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        },
    }
    health_data = {"system_status": "healthy"}
    result = hda.apply_ops_monitor_to_dashboard_health(
        health_data,
        report,
        data_dir=data_dir,
        public_dir=public_dir,
    )

    assert result["ops_health_status"] == "ok"
    assert result["ops_health_timestamp"] == "2026-08-17T08:00:00Z"
    assert result["ops_health_source"] == "monitor.health_check"
    # Disk kill/open re-projected (clear)
    assert result["kill_switch"]["enabled"] is False
    assert result["open_incidents"]["open_count"] == 0
    # Mirror lag SLI stamped clean
    assert result["repo_public_mirror_lag_status"] == "ok"
    assert result["repo_public_mirror_lag_badge"] == "lagging=0/12"
    # Final worst-wins rollup keeps healthy
    assert result["system_status"] == "healthy"


def test_apply_elevates_status_on_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Armed kill switch on disk elevates dashboard system_status to critical."""
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()
    (data_dir / "kill_switch.json").write_text(
        json.dumps({"enabled": True, "level": "halt", "reason": "test"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 0,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    report = {
        "status": "ok",
        "timestamp": "2026-08-17T08:00:00Z",
        "checks": {
            "kill_switch": {"enabled": True, "status": "critical"},
            "open_incidents": {"status": "ok", "open_count": 0},
        },
    }
    result = hda.apply_ops_monitor_to_dashboard_health(
        {"system_status": "healthy"},
        report,
        data_dir=data_dir,
        public_dir=public_dir,
    )

    assert result["kill_switch"]["enabled"] is True
    # Disk kill SSOT identity wins over report
    assert result["kill_switch"]["level"] == "halt"
    assert result["system_status"] in {"critical", "degraded", "warning"}


def test_apply_retains_degraded_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded ops report retains (does not wipe) degraded system_status."""
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 0,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    report = {
        "status": "degraded",
        "timestamp": "2026-08-17T08:00:00Z",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"status": "ok", "open_count": 0},
        },
    }
    result = hda.apply_ops_monitor_to_dashboard_health(
        {"system_status": "degraded"},
        report,
        data_dir=data_dir,
        public_dir=public_dir,
    )

    assert result["ops_health_status"] == "degraded"
    # SLO-derived degraded is not demoted by the clean-kill branch
    assert result["system_status"] == "degraded"


def test_project_mirror_lag_lagging_badge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lagging probe stamps lag status/badge and soft-elevates system_status."""
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 3,
            "total": 33,
            "lagging_paths": ["a.json", "b.json"],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    ops_report: dict[str, Any] = {
        "repo_public_mirror_lagging_count": 3,
        "repo_public_mirror_total": 33,
    }
    health_data: dict[str, Any] = {"system_status": "healthy", "status": "ok"}
    hda._project_mirror_lag_onto_dashboard_health(
        health_data,
        ops_report,
        data_dir=tmp_path,
        public_dir=tmp_path,
    )

    assert health_data["repo_public_mirror_lagging_count"] == 3
    assert health_data["repo_public_mirror_lag_status"] == "lagging"
    assert health_data["repo_public_mirror_lag_badge"] == "lagging=3/33"
    assert health_data["repo_public_mirror_lagging_paths"] == ["a.json", "b.json"]
    # Soft-elevate: healthy -> warning (ops hygiene, not trading halt)
    assert health_data["system_status"] == "warning"
    assert health_data["repo_public_mirror_lag"]["status"] == "lagging"


def test_project_mirror_lag_critical_badge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical threshold probes stamp critical status."""
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 15,
            "total": 20,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    health_data: dict[str, Any] = {"system_status": "healthy"}
    hda._project_mirror_lag_onto_dashboard_health(
        health_data,
        {"repo_public_mirror_lagging_count": 15, "repo_public_mirror_total": 20},
        data_dir=tmp_path,
        public_dir=tmp_path,
    )

    assert health_data["repo_public_mirror_lag_status"] == "critical"
    assert health_data["repo_public_mirror_lag_badge"] == "lagging=15/20"
    assert health_data["system_status"] == "warning"


def test_project_mirror_lag_empty_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty catalog probe stamps unknown status with no_catalog badge."""
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 0,
            "lagging_paths": [],
            "source": "/live/data",
            "dest": "/repo/public/data",
            "ok": True,
        },
    )

    health_data: dict[str, Any] = {"system_status": "healthy"}
    hda._project_mirror_lag_onto_dashboard_health(
        health_data,
        {},
        data_dir=tmp_path,
        public_dir=tmp_path,
    )

    assert health_data["repo_public_mirror_lag_status"] == "unknown"
    assert health_data["repo_public_mirror_lag_badge"] == "no_catalog"
    assert health_data["repo_public_mirror_lagging_paths"] == []
    # Unknown lag does not soft-elevate
    assert health_data["system_status"] == "healthy"


def test_project_mirror_lag_stamp_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the live probe fails, ops-report stamp alone is projected."""
    def _fail_probe(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        _fail_probe,
    )

    health_data: dict[str, Any] = {"system_status": "healthy"}
    hda._project_mirror_lag_onto_dashboard_health(
        health_data,
        {
            "repo_public_mirror_lagging_count": 2,
            "repo_public_mirror_total": 10,
            "repo_public_mirror_lagging_paths": ["c.json"],
            "repo_public_mirror_lag_status": "lagging",
        },
        data_dir=tmp_path,
        public_dir=tmp_path,
    )

    assert health_data["repo_public_mirror_lag_status"] == "lagging"
    assert health_data["repo_public_mirror_lagging_count"] == 2
    assert health_data["mirror_lag_source_of_truth"] == "stamp"
    assert health_data["mirror_lag_stamp_lagging_count"] == 2


def test_refresh_signals_health_kill_fields_armed(
    tmp_path: Path,
) -> None:
    """Live disk kill authority is projected onto signals.json#health."""
    public_dir = tmp_path / "public"
    data_dir = tmp_path / "data"
    public_dir.mkdir()
    data_dir.mkdir()
    _write_signals_public(public_dir, data_dir)

    (data_dir / "kill_switch.json").write_text(
        json.dumps(
            {"enabled": True, "level": "halt", "reason": "test_halt", "status": "critical"}
        ),
        encoding="utf-8",
    )
    (data_dir / "incidents.json").write_text(
        json.dumps({"open_count": 1, "status": "critical"}),
        encoding="utf-8",
    )

    report = {
        "status": "ok",
        "timestamp": "2026-08-17T08:00:00Z",
        "signal_health": {"healthy": 1, "unhealthy": 0},
    }
    hda.refresh_signals_health_kill_fields(
        report, public_dir=public_dir, data_dir=data_dir
    )

    out = json.loads((public_dir / "signals.json").read_text(encoding="utf-8"))
    h = out["health"]
    assert h["kill_switch_enabled"] is True
    assert h["kill_switch_level"] == "halt"
    assert h["open_incidents_count"] == 1
    # Partial-patch honesty
    assert out["generator_git_sha_status"] == "partial_patch"
    assert out["content_patch_source"] == "health_kill_refresh"
    assert h["ensemble_may_lag_full_generate"] is True
    # Authority keys preserved
    assert out["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}


def test_refresh_signals_health_kill_fields_clear(
    tmp_path: Path,
) -> None:
    """Cleared kill switch is projected as enabled=false / level null."""
    public_dir = tmp_path / "public"
    data_dir = tmp_path / "data"
    public_dir.mkdir()
    data_dir.mkdir()
    _write_signals_public(public_dir, data_dir)

    (data_dir / "kill_switch.json").write_text(
        json.dumps({"enabled": False, "status": "ok"}),
        encoding="utf-8",
    )
    (data_dir / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}),
        encoding="utf-8",
    )

    report = {"status": "ok", "timestamp": "2026-08-17T08:00:00Z"}
    hda.refresh_signals_health_kill_fields(
        report, public_dir=public_dir, data_dir=data_dir
    )

    out = json.loads((public_dir / "signals.json").read_text(encoding="utf-8"))
    h = out["health"]
    assert h["kill_switch_enabled"] is False
    assert h.get("kill_switch_level") is None
    assert h["open_incidents_count"] == 0


def test_refresh_signals_health_kill_fields_missing_public(
    tmp_path: Path,
) -> None:
    """Absent public signals.json is a no-op (never create authority empty files)."""
    public_dir = tmp_path / "public"
    data_dir = tmp_path / "data"
    public_dir.mkdir()
    data_dir.mkdir()

    hda.refresh_signals_health_kill_fields(
        {"status": "ok"},
        public_dir=public_dir,
        data_dir=data_dir,
    )

    assert not (public_dir / "signals.json").exists()