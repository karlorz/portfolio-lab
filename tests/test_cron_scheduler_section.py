"""Tests for cron_scheduler_section builder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.dashboard.cron_scheduler_section import (
    build_cron_scheduler_section,
    cron_scheduler_unavailable_payload,
    refresh_public_health_cron_section,
)


def test_cron_scheduler_unavailable_payload_shape() -> None:
    payload = cron_scheduler_unavailable_payload(RuntimeError("boom"))
    assert payload["cron_jobs"] == []
    assert payload["scheduler_status"]["status"] == "unavailable"
    assert "boom" in payload["scheduler_status"]["error"]


def test_build_cron_scheduler_section_local_only(tmp_path: Path) -> None:
    local_jobs = [{"name": "signals", "status": "ok"}]
    local_backend = {"status": "ok", "backend": "local"}
    with patch("src.monitor.hermes_cron.load_local_cron_jobs", return_value=(local_jobs, local_backend)) as ll:
        out = build_cron_scheduler_section(cron_status_file=tmp_path / "cron_status.json")
    ll.assert_called_once()
    assert out["cron_jobs"] == local_jobs
    assert out["scheduler_status"]["status"] == "ok"
    assert out["scheduler_status"]["backends"]["local"] is local_backend
    assert "hermes" not in out["scheduler_status"]["backends"]


def test_build_cron_scheduler_section_with_hermes(tmp_path: Path) -> None:
    local_backend = {"status": "ok", "backend": "local"}
    hermes_backend = {"status": "ok", "backend": "hermes"}
    hermes_jobs = [{"name": "eval", "status": "ok"}]
    with patch("src.monitor.hermes_cron.load_local_cron_jobs", return_value=([], local_backend)):
        with patch("src.monitor.hermes_cron.load_hermes_portfolio_cron_jobs", return_value=(hermes_jobs, hermes_backend)) as lh:
            out = build_cron_scheduler_section(
                cron_status_file=tmp_path / "cron_status.json",
                resolve_hermes_path=lambda: tmp_path / "jobs.json",
            )
    lh.assert_called_once_with(tmp_path / "jobs.json")
    assert out["cron_jobs"] == hermes_jobs
    assert out["scheduler_status"]["backends"]["hermes"] is hermes_backend


def test_build_cron_scheduler_section_no_hermes_when_path_none(tmp_path: Path) -> None:
    local_backend = {"status": "ok", "backend": "local"}
    with patch("src.monitor.hermes_cron.load_local_cron_jobs", return_value=([], local_backend)):
        with patch("src.monitor.hermes_cron.load_hermes_portfolio_cron_jobs") as lh:
            out = build_cron_scheduler_section(
                cron_status_file=tmp_path / "cron_status.json",
                resolve_hermes_path=lambda: None,
            )
    lh.assert_not_called()
    assert "hermes" not in out["scheduler_status"]["backends"]


def test_build_cron_scheduler_section_import_failure(tmp_path: Path) -> None:
    with patch("src.monitor.hermes_cron.load_local_cron_jobs", side_effect=ImportError("no module")):
        out = build_cron_scheduler_section(cron_status_file=tmp_path / "cron_status.json")
    assert out["cron_jobs"] == []
    assert out["scheduler_status"]["status"] == "unavailable"
    assert "no module" in out["scheduler_status"]["error"]


def test_build_cron_scheduler_section_log_error_invoked(tmp_path: Path) -> None:
    calls: list[tuple[str, Exception]] = []

    def log_error(name: str, exc: Exception) -> None:
        calls.append((name, exc))

    with patch("src.monitor.hermes_cron.load_local_cron_jobs", side_effect=OSError("io fail")):
        out = build_cron_scheduler_section(
            cron_status_file=tmp_path / "cron_status.json",
            log_error=log_error,
        )
    assert out["scheduler_status"]["status"] == "unavailable"
    assert len(calls) == 1
    assert calls[0][0] == "cron_scheduler"
    assert isinstance(calls[0][1], OSError)


def test_refresh_public_health_cron_section_updates_stale_data_last_run(tmp_path: Path) -> None:
    """Post-stamp refresh must pull portfolio-lab-data last_run after cron_update."""
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()
    (data_dir / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-data",
                        "status": "success",
                        "last_run": "2026-07-20T14:06:24+00:00",
                        "enabled": True,
                        "state": "scheduled",
                    },
                    {
                        "name": "portfolio-lab-health",
                        "status": "error",
                        "last_run": "2026-07-20T14:00:06+00:00",
                        "enabled": True,
                        "state": "scheduled",
                    },
                ],
            }
        )
    )
    health_path = public_dir / "health.json"
    health_path.write_text(
        json.dumps(
            {
                "system_status": "warning",
                "cron_jobs": [
                    {
                        "name": "portfolio-lab-data",
                        "status": "ok",
                        "last_run": "2026-07-20T13:06:29+00:00",
                    }
                ],
                "scheduler_status": {
                    "status": "degraded",
                    "backends": {
                        "local": {"status": "degraded", "failed_jobs": 1},
                    },
                },
                "data_freshness": {},
            }
        )
    )

    wrote = refresh_public_health_cron_section(
        public_health_path=health_path,
        cron_status_file=data_dir / "cron_status.json",
    )
    assert wrote is True
    health = json.loads(health_path.read_text())
    data_row = next(j for j in health["cron_jobs"] if j["name"] == "portfolio-lab-data")
    assert data_row["last_run"] == "2026-07-20T14:06:24+00:00"
    # Health self-job error must not inflate backend failed_jobs.
    assert health["scheduler_status"]["backends"]["local"]["failed_jobs"] == 0
    assert health["scheduler_status"]["status"] == "ok"
    assert health.get("cron_section_refreshed_at")
