"""Batch IU: DQ1 conflict clear-on-heal, DO2 health path, DT1 timeout soft-ok.

Session A plan (IU):
- DQ1: promote_blocked_kill sticky after kill heal → clear markers
- DO2: health cadence must invoke clear (not only dashboard write_promote)
- DT1: wall-clock timeout + fresh producer artifact → scheduler soft-ok
- DT2: shell cron dashboard guard ≥180 (covered in test_batch_ih)

Does not touch signals.json.target_allocations / order_router.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.strategy.graduation_checklist import GraduationChecklist
from tests.test_graduation_promote_ssot import _results


# ---------------------------------------------------------------------------
# DQ1 — kill-gated promote conflict clear-on-heal
# ---------------------------------------------------------------------------


def test_dq1_clear_kill_gated_conflict_when_kill_healed(tmp_path: Path) -> None:
    """Sticky promote_blocked_kill must clear when kill authority is gone."""
    conflict = tmp_path / ".graduation_conflict.json"
    promote = tmp_path / ".promote_to_live"
    body = {
        "graduation_conflict": True,
        "action": "promote_blocked_kill",
        "reason": "kill_authority",
        "is_graduation_ready": False,
        "kill_level": "warning",
        "kill_incident_id": "5d0ad4c8-2436-424c-9305-38d33c831496",
    }
    conflict.write_text(json.dumps(body), encoding="utf-8")
    promote.write_text(json.dumps(body), encoding="utf-8")

    # No kill_switch.json → kill clear
    out = GraduationChecklist().clear_kill_gated_promote_markers(data_dir=tmp_path)
    assert out["kill_clear"] is True
    assert out["cleared"] is True
    assert not conflict.exists()
    assert not promote.exists()


def test_dq1_does_not_clear_while_kill_armed(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "warning",
                "reason": "unresolved_incident:signal_staleness",
                "source": "incident_lifecycle",
            }
        ),
        encoding="utf-8",
    )
    conflict = tmp_path / ".graduation_conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "graduation_conflict": True,
                "action": "promote_blocked_kill",
                "reason": "kill_authority",
            }
        ),
        encoding="utf-8",
    )
    out = GraduationChecklist().clear_kill_gated_promote_markers(data_dir=tmp_path)
    assert out["cleared"] is False
    assert conflict.exists()


def test_dq1_write_promote_clears_kill_tombstone_when_checklist_not_ready(
    tmp_path: Path,
) -> None:
    """Even when checklist fails, kill heal must drop promote_blocked_kill."""
    conflict = tmp_path / ".graduation_conflict.json"
    promote = tmp_path / ".promote_to_live"
    body = {
        "graduation_conflict": True,
        "action": "promote_blocked_kill",
        "reason": "kill_authority",
        "is_graduation_ready": False,
    }
    conflict.write_text(json.dumps(body), encoding="utf-8")
    promote.write_text(json.dumps(body), encoding="utf-8")

    path = GraduationChecklist().write_promote_to_live_if_ready(
        _results(ready=False), data_dir=tmp_path
    )
    assert path is None
    assert not conflict.exists()
    # Kill-gated promote tombstone removed; no checklist re-tombstone of non-candidacy
    assert not promote.exists()


def test_dq1_preserves_checklist_blocked_tombstone(tmp_path: Path) -> None:
    """Checklist-not-ready markers are not kill-gated — leave them."""
    conflict = tmp_path / ".graduation_conflict.json"
    body = {
        "graduation_conflict": True,
        "action": "promote_blocked_checklist",
        "reason": "checklist_not_ready",
    }
    conflict.write_text(json.dumps(body), encoding="utf-8")
    out = GraduationChecklist().clear_kill_gated_promote_markers(data_dir=tmp_path)
    assert out["cleared"] is False
    assert conflict.exists()


# ---------------------------------------------------------------------------
# DO2 — health path invokes clear-on-heal
# ---------------------------------------------------------------------------


def test_do2_run_health_check_clears_kill_gated_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Health cadence (not only dashboard) must clear sticky kill conflict."""
    import src.monitor.health_check as hc

    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path / "public")
    (tmp_path / "public").mkdir(parents=True, exist_ok=True)

    conflict = tmp_path / ".graduation_conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "graduation_conflict": True,
                "action": "promote_blocked_kill",
                "reason": "kill_authority",
                "kill_incident_id": "inc-zombie",
            }
        ),
        encoding="utf-8",
    )

    # Minimal stubs so run_health_check does not depend on live SSOT
    monkeypatch.setattr(
        hc,
        "_check_data_freshness",
        lambda: {
            "prices": {"status": "ok", "age_hours": 0.1},
            "signals": {"status": "ok", "age_hours": 0.1},
            "cron": {"status": "ok", "total_jobs": 0, "failed_jobs": 0},
        },
    )
    monkeypatch.setattr(
        hc,
        "_check_circuit_breaker",
        lambda: {"status": "ok", "state": "closed", "fail_count": 0},
    )
    monkeypatch.setattr(
        hc,
        "_check_kill_switch",
        lambda: {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        hc,
        "_check_open_incidents",
        lambda: {"status": "ok", "open_count": 0, "incidents": []},
    )
    monkeypatch.setattr(
        hc,
        "_check_fred_md_cache",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(hc, "publish_ops_health_surfaces", lambda report: None)
    monkeypatch.setattr(hc, "publish_health_alerts_json", lambda report: None)
    monkeypatch.setattr(
        hc,
        "update_graduation_circuit_breaker_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        hc,
        "attach_shared_freshness_slis_to_ops_report",
        lambda report, data_dir=None: report,
    )
    monkeypatch.setattr(hc, "_stamp_health_self_job_running_success", lambda freshness: None)

    report = hc.run_health_check()
    assert report["status"] in {"ok", "warning", "degraded", "critical"}
    assert not conflict.exists(), "health must clear kill-gated conflict on heal"


# ---------------------------------------------------------------------------
# DT1 — timeout + fresh artifact soft-ok
# ---------------------------------------------------------------------------


def test_dt1_timeout_soft_ok_when_signals_fresh(tmp_path: Path) -> None:
    """Dashboard timeout must not sticky-fail SLO when signals.json is fresh."""
    from src.monitor.hermes_cron import normalize_cron_job

    data = tmp_path / "data"
    data.mkdir()
    signals = data / "signals.json"
    signals.write_text(
        json.dumps({"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}),
        encoding="utf-8",
    )
    # Artifact mtime slightly before timeout finish (partial publish then kill)
    finished = time.time()
    early = finished - 90.0
    import os

    os.utime(signals, (early, early))

    job = {
        "name": "portfolio-lab-dashboard",
        "status": "timeout",
        "last_run": datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
        "duration_seconds": 120.0,
        "enabled": True,
        "state": "scheduled",
    }
    with patch(
        "src.monitor.hermes_cron.recovery_data_dirs",
        return_value=[data],
    ):
        normalized = normalize_cron_job(job, backend="tasker", source="test", now=finished)

    assert normalized["status"] == "ok"
    assert normalized.get("timeout_artifact_reconciled") is True
    assert normalized["timeout_artifact_evidence"]["artifact"] == "signals.json"


def test_dt1_data_timeout_soft_ok_when_prices_written_during_run(
    tmp_path: Path,
) -> None:
    """Data job: prices written early, outer timeout later → soft-ok via duration grace."""
    from src.monitor.hermes_cron import normalize_cron_job

    data = tmp_path / "data"
    data.mkdir()
    prices = data / "prices.json"
    prices.write_text(json.dumps({"SPY": 500.0}), encoding="utf-8")

    finished = time.time()
    # prices ~280s before finish (within 301s duration window)
    early = finished - 280.0
    import os

    os.utime(prices, (early, early))

    job = {
        "name": "portfolio-lab-data",
        "status": "timeout",
        "last_run": datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
        "duration_seconds": 301.0,
        "enabled": True,
        "state": "scheduled",
    }
    with patch(
        "src.monitor.hermes_cron.recovery_data_dirs",
        return_value=[data],
    ):
        normalized = normalize_cron_job(job, backend="tasker", source="test", now=finished)

    assert normalized["status"] == "ok"
    assert normalized.get("timeout_artifact_reconciled") is True


def test_dt1_timeout_not_soft_ok_without_fresh_artifact(tmp_path: Path) -> None:
    from src.monitor.hermes_cron import normalize_cron_job

    data = tmp_path / "data"
    data.mkdir()
    # Stale artifact (2 days old)
    prices = data / "prices.json"
    prices.write_text("{}", encoding="utf-8")
    finished = time.time()
    import os

    os.utime(prices, (finished - 2 * 86400, finished - 2 * 86400))

    job = {
        "name": "portfolio-lab-data",
        "status": "timeout",
        "last_run": datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
        "duration_seconds": 301.0,
        "enabled": True,
    }
    with patch(
        "src.monitor.hermes_cron.recovery_data_dirs",
        return_value=[data],
    ):
        normalized = normalize_cron_job(job, backend="tasker", source="test", now=finished)

    assert normalized["status"] == "error"
    assert not normalized.get("timeout_artifact_reconciled")
