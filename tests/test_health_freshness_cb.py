"""Unit tests for src.monitor.health_freshness_cb (Item Q51).

Tests cover:
- _resolve_freshness_public_root (under pytest vs fallback)
- _freshness_artifact_check (ok, stale, missing, root priority)
- _check_data_freshness (prices, signals, cron jobs)
- _check_circuit_breaker (closed ok, degraded, unavailable fallback)
- update_graduation_circuit_breaker_state (consecutive_ok climb capped at 30,
  reset on broker open/trips, reset on ops failure, signal_health outage hold)
- load_graduation_cb_ssot & project_graduation_cb_onto_report & project_graduation_cb_onto_compact_health
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from src.monitor import health_freshness_cb as hfc


def test_resolve_freshness_public_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Under pytest, returns PUBLIC_DATA_DIR directly
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_case")
    monkeypatch.setattr(hfc, "PUBLIC_DATA_DIR", tmp_path / "public")
    assert hfc._resolve_freshness_public_root() == tmp_path / "public"

    # Outside pytest with ephemeral path, falls back to live WWW or DEFAULT_LIVE_PUBLIC_DATA_DIR
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hfc, "PUBLIC_DATA_DIR", "/tmp/plab-pytest-public.abc/data")
    assert str(hfc._resolve_freshness_public_root()) in ("/var/www/portfolio-lab/data", "/tmp/plab-pytest-public.abc/data")


def test_freshness_artifact_check_ok_and_stale(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    f = root / "prices.json"
    f.write_text("{}", encoding="utf-8")

    # Fresh (< 24h)
    res_ok = hfc._freshness_artifact_check(
        basenames=("prices.json",),
        roots=[root],
        stale_hours=24.0,
    )
    assert res_ok["status"] == "ok"
    assert res_ok["age_hours"] < 1.0
    assert res_ok["path"] == str(f)

    # Set mtime to 30 hours ago
    past_mtime = time.time() - (30 * 3600)
    os.utime(f, (past_mtime, past_mtime))

    res_stale = hfc._freshness_artifact_check(
        basenames=("prices.json",),
        roots=[root],
        stale_hours=24.0,
    )
    assert res_stale["status"] == "stale"
    assert res_stale["age_hours"] >= 30.0


def test_freshness_artifact_check_missing_and_roots_fallback(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()

    # Missing in empty root
    res_missing = hfc._freshness_artifact_check(
        basenames=("signals.json",),
        roots=[empty_root],
        stale_hours=4.0,
    )
    assert res_missing["status"] == "missing"
    assert res_missing["age_hours"] is None

    # Fallback to secondary root
    (fallback_root / "signals.json").write_text("{}", encoding="utf-8")
    res_fallback = hfc._freshness_artifact_check(
        basenames=("signals.json",),
        roots=[empty_root, fallback_root],
        stale_hours=4.0,
    )
    assert res_fallback["status"] == "ok"
    assert res_fallback["path"] == str(fallback_root / "signals.json")


def test_check_data_freshness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_dir = tmp_path / "public"
    private_dir = tmp_path / "data"
    public_dir.mkdir()
    private_dir.mkdir()

    (public_dir / "prices.json").write_text("{}", encoding="utf-8")
    (public_dir / "signals.json").write_text("{}", encoding="utf-8")
    (private_dir / "cron_status.json").write_text(
        json.dumps({
            "jobs": [
                {"name": "fetch-data", "status": "success", "enabled": True}
            ],
            "backend": "tasker",
            "status": "ok",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(hfc, "PUBLIC_DATA_DIR", public_dir)
    monkeypatch.setattr(hfc, "DATA_DIR", private_dir)

    freshness = hfc._check_data_freshness()
    assert freshness["prices"]["status"] == "ok"
    assert freshness["signals"]["status"] == "ok"
    assert "cron" in freshness
    assert freshness["cron"]["total_jobs"] >= 1
    assert freshness["cron"]["failed_jobs"] == 0


def test_check_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    # State closed -> ok
    monkeypatch.setattr(
        "src.broker.circuit_breaker.get_circuit_state",
        lambda: {"state": "closed", "fail_count": 0, "reset_timeout": 60},
    )
    cb_ok = hfc._check_circuit_breaker()
    assert cb_ok["status"] == "ok"
    assert cb_ok["state"] == "closed"

    # State open -> degraded
    monkeypatch.setattr(
        "src.broker.circuit_breaker.get_circuit_state",
        lambda: {"state": "open", "fail_count": 3, "reset_timeout": 60},
    )
    cb_open = hfc._check_circuit_breaker()
    assert cb_open["status"] == "degraded"
    assert cb_open["state"] == "open"


def test_update_graduation_circuit_breaker_state_climb(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Step 1: 0 -> 1
    res1 = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
    )
    assert res1["status"] == "green"
    assert res1["consecutive_ok"] == 1
    assert res1["trips"] == 0

    # Step 2: Seed 29 and verify cap at 30
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 29, "status": "green", "trips": 0}),
        encoding="utf-8",
    )
    res2 = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
    )
    assert res2["consecutive_ok"] == 30

    # Step 3: Verify 30 stays capped at 30
    res3 = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
    )
    assert res3["consecutive_ok"] == 30


def test_update_graduation_circuit_breaker_state_resets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Reset on broker open
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 10, "status": "green", "trips": 0}),
        encoding="utf-8",
    )
    res_open = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "open", "fail_count": 5},
        data_dir=data_dir,
    )
    assert res_open["consecutive_ok"] == 0
    assert res_open["status"] == "red"
    assert res_open["trips"] == 1

    # Reset on ops failure (system_status degraded)
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 15, "status": "green", "trips": 1}),
        encoding="utf-8",
    )
    res_ops_fail = hfc.update_graduation_circuit_breaker_state(
        system_status="degraded",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
    )
    assert res_ops_fail["consecutive_ok"] == 0
    assert res_ops_fail["status"] == "yellow"


def test_update_graduation_circuit_breaker_signal_health_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 8, "status": "green", "trips": 0}),
        encoding="utf-8",
    )

    # Hard quality outage (summary healthy: 0 -> degraded) freezes climb (holds at 8)
    res_hold = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
        signal_health={"summary": {"healthy": 0, "unhealthy": 9, "total_tracked": 9}},
    )
    assert res_hold["consecutive_ok"] == 8
    assert res_hold["signal_health_blocked"] is True
    assert res_hold["status"] == "yellow"

    # Warning signal health (partial healthy sleeves) does NOT freeze climb (increments 8 -> 9)
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 8, "status": "green", "trips": 0}),
        encoding="utf-8",
    )
    res_warn = hfc.update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=data_dir,
        signal_health={"summary": {"healthy": 1, "degraded": 5, "unhealthy": 3, "total_tracked": 9}},
    )
    assert res_warn["consecutive_ok"] == 9
    assert res_warn.get("signal_health_blocked") is not True


def test_load_and_project_graduation_cb_ssot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Missing SSOT
    assert hfc.load_graduation_cb_ssot(data_dir) is None
    proj_missing = hfc.project_graduation_cb_onto_report({}, data_dir=data_dir)
    assert proj_missing["graduation_circuit_breaker"]["graduation_cb_source"] == "missing"

    compact_missing = hfc.project_graduation_cb_onto_compact_health({}, data_dir=data_dir)
    assert compact_missing["graduation_cb_source"] == "missing"

    # Present SSOT
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({
            "consecutive_ok": 12,
            "status": "green",
            "updated_at": "2026-08-17T08:00:00Z",
            "signal_health_blocked": False,
            "producer": "test",
        }),
        encoding="utf-8",
    )

    ssot = hfc.load_graduation_cb_ssot(data_dir)
    assert ssot is not None
    assert ssot["consecutive_ok"] == 12

    report: dict[str, Any] = {}
    hfc.project_graduation_cb_onto_report(report, data_dir=data_dir, ssot=ssot)
    assert report["graduation_circuit_breaker"]["consecutive_ok"] == 12
    assert report["graduation_circuit_breaker"]["graduation_cb_source"] == "disk_ssot"

    compact: dict[str, Any] = {}
    hfc.project_graduation_cb_onto_compact_health(compact, data_dir=data_dir, ssot=ssot)
    assert compact["graduation_circuit_breaker_consecutive_ok"] == 12
    assert compact["graduation_cb_source"] == "disk_ssot"


def test_reconcile_graduation_cb_projection(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    health_file = data_dir / "health.json"

    # No health.json file -> returns False
    assert hfc.reconcile_graduation_cb_projection(data_dir=data_dir) is False

    # Seed health.json with stale consecutive_ok
    health_file.write_text(
        json.dumps({"graduation_circuit_breaker": {"consecutive_ok": 0, "status": "yellow"}}),
        encoding="utf-8",
    )
    # Seed .circuit_breaker.json
    (data_dir / ".circuit_breaker.json").write_text(
        json.dumps({"consecutive_ok": 5, "status": "green", "signal_health_blocked": False}),
        encoding="utf-8",
    )

    wrote = hfc.reconcile_graduation_cb_projection(data_dir=data_dir, health_path=health_file)
    assert wrote is True
    updated = json.loads(health_file.read_text(encoding="utf-8"))
    assert updated["graduation_circuit_breaker"]["consecutive_ok"] == 5
    assert updated["graduation_circuit_breaker"]["graduation_cb_source"] == "disk_ssot"

