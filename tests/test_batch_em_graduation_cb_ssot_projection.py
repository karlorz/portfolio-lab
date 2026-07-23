"""Batch EM: re-project graduation CB SSOT onto health surfaces.

Live friction (c407): Batch EL climbed ``.circuit_breaker.json`` to
consecutive_ok=1 / green, but sticky ``data/health.json`` still showed
graduation_circuit_breaker consecutive_ok=0 / yellow until the next full
health job. Dual-surface split-brain for operators.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.health_check import (
    load_graduation_cb_ssot,
    project_graduation_cb_onto_compact_health,
    project_graduation_cb_onto_report,
    reconcile_graduation_cb_projection,
    refresh_signals_health_kill_fields,
    update_graduation_circuit_breaker_state,
)


def _write_ssot(root: Path, **fields) -> None:
    payload = {
        "schema_version": "graduation-circuit-breaker/v1",
        "status": "green",
        "consecutive_ok": 1,
        "trips": 0,
        "broker_state": "closed",
        "broker_fail_count": 0,
        "system_status": "ok",
        "updated_at": "2026-07-22T08:49:50.532465+00:00",
        "producer": "test",
        "signal_health_blocked": False,
        "signal_health_contribution": None,
    }
    payload.update(fields)
    (root / ".circuit_breaker.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def test_project_report_overwrites_sticky_stale_yellow(tmp_path: Path) -> None:
    _write_ssot(tmp_path, consecutive_ok=3, status="green")
    sticky = {
        "status": "ok",
        "graduation_circuit_breaker": {
            "consecutive_ok": 0,
            "status": "yellow",
            "updated_at": "2026-07-22T08:30:00+00:00",
            "signal_health_blocked": True,
            "signal_health_contribution": "warning",
        },
    }
    out = project_graduation_cb_onto_report(sticky, data_dir=tmp_path)
    g = out["graduation_circuit_breaker"]
    assert g["consecutive_ok"] == 3
    assert g["status"] == "green"
    assert g["graduation_cb_source"] == "disk_ssot"
    assert g.get("signal_health_blocked") is False


def test_reconcile_rewrites_monitor_health_when_stale(tmp_path: Path) -> None:
    _write_ssot(tmp_path, consecutive_ok=2, status="green")
    health_path = tmp_path / "health.json"
    health_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "graduation_circuit_breaker": {
                    "consecutive_ok": 0,
                    "status": "yellow",
                    "updated_at": "old",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert reconcile_graduation_cb_projection(
        data_dir=tmp_path, health_path=health_path
    )
    on_disk = json.loads(health_path.read_text(encoding="utf-8"))
    g = on_disk["graduation_circuit_breaker"]
    assert g["consecutive_ok"] == 2
    assert g["status"] == "green"
    assert g["graduation_cb_source"] == "disk_ssot"
    assert on_disk.get("graduation_cb_reconciled_at")


def test_reconcile_noop_when_already_aligned(tmp_path: Path) -> None:
    _write_ssot(tmp_path, consecutive_ok=1, status="green")
    health_path = tmp_path / "health.json"
    projected = project_graduation_cb_onto_report({}, data_dir=tmp_path)
    health_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "graduation_circuit_breaker": projected["graduation_circuit_breaker"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert not reconcile_graduation_cb_projection(
        data_dir=tmp_path, health_path=health_path
    )


def test_producer_projects_onto_existing_health(tmp_path: Path) -> None:
    """update_graduation_circuit_breaker_state dual-writes projection (Batch EM)."""
    health_path = tmp_path / "health.json"
    health_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "graduation_circuit_breaker": {
                    "consecutive_ok": 0,
                    "status": "yellow",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    p = update_graduation_circuit_breaker_state(
        system_status="ok",
        broker_circuit={"state": "closed", "fail_count": 0},
        data_dir=tmp_path,
        signal_health={
            "summary": {
                "healthy": 1,
                "degraded": 6,
                "unhealthy": 2,
                "total_tracked": 9,
            }
        },
    )
    assert p["consecutive_ok"] == 1
    assert p["status"] == "green"
    on_disk = json.loads(health_path.read_text(encoding="utf-8"))
    g = on_disk["graduation_circuit_breaker"]
    assert g["consecutive_ok"] == 1
    assert g["status"] == "green"
    assert g["graduation_cb_source"] == "disk_ssot"


def test_partial_signals_health_patch_projects_cb(tmp_path, monkeypatch) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    _write_ssot(private, consecutive_ok=4, status="green")
    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {
            "status": "ok",
            "graduation_circuit_breaker_consecutive_ok": 0,
            "graduation_circuit_breaker_status": "yellow",
        }
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)

    refresh_signals_health_kill_fields(
        {"status": "ok", "kill_switch": {"enabled": False}},
        public_dir=public,
        data_dir=private,
    )
    h = json.loads((public / "signals.json").read_text(encoding="utf-8")).get(
        "health"
    ) or {}
    assert h.get("graduation_circuit_breaker_consecutive_ok") == 4
    assert h.get("graduation_circuit_breaker_status") == "green"
    assert h.get("graduation_cb_source") == "disk_ssot"


def test_load_ssot_missing_returns_none(tmp_path: Path) -> None:
    assert load_graduation_cb_ssot(tmp_path) is None
    h = project_graduation_cb_onto_compact_health({}, data_dir=tmp_path)
    assert h.get("graduation_cb_source") == "missing"


def test_apply_ops_projects_graduation_cb_onto_dashboard_health(
    tmp_path: Path, monkeypatch
) -> None:
    """Batch IG: public dashboard health.json must surface graduation CB SSOT.

    Residual after EM: signals.health + private ops carry CB; public
    health.json only had ops_health_* (SPA split-brain on consecutive_ok).
    """
    from src.monitor.health_check import apply_ops_monitor_to_dashboard_health

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _write_ssot(data, consecutive_ok=5, status="green")
    (data / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )
    # Monitor ops report without nested graduation block (merge must use SSOT)
    ops = {
        "status": "warning",
        "timestamp": "2026-07-23T12:00:00+00:00",
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "ok",
    }
    (data / "health.json").write_text(json.dumps(ops), encoding="utf-8")

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 0,
            "total": 36,
            "lagging_paths": [],
            "source": str(public),
            "dest": str(public),
            "ok": True,
        },
    )

    dash = {
        "system_status": "warning",
        "generated_at": "2026-07-23T11:00:00+00:00",
        "cron_jobs": [],
        # Sticky stale compact keys from prior cycle
        "graduation_circuit_breaker_consecutive_ok": 0,
        "graduation_circuit_breaker_status": "yellow",
    }
    out = apply_ops_monitor_to_dashboard_health(
        dash, ops, data_dir=data, public_dir=public
    )
    assert out["graduation_circuit_breaker_consecutive_ok"] == 5
    assert out["graduation_circuit_breaker_status"] == "green"
    assert out["graduation_cb_source"] == "disk_ssot"
    # Nested block for operators who expect ops-shape keys on dashboard too
    nested = out.get("graduation_circuit_breaker")
    assert isinstance(nested, dict)
    assert nested.get("consecutive_ok") == 5
    assert nested.get("status") == "green"
    assert nested.get("graduation_cb_source") == "disk_ssot"
