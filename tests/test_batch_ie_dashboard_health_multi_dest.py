"""Batch IE: generate_health_json multi-dest (public + repo soft-mirror).

Session A residual after ID (HM Cases DA–DC P1 dashboard multi-dest):
- ``generate_health_json`` still used single-path ``save_results_json`` to
  PUBLIC_DIR/health.json only — no serialize-once repo soft-mirror.
- Private DATA_DIR/health.json remains the monitor schema SSOT and must
  **not** be overwritten by the dashboard health payload (dual schema).
- Live: public/repo health often re-equalized by later soft-mirror, but the
  full-generate path itself must land same-bytes @ 0o644 without waiting
  for a separate mirror job (parity with Batch IC merge path).

Authority: never touches ``signals.json.target_allocations`` / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


def _minimal_generator(tmp_path, monkeypatch, *, public, private, repo):
    """Build a DashboardGenerator with isolated paths and quiet externals."""
    from src.dashboard import generator as gen_mod

    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)
    monkeypatch.setattr(gen_mod, "DATA_DIR", private)
    monkeypatch.setattr(gen_mod, "PUBLIC_DATA_DIR", public)

    # Path constants used by project helpers / kill loaders
    monkeypatch.setattr("src.paths.DATA_DIR", private)
    monkeypatch.setattr("src.paths.PUBLIC_DATA_DIR", public)

    monkeypatch.setattr(
        gen_mod,
        "build_cron_scheduler_section",
        lambda **kwargs: {
            "cron_jobs": [],
            "scheduler_status": {"status": "ok", "backends": {}},
        },
    )
    monkeypatch.setattr(
        gen_mod,
        "build_data_freshness_section",
        lambda **kwargs: {"data_freshness": {}},
    )
    monkeypatch.setattr(
        gen_mod,
        "build_signal_health_section",
        lambda **kwargs: {
            "status": "ok",
            "overall_health": "ok",
            "summary": {"healthy": 2},
            "scores": {},
        },
    )
    monkeypatch.setattr(
        gen_mod,
        "build_fred_readiness_section",
        lambda **kwargs: {
            "status": "ok",
            "readiness": "ok",
            "ready": True,
            "blocking": False,
        },
    )
    monkeypatch.setattr(
        gen_mod,
        "build_data_pipeline_slo_section",
        lambda **kwargs: {"status": "ok", "dimensions": {}},
    )
    monkeypatch.setattr(
        gen_mod,
        "load_kill_switch_payload",
        lambda *a, **k: {"enabled": False},
    )
    monkeypatch.setattr(
        gen_mod,
        "project_kill_switch_fields",
        lambda payload: {
            "status": "ok",
            "enabled": False,
            "level": None,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        gen_mod,
        "load_open_incidents_summary",
        lambda *a, **k: {"status": "ok", "open_count": 0, "incidents": []},
    )
    monkeypatch.setattr(
        gen_mod,
        "derive_system_status",
        lambda **kwargs: "healthy",
    )
    monkeypatch.setattr(
        gen_mod,
        "elevate_system_status_for_kill",
        lambda current, *a, **k: current,
    )
    monkeypatch.setattr(
        gen_mod,
        "summarize_stale_symbol_count",
        lambda *a, **k: 0,
    )
    monkeypatch.setattr(
        gen_mod,
        "_stamp_generator_git_sha",
        lambda data: {**data, "generator_git_sha": "batch-ie-test"},
    )

    # Quiet ops merge / monitor reconcile (import inside generate_health_json)
    monkeypatch.setattr(
        "src.monitor.health_check.apply_ops_monitor_to_dashboard_health",
        lambda health_data, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        "src.monitor.health_check.reconcile_monitor_health_with_disk_ssot",
        lambda **kwargs: None,
        raising=False,
    )

    gen = gen_mod.DashboardGenerator.__new__(gen_mod.DashboardGenerator)
    gen.conn = MagicMock()
    return gen


def test_generate_health_json_public_repo_same_bytes_0644(tmp_path, monkeypatch):
    """Case DV: full generate multi-dests dashboard health public+repo @ 0o644.

    Private DATA_DIR/health.json must remain untouched (monitor schema SSOT).
    """
    from src.dashboard import generator as gen_mod
    from src.monitor import signal_authority as sa

    public = tmp_path / "www"
    private = tmp_path / "data"
    repo = tmp_path / "repo_public"
    public.mkdir()
    private.mkdir()
    repo.mkdir()

    # Pre-seed private monitor health with distinct schema so overwrite is detectable
    private_monitor = {
        "status": "ok",
        "scope": "operational_readiness",
        "service": "portfolio-lab",
        "checks": {"probe": True},
        "timestamp": "2026-07-23T00:00:00+00:00",
    }
    (private / "health.json").write_text(
        json.dumps(private_monitor, indent=2) + "\n", encoding="utf-8"
    )
    private_before = (private / "health.json").read_bytes()

    # Under pytest auto soft-mirror is off — force explicit repo path for contract.
    real_write = sa.write_json_multi_dest

    def _write_with_repo(payload, **kwargs):
        # Only inject repo for health.json dashboard fan-out
        pub = kwargs.get("public_path")
        if pub is not None and Path(pub).name == "health.json":
            kwargs = {
                **kwargs,
                "repo_path": repo / "health.json",
                # Must never receive private DATA_DIR/health.json
                "private_path": None,
            }
        return real_write(payload, **kwargs)

    monkeypatch.setattr(sa, "write_json_multi_dest", _write_with_repo)
    monkeypatch.setattr(
        "src.monitor.signal_authority.write_json_multi_dest", _write_with_repo
    )

    gen = _minimal_generator(
        tmp_path, monkeypatch, public=public, private=private, repo=repo
    )
    out = gen.generate_health_json()

    assert out == public / "health.json"
    assert (public / "health.json").is_file()
    assert (repo / "health.json").is_file()

    pub_body = (public / "health.json").read_bytes()
    repo_body = (repo / "health.json").read_bytes()
    assert pub_body == repo_body
    assert (public / "health.json").stat().st_mode & 0o777 == 0o644
    assert (repo / "health.json").stat().st_mode & 0o777 == 0o644

    payload = json.loads(pub_body)
    assert payload.get("system_status") == "healthy"
    assert payload.get("generator_git_sha") == "batch-ie-test"
    # Dashboard schema markers (not monitor scope)
    assert "cron_jobs" in payload or "data_freshness" in payload

    # Private monitor schema untouched
    assert (private / "health.json").read_bytes() == private_before
    private_after = json.loads((private / "health.json").read_text())
    assert private_after.get("scope") == "operational_readiness"
    assert private_after.get("checks", {}).get("probe") is True


def test_generate_health_json_never_passes_private_monitor_path(
    tmp_path, monkeypatch
):
    """Case DW: multi-dest call must use private_path=None for dashboard health."""
    from src.dashboard import generator as gen_mod
    from src.monitor import signal_authority as sa

    public = tmp_path / "www"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    (private / "health.json").write_text("{}", encoding="utf-8")

    calls: list[dict] = []
    real_write = sa.write_json_multi_dest

    def _capture(payload, **kwargs):
        calls.append(dict(kwargs))
        # Provide repo so write succeeds under pytest auto-skip
        kwargs = {
            **kwargs,
            "repo_path": tmp_path / "repo" / "health.json",
        }
        (tmp_path / "repo").mkdir(exist_ok=True)
        return real_write(payload, **kwargs)

    monkeypatch.setattr(
        "src.monitor.signal_authority.write_json_multi_dest", _capture
    )

    gen = _minimal_generator(
        tmp_path,
        monkeypatch,
        public=public,
        private=private,
        repo=tmp_path / "repo",
    )
    gen.generate_health_json()

    assert calls, "expected write_json_multi_dest to be invoked"
    for c in calls:
        assert c.get("private_path") in (None, ""), c
        pub = c.get("public_path")
        assert pub is not None
        assert Path(pub).name == "health.json"
        # Must not target private monitor path as public either
        assert Path(pub).resolve() != (private / "health.json").resolve()


def test_generate_health_json_fallback_save_results_on_multi_dest_failure(
    tmp_path, monkeypatch
):
    """Case DX: if multi-dest raises, fall back to save_results_json @ public."""
    from src.dashboard import generator as gen_mod

    public = tmp_path / "www"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    (private / "health.json").write_text(
        json.dumps({"status": "ok", "scope": "operational_readiness"}),
        encoding="utf-8",
    )
    private_before = (private / "health.json").read_bytes()

    def _boom(*a, **k):
        raise RuntimeError("forced multi-dest failure")

    monkeypatch.setattr(
        "src.monitor.signal_authority.write_json_multi_dest", _boom
    )

    gen = _minimal_generator(
        tmp_path,
        monkeypatch,
        public=public,
        private=private,
        repo=tmp_path / "repo",
    )
    out = gen.generate_health_json()
    assert out == public / "health.json"
    assert (public / "health.json").is_file()
    body = json.loads((public / "health.json").read_text())
    assert body.get("generator_git_sha") == "batch-ie-test"
    # Fallback must still chmod public (save_results_json Batch HZ)
    assert (public / "health.json").stat().st_mode & 0o777 == 0o644
    assert (private / "health.json").read_bytes() == private_before
