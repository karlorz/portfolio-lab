"""Batch IC: health_ops + public health merge multi-dest + 0o644.

Session A residual after IB:
- ``publish_ops_health_surfaces`` used bare ``write_text`` for PUBLIC
  ``health_ops.json`` and dashboard ``health.json`` merge — no fchmod, no
  repo soft-mirror, no private ops twin. Live probe showed private
  ``data/health_ops.json`` missing while www/public EQ only after satellite
  mirror soft-gates.

Private monitor ``DATA_DIR/health.json`` remains a different schema (SSOT) and
must **not** be overwritten by the dashboard health merge payload.

Authority: never touches ``signals.json.target_allocations`` / order_router
except via existing kill-refresh multi-dest (already gated).
"""

from __future__ import annotations

import json
from pathlib import Path


def test_health_ops_multi_dest_same_bytes_0644(tmp_path, monkeypatch) -> None:
    """Case DR: health_ops lands public + private twin @ 0o644 (repo via explicit path).

    Under pytest, write_json_multi_dest disables *auto* repo soft-mirror; pass
    repo_path explicitly to prove the fan-out contract (production uses auto).
    """
    from src.monitor import health_check as hc
    from src.monitor import signal_authority as sa

    public = tmp_path / "www"
    private = tmp_path / "data"
    repo = tmp_path / "repo_public"
    public.mkdir()
    private.mkdir()
    repo.mkdir()

    monkeypatch.setattr(hc, "DATA_DIR", private)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "HEALTH_PATH", private / "health.json")
    monkeypatch.setattr(hc, "health_ops_path", lambda: public / "health_ops.json")
    monkeypatch.setattr(hc, "refresh_signals_health_kill_fields", lambda *a, **k: None)

    real_write = sa.write_json_multi_dest

    def _write_with_repo(payload, **kwargs):
        if kwargs.get("repo_filename") == "health_ops.json" or (
            kwargs.get("public_path") and Path(kwargs["public_path"]).name == "health_ops.json"
        ):
            kwargs = {**kwargs, "repo_path": repo / "health_ops.json"}
        return real_write(payload, **kwargs)

    monkeypatch.setattr(sa, "write_json_multi_dest", _write_with_repo)
    monkeypatch.setattr(hc, "write_json_multi_dest", _write_with_repo, raising=False)
    # health_check imports write_json_multi_dest inside the function
    monkeypatch.setattr(
        "src.monitor.signal_authority.write_json_multi_dest", _write_with_repo
    )

    report = {
        "status": "ok",
        "timestamp": "2026-07-23T00:00:00+00:00",
        "checks": {"kill_switch": {"enabled": False, "status": "ok"}},
        "service": "portfolio-lab",
        "scope": "operational_readiness",
    }
    hc.publish_ops_health_surfaces(report)

    ops_pub = public / "health_ops.json"
    ops_priv = private / "health_ops.json"
    ops_repo = repo / "health_ops.json"
    assert ops_pub.is_file()
    assert ops_priv.is_file()
    assert ops_repo.is_file()
    body = ops_pub.read_bytes()
    assert ops_priv.read_bytes() == body
    assert ops_repo.read_bytes() == body
    assert (ops_pub.stat().st_mode & 0o777) == 0o644
    assert (ops_priv.stat().st_mode & 0o777) == 0o644
    assert (ops_repo.stat().st_mode & 0o777) == 0o644
    assert json.loads(body)["scope"] == "operational_readiness"


def test_public_health_merge_soft_mirrors_not_private_monitor(
    tmp_path, monkeypatch
) -> None:
    """Case DS: dashboard health merge fans out public+repo; private monitor intact."""
    from src.monitor import health_check as hc
    from src.monitor import signal_authority as sa

    public = tmp_path / "www"
    private = tmp_path / "data"
    repo = tmp_path / "repo_public"
    public.mkdir()
    private.mkdir()
    repo.mkdir()

    monitor_ssot = {
        "status": "ok",
        "scope": "operational_readiness",
        "checks": {"kill_switch": {"enabled": False}},
        "schema": "monitor",
    }
    (private / "health.json").write_text(
        json.dumps(monitor_ssot), encoding="utf-8"
    )
    before_monitor = (private / "health.json").read_bytes()

    dash = {
        "system_status": "healthy",
        "generated_at": "2026-07-01T00:00:00+00:00",
        "kill_switch": {"enabled": False, "status": "ok"},
        "cron_jobs": [],
        "schema": "dashboard",
    }
    (public / "health.json").write_text(json.dumps(dash), encoding="utf-8")

    monkeypatch.setattr(hc, "DATA_DIR", private)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "HEALTH_PATH", private / "health.json")
    monkeypatch.setattr(hc, "health_ops_path", lambda: public / "health_ops.json")
    monkeypatch.setattr(hc, "refresh_signals_health_kill_fields", lambda *a, **k: None)

    real_write = sa.write_json_multi_dest

    def _write_with_repo(payload, **kwargs):
        name = None
        if kwargs.get("repo_filename"):
            name = kwargs["repo_filename"]
        elif kwargs.get("public_path"):
            name = Path(kwargs["public_path"]).name
        if name in ("health.json", "health_ops.json"):
            kwargs = {**kwargs, "repo_path": repo / name}
        return real_write(payload, **kwargs)

    monkeypatch.setattr(
        "src.monitor.signal_authority.write_json_multi_dest", _write_with_repo
    )

    report = {
        "status": "ok",
        "timestamp": "2026-07-23T00:00:00+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "status": "ok"},
            "open_incidents": {"count": 0, "status": "ok"},
        },
        "service": "portfolio-lab",
        "scope": "operational_readiness",
    }
    hc.publish_ops_health_surfaces(report)

    # Private monitor SSOT must not become dashboard schema
    assert (private / "health.json").read_bytes() == before_monitor
    assert json.loads(before_monitor)["schema"] == "monitor"

    pub_h = public / "health.json"
    repo_h = repo / "health.json"
    assert pub_h.is_file()
    assert repo_h.is_file()
    assert pub_h.read_bytes() == repo_h.read_bytes()
    assert (pub_h.stat().st_mode & 0o777) == 0o644
    assert (repo_h.stat().st_mode & 0o777) == 0o644
    merged = json.loads(pub_h.read_text(encoding="utf-8"))
    assert merged.get("content_patch_source") == "ops_health_merge"


def test_ops_publish_does_not_touch_target_allocations(tmp_path, monkeypatch) -> None:
    """Case DT: health_ops multi-dest must not rewrite signals authority surface."""
    from src.monitor import health_check as hc

    public = tmp_path / "www"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    signals = public / "signals.json"
    signals.write_text(
        json.dumps({"target_allocations": champion, "health": {"status": "ok"}}),
        encoding="utf-8",
    )
    before = signals.read_bytes()

    monkeypatch.setattr(hc, "DATA_DIR", private)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(hc, "HEALTH_PATH", private / "health.json")
    monkeypatch.setattr(hc, "health_ops_path", lambda: public / "health_ops.json")
    # Kill refresh would rewrite signals — stub it for authority isolation
    monkeypatch.setattr(hc, "refresh_signals_health_kill_fields", lambda *a, **k: None)

    hc.publish_ops_health_surfaces(
        {
            "status": "ok",
            "checks": {},
            "scope": "operational_readiness",
        }
    )
    assert signals.read_bytes() == before
