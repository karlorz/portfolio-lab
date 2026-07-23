"""Partial signals.json writers must stamp top-level generated_at."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_health_kill_refresh_stamps_generated_at(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    signals = {
        "generated_at": "2026-07-20T07:15:00+00:00",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"status": "ok"},
        "collar": {"call_strike": 1},
        "generator_git_sha": "fulltipsha001",
        "generator_git_sha_status": "full_generate",
    }
    public_dir = tmp_path / "www"
    data_dir = tmp_path / "data"
    public_dir.mkdir()
    data_dir.mkdir()
    sp = public_dir / "signals.json"
    sp.write_text(json.dumps(signals))
    (data_dir / "signals.json").write_text(json.dumps(signals))
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public_dir)
    monkeypatch.setattr(hc, "DATA_DIR", data_dir)

    report = {
        "status": "ok",
        "timestamp": "2026-07-20T16:00:00+00:00",
        "checks": {
            "kill_switch": {"enabled": False, "level": None},
            "open_incidents": {"open_count": 0, "status": "ok"},
        },
    }
    # project_compact_kill_fields may need real structure — call with monkeypatch
    with patch("src.dashboard.kill_authority.project_compact_kill_fields", return_value={"kill_switch_enabled": False}):
        with patch("src.dashboard.generator._compact_health_summary", return_value={"status": "ok"}):
            hc.refresh_signals_health_kill_fields(
                report, public_dir=public_dir, data_dir=data_dir
            )

    out = json.loads(sp.read_text())
    assert out["generated_at"] != "2026-07-20T07:15:00+00:00"
    assert "content_patched_at" in out
    assert out.get("content_patch_source") == "health_kill_refresh"
    assert out["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    # Partial must not keep a full-run generator_git_sha as live tip
    assert out.get("generator_git_sha") is None
    assert out.get("generator_git_sha_status") == "partial_patch"
    assert (sp.stat().st_mode & 0o777) == 0o644


def test_alt_data_projection_stamps_generated_at(tmp_path, monkeypatch):
    from src.dashboard.generator import refresh_public_alternative_data_projection
    import src.dashboard.generator as gen

    data_dir = tmp_path / "data"
    public = tmp_path / "public"
    (data_dir / "signals").mkdir(parents=True)
    public.mkdir()
    (data_dir / "signals" / "alternative_data_latest.json").write_text(json.dumps({
        "timestamp": "2026-07-20T12:00:00",
        "score": 0.1,
    }))
    full = {
        "generated_at": "2026-07-20T07:15:00+00:00",
        "generator_git_sha": "4440aad5c996",
        "generator_git_sha_status": "full_generate",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "alternative_data": {},
    }
    (public / "signals.json").write_text(json.dumps(full))
    (data_dir / "signals.json").write_text(json.dumps(full))

    # project may need more fields — mock project_alternative_data_signal.
    # Real multi-dest write (authority gate + 0o644) replaces legacy save_results_json mock.
    with patch.object(gen, "project_alternative_data_signal", return_value={
        "timestamp": "2026-07-20T12:00:00",
        "score": 0.1,
    }):
        ok = refresh_public_alternative_data_projection(data_dir=data_dir, public_dir=public)
    assert ok is True
    out = json.loads((public / "signals.json").read_text())
    assert out["generated_at"] != "2026-07-20T07:15:00+00:00"
    assert out.get("content_patch_source") == "bounded_alt_data_refresh"
    assert out["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    # Sticky full-run sha must not survive partial alt-data refresh
    assert out.get("generator_git_sha") is None
    assert out.get("generator_git_sha_status") == "partial_patch"
    assert out.get("last_full_generator_git_sha") == "4440aad5c996"
    assert ((public / "signals.json").stat().st_mode & 0o777) == 0o644


def test_apply_partial_patch_git_sha_honesty_preserves_prior():
    from src.dashboard.generator import _apply_partial_patch_git_sha_honesty

    payload = {"generator_git_sha": "deadbeef"}
    _apply_partial_patch_git_sha_honesty(payload, patch_source="unit_test")
    assert payload["generator_git_sha"] is None
    assert payload["last_full_generator_git_sha"] == "deadbeef"
    assert payload["generator_git_sha_status"] == "partial_patch"
