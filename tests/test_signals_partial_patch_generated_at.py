"""Partial signals.json writers must stamp top-level generated_at."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_health_kill_refresh_stamps_generated_at(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    signals = {
        "generated_at": "2026-07-20T07:15:00+00:00",
        "health": {"status": "ok"},
        "collar": {"call_strike": 1},
    }
    sp = tmp_path / "signals.json"
    sp.write_text(json.dumps(signals))
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path)

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
            hc.refresh_signals_health_kill_fields(report, public_dir=tmp_path)

    out = json.loads(sp.read_text())
    assert out["generated_at"] != "2026-07-20T07:15:00+00:00"
    assert "content_patched_at" in out
    assert out.get("content_patch_source") == "health_kill_refresh"


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
    (public / "signals.json").write_text(json.dumps({
        "generated_at": "2026-07-20T07:15:00+00:00",
        "alternative_data": {},
    }))

    # project may need more fields — mock project_alternative_data_signal
    with patch.object(gen, "project_alternative_data_signal", return_value={
        "timestamp": "2026-07-20T12:00:00",
        "score": 0.1,
    }):
        with patch.object(gen, "save_results_json", side_effect=lambda data, output_path: Path(output_path).write_text(json.dumps(data))):
            ok = refresh_public_alternative_data_projection(data_dir=data_dir, public_dir=public)
    assert ok is True
    out = json.loads((public / "signals.json").read_text())
    assert out["generated_at"] != "2026-07-20T07:15:00+00:00"
    assert out.get("content_patch_source") == "bounded_alt_data_refresh"
