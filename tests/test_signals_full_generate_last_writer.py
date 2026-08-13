"""Last-writer contract: full generate leaves full_generate when it is last."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch



def test_full_then_partial_preserves_last_full_and_clears_live_sha():
    """Sequence A: full stamp then partial honesty → partial_patch, last_full kept."""
    from src.dashboard.generator import (
        _apply_partial_patch_git_sha_honesty,
        _stamp_generator_git_sha,
    )

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="fulltipsha001",
    ):
        payload = _stamp_generator_git_sha(
            {"target_allocations": {"SPY": 0.46}},
            status="full_generate",
        )
    assert payload["generator_git_sha"] == "fulltipsha001"
    assert payload["generator_git_sha_status"] == "full_generate"
    assert payload["last_full_generator_git_sha"] == "fulltipsha001"

    _apply_partial_patch_git_sha_honesty(payload, patch_source="health_kill_refresh")
    assert payload["generator_git_sha"] is None
    assert payload["generator_git_sha_status"] == "partial_patch"
    assert payload["last_full_generator_git_sha"] == "fulltipsha001"


def test_full_finalize_as_last_writer_leaves_full_generate():
    """Sequence B: _finalize_signal_metadata (full path last writer) → full_generate."""
    from src.dashboard.generator import _finalize_signal_metadata

    prior = {
        "generator_git_sha": None,
        "generator_git_sha_status": "partial_patch",
        "last_full_generator_git_sha": "oldfullsha000",
        "content_patch_source": "health_kill_refresh",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    }
    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="newtifulsha01",
    ):
        out = _finalize_signal_metadata(prior, finalized_at="2026-07-22T18:30:00+00:00")

    assert out["generator_git_sha"] == "newtifulsha01"
    assert out["generator_git_sha_status"] == "full_generate"
    assert out["last_full_generator_git_sha"] == "oldfullsha000"
    assert out["generated_at"] == "2026-07-22T18:30:00+00:00"


def test_generate_signals_json_last_writer_stamps_full_generate(tmp_path, monkeypatch):
    """Real generate_signals_json write path leaves full_generate on disk."""
    from src.dashboard import generator as gen_mod
    from src.dashboard.generator import DashboardGenerator

    public = tmp_path / "public"
    public.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)
    monkeypatch.setattr(gen_mod, "DATA_DIR", data)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    base = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"status": "ok"},
    }

    with patch.object(
        DashboardGenerator,
        "_load_signal_generation_context",
        return_value={},
    ), patch.object(
        DashboardGenerator,
        "_build_base_signal_sections",
        return_value=base,
    ), patch.object(
        DashboardGenerator,
        "_build_optional_signal_sections",
        side_effect=lambda output, context: output,
    ), patch.object(
        DashboardGenerator,
        "_apply_signal_postprocessors",
        side_effect=lambda output, context: output,
    ), patch(
        "src.dashboard.generator._attach_signal_metadata",
        side_effect=lambda o: o,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="signalsfull1",
    ), patch(
        "src.dashboard.generator.save_results_json",
        side_effect=lambda data, output_path, validator=None: Path(output_path).write_text(
            json.dumps(data)
        ),
    ), patch(
        "src.monitor.decision_registry.record_dashboard_cycle_decision",
        side_effect=ImportError("skip"),
    ):
        out_path = DashboardGenerator.generate_signals_json(gen)

    payload = json.loads(Path(out_path).read_text())
    assert payload["generator_git_sha"] == "signalsfull1"
    assert payload["generator_git_sha_status"] == "full_generate"
    assert payload["last_full_generator_git_sha"] == "signalsfull1"


def test_ops_regen_makefile_dashboard_is_last_signals_writer():
    """Contract A: ops-regen runs dashboard after health so full stamp is last writer."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    # Extract ops-regen recipe body (until next .PHONY or bare target at col 0)
    start = makefile.find(".PHONY: ops-regen")
    assert start >= 0
    body = makefile[start : start + 1200]
    # Next top-level target after ops-regen ends the block roughly
    _ = body.find("health")
    _ = body.find("dashboard")
    # Both must appear; last dashboard invoke must be after last health invoke
    # among make --no-print-directory lines
    make_lines = [
        ln
        for ln in body.splitlines()
        if "MAKE" in ln and "--no-print-directory" in ln
    ]
    health_idxs = [i for i, ln in enumerate(make_lines) if "health" in ln and "garch" not in ln]
    dash_idxs = [i for i, ln in enumerate(make_lines) if "dashboard" in ln]
    assert health_idxs, "ops-regen must invoke health"
    assert dash_idxs, "ops-regen must invoke dashboard"
    assert max(dash_idxs) > max(health_idxs), (
        "dashboard must run after health so full generate is last writer of signals.json"
    )
    assert "last writer" in body.lower() or "full_generate" in body or "LAST" in body
