"""Batch AW residual honesty: overlay_dashboard dual-write provenance + mid-suite GC."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_overlay_save_dual_write_provenance(tmp_path, monkeypatch):
    from src.dashboard.overlay_dashboard import (
        OverlayDashboardData,
        OverlayDashboardGenerator,
    )

    private = tmp_path / "private" / "overlay_dashboard.json"
    public = tmp_path / "public"
    public.mkdir()
    private.parent.mkdir()

    monkeypatch.setattr(OverlayDashboardGenerator, "OUTPUT_PATH", private)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "overlaysha123",
    )

    # Patch PUBLIC_DATA_DIR at the import site used inside save()
    import src.paths as paths_mod

    monkeypatch.setattr(paths_mod, "PUBLIC_DATA_DIR", public)

    gen = OverlayDashboardGenerator.__new__(OverlayDashboardGenerator)
    gen.OUTPUT_PATH = private

    dash = OverlayDashboardData(
        timestamp="2026-07-21T00:00:00+00:00",
        generated_at="2026-07-21T00:00:00+00:00",
        collar={"active": False},
        crypto={"active": False},
        bond_duration={"active": False},
        calendar={"active": False},
        kurtosis={"active": False},
        mean_reversion={"active": False},
        unified={"active": False},
        active_overlays=0,
        total_overlays=7,
        portfolio_risk="low",
        alerts=[],
    )
    gen.save(dash)

    priv = json.loads(private.read_text())
    pub = json.loads((public / "overlay_dashboard.json").read_text())
    for body in (priv, pub):
        assert body.get("generator_git_sha") == "overlaysha123"
        pc = body["provenance_completeness"]
        assert pc["dual_write_attempted"] is True
        assert pc["dual_write_ok"] is True
        assert pc["paths_identical"] is False


def test_overlay_save_paths_identical_skips_public_dual_write(tmp_path, monkeypatch):
    from src.dashboard.overlay_dashboard import (
        OverlayDashboardData,
        OverlayDashboardGenerator,
    )

    out = tmp_path / "overlay_dashboard.json"
    monkeypatch.setattr(OverlayDashboardGenerator, "OUTPUT_PATH", out)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "samepathsha12",
    )
    import src.paths as paths_mod

    # PUBLIC dir is parent of out so public_path == out when file name matches...
    # Actually public is PUBLIC_DATA_DIR / overlay_dashboard.json — set PUBLIC to
    # the same parent so paths resolve identical.
    monkeypatch.setattr(paths_mod, "PUBLIC_DATA_DIR", tmp_path)

    gen = OverlayDashboardGenerator.__new__(OverlayDashboardGenerator)
    gen.OUTPUT_PATH = out
    dash = OverlayDashboardData(
        timestamp="2026-07-21T00:00:00+00:00",
        generated_at="2026-07-21T00:00:00+00:00",
        collar={"active": False},
        crypto={"active": False},
        bond_duration={"active": False},
        calendar={"active": False},
        kurtosis={"active": False},
        mean_reversion={"active": False},
        unified={"active": False},
        active_overlays=0,
        total_overlays=7,
        portfolio_risk="low",
        alerts=[],
    )
    gen.save(dash)

    body = json.loads(out.read_text())
    assert body.get("generator_git_sha") == "samepathsha12"
    pc = body["provenance_completeness"]
    assert pc["paths_identical"] is True
    assert pc["dual_write_attempted"] is False


def test_dual_write_canary_includes_overlay_dashboard():
    from scripts.check_public_data_consistency import DUAL_WRITE_PROVENANCE_FILES

    assert "overlay_dashboard.json" in DUAL_WRITE_PROVENANCE_FILES


def test_batch_aw_source_contracts():
    overlay = Path("src/dashboard/overlay_dashboard.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in overlay
    assert "dual_write_ok" in overlay

    conftest = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "_mid_suite_gc_hygiene" in conftest
    assert "PORTFOLIO_LAB_MID_SUITE_GC_EVERY" in conftest
    assert "CHECK_LEAKS" in conftest
