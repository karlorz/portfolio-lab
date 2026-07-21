"""Batch BP residual honesty: graduation CB SSOT + dual-surface write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_refresh_graduation_dual_surfaces_matches_cb_ssot(tmp_path, monkeypatch):
    from src.dashboard import generator as gen_mod
    import src.strategy.graduation_checklist as gc
    from src.dashboard.generator import refresh_graduation_dual_surfaces

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    monkeypatch.setattr(gc, "DATA_DIR", data)
    monkeypatch.setattr(gen_mod, "DATA_DIR", data)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)

    (data / ".circuit_breaker.json").write_text(
        json.dumps(
            {
                "schema_version": "graduation-circuit-breaker/v1",
                "status": "green",
                "consecutive_ok": 7,
                "trips": 0,
            }
        ),
        encoding="utf-8",
    )

    out = refresh_graduation_dual_surfaces(public_dir=public, data_dir=data)
    assert out is not None
    assert out.is_file()

    public_payload = json.loads(out.read_text(encoding="utf-8"))
    private_path = data / ".graduation_report.json"
    assert private_path.is_file()
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))

    assert public_payload.get("circuit_breaker_consecutive_ok") == 7
    assert private_payload.get("circuit_breaker_consecutive_ok") == 7
    assert public_payload.get("circuit_breaker_ssot") == ".circuit_breaker.json"
    assert private_payload.get("circuit_breaker_ssot") == ".circuit_breaker.json"
    # readiness_score must match between dual surfaces
    assert public_payload.get("readiness_score") == private_payload.get(
        "readiness_score"
    )

    # CB criterion in public criteria list
    cb_crit = next(
        c
        for c in public_payload["criteria"]
        if c.get("name") == "circuit_breaker_confidence"
    )
    assert cb_crit["value"] == 7
    assert cb_crit["passed"] is True


def test_graduation_checklist_ignores_legacy_state_file_invented_streak(tmp_path, monkeypatch):
    import src.strategy.graduation_checklist as gc
    from src.strategy.graduation_checklist import GraduationChecklist

    monkeypatch.setattr(gc, "DATA_DIR", tmp_path)
    (tmp_path / ".circuit_breaker_state.json").write_text(
        json.dumps({"status": "green", "max_drawdown": 0.0}),
        encoding="utf-8",
    )
    cl = GraduationChecklist()
    state = cl._load_state()
    assert int(state["circuit_breaker"].get("consecutive_ok") or 0) == 0
    assert cl._check_circuit_breaker(state).passed is False
