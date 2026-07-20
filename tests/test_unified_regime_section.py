"""Unified dashboard regime section must read regime_state.json SSOT."""
import json
from pathlib import Path

import pytest


def test_get_regime_section_available_from_regime_state(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "regime_state.json").write_text(json.dumps({
        "regime": "NORMAL",
        "confidence": 0.55,
        "source": "classify_vix_regime",
        "updated_at": "2026-07-20T21:16:01",
    }))
    section = ud._get_regime_section()
    assert section["available"] is True
    assert section["regime"] == "NORMAL"
    assert abs(float(section["confidence"]) - 0.55) < 1e-9


def test_get_regime_section_missing_file(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    section = ud._get_regime_section()
    assert section["available"] is False
