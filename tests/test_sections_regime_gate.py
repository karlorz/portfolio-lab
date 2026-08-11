#!/usr/bin/env python3
"""
Tests for the C10 regime-gate-state mixin extracted by Item 26 (2026-08-12):
``src/dashboard/sections_regime_gate.py`` ``_RegimeGateStateMixin``.

A1: getattr smoke — all 3 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_RegimeGateStateMixin``.
A2: behavior-equality — canned fixtures for ``_resolve_current_regime_for_gate``
    (ensemble_voting path + default path) and ``_persist_regime_state``
    (history trim to last 50, FakeDateTime-deferred timestamp, regime_log
    append). Lazy-import contract: patches on ``src.dashboard.generator.*``
    are respected (never write to the real DATA_DIR).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_regime_gate import _RegimeGateStateMixin

REGIME_GATE_NAMES = (
    "_normalize_gate_regime_name",
    "_resolve_current_regime_for_gate",
    "_persist_regime_state",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 3 C10 names resolve via DashboardGenerator MRO and the mixin."""
    for name in REGIME_GATE_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_RegimeGateStateMixin, name), name


def test_a2_normalize_gate_regime_name_canned_inputs():
    """Alias mapping identical via both surfaces."""
    for surface in (_RegimeGateStateMixin, DashboardGenerator):
        assert surface._normalize_gate_regime_name(None) == "NORMAL"
        assert surface._normalize_gate_regime_name("") == "NORMAL"
        assert surface._normalize_gate_regime_name("normal") == "NORMAL"
        assert surface._normalize_gate_regime_name("vol_spike") == "HIGH_VOL"
        assert surface._normalize_gate_regime_name("vol-spike") == "HIGH_VOL"
        assert surface._normalize_gate_regime_name("lowvol") == "LOW_VOL"
        assert surface._normalize_gate_regime_name("crisis") == "CRISIS"
        assert surface._normalize_gate_regime_name("recovery") == "RECOVERY"
        assert surface._normalize_gate_regime_name("unknown") == "UNKNOWN"


def test_a2_resolve_current_regime_ensemble_voting_path(tmp_path):
    """ensemble_voting on published signals wins when no DB connection."""
    signals = {
        "ensemble_voting": {"regime": "normal", "regime_confidence": 0.755}
    }
    (tmp_path / "signals.json").write_text(json.dumps(signals))
    gen = DashboardGenerator.__new__(DashboardGenerator)  # no conn attr

    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            result = gen._resolve_current_regime_for_gate()

    assert result == ("NORMAL", 0.755, "ensemble_voting")


def test_a2_resolve_current_regime_default_path(tmp_path):
    """No signals / classifier state → explicit default with disclosed source."""
    gen = DashboardGenerator.__new__(DashboardGenerator)

    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            result = gen._resolve_current_regime_for_gate()

    assert result == ("NORMAL", 0.5, "default_missing_state")


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def test_a2_persist_regime_state_history_trim_and_timestamp(tmp_path):
    """History trims to last 50; updated_at deferred; regime_log appended."""
    old_history = [
        {"timestamp": f"2026-01-{i:02d}T00:00:00", "regime": "old", "confidence": 0.5}
        for i in range(1, 61)  # 60 entries
    ]
    (tmp_path / "regime_state.json").write_text(
        json.dumps({"regime": "old", "history": old_history})
    )

    gen = DashboardGenerator.__new__(DashboardGenerator)
    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        with patch("src.dashboard.generator.datetime", FakeDateTime):
            result = gen._persist_regime_state("HIGH_VOL", 0.9, "test")

    assert result == tmp_path / "regime_state.json"
    payload = json.loads((tmp_path / "regime_state.json").read_text())
    expected_ts = FakeDateTime.now().isoformat()
    assert payload["regime"] == "HIGH_VOL"
    assert payload["previous_regime"] == "old"
    assert payload["updated_at"] == expected_ts
    assert len(payload["history"]) == 50  # 49 kept + 1 appended
    assert payload["history"][-1] == {
        "timestamp": expected_ts,
        "regime": "HIGH_VOL",
        "confidence": 0.9,
        "source": "test",
    }
    # regime_log.json appended for graduation coverage
    log_line = json.loads((tmp_path / "regime_log.json").read_text().strip().split("\n")[-1])
    assert log_line == {
        "regime": "HIGH_VOL",
        "confidence": 0.9,
        "source": "test",
        "detected_at": expected_ts,
    }
