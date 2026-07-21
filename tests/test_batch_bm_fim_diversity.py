"""Batch BM residual honesty: FIM external-append classify + diversity vs health sleep."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_classify_performance_growth_external_capture_daily_pnl(tmp_path: Path):
    from tests.conftest import _classify_performance_jsonl_growth

    path = tmp_path / "performance.jsonl"
    prior = (
        json.dumps(
            {
                "timestamp": "2026-07-20T00:00:00",
                "total_value": 100.0,
                "mode": "paper",
                "source": "capture_daily_pnl",
            }
        )
        + "\n"
    )
    path.write_text(prior, encoding="utf-8")
    before = path.stat().st_size
    append = (
        json.dumps(
            {
                "timestamp": "2026-07-21T13:40:00",
                "date": "2026-07-21",
                "total_value": 99.0,
                "mode": "paper",
                "source": "capture_daily_pnl",
            }
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(append)
    after = path.stat().st_size
    assert after > before
    assert (
        _classify_performance_jsonl_growth(path, before, after) == "external_append"
    )


def test_classify_performance_growth_test_leak_markers(tmp_path: Path):
    from tests.conftest import _classify_performance_jsonl_growth

    path = tmp_path / "performance.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    before = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"note": "wrote from pytest tmp_path isolation fail"}\n')
    after = path.stat().st_size
    assert _classify_performance_jsonl_growth(path, before, after) == "test_leak"


def test_classify_truncate_is_test_leak(tmp_path: Path):
    from tests.conftest import _classify_performance_jsonl_growth

    path = tmp_path / "performance.jsonl"
    path.write_text("x" * 100, encoding="utf-8")
    assert _classify_performance_jsonl_growth(path, 100, 10) == "test_leak"


def test_diversity_floor_does_not_reinflate_health_slept_arms():
    from src.strategy.ensemble_voter import EnsembleVoter, DEFAULT_DIVERSITY_FLOOR
    from src.signals.signal_source import SignalSource

    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter._health_gate_slept = [
        SignalSource.VIX_TERM_STRUCTURE.value,
        SignalSource.UNIFIED_OVERLAY.value,
    ]
    # Unhealthy arms already zeroed; healthy ones uneven
    weights = {
        SignalSource.MULTI_SPEED_MOM: 0.9,
        SignalSource.ALTERNATIVE_DATA: 0.1,
        SignalSource.VIX_TERM_STRUCTURE: 0.0,
        SignalSource.UNIFIED_OVERLAY: 0.0,
    }
    # Use a high floor so healthy legs get raised
    out = voter._apply_diversity_floor(weights, floor=0.2)
    assert out[SignalSource.VIX_TERM_STRUCTURE] == 0.0
    assert out[SignalSource.UNIFIED_OVERLAY] == 0.0
    assert out[SignalSource.MULTI_SPEED_MOM] > 0
    assert out[SignalSource.ALTERNATIVE_DATA] > 0
    # Slept mass stays out of the renorm pool
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out[SignalSource.ALTERNATIVE_DATA] > weights[SignalSource.ALTERNATIVE_DATA] * 0.9


def test_diversity_floor_forces_slept_zero_if_positive_slip():
    """Defense-in-depth: slept name stays 0 even if weight slipped positive."""
    from src.strategy.ensemble_voter import EnsembleVoter
    from src.signals.signal_source import SignalSource

    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter._health_gate_slept = [SignalSource.VIX_TERM_STRUCTURE.value]
    weights = {
        SignalSource.MULTI_SPEED_MOM: 0.5,
        SignalSource.VIX_TERM_STRUCTURE: 0.01,  # should not get diversity raise
        SignalSource.ALTERNATIVE_DATA: 0.49,
    }
    out = voter._apply_diversity_floor(weights, floor=0.15)
    assert out[SignalSource.VIX_TERM_STRUCTURE] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9
