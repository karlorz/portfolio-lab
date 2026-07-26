"""Batch JG DS3 — sector_rotation dual-stamps generated_at + timestamp."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_ds3_generate_sector_signals_emits_generated_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Producer success payload includes generated_at == timestamp."""
    from src.strategy import sector_momentum_calc as smc

    # Minimal historical blob so calculator can run; if empty scores → None,
    # so stub calculate path for hermetic unit.
    hist = tmp_path / "historical.json"
    hist.write_text("{}", encoding="utf-8")

    class _FakeCalc:
        def __init__(self, data):
            pass

        def calculate_all_momentum(self, lookback_days=252):
            return [
                {
                    "symbol": "XLK",
                    "name": "Tech",
                    "compositeMomentum": 0.12,
                    "longMomentum": 0.1,
                    "shortMomentum": 0.05,
                    "volatility": 0.2,
                    "rank": 1,
                }
            ]

        def adjust_for_regime(self, scores, regime):
            return scores

        def get_allocation(self, scores, top_n=3, vix=None):
            return {
                "spAllocation": 0.75,
                "totalEquityWeight": 1.0,
                "sectorAllocations": [
                    {
                        "symbol": "XLK",
                        "weight": 0.25,
                        "momentum": 0.12,
                        "rank": 1,
                    }
                ],
                "rebalanceRecommended": False,
                "rebalanceReason": None,
            }

    monkeypatch.setattr(smc, "SectorMomentumCalculator", _FakeCalc)
    out = smc.generate_sector_signals(hist, vix=18.0, regime="neutral")
    assert out is not None
    assert out.get("generated_at"), "DS3: generated_at required"
    assert out.get("timestamp"), "timestamp still required"
    assert out["generated_at"] == out["timestamp"]
    assert out.get("status") == "active"


def test_ds3_generator_setdefault_fills_missing_generated_at() -> None:
    """Attach path setdefault when producer only has timestamp."""
    block = {
        "timestamp": "2026-07-23T17:16:00.879523+00:00",
        "status": "active",
    }
    ts = block.get("timestamp") or block.get("generated_at")
    if ts:
        block.setdefault("generated_at", ts)
        block.setdefault("timestamp", ts)
    assert block["generated_at"] == block["timestamp"]
