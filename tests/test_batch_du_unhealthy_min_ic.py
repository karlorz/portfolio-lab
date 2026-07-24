"""Batch DU: unhealthy arms need IC >= min to soft-floor; else hard sleep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.ensemble_voter import EnsembleVoter, Regime, SignalSource


class _H:
    def __init__(self, status: str, score: float, ic, predictions_count: int = 20):
        self.status = status
        self.health_score = score
        self.ic = ic
        self.predictions_count = predictions_count


def test_unhealthy_weak_ic_hard_sleeps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", "0.08")
    scores = {
        "cross_asset_rv": _H("degraded", 0.54, 0.15),
        "multi_timeframe_fusion": _H("unhealthy", 0.39, 0.05),  # weak
        "vix_term_structure": _H("unhealthy", 0.46, 0.059),  # weak
        "google_trends": _H("degraded", 0.49, 0.29),
        "alternative_data": _H("degraded", 0.51, -0.07),
        "cross_asset_regime_arb": _H("degraded", 0.51, -0.30),
        "international_momentum": _H("degraded", 0.52, 0.17),
        "multi_speed_momentum": _H("healthy", 0.55, 0.03),
        "unified_overlay": _H("healthy", 0.55, 0.02),
    }

    class _FakeTracker:
        def calculate_all_health_scores(self):
            return scores

    monkeypatch.setattr(
        "src.signals.health_tracker.SignalHealthTracker",
        _FakeTracker,
    )
    voter = EnsembleVoter(data_path=tmp_path)
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.4,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.2,
        SignalSource.VIX_TERM_STRUCTURE: 0.2,
        SignalSource.GOOGLE_TRENDS: 0.2,
        SignalSource.ALTERNATIVE_DATA: 0.0,
        SignalSource.MULTI_SPEED_MOM: 0.0,
    }
    out = voter._apply_health_weights(weights)
    # Weak unhealthy → 0
    assert float(out[SignalSource.MULTI_TIMEFRAME_FUSION]) == 0.0
    assert float(out[SignalSource.VIX_TERM_STRUCTURE]) == 0.0
    assert "multi_timeframe_fusion" in voter._health_gate_slept
    assert "vix_term_structure" in voter._health_gate_slept
    reasons = voter._health_gate_sleep_reasons
    assert "unhealthy_weak_ic" in reasons.get("multi_timeframe_fusion", "")
    # Strong IC degraded/unhealthy can remain
    assert float(out[SignalSource.CROSS_ASSET_RV]) > 0
    assert float(out[SignalSource.GOOGLE_TRENDS]) > 0


def test_unhealthy_strong_ic_soft_floor_disclosed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENSEMBLE_UNHEALTHY_MIN_IC", "0.08")
    scores = {
        "cross_asset_rv": _H("unhealthy", 0.40, 0.15),  # strong IC → soft floor
        "google_trends": _H("healthy", 0.70, 0.20),
    }

    class _FakeTracker:
        def calculate_all_health_scores(self):
            return scores

    monkeypatch.setattr(
        "src.signals.health_tracker.SignalHealthTracker",
        _FakeTracker,
    )
    voter = EnsembleVoter(data_path=tmp_path)
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.6,
        SignalSource.GOOGLE_TRENDS: 0.4,
    }
    out = voter._apply_health_weights(weights)
    assert float(out[SignalSource.CROSS_ASSET_RV]) > 0
    assert "cross_asset_rv" not in (voter._health_gate_slept or [])
    assert "cross_asset_rv" in voter._health_gate_soft_floor
    assert "unhealthy_soft_floor" in voter._health_gate_soft_floor["cross_asset_rv"]
