"""Batch DN: enforce documented 50% per-signal weight cap after health renorm."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.strategy.ensemble_voter import (
    EnsembleVoter,
    Regime,
    SignalReading,
    SignalSource,
)


def test_apply_per_signal_weight_cap_clips_and_renorms() -> None:
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.85,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.05,
        SignalSource.GOOGLE_TRENDS: 0.05,
        SignalSource.VIX_TERM_STRUCTURE: 0.05,
    }
    out = EnsembleVoter._apply_per_signal_weight_cap(weights, max_weight=0.50)
    assert out[SignalSource.CROSS_ASSET_RV] <= 0.50 + 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # Excess redistributes to smaller arms
    assert out[SignalSource.MULTI_TIMEFRAME_FUSION] > 0.05


def test_cap_respects_soft_delete_zeros() -> None:
    weights = {
        SignalSource.MULTI_SPEED_MOM: 0.20,  # should stay 0 via soft_delete
        SignalSource.CROSS_ASSET_RV: 0.80,
        SignalSource.GOOGLE_TRENDS: 0.0,
    }
    soft = {SignalSource.MULTI_SPEED_MOM}
    out = EnsembleVoter._apply_per_signal_weight_cap(
        weights, max_weight=0.50, soft_delete=soft
    )
    assert out[SignalSource.MULTI_SPEED_MOM] == 0.0
    # Only one non-soft positive arm → residual 1.0 allowed
    assert abs(out[SignalSource.CROSS_ASSET_RV] - 1.0) < 1e-9


def test_cap_instance_records_disclosure(tmp_path: Path) -> None:
    voter = EnsembleVoter(data_path=tmp_path)
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.848,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.045,
        SignalSource.GOOGLE_TRENDS: 0.061,
        SignalSource.VIX_TERM_STRUCTURE: 0.046,
        SignalSource.MULTI_SPEED_MOM: 0.0,
    }
    out = voter._cap_per_signal_weights(weights, "NORMAL")
    assert out[SignalSource.CROSS_ASSET_RV] <= 0.50 + 1e-9
    assert out[SignalSource.MULTI_SPEED_MOM] == 0.0
    info = voter._last_per_signal_cap
    assert info["cap"] == 0.50
    assert info["applied"] is True
    assert "cross_asset_rv" in info["breached_before"]
    assert info["max_weight_after"] <= 0.50 + 1e-6


def _reading(
    source: SignalSource,
    value: float,
    confidence: float = 0.6,
    *,
    is_active: bool = True,
) -> SignalReading:
    return SignalReading(
        source=source,
        timestamp="2026-07-22",
        value=value,
        confidence=confidence,
        weight=0.0,
        regime_fit="all",
        asset_signals={"SPY": value},
        explanation=source.value,
        is_active=is_active,
    )


def test_compute_vote_pipeline_caps_after_health(tmp_path: Path, monkeypatch) -> None:
    """Health sleep of peers must not leave a single arm >50% when peers remain."""
    from src.signals.health_tracker import SignalHealthStatus

    class _H:
        def __init__(self, status: str, score: float, ic: float):
            self.status = status
            self.health_score = score
            self.ic = ic

    class _FakeTracker:
        def calculate_all_health_scores(self):
            return {
                "cross_asset_rv": _H(SignalHealthStatus.DEGRADED.value, 0.54, 0.15),
                "multi_timeframe_fusion": _H(
                    SignalHealthStatus.UNHEALTHY.value, 0.39, 0.12
                ),
                "google_trends": _H(SignalHealthStatus.DEGRADED.value, 0.49, 0.29),
                "vix_term_structure": _H(
                    SignalHealthStatus.UNHEALTHY.value, 0.46, 0.06
                ),
                "alternative_data": _H(
                    SignalHealthStatus.DEGRADED.value, 0.51, -0.07
                ),
                "cross_asset_regime_arb": _H(
                    SignalHealthStatus.DEGRADED.value, 0.51, -0.30
                ),
                "international_momentum": _H(
                    SignalHealthStatus.DEGRADED.value, 0.52, 0.17
                ),
                "multi_speed_momentum": _H(
                    SignalHealthStatus.HEALTHY.value, 0.55, 0.03
                ),
                "unified_overlay": _H(SignalHealthStatus.HEALTHY.value, 0.55, 0.02),
            }

    monkeypatch.setattr(
        "src.signals.health_tracker.SignalHealthTracker",
        _FakeTracker,
    )

    voter = EnsembleVoter(data_path=tmp_path)
    monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "0")
    readings = {
        SignalSource.CROSS_ASSET_RV: _reading(SignalSource.CROSS_ASSET_RV, 0.5, 0.7),
        SignalSource.MULTI_TIMEFRAME_FUSION: _reading(
            SignalSource.MULTI_TIMEFRAME_FUSION, -0.01, 0.5
        ),
        SignalSource.GOOGLE_TRENDS: _reading(SignalSource.GOOGLE_TRENDS, 0.1, 0.5),
        SignalSource.VIX_TERM_STRUCTURE: _reading(
            SignalSource.VIX_TERM_STRUCTURE, 0.05, 0.5
        ),
        SignalSource.ALTERNATIVE_DATA: _reading(
            SignalSource.ALTERNATIVE_DATA, 0.5, 0.6
        ),
        SignalSource.CROSS_ASSET_REGIME_ARB: _reading(
            SignalSource.CROSS_ASSET_REGIME_ARB, 0.05, 0.7
        ),
        SignalSource.INTERNATIONAL_MOMENTUM: _reading(
            SignalSource.INTERNATIONAL_MOMENTUM, 0.0, 0.0, is_active=False
        ),
        SignalSource.MULTI_SPEED_MOM: _reading(
            SignalSource.MULTI_SPEED_MOM, -0.3, 0.6
        ),
        SignalSource.UNIFIED_OVERLAY: _reading(
            SignalSource.UNIFIED_OVERLAY, -0.5, 0.7
        ),
    }
    vote = voter.compute_vote(readings, Regime.NORMAL, 0.8)
    pos = [r for r in vote.source_votes if float(getattr(r, "weight", 0) or 0) > 1e-9]
    for r in pos:
        assert float(r.weight) <= 0.50 + 1e-6, (
            f"{r.source.value} weight {r.weight} exceeds 50% cap"
        )
    msm_votes = [
        r for r in vote.source_votes if r.source == SignalSource.MULTI_SPEED_MOM
    ]
    for r in msm_votes:
        assert float(r.weight or 0.0) == 0.0
    cap = (vote.adaptive_learning or {}).get("per_signal_weight_cap") or {}
    assert cap.get("cap") == 0.50
