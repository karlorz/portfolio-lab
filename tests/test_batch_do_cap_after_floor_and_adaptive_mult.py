"""Batch DO: recap after analysis floor; zero adaptive multipliers for soft-delete."""

from __future__ import annotations

from pathlib import Path

from src.strategy.adaptive_ensemble_weights import AdaptiveEnsembleWeights
from src.strategy.ensemble_voter import (
    EnsembleVoter,
    Regime,
    SignalReading,
    SignalSource,
)


def _reading(source: SignalSource, value: float, conf: float = 0.6) -> SignalReading:
    return SignalReading(
        source=source,
        timestamp="2026-07-22",
        value=value,
        confidence=conf,
        weight=0.0,
        regime_fit="all",
        asset_signals={"SPY": value},
        explanation=source.value,
        is_active=True,
    )


def test_analysis_floor_then_cap_keeps_max_at_50(tmp_path: Path, monkeypatch) -> None:
    """Equal-weight / floor renorm after DN cap must not leave arm >50%."""
    monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "0")
    voter = EnsembleVoter(data_path=tmp_path)

    # Force a path where floor might re-inflate: many zero-weight readings
    # with one heavy arm after health — use direct post-cap floor simulation
    # via compute_vote with caller-supplied readings that trigger floor.
    readings = {
        SignalSource.CROSS_ASSET_RV: _reading(SignalSource.CROSS_ASSET_RV, 0.9, 0.8),
        SignalSource.MULTI_TIMEFRAME_FUSION: _reading(
            SignalSource.MULTI_TIMEFRAME_FUSION, 0.0, 0.5
        ),
        SignalSource.GOOGLE_TRENDS: _reading(SignalSource.GOOGLE_TRENDS, 0.0, 0.5),
        SignalSource.VIX_TERM_STRUCTURE: _reading(
            SignalSource.VIX_TERM_STRUCTURE, 0.0, 0.5
        ),
        SignalSource.MULTI_SPEED_MOM: _reading(SignalSource.MULTI_SPEED_MOM, -0.3),
        SignalSource.ALTERNATIVE_DATA: _reading(SignalSource.ALTERNATIVE_DATA, 0.2),
        SignalSource.CROSS_ASSET_REGIME_ARB: _reading(
            SignalSource.CROSS_ASSET_REGIME_ARB, 0.1
        ),
        SignalSource.INTERNATIONAL_MOMENTUM: _reading(
            SignalSource.INTERNATIONAL_MOMENTUM, 0.1
        ),
        SignalSource.UNIFIED_OVERLAY: _reading(SignalSource.UNIFIED_OVERLAY, -0.2),
    }
    vote = voter.compute_vote(readings, Regime.NORMAL, 0.8)
    for r in vote.source_votes:
        if float(r.weight or 0) > 1e-9:
            assert float(r.weight) <= 0.50 + 1e-6, (
                f"{r.source.value}={r.weight} exceeds cap after floor+DO recap"
            )
    # Soft-delete MSM still zero
    for r in vote.source_votes:
        if r.source == SignalSource.MULTI_SPEED_MOM:
            assert float(r.weight or 0.0) == 0.0


def test_cap_helper_after_equal_share_renorm() -> None:
    """Simulated floor renorm 0.85/0.05/0.05/0.05 → cap redistributes."""
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.85,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.05,
        SignalSource.GOOGLE_TRENDS: 0.05,
        SignalSource.VIX_TERM_STRUCTURE: 0.05,
        SignalSource.MULTI_SPEED_MOM: 0.0,
    }
    out = EnsembleVoter._apply_per_signal_weight_cap(
        weights,
        max_weight=0.50,
        soft_delete={SignalSource.MULTI_SPEED_MOM},
    )
    assert out[SignalSource.CROSS_ASSET_RV] <= 0.50 + 1e-9
    assert out[SignalSource.MULTI_SPEED_MOM] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_adaptive_soft_delete_multiplier_zero(tmp_path: Path) -> None:
    """MSM baseline 0 → multiplier disclosed as 0, not attribution boost."""
    baseline = {
        "multi_speed_momentum": 0.0,
        "cross_asset_rv": 0.5,
        "alternative_data": 0.3,
        "google_trends": 0.2,
    }
    attribution = {
        "sources": {
            "multi_speed_momentum": {
                "sharpe_contribution": 2.0,
                "total_readings": 100,
                "hit_rate": 0.7,
            },
            "cross_asset_rv": {
                "sharpe_contribution": 0.1,
                "total_readings": 100,
                "hit_rate": 0.5,
            },
            "alternative_data": {
                "sharpe_contribution": 0.1,
                "total_readings": 100,
                "hit_rate": 0.5,
            },
            "google_trends": {
                "sharpe_contribution": 0.1,
                "total_readings": 100,
                "hit_rate": 0.5,
            },
        },
        "timestamp": "2026-07-22",
    }
    aw = AdaptiveEnsembleWeights(
        base_weights=baseline,
        state_file=tmp_path / "adaptive.json",
    )
    aw.update_weights(attribution, "normal")

    assert aw.adjusted_weights.get("multi_speed_momentum", 0.0) == 0.0
    assert aw.multipliers.get("multi_speed_momentum", 1.0) == 0.0
    assert "multi_speed_momentum" in (aw.zero_baseline_exclusions or [])


def test_adaptive_empty_attribution_zero_baseline_mult(tmp_path: Path) -> None:
    baseline = {"multi_speed_momentum": 0.0, "cross_asset_rv": 1.0}
    aw = AdaptiveEnsembleWeights(
        base_weights=baseline,
        state_file=tmp_path / "adaptive2.json",
    )
    aw.update_weights({"sources": {}, "timestamp": "now"}, "normal")
    assert aw.multipliers.get("multi_speed_momentum") == 0.0
    assert aw.multipliers.get("cross_asset_rv") == 1.0
