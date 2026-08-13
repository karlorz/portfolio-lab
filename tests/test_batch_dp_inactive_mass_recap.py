"""Batch DP: zero inactive_signal mass then renorm+cap; dashboard rollup cap."""

from __future__ import annotations

from pathlib import Path

from src.dashboard.generator import DashboardGenerator
from src.strategy.ensemble_voter import (
    EnsembleVoter,
    Regime,
    SignalReading,
    SignalSource,
)


def _reading(
    source: SignalSource,
    value: float,
    conf: float = 0.6,
    *,
    is_active: bool = True,
) -> SignalReading:
    return SignalReading(
        source=source,
        timestamp="2026-07-22",
        value=value,
        confidence=conf,
        weight=0.0,
        regime_fit="all",
        asset_signals={"SPY": value},
        explanation=source.value,
        is_active=is_active,
    )


def test_zero_inactive_and_recap_redistributes_intl_mass() -> None:
    """~20% on inactive intl must redistribute; max stays ≤50%."""
    weights = {
        SignalSource.CROSS_ASSET_RV: 0.50,
        SignalSource.INTERNATIONAL_MOMENTUM: 0.20,
        SignalSource.MULTI_TIMEFRAME_FUSION: 0.10,
        SignalSource.GOOGLE_TRENDS: 0.08,
        SignalSource.VIX_TERM_STRUCTURE: 0.12,
        SignalSource.MULTI_SPEED_MOM: 0.0,
    }
    readings = {
        SignalSource.CROSS_ASSET_RV: _reading(SignalSource.CROSS_ASSET_RV, 0.5),
        SignalSource.INTERNATIONAL_MOMENTUM: _reading(
            SignalSource.INTERNATIONAL_MOMENTUM, 0.0, 0.0, is_active=False
        ),
        SignalSource.MULTI_TIMEFRAME_FUSION: _reading(
            SignalSource.MULTI_TIMEFRAME_FUSION, 0.0
        ),
        SignalSource.GOOGLE_TRENDS: _reading(SignalSource.GOOGLE_TRENDS, 0.1),
        SignalSource.VIX_TERM_STRUCTURE: _reading(
            SignalSource.VIX_TERM_STRUCTURE, 0.05
        ),
        SignalSource.MULTI_SPEED_MOM: _reading(SignalSource.MULTI_SPEED_MOM, -0.3),
    }
    voter = EnsembleVoter(data_path=Path("/tmp/batch-dp"))
    out = voter._zero_inactive_and_recap(weights, readings, "NORMAL")
    assert out[SignalSource.INTERNATIONAL_MOMENTUM] == 0.0
    assert out[SignalSource.MULTI_SPEED_MOM] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-6
    assert max(out.values()) <= 0.50 + 1e-6
    # Mass that was on intl redistributed to awake arms
    assert out[SignalSource.CROSS_ASSET_RV] >= 0.50 - 1e-6  # may stay at cap


def test_apply_weights_after_dp_sum_active_is_one(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: inactive intl + soft-delete MSM → active vote mass sums ~1, max≤0.5."""
    monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "0")
    voter = EnsembleVoter(data_path=tmp_path)
    readings = {
        SignalSource.CROSS_ASSET_RV: _reading(SignalSource.CROSS_ASSET_RV, 0.5, 0.7),
        SignalSource.INTERNATIONAL_MOMENTUM: _reading(
            SignalSource.INTERNATIONAL_MOMENTUM, 0.0, 0.0, is_active=False
        ),
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
        SignalSource.MULTI_SPEED_MOM: _reading(SignalSource.MULTI_SPEED_MOM, -0.3),
        SignalSource.UNIFIED_OVERLAY: _reading(SignalSource.UNIFIED_OVERLAY, -0.5),
    }
    vote = voter.compute_vote(readings, Regime.NORMAL, 0.8)
    # Health sleep may zero more arms; remaining active mass should be ~1
    # when any contribute
    pos = [float(r.weight) for r in vote.source_votes if float(r.weight or 0) > 1e-9]
    for w in pos:
        assert w <= 0.50 + 1e-6
    if pos:
        assert abs(sum(pos) - 1.0) < 0.02
    for r in vote.source_votes:
        if r.source == SignalSource.INTERNATIONAL_MOMENTUM:
            assert float(r.weight or 0) == 0.0
        if r.source == SignalSource.MULTI_SPEED_MOM:
            assert float(r.weight or 0) == 0.0


def test_dashboard_rollup_caps_after_renorm() -> None:
    """Simulated post-inactive renorm that would put CAR at 85% gets clipped."""
    statuses = [
        {
            "source": "cross_asset_rv",
            "contributing": True,
            "active_weight": 0.85,
            "effective_weight": 0.68,
            "configured_weight": 0.11,
        },
        {
            "source": "multi_timeframe_fusion",
            "contributing": True,
            "active_weight": 0.05,
            "effective_weight": 0.04,
            "configured_weight": 0.09,
        },
        {
            "source": "google_trends",
            "contributing": True,
            "active_weight": 0.06,
            "effective_weight": 0.048,
            "configured_weight": 0.05,
        },
        {
            "source": "vix_term_structure",
            "contributing": True,
            "active_weight": 0.04,
            "effective_weight": 0.032,
            "configured_weight": 0.05,
        },
        {
            "source": "international_momentum",
            "contributing": False,
            "active_weight": 0.0,
            "effective_weight": 0.0,
            "configured_weight": 0.21,
        },
    ]
    rollup = DashboardGenerator._ensemble_active_weights_rollup(statuses)
    assert rollup["per_signal_active_weight_cap_applied"] is True
    assert max(rollup["active_weights"].values()) <= 0.50 + 1e-6
    assert abs(rollup["active_weights_sum"] - 1.0) < 1e-3
