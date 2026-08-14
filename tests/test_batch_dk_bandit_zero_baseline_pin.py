"""Batch DK: bandit/adaptive/noise must not reinflate static-zero soft-delete arms."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.strategy.ensemble_voter import (
    EnsembleVoter,
    REGIME_WEIGHTS,
    Regime,
    SignalSource,
)


def _voter_with_bandit_warmup(tmp_path: Path) -> EnsembleVoter:
    voter = EnsembleVoter(data_path=tmp_path)
    # Force full blend so bandit mass can reinflate soft-delete arms
    voter.bandit_days = 500
    voter.bandit_observations = 5000
    rng = np.random.RandomState(7)
    for _ in range(30):
        for src in SignalSource:
            voter.update_bandit(src.value, "NORMAL", float(rng.normal(0.001, 0.01)))
    return voter


def test_get_blended_weights_pins_static_zero_msm(tmp_path: Path) -> None:
    voter = _voter_with_bandit_warmup(tmp_path)
    blended = voter.get_blended_weights("NORMAL")
    assert SignalSource.MULTI_SPEED_MOM in blended
    assert float(blended[SignalSource.MULTI_SPEED_MOM]) == 0.0
    # Soft-delete static remains zero across regimes
    for regime in Regime:
        static = REGIME_WEIGHTS.get(regime, {})
        if float(static.get(SignalSource.MULTI_SPEED_MOM, 0.0) or 0.0) == 0.0:
            w = voter.get_blended_weights(regime.name)
            assert float(w.get(SignalSource.MULTI_SPEED_MOM, 0.0)) == 0.0


def test_pin_helper_zeros_and_renorms() -> None:
    weights = {
        SignalSource.MULTI_SPEED_MOM: 0.12,
        SignalSource.CROSS_ASSET_RV: 0.50,
        SignalSource.ALTERNATIVE_DATA: 0.38,
    }
    pinned = EnsembleVoter._pin_zero_baseline_weights(weights, "NORMAL")
    assert pinned[SignalSource.MULTI_SPEED_MOM] == 0.0
    total = sum(pinned.values())
    assert abs(total - 1.0) < 1e-9
    assert pinned[SignalSource.CROSS_ASSET_RV] > 0.5


def test_exploration_noise_does_not_reinflate_soft_delete(tmp_path: Path) -> None:
    voter = EnsembleVoter(data_path=tmp_path)
    weights = {
        SignalSource.MULTI_SPEED_MOM: 0.0,
        SignalSource.CROSS_ASSET_RV: 0.7,
        SignalSource.ALTERNATIVE_DATA: 0.3,
    }
    # Force exploration path; _apply_exploration_noise resolves random/np
    # from its owner module (post ENSEMBLE-VOTER-MIXINS split). Stub those
    # module bindings so the process-global random module / numpy are
    # never patched.
    fake_random = MagicMock()
    fake_random.random.return_value = 0.0
    fake_np = MagicMock()
    # Dirichlet would try to put mass on MSM if alpha>0 for zero arm
    fake_np.random.dirichlet.return_value = np.array([0.2, 0.5, 0.3])
    with patch("src.strategy.ensemble_voter_vote.random", fake_random):
        with patch("src.strategy.ensemble_voter_vote.np", fake_np):
            out = voter._apply_exploration_noise(weights, Regime.NORMAL)
    assert float(out[SignalSource.MULTI_SPEED_MOM]) == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_compute_vote_pipeline_keeps_msm_zero(tmp_path: Path) -> None:
    """End-to-end: soft-delete MSM stays at 0 through full weight pipeline."""
    from src.strategy.ensemble_voter import SignalReading

    voter = _voter_with_bandit_warmup(tmp_path)
    readings = {
        SignalSource.MULTI_SPEED_MOM: SignalReading(
            source=SignalSource.MULTI_SPEED_MOM,
            timestamp="2026-07-22",
            value=-0.33,
            confidence=0.67,
            weight=0.0,
            regime_fit="all",
            asset_signals={"SPY": -0.3},
            explanation="msm shadow",
        ),
        SignalSource.CROSS_ASSET_RV: SignalReading(
            source=SignalSource.CROSS_ASSET_RV,
            timestamp="2026-07-22",
            value=0.5,
            confidence=0.7,
            weight=0.0,
            regime_fit="all",
            asset_signals={"SPY": 0.5},
            explanation="rv",
        ),
        SignalSource.ALTERNATIVE_DATA: SignalReading(
            source=SignalSource.ALTERNATIVE_DATA,
            timestamp="2026-07-22",
            value=0.2,
            confidence=0.6,
            weight=0.0,
            regime_fit="all",
            asset_signals={"SPY": 0.2},
            explanation="alt",
        ),
    }
    # Disable analysis floor path that can reinflate for caller-supplied NORMAL tests
    with patch.dict("os.environ", {"ENSEMBLE_EXPLORATION_EPSILON": "0"}):
        vote = voter.compute_vote(readings, Regime.NORMAL, 0.8)

    msm_votes = [s for s in vote.source_votes if s.source == SignalSource.MULTI_SPEED_MOM]
    # May or may not appear in votes depending on weight filter; weight must be 0
    for sv in msm_votes:
        assert float(sv.weight) == 0.0
    # Active weights mass on MSM via source_votes weights
    msm_w = sum(float(s.weight) for s in vote.source_votes if s.source == SignalSource.MULTI_SPEED_MOM)
    assert msm_w == 0.0
