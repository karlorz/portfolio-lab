"""Unit tests for src.strategy.ensemble_voter_weights WeightsMixin."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.signals.regime_spec import Regime, SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_voter_weights import WeightsMixin


class DummyWeighter(WeightsMixin):
    """Test harness incorporating WeightsMixin."""

    _ConsensusResult = namedtuple(
        "_ConsensusResult",
        [
            "weighted_consensus",
            "agreement",
            "equity_bias",
            "duration_bias",
            "gold_bias",
            "action",
            "action_confidence",
        ],
    )

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db_path = data_path / "ensemble_signals.db"
        self.bandit = None
        self.current_regime = Regime.NORMAL

    def _static_zero_baseline_sources(self, regime_name: str) -> set:
        return set()

    def _pin_zero_baseline_weights(self, weights: dict, regime_name: str) -> dict:
        return weights


class TestApplyPerSignalWeightCap:
    """Test _apply_per_signal_weight_cap water-filling redistribution."""

    def test_empty_weights_returns_input(self):
        assert WeightsMixin._apply_per_signal_weight_cap({}) == {}

    def test_invalid_or_extreme_cap_passthrough(self):
        weights = {"s1": 0.8, "s2": 0.2}
        assert WeightsMixin._apply_per_signal_weight_cap(weights, max_weight=0.0) == weights
        assert WeightsMixin._apply_per_signal_weight_cap(weights, max_weight=1.0) == weights

    def test_single_positive_arm_uncapped(self):
        weights = {"s1": 1.0, "s2": 0.0}
        capped = WeightsMixin._apply_per_signal_weight_cap(weights, max_weight=0.50)
        assert capped["s1"] == 1.0

    def test_water_filling_redistributes_excess(self):
        # 3 arms: one over cap (0.70), two under (0.15, 0.15), cap = 0.50
        weights = {"s1": 0.70, "s2": 0.15, "s3": 0.15}
        capped = WeightsMixin._apply_per_signal_weight_cap(weights, max_weight=0.50)
        assert capped["s1"] == pytest.approx(0.50)
        assert capped["s2"] == pytest.approx(0.25)
        assert capped["s3"] == pytest.approx(0.25)
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_soft_delete_remains_zero(self):
        weights = {"s1": 0.60, "s2": 0.30, "s3": 0.10}
        capped = WeightsMixin._apply_per_signal_weight_cap(
            weights, max_weight=0.50, soft_delete={"s3"}
        )
        assert capped["s3"] == 0.0
        assert capped["s1"] == pytest.approx(0.50)
        assert capped["s2"] == pytest.approx(0.50)


class TestGetBlendedWeights:
    """Test get_blended_weights blending logic."""

    def test_uninitialized_or_cold_start_bandit_returns_static(self, tmp_path: Path):
        weighter = DummyWeighter(tmp_path)
        weights = weighter.get_blended_weights("NORMAL")
        assert isinstance(weights, dict)
        assert len(weights) > 0


class TestDetermineAction:
    """Test _determine_action regime-conditional logic."""

    def test_crisis_regime_forces_risk_off(self):
        action, conf = WeightsMixin._determine_action(Regime.CRISIS, 0.9, 0.5, 0.8)
        assert action == "risk_off"
        assert conf == 0.9

    def test_high_equity_bias_increases_equity(self):
        action, conf = WeightsMixin._determine_action(Regime.NORMAL, 0.8, 0.5, 0.9)
        assert action == "increase_equity"
        assert conf == pytest.approx(0.9 * 0.5)

    def test_low_equity_bias_decreases_equity(self):
        action, conf = WeightsMixin._determine_action(Regime.NORMAL, 0.8, -0.5, 0.9)
        assert action == "decrease_equity"
        assert conf == pytest.approx(0.9 * 0.5)

    def test_neutral_action_when_agreement_below_threshold(self):
        action, conf = WeightsMixin._determine_action(Regime.NORMAL, 0.8, 0.1, 0.5)
        assert action == "neutral"
        assert conf == pytest.approx(0.5 * 0.8)


class TestComputeConsensusAndBuildVote:
    """Test _compute_consensus and _build_vote."""

    def test_compute_consensus_and_build_vote(self, tmp_path: Path):
        weighter = DummyWeighter(tmp_path)
        reading1 = SignalReading(
            source=SignalSource.MULTI_SPEED_MOM,
            timestamp="2026-08-17T12:00:00Z",
            value=0.4,
            confidence=0.8,
            weight=0.6,
            regime_fit="normal",
            asset_signals={"SPY": 0.4, "TLT": -0.2, "GLD": 0.1},
        )
        reading2 = SignalReading(
            source=SignalSource.CROSS_ASSET_RV,
            timestamp="2026-08-17T12:00:00Z",
            value=0.2,
            confidence=0.7,
            weight=0.4,
            regime_fit="normal",
            asset_signals={"SPY": 0.2, "TLT": -0.1, "GLD": 0.0},
        )
        signals = [reading1, reading2]

        consensus = weighter._compute_consensus(signals, Regime.NORMAL, 0.85)
        assert consensus.weighted_consensus == pytest.approx(0.4 * 0.6 + 0.2 * 0.4)
        assert consensus.equity_bias == pytest.approx(0.4 * 0.6 + 0.2 * 0.4)

        vote = weighter._build_vote(signals, consensus, Regime.NORMAL, 0.85)
        assert vote.regime == Regime.NORMAL
        assert vote.num_sources == 2
        assert vote.n_eff > 1.0
        assert vote.weight_entropy > 0.0


class TestRecommendAllocation:
    """Test recommend_allocation output formatting and bounds."""

    def test_recommend_allocation(self, tmp_path: Path):
        weighter = DummyWeighter(tmp_path)
        reading = SignalReading(
            source=SignalSource.MULTI_SPEED_MOM,
            timestamp="2026-08-17T12:00:00Z",
            value=0.5,
            confidence=0.8,
            weight=1.0,
            regime_fit="normal",
            asset_signals={"SPY": 0.5, "TLT": 0.0, "GLD": 0.0},
        )
        consensus = weighter._compute_consensus([reading], Regime.NORMAL, 0.8)
        vote = weighter._build_vote([reading], consensus, Regime.NORMAL, 0.8)

        rec = weighter.recommend_allocation(
            base_allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            vote=vote,
            max_shift=0.10,
        )
        assert "assets" in rec
        assert "SPY" in rec["assets"]
        assert sum(a["new"] for a in rec["assets"].values()) == pytest.approx(1.0)
