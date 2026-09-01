#!/usr/bin/env python3
"""Bounded characterization suite for EnsembleVoter VoteMixin (Q66).

Exercises real VoteMixin behavior and observable vote transformations:
1. compute_vote caller path vs. default resolution path
2. _resolve_inputs fallback, state storage, and reading preservation
3. _apply_regime_gating active filtering, renormalization, and disclosure
4. _apply_adaptive_weights fresh attribution vs. stale/insufficient passthrough
5. _apply_ic_weights ICMonitor integration, online blending, and status tracking
6. _apply_health_weights hard-zero, soft-floor, freeze/all-zero, and per-signal cap
7. _apply_correlation_penalty pairwise penalty attenuation, clipping, and renorm
8. _apply_utility_reweighting Sharpe contribution / hit rate blend
9. _apply_exploration_noise Dirichlet noise, static-zero preservation, and identity
10. _apply_diversity_floor floor enforcement, health-slept exclusion, and normalization
11. _apply_turnover_validation string/enum mapping, finite-value filtering, and renorm
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.signals.health_tracker import HealthScore, SignalHealthStatus
from src.signals.regime_gate import RegimeGate
from src.signals.regime_spec import Regime, SignalReading
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_support import EnsembleVote
from src.strategy.ensemble_voter import EnsembleVoter
from src.strategy.online_ic_weighter import OnlineICWeighter


pytestmark = pytest.mark.usefixtures("_hermetic_health_db")


@pytest.fixture(autouse=True)
def _isolate_vote_module_state(tmp_path: Path, monkeypatch):
    """Keep attribution, adaptive state, health logging, and correlation data hermetic."""
    attribution_dir = tmp_path / "vote_attribution"
    attribution_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "src.strategy.ensemble_voter_vote.ATTRIBUTION_DIR",
        attribution_dir,
    )
    monkeypatch.setattr(
        "src.strategy.adaptive_ensemble_weights.STATE_FILE",
        tmp_path / "adaptive_weights_state.json",
    )
    monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "0")
    with patch(
        "src.strategy.ensemble_voter_vote._get_health_tracker",
        return_value=None,
    ), patch(
        "src.strategy.ensemble_voter_vote.compute_signal_correlation_matrix",
        return_value={},
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_reading(
    source: SignalSource = SignalSource.CROSS_ASSET_RV,
    value: float = 0.5,
    confidence: float = 0.8,
    is_active: bool = True,
    asset_signals: Optional[Dict[str, float]] = None,
) -> SignalReading:
    """Construct a typed SignalReading for characterization tests."""
    return SignalReading(
        source=source,
        timestamp="2026-01-01T00:00:00",
        value=value,
        confidence=confidence,
        weight=0.0,
        regime_fit="all",
        asset_signals=asset_signals or {"SPY": 0.5, "TLT": -0.2, "GLD": 0.1},
        explanation="q66_characterization",
        is_active=is_active,
    )


def _make_test_voter(tmp_path: Path) -> EnsembleVoter:
    """Instantiate EnsembleVoter with an isolated SQLite db path."""
    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter.data_path = tmp_path
    voter.db_path = tmp_path / "ensemble_test.db"
    voter.current_readings = {}
    voter.current_regime = Regime.NORMAL
    voter.current_regime_confidence = 0.5
    voter._init_db()
    return voter


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestEnsembleVoterVoteSuite:
    """Characterization tests for VoteMixin pipeline and adjustment stages."""

    def test_compute_vote_caller_supplied_readings(self, tmp_path: Path):
        """Caller-supplied readings are evaluated directly and produce an observable vote."""
        voter = _make_test_voter(tmp_path)
        readings = {
            SignalSource.CROSS_ASSET_RV: _make_reading(
                source=SignalSource.CROSS_ASSET_RV,
                value=0.6,
                confidence=0.9,
                asset_signals={"SPY": 0.6, "TLT": -0.2, "GLD": 0.1},
            ),
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                source=SignalSource.ALTERNATIVE_DATA,
                value=0.4,
                confidence=0.8,
                asset_signals={"SPY": 0.4, "TLT": -0.1, "GLD": 0.2},
            ),
        }

        with patch.object(voter, "_persist_vote") as mock_persist:
            vote = voter.compute_vote(
                readings=readings,
                regime=Regime.NORMAL,
                regime_confidence=0.85,
            )

        assert isinstance(vote, EnsembleVote)
        assert vote.regime == Regime.NORMAL
        assert vote.regime_confidence == 0.85
        assert vote.num_sources == 2
        assert vote.weighted_consensus > 0.0
        assert vote.equity_bias > 0.0
        assert vote.action in ["increase_equity", "neutral"]
        assert mock_persist.called
        assert len(vote.source_votes) > 0

    def test_compute_vote_default_path_resolves_readings_and_regime(self, tmp_path: Path):
        """When inputs are None, _resolve_inputs falls back to detect_regime and current_readings."""
        voter = _make_test_voter(tmp_path)
        seeded_readings = {
            SignalSource.CROSS_ASSET_RV: _make_reading(
                source=SignalSource.CROSS_ASSET_RV,
                value=0.35,
            ),
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                source=SignalSource.ALTERNATIVE_DATA,
                value=0.25,
            ),
        }
        voter.current_readings = seeded_readings

        with patch.object(voter, "detect_regime", return_value=(Regime.NORMAL, 0.77)) as mock_detect, \
             patch.object(voter, "_persist_vote"):
            vote = voter.compute_vote()

        assert mock_detect.called
        assert vote.regime == Regime.NORMAL
        assert vote.regime_confidence == 0.77
        assert vote.num_sources == 2
        assert voter.current_regime == Regime.NORMAL
        assert voter.current_regime_confidence == 0.77

    def test_resolve_inputs_stores_state_and_defaults_confidence(self, tmp_path: Path):
        """_resolve_inputs sets instance state, applies 0.5 confidence fallback, and preserves reading dict."""
        voter = _make_test_voter(tmp_path)
        custom_readings = {
            SignalSource.CROSS_ASSET_RV: _make_reading(value=0.1),
            "custom_key": {"some": "raw_payload"},  # type: ignore[dict-item]
        }

        resolved_readings, resolved_regime, resolved_conf = voter._resolve_inputs(
            readings=custom_readings,  # type: ignore[arg-type]
            regime=Regime.RECOVERY,
            regime_confidence=None,
        )

        assert resolved_readings == custom_readings
        assert resolved_regime == Regime.RECOVERY
        assert resolved_conf == 0.5
        assert voter.current_regime == Regime.RECOVERY
        assert voter.current_regime_confidence == 0.5

    def test_apply_regime_gating_zeros_explicit_gated_arm_and_renormalizes(self, tmp_path: Path):
        """Regime gating zeros out signals inactive in the current regime and renormalizes remaining mass."""
        voter = _make_test_voter(tmp_path)
        gate = RegimeGate(
            gate_rules={
                "cross_asset_rv": {"HIGH_VOL", "CRISIS"},
                "alternative_data": set(),
            },
            confidence_threshold=0.5,
        )
        voter.regime_gate = gate

        initial_weights = {
            SignalSource.CROSS_ASSET_RV: 0.6,
            SignalSource.ALTERNATIVE_DATA: 0.4,
        }

        gated = voter._apply_regime_gating(initial_weights, "CRISIS", regime_confidence=0.8)

        assert gated[SignalSource.CROSS_ASSET_RV] == 0.0
        assert gated[SignalSource.ALTERNATIVE_DATA] == pytest.approx(1.0)
        assert "cross_asset_rv" in voter._regime_gated
        assert "regime_gate_off" in voter._regime_gated["cross_asset_rv"]

    def test_apply_adaptive_weights_fresh_data_adjusts_weights(self, tmp_path: Path, monkeypatch):
        """Fresh performance attribution data dynamically modifies relative signal weights."""
        voter = _make_test_voter(tmp_path)
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.strategy.ensemble_voter_vote.ATTRIBUTION_DIR", attr_dir)
        monkeypatch.setattr(
            "src.strategy.adaptive_ensemble_weights.STATE_FILE",
            tmp_path / "adaptive_weights_state.json",
        )

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {
                "cross_asset_rv": {
                    "total_readings": 30,
                    "sharpe_contribution": 1.8,
                    "hit_rate": 0.70,
                },
                "alternative_data": {
                    "total_readings": 30,
                    "sharpe_contribution": -1.2,
                    "hit_rate": 0.35,
                },
            },
        }
        (attr_dir / "attribution_20260901.json").write_text(json.dumps(payload))

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }

        adapted = voter._apply_adaptive_weights(base_weights, Regime.NORMAL)

        assert adapted[SignalSource.CROSS_ASSET_RV] > adapted[SignalSource.ALTERNATIVE_DATA]
        assert sum(adapted.values()) == pytest.approx(1.0)

    def test_apply_adaptive_weights_stale_or_insufficient_returns_original(self, tmp_path: Path, monkeypatch):
        """Stale attribution (>7 days) or low average readings (<5) leaves base weights unmodified."""
        voter = _make_test_voter(tmp_path)
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.strategy.ensemble_voter_vote.ATTRIBUTION_DIR", attr_dir)

        stale_payload = {
            "timestamp": "2025-01-01 00:00:00",
            "sources": {
                "cross_asset_rv": {"total_readings": 50, "sharpe_contribution": 1.0},
            },
        }
        (attr_dir / "attribution_stale.json").write_text(json.dumps(stale_payload))

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.6,
            SignalSource.ALTERNATIVE_DATA: 0.4,
        }
        adapted_stale = voter._apply_adaptive_weights(base_weights, Regime.NORMAL)
        assert adapted_stale == base_weights

        # Clean stale and write insufficient readings payload
        for f in attr_dir.glob("*.json"):
            f.unlink()

        insufficient_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {
                "cross_asset_rv": {"total_readings": 2, "sharpe_contribution": 1.0},
                "alternative_data": {"total_readings": 1, "sharpe_contribution": 1.0},
            },
        }
        (attr_dir / "attribution_low.json").write_text(json.dumps(insufficient_payload))

        adapted_low = voter._apply_adaptive_weights(base_weights, Regime.NORMAL)
        assert adapted_low == base_weights

    def test_apply_ic_weights_blends_and_updates_status(self, tmp_path: Path, monkeypatch):
        """Online IC weighting blends dynamic IC weights with current weights and sets active status."""
        voter = _make_test_voter(tmp_path)
        voter._use_ic_weights = True
        voter._ic_weighter = OnlineICWeighter(temperature=1.0)
        monkeypatch.setenv("ENSEMBLE_IC_WEIGHT_BLEND_ALPHA", "0.4")

        mock_monitor = MagicMock()
        mock_monitor.compute_ic.side_effect = lambda src: 0.25 if src == "cross_asset_rv" else 0.02
        mock_monitor.compute_ic_trend.return_value = "improving"

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }

        with patch("src.monitor.ic_decay_monitor.ICMonitor", return_value=mock_monitor):
            adjusted = voter._apply_ic_weights(base_weights, Regime.NORMAL)

        assert adjusted[SignalSource.CROSS_ASSET_RV] > adjusted[SignalSource.ALTERNATIVE_DATA]
        assert sum(adjusted.values()) == pytest.approx(1.0)
        assert voter._last_online_ic_learning_status["status"] == "active"
        assert voter._last_online_ic_learning_status["reason"] == "blending_with_static_weights"

    def test_apply_health_weights_hard_zero_and_soft_floor(self, tmp_path: Path):
        """Health gate hard-zeros eligible negative-IC arm and soft-floors degraded arm."""
        voter = _make_test_voter(tmp_path)

        mock_tracker = MagicMock()
        mock_scores = {
            "cross_asset_rv": HealthScore(
                source="cross_asset_rv",
                timestamp="2026-09-01T00:00:00",
                health_score=0.10,
                accuracy_30d=0.45,
                accuracy_60d=0.50,
                accuracy_90d=0.55,
                decay_rate=-0.001,
                status=SignalHealthStatus.UNHEALTHY.value,
                ic=-0.15,
                predictions_count=35,
            ),
            "alternative_data": HealthScore(
                source="alternative_data",
                timestamp="2026-09-01T00:00:00",
                health_score=0.40,
                accuracy_30d=0.48,
                accuracy_60d=0.51,
                accuracy_90d=0.52,
                decay_rate=-0.001,
                status=SignalHealthStatus.DEGRADED.value,
                ic=0.12,
                predictions_count=5,  # Insufficient for hard-zero, gets soft-floor
            ),
            "vix_term_structure": HealthScore(
                source="vix_term_structure",
                timestamp="2026-09-01T00:00:00",
                health_score=0.90,
                accuracy_30d=0.65,
                accuracy_60d=0.62,
                accuracy_90d=0.60,
                decay_rate=0.001,
                status=SignalHealthStatus.HEALTHY.value,
                ic=0.20,
                predictions_count=40,
            ),
        }
        mock_tracker.calculate_all_health_scores.return_value = mock_scores

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.3,
            SignalSource.ALTERNATIVE_DATA: 0.3,
            SignalSource.VIX_TERM_STRUCTURE: 0.4,
        }

        with patch("src.signals.health_tracker.SignalHealthTracker", return_value=mock_tracker):
            adjusted = voter._apply_health_weights(base_weights)

        assert adjusted[SignalSource.CROSS_ASSET_RV] == 0.0
        assert "cross_asset_rv" in voter._health_gate_slept
        assert "negative_ic" in voter._health_gate_sleep_reasons["cross_asset_rv"]
        assert "alternative_data" in voter._health_gate_soft_floor
        assert adjusted[SignalSource.VIX_TERM_STRUCTURE] > adjusted[SignalSource.ALTERNATIVE_DATA]
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_apply_health_weights_all_zero_triggers_freeze(self, tmp_path: Path):
        """When all active arms are hard-zeroed by health policy, health gate freezes without reinflating."""
        voter = _make_test_voter(tmp_path)

        mock_tracker = MagicMock()
        mock_scores = {
            "cross_asset_rv": HealthScore(
                source="cross_asset_rv",
                timestamp="2026-09-01T00:00:00",
                health_score=0.10,
                accuracy_30d=0.45,
                accuracy_60d=0.50,
                accuracy_90d=0.55,
                decay_rate=-0.001,
                status=SignalHealthStatus.UNHEALTHY.value,
                ic=-0.20,
                predictions_count=30,
            ),
            "alternative_data": HealthScore(
                source="alternative_data",
                timestamp="2026-09-01T00:00:00",
                health_score=0.05,
                accuracy_30d=0.40,
                accuracy_60d=0.45,
                accuracy_90d=0.48,
                decay_rate=-0.002,
                status=SignalHealthStatus.UNHEALTHY.value,
                ic=-0.30,
                predictions_count=30,
            ),
        }
        mock_tracker.calculate_all_health_scores.return_value = mock_scores

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }

        with patch("src.signals.health_tracker.SignalHealthTracker", return_value=mock_tracker):
            adjusted = voter._apply_health_weights(base_weights)

        assert voter._health_gate_freeze is True
        assert all(w == 0.0 for w in adjusted.values())

    def test_health_stage_renormalization_cap_enforcement(self, tmp_path: Path):
        """When health adjustment eliminates an arm and concentrates mass, cap enforcement restricts each arm to <= 50%."""
        voter = _make_test_voter(tmp_path)

        mock_tracker = MagicMock()
        mock_scores = {
            "cross_asset_rv": HealthScore(
                source="cross_asset_rv",
                timestamp="2026-09-01T00:00:00",
                health_score=0.05,
                accuracy_30d=0.40,
                accuracy_60d=0.45,
                accuracy_90d=0.48,
                decay_rate=-0.002,
                status=SignalHealthStatus.UNHEALTHY.value,
                ic=-0.25,
                predictions_count=30,
            ),
            "alternative_data": HealthScore(
                source="alternative_data",
                timestamp="2026-09-01T00:00:00",
                health_score=0.95,
                accuracy_30d=0.70,
                accuracy_60d=0.68,
                accuracy_90d=0.65,
                decay_rate=0.001,
                status=SignalHealthStatus.HEALTHY.value,
                ic=0.30,
                predictions_count=30,
            ),
            "vix_term_structure": HealthScore(
                source="vix_term_structure",
                timestamp="2026-09-01T00:00:00",
                health_score=0.90,
                accuracy_30d=0.65,
                accuracy_60d=0.62,
                accuracy_90d=0.60,
                decay_rate=0.001,
                status=SignalHealthStatus.HEALTHY.value,
                ic=0.25,
                predictions_count=30,
            ),
            "cross_asset_regime_arb": HealthScore(
                source="cross_asset_regime_arb",
                timestamp="2026-09-01T00:00:00",
                health_score=0.85,
                accuracy_30d=0.60,
                accuracy_60d=0.58,
                accuracy_90d=0.55,
                decay_rate=0.001,
                status=SignalHealthStatus.HEALTHY.value,
                ic=0.20,
                predictions_count=30,
            ),
        }
        mock_tracker.calculate_all_health_scores.return_value = mock_scores

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.30,
            SignalSource.ALTERNATIVE_DATA: 0.50,
            SignalSource.VIX_TERM_STRUCTURE: 0.10,
            SignalSource.CROSS_ASSET_REGIME_ARB: 0.10,
        }

        with patch("src.signals.health_tracker.SignalHealthTracker", return_value=mock_tracker):
            adjusted = voter._apply_health_weights(base_weights)
            capped = voter._cap_per_signal_weights(adjusted, "NORMAL")

        # After hard-zero of cross_asset_rv, alternative_data raw share became > 0.50, but capped at 0.50
        assert adjusted[SignalSource.CROSS_ASSET_RV] == 0.0
        assert adjusted[SignalSource.ALTERNATIVE_DATA] > 0.50
        assert capped[SignalSource.ALTERNATIVE_DATA] == pytest.approx(0.50)
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_apply_correlation_penalty_attenuates_redundant_arm(self, tmp_path: Path):
        """Redundant correlated signals receive a capped correlation penalty factor."""
        voter = _make_test_voter(tmp_path)

        corr_payload = {
            "matrix": {"cross_asset_rv": {"alternative_data": 0.85}},
            "redundant_pairs": [("cross_asset_rv", "alternative_data", 0.85)],
            "correlation_penalties": {
                "cross_asset_rv": 0.40,  # Below 0.50 clip limit -> should clip to 0.50
                "alternative_data": 0.90,
            },
        }

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }

        with patch("src.strategy.ensemble_voter_vote.compute_signal_correlation_matrix", return_value=corr_payload):
            adjusted = voter._apply_correlation_penalty(base_weights)

        # alternative_data penalty (0.90) > cross_asset_rv clipped penalty (0.50)
        assert adjusted[SignalSource.ALTERNATIVE_DATA] > adjusted[SignalSource.CROSS_ASSET_RV]
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_apply_utility_reweighting_positive_sharpe_boost(self, tmp_path: Path, monkeypatch):
        """Utility reweighting boosts signals with positive Sharpe contributions and good hit rates."""
        voter = _make_test_voter(tmp_path)
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.strategy.ensemble_voter_vote.ATTRIBUTION_DIR", attr_dir)

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {
                "cross_asset_rv": {
                    "total_readings": 25,
                    "sharpe_contribution": 1.5,
                    "hit_rate": 0.65,
                },
                "alternative_data": {
                    "total_readings": 25,
                    "sharpe_contribution": -1.0,
                    "hit_rate": 0.40,
                },
            },
        }
        (attr_dir / "attribution_util.json").write_text(json.dumps(payload))

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
        }

        adjusted = voter._apply_utility_reweighting(base_weights, Regime.NORMAL)

        assert adjusted[SignalSource.CROSS_ASSET_RV] > adjusted[SignalSource.ALTERNATIVE_DATA]
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_apply_exploration_noise_deterministic_and_static_zero_preservation(self, tmp_path: Path, monkeypatch):
        """Exploration noise changes active weights under positive epsilon, but preserves static zeros and identity when epsilon=0."""
        voter = _make_test_voter(tmp_path)

        # Baseline weights in NORMAL regime where MULTI_SPEED_MOM has 0 static weight
        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.5,
            SignalSource.MULTI_SPEED_MOM: 0.0,
        }

        # Case 1: epsilon = 0 -> identity
        monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "0.0")
        unchanged = voter._apply_exploration_noise(base_weights, Regime.NORMAL)
        assert unchanged == base_weights

        # Case 2: forced exploration (epsilon = 1.0) with deterministic RNG
        monkeypatch.setenv("ENSEMBLE_EXPLORATION_EPSILON", "1.0")
        monkeypatch.setenv("ENSEMBLE_EXPLORATION_ALPHA", "10.0")

        with patch("numpy.random.dirichlet", return_value=np.array([0.7, 0.3, 0.0001])):
            adjusted = voter._apply_exploration_noise(base_weights, Regime.NORMAL)

        # Static zero arm (MULTI_SPEED_MOM) remains pinned to 0
        assert adjusted[SignalSource.MULTI_SPEED_MOM] == 0.0
        assert adjusted[SignalSource.CROSS_ASSET_RV] == pytest.approx(0.7)
        assert adjusted[SignalSource.ALTERNATIVE_DATA] == pytest.approx(0.3)
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_apply_diversity_floor_raises_low_arms_and_preserves_slept(self, tmp_path: Path):
        """Diversity floor lifts low non-slept weights relative to base while keeping health-slept arms at zero."""
        voter = _make_test_voter(tmp_path)
        voter._health_gate_slept = ["cross_asset_rv"]

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.0,      # Health slept
            SignalSource.ALTERNATIVE_DATA: 0.98,   # Dominant
            SignalSource.VIX_TERM_STRUCTURE: 0.02, # Below 10% floor
        }

        adjusted = voter._apply_diversity_floor(base_weights, floor=0.10)

        assert adjusted[SignalSource.CROSS_ASSET_RV] == 0.0
        # VIX_TERM_STRUCTURE was 0.02; with floor=0.10 and renorm, its weight increases substantially
        assert adjusted[SignalSource.VIX_TERM_STRUCTURE] > base_weights[SignalSource.VIX_TERM_STRUCTURE]
        assert adjusted[SignalSource.ALTERNATIVE_DATA] < base_weights[SignalSource.ALTERNATIVE_DATA]
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_apply_turnover_validation_string_mapping_finite_values_and_renorm(self, tmp_path: Path):
        """Turnover validation filters NaN readings, translates string/enum dicts, applies penalty, and renormalizes."""
        voter = _make_test_voter(tmp_path)

        readings = {
            SignalSource.CROSS_ASSET_RV: _make_reading(
                source=SignalSource.CROSS_ASSET_RV,
                value=0.5,
            ),
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                source=SignalSource.ALTERNATIVE_DATA,
                value=float("nan"),  # Must be filtered by _extract_signal_values
            ),
            SignalSource.VIX_TERM_STRUCTURE: _make_reading(
                source=SignalSource.VIX_TERM_STRUCTURE,
                value=-0.3,
            ),
        }

        base_weights = {
            SignalSource.CROSS_ASSET_RV: 0.5,
            SignalSource.ALTERNATIVE_DATA: 0.2,
            SignalSource.VIX_TERM_STRUCTURE: 0.3,
        }

        mock_validator = MagicMock()
        # Returns penalized string dict (penalizes cross_asset_rv from 0.5 to 0.25)
        mock_validator.get_adjusted_weights.return_value = {
            "cross_asset_rv": 0.25,
            "alternative_data": 0.20,
            "vix_term_structure": 0.30,
        }

        with patch("src.strategy.turnover_validator.TurnoverValidator", return_value=mock_validator), \
             patch.object(voter, "_apply_basis_pursuit", side_effect=lambda sv, bw, r: bw), \
             patch.object(voter, "_apply_regret_weighting", side_effect=lambda sv, bw, r: bw):
            adjusted = voter._apply_turnover_validation(base_weights, readings, Regime.NORMAL)

        # Cross asset was penalized relative to vix_term_structure; weights sum to 1.0
        assert adjusted[SignalSource.CROSS_ASSET_RV] < adjusted[SignalSource.VIX_TERM_STRUCTURE]
        assert sum(adjusted.values()) == pytest.approx(1.0)
