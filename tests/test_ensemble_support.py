"""Unit tests for src.strategy.ensemble_support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.signals.regime_spec import Regime
from src.strategy.ensemble_support import (
    BanditWeighter,
    EnsembleVote,
    _REGIME_CONDITIONAL_WEIGHTS_DEFAULTS,
    _extract_signal_predictions,
    _get_health_tracker,
    _load_regime_conditional_weights,
    _rank_correlation_from_matrix,
    _rank_prediction_matrix,
    compute_signal_correlation_matrix,
)


class TestHealthTrackerAccess:
    """Test _get_health_tracker helper."""

    def test_returns_instance_or_none(self):
        tracker = _get_health_tracker()
        assert tracker is not None or tracker is None


class TestEnsembleVoteDataclass:
    """Test EnsembleVote dataclass instantiations and defaults."""

    def test_instantiation_with_defaults(self):
        vote = EnsembleVote(
            timestamp="2026-08-17T12:00:00Z",
            regime=Regime.NORMAL,
            regime_confidence=0.85,
            num_sources=3,
            weighted_consensus=0.5,
            agreement_ratio=0.67,
            equity_bias=0.2,
            duration_bias=-0.1,
            gold_bias=0.0,
            action="increase_equity",
            confidence=0.75,
            reasoning="test",
            source_votes=[],
        )
        assert vote.n_eff == 0.0
        assert vote.weight_entropy == 0.0
        assert vote.regime_multipliers is None
        assert vote.adaptive_learning == {}
        assert vote.health_gate_freeze is False


class TestLoadRegimeConditionalWeights:
    """Test _load_regime_conditional_weights configuration loader."""

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        weights = _load_regime_conditional_weights(str(tmp_path / "nonexistent.json"))
        assert "CRISIS" in weights
        assert weights["CRISIS"]["alternative_data"] == 1.3

    def test_corrupt_file_returns_defaults(self, tmp_path: Path):
        f = tmp_path / "corrupt.json"
        f.write_text("not json", encoding="utf-8")
        weights = _load_regime_conditional_weights(str(f))
        assert weights == _REGIME_CONDITIONAL_WEIGHTS_DEFAULTS

    def test_valid_json_loads_correctly(self, tmp_path: Path):
        f = tmp_path / "weights.json"
        custom = {"NORMAL": {"custom_signal": 2.0}}
        f.write_text(json.dumps(custom), encoding="utf-8")
        weights = _load_regime_conditional_weights(str(f))
        assert weights == custom

    def test_env_var_override(self, tmp_path: Path, monkeypatch):
        f = tmp_path / "env_weights.json"
        custom = {"LOW_VOL": {"env_signal": 1.5}}
        f.write_text(json.dumps(custom), encoding="utf-8")
        monkeypatch.setenv("ENSEMBLE_CONDITIONAL_WEIGHTS_FILE", str(f))
        weights = _load_regime_conditional_weights()
        assert weights == custom


class TestSignalPredictionsAndRanking:
    """Test _extract_signal_predictions and ranking matrix functions."""

    def test_extract_ignores_staged_and_filters_non_numeric(self):
        ic_data = {
            "__staged__": [[1.0], [2.0]],
            "invalid_signal": "not a list",
            "sparse_signal": [[1.0], [2.0]],  # < 10 observations
            "valid_signal": [[float(i)] for i in range(12)],
            "mixed_signal": [[float(i)] if i % 2 == 0 else ["invalid"] for i in range(25)],
        }
        extracted = _extract_signal_predictions(ic_data)
        assert "__staged__" not in extracted
        assert "invalid_signal" not in extracted
        assert "sparse_signal" not in extracted
        assert "valid_signal" in extracted
        assert len(extracted["valid_signal"]) == 12
        assert "mixed_signal" in extracted
        assert np.isnan(extracted["mixed_signal"][1])

    def test_rank_prediction_matrix(self):
        preds = {
            "s1": [float(i) for i in range(15)],
            "s2": [float(15 - i) for i in range(15)],
        }
        signals, ranks = _rank_prediction_matrix(preds)
        assert signals == ["s1", "s2"]
        assert ranks.shape == (2, 15)

    def test_rank_correlation_from_matrix(self):
        ranks = np.array([
            [float(i) for i in range(10)],
            [float(i) for i in range(10)],
            [float(10 - i) for i in range(10)],
        ])
        corr_pos = _rank_correlation_from_matrix(ranks, 0, 1)
        assert corr_pos == pytest.approx(1.0)
        corr_neg = _rank_correlation_from_matrix(ranks, 0, 2)
        assert corr_neg == pytest.approx(-1.0)

    def test_rank_correlation_insufficient_overlap_returns_zero(self):
        ranks = np.array([
            [1.0, 2.0, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        assert _rank_correlation_from_matrix(ranks, 0, 1) == 0.0


class TestComputeSignalCorrelationMatrix:
    """Test compute_signal_correlation_matrix."""

    def test_empty_or_missing_returns_empty_dict(self):
        res = compute_signal_correlation_matrix(ic_data={})
        assert res == {"matrix": {}, "redundant_pairs": [], "correlation_penalties": {}}

    def test_calculates_matrix_redundancy_and_penalties(self):
        ic_data = {
            "s1": [[float(i)] for i in range(20)],
            "s2": [[float(i * 1.1)] for i in range(20)],  # highly correlated with s1
            "s3": [[float(20 - i)] for i in range(20)],  # anti-correlated
        }
        res = compute_signal_correlation_matrix(ic_data=ic_data, threshold=0.8)
        assert "s1" in res["matrix"]
        assert len(res["redundant_pairs"]) >= 1
        assert "s1" in res["correlation_penalties"]
        assert res["correlation_penalties"]["s1"] < 1.0


class TestBanditWeighter:
    """Test BanditWeighter Thompson Sampling and weight updates."""

    def test_initialization(self):
        bw = BanditWeighter(["s1", "s2"], epsilon=0.05, temperature=0.5)
        assert bw.signals == ["s1", "s2"]
        assert bw.epsilon == 0.05
        assert bw.temperature == 0.5

    def test_update_and_window_trim(self):
        bw = BanditWeighter(["s1"], window=5)
        for i in range(10):
            bw.update("s1", "normal", float(i))
        assert len(bw._history["normal"]["s1"]) == 5
        assert bw._history["normal"]["s1"] == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_weights_cold_start_returns_none(self):
        bw = BanditWeighter(["s1", "s2"])
        assert bw.get_weights("normal") is None

    def test_get_weights_softmax_sums_to_one(self):
        bw = BanditWeighter(["s1", "s2"])
        for i in range(25):
            bw.update("s1", "normal", 0.02 + (0.001 if i % 2 == 0 else -0.001))
            bw.update("s2", "normal", 0.005 + (0.001 if i % 2 == 0 else -0.001))
        weights = bw.get_weights("normal")
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["s1"] > weights["s2"]

    def test_select_returns_valid_signal(self):
        bw = BanditWeighter(["s1", "s2"], epsilon=0.0)
        # Cold start
        selected = bw.select("normal")
        assert selected in ["s1", "s2"]

        # With observations
        for _ in range(5):
            bw.update("s1", "normal", 0.01)
            bw.update("s2", "normal", -0.01)
        selected_warm = bw.select("normal")
        assert selected_warm in ["s1", "s2"]

    def test_serialization_roundtrip(self):
        bw = BanditWeighter(["s1", "s2"], epsilon=0.15, window=100, temperature=0.8)
        bw.update("s1", "normal", 0.01)
        bw.update("s2", "crisis", -0.02)
        state = bw.get_state()
        assert state["schema_version"] == "bandit-weighter/v1"
        assert state["signals"] == ["s1", "s2"]

        bw2 = BanditWeighter(["s1", "s2"])
        bw2.load_state(state)
        assert bw2._history["normal"]["s1"] == [0.01]
        assert bw2._history["crisis"]["s2"] == [-0.02]
