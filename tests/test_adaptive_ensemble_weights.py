#!/usr/bin/env python3
"""
Tests for v6.09 Adaptive Ensemble Signal Weighting.

Tests the AdaptiveEnsembleWeights class including:
- Multiplier computation from Sharpe values
- Weight normalization
- Edge cases (NaN, None, zero readings)
- Attribution data integration
- State persistence
- Fallback behavior
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np

from src.strategy.adaptive_ensemble_weights import (
    AdaptiveEnsembleWeights,
    DEFAULT_CONFIG,
    WeightAdjustment,
    AdaptiveWeightsState,
)
from dataclasses import asdict


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_state_dir(monkeypatch):
    """Temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "adaptive_weights_state.json"
        monkeypatch.setattr(
            "src.strategy.adaptive_ensemble_weights.STATE_FILE",
            state_path,
        )
        # Also set DATA_DIR to use the tmp dir for attribution
        monkeypatch.setattr(
            "src.strategy.adaptive_ensemble_weights.DATA_DIR",
            Path(tmp),
        )
        monkeypatch.setattr(
            "src.strategy.adaptive_ensemble_weights.ATTRIBUTION_DIR",
            Path(tmp) / "attribution",
        )
        yield tmp


@pytest.fixture
def sample_base_weights():
    """Realistic base weights mirroring ensemble_voter NORMAL regime."""
    return {
        "tsfm_momentum": 0.28,
        "multi_speed_momentum": 0.19,
        "cta_trend": 0.10,
        "macro_momentum": 0.08,
        "factor_rotation": 0.05,
        "duration_regime": 0.05,
        "mean_reversion": 0.03,
        "hmm_regime": 0.02,
        "circuit_breaker": 0.0,
        "closing_auction": 0.03,
        "unified_overlay": 0.02,
        "transformer_regime": 0.03,
        "transient_factors": 0.03,
        "visibility_graph": 0.02,
        "vp_macd": 0.01,
        "cross_asset_rv": 0.01,
        "regime_classifier": 0.03,
        "factor_timing": 0.03,
        "risk_budget": 0.02,
    }


@pytest.fixture
def sample_attribution_good():
    """Attribution data where most sources have positive Sharpe."""
    sources = {}
    base_sources = [
        ("tsfm_momentum", 0.8, 45),
        ("multi_speed_momentum", 0.6, 50),
        ("cta_trend", 0.95, 60),
        ("macro_momentum", 0.4, 30),
        ("factor_rotation", 0.55, 25),
        ("duration_regime", 0.3, 20),
        ("mean_reversion", 0.7, 35),
        ("hmm_regime", 0.2, 15),
        ("closing_auction", 0.5, 22),
        ("unified_overlay", 0.35, 18),
        ("transformer_regime", 0.45, 28),
        ("transient_factors", 0.6, 32),
        ("visibility_graph", 0.3, 20),
        ("vp_macd", 0.4, 15),
        ("cross_asset_rv", 0.25, 12),
        ("regime_classifier", 0.5, 25),
        ("factor_timing", 0.65, 30),
        ("risk_budget", 0.55, 22),
    ]
    for name, sharpe, readings in base_sources:
        sources[name] = {
            "source": name,
            "display_name": name.replace("_", " ").title(),
            "category": "trend",
            "total_readings": readings,
            "active_days": readings,
            "hit_rate": 0.55,
            "win_rate": 0.52,
            "avg_return_bps": 2.5,
            "total_return_bps": readings * 2.5,
            "sharpe_contribution": sharpe,
            "max_consecutive_losses": 3,
            "avg_correlation": 0.3,
            "avg_weight": 0.05,
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "start_date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "analysis_days": 90,
        "sources": sources,
        "best_source": "cta_trend",
        "worst_source": "hmm_regime",
        "avg_hit_rate": 0.52,
        "avg_correlation": 0.28,
    }


@pytest.fixture
def sample_attribution_mixed():
    """Attribution with mixed performance - some negative Sharpe sources."""
    sources = {}
    base_data = [
        ("tsfm_momentum", 0.9, 60),
        ("multi_speed_momentum", -0.3, 55),  # Negative!
        ("cta_trend", 1.2, 65),
        ("macro_momentum", 0.5, 32),
        ("mean_reversion", -0.8, 40),  # Very negative!
        ("hmm_regime", 0.1, 10),
        ("factor_timing", 0.75, 35),
        ("risk_budget", 0.4, 20),
    ]
    for name, sharpe, readings in base_data:
        sources[name] = {
            "source": name,
            "total_readings": readings,
            "sharpe_contribution": sharpe,
            "avg_weight": 0.05,
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "start_date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "analysis_days": 90,
        "sources": sources,
    }


@pytest.fixture
def adaptive_weights(tmp_state_dir, sample_base_weights):
    """Create AdaptiveEnsembleWeights instance with sample base weights."""
    return AdaptiveEnsembleWeights(base_weights=sample_base_weights)


# ── Multiplier Computation Tests ──────────────────────────────────────────


class TestMultiplierComputation:
    """Verify the multiplier function under various Sharpe scenarios."""

    def test_positive_sharpe_produces_boost(self, adaptive_weights):
        """Sharpe > baseline → multiplier between 1.0 and 2.0."""
        attrib = {"total_readings": 30, "sharpe_contribution": 0.8}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert 1.0 <= mult <= 2.0, f"Expected multiplier in [1.0, 2.0], got {mult}"

    def test_low_sharpe_slight_boost(self, adaptive_weights):
        """Sharpe slightly above zero → small boost."""
        attrib = {"total_readings": 30, "sharpe_contribution": 0.3}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert 0.25 <= mult <= 1.0, f"Expected multiplier < 1.0, got {mult}"

    def test_negative_sharpe_penalty(self, adaptive_weights):
        """Sharpe < 0 → multiplier = neg_sharpe_penalty (0.25)."""
        attrib = {"total_readings": 30, "sharpe_contribution": -0.5}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 0.25, f"Expected 0.25 for negative Sharpe, got {mult}"

    def test_very_negative_sharpe_penalty(self, adaptive_weights):
        """Very negative Sharpe → still 0.25 (no worse than penalty)."""
        attrib = {"total_readings": 30, "sharpe_contribution": -3.0}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 0.25, f"Expected 0.25 for very negative Sharpe, got {mult}"

    def test_zero_sharpe(self, adaptive_weights):
        """Zero Sharpe → no_data_multiplier (1.0)."""
        attrib = {"total_readings": 30, "sharpe_contribution": 0.0}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 1.0, f"Expected 1.0 for zero Sharpe, got {mult}"

    def test_zero_readings(self, adaptive_weights):
        """Zero readings → no_data_multiplier (1.0)."""
        attrib = {"total_readings": 0, "sharpe_contribution": 0.8}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 1.0, f"Expected 1.0 for no readings, got {mult}"

    def test_nan_sharpe(self, adaptive_weights):
        """NaN Sharpe → no_data_multiplier (1.0)."""
        attrib = {"total_readings": 30, "sharpe_contribution": float('nan')}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 1.0, f"Expected 1.0 for NaN Sharpe, got {mult}"

    def test_none_sharpe(self, adaptive_weights):
        """None Sharpe → no_data_multiplier (1.0)."""
        attrib = {"total_readings": 30, "sharpe_contribution": None}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 1.0, f"Expected 1.0 for None Sharpe, got {mult}"

    def test_very_high_sharpe_capped(self, adaptive_weights):
        """Very high Sharpe → capped at max_multiplier (2.0)."""
        attrib = {"total_readings": 30, "sharpe_contribution": 5.0}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 2.0, f"Expected 2.0 (capped) for high Sharpe, got {mult}"

    def test_missing_fields(self, adaptive_weights):
        """Missing fields in attribution dict → safe default."""
        attrib = {}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        assert mult == 1.0, f"Expected 1.0 for empty attrib, got {mult}"

    def test_few_readings_scales_multiplier(self, adaptive_weights):
        """Few readings (< min) should scale multiplier toward 1.0."""
        # With positive Sharpe but only 5 readings (min is 20)
        attrib = {"total_readings": 5, "sharpe_contribution": 1.0}
        mult = adaptive_weights._compute_multiplier("test_source", attrib)
        # Should be closer to 1.0 than raw multiplier
        raw = adaptive_weights._raw_multiplier(1.0)
        assert abs(mult - 1.0) < abs(raw - 1.0), \
            f"Data scarcity should pull mult toward 1.0: raw={raw:.2f}, adjusted={mult:.2f}"
        assert 1.0 < mult < 2.0, f"Expected between 1.0 and 2.0, got {mult}"


# ── Weight Update Tests ────────────────────────────────────────────────────


class TestWeightUpdate:
    """Verify full weight update from attribution data."""

    def test_good_attribution_produces_adjusted_weights(
        self, adaptive_weights, sample_attribution_good
    ):
        """Good attribution data → all sources get adjusted weights."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        assert len(adapted) > 0
        # Sum should be ≈ 1.0
        total = sum(adapted.values())
        assert abs(total - 1.0) < 0.01, f"Expected sum ≈ 1.0, got {total}"

    def test_no_sources_falls_back_to_base(self, adaptive_weights):
        """Empty sources dict → base weights used."""
        empty_attribution = {
            "timestamp": datetime.now().isoformat(),
            "sources": {},
        }
        adapted = adaptive_weights.update_weights(empty_attribution, "normal")
        assert adapted == adaptive_weights.base_weights

    def test_mixed_performance_adjusts_weights(
        self, adaptive_weights, sample_attribution_mixed
    ):
        """Mixed attribution → boost strong sources, reduce weak ones."""
        adapted = adaptive_weights.update_weights(sample_attribution_mixed, "normal")
        # Sum should be ≈ 1.0
        total = sum(adapted.values())
        assert abs(total - 1.0) < 0.01, f"Expected sum ≈ 1.0, got {total}"

        # CTA Trend (Sharpe 1.2) should have higher weight than multi_speed_momentum (Sharpe -0.3)
        cta_weight = adapted.get("cta_trend", 0)
        msm_weight = adapted.get("multi_speed_momentum", 0)
        assert cta_weight > msm_weight, \
            f"CTA ({cta_weight:.4f}) should > MSM ({msm_weight:.4f})"

    def test_min_weight_enforced(self, adaptive_weights, sample_attribution_good):
        """Positive-baseline sources should not fall below min_weight.

        Zero-baseline arms are hard-excluded (see test_zero_baseline_arm_stays_zero).
        """
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        for source, weight in adapted.items():
            base = float(adaptive_weights.base_weights.get(source, 0) or 0)
            if base <= 0:
                assert weight == 0.0, f"zero-baseline {source} resurrected to {weight}"
                continue
            assert weight >= adaptive_weights.config["min_weight"], \
                f"{source} weight {weight:.4f} < min {adaptive_weights.config['min_weight']}"

    def test_max_weight_enforced(self, adaptive_weights, sample_attribution_good):
        """No source should exceed max_weight."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        for source, weight in adapted.items():
            assert weight <= adaptive_weights.config["max_weight"], \
                f"{source} weight {weight:.4f} > max {adaptive_weights.config['max_weight']}"

    def test_multipliers_recorded(self, adaptive_weights, sample_attribution_good):
        """Multipliers should be recorded for each source."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        multipliers = adaptive_weights.get_multipliers()
        assert len(multipliers) > 0
        for source in adapted:
            assert source in multipliers, f"Missing multiplier for {source}"

    def test_persistent_state(self, adaptive_weights, sample_attribution_good):
        """State should be saved and loadable."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")

        # Create new instance and verify load
        new_adaptive = AdaptiveEnsembleWeights(base_weights=adaptive_weights.base_weights)
        loaded = new_adaptive._load_state()
        assert loaded, "State should load successfully"
        assert len(new_adaptive.adjusted_weights) > 0


# ── Reset Tests ────────────────────────────────────────────────────────────


class TestReset:
    """Verify reset restores baseline weights."""

    def test_reset_returns_base_weights(self, adaptive_weights, sample_attribution_good):
        """After reset, weights should equal base weights."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        reset_weights = adaptive_weights.reset_to_baseline()
        for source in reset_weights:
            assert reset_weights[source] == adaptive_weights.base_weights.get(source, 0), \
                f"{source}: {reset_weights[source]} != base {adaptive_weights.base_weights.get(source, 0)}"

    def test_multipliers_reset_to_one(self, adaptive_weights, sample_attribution_good):
        """After reset, all multipliers should be 1.0."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        adaptive_weights.reset_to_baseline()
        multipliers = adaptive_weights.get_multipliers()
        for source, mult in multipliers.items():
            assert mult == 1.0, f"{source} multiplier {mult} != 1.0"


# ── Integration Tests ──────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests with realistic data flow."""

    def test_end_to_end_attribution_flow(
        self, tmp_state_dir, sample_base_weights, sample_attribution_good
    ):
        """Full flow: attribution → weight update → normalized output."""
        weights = AdaptiveEnsembleWeights(base_weights=sample_base_weights)
        adapted = weights.update_weights(sample_attribution_good, "normal")

        # Check all expected properties
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        # Positive-baseline arms respect min floor; zero-baseline hard-exclude stays 0
        for src, w in adapted.items():
            base = float(sample_base_weights.get(src, 0) or 0)
            if base <= 0:
                assert w == 0.0, f"zero-baseline {src} resurrected to {w}"
            else:
                assert w >= 0.01, f"{src} weight {w} < min floor"
        assert all(w <= 0.40 for w in adapted.values())

        # Check that top performer got highest weight
        assert adapted.get("cta_trend", 0) > adapted.get("hmm_regime", 0), \
            "CTA trend (0.95 Sharpe) should outweigh HMM (0.2 Sharpe)"

    def test_attribution_with_extra_sources(
        self, tmp_state_dir, sample_base_weights
    ):
        """Attribution-only ghosts must not enter live adaptive mass (Batch BJ)."""
        attribution = {
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "tsfm_momentum": {"total_readings": 30, "sharpe_contribution": 0.8, "avg_weight": 0.05},
                "multi_speed_momentum": {"total_readings": 30, "sharpe_contribution": 0.6, "avg_weight": 0.05},
                "brand_new_source_x": {"total_readings": 40, "sharpe_contribution": 1.5, "avg_weight": 0.01},
            },
        }
        weights = AdaptiveEnsembleWeights(base_weights=sample_base_weights)
        adapted = weights.update_weights(attribution, "normal")
        assert "brand_new_source_x" not in adapted, "Ghost must not dilute live weights"
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        assert "tsfm_momentum" in adapted

    def test_stale_attribution_handling(self, tmp_state_dir, sample_base_weights):
        """
        Fallback in EnsembleVoter: when attribution is stale (>7 days),
        the voter should not activate adaptive weights (handled in voter code).
        The weights module itself doesn't check staleness - it works with any data.
        """
        old_date = (datetime.now() - timedelta(days=14)).isoformat()
        attribution = {
            "timestamp": old_date,
            "sources": {
                "tsfm_momentum": {"total_readings": 30, "sharpe_contribution": 0.8},
            },
        }
        weights = AdaptiveEnsembleWeights(base_weights=sample_base_weights)
        # Module still processes the data (staleness check is in voter)
        adapted = weights.update_weights(attribution, "normal")
        assert len(adapted) > 0


# ── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_base_weights(self, tmp_state_dir):
        """Empty base weights should not crash (isolated state path)."""
        weights = AdaptiveEnsembleWeights(base_weights={})
        adapted = weights.update_weights({"sources": {}, "timestamp": "now"}, "normal")
        assert adapted == {}

    def test_single_source_only(self):
        """Single source with perfect Sharpe should get full weight."""
        weights = AdaptiveEnsembleWeights(
            base_weights={"only_source": 1.0},
        )
        attribution = {
            "timestamp": "now",
            "sources": {
                "only_source": {"total_readings": 100, "sharpe_contribution": 2.0},
            },
        }
        adapted = weights.update_weights(attribution, "normal")
        assert abs(adapted.get("only_source", 0) - 1.0) < 0.01

    def test_all_sources_negative_sharpe(self):
        """All sources negative → all penalized, but min constraint applies."""
        weights = AdaptiveEnsembleWeights(
            base_weights={"src_a": 0.6, "src_b": 0.4},
        )
        attribution = {
            "timestamp": "now",
            "sources": {
                "src_a": {"total_readings": 50, "sharpe_contribution": -0.5},
                "src_b": {"total_readings": 50, "sharpe_contribution": -1.0},
            },
        }
        adapted = weights.update_weights(attribution, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        for w in adapted.values():
            assert w >= 0.01


# ── CLI Tests ──────────────────────────────────────────────────────────────


class TestCLI:
    """Basic CLI smoke tests."""

    def test_cli_update_no_attribution(self, tmp_state_dir, monkeypatch):
        """CLI update with no attribution files should exit with error."""
        monkeypatch.setattr(
            "sys.argv", ["adaptive_ensemble_weights", "update", "--regime", "normal"]
        )
        with pytest.raises(SystemExit) as exc:
            from src.strategy.adaptive_ensemble_weights import main
            main()
        # Should return error code 1
        assert exc.value.code != 0

    def test_cli_show_no_state(self, tmp_state_dir, monkeypatch):
        """CLI show with no state file should print error."""
        monkeypatch.setattr(
            "sys.argv", ["adaptive_ensemble_weights", "show"]
        )
        with pytest.raises(SystemExit) as exc:
            from src.strategy.adaptive_ensemble_weights import main
            main()
        assert exc.value.code != 0


# ── State Persistence Tests ────────────────────────────────────────────────


class TestStatePersistence:
    """Verify state save/load works correctly."""

    def test_state_file_created(self, adaptive_weights, sample_attribution_good):
        """After update, state file should exist."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        assert adaptive_weights.state_file.exists()

    def test_load_restores_weights(self, adaptive_weights, sample_attribution_good):
        """Loading should restore previously computed weights."""
        original = adaptive_weights.update_weights(sample_attribution_good, "normal")
        adaptive_weights._load_state()
        for source in original:
            loaded = adaptive_weights.adjusted_weights.get(source)
            expected = float(original[source])
            assert loaded is not None
            assert abs(loaded - expected) <= 0.01, f"weight mismatch for {source}: {loaded} != {expected}"

    def test_load_nonexistent_file(self, adaptive_weights):
        """Loading non-existent file should not crash."""
        if adaptive_weights.state_file.exists():
            os.remove(adaptive_weights.state_file)
        result = adaptive_weights._load_state()
        assert not result

    def test_corrupted_state_file(self, adaptive_weights):
        """Corrupted JSON should not crash."""
        adaptive_weights.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(adaptive_weights.state_file, "w") as f:
            f.write("not valid json {{{")
        result = adaptive_weights._load_state()
        assert not result


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestRawMultiplier:
    """Test _raw_multiplier edge cases."""

    def test_negative_sharpe_gives_penalty(self, adaptive_weights):
        """Negative Sharpe should return neg_sharpe_penalty."""
        result = adaptive_weights._raw_multiplier(-0.5)
        assert result == adaptive_weights.config["neg_sharpe_penalty"]

    def test_zero_sharpe_gives_no_data(self, adaptive_weights):
        """Zero Sharpe should return no_data_multiplier."""
        result = adaptive_weights._raw_multiplier(0.0)
        assert result == adaptive_weights.config["no_data_multiplier"]

    def test_positive_sharpe_scales(self, adaptive_weights):
        """Positive Sharpe should scale relative to baseline."""
        # Sharpe = baseline_sharpe → multiplier = 1.0
        result = adaptive_weights._raw_multiplier(adaptive_weights.config["baseline_sharpe"])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_high_sharpe_capped(self, adaptive_weights):
        """Very high Sharpe should be capped at max_multiplier."""
        result = adaptive_weights._raw_multiplier(10.0)
        assert result == adaptive_weights.config["max_multiplier"]

    def test_very_small_positive_sharpe(self, adaptive_weights):
        """Very small positive Sharpe should give low multiplier."""
        result = adaptive_weights._raw_multiplier(0.001)
        # 0.001 / 0.5 = 0.002, but clipped to min_multiplier
        assert result == adaptive_weights.config["min_multiplier"]


class TestComputeMultiplier:
    """Test _compute_multiplier edge cases."""

    def test_none_sharpe_returns_no_data(self, adaptive_weights):
        """None Sharpe should return no_data_multiplier."""
        attr = {"sharpe_contribution": None, "total_readings": 100}
        result = adaptive_weights._compute_multiplier("test_source", attr)
        assert result == adaptive_weights.config["no_data_multiplier"]

    def test_nan_sharpe_returns_no_data(self, adaptive_weights):
        """NaN Sharpe should return no_data_multiplier."""
        attr = {"sharpe_contribution": float('nan'), "total_readings": 100}
        result = adaptive_weights._compute_multiplier("test_source", attr)
        assert result == adaptive_weights.config["no_data_multiplier"]

    def test_zero_readings_returns_no_data(self, adaptive_weights):
        """Zero total_readings should return no_data_multiplier."""
        attr = {"sharpe_contribution": 0.5, "total_readings": 0}
        result = adaptive_weights._compute_multiplier("test_source", attr)
        assert result == adaptive_weights.config["no_data_multiplier"]

    def test_scarce_data_scales_toward_one(self, adaptive_weights):
        """Scarce data should scale multiplier toward 1.0."""
        # With 5 readings (min_readings=20), data_ratio=0.25
        attr = {"sharpe_contribution": 1.0, "total_readings": 5}
        result = adaptive_weights._compute_multiplier("test_source", attr)
        # raw_mult for sharpe=1.0 = 1.0/0.5 = 2.0, but clipped to max=2.0
        # adjusted = 1.0 + (2.0 - 1.0) * 0.25 = 1.25
        assert 1.0 <= result <= 2.0

    def test_sufficient_data_uses_raw(self, adaptive_weights):
        """Enough readings should use raw multiplier."""
        attr = {"sharpe_contribution": 1.0, "total_readings": 100}
        result = adaptive_weights._compute_multiplier("test_source", attr)
        raw = adaptive_weights._raw_multiplier(1.0)
        assert result == raw


class TestResetToBaseline:
    """Test reset_to_baseline behavior."""

    def test_reset_returns_baseline(self, adaptive_weights, sample_attribution_good):
        """After update and reset, weights should match baseline."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        reset = adaptive_weights.reset_to_baseline()
        for source in adaptive_weights.base_weights:
            assert abs(reset[source] - adaptive_weights.base_weights[source]) < 0.001

    def test_reset_multipliers_to_one(self, adaptive_weights, sample_attribution_good):
        """After reset, all multipliers should be 1.0."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        adaptive_weights.reset_to_baseline()
        for mult in adaptive_weights.multipliers.values():
            assert mult == 1.0

    def test_reset_persists(self, adaptive_weights, sample_attribution_good):
        """Reset should persist to state file."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        adaptive_weights.reset_to_baseline()
        # Load into new instance
        new_weights = AdaptiveEnsembleWeights(
            base_weights=adaptive_weights.base_weights,
        )
        new_weights.state_file = adaptive_weights.state_file
        new_weights._load_state()
        for source in adaptive_weights.base_weights:
            assert abs(new_weights.adjusted_weights[source] - adaptive_weights.base_weights[source]) < 0.01


class TestGetMultipliers:
    """Test get_multipliers behavior."""

    def test_before_update_returns_empty_or_loaded(self, adaptive_weights):
        """Before update, multipliers should be empty or loaded from state."""
        result = adaptive_weights.get_multipliers()
        assert isinstance(result, dict)

    def test_after_update_returns_multipliers(self, adaptive_weights, sample_attribution_good):
        """After update, multipliers should have entries."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        mults = adaptive_weights.get_multipliers()
        assert len(mults) > 0


class TestGetStateDict:
    """Test get_state_dict for dashboard integration."""

    def test_state_dict_has_expected_keys(self, adaptive_weights, sample_attribution_good):
        """State dict should contain expected keys."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        state = adaptive_weights.get_state_dict()
        assert "timestamp" in state
        assert "regime" in state
        assert "adjusted_weights" in state
        assert "multipliers" in state
        assert "top_changes" in state
        assert "history_count" in state

    def test_state_dict_regime(self, adaptive_weights, sample_attribution_good):
        """State dict should reflect the current regime."""
        adaptive_weights.update_weights(sample_attribution_good, "crisis")
        state = adaptive_weights.get_state_dict()
        assert state["regime"] == "crisis"


class TestUpdateWeightsExtended:
    """Extended update_weights edge cases."""

    def test_regime_stored(self, adaptive_weights, sample_attribution_good):
        """update_weights should store the regime."""
        adaptive_weights.update_weights(sample_attribution_good, "high_vol")
        assert adaptive_weights.current_regime == "high_vol"

    def test_weights_sum_to_one(self, adaptive_weights, sample_attribution_good):
        """Adjusted weights should sum to approximately 1.0."""
        result = adaptive_weights.update_weights(sample_attribution_good, "normal")
        total = sum(result.values())
        assert abs(total - 1.0) < 0.05

    def test_history_populated(self, adaptive_weights, sample_attribution_good):
        """History should have entries after update."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        assert len(adaptive_weights.history) > 0

    def test_custom_config(self):
        """Custom config should override defaults."""
        custom_config = {
            "baseline_sharpe": 1.0,
            "min_multiplier": 0.5,
            "max_multiplier": 3.0,
        }
        aew = AdaptiveEnsembleWeights(config=custom_config)
        assert aew.config["baseline_sharpe"] == 1.0
        assert aew.config["min_multiplier"] == 0.5
        assert aew.config["max_multiplier"] == 3.0
        # Defaults should still be present
        assert "neg_sharpe_penalty" in aew.config

    def test_custom_base_weights(self):
        """Custom base weights should be used instead of empty dict."""
        custom_base = {"source_a": 0.6, "source_b": 0.4}
        aew = AdaptiveEnsembleWeights(base_weights=custom_base)
        assert aew.base_weights == custom_base


class TestWeightAdjustmentDataclass:
    """Test WeightAdjustment dataclass."""

    def test_creation(self):
        wa = WeightAdjustment(
            timestamp="2026-05-24T00:00:00",
            regime="normal",
            source="test_source",
            base_weight=0.3,
            multiplier=1.5,
            adjusted_weight=0.45,
            sharpe_contribution=0.75,
            total_readings=50,
        )
        assert wa.source == "test_source"
        assert wa.multiplier == 1.5

    def test_asdict(self):
        wa = WeightAdjustment(
            timestamp="2026-05-24", regime="crisis", source="s1",
            base_weight=0.2, multiplier=0.8, adjusted_weight=0.16,
            sharpe_contribution=-0.3, total_readings=10,
        )
        d = asdict(wa)
        assert d["regime"] == "crisis"
        assert d["source"] == "s1"


class TestAdaptiveWeightsStateDataclass:
    """Test AdaptiveWeightsState dataclass."""

    def test_creation(self):
        state = AdaptiveWeightsState(
            timestamp="2026-05-24",
            regime="normal",
            adjusted_weights={"s1": 0.5},
            multipliers={"s1": 1.0},
            history=[],
            baseline_weights={"s1": 0.5},
            config=DEFAULT_CONFIG,
        )
        assert state.regime == "normal"
        assert state.adjusted_weights["s1"] == 0.5


# ── Config Edge Cases ─────────────────────────────────────────────────────────


class TestConfigEdgeCases:
    """Configuration override and edge-case behavior."""

    def test_empty_config_falls_back_to_defaults(self):
        """Empty config dict should use all DEFAULT_CONFIG values."""
        aew = AdaptiveEnsembleWeights(config={})
        for key, value in DEFAULT_CONFIG.items():
            assert aew.config[key] == value, f"Mismatch for config key {key}"

    def test_partial_config_merges_with_defaults(self):
        """Partial override keeps unspecified defaults."""
        aew = AdaptiveEnsembleWeights(config={"baseline_sharpe": 2.0})
        assert aew.config["baseline_sharpe"] == 2.0
        assert aew.config["min_multiplier"] == DEFAULT_CONFIG["min_multiplier"]
        assert aew.config["max_multiplier"] == DEFAULT_CONFIG["max_multiplier"]
        assert aew.config["neg_sharpe_penalty"] == DEFAULT_CONFIG["neg_sharpe_penalty"]

    def test_clamp_with_inverted_bounds(self):
        """min_multiplier > max_multiplier: np.clip caps at max, does not crash."""
        aew = AdaptiveEnsembleWeights(config={
            "min_multiplier": 1.5,
            "max_multiplier": 0.5,
        })
        result = aew._raw_multiplier(2.0)
        assert isinstance(result, float)
        # np.clip with inverted bounds clips to the upper bound (max)
        assert result == 0.5


# ── Multiplier Edge Cases (precedence, boundaries) ──────────────────────────


class TestMultiplierEdgeCases:
    """Multiplier-computation precedence & combined edge cases."""

    def test_negative_sharpe_with_scarce_data(self, adaptive_weights):
        """Negative Sharpe + scarce data: penalty scaled toward 1.0."""
        attrib = {"total_readings": 5, "sharpe_contribution": -0.5}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        # raw = 0.25 (penalty), data_ratio = 5/20 = 0.25
        # result = 1.0 + (0.25 - 1.0) * 0.25 = 0.8125
        assert 0.25 <= mult <= 1.0
        assert mult < 1.0

    def test_zero_readings_with_none_sharpe(self, adaptive_weights):
        """Zero readings check precedes None-Sharpe check."""
        attrib = {"total_readings": 0, "sharpe_contribution": None}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        assert mult == 1.0

    def test_zero_readings_with_nan_sharpe(self, adaptive_weights):
        """Zero readings check precedes NaN-Sharpe check."""
        attrib = {"total_readings": 0, "sharpe_contribution": float("nan")}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        assert mult == 1.0

    def test_exactly_min_readings_boundary(self, adaptive_weights):
        """Exactly min_readings_per_source: data_ratio=1.0 => raw multiplier."""
        min_r = adaptive_weights.config["min_readings_per_source"]
        attrib = {"total_readings": min_r, "sharpe_contribution": 0.8}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        raw = adaptive_weights._raw_multiplier(0.8)
        assert mult == raw

    def test_one_reading_below_min(self, adaptive_weights):
        """One reading: heavily pulled toward 1.0."""
        attrib = {"total_readings": 1, "sharpe_contribution": 2.0}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        assert mult == pytest.approx(1.05, abs=0.01)

    def test_none_sharpe_with_plenty_readings(self, adaptive_weights):
        """None Sharpe with sufficient readings: None check triggers first."""
        attrib = {"total_readings": 100, "sharpe_contribution": None}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        assert mult == 1.0

    def test_nan_sharpe_with_plenty_readings(self, adaptive_weights):
        """NaN Sharpe with sufficient readings: NaN check triggers first."""
        attrib = {"total_readings": 100, "sharpe_contribution": float("nan")}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        assert mult == 1.0

    def test_scarce_data_on_negative_sharpe_boundary(self, adaptive_weights):
        """One reading below min, negative Sharpe: penalty scaled up toward 1."""
        attrib = {"total_readings": 1, "sharpe_contribution": -0.5}
        mult = adaptive_weights._compute_multiplier("test", attrib)
        # raw = 0.25, data_ratio = 0.05 => 1.0 + (0.25 - 1.0) * 0.05 = 0.9625
        assert 0.25 <= mult < 1.0


# ── Weight Update Edge Cases ──────────────────────────────────────────────────


class TestWeightUpdateEdgeCases:
    """Edge cases in update_weights beyond normal attribution data."""

    def test_missing_sources_key(self, adaptive_weights):
        """Attribution without 'sources' key should fall back to base weights."""
        attribution = {"timestamp": "now"}
        adapted = adaptive_weights.update_weights(attribution, "normal")
        assert adapted == adaptive_weights.base_weights
        # multipliers should all be 1.0
        for mult in adaptive_weights.multipliers.values():
            assert mult == 1.0

    def test_sources_is_none(self, adaptive_weights):
        """sources=None should be treated same as empty dict."""
        attribution = {"timestamp": "now", "sources": None}
        adapted = adaptive_weights.update_weights(attribution, "normal")
        # .get("sources", {}) on None returns None, which is falsy
        # Actually: if not sources -> True since None is falsy -> base weights
        assert adapted == adaptive_weights.base_weights

    def test_all_sources_identical_sharpe(self):
        """All sources with same Sharpe => weights proportional to base (no min clamp distortion)."""
        # Use high base weights and no min/max clamping to verify proportionality
        weights = AdaptiveEnsembleWeights(
            base_weights={"src_a": 0.5, "src_b": 0.3, "src_c": 0.2},
            config={"min_weight": 0.001, "max_weight": 0.999, "min_multiplier": 0.1, "max_multiplier": 10.0},
        )
        attr = {
            "timestamp": "now",
            "sources": {
                "src_a": {"total_readings": 40, "sharpe_contribution": 0.6},
                "src_b": {"total_readings": 40, "sharpe_contribution": 0.6},
                "src_c": {"total_readings": 40, "sharpe_contribution": 0.6},
            },
        }
        adapted = weights.update_weights(attr, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        # With identical multipliers and no clamping, weights stay proportional to base
        # src_a / src_b should be ~ 0.5/0.3 = 1.667
        ratio_ab = adapted["src_a"] / adapted["src_b"]
        assert abs(ratio_ab - 1.667) < 0.1

    def test_attribution_only_extra_sources(self, adaptive_weights):
        """Ghost-only attribution with empty baseline → empty adapted (Batch BJ)."""
        attribution = {
            "timestamp": "now",
            "sources": {
                "extra_one": {
                    "total_readings": 40,
                    "sharpe_contribution": 1.2,
                    "avg_weight": 0.02,
                },
                "extra_two": {
                    "total_readings": 40,
                    "sharpe_contribution": 0.8,
                    "avg_weight": 0.03,
                },
            },
        }
        weights = AdaptiveEnsembleWeights(base_weights={})  # empty baseline
        adapted = weights.update_weights(attribution, "normal")
        assert adapted == {}
        assert "extra_one" not in adapted
        assert "extra_two" not in adapted

    def test_single_source_zero_sharpe(self):
        """Single source with zero Sharpe yields no_data_multiplier."""
        weights = AdaptiveEnsembleWeights(base_weights={"src": 1.0})
        attr = {
            "timestamp": "now",
            "sources": {"src": {"total_readings": 30, "sharpe_contribution": 0.0}},
        }
        adapted = weights.update_weights(attr, "normal")
        assert abs(adapted.get("src", 0) - 1.0) < 0.01
        assert weights.multipliers["src"] == 1.0

    def test_single_source_negative_sharpe(self):
        """Single source with negative Sharpe gets penalty but min weight."""
        weights = AdaptiveEnsembleWeights(base_weights={"src": 1.0})
        attr = {
            "timestamp": "now",
            "sources": {"src": {"total_readings": 30, "sharpe_contribution": -0.5}},
        }
        adapted = weights.update_weights(attr, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        assert adapted["src"] >= weights.config["min_weight"]

    def test_attribution_source_missing_sharpe_field(self, adaptive_weights):
        """Attribution source dict without sharpe_contribution key."""
        attribution = {
            "timestamp": "now",
            "sources": {
                "tsfm_momentum": {"total_readings": 30},  # no sharpe_contribution
            },
        }
        adapted = adaptive_weights.update_weights(attribution, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        # missing sharpe defaults to 0 in .get(), which returns no_data_multiplier=1.0
        assert adaptive_weights.multipliers.get("tsfm_momentum") == 1.0

    def test_attribution_source_missing_total_readings(self, adaptive_weights):
        """Attribution source dict without total_readings key."""
        attribution = {
            "timestamp": "now",
            "sources": {
                "tsfm_momentum": {"sharpe_contribution": 0.8},  # no total_readings
            },
        }
        # default total_readings = 0 => no_data_multiplier
        adapted = adaptive_weights.update_weights(attribution, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        assert adaptive_weights.multipliers.get("tsfm_momentum") == 1.0

    def test_different_regime_preserved(self, adaptive_weights, sample_attribution_good):
        """Regime passed to update_weights is stored."""
        for regime in ("normal", "crisis", "high_vol", "low_growth"):
            aew = AdaptiveEnsembleWeights(base_weights=adaptive_weights.base_weights)
            aew.state_file = adaptive_weights.state_file
            aew.update_weights(sample_attribution_good, regime)
            assert aew.current_regime == regime


# ── Normalization Edge Cases ──────────────────────────────────────────────────


class TestNormalizationEdgeCases:
    """Normalization loop and edge-case handling."""

    def test_all_sources_at_max_weight(self):
        """Many sources all clipped to max_weight re-normalize correctly."""
        weights = AdaptiveEnsembleWeights(
            base_weights={f"s{i}": 0.5 for i in range(5)},
            config={"max_weight": 0.30, "min_weight": 0.01},
        )
        attr = {
            "timestamp": "now",
            "sources": {f"s{i}": {"total_readings": 50, "sharpe_contribution": 3.0}
                        for i in range(5)},
        }
        adapted = weights.update_weights(attr, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        for w in adapted.values():
            assert w <= weights.config["max_weight"] + 1e-6

    def test_single_source_min_weight(self):
        """Single source with very negative Sharpe: floor at min_weight."""
        weights = AdaptiveEnsembleWeights(
            base_weights={"src": 1.0},
            config={"min_weight": 0.02},
        )
        attr = {
            "timestamp": "now",
            "sources": {"src": {"total_readings": 50, "sharpe_contribution": -5.0}},
        }
        adapted = weights.update_weights(attr, "normal")
        assert adapted["src"] >= weights.config["min_weight"]
        assert abs(sum(adapted.values()) - 1.0) < 0.01

    def test_many_iterations_to_converge(self):
        """Multiple sources bouncing below min_weight after normalization."""
        config = {
            "min_weight": 0.10,
            "max_weight": 0.60,
            "neg_sharpe_penalty": 0.25,
        }
        weights = AdaptiveEnsembleWeights(
            base_weights={"good": 0.7, "bad1": 0.1, "bad2": 0.1, "bad3": 0.1},
            config=config,
        )
        attr = {
            "timestamp": "now",
            "sources": {
                "good": {"total_readings": 50, "sharpe_contribution": 1.5},
                "bad1": {"total_readings": 50, "sharpe_contribution": -1.0},
                "bad2": {"total_readings": 50, "sharpe_contribution": -2.0},
                "bad3": {"total_readings": 50, "sharpe_contribution": -3.0},
            },
        }
        adapted = weights.update_weights(attr, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        for w in adapted.values():
            assert w >= config["min_weight"] - 1e-6

    def test_empty_attribution_returns_empty(self):
        """Empty base + no sources => empty result."""
        weights = AdaptiveEnsembleWeights(base_weights={})
        adapted = weights.update_weights({"sources": {}}, "normal")
        assert adapted == {}

    def test_very_many_sources(self):
        """Many sources (30+) should normalize without numerical issues."""
        base = {f"s{i}": 1.0 / 30 for i in range(30)}
        sources = {
            f"s{i}": {"total_readings": 30, "sharpe_contribution": 0.3 + (i % 10) * 0.2}
            for i in range(30)
        }
        weights = AdaptiveEnsembleWeights(base_weights=base)
        adapted = weights.update_weights({"sources": sources, "timestamp": "now"}, "normal")
        assert abs(sum(adapted.values()) - 1.0) < 0.01


# ── State Persistence Edge Cases ──────────────────────────────────────────────


class TestStatePersistenceEdgeCases:
    """Save/load edge cases beyond basic round-trip."""

    def test_history_trimmed_to_fifty(self, adaptive_weights, sample_attribution_good):
        """History list is trimmed to last 50 entries on save."""
        # Manually add 60 history entries
        for i in range(60):
            adaptive_weights.history.append(WeightAdjustment(
                timestamp=f"2026-01-{i+1:02d}T00:00:00",
                regime="normal", source="s1",
                base_weight=0.1, multiplier=1.0, adjusted_weight=0.1,
                sharpe_contribution=0.5, total_readings=30,
            ))
        adaptive_weights._save_state()
        # Reload
        adaptive_weights._load_state()
        assert len(adaptive_weights.history) <= 50
        # The oldest entry should be gone
        timestamps = [h.timestamp for h in adaptive_weights.history]
        assert "2026-01-01T00:00:00" not in timestamps

    def test_save_load_io_error_does_not_crash(self, adaptive_weights):
        """_save_state catches IOError and logs warning."""
        from unittest.mock import patch
        with patch("builtins.open", side_effect=IOError("Disk full")):
            adaptive_weights.adjusted_weights = {"s1": 0.6}
            adaptive_weights._save_state()  # should not raise

    def test_save_load_json_decode_error(self, adaptive_weights):
        """Corrupted state file should return False from _load_state."""
        adaptive_weights.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(adaptive_weights.state_file, "w") as f:
            f.write("{not json}")
        assert not adaptive_weights._load_state()

    def test_save_load_empty_baseline(self, tmp_state_dir):
        """Save/load with empty baseline weights (isolated state path)."""
        weights = AdaptiveEnsembleWeights(base_weights={})
        weights.adjusted_weights = {}
        weights.multipliers = {}
        # force_empty allows intentional empty baseline persistence in tests
        weights._save_state(force_empty=True)
        loaded = weights._load_state()
        assert loaded
        assert weights.base_weights == {}

    def test_load_without_prior_save(self, adaptive_weights):
        """State file does not exist => _load_state returns False."""
        if adaptive_weights.state_file.exists():
            adaptive_weights.state_file.unlink()
        assert not adaptive_weights._load_state()
        # adjusted_weights remain empty
        assert adaptive_weights.adjusted_weights == {}


# ── GetStateDict Edge Cases ───────────────────────────────────────────────────


class TestGetStateDictEdgeCases:
    """get_state_dict before/after updates and with empty state."""

    def test_before_any_update(self, adaptive_weights):
        """get_state_dict before update: available=False."""
        state = adaptive_weights.get_state_dict()
        assert state["available"] is False
        assert state["num_adjusted_sources"] == 0
        assert state["top_changes"] == []

    def test_after_reset(self, adaptive_weights, sample_attribution_good):
        """get_state_dict after reset: top_changes should be empty/small."""
        adaptive_weights.update_weights(sample_attribution_good, "normal")
        adaptive_weights.reset_to_baseline()
        state = adaptive_weights.get_state_dict()
        assert state["available"] is True

    def test_state_dict_without_baseline(self):
        """get_state_dict without baseline: no top_changes."""
        weights = AdaptiveEnsembleWeights(base_weights={})
        weights.adjusted_weights = {"s1": 0.5}
        weights.multipliers = {"s1": 1.0}
        state = weights.get_state_dict()
        assert state["num_adjusted_sources"] == 1
        assert state["top_changes"] == []  # no baseline to compare against


# ── CLI Edge Cases ────────────────────────────────────────────────────────────


class TestCLIEdgeCases:
    """Additional CLI command test coverage."""

    def test_cli_reset_no_state(self, tmp_state_dir, monkeypatch):
        """CLI reset without any prior update should not crash."""
        monkeypatch.setattr(
            "sys.argv", ["adaptive_ensemble_weights", "reset"],
        )
        from src.strategy.adaptive_ensemble_weights import main
        # reset calls _save_state which creates parent dirs
        rc = main()
        assert rc == 0

    def test_cli_update_with_attribution_file_direct(self, tmp_state_dir, monkeypatch, sample_attribution_good):
        """CLI update flow when attribution exists in default location."""
        # Write attribution to ATTRIBUTION_DIR with expected naming convention
        attr_dir = Path(tmp_state_dir) / "attribution"
        attr_dir.mkdir(parents=True, exist_ok=True)
        attr_file = attr_dir / "attribution_20260524.json"
        with open(attr_file, "w") as f:
            json.dump(sample_attribution_good, f)

        monkeypatch.setattr(
            "sys.argv", ["adaptive_ensemble_weights", "update", "--regime", "normal"],
        )
        from unittest.mock import patch
        with patch("src.strategy.adaptive_ensemble_weights._get_base_weights_from_voter",
                   return_value={s: 0.05 for s in sample_attribution_good["sources"]}):
            from src.strategy.adaptive_ensemble_weights import main
            rc = main()
        assert rc == 0

    def test_cli_unknown_command(self, tmp_state_dir, monkeypatch):
        """CLI with unknown command prints help and exits with error code 2."""
        monkeypatch.setattr(
            "sys.argv", ["adaptive_ensemble_weights", "unknown_command"],
        )
        with pytest.raises(SystemExit) as exc:
            from src.strategy.adaptive_ensemble_weights import main
            main()
        assert exc.value.code == 2


# ── Dataclass Field Validation ────────────────────────────────────────────────


class TestWeightAdjustmentDataclassExtended:
    """Extended dataclass validation for WeightAdjustment."""

    def test_negative_base_weight(self):
        """WeightAdjustment can hold a negative base_weight."""
        wa = WeightAdjustment(
            timestamp="now", regime="normal", source="s1",
            base_weight=-0.1, multiplier=1.0, adjusted_weight=-0.1,
            sharpe_contribution=-0.5, total_readings=30,
        )
        assert wa.base_weight == -0.1

    def test_zero_total_readings(self):
        """WeightAdjustment with zero total_readings is valid."""
        wa = WeightAdjustment(
            timestamp="now", regime="normal", source="s1",
            base_weight=0.1, multiplier=1.0, adjusted_weight=0.1,
            sharpe_contribution=0.0, total_readings=0,
        )
        assert wa.total_readings == 0

    def test_empty_source_string(self):
        """WeightAdjustment with empty source string."""
        wa = WeightAdjustment(
            timestamp="now", regime="normal", source="",
            base_weight=0.0, multiplier=1.0, adjusted_weight=0.0,
            sharpe_contribution=0.0, total_readings=0,
        )
        assert wa.source == ""

    def test_large_float_values(self):
        """Very large multiplier should be storable."""
        wa = WeightAdjustment(
            timestamp="now", regime="normal", source="s1",
            base_weight=1e6, multiplier=1e6, adjusted_weight=1e12,
            sharpe_contribution=1e6, total_readings=1000000,
        )
        assert wa.base_weight == 1e6
        assert wa.multiplier == 1e6


class TestAdaptiveWeightsStateDataclassExtended:
    """Extended dataclass validation for AdaptiveWeightsState."""

    def test_empty_dicts(self):
        """State with all empty dicts."""
        state = AdaptiveWeightsState(
            timestamp="now", regime="normal",
            adjusted_weights={}, multipliers={}, history=[],
            baseline_weights={}, config=DEFAULT_CONFIG,
        )
        assert state.adjusted_weights == {}
        assert state.history == []

    def test_minimal_config(self):
        """State with minimal config subset."""
        state = AdaptiveWeightsState(
            timestamp="now", regime="crisis",
            adjusted_weights={"s1": 0.4, "s2": 0.6},
            multipliers={"s1": 0.8, "s2": 1.2},
            history=[],
            baseline_weights={"s1": 0.5, "s2": 0.5},
            config={"window_days": 30},
        )
        assert state.config["window_days"] == 30

    def test_dict_conversion_roundtrip(self):
        """asdict() output should recreate the same object."""
        original = AdaptiveWeightsState(
            timestamp="2026-05-24", regime="high_vol",
            adjusted_weights={"a": 0.3, "b": 0.7},
            multipliers={"a": 0.6, "b": 1.4},
            history=[{"ts": "now", "src": "a"}],
            baseline_weights={"a": 0.5, "b": 0.5},
            config=DEFAULT_CONFIG,
        )
        d = asdict(original)
        restored = AdaptiveWeightsState(**d)
        assert restored.regime == original.regime
        assert restored.adjusted_weights == original.adjusted_weights
        assert restored.history == original.history


# ── Reset Edge Cases ──────────────────────────────────────────────────────────


class TestResetEdgeCases:
    """Additional reset behavior."""

    def test_reset_before_any_update(self, adaptive_weights):
        """Reset without prior update returns base weights."""
        result = adaptive_weights.reset_to_baseline()
        assert result == adaptive_weights.base_weights
        assert adaptive_weights.multipliers == {k: 1.0 for k in adaptive_weights.base_weights}

    def test_reset_empty_base_weights(self):
        """Reset with empty base weights returns empty dict."""
        weights = AdaptiveEnsembleWeights(base_weights={})
        result = weights.reset_to_baseline()
        assert result == {}
        assert weights.multipliers == {}

    def test_reset_saves_state(self, tmp_state_dir):
        """Reset persists state to file."""
        weights = AdaptiveEnsembleWeights(
            base_weights={"a": 0.7, "b": 0.3},
        )
        # Trigger save by reset
        weights.reset_to_baseline()
        assert weights.state_file.exists()
        with open(weights.state_file) as f:
            state = json.load(f)
        assert state["adjusted_weights"]["a"] == 0.7


# ── GetAdjustedWeights / GetMultipliers Edge Cases ────────────────────────────


class TestGetAccessorsEdgeCases:
    """get_adjusted_weights / get_multipliers edge cases."""

    def test_get_adjusted_weights_before_update_loads_state(self, adaptive_weights):
        """get_adjusted_weights with no in-memory data tries to load from file."""
        # No state file exists either
        result = adaptive_weights.get_adjusted_weights()
        # Should be empty dict since no state file
        assert result == {}

    def test_get_multipliers_before_update_loads_state(self, adaptive_weights):
        """get_multipliers with no in-memory data tries to load from file."""
        result = adaptive_weights.get_multipliers()
        assert result == {}


# ── Save State Parent Dir Creation ────────────────────────────────────────────


class TestSaveStateParentDir:
    """_save_state creates parent directories automatically."""

    def test_save_creates_parent_dirs(self, monkeypatch):
        """_save_state creates parent directory if it doesn't exist."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            deep_path = Path(tmp) / "a" / "b" / "c" / "state.json"
            monkeypatch.setattr(
                "src.strategy.adaptive_ensemble_weights.STATE_FILE", deep_path,
            )
            weights = AdaptiveEnsembleWeights(
                base_weights={"src": 1.0},
                config={"min_weight": 0.01},
            )
            weights.adjusted_weights = {"src": 1.0}
            weights._save_state()
            assert deep_path.exists()


# ── Zero-baseline hard exclude (Batch AQ) ──────────────────────────────────


class TestZeroBaselineSkipsMinWeightFloor:
    """Zero baseline weight must mean hard exclude — no min_weight resurrection."""

    def test_zero_baseline_arm_stays_zero(self, tmp_state_dir):
        """baseline multi_speed_momentum=0 yields adjusted msm=0 after update."""
        base = {
            "tsfm_momentum": 0.50,
            "cta_trend": 0.50,
            "multi_speed_momentum": 0.0,
        }
        attr = {
            "timestamp": "now",
            "sources": {
                "tsfm_momentum": {
                    "total_readings": 40,
                    "sharpe_contribution": 0.8,
                },
                "cta_trend": {
                    "total_readings": 40,
                    "sharpe_contribution": 0.6,
                },
                # Strong attribution must not resurrect a zero-baseline arm
                "multi_speed_momentum": {
                    "total_readings": 40,
                    "sharpe_contribution": 1.5,
                },
            },
        }
        weights = AdaptiveEnsembleWeights(
            base_weights=base,
            state_file=Path(tmp_state_dir) / "adaptive_zero.json",
        )
        adapted = weights.update_weights(attr, "normal")
        assert adapted.get("multi_speed_momentum", None) == 0.0, (
            f"zero-baseline msm resurrected to {adapted.get('multi_speed_momentum')}"
        )
        # Active mass still renormalizes
        assert abs(sum(adapted.values()) - 1.0) < 0.01
        assert adapted["tsfm_momentum"] > 0
        assert adapted["cta_trend"] > 0
        # Disclosure on instance + state payload
        assert "multi_speed_momentum" in weights.zero_baseline_exclusions
        state = weights.get_state_dict()
        assert "multi_speed_momentum" in state["zero_baseline_exclusions"]
        assert state["respect_zero_baseline"] is True

    def test_respect_zero_baseline_config_default_true(self):
        """respect_zero_baseline is True by default."""
        assert DEFAULT_CONFIG.get("respect_zero_baseline") is True
        w = AdaptiveEnsembleWeights(base_weights={"a": 1.0})
        assert w.config.get("respect_zero_baseline") is True

    def test_respect_zero_baseline_false_restores_floor(self, tmp_state_dir):
        """Opt-out: respect_zero_baseline=false re-applies min_weight to zero base."""
        base = {
            "tsfm_momentum": 0.50,
            "cta_trend": 0.50,
            "multi_speed_momentum": 0.0,
        }
        attr = {
            "timestamp": "now",
            "sources": {
                "tsfm_momentum": {
                    "total_readings": 40,
                    "sharpe_contribution": 0.8,
                },
                "cta_trend": {
                    "total_readings": 40,
                    "sharpe_contribution": 0.6,
                },
                "multi_speed_momentum": {
                    "total_readings": 40,
                    "sharpe_contribution": 1.5,
                },
            },
        }
        weights = AdaptiveEnsembleWeights(
            base_weights=base,
            config={"respect_zero_baseline": False, "min_weight": 0.01},
            state_file=Path(tmp_state_dir) / "adaptive_legacy.json",
        )
        adapted = weights.update_weights(attr, "normal")
        assert adapted["multi_speed_momentum"] >= weights.config["min_weight"] - 1e-9
        assert weights.zero_baseline_exclusions == []

    def test_sample_base_circuit_breaker_stays_zero(
        self, adaptive_weights, sample_attribution_good
    ):
        """circuit_breaker baseline 0.0 in sample fixtures stays zero after update."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        if "circuit_breaker" in adaptive_weights.base_weights:
            assert adapted.get("circuit_breaker", 0.0) == 0.0
            assert "circuit_breaker" in adaptive_weights.zero_baseline_exclusions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
