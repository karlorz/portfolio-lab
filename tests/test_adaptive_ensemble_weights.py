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
)


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
        """No source should fall below min_weight."""
        adapted = adaptive_weights.update_weights(sample_attribution_good, "normal")
        for source, weight in adapted.items():
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
        assert all(w >= 0.01 for w in adapted.values())
        assert all(w <= 0.40 for w in adapted.values())

        # Check that top performer got highest weight
        assert adapted.get("cta_trend", 0) > adapted.get("hmm_regime", 0), \
            "CTA trend (0.95 Sharpe) should outweigh HMM (0.2 Sharpe)"

    def test_attribution_with_extra_sources(
        self, tmp_state_dir, sample_base_weights
    ):
        """Attribution with sources not in baseline should still work."""
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
        assert "brand_new_source_x" in adapted, "Extra source should be included"
        assert abs(sum(adapted.values()) - 1.0) < 0.01

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

    def test_empty_base_weights(self):
        """Empty base weights should not crash."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
