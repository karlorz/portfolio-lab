"""
Tests for Unified Overlay Orchestrator (v4.90)
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from src.strategy.unified_orchestrator import (
    UnifiedOrchestrator,
    UnifiedRecommendation,
    OverlayContribution,
    OverlayStatus,
    get_unified_recommendation,
)


class TestOverlayStatus:
    """Test overlay status enum."""

    def test_status_values(self):
        assert OverlayStatus.ACTIVE.value == "active"
        assert OverlayStatus.SUPPRESSED.value == "suppressed"
        assert OverlayStatus.DISABLED.value == "disabled"


class TestOverlayContribution:
    """Test overlay contribution dataclass."""

    def test_serializable(self):
        contrib = OverlayContribution(
            name="collar", version="v4.60", status="active", weight=0.25,
            spy_delta=-0.02, gld_delta=0.01, tlt_delta=0.0,
            ief_delta=0.0, shy_delta=0.0, btc_delta=0.0, eth_delta=0.0,
            vol_impact=-0.005, sharpe_contribution=0.03,
            confidence=75.0, reason="Test",
        )
        d = contrib.to_dict()
        assert d["name"] == "collar"
        assert d["spy_delta"] == -0.02


class TestUnifiedOrchestrator:
    """Test unified orchestrator core functionality."""

    @pytest.fixture
    def orch(self):
        return UnifiedOrchestrator()

    def test_collect_contributions(self, orch):
        contributions = orch.collect_overlay_contributions()
        assert isinstance(contributions, list)
        assert len(contributions) >= 1  # At least calendar should always work
        for c in contributions:
            assert isinstance(c, OverlayContribution)
            assert c.name in ("collar", "crypto", "bond_duration", "calendar")

    def test_contributions_have_versions(self, orch):
        contributions = orch.collect_overlay_contributions()
        for c in contributions:
            assert c.version is not None
            assert len(c.version) > 0

    def test_recommend_generates(self, orch):
        rec = orch.recommend()
        assert isinstance(rec, UnifiedRecommendation)
        assert rec.timestamp is not None
        assert rec.spy > 0
        assert rec.gld > 0

    def test_recommend_weights_sum_to_one(self, orch):
        rec = orch.recommend()
        total = rec.spy + rec.gld + rec.tlt + rec.ief + rec.shy + rec.btc + rec.eth
        assert abs(total - 1.0) < 0.02

    def test_baseline_is_46_38_16(self, orch):
        rec = orch.recommend()
        assert rec.baseline_spy == 0.46
        assert rec.baseline_gld == 0.38
        assert rec.baseline_tlt == 0.16

    def test_recommendation_is_string(self, orch):
        rec = orch.recommend()
        assert isinstance(rec.recommendation, str)
        assert len(rec.recommendation) > 0

    def test_serializable(self, orch):
        rec = orch.recommend()
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert "spy" in d
        assert "contributions" in d
        for c in d["contributions"]:
            assert "name" in c

    def test_save_recommendation(self, orch, tmp_path):
        orch.STATE_FILE = tmp_path / "state.json"
        rec = orch.recommend()
        orch.save_recommendation(rec)

        out = tmp_path / "signals" / "unified_recommendation.json"
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)
        assert "spy" in loaded

    def test_crypto_within_bounds(self, orch):
        rec = orch.recommend()
        assert 0 <= rec.btc <= 0.03
        assert 0 <= rec.eth <= 0.02

    def test_spy_within_bounds(self, orch):
        rec = orch.recommend()
        assert 0.36 <= rec.spy <= 0.56

    def test_gld_within_bounds(self, orch):
        rec = orch.recommend()
        assert 0.28 <= rec.gld <= 0.48

    def test_convenience_function(self):
        rec = get_unified_recommendation()
        assert isinstance(rec, UnifiedRecommendation)


class TestConflictResolution:
    """Test conflict detection and resolution."""

    @pytest.fixture
    def orch(self):
        return UnifiedOrchestrator()

    def test_resolve_no_conflicts_with_valid_inputs(self, orch):
        contributions = [
            OverlayContribution("collar", "v4.60", "active", 0.25,
                                -0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                -0.005, 0.03, 75.0, "ok"),
            OverlayContribution("crypto", "v4.70", "active", 0.15,
                                0.0, -0.03, 0.0, 0.0, 0.0, 0.02, 0.01,
                                0.003, 0.02, 65.0, "ok"),
        ]
        weights, conflicts = orch.resolve_conflicts(contributions)
        assert isinstance(weights, dict)
        assert isinstance(conflicts, list)

    def test_resolve_with_disabled_leaves_baseline(self, orch):
        contributions = [
            OverlayContribution("test", "v1.0", "disabled", 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, "disabled"),
        ]
        weights, conflicts = orch.resolve_conflicts(contributions)
        for k, v in orch.BASELINE.items():
            assert abs(weights[k] - v) < 0.01

    def test_resolve_conflicting_spy_signals(self, orch):
        """One says buy SPY, other says sell."""
        contributions = [
            OverlayContribution("bull", "v1", "active", 0.3,
                                0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 80.0, "bull"),
            OverlayContribution("bear", "v1", "active", 0.3,
                                -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 80.0, "bear"),
        ]
        weights, conflicts = orch.resolve_conflicts(contributions)
        # Should detect the SPY conflict
        assert len(conflicts) >= 1
        # SPY should still be within bounds
        assert 0.36 <= weights["spy"] <= 0.56

    def test_empty_contributions_returns_baseline(self, orch):
        weights, conflicts = orch.resolve_conflicts([])
        for k, v in orch.BASELINE.items():
            assert abs(weights[k] - v) < 0.01


class TestEdgeCases:
    """Edge cases for orchestrator."""

    @pytest.fixture
    def orch(self):
        return UnifiedOrchestrator()

    def test_all_weights_non_negative(self, orch):
        rec = orch.recommend()
        assert rec.spy >= 0
        assert rec.gld >= 0
        assert rec.tlt >= 0
        assert rec.ief >= 0
        assert rec.shy >= 0
        assert rec.btc >= 0
        assert rec.eth >= 0

    def test_estimated_sharpe_reasonable(self, orch):
        rec = orch.recommend()
        assert 0.5 < rec.estimated_sharpe < 1.5

    def test_multiple_recommends_consistent(self, orch):
        rec1 = orch.recommend()
        rec2 = orch.recommend()
        assert rec1.spy > 0 and rec2.spy > 0
        assert abs(rec1.spy - rec2.spy) < 0.10  # Should be similar

    def test_calendar_modifier_in_range(self, orch):
        rec = orch.recommend()
        assert 0.0 < rec.calendar_modifier <= 1.0

    def test_execution_recommendation_present(self, orch):
        rec = orch.recommend()
        assert rec.execution_recommendation is not None
        assert len(rec.execution_recommendation) > 0
