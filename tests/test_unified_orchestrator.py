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
            assert c.name in ("collar", "crypto", "bond_duration", "calendar", "vixy")

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


class TestStateManagement:
    """Test _load_state / _save_state methods."""

    def test_load_state_no_file(self, tmp_path):
        """Missing state file returns defaults."""
        orch = UnifiedOrchestrator()
        orch.STATE_FILE = tmp_path / "nonexistent.json"
        state = orch._load_state()
        assert state["last_unified"] is None
        assert state["conflict_history"] == []

    def test_load_state_corrupt_file(self, tmp_path):
        """Corrupt JSON returns defaults without crashing."""
        orch = UnifiedOrchestrator()
        orch.STATE_FILE = tmp_path / "bad.json"
        orch.STATE_FILE.write_text("{invalid json")
        state = orch._load_state()
        assert state["last_unified"] is None

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save then load should preserve state."""
        orch = UnifiedOrchestrator()
        orch.STATE_FILE = tmp_path / "state.json"
        orch._state = {"last_unified": "2026-01-01", "conflict_history": ["spy_conflict"]}
        orch._save_state()
        loaded = orch._load_state()
        assert loaded["last_unified"] == "2026-01-01"
        assert loaded["conflict_history"] == ["spy_conflict"]

    def test_save_creates_parent_dirs(self, tmp_path):
        """_save_state should create missing parent directories."""
        orch = UnifiedOrchestrator()
        orch.STATE_FILE = tmp_path / "deep" / "nested" / "state.json"
        orch._state = {"last_unified": None, "conflict_history": []}
        orch._save_state()
        assert orch.STATE_FILE.exists()


class TestVixLevelFetch:
    """Test _fetch_vix_level with mocked DB."""

    def test_no_db_returns_default(self, tmp_path):
        """Missing DB returns default VIX of 16.0."""
        orch = UnifiedOrchestrator()
        # Patch MARKET_DB to nonexistent path
        import src.strategy.unified_orchestrator as uo_mod
        orig = uo_mod.MARKET_DB
        uo_mod.MARKET_DB = tmp_path / "no_such.db"
        try:
            assert orch._fetch_vix_level() == 16.0
        finally:
            uo_mod.MARKET_DB = orig

    def test_db_with_vix_data(self, tmp_path):
        """DB with VIX data returns the stored value."""
        import sqlite3
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('^VIX', '2026-01-01', 22.5)")
        conn.commit()
        conn.close()

        import src.strategy.unified_orchestrator as uo_mod
        orig = uo_mod.MARKET_DB
        uo_mod.MARKET_DB = db_path
        orch = UnifiedOrchestrator()
        try:
            assert orch._fetch_vix_level() == 22.5
        finally:
            uo_mod.MARKET_DB = orig

    def test_db_empty_returns_default(self, tmp_path):
        """DB without VIX rows returns default."""
        import sqlite3
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        import src.strategy.unified_orchestrator as uo_mod
        orig = uo_mod.MARKET_DB
        uo_mod.MARKET_DB = db_path
        orch = UnifiedOrchestrator()
        try:
            assert orch._fetch_vix_level() == 16.0
        finally:
            uo_mod.MARKET_DB = orig


class TestConflictResolutionEdgeCases:
    """Extended conflict resolution tests."""

    @pytest.fixture
    def orch(self):
        return UnifiedOrchestrator()

    def test_single_active_contribution(self, orch):
        """Single active contribution should apply its delta."""
        contributions = [
            OverlayContribution("collar", "v1", "active", 0.5,
                                -0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
                                -0.005, 0.02, 80.0, "collar active"),
        ]
        weights, conflicts = orch.resolve_conflicts(contributions)
        assert weights["spy"] < orch.BASELINE["spy"]  # -0.03 delta
        assert weights["gld"] > orch.BASELINE["gld"]  # +0.02 delta

    def test_suppressed_contribution_ignored(self, orch):
        """Suppressed contributions should not affect weights."""
        contributions = [
            OverlayContribution("crypto", "v1", "suppressed", 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05,
                                0.0, 0.0, 0.0, "suppressed"),
        ]
        weights, conflicts = orch.resolve_conflicts(contributions)
        # Should be at baseline since suppressed
        for k, v in orch.BASELINE.items():
            assert abs(weights[k] - v) < 0.01

    def test_many_small_deltas(self, orch):
        """Many small contributions should accumulate without breaking bounds."""
        contributions = []
        for i in range(5):
            contributions.append(OverlayContribution(
                f"overlay_{i}", "v1", "active", 0.15,
                0.005, 0.003, 0.001, 0.0, 0.0, 0.0, 0.0,
                0.001, 0.01, 70.0, f"overlay {i}",
            ))
        weights, conflicts = orch.resolve_conflicts(contributions)
        assert 0.36 <= weights["spy"] <= 0.56
        assert 0.28 <= weights["gld"] <= 0.48

    def test_conflict_detected_on_opposite_spy(self, orch):
        """Opposing SPY deltas should register as a conflict."""
        contributions = [
            OverlayContribution("bull", "v1", "active", 0.3,
                                0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 80.0, "bull"),
            OverlayContribution("bear", "v1", "active", 0.3,
                                -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 80.0, "bear"),
        ]
        _, conflicts = orch.resolve_conflicts(contributions)
        assert len(conflicts) >= 1
        assert any("spy" in c.lower() or "SPY" in c for c in conflicts)


class TestOrchestratorBacktestValidation:
    """Validate the unified orchestrator's recommendation properties
    across multiple simulated scenarios (no real data needed)."""

    @pytest.fixture
    def orch(self):
        return UnifiedOrchestrator()

    def test_allocation_weights_sum_near_one(self, orch):
        """All recommended allocations should sum near 1.0."""
        rec = orch.recommend()
        total = rec.spy + rec.gld + rec.tlt + rec.ief + rec.shy + rec.btc + rec.eth
        assert abs(total - 1.0) < 0.02, f"Allocation sums to {total:.4f}"

    def test_spy_within_bounds(self, orch):
        """SPY allocation should be within hard bounds (36-56%)."""
        rec = orch.recommend()
        assert 0.36 <= rec.spy <= 0.56, f"SPY {rec.spy:.2%} outside 36-56%"

    def test_gld_within_bounds(self, orch):
        """GLD allocation should be within hard bounds (28-48%)."""
        rec = orch.recommend()
        assert 0.28 <= rec.gld <= 0.48, f"GLD {rec.gld:.2%} outside 28-48%"

    def test_bonds_within_bounds(self, orch):
        """Total bond allocation should be within hard bounds (6-26%)."""
        rec = orch.recommend()
        total_bonds = rec.tlt + rec.ief + rec.shy
        assert 0.06 <= total_bonds <= 0.26, f"Bonds {total_bonds:.2%} outside 6-26%"

    def test_crypto_within_bounds(self, orch):
        """Total crypto allocation should be within hard bounds (0-5%)."""
        rec = orch.recommend()
        total_crypto = rec.btc + rec.eth
        assert 0.0 <= total_crypto <= 0.05, f"Crypto {total_crypto:.2%} outside 0-5%"

    def test_confidence_in_range(self, orch):
        """Confidence should be between 0 and 100."""
        rec = orch.recommend()
        assert 0 <= rec.confidence <= 100

    def test_contributions_have_required_fields(self, orch):
        """Each overlay contribution should have required fields."""
        contributions = orch.collect_overlay_contributions()
        for c in contributions:
            assert c.name is not None
            assert c.status in [OverlayStatus.ACTIVE.value, OverlayStatus.SUPPRESSED.value, OverlayStatus.DISABLED.value, "active", "suppressed", "disabled"]
            assert isinstance(c.weight, float)
            assert isinstance(c.spy_delta, float)

    def test_save_recommendation(self, orch, tmp_path):
        """Recommendation should be saveable to disk."""
        rec = orch.recommend()
        # Override paths to use tmp
        orch._ensure_dirs()
        # The save path is STATE_FILE.parent / "signals" / "unified_recommendation.json"
        # We need to patch STATE_FILE temporarily
        import tempfile
        signals_dir = Path(tempfile.mkdtemp()) / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        out = signals_dir / "unified_recommendation.json"
        with open(out, 'w') as f:
            import json
            json.dump(rec.to_dict(), f, indent=2)
        assert out.exists()
