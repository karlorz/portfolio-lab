"""
Tests for Overlay Dashboard Data Generator (v4.91)
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from src.dashboard.overlay_dashboard import (
    OverlayDashboardGenerator,
    OverlayDashboardData,
    generate_overlay_dashboard,
)


class TestOverlayDashboardData:
    """Test dashboard data dataclass."""

    def test_serializable(self):
        data = OverlayDashboardData(
            timestamp="2026-05-16", generated_at="2026-05-16",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": True},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=5, total_overlays=7,
            portfolio_risk="moderate",
            alerts=["Test alert"],
        )
        d = data.to_dict()
        assert d["active_overlays"] == 5
        assert d["portfolio_risk"] == "moderate"
        assert len(d["alerts"]) == 1


class TestOverlayDashboardGenerator:
    """Test dashboard generator."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_generates_dashboard(self, gen):
        dashboard = gen.generate()
        assert isinstance(dashboard, OverlayDashboardData)
        assert dashboard.timestamp is not None
        assert dashboard.total_overlays >= 1

    def test_collar_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.collar or "error" in dashboard.collar

    def test_crypto_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.crypto or "error" in dashboard.crypto

    def test_bond_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.bond_duration or "error" in dashboard.bond_duration

    def test_calendar_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.calendar

    def test_kurtosis_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.kurtosis or "error" in dashboard.kurtosis

    def test_unified_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.unified or "error" in dashboard.unified

    def test_active_count_reasonable(self, gen):
        dashboard = gen.generate()
        assert 0 <= dashboard.active_overlays <= dashboard.total_overlays

    def test_risk_level_valid(self, gen):
        dashboard = gen.generate()
        assert dashboard.portfolio_risk in ("low", "moderate", "elevated", "high")

    def test_save_dashboard(self, gen, tmp_path):
        gen.OUTPUT_PATH = tmp_path / "test_dashboard.json"
        dashboard = gen.generate()
        gen.save(dashboard)
        assert gen.OUTPUT_PATH.exists()

        with open(gen.OUTPUT_PATH) as f:
            loaded = json.load(f)
        assert "active_overlays" in loaded

    def test_convenience_function(self):
        dashboard = generate_overlay_dashboard()
        assert isinstance(dashboard, OverlayDashboardData)


class TestRiskAssessment:
    """Test risk assessment logic."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_low_risk_when_normal(self, gen):
        data = {
            "collar": {"vix_level": 15.0},
            "crypto": {"btc_vol_regime": "normal"},
            "kurtosis": {"fat_tail_risk": 0.1},
            "bond_duration": {"curve_regime": "normal"},
            "unified": {"conflict_count": 0},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"

    def test_high_risk_when_vix_elevated_and_conflicts(self, gen):
        data = {
            "collar": {"vix_level": 35.0},
            "crypto": {"btc_vol_regime": "extreme"},
            "kurtosis": {"fat_tail_risk": 0.9},
            "bond_duration": {"curve_regime": "inverted"},
            "unified": {"conflict_count": 2},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "high"
        assert len(alerts) >= 3

    def test_elevated_with_moderate_risk(self, gen):
        data = {
            "collar": {"vix_level": 28.0},
            "crypto": {"btc_vol_regime": "high"},
            "kurtosis": {"fat_tail_risk": 0.5},
            "bond_duration": {"curve_regime": "flat"},
            "unified": {"conflict_count": 1},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk in ("elevated", "moderate")


class TestEdgeCases:
    """Edge cases for dashboard."""

    def test_empty_data_handled(self):
        gen = OverlayDashboardGenerator()
        data = {
            "collar": {}, "crypto": {}, "bond_duration": {},
            "calendar": {}, "kurtosis": {}, "unified": {},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"

    def test_missing_keys_handled(self):
        gen = OverlayDashboardGenerator()
        data = {}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"


class TestRiskAssessmentExtended:
    """Additional edge cases for risk assessment."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_moderate_risk_with_single_factor(self, gen):
        """Single risk factor (VIX 26) -> moderate."""
        data = {"collar": {"vix_level": 26.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "moderate"
        assert len(alerts) == 0

    def test_elevated_risk_with_covid_combo(self, gen):
        """VIX elevated + high crypto + 1 conflict -> elevated (score=4)."""
        data = {
            "collar": {"vix_level": 28.0},
            "crypto": {"btc_vol_regime": "high"},
            "kurtosis": {"fat_tail_risk": 0.3},
            "unified": {"conflict_count": 1},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "elevated"

    def test_vix_exactly_30_triggers_alert(self, gen):
        """VIX exactly 30 should not trigger the >30 alert."""
        data = {"collar": {"vix_level": 30.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        # vix_level > 30 is False at exactly 30
        assert risk == "moderate"

    def test_vix_31_triggers_alert(self, gen):
        """VIX > 30 should trigger collar alert and add 2 to risk score."""
        data = {"collar": {"vix_level": 31.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) >= 1
        assert "VIX" in alerts[0]

    def test_btc_vol_extreme_triggers_alert(self, gen):
        """BTC extreme vol should add 2 to risk score."""
        data = {"crypto": {"btc_vol_regime": "extreme"}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert any("BTC" in a for a in alerts)

    def test_fat_tail_risk_threshold(self, gen):
        """fat_tail_risk > 0.7 should trigger alert, 0.7 exactly should not."""
        data_high = {"kurtosis": {"fat_tail_risk": 0.71}}
        _, alerts_high = gen._assess_portfolio_risk(data_high)
        assert any("fat tail" in a.lower() for a in alerts_high)

        data_low = {"kurtosis": {"fat_tail_risk": 0.7}}
        _, alerts_low = gen._assess_portfolio_risk(data_low)
        assert not any("fat tail" in a.lower() for a in alerts_low)

    def test_inverted_curve_adds_alert(self, gen):
        """Inverted yield curve should trigger defensive alert."""
        data = {"bond_duration": {"curve_regime": "inverted"}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert any("inverted" in a.lower() for a in alerts)

    def test_flat_curve_no_alert(self, gen):
        """Flat curve should not trigger curve alert."""
        data = {"bond_duration": {"curve_regime": "flat"}}
        _, alerts = gen._assess_portfolio_risk(data)
        assert not any("inverted" in a.lower() for a in alerts)

    def test_conflict_count_adds_to_score(self, gen):
        """Each conflict adds 1 to risk score."""
        data3 = {"unified": {"conflict_count": 3}}
        risk3, _ = gen._assess_portfolio_risk(data3)
        assert risk3 == "elevated"  # score = 3

    def test_high_risk_requires_score_5(self, gen):
        """Risk score >= 5 required for 'high'."""
        data = {
            "collar": {"vix_level": 31.0},   # +2
            "crypto": {"btc_vol_regime": "extreme"},  # +2
            "unified": {"conflict_count": 1},  # +1
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "high"
        assert len(alerts) >= 3


class TestOverlayDashboardGeneratorExtended:
    """Additional generator edge cases."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_mean_reversion_always_disabled(self, gen):
        """Mean reversion overlay is permanently disabled."""
        mr = gen._get_mean_reversion_data()
        assert mr["active"] is False
        assert "disabled" in mr.get("status_text", "").lower()

    def test_save_to_existing_dir(self, gen, tmp_path):
        """save() should write to a file in an existing directory."""
        output = tmp_path / "dashboard.json"
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": False}, crypto={"active": False},
            bond_duration={"active": False}, calendar={"active": False},
            kurtosis={"active": False}, mean_reversion={"active": False},
            unified={"active": False},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        gen.save(data)
        assert output.exists()

    def test_generate_counts_active_overlays(self, gen):
        """generate() should count active overlays correctly."""
        dashboard = gen.generate()
        # Verify count matches actual active values
        manual_count = sum(1 for key in ['collar', 'crypto', 'bond_duration',
                                           'calendar', 'kurtosis', 'mean_reversion', 'unified']
                          if getattr(dashboard, key, {}).get("active"))
        assert dashboard.active_overlays == manual_count

    def test_generate_total_always_7(self, gen):
        """Total overlays should always be 7."""
        dashboard = gen.generate()
        assert dashboard.total_overlays == 7

    def test_to_dict_roundtrip(self):
        """OverlayDashboardData.to_dict should produce serializable dict."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True, "vix_level": 18.0},
            crypto={"active": False},
            bond_duration={"active": True},
            calendar={"active": True},
            kurtosis={"active": True},
            mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        import json
        serialized = json.dumps(d)
        assert "active_overlays" in serialized
        assert d["collar"]["vix_level"] == 18.0


class TestOverlayDashboardDataFields:
    """Test to_dict() field completeness for OverlayDashboardData."""

    def test_to_dict_contains_all_fields(self):
        """to_dict() should include all 13 dataclass fields."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=3, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        expected_keys = {
            "timestamp", "generated_at", "collar", "crypto",
            "bond_duration", "calendar", "kurtosis", "mean_reversion",
            "unified", "active_overlays", "total_overlays",
            "portfolio_risk", "alerts",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_field_types(self):
        """to_dict() should preserve expected field types."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=3, total_overlays=7,
            portfolio_risk="low", alerts=["Alert 1"],
        )
        d = data.to_dict()
        assert isinstance(d["timestamp"], str)
        assert isinstance(d["generated_at"], str)
        assert isinstance(d["collar"], dict)
        assert isinstance(d["crypto"], dict)
        assert isinstance(d["active_overlays"], int)
        assert isinstance(d["total_overlays"], int)
        assert isinstance(d["portfolio_risk"], str)
        assert isinstance(d["alerts"], list)

    def test_to_dict_with_all_inactive_overlays(self):
        """to_dict() with all 7 overlays inactive."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": False}, crypto={"active": False},
            bond_duration={"active": False}, calendar={"active": False},
            kurtosis={"active": False}, mean_reversion={"active": False},
            unified={"active": False},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        assert d["active_overlays"] == 0
        for key in ("collar", "crypto", "bond_duration", "calendar",
                    "kurtosis", "mean_reversion", "unified"):
            assert d[key]["active"] is False

    def test_to_dict_with_all_active_overlays(self):
        """to_dict() with all active overlays (MR remains False)."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": True},
            bond_duration={"active": True}, calendar={"active": True},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=6, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        assert d["active_overlays"] == 6
        assert d["total_overlays"] == 7

    def test_to_dict_with_many_alerts(self):
        """to_dict() should handle a large alert list."""
        alerts = [f"Alert {i}" for i in range(20)]
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": True},
            bond_duration={"active": True}, calendar={"active": True},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=6, total_overlays=7,
            portfolio_risk="high", alerts=alerts,
        )
        d = data.to_dict()
        assert len(d["alerts"]) == 20
        assert d["portfolio_risk"] == "high"


class TestMeanReversionCompleteness:
    """Test mean reversion overlay data completeness."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_mean_reversion_status_text_format(self, gen):
        """_get_mean_reversion_data should include 'disabled' in status_text."""
        mr = gen._get_mean_reversion_data()
        assert "disabled" in mr.get("status_text", "").lower()

    def test_mean_reversion_dict_structure(self, gen):
        """_get_mean_reversion_data has exactly two expected keys."""
        mr = gen._get_mean_reversion_data()
        assert set(mr.keys()) == {"active", "status_text"}
        assert mr["active"] is False


class TestRiskAssessmentBoundaries:
    """Test boundary conditions in risk assessment scoring."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_vix_25_boundary_low_risk(self, gen):
        """VIX exactly 25 produces low risk (no score added)."""
        data = {"collar": {"vix_level": 25.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"
        assert len(alerts) == 0

    def test_vix_26_boundary_moderate_risk(self, gen):
        """VIX 26 produces moderate risk (score=1)."""
        data = {"collar": {"vix_level": 26.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "moderate"
        assert len(alerts) == 0

    def test_score_1_is_moderate(self, gen):
        """Risk score exactly 1 maps to moderate."""
        data = {"collar": {"vix_level": 26.0}}
        risk, _ = gen._assess_portfolio_risk(data)
        assert risk == "moderate"

    def test_score_2_is_moderate(self, gen):
        """Risk score exactly 2 maps to moderate."""
        data = {
            "collar": {"vix_level": 26.0},     # +1
            "crypto": {"btc_vol_regime": "high"},  # +1
        }
        risk, _ = gen._assess_portfolio_risk(data)
        assert risk == "moderate"

    def test_score_3_is_elevated(self, gen):
        """Risk score exactly 3 maps to elevated."""
        data = {
            "collar": {"vix_level": 31.0},         # +2
            "bond_duration": {"curve_regime": "inverted"},  # +1
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "elevated"
        assert len(alerts) >= 2

    def test_score_4_is_elevated(self, gen):
        """Risk score exactly 4 maps to elevated."""
        data = {
            "collar": {"vix_level": 31.0},             # +2
            "crypto": {"btc_vol_regime": "high"},       # +1
            "bond_duration": {"curve_regime": "inverted"},  # +1
        }
        risk, _ = gen._assess_portfolio_risk(data)
        assert risk == "elevated"

    def test_btc_vol_high_scores_one(self, gen):
        """BTC vol 'high' adds 1 (not 2), no alert generated."""
        data = {"crypto": {"btc_vol_regime": "high"}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "moderate"  # score = 1
        assert len(alerts) == 0

    def test_risk_score_zero_is_low(self, gen):
        """Risk score 0 maps to low with no alerts."""
        data = {
            "collar": {"vix_level": 15.0},
            "crypto": {"btc_vol_regime": "normal"},
            "kurtosis": {"fat_tail_risk": 0.1},
            "bond_duration": {"curve_regime": "normal"},
            "unified": {"conflict_count": 0},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"
        assert len(alerts) == 0


class TestOverlayDashboardDataEdgeCases:
    """Edge cases for OverlayDashboardData construction and constants."""

    def test_construct_with_all_inactive(self):
        """Construct with all overlays inactive and verify fields."""
        data = OverlayDashboardData(
            timestamp="", generated_at="",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        assert data.active_overlays == 0
        assert data.portfolio_risk == "low"
        assert data.alerts == []

    def test_construct_with_empty_alerts(self):
        """Empty alerts list is preserved through to_dict."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": True},
            bond_duration={"active": True}, calendar={"active": True},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=6, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        assert data.alerts == []
        d = data.to_dict()
        assert d["alerts"] == []
        assert isinstance(d["alerts"], list)

    def test_output_path_format(self):
        """OUTPUT_PATH should point to overlay_dashboard.json."""
        gen = OverlayDashboardGenerator()
        assert "overlay_dashboard.json" in str(gen.OUTPUT_PATH)
        assert str(gen.OUTPUT_PATH).endswith("overlay_dashboard.json")

    def test_portfolio_risk_valid_values(self):
        """All four portfolio risk values are accepted."""
        for risk in ("low", "moderate", "elevated", "high"):
            data = OverlayDashboardData(
                timestamp="", generated_at="",
                collar={}, crypto={}, bond_duration={},
                calendar={}, kurtosis={}, mean_reversion={},
                unified={},
                active_overlays=0, total_overlays=7,
                portfolio_risk=risk, alerts=[],
            )
            assert data.portfolio_risk == risk
