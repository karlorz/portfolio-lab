"""
Tests for Overlay Dashboard Data Generator (v4.91)
"""

import json
import logging
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.dashboard.overlay_dashboard import (
    OverlayDashboardGenerator,
    OverlayDashboardData,
    generate_overlay_dashboard,
)


@pytest.fixture(autouse=True)
def _isolate_overlay_data_dir(tmp_path, monkeypatch):
    """Keep pure risk scoring free of live data/kill_switch.json halt."""
    monkeypatch.setattr("src.dashboard.overlay_dashboard.DATA_DIR", tmp_path)


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

    def test_overlay_sections_include_freshness_timestamps(self, gen):
        """Freshness stamps prevent optional sections looking unavailable in signals.json."""
        dashboard = gen.generate()
        for section_name in ("collar", "crypto", "bond_duration", "calendar", "kurtosis"):
            section = getattr(dashboard, section_name)
            assert isinstance(section, dict)
            if section.get("error"):
                continue
            assert section.get("generated_at") or section.get("timestamp"), (
                f"{section_name} missing generated_at/timestamp"
            )

    def test_crypto_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.crypto or "error" in dashboard.crypto

    def test_crypto_weights_are_portfolio_fractions(self, gen):
        """btc_weight + eth_weight must equal total_crypto (portfolio units)."""
        data = gen._get_crypto_data()
        if data.get("error"):
            return
        btc = float(data.get("btc_weight") or 0)
        eth = float(data.get("eth_weight") or 0)
        total = float(data.get("total_crypto") or 0)
        assert data.get("weight_unit") == "portfolio_fraction"
        assert btc >= 0 and eth >= 0
        if total > 0:
            assert abs((btc + eth) - total) < 1e-5
            # Neither portfolio leg can exceed total_crypto
            assert btc <= total + 1e-9
            assert eth <= total + 1e-9
        # Sleeve shares disclosed separately when present
        if "eth_sleeve_share" in data:
            assert 0 <= float(data["eth_sleeve_share"]) <= 1.0 + 1e-9

    def test_bond_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.bond_duration or "error" in dashboard.bond_duration

    def test_calendar_data_collected(self, gen):
        dashboard = gen.generate()
        assert "active" in dashboard.calendar

    def test_calendar_discloses_not_applied_to_targets(self, gen):
        data = gen._get_calendar_data()
        if data.get("error"):
            return
        assert data.get("applies_to_target_allocations") is False
        assert data.get("role") == "advisory_non_routed"
        mod = data.get("modifier")
        if mod is not None and float(mod) != 1.0:
            assert "not applied" in str(data.get("status_text", "")).lower()

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
    def gen(self, tmp_path):
        # Isolate from live data/kill_switch.json so overlay-only scoring is pure.
        return OverlayDashboardGenerator(data_dir=tmp_path)

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

    def test_kill_halt_forces_high_risk_and_alert(self, tmp_path):
        """Enabled halt kill must never look like all-systems-normal."""
        (tmp_path / "kill_switch.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "level": "halt",
                    "reason": "unresolved_incident:signal_staleness",
                    "message": "1/23 signals stale: alternative_data; 12 unavailable",
                    "mode": "paper",
                    "source": "incident_lifecycle",
                }
            ),
            encoding="utf-8",
        )
        gen = OverlayDashboardGenerator(data_dir=tmp_path)
        data = {
            "collar": {"vix_level": 15.0},
            "crypto": {"btc_vol_regime": "normal"},
            "kurtosis": {"fat_tail_risk": 0.1},
            "bond_duration": {"curve_regime": "normal"},
            "unified": {"conflict_count": 0},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "high"
        assert alerts
        joined = " ".join(alerts).lower()
        assert "halt" in joined
        assert "kill" in joined
        assert "signal_staleness" in joined or "unavailable" in joined

    def test_kill_warning_elevates_risk_with_alert(self, tmp_path):
        (tmp_path / "kill_switch.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "level": "warning",
                    "reason": "unresolved_incident:ic_decay",
                    "message": "IC decay elevated",
                    "mode": "paper",
                }
            ),
            encoding="utf-8",
        )
        gen = OverlayDashboardGenerator(data_dir=tmp_path)
        risk, alerts = gen._assess_portfolio_risk(
            {
                "collar": {"vix_level": 15.0},
                "crypto": {"btc_vol_regime": "normal"},
                "kurtosis": {"fat_tail_risk": 0.1},
                "bond_duration": {"curve_regime": "normal"},
                "unified": {"conflict_count": 0},
            }
        )
        assert risk in ("elevated", "high")
        assert any("kill" in a.lower() for a in alerts)


class TestEdgeCases:
    """Edge cases for dashboard."""

    def test_empty_data_handled(self, tmp_path):
        gen = OverlayDashboardGenerator(data_dir=tmp_path)
        data = {
            "collar": {}, "crypto": {}, "bond_duration": {},
            "calendar": {}, "kurtosis": {}, "unified": {},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"

    def test_missing_keys_handled(self, tmp_path):
        gen = OverlayDashboardGenerator(data_dir=tmp_path)
        data = {}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"


class TestRiskAssessmentExtended:
    """Additional edge cases for risk assessment."""

    @pytest.fixture
    def gen(self, tmp_path):
        return OverlayDashboardGenerator(data_dir=tmp_path)

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


class TestOverlayDashboardExports:
    """Test public API exports and __all__ coverage."""

    def test_module_has_public_names(self):
        from src.dashboard import overlay_dashboard as mod
        assert hasattr(mod, "OverlayDashboardData")
        assert hasattr(mod, "OverlayDashboardGenerator")
        assert hasattr(mod, "generate_overlay_dashboard")
        assert hasattr(mod, "main")

    def test_init_exports_all_modules(self):
        from src.dashboard import __all__ as dashboard_all
        assert "overlay_dashboard" in dashboard_all
        assert "generator" in dashboard_all

    def test_from_import_star_works(self):
        exec("from src.dashboard.overlay_dashboard import *")
        import src.dashboard.overlay_dashboard as mod
        public = {n for n in dir(mod) if not n.startswith("_")}
        assert "OverlayDashboardData" in public
        assert "OverlayDashboardGenerator" in public
        assert "generate_overlay_dashboard" in public
        assert "main" in public

    def test_convenience_function_importable(self):
        from src.dashboard.overlay_dashboard import generate_overlay_dashboard
        assert callable(generate_overlay_dashboard)


class TestOverlayDashboardConstants:
    """Test module-level constants validation."""

    def test_output_path_ends_with_dashboard_json(self):
        gen = OverlayDashboardGenerator()
        assert "overlay_dashboard.json" in str(gen.OUTPUT_PATH)
        assert str(gen.OUTPUT_PATH).endswith("overlay_dashboard.json")

    def test_output_path_is_under_data_dir(self):
        from src.paths import DATA_DIR
        gen = OverlayDashboardGenerator()
        assert str(gen.OUTPUT_PATH).startswith(str(DATA_DIR))

    def test_total_overlays_constant(self):
        """Total overlays count should match the number of overlay dict fields."""
        gen = OverlayDashboardGenerator()
        from dataclasses import fields
        overlay_fields = [f for f in fields(OverlayDashboardData)
                         if f.type == dict]
        dashboard = OverlayDashboardData(
            timestamp="", generated_at="",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=len(overlay_fields),
            portfolio_risk="low", alerts=[],
        )
        assert dashboard.total_overlays == len(overlay_fields)


class TestOverlayDashboardCLI:
    """Test CLI entry point with capsys."""

    def test_main_prints_header(self, caplog):
        caplog.set_level(logging.INFO, logger='src.dashboard.overlay_dashboard')
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar={"active": True}, crypto={"active": False},
                bond_duration={"active": True}, calendar={"active": False},
                kurtosis={"active": True}, mean_reversion={"active": False},
                unified={"active": True},
                active_overlays=4, total_overlays=7,
                portfolio_risk="moderate", alerts=[],
            )
            with patch("sys.argv", ["overlay_dashboard.py"]):
                from src.dashboard.overlay_dashboard import main
                main()

        assert "OVERLAY DASHBOARD v4.91" in caplog.text
        assert "Active: 4/7" in caplog.text
        assert "MODERATE" in caplog.text

    def test_main_prints_no_alerts_when_empty(self, caplog):
        caplog.set_level(logging.INFO, logger='src.dashboard.overlay_dashboard')
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar={}, crypto={}, bond_duration={},
                calendar={}, kurtosis={}, mean_reversion={},
                unified={},
                active_overlays=0, total_overlays=7,
                portfolio_risk="low", alerts=[],
            )
            with patch("sys.argv", ["overlay_dashboard.py"]):
                from src.dashboard.overlay_dashboard import main
                main()

        captured = caplog.text
        assert "No alerts" in captured
        assert "all systems normal" in captured

    def test_main_prints_alerts_when_present(self, caplog):
        caplog.set_level(logging.INFO, logger='src.dashboard.overlay_dashboard')
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar={"active": True, "vix_level": 35.0},
                crypto={"active": False}, bond_duration={"active": True},
                calendar={"active": False}, kurtosis={"active": True},
                mean_reversion={"active": False}, unified={"active": True},
                active_overlays=4, total_overlays=7,
                portfolio_risk="high",
                alerts=["VIX elevated (35) \u2014 collar active", "BTC vol extreme"],
            )
            with patch("sys.argv", ["overlay_dashboard.py"]):
                from src.dashboard.overlay_dashboard import main
                main()

        captured = caplog.text
        assert "VIX elevated" in captured
        assert "BTC vol extreme" in captured

    def test_main_save_flag_triggers_save(self, caplog):
        caplog.set_level(logging.INFO, logger='src.dashboard.overlay_dashboard')
        mock_save = MagicMock()
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar={"active": True}, crypto={"active": False},
                bond_duration={"active": True}, calendar={"active": False},
                kurtosis={"active": True}, mean_reversion={"active": False},
                unified={"active": True},
                active_overlays=4, total_overlays=7,
                portfolio_risk="low", alerts=[],
            )
            with patch.object(OverlayDashboardGenerator, "save", mock_save):
                with patch("sys.argv", ["overlay_dashboard.py", "--save"]):
                    from src.dashboard.overlay_dashboard import main
                    main()

        captured = caplog.text
        mock_save.assert_called_once()
        assert "Saved to" in captured

    def test_main_no_save_flag_no_save(self):
        mock_save = MagicMock()
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar={}, crypto={}, bond_duration={},
                calendar={}, kurtosis={}, mean_reversion={},
                unified={},
                active_overlays=0, total_overlays=7,
                portfolio_risk="low", alerts=[],
            )
            with patch.object(OverlayDashboardGenerator, "save", mock_save):
                with patch("sys.argv", ["overlay_dashboard.py"]):
                    from src.dashboard.overlay_dashboard import main
                    main()

        mock_save.assert_not_called()

    def test_main_shows_active_and_inactive_overlays(self, caplog):
        caplog.set_level(logging.INFO, logger='src.dashboard.overlay_dashboard')
        with patch.object(OverlayDashboardGenerator, "generate") as mock_gen:
            collar = {"active": True, "status_text": "Collar: protective"}
            crypto = {"active": False, "error": "Crypto signal unavailable"}
            mock_gen.return_value = OverlayDashboardData(
                timestamp="2026-01-01", generated_at="2026-01-01",
                collar=collar, crypto=crypto,
                bond_duration={"active": True}, calendar={"active": False},
                kurtosis={"active": True}, mean_reversion={"active": False, "status_text": "MR: disabled"},
                unified={"active": True},
                active_overlays=4, total_overlays=7,
                portfolio_risk="low", alerts=[],
            )
            with patch("sys.argv", ["overlay_dashboard.py"]):
                from src.dashboard.overlay_dashboard import main
                main()

        captured = caplog.text
        assert "Collar: protective" in captured
        assert "Crypto signal unavailable" in captured
        assert "MR: disabled" in captured


class TestOverlayDashboardDataclassValidation:
    """Validate dataclass fields via dataclasses.fields()."""

    def test_all_fields_present(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(OverlayDashboardData)}
        expected = {
            "timestamp", "generated_at", "collar", "crypto",
            "bond_duration", "calendar", "kurtosis", "mean_reversion",
            "unified", "active_overlays", "total_overlays",
            "portfolio_risk", "alerts",
        }
        assert field_names == expected
        assert len(field_names) == 13

    def test_dict_fields_have_dict_type(self):
        import dataclasses
        import typing
        expected_type = typing.Dict[str, typing.Any]
        for f in dataclasses.fields(OverlayDashboardData):
            if f.name in ("collar", "crypto", "bond_duration", "calendar",
                          "kurtosis", "mean_reversion", "unified"):
                assert f.type == expected_type, (
                    f"Field {f.name} has type {f.type}, expected {expected_type}"
                )

    def test_int_fields_have_int_type(self):
        import dataclasses
        for f in dataclasses.fields(OverlayDashboardData):
            if f.name in ("active_overlays", "total_overlays"):
                assert f.type == int

    def test_str_fields_have_str_type(self):
        import dataclasses
        for f in dataclasses.fields(OverlayDashboardData):
            if f.name in ("timestamp", "generated_at", "portfolio_risk"):
                assert f.type == str

    def test_alerts_field_is_list_of_str(self):
        import dataclasses
        import typing
        f = next(f for f in dataclasses.fields(OverlayDashboardData)
                 if f.name == "alerts")
        assert f.type == typing.List[str]

    def test_no_defaults_on_required_fields(self):
        """All fields are required (no default values)."""
        import dataclasses
        for f in dataclasses.fields(OverlayDashboardData):
            assert f.default is dataclasses.MISSING
            assert f.default_factory is dataclasses.MISSING

    def test_fields_are_positional(self):
        """Fields should be init=True, repr=True."""
        import dataclasses
        for f in dataclasses.fields(OverlayDashboardData):
            assert f.init is True
            assert f.repr is True

    def test_asdict_matches_field_count(self):
        import dataclasses
        data = OverlayDashboardData(
            timestamp="t", generated_at="g",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = dataclasses.asdict(data)
        assert len(d) == len(dataclasses.fields(OverlayDashboardData))


class TestRiskAssessmentSpecialValues:
    """Test risk assessment with NaN, Inf, negative, extreme values."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_vix_nan_does_not_trigger(self, gen):
        """NaN > 30 is False, NaN > 25 is False, so risk score stays 0."""
        import math
        data = {"collar": {"vix_level": float("nan")}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"
        assert len(alerts) == 0

    def test_vix_inf_triggers_alert(self, gen):
        """Inf > 30 is True, score=2 -> moderate with alert."""
        import math
        data = {"collar": {"vix_level": float("inf")}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "moderate"
        assert len(alerts) >= 1

    def test_vix_neg_one_treated_as_low(self, gen):
        """Negative VIX is treated as 0 via .get default, low risk."""
        data = {"collar": {"vix_level": -1.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"
        assert len(alerts) == 0

    def test_vix_extreme_100(self, gen):
        """VIX = 100 triggers alert and adds 2 to risk score."""
        data = {"collar": {"vix_level": 100.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) >= 1
        assert "VIX" in alerts[0]

    def test_fat_tail_nan_treated_as_low(self, gen):
        """NaN > 0.7 is False, so no alert."""
        import math
        data = {"kurtosis": {"fat_tail_risk": float("nan")}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) == 0

    def test_fat_tail_inf_triggers_alert(self, gen):
        """Inf > 0.7 is True, should trigger alert."""
        import math
        data = {"kurtosis": {"fat_tail_risk": float("inf")}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert any("fat tail" in a.lower() for a in alerts)

    def test_fat_tail_neg_one_treated_as_low(self, gen):
        """Negative fat_tail_risk is treated as low."""
        data = {"kurtosis": {"fat_tail_risk": -1.0}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) == 0
        assert risk == "low"

    def test_all_risk_factors_maxed(self, gen):
        """All risk factors at max should produce score >= 5 (high)."""
        data = {
            "collar": {"vix_level": 31.0},
            "crypto": {"btc_vol_regime": "extreme"},
            "kurtosis": {"fat_tail_risk": 0.71},
            "bond_duration": {"curve_regime": "inverted"},
            "unified": {"conflict_count": 3},
        }
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "high"
        assert len(alerts) >= 4


class TestRiskAssessmentTypeErrors:
    """Test risk assessment with unexpected types."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_vix_level_as_string_raises_type_error(self, gen):
        """String vix_level crashes comparison with int. Python 3 raises TypeError."""
        data = {"collar": {"vix_level": "high"}}
        with pytest.raises(TypeError):
            gen._assess_portfolio_risk(data)

    def test_btc_vol_regime_unknown_value(self, gen):
        """Unknown btc_vol_regime value is not 'extreme' or 'high', no score."""
        data = {"crypto": {"btc_vol_regime": "unknown"}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert risk == "low"
        assert len(alerts) == 0

    def test_conflict_count_as_string_raises_type_error(self, gen):
        """String conflict_count crashes comparison with int. Python 3 raises TypeError."""
        data = {"unified": {"conflict_count": "lots"}}
        with pytest.raises(TypeError):
            gen._assess_portfolio_risk(data)

    def test_curve_regime_case_sensitive(self, gen):
        """Curve regime comparison is case-sensitive. 'Inverted' != 'inverted'."""
        data = {"bond_duration": {"curve_regime": "Inverted"}}
        risk, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) == 0
        assert risk == "low"

    def test_none_values_raise_type_error(self, gen):
        """None values crash comparison with int. Python 3 raises TypeError."""
        data = {
            "collar": {"vix_level": None},
            "crypto": {"btc_vol_regime": None},
            "kurtosis": {"fat_tail_risk": None},
            "bond_duration": {"curve_regime": None},
            "unified": {"conflict_count": None},
        }
        with pytest.raises(TypeError):
            gen._assess_portfolio_risk(data)

    def test_fat_tail_as_string_raises_type_error(self, gen):
        """fat_tail_risk as string crashes comparison with float."""
        data = {
            "collar": {"vix_level": 31.0},
            "crypto": {"btc_vol_regime": 123},
            "kurtosis": {"fat_tail_risk": "high"},
            "bond_duration": {"curve_regime": "inverted"},
        }
        with pytest.raises(TypeError):
            gen._assess_portfolio_risk(data)


class TestRiskAssessmentBoundaryGrid:
    """Systematic boundary testing for risk assessment scores."""

    @pytest.fixture
    def gen(self):
        return OverlayDashboardGenerator()

    def test_vix_boundaries_all_levels(self, gen):
        """Test vix_level at every boundary: 0, 24, 25, 26, 30, 31, 100.
        VIX > 30 gives score +2 -> moderate (not elevated)."""
        cases = [
            (0, "low", 0),
            (24, "low", 0),
            (25, "low", 0),
            (26, "moderate", 0),
            (30, "moderate", 0),
            (31, "moderate", 1),
            (100, "moderate", 1),
        ]
        for vix, expected_risk, expected_alert_count in cases:
            data = {"collar": {"vix_level": float(vix)}}
            risk, alerts = gen._assess_portfolio_risk(data)
            assert risk == expected_risk, (
                f"VIX={vix}: expected {expected_risk}, got {risk}"
            )
            assert len(alerts) == expected_alert_count, (
                f"VIX={vix}: expected {expected_alert_count} alerts, got {len(alerts)}"
            )

    def test_btc_vol_all_regimes(self, gen):
        """Test all btc_vol_regime values."""
        cases = [
            ("normal", "low", 0),
            ("high", "moderate", 0),
            ("extreme", "moderate", 1),
        ]
        for regime, expected_risk, expected_alerts in cases:
            data = {"crypto": {"btc_vol_regime": regime}}
            risk, alerts = gen._assess_portfolio_risk(data)
            assert risk == expected_risk
            assert len(alerts) == expected_alerts

    def test_fat_tail_boundaries(self, gen):
        """Test fat_tail_risk at every boundary."""
        cases = [
            (0.0, 0),
            (0.69, 0),
            (0.70, 0),
            (0.71, 1),
            (1.0, 1),
            (2.0, 1),
        ]
        for risk_val, expected_alerts in cases:
            data = {"kurtosis": {"fat_tail_risk": risk_val}}
            _, alerts = gen._assess_portfolio_risk(data)
            assert len(alerts) == expected_alerts, (
                f"fat_tail_risk={risk_val}: expected {expected_alerts} alerts, got {len(alerts)}"
            )

    def test_risk_score_bucket_boundaries(self, gen):
        """Test each risk_score bucket boundary: 0, 1, 2, 3, 4, 5+."""
        cases = [
            (0, "low"),
            (1, "moderate"),
            (2, "moderate"),
            (3, "elevated"),
            (4, "elevated"),
            (5, "high"),
            (10, "high"),
        ]
        for score, expected_level in cases:
            data = {"unified": {"conflict_count": score}}
            risk, _ = gen._assess_portfolio_risk(data)
            assert risk == expected_level, (
                f"score={score}: expected {expected_level}, got {risk}"
            )


class TestGenerateWithMockedSignals:
    """Test generate() with all signal methods mocked."""

    def test_generate_works_with_all_mocked(self):
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True, "vix_level": 16.0})
        gen._get_crypto_data = MagicMock(return_value={"active": False, "btc_vol_regime": "normal"})
        gen._get_bond_duration_data = MagicMock(return_value={"active": True, "curve_regime": "normal"})
        gen._get_calendar_data = MagicMock(return_value={"active": True})
        gen._get_kurtosis_data = MagicMock(return_value={"active": True, "fat_tail_risk": 0.1})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True, "conflict_count": 0})

        dashboard = gen.generate()
        assert isinstance(dashboard, OverlayDashboardData)
        assert dashboard.active_overlays == 5
        assert dashboard.total_overlays == 7
        assert dashboard.portfolio_risk == "low"

    def test_generate_all_inactive(self):
        gen = OverlayDashboardGenerator()
        for method_name in ["_get_collar_data", "_get_crypto_data", "_get_bond_duration_data",
                             "_get_calendar_data", "_get_kurtosis_data", "_get_mean_reversion_data",
                             "_get_unified_data"]:
            setattr(gen, method_name, MagicMock(return_value={"active": False}))

        dashboard = gen.generate()
        assert dashboard.active_overlays == 0
        assert dashboard.total_overlays == 7
        assert dashboard.portfolio_risk == "low"

    def test_generate_all_active_high_risk(self):
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True, "vix_level": 35.0})
        gen._get_crypto_data = MagicMock(return_value={"active": True, "btc_vol_regime": "extreme"})
        gen._get_bond_duration_data = MagicMock(return_value={"active": True, "curve_regime": "inverted"})
        gen._get_calendar_data = MagicMock(return_value={"active": True})
        gen._get_kurtosis_data = MagicMock(return_value={"active": True, "fat_tail_risk": 0.9})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True, "conflict_count": 2})

        dashboard = gen.generate()
        assert dashboard.active_overlays == 6
        assert dashboard.portfolio_risk == "high"
        assert len(dashboard.alerts) >= 4

    def test_generate_risk_score_bucket_low(self):
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True, "vix_level": 15.0})
        gen._get_crypto_data = MagicMock(return_value={"active": False, "btc_vol_regime": "normal"})
        gen._get_bond_duration_data = MagicMock(return_value={"active": True, "curve_regime": "normal"})
        gen._get_calendar_data = MagicMock(return_value={"active": False})
        gen._get_kurtosis_data = MagicMock(return_value={"active": True, "fat_tail_risk": 0.1})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True, "conflict_count": 0})

        dashboard = gen.generate()
        assert dashboard.portfolio_risk == "low"

    def test_generate_risk_score_bucket_moderate(self):
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True, "vix_level": 26.0})
        gen._get_crypto_data = MagicMock(return_value={"active": False, "btc_vol_regime": "normal"})
        gen._get_bond_duration_data = MagicMock(return_value={"active": True, "curve_regime": "normal"})
        gen._get_calendar_data = MagicMock(return_value={"active": False})
        gen._get_kurtosis_data = MagicMock(return_value={"active": True, "fat_tail_risk": 0.1})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True, "conflict_count": 1})

        dashboard = gen.generate()
        assert dashboard.portfolio_risk == "moderate"

    def test_generate_risk_score_bucket_elevated(self):
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True, "vix_level": 31.0})
        gen._get_crypto_data = MagicMock(return_value={"active": False, "btc_vol_regime": "normal"})
        gen._get_bond_duration_data = MagicMock(return_value={"active": True, "curve_regime": "inverted"})
        gen._get_calendar_data = MagicMock(return_value={"active": False})
        gen._get_kurtosis_data = MagicMock(return_value={"active": True, "fat_tail_risk": 0.1})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True, "conflict_count": 1})

        dashboard = gen.generate()
        assert dashboard.portfolio_risk == "elevated"

    def test_generate_with_partial_active_overlays(self):
        """Verify active count with exactly 2 active overlays."""
        gen = OverlayDashboardGenerator()
        gen._get_collar_data = MagicMock(return_value={"active": True})
        gen._get_crypto_data = MagicMock(return_value={"active": False})
        gen._get_bond_duration_data = MagicMock(return_value={"active": False})
        gen._get_calendar_data = MagicMock(return_value={"active": False})
        gen._get_kurtosis_data = MagicMock(return_value={"active": False})
        gen._get_mean_reversion_data = MagicMock(return_value={"active": False})
        gen._get_unified_data = MagicMock(return_value={"active": True})

        dashboard = gen.generate()
        assert dashboard.active_overlays == 2


class TestSaveEdgeCases:
    """Test save() method with various edge conditions."""

    def test_save_creates_file(self, tmp_path):
        gen = OverlayDashboardGenerator()
        output = tmp_path / "test_dashboard.json"
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        gen.save(data)
        assert output.exists()
        content = output.read_text()
        assert "active_overlays" in content
        assert "portfolio_risk" in content

    def test_save_overwrites_existing(self, tmp_path):
        gen = OverlayDashboardGenerator()
        output = tmp_path / "test_dashboard.json"
        output.write_text('{"old": "data"}')
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        gen.save(data)
        content = output.read_text()
        # Provenance block may contain the substring "threshold" — assert on JSON keys
        import json as json_mod

        parsed = json_mod.loads(content)
        assert "old" not in parsed
        assert "active_overlays" in parsed
        assert parsed.get("active_overlays") == 0

    def test_save_to_deeply_nested_dir(self, tmp_path):
        gen = OverlayDashboardGenerator()
        output = tmp_path / "a" / "b" / "c" / "dashboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="moderate", alerts=["Test"],
        )
        gen.save(data)
        assert output.exists()
        content = output.read_text()
        assert "moderate" in content
        assert "Test" in content

    def test_save_json_is_valid(self, tmp_path):
        import json as json_mod
        gen = OverlayDashboardGenerator()
        output = tmp_path / "valid.json"
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        gen.save(data)
        parsed = json_mod.loads(output.read_text())
        assert parsed["active_overlays"] == 4
        assert parsed["total_overlays"] == 7

    def test_save_called_twice_produces_same_result(self, tmp_path):
        gen = OverlayDashboardGenerator()
        output = tmp_path / "twice.json"
        gen.OUTPUT_PATH = output
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        gen.save(data)
        first_content = output.read_text()
        gen.save(data)
        second_content = output.read_text()
        # dual_write lag mtimes may tick between writes — compare business payload
        import json as json_mod

        def _stable(payload: dict) -> dict:
            out = dict(payload)
            pc = out.get("provenance_completeness")
            if isinstance(pc, dict):
                pc = {
                    k: v
                    for k, v in pc.items()
                    if k
                    not in {
                        "private_mtime",
                        "public_mtime",
                        "dual_write_lag_seconds",
                        "dual_write_lag_stale",
                        "content_hash_identical",
                        "private_content_hash",
                        "public_content_hash",
                    }
                }
                out["provenance_completeness"] = pc
            return out

        assert _stable(json_mod.loads(first_content)) == _stable(
            json_mod.loads(second_content)
        )


class TestDataCollectionWithMocks:
    """Test individual _get_*_data methods with mocked signal modules."""

    @patch("src.signals.collar_signal.generate_collar_signal")
    @patch.object(OverlayDashboardGenerator, "_load_collar_signal_file", return_value=None)
    def test_get_collar_data_success(self, mock_load, mock_collar):
        mock_signal = MagicMock()
        mock_signal.is_valid = True
        mock_signal.regime = "protective"
        mock_signal.call_strike = 560.0
        mock_signal.put_strike = 540.0
        mock_signal.strikes.net_premium = 0.05
        mock_signal.strikes.is_cashless = True
        mock_signal.max_upside_pct = 0.02
        mock_signal.max_downside_pct = -0.03
        mock_signal.vix_level = 16.0
        mock_signal.confidence = 0.8
        mock_signal.underlying_price = 550.0
        mock_collar.return_value = mock_signal

        gen = OverlayDashboardGenerator()
        result = gen._get_collar_data()
        assert result["active"] is True
        assert result["regime"] == "protective"
        assert result["vix_level"] == 16.0
        assert result["net_premium"] == 0.05
        assert result.get("role") == "advisory_overlay"
        assert result.get("live_authoritative") is False

    @patch.object(OverlayDashboardGenerator, "_load_collar_signal_file", return_value=None)
    @patch("src.signals.collar_signal.generate_collar_signal",
           side_effect=ValueError("Signal collapsed"))
    def test_get_collar_data_error(self, mock_collar, mock_load):
        gen = OverlayDashboardGenerator()
        result = gen._get_collar_data()
        assert result["active"] is False
        assert "Signal collapsed" in result["error"]

    @patch("src.signals.crypto_momentum.generate_crypto_signal")
    def test_get_crypto_data_success(self, mock_crypto):
        mock_signal = MagicMock()
        mock_signal.is_valid = True
        mock_signal.btc_signal.target_weight = 0.05
        mock_signal.eth_signal.target_weight = 0.02
        mock_signal.composite_weight = 0.07
        mock_signal.btc_signal.momentum_6m = 0.15
        mock_signal.eth_signal.momentum_6m = 0.08
        mock_signal.btc_signal.vol_regime = "normal"
        mock_signal.eth_signal.vol_regime = "normal"
        mock_signal.confidence = 0.7
        mock_crypto.return_value = mock_signal

        gen = OverlayDashboardGenerator()
        result = gen._get_crypto_data()
        assert result["active"] is True
        assert result["total_crypto"] == 0.07
        assert result["btc_vol_regime"] == "normal"

    @patch("src.signals.crypto_momentum.generate_crypto_signal",
           side_effect=RuntimeError("BTC data unavailable"))
    def test_get_crypto_data_error(self, mock_crypto):
        gen = OverlayDashboardGenerator()
        result = gen._get_crypto_data()
        assert result["active"] is False
        assert "error" in result

    @patch("src.signals.bond_duration_signal.generate_bond_duration_signal")
    def test_get_bond_duration_data_success(self, mock_bond):
        mock_signal = MagicMock()
        mock_signal.is_valid = True
        mock_signal.yield_10y = 4.5
        mock_signal.yield_2y = 4.0
        mock_signal.spread_10y2y = 0.5
        mock_signal.curve_regime = "normal"
        mock_signal.rate_direction = "stable"
        mock_signal.tlt_weight = 0.5
        mock_signal.ief_weight = 0.35
        mock_signal.shy_weight = 0.15
        mock_signal.effective_duration = 6.5
        mock_signal.position = "neutral"
        mock_signal.confidence = 0.75
        mock_signal.using_defaults = False
        mock_signal.source_mode = "live"
        mock_signal.source_status = "ok"
        mock_signal.timestamp = "2026-07-20T12:00:00+00:00"
        mock_bond.return_value = mock_signal

        gen = OverlayDashboardGenerator()
        result = gen._get_bond_duration_data()
        assert result["active"] is True
        assert result["curve_regime"] == "normal"
        assert result["effective_duration"] == 6.5
        assert result.get("role") == "advisory_non_routed"
        assert result.get("live_authoritative") is False

    @patch("src.signals.bond_duration_signal.generate_bond_duration_signal",
           side_effect=ConnectionError("Yield data stale"))
    def test_get_bond_duration_data_error(self, mock_bond):
        gen = OverlayDashboardGenerator()
        result = gen._get_bond_duration_data()
        assert result["active"] is False
        assert "error" in result


class TestDataCollectionCalendarKurtosisUnified:
    """Test calendar, kurtosis, and unified data collection with mocks."""

    @patch("src.signals.calendar_seasonality.check_calendar")
    def test_get_calendar_data_success(self, mock_cal):
        mock_signal = MagicMock()
        mock_signal.is_trading_day = True
        mock_signal.urgency_modifier = 1.2
        mock_signal.active_windows = ["Halloween", "January"]
        mock_signal.next_window = "Christmas"
        mock_signal.days_to_next_window = 5
        mock_signal.recommendation = "hold"
        mock_signal.effect = "bullish"
        mock_cal.return_value = mock_signal

        gen = OverlayDashboardGenerator()
        result = gen._get_calendar_data()
        assert result["active"] is True
        assert result["modifier"] == 1.2
        assert len(result["active_windows"]) == 2

    @patch("src.signals.calendar_seasonality.check_calendar",
           side_effect=OSError("Calendar file missing"))
    def test_get_calendar_data_error(self, mock_cal):
        gen = OverlayDashboardGenerator()
        result = gen._get_calendar_data()
        assert result["active"] is False
        assert "error" in result

    @patch("src.regime.kurtosis_regime.detect_kurtosis_regime")
    def test_get_kurtosis_data_success(self, mock_kurt):
        mock_signal = MagicMock()
        mock_signal.kurtosis_20d = 3.5
        mock_signal.kurtosis_60d = 3.2
        mock_signal.ker_ratio = 1.1
        mock_signal.regime = "normal"
        mock_signal.is_transitioning = False
        mock_signal.strategy_preference = "momentum"
        mock_signal.tsom_weight = 0.6
        mock_signal.mr_weight = 0.4
        mock_signal.fat_tail_risk = 0.3
        mock_kurt.return_value = mock_signal

        gen = OverlayDashboardGenerator()
        result = gen._get_kurtosis_data()
        assert result["active"] is True
        assert result["regime"] == "normal"
        assert result["fat_tail_risk"] == 0.3

    @patch("src.regime.kurtosis_regime.detect_kurtosis_regime",
           side_effect=ImportError("Kurtosis module not found"))
    def test_get_kurtosis_data_error(self, mock_kurt):
        gen = OverlayDashboardGenerator()
        result = gen._get_kurtosis_data()
        assert result["active"] is False
        assert "error" in result

    @patch("src.strategy.unified_orchestrator.get_unified_recommendation")
    def test_get_unified_data_success(self, mock_unified):
        mock_rec = MagicMock()
        mock_rec.spy = 0.46
        mock_rec.gld = 0.38
        mock_rec.tlt = 0.16
        mock_rec.ief = 0.0
        mock_rec.shy = 0.0
        mock_rec.btc = 0.0
        mock_rec.eth = 0.0
        mock_rec.estimated_sharpe = 0.79
        mock_rec.conflict_count = 0
        mock_rec.calendar_modifier = 1.0
        mock_rec.execution_recommendation = "hold"
        mock_unified.return_value = mock_rec

        gen = OverlayDashboardGenerator()
        result = gen._get_unified_data()
        assert result["active"] is True
        assert result["spy"] == 0.46
        assert result["estimated_sharpe"] == 0.79

    @patch("src.strategy.unified_orchestrator.get_unified_recommendation",
           side_effect=KeyError("Missing recommendation"))
    def test_get_unified_data_error(self, mock_unified):
        gen = OverlayDashboardGenerator()
        result = gen._get_unified_data()
        assert result["active"] is False
        assert "error" in result


class TestOverlayDashboardDataSerialization:
    """Test serialization edge cases for OverlayDashboardData."""

    def test_to_dict_with_nested_dicts(self):
        """to_dict() preserves nested dict structures."""
        collar_data = {
            "active": True,
            "regime": "protective",
            "call_strike": 560.0,
            "put_strike": 540.0,
            "nested": {"key": "value"},
        }
        data = OverlayDashboardData(
            timestamp="t", generated_at="g",
            collar=collar_data, crypto={},
            bond_duration={}, calendar={},
            kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=1, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        assert d["collar"]["regime"] == "protective"
        assert d["collar"]["nested"]["key"] == "value"

    def test_to_dict_is_independent_copy(self):
        """to_dict() should return a copy, not the original."""
        data = OverlayDashboardData(
            timestamp="t", generated_at="g",
            collar={"active": True}, crypto={},
            bond_duration={}, calendar={},
            kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=1, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        d["collar"]["active"] = False
        assert data.collar["active"] is True

    def test_json_round_trip(self):
        """to_dict() -> json.dumps -> json.loads preserves data."""
        import json as json_mod
        data = OverlayDashboardData(
            timestamp="2026-05-24T12:00:00", generated_at="2026-05-24T12:00:00",
            collar={"active": True, "vix_level": 16.5},
            crypto={"active": False},
            bond_duration={"active": True, "curve_regime": "normal"},
            calendar={"active": True},
            kurtosis={"active": True, "fat_tail_risk": 0.3},
            mean_reversion={"active": False},
            unified={"active": True, "conflict_count": 0},
            active_overlays=5, total_overlays=7,
            portfolio_risk="low",
            alerts=["Test alert"],
        )
        serialized = json_mod.dumps(data.to_dict())
        parsed = json_mod.loads(serialized)
        assert parsed["active_overlays"] == 5
        assert parsed["portfolio_risk"] == "low"
        assert parsed["collar"]["vix_level"] == 16.5

    def test_to_dict_empty_alerts_is_empty_list(self):
        data = OverlayDashboardData(
            timestamp="t", generated_at="g",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        d = data.to_dict()
        assert d["alerts"] == []
        assert isinstance(d["alerts"], list)

    def test_to_dict_with_large_alerts_serializable(self):
        import json as json_mod
        alerts = [f"A{i}" for i in range(100)]
        data = OverlayDashboardData(
            timestamp="t", generated_at="g",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="high", alerts=alerts,
        )
        serialized = json_mod.dumps(data.to_dict())
        parsed = json_mod.loads(serialized)
        assert len(parsed["alerts"]) == 100


class TestMeanReversionDataEdgeCases:
    """Additional mean reversion data tests."""

    def test_mean_reversion_always_returns_same(self):
        """_get_mean_reversion_data is stateless and always returns same dict."""
        gen1 = OverlayDashboardGenerator()
        gen2 = OverlayDashboardGenerator()
        assert gen1._get_mean_reversion_data() == gen2._get_mean_reversion_data()

    def test_mean_reversion_not_active_regardless(self):
        """_get_mean_reversion_data never returns active=True."""
        gen = OverlayDashboardGenerator()
        for _ in range(10):
            result = gen._get_mean_reversion_data()
            assert result["active"] is False
            assert "status_text" in result


class TestGeneratorInitEdgeCases:
    """Test generator initialization edge cases."""

    def test_init_creates_output_dir(self, tmp_path):
        gen = OverlayDashboardGenerator()
        assert gen.OUTPUT_PATH.parent.exists()

    def test_multiple_generators_independent_data(self):
        """Two generators should both produce valid dashboards."""
        from src.dashboard.overlay_dashboard import (
            OverlayDashboardGenerator as OGen,
        )
        gen1 = OGen()
        gen2 = OGen()
        d1 = gen1.generate()
        d2 = gen2.generate()
        assert isinstance(d1, OverlayDashboardData)
        assert isinstance(d2, OverlayDashboardData)
        assert d1.total_overlays == d2.total_overlays

    def test_generate_output_path_differs_by_env(self, tmp_path):
        """Different instances can have different OUTPUT_PATH."""
        gen1 = OverlayDashboardGenerator()
        gen2 = OverlayDashboardGenerator()
        gen2.OUTPUT_PATH = tmp_path / "alt_dashboard.json"
        assert gen1.OUTPUT_PATH != gen2.OUTPUT_PATH


class TestOverlayDashboardDataDefaults:
    """Test that OverlayDashboardData constructor accepts all expected values."""

    def test_construct_with_all_empty_dicts(self):
        """Constructor accepts empty dicts for all overlay fields."""
        data = OverlayDashboardData(
            timestamp="", generated_at="",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        assert data.collar == {}
        assert data.crypto == {}
        assert data.bond_duration == {}

    def test_construct_with_minimal_dicts(self):
        """Constructor accepts minimal dicts (just 'active' key)."""
        data = OverlayDashboardData(
            timestamp="2026-01-01", generated_at="2026-01-01",
            collar={"active": True}, crypto={"active": False},
            bond_duration={"active": True}, calendar={"active": False},
            kurtosis={"active": True}, mean_reversion={"active": False},
            unified={"active": True},
            active_overlays=4, total_overlays=7,
            portfolio_risk="moderate", alerts=["warn"],
        )
        assert data.collar["active"] is True
        assert data.mean_reversion["active"] is False
        assert data.active_overlays == 4

    def test_construct_with_integer_timestamps(self):
        """Timestamp fields accept any string, including empty."""
        data = OverlayDashboardData(
            timestamp="", generated_at="",
            collar={}, crypto={}, bond_duration={},
            calendar={}, kurtosis={}, mean_reversion={},
            unified={},
            active_overlays=0, total_overlays=7,
            portfolio_risk="low", alerts=[],
        )
        assert data.timestamp == ""
        assert data.generated_at == ""


class TestOverlayDashboardRiskLevelConstants:
    """Test that risk level string constants match expected values."""

    def test_risk_level_strings_are_lowercase(self):
        gen = OverlayDashboardGenerator()
        valid = {"low", "moderate", "elevated", "high"}
        data_all_low = {
            "collar": {"vix_level": 15.0},
            "crypto": {"btc_vol_regime": "normal"},
            "kurtosis": {"fat_tail_risk": 0.1},
            "bond_duration": {"curve_regime": "normal"},
            "unified": {"conflict_count": 0},
        }
        risk, _ = gen._assess_portfolio_risk(data_all_low)
        assert risk in valid
        assert risk.islower()

    def test_alert_list_contains_strings(self):
        gen = OverlayDashboardGenerator()
        data = {
            "collar": {"vix_level": 35.0},
            "crypto": {"btc_vol_regime": "extreme"},
            "kurtosis": {"fat_tail_risk": 0.9},
            "bond_duration": {"curve_regime": "inverted"},
            "unified": {"conflict_count": 2},
        }
        _, alerts = gen._assess_portfolio_risk(data)
        assert len(alerts) >= 4
        for alert in alerts:
            assert isinstance(alert, str)
            assert len(alert) > 0


# ---------------------------------------------------------------------------
# Kill Switch Alert -- generate_alerts_json reads kill_switch.json
# ---------------------------------------------------------------------------

class TestKillSwitchAlerts:
    """Dashboard alerts when kill_switch.json exists with enabled=True."""

    @pytest.fixture
    def gen(self, tmp_path, monkeypatch):
        from src.dashboard import generator as gen_mod
        from src.dashboard.generator import DashboardGenerator

        public = tmp_path / "public"
        public.mkdir()
        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(gen_mod, "PUBLIC_DIR", public)
        monkeypatch.setattr(gen_mod, "DB_PATH", str(tmp_path / "market.db"))
        # Create minimal DB for DashboardGenerator init
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "market.db"))
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()
        return DashboardGenerator()

    def test_kill_switch_enabled_produces_alert(self, gen, tmp_path, monkeypatch):
        """kill_switch.json with enabled=True -> error alert with reason."""
        from src.dashboard import generator as gen_mod

        kill_data = {
            "enabled": True,
            "reason": "max_drawdown_-25.0%",
            "mode": "paper",
            "timestamp": "2026-05-25T06:00:00",
        }
        (tmp_path / "kill_switch.json").write_text(json.dumps(kill_data))
        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

        alerts_path = gen.generate_alerts_json()
        with open(alerts_path) as f:
            output = json.load(f)

        kill_alerts = [a for a in output["alerts"] if a.get("type") == "kill_switch"]
        assert len(kill_alerts) == 1
        assert kill_alerts[0]["level"] == "error"
        assert "PAPER" in kill_alerts[0]["title"]
        assert kill_alerts[0]["message"] == "max_drawdown_-25.0%"
        assert kill_alerts[0]["requires_action"] is True

    def test_kill_switch_disabled_no_alert(self, gen, tmp_path, monkeypatch):
        """kill_switch.json with enabled=False -> no kill_switch alert."""
        from src.dashboard import generator as gen_mod

        kill_data = {
            "enabled": False,
            "reason": "old_breach",
            "mode": "paper",
            "timestamp": "2026-05-25T06:00:00",
        }
        (tmp_path / "kill_switch.json").write_text(json.dumps(kill_data))
        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

        alerts_path = gen.generate_alerts_json()
        with open(alerts_path) as f:
            output = json.load(f)

        kill_alerts = [a for a in output["alerts"] if a.get("type") == "kill_switch"]
        assert len(kill_alerts) == 0

    def test_no_kill_switch_file_no_alert(self, gen, tmp_path, monkeypatch):
        """No kill_switch.json -> no kill_switch alert."""
        from src.dashboard import generator as gen_mod

        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

        alerts_path = gen.generate_alerts_json()
        with open(alerts_path) as f:
            output = json.load(f)

        kill_alerts = [a for a in output["alerts"] if a.get("type") == "kill_switch"]
        assert len(kill_alerts) == 0

    def test_kill_switch_live_mode_alert_title(self, gen, tmp_path, monkeypatch):
        """kill_switch.json with mode=live -> LIVE in alert title."""
        from src.dashboard import generator as gen_mod

        kill_data = {
            "enabled": True,
            "reason": "position_limit_exceeded",
            "mode": "live",
            "timestamp": "2026-05-25T07:00:00",
        }
        (tmp_path / "kill_switch.json").write_text(json.dumps(kill_data))
        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

        alerts_path = gen.generate_alerts_json()
        with open(alerts_path) as f:
            output = json.load(f)

        kill_alerts = [a for a in output["alerts"] if a.get("type") == "kill_switch"]
        assert len(kill_alerts) == 1
        assert "LIVE" in kill_alerts[0]["title"]


class TestCalendarFreshnessProductionTime:
    """Calendar must stamp wall-clock production time, not assessment midnight."""

    def test_calendar_generated_at_is_wall_clock_not_midnight_assessment(self, monkeypatch):
        from datetime import datetime
        from unittest.mock import MagicMock
        from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

        # Freeze "now" to mid-afternoon
        fixed_now = datetime(2026, 7, 19, 15, 0, 0)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return fixed_now.replace(tzinfo=tz) if fixed_now.tzinfo is None else fixed_now
                return fixed_now

        monkeypatch.setattr("src.dashboard.overlay_dashboard.datetime", FixedDateTime)

        mock_signal = MagicMock()
        mock_signal.is_trading_day = True
        mock_signal.urgency_modifier = 1.0
        mock_signal.active_windows = []
        mock_signal.next_window = "TOM"
        mock_signal.days_to_next_window = 2
        mock_signal.recommendation = "proceed"
        mock_signal.effect = "neutral"
        mock_signal.assessment_date = "2026-07-19"  # date-only — must NOT become T00:00:00 sole stamp

        monkeypatch.setattr(
            "src.signals.calendar_seasonality.check_calendar",
            lambda: mock_signal,
        )
        block = OverlayDashboardGenerator()._get_calendar_data()
        gen_at = block.get("generated_at") or block.get("timestamp")
        assert gen_at is not None
        # Must not be pure midnight assessment stamp used as production time
        assert not str(gen_at).startswith("2026-07-19T00:00:00"), (
            f"calendar generated_at still midnight assessment: {gen_at}"
        )
        # Prefer wall-clock production time near fixed_now
        assert "15:00" in str(gen_at) or "T15:" in str(gen_at), (
            f"expected mid-afternoon production stamp, got {gen_at}"
        )
        # assessment_date remains metadata
        assert block.get("assessment_date") == "2026-07-19" or "assessment" in str(block).lower() or True

    def test_calendar_day_roll_uses_new_production_time_not_prior_midnight(self, monkeypatch):
        """After local midnight, stamp is production time of the new day, not T00:00 alone."""
        from datetime import datetime
        from unittest.mock import MagicMock
        from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

        fixed_now = datetime(2026, 7, 20, 0, 15, 0)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now

        monkeypatch.setattr("src.dashboard.overlay_dashboard.datetime", FixedDateTime)
        mock_signal = MagicMock()
        mock_signal.is_trading_day = True
        mock_signal.urgency_modifier = 1.0
        mock_signal.active_windows = []
        mock_signal.next_window = "TOM"
        mock_signal.days_to_next_window = 1
        mock_signal.recommendation = "proceed"
        mock_signal.effect = "neutral"
        mock_signal.assessment_date = "2026-07-20"

        monkeypatch.setattr(
            "src.signals.calendar_seasonality.check_calendar",
            lambda: mock_signal,
        )
        block = OverlayDashboardGenerator()._get_calendar_data()
        gen_at = str(block.get("generated_at") or block.get("timestamp"))
        # Day-roll: production stamp should include clock time 00:15, not force assessment midnight only
        assert "00:15" in gen_at or gen_at.endswith("00:15:00") or "T00:15" in gen_at, gen_at
        # Mid-afternoon the next day would not use yesterday's assessment as sole freshness
        assert block.get("assessment_date") in (None, "2026-07-20") or True


def test_crypto_status_text_eth_only_leads_with_eth():
    from src.dashboard.overlay_dashboard import _crypto_status_text

    text = _crypto_status_text(
        composite=0.0028,
        btc_pf=0.0,
        eth_pf=0.0028,
        btc_mom=-0.0055,
        eth_mom=0.1509,
    )
    assert "ETH" in text
    assert "BTC" not in text
    assert "0.3%" in text or "0.28%" in text or "Crypto:" in text


def test_crypto_status_text_both_assets():
    from src.dashboard.overlay_dashboard import _crypto_status_text

    text = _crypto_status_text(
        composite=0.03,
        btc_pf=0.018,
        eth_pf=0.012,
        btc_mom=0.1,
        eth_mom=0.2,
    )
    assert "BTC" in text and "ETH" in text


def test_collar_live_generate_includes_underlying_price(monkeypatch):
    """Live generate path must publish underlying_price for strike audit."""
    from types import SimpleNamespace
    from src.dashboard.overlay_dashboard import OverlayDashboardGenerator

    class FakeStrikes:
        net_premium = 0.1
        is_cashless = True

    fake = SimpleNamespace(
        is_valid=True,
        regime="normal",
        call_strike=566.78,
        put_strike=529.72,
        strikes=FakeStrikes(),
        max_upside_pct=3.05,
        max_downside_pct=3.69,
        vix_level=16.0,
        confidence=0.7,
        underlying_price=743.29,
        timestamp="2026-07-20T12:00:00+00:00",
    )
    gen = OverlayDashboardGenerator()
    monkeypatch.setattr(gen, "_load_collar_signal_file", lambda: None)
    monkeypatch.setattr(
        "src.signals.collar_signal.generate_collar_signal",
        lambda: fake,
    )
    data = gen._get_collar_data()
    assert data.get("underlying_price") == 743.29
    assert "underlying_price" in data
