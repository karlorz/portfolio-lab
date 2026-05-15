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
