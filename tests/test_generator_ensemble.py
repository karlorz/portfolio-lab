#!/usr/bin/env python3
"""
Generator ensemble tests — ensemble post-decay metrics class
(TEST-GENERATOR-SPLIT s3, 2026-08-12).

Moved verbatim from tests/test_generator.py (TestEnsemblePostDecayMetrics)
— no tests renamed or weakened. Shared helpers live in tests/helpers.py (plain
module; the autouse fixture below is duplicated verbatim per split file —
never move it to conftest.py, it would pollute the full ~15k-test suite).
"""
from datetime import datetime, timezone

import pytest

from src.dashboard.generator import DashboardGenerator
from tests.helpers import _make_generator


@pytest.fixture(autouse=True)
def _isolate_live_ensemble_and_ic_health(request, monkeypatch):
    """Keep generator tests off live SignalHealthTracker.compute_ic / compute_vote.

    gen.run() and generate_health_json() otherwise call get_health_report() which
    runs hundreds of Spearman IC queries (~15–35s each on lab hosts). That was
    stalling make-test around the TestRun / health-json region (~44%).

    Opt out with @pytest.mark.allow_live_signal_health when a test intentionally
    exercises the real tracker (or already patches get_health_report itself).
    """
    if request.node.get_closest_marker("allow_live_signal_health"):
        yield
        return

    from src.strategy.ensemble_voter import EnsembleVote, Regime

    def _fake_vote(self, *args, **kwargs):
        return EnsembleVote(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=1,
            weighted_consensus=0.1,
            agreement_ratio=0.5,
            equity_bias=0.1,
            duration_bias=0.0,
            gold_bias=0.0,
            action="neutral",
            confidence=0.5,
            reasoning="test-isolation",
            source_votes=[],
        )

    def _fake_bl_views(self, *args, **kwargs):
        from src.strategy.black_litterman_mapper import map_biases_to_views

        views = map_biases_to_views(
            0.1, 0.0, 0.0, health_scores=None, tau=0.15, prior="equal"
        )
        return {
            "views": views,
            "tau": 0.15,
            "prior": "equal",
            "health_scores_used": {},
            "equity_bias": 0.1,
            "duration_bias": 0.0,
            "gold_bias": 0.0,
        }

    def _fake_signal_health_section(**kwargs):
        return {
            "status": "ok",
            "sources": {},
            "summary": {"healthy": 0, "warning": 0, "critical": 0, "total": 0},
            "label_resolve": {"resolved": 0, "pending": 0, "skipped": True},
        }

    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.compute_vote",
        _fake_vote,
        raising=False,
    )
    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.get_bl_views",
        _fake_bl_views,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.signal_health_section.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.generator.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    yield
class TestEnsemblePostDecayMetrics:
    """Ensemble post-decay metrics should share one signed source contract."""

    def test_allocation_surface_roles_disclose_current_live_routing(self, tmp_path):
        roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)

        assert roles["schema_version"] == "allocation-surface-roles/v1"
        assert roles["routed_surface"] == "target_allocations"
        assert roles["surfaces"]["target_allocations"]["routed"] is True
        assert roles["surfaces"]["target_allocations"]["routed_by"] == "src.broker.order_router"
        assert roles["surfaces"]["target_allocations"]["live_authoritative"] is True
        assert roles["surfaces"]["ensemble_voting"]["routed"] is False
        assert roles["surfaces"]["ensemble_voting"]["role"] == "advisory_non_routed"

    def test_allocation_surface_roles_include_standalone_advisory_artifacts(self, tmp_path):
        roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)

        for surface in ("adaptive_sizing", "black_litterman", "calendar_seasonality", "factor_rotation"):
            role = roles["surfaces"][surface]
            assert role["role"] == "advisory_non_routed"
            assert role["routed"] is False
            assert role["routed_by"] is None
            assert role["live_authoritative"] is False
            assert role["canonical_controller"] == "signals.json.target_allocations"
            assert "target_allocations" in role["description"]

        cal = roles["surfaces"]["calendar_seasonality"]
        assert cal.get("applies_to_target_allocations") is False

    def test_advisory_allocation_artifact_role_block_is_machine_readable(self):
        role = DashboardGenerator._build_advisory_allocation_artifact_role(
            surface="black_litterman",
            allocation_field="posterior_weights",
        )

        assert role == {
            "schema_version": "allocation-artifact-role/v1",
            "surface": "black_litterman",
            "allocation_field": "posterior_weights",
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
            "routed_surface_path": "public/data/signals.json#target_allocations",
            "description": (
                "black_litterman is published for advisory diagnostics; live order routing "
                "continues to consume signals.json.target_allocations."
            ),
        }

    def test_black_litterman_public_weights_are_uppercase_with_exclusion_diagnostics(self):
        weights = DashboardGenerator._canonicalize_public_weights(
            {"spy": 0.46, "gld": 0.0, "tlt": 0.16},
            canonical_assets=("SPY", "GLD", "TLT", "IEF"),
        )

        assert weights["weights"] == {"SPY": 0.46, "GLD": 0.0, "TLT": 0.16, "IEF": 0.0}
        assert weights["excluded_assets"] == []
        assert weights["zero_weight_assets"] == ["GLD", "IEF"]

    def test_regime_authority_discloses_live_controller_and_shadow_roles(self):
        authority = DashboardGenerator._build_regime_authority(
            current_regime="vol_spike",
            target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        )

        assert authority["schema_version"] == "regime-authority/v1"
        assert authority["live_controller"] == "signals.json.target_allocations"
        assert authority["live_controller_module"] == "src.broker.order_router"
        assert authority["regime_controller"] == "classify_vix_regime"
        assert authority["regime_routed"] is False
        assert authority["live_regime"] == "vol_spike"
        assert authority["allocation_regime"] == "high_vol"
        assert authority["routed_surface"] == "target_allocations"
        assert authority["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        assert authority["advanced_regime_signals"]["two_stage_regime"]["role"] == "advisory_shadow"
        assert authority["advanced_regime_signals"]["bocd_regime"]["routed"] is False

    def test_regime_allocation_diagnostic_is_advisory_non_routed(self):
        diagnostic = DashboardGenerator._build_regime_allocation_diagnostic("vol_spike")

        assert diagnostic["role"] == "advisory_non_routed"
        assert diagnostic["live_authoritative"] is False
        assert diagnostic["routed"] is False
        assert diagnostic["allocation_regime"] == "high_vol"
        assert diagnostic["candidate_target_allocations"] == {
            "SPY": 0.30,
            "GLD": 0.45,
            "TLT": 0.25,
        }
        assert diagnostic["canonical_controller"] == "signals.json.target_allocations"

    def test_regime_authority_marks_missing_advanced_sections_unpublished(self):
        output = {
            "regime_authority": DashboardGenerator._build_regime_authority(
                current_regime="normal",
                target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ),
            "staleness": {
                "unavailable_signals": ["two_stage_regime", "regime_transition"],
                "stale_signals": [],
            },
        }

        DashboardGenerator._update_regime_authority_availability(output)

        advanced = output["regime_authority"]["advanced_regime_signals"]
        for signal_name in ("two_stage_regime", "regime_transition"):
            entry = advanced[signal_name]
            assert entry["published"] is False
            assert entry["availability"] == "unavailable"
            assert entry["routed"] is False
            assert entry["role"] == "advisory_shadow"
            assert "Published" not in entry["description"]

    def test_regime_authority_marks_present_fresh_advanced_sections_published(self):
        fresh = datetime.now(timezone.utc).isoformat()
        output = {
            "regime_authority": DashboardGenerator._build_regime_authority(
                current_regime="normal",
                target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ),
            "two_stage_regime": {"timestamp": fresh, "regime": "NORMAL"},
            "staleness": {
                "unavailable_signals": [],
                "stale_signals": [],
            },
        }

        DashboardGenerator._update_regime_authority_availability(output)

        entry = output["regime_authority"]["advanced_regime_signals"]["two_stage_regime"]
        assert entry["published"] is True
        assert entry["availability"] == "present"
        assert entry["routed"] is False
        assert entry["role"] == "advisory_shadow"

    def test_source_breakdown_preserves_signed_signal_values(self):
        """Serialized source rows include the signed value used downstream."""
        from src.signals.signal_source import SignalSource
        from src.strategy.ensemble_voter import SignalReading

        rows = DashboardGenerator._build_ensemble_source_breakdown([
            SignalReading(
                source=SignalSource.ALTERNATIVE_DATA,
                timestamp="2026-07-05T00:00:00+00:00",
                value=0.8,
                confidence=0.9,
                weight=0.6,
                regime_fit="normal",
            ),
            SignalReading(
                source=SignalSource.CROSS_ASSET_RV,
                timestamp="2026-07-05T00:00:00+00:00",
                value=-0.4,
                confidence=0.7,
                weight=0.4,
                regime_fit="normal",
            ),
        ])

        assert rows[0]["source"] == "alternative_data"
        assert rows[0]["value"] == pytest.approx(0.8)
        assert rows[1]["source"] == "cross_asset_rv"
        assert rows[1]["value"] == pytest.approx(-0.4)

    def test_vix_source_breakdown_uses_fractional_bridge_confidence(self):
        """VIX source rows publish the normalized typed-bridge confidence."""
        from src.signals.vix_term_structure import VIXTermStructureSignal

        vix_signal = VIXTermStructureSignal(
            timestamp="2026-07-05T00:00:00+00:00",
            signal_state="NEUTRAL",
            signal_value=0.2,
            vix_spot=18.0,
            vix3m=19.5,
            vix6m=20.0,
            slope_vix3m_vix=1.083,
            regime="contango",
            regime_strength=0.5,
            slope_signal=0.3,
            roll_yield_signal=0.08,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.25,
            spy_shift=0.02,
            gld_shift=-0.01,
            tlt_shift=-0.01,
            confidence=90.0,
            is_valid=True,
            reason="VIX=18.00, Slope=1.083, Regime=contango",
        )
        reading = vix_signal.to_signal_snapshot().to_signal_reading()
        reading.weight = 0.05

        rows = DashboardGenerator._build_ensemble_source_breakdown([reading])

        assert rows[0]["source"] == "vix_term_structure"
        assert rows[0]["confidence"] == pytest.approx(0.9)
        assert 0.0 <= rows[0]["confidence"] <= 1.0

    def test_ensemble_adaptive_learning_disclosure_preserves_runtime_status(self):
        disclosure = {
            "bandit": {
                "status": "non_effective",
                "enabled": True,
                "reason": "cold_start_no_regime_weights",
            },
            "online_ic": {
                "status": "disabled",
                "enabled": False,
                "reason": "env_disabled",
            },
        }
        ensemble_result = type("EnsembleResult", (), {"adaptive_learning": disclosure})()

        assert DashboardGenerator._build_ensemble_adaptive_learning_disclosure(ensemble_result) == disclosure

    def test_ensemble_source_count_metadata_distinguishes_configured_collected_and_contributing(self):
        """Source count metadata separates roster, collected rows, and live contributors."""
        source_breakdown = [
            {"source": "alternative_data", "weight": 0.24},
            {"source": "cross_asset_rv", "weight": 0.0},
            {"source": "google_trends", "weight": 0.05},
            {"source": "multi_speed_momentum", "weight": 0.0},
        ]

        counts = DashboardGenerator._build_ensemble_source_count_metadata(
            regime="normal",
            source_breakdown=source_breakdown,
        )

        assert counts["configured_source_count"] == 9
        assert counts["collected_source_count"] == 4
        assert counts["contributing_source_count"] == 2
        assert counts["inactive_source_count"] == 2
        assert counts["inactive_sources"] == ["cross_asset_rv", "multi_speed_momentum"]
        assert counts["num_sources"] == counts["collected_source_count"]


    def test_inactive_count_uses_configured_source_status_missing_stale(self):
        """Headline inactive_* must include missing/stale configured rows."""
        source_breakdown = [
            {"source": "alternative_data", "weight": 0.24},
            {"source": "cross_asset_rv", "weight": 0.10},
        ]
        configured_status = [
            {"source": "alternative_data", "status": "active", "contributing": True},
            {"source": "cross_asset_rv", "status": "active", "contributing": True},
            {"source": "multi_speed_momentum", "status": "missing", "contributing": False},
            {"source": "international_momentum", "status": "missing", "contributing": False},
            {"source": "google_trends", "status": "stale", "contributing": False},
        ]
        counts = DashboardGenerator._build_ensemble_source_count_metadata(
            regime="normal",
            source_breakdown=source_breakdown,
            configured_source_status=configured_status,
        )
        assert counts["inactive_source_count"] == 3
        assert set(counts["inactive_sources"]) == {
            "multi_speed_momentum",
            "international_momentum",
            "google_trends",
        }
        assert counts["contributing_source_count"] == 2
        assert counts["collected_source_count"] == 2

    def test_configured_source_status_discloses_stale_google_trends(self, monkeypatch):
        """Configured source status explains stale Google Trends omission from source rows."""
        from src.signals.signal_snapshot import SignalSnapshot

        class FakeGoogleTrendsSignal:
            def get_signal_snapshot(self):
                return SignalSnapshot(
                    source="google_trends",
                    timestamp="2026-07-05T00:00:00+00:00",
                    value=0.0,
                    confidence=0.0,
                    is_active=False,
                    explanation="Google Trends: Data is 37 days old (max 14)",
                    metadata={
                        "inactive_reason": "Data is 37 days old (max 14)",
                        "inactive_category": "stale",
                    },
                )

        monkeypatch.setattr(
            "src.signals.google_trends_signal.GoogleTrendsSignal",
            FakeGoogleTrendsSignal,
        )

        statuses = DashboardGenerator._build_configured_source_status(
            regime="normal",
            source_breakdown=[{"source": "alternative_data", "weight": 0.24}],
        )

        google_trends = next(status for status in statuses if status["source"] == "google_trends")
        assert google_trends["status"] == "stale"
        assert google_trends["active"] is False
        assert google_trends["collected"] is False
        assert google_trends["configured_weight"] == pytest.approx(0.04762)
        assert google_trends["reason"] == "Data is 37 days old (max 14)"

    def test_marl_status_discloses_controller_runtime_non_routed(self, monkeypatch):
        """MARL status publishes the controller contract without implying routing authority."""
        controller_status = {
            "version": "2.51.0",
            "device": "cpu",
            "agents_loaded": ["analyst", "sentiment", "risk", "execution", "controller"],
            "signal_integrator_connected": False,
            "checkpoint_loaded": False,
            "inference_count": 0,
            "current_allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0},
            "graph_metrics": {"messages_routed": 0},
        }

        class FakeAIController:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get_status(self):
                return controller_status

        monkeypatch.setattr("src.agents.ai_controller.AIController", FakeAIController)

        status = DashboardGenerator._generate_marl_status()

        # Without checkpoint, available is false; module is still importable
        assert status["available"] is False
        assert status.get("module_importable") is True
        assert status.get("reason") == "checkpoint_not_loaded"
        assert status["schema_version"] == "marl-runtime-status/v1"
        assert status["runtime"]["version"] == "2.51.0"
        assert status["runtime"]["agents_loaded"] == controller_status["agents_loaded"]
        assert status["runtime"]["signal_integrator_connected"] is False
        assert status["runtime"]["inference_count"] == 0
        assert status["execution_role"]["routed"] is False
        assert status["execution_role"]["role"] == "research_shadow_non_routed"
        assert status["execution_role"]["routed_by"] is None
        assert "target_allocations" in status["execution_role"]["description"]

    def test_marl_status_available_when_checkpoint_loaded(self, monkeypatch):
        controller_status = {
            "version": "2.51.0",
            "device": "cpu",
            "agents_loaded": ["analyst"],
            "signal_integrator_connected": False,
            "checkpoint_loaded": True,
            "inference_count": 3,
            "current_allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0},
            "graph_metrics": {},
        }

        class FakeAIController:
            def __init__(self, *args, **kwargs):
                pass
            def get_status(self):
                return controller_status

        monkeypatch.setattr("src.agents.ai_controller.AIController", FakeAIController)
        status = DashboardGenerator._generate_marl_status()
        assert status["available"] is True
        assert status.get("module_importable") is True
        assert status["runtime"]["checkpoint_loaded"] is True

    def test_staleness_decay_recomputes_consensus_and_agreement(self, tmp_path):
        """Post-decay consensus and agreement derive from decayed source rows."""
        gen, _ = _make_generator(tmp_path)
        output = {
            "staleness": {
                "staleness_decay": {
                    "alternative_data": 0.1,
                    "ensemble_voting": 1.0,
                },
            },
            "ensemble_voting": {
                "weighted_consensus": 0.0,
                "agreement_ratio": 0.5,
                "source_breakdown": [
                    {
                        "source": "alternative_data",
                        "value": 1.0,
                        "weight": 0.5,
                    },
                    {
                        "source": "cross_asset_rv",
                        "value": -1.0,
                        "weight": 0.5,
                    },
                ],
            },
        }

        try:
            result = gen._apply_staleness_decay(output)
        finally:
            gen.conn.close()

        ensemble = result["ensemble_voting"]
        assert ensemble["total_weight_after_decay"] == pytest.approx(0.55)
        assert ensemble["weighted_consensus"] == pytest.approx(-0.8182)
        assert ensemble["agreement_ratio"] == pytest.approx(0.9091)


# ---------------------------------------------------------------------------
# VIX regime detection tests
# ---------------------------------------------------------------------------

