#!/usr/bin/env python3
"""
Generator domain tests — incident/broker/yield-curve/garch-cvar/entropy/
constants data classes (TEST-GENERATOR-SPLIT s7, 2026-08-12).

Moved verbatim from tests/test_generator.py (14 domain classes per the
15:20Z table) — no tests renamed or weakened. Shared helpers live in
tests/helpers.py (plain module; the autouse fixture below is duplicated
verbatim per split file — never move it to conftest.py, it would pollute
the full ~15k-test suite).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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

class TestIncidentLifecycleJSON:
    """Test generate_incidents_json."""

    def test_copies_incident_summary_to_public_data(self, tmp_path):
        """Existing incident lifecycle state is published for dashboard fetches."""
        gen, _ = _make_generator(tmp_path)
        source = tmp_path / "incidents.json"
        source.write_text(json.dumps({
            "generated_at": "2026-07-06T00:00:00+00:00",
            "open_count": 1,
            "incidents": [
                {
                    "incident_id": "incident-123",
                    "channel": "signal_staleness",
                    "severity": "p0",
                    "state": "firing",
                    "message": "signals stale",
                    "details": {},
                    "created_at": "2026-07-06T00:00:00+00:00",
                    "updated_at": "2026-07-06T00:00:00+00:00",
                    "resolved_at": None,
                    "resolution_notes": None,
                    "mttr_seconds": None,
                    "alert_count": 6,
                    "kill_switch_level": "halt",
                }
            ],
            "metrics": {
                "incident_frequency": 1,
                "open_count": 1,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            },
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path / "public"), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_incidents_json()

        assert path == tmp_path / "public" / "incidents.json"
        assert json.loads(path.read_text())["incidents"][0]["kill_switch_level"] == "halt"
        gen.conn.close()

    def test_missing_incident_summary_publishes_empty_summary(self, tmp_path):
        """Dashboard core endpoint exists even before the first incident event."""
        gen, _ = _make_generator(tmp_path)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path / "public"), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_incidents_json()

        data = json.loads(path.read_text())
        assert data["open_count"] == 0
        assert data["incidents"] == []
        assert data["metrics"]["open_count"] == 0
        gen.conn.close()

class TestBrokerData:
    """Test _load_broker_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert "connected" in broker
        assert "positions" in broker
        assert "drift" in broker
        assert "kill_switch" in broker
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_detected(self, tmp_path):
        """Kill switch file is detected."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": True}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is True
        gen.conn.close()

    def test_sync_log_detected(self, tmp_path):
        """Position sync log is loaded."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "broker_positions": [{"symbol": "SPY", "qty": 10}],
            "drift": [{"symbol": "SPY", "drift_pct": 0.02}],
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is True
        assert len(broker["positions"]) == 1
        gen.conn.close()

class TestYieldCurve:
    """Test _get_yield_curve_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._get_yield_curve_data()
        assert "yield_curve" in data or "duration_allocation" in data
        gen.conn.close()

    def test_yield_curve_includes_yields_source_manifest_provenance(self, tmp_path):
        """Synthetic/degraded yields provenance follows the yield curve payload."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([
            {"spread2s10s": -10.0, "dgs2": 4.6, "dgs10": 4.5},
        ]))
        (tmp_path / "source_manifest.json").write_text(json.dumps({
            "artifacts": [
                {
                    "artifact": "yields.json",
                    "provider": "FRED",
                    "source_mode": "synthetic",
                    "status": "degraded",
                    "failure_reason": "FRED_API_KEY missing",
                    "generated_at": "2026-07-06T00:00:00Z",
                    "latest_observation": "2026-07-02",
                },
            ],
        }))

        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
                data = gen._get_yield_curve_data()

        assert data["yield_curve"]["source_mode"] == "synthetic"
        assert data["yield_curve"]["source_status"] == "degraded"
        assert data["yield_curve"]["source_reason"] == "FRED_API_KEY missing"
        assert data["yield_curve"]["source_provider"] == "FRED"
        assert data["yield_curve"]["source_latest_observation"] == "2026-07-02"
        gen.conn.close()

class TestConstants:
    """Test module-level constants."""

    def test_base_allocation_has_symbols(self):
        """BASE_ALLOCATION contains SPY, GLD, TLT."""
        from src.paths import BASE_ALLOCATION
        assert "SPY" in BASE_ALLOCATION
        assert "GLD" in BASE_ALLOCATION
        assert "TLT" in BASE_ALLOCATION

    def test_public_dir_is_path_instance(self):
        """PUBLIC_DIR is a Path instance."""
        from src.dashboard.generator import PUBLIC_DIR
        assert isinstance(PUBLIC_DIR, Path)

    def test_base_allocation_weights_sum_to_one(self):
        """BASE_ALLOCATION weights sum to 1.0."""
        from src.paths import BASE_ALLOCATION
        total = sum(BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

class TestGarchCvarData:
    """Test _load_garch_cvar_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Returns expected defaults when no health file exists."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["cvar_95_garch"] == -0.0215
        assert data["var_95"] == -0.0127
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_conformal_coverage_diagnostics_are_optional_monitoring_metadata(self, tmp_path):
        """GARCH-CVaR payload includes optional conformal coverage diagnostics."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()

        diagnostics = data["coverage_diagnostics"]
        assert diagnostics["schema_version"] == "conformal-coverage/v1"
        assert diagnostics["alpha"] == pytest.approx(0.05)
        assert diagnostics["observations"] >= 21
        assert "kupiec_pass" in diagnostics
        assert "christoffersen_pass" in diagnostics
        assert "conditional_coverage_pass" in diagnostics
        gen.conn.close()

    def test_flat_format_normalizes_percentages(self, tmp_path):
        """Values >1 in health report are divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 2.5,
            "var_95": 1.8,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.2,
            "garch_persistence": 0.97,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.025
        assert data["var_95"] == pytest.approx(0.018)
        assert data["garch_active"] is True
        gen.conn.close()

    def test_flat_format_keeps_decimal_values(self, tmp_path):
        """Values <=1 are kept as-is."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": -0.0179,
            "var_95": -0.0127,
            "cvar_ratio": 1.51,
            "filter_active": True,
            "conditional_volatility_current": 1.5,
            "garch_persistence": 0.88,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["var_95"] == -0.0127
        gen.conn.close()

    def test_legacy_nested_format(self, tmp_path):
        """Parses legacy nested check format from health report."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "cvar_metrics": {
                    "garch_filtered": True,
                    "cvar_95": -0.0250,
                    "var_95": -0.0150,
                    "cvar_ratio": 1.75,
                    "garch_active": True,
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0250
        assert data["cvar_ratio"] == 1.75
        gen.conn.close()

    def test_volatility_clustering_boundaries(self, tmp_path):
        """Tests all persistence thresholds for clustering label."""
        gen, _ = _make_generator(tmp_path)
        for persistence, expected in [(0.96, "high"), (0.90, "elevated"), (0.80, "normal")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "var_95": -0.0127,
                "cvar_ratio": 1.51,
                "filter_active": True,
                "conditional_volatility_current": 1.5,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}"
            )
        gen.conn.close()

class TestEntropyData:
    """Test _load_entropy_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Without health metrics, correlation axes are null (not fake 0.95/2.5)."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] is None
        assert data["effective_n"] is None
        assert data["concentration_risk"] == "unknown"
        assert data["hhi_index"] is None
        assert data["correlation_entropy"] is None
        assert data["participation_ratio"] is None
        assert data["correlation_metrics_status"] == "unavailable"
        gen.conn.close()

    def test_concentration_risk_all_levels(self, tmp_path):
        """All score thresholds map to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        for score, expected in [(92, "good"), (80, "low"), (60, "medium"),
                                 (40, "high"), (20, "critical")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.02,
                            "effective_n": 2.77,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}"
            )
        gen.conn.close()

    def test_loads_from_health_file_metrics(self, tmp_path):
        """All fields are populated from health file metrics."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {
                    "metrics": {
                        "shannon_entropy": 0.85,
                        "effective_n": 2.1,
                        "normalized_score": 77.0,
                        "hhi_index": 0.45,
                    }
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 0.85
        assert data["effective_n"] == 2.1
        assert data["normalized_score"] == 77.0
        assert data["hhi_index"] == 0.45
        # H_max derived when producer omits max_possible (ln(n) for champion book)
        assert data["max_possible"] is not None
        assert data["max_possible"] > 0
        gen.conn.close()

    def test_missing_metrics_section(self, tmp_path):
        """Missing metrics section stays partial/unavailable (no invented numbers)."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {}
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] is None
        assert data["effective_n"] is None
        assert data["correlation_metrics_status"] == "unavailable"
        gen.conn.close()

class TestYieldCurveEdgeCases:
    """Test _get_yield_curve_data edge cases."""

    def test_no_file_returns_empty(self, tmp_path):
        """Returns empty result when yields file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.YIELDS_JSON", tmp_path / "nonexistent.json"):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        assert data["duration_allocation"] is None
        gen.conn.close()

    def test_empty_list_returns_empty(self, tmp_path):
        """Empty yields list returns empty result."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("[]")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        gen.conn.close()

    def test_spread_classification_steep(self, tmp_path):
        """Spread > 100 bps classifies as steep."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": 150, "dgs2": 4.0, "dgs10": 5.5} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "steep"
        gen.conn.close()

    def test_spread_classification_inverted(self, tmp_path):
        """Spread <= 0 bps classifies as inverted."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": -25, "dgs2": 5.0, "dgs10": 4.75} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_spread_boundary_values(self, tmp_path):
        """Boundary spread values map to correct regimes."""
        gen, _ = _make_generator(tmp_path)
        cases = [(100, "normal"), (50, "flat"), (1, "flat"), (0, "inverted")]
        for spread, expected in cases:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            assert data["yield_curve"]["duration_regime"] == expected, (
                f"Spread {spread} should be {expected}, got {data['yield_curve']['duration_regime']}"
            )
        gen.conn.close()

    def test_duration_allocation_by_regime(self, tmp_path):
        """Each regime maps to correct duration allocation."""
        gen, _ = _make_generator(tmp_path)
        allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, spread_val in [("steep", 150), ("inverted", -25)]:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread_val, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            expected = allocations[regime]
            for k, v in expected.items():
                assert data["duration_allocation"][k] == v, (
                    f"Regime {regime}: {k} expected {v}, got {data['duration_allocation'][k]}"
                )
        gen.conn.close()

    def test_spread_history_length(self, tmp_path):
        """Spread history contains up to 30 entries."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": i * 5, "dgs2": 4.0, "dgs10": 5.0} for i in range(40)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert len(data["yield_curve"]["spread_history"]) == 30
        gen.conn.close()

class TestBrokerDataEdgeCases:
    """Test _load_broker_data edge cases."""

    def test_empty_sync_log(self, tmp_path):
        """Empty sync log file returns default structure."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is False
        gen.conn.close()

    def test_malformed_sync_log(self, tmp_path):
        """Malformed JSON in sync log is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("not valid json\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, broker stays in default state
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_disabled(self, tmp_path):
        """Kill switch with enabled=False reports no kill switch."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": False}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()

    def test_broker_orders_loaded(self, tmp_path):
        """Broker orders log is loaded into recent_orders."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text(
            json.dumps({"order_id": 1, "symbol": "SPY", "side": "buy", "qty": 10})
            + "\n"
            + json.dumps({"order_id": 2, "symbol": "GLD", "side": "sell", "qty": 5})
            + "\n"
        )
        # Also need sync log so connected=True
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert len(broker["recent_orders"]) == 2
        assert broker["recent_orders"][0]["order_id"] == 2  # Reversed order
        gen.conn.close()

    def test_malformed_orders_line(self, tmp_path):
        """Malformed line in broker orders is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text("not valid json\n")
        # Also need sync log
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, default recent_orders returned
        assert broker["recent_orders"] == []
        gen.conn.close()

    def test_malformed_kill_switch(self, tmp_path):
        """Malformed kill_switch.json returns default state."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text("not valid json")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()

class TestGarchCvarEdgeCases:
    """Additional _load_garch_cvar_data edge cases."""

    def test_flat_format_empty_dict(self, tmp_path):
        """Empty health report file returns all defaults."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_flat_format_zero_values(self, tmp_path):
        """Zero values in health report are handled correctly."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 0.0,
            "var_95": 0.0,
            "cvar_ratio": 0.0,
            "filter_active": False,
            "conditional_volatility_current": 0.0,
            "garch_persistence": 0.0,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0
        assert data["var_95"] == 0.0
        assert data["cvar_ratio"] == 0.0
        assert data["garch_active"] is False
        assert data["volatility_clustering"] == "normal"
        gen.conn.close()

class TestEntropyEdgeCases:
    """Additional _load_entropy_data edge cases."""

    def test_concentration_risk_exact_boundaries(self, tmp_path):
        """Normalized score at exact boundaries maps to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        boundaries = [(91, "good"), (71, "low"), (51, "medium"), (31, "high"), (0, "critical")]
        for score, expected in boundaries:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.0,
                            "effective_n": 2.5,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}, got {data['concentration_risk']}"
            )
        gen.conn.close()

    def test_empty_health_file_returns_defaults(self, tmp_path):
        """Empty JSON health file stays partial/unavailable."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] is None
        assert data["concentration_risk"] == "unknown"
        gen.conn.close()

class TestConstantsAdditional:
    """Additional module-level constant validation."""

    def test_data_dir_is_path_instance(self):
        """DATA_DIR is a Path instance."""
        from src.dashboard.generator import DATA_DIR
        assert isinstance(DATA_DIR, Path)

    def test_db_path_is_path_instance(self):
        """DB_PATH is a Path instance."""
        from src.dashboard.generator import DB_PATH
        assert isinstance(DB_PATH, Path)

class TestYieldCurveMalformed:
    """Test _get_yield_curve_data with malformed data."""

    def test_missing_spread_key(self, tmp_path):
        """Missing spread2s10s key defaults to 0 spread."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["spread2s10s"] == 0
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_none_spread_entries_skipped(self, tmp_path):
        """None values in spread entries are excluded from spread_history."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        entries = []
        for i in range(35):
            if i % 3 == 0:
                entries.append({"spread2s10s": None, "dgs2": 4.0, "dgs10": 5.0})
            else:
                entries.append({"spread2s10s": i * 3, "dgs2": 4.0, "dgs10": 5.0})
        yields_path.write_text(json.dumps(entries))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # None entries should be excluded from spread_history
        assert None not in data["yield_curve"]["spread_history"]
        gen.conn.close()

    def test_yields_file_empty_json_object(self, tmp_path):
        """Non-list JSON in yields file is handled gracefully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("{}")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # Should return empty result
        assert data["yield_curve"] is None
        gen.conn.close()

class TestGarchCvarEdgeCasesExtended:
    """Additional _load_garch_cvar_data edge cases — boundary values."""

    def test_value_exactly_one_not_divided(self, tmp_path):
        """Value exactly 1.0 is kept as-is (not divided by 100 because abs(1) <= 1)."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.0,
            "var_95": 1.0,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.0,
            "garch_persistence": 0.9,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        # abs(1.0) <= 1, so no division
        assert data["cvar_95"] == 1.0
        assert data["var_95"] == 1.0
        gen.conn.close()

    def test_value_slightly_above_one_divided(self, tmp_path):
        """Value 1.01 (> 1.0) is divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.01,
            "var_95": 1.01,
            "filter_active": True,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0101
        assert data["var_95"] == 0.0101
        gen.conn.close()

    def test_persistence_at_exact_boundaries(self, tmp_path):
        """GARCH persistence at exact boundary values."""
        gen, _ = _make_generator(tmp_path)
        # boundary cases: 0.85 -> elevated, 0.95 -> high, 0.951 -> high
        for persistence, expected in [
            # Code uses > 0.85 and > 0.95 (strict), not >=
            (0.85, "normal"),
            (0.86, "elevated"),
            (0.94, "elevated"),
            (0.95, "elevated"),
            (0.951, "high"),
            (0.80, "normal"),
            (0.84, "normal"),
        ]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "filter_active": True,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}, got {data['volatility_clustering']}"
            )
        gen.conn.close()

class TestConstantsExtended:
    """Extended module-level constant validation."""

    def test_logger_is_logger_instance(self):
        """logger is a Logger instance."""
        import logging
        from src.dashboard.generator import logger
        assert isinstance(logger, logging.Logger)

    def test_base_allocation_keys_are_uppercase(self):
        """BASE_ALLOCATION keys are uppercase symbol names."""
        from src.paths import BASE_ALLOCATION
        for key in BASE_ALLOCATION:
            assert key == key.upper(), f"Key '{key}' should be uppercase"

    def test_regime_overrides_keys_match(self):
        """Regime override keys correspond to valid regimes."""
        from src.dashboard.generator import DashboardGenerator
        gen = DashboardGenerator.__new__(DashboardGenerator)
        # Extract the regime_overrides from generate_signals_json logic
        expected_regimes = {"crisis", "vol_spike", "low_vol"}
        overrides = {
            "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
            "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},
            "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},
        }
        assert set(overrides.keys()) == expected_regimes
        for regime, alloc in overrides.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Regime '{regime}' allocations sum to {total}, expected 1.0"
            )
            for sym in alloc:
                assert isinstance(alloc[sym], float)

    def test_yield_regime_allocations_sum_to_one(self):
        """Each yield regime allocation sums to ~1.0."""
        regime_allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
            "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, alloc in regime_allocations.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Yield regime '{regime}' allocations sum to {total}, expected 1.0"
            )

    def test_public_dir_exists_after_creation(self):
        """PUBLIC_DIR is a Path pointing to an existing or creatable directory."""
        from src.dashboard.generator import PUBLIC_DIR
        # This test just validates the constant, directory creation is tested in init
        import os
        parent = PUBLIC_DIR.parent
        assert parent.exists(), f"Parent dir {parent} should exist"

