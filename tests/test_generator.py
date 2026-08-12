#!/usr/bin/env python3
"""
Tests for dashboard generator — VIX regime detection, data freshness,
health status, alerts, broker data, and stats calculation.
"""
import json
import inspect
import sqlite3
import sys
import types
import numpy as np

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.dashboard.generator import DashboardGenerator, DATA_DIR, PUBLIC_DIR, DB_PATH


# ---------------------------------------------------------------------------
# Helpers — moved verbatim to tests/helpers.py (TEST-GENERATOR-SPLIT);
# the autouse _isolate_live_ensemble_and_ic_health fixture stays HERE and is
# duplicated verbatim into each split file (never conftest.py).
# ---------------------------------------------------------------------------
from tests.helpers import (  # noqa: E402
    _create_market_db,
    _make_generator,
    _write_data_quality_report,
    _write_ok_source_manifest,
)


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



# ---------------------------------------------------------------------------
# Data freshness tests
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """Test data freshness classification."""

    def _classify_freshness(self, days_stale):
        """Extract freshness classification logic."""
        if days_stale <= 1:
            return "fresh"
        elif days_stale <= 3:
            return "stale"
        else:
            return "critical"

    def test_fresh(self):
        assert self._classify_freshness(0) == "fresh"
        assert self._classify_freshness(1) == "fresh"

    def test_stale(self):
        assert self._classify_freshness(2) == "stale"
        assert self._classify_freshness(3) == "stale"

    def test_critical(self):
        assert self._classify_freshness(4) == "critical"
        assert self._classify_freshness(30) == "critical"


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------

class TestHealthStatus:
    """Test system health status determination."""

    def _determine_health(self, failed_jobs, stale_count):
        """Extract health status logic."""
        status = "healthy"
        if failed_jobs > 0 or stale_count > 5:
            status = "warning"
        if failed_jobs > 2 or stale_count > 10:
            status = "critical"
        return status

    def test_healthy(self):
        assert self._determine_health(0, 0) == "healthy"
        assert self._determine_health(0, 5) == "healthy"

    def test_warning(self):
        assert self._determine_health(1, 0) == "warning"
        assert self._determine_health(0, 6) == "warning"

    def test_critical(self):
        assert self._determine_health(3, 0) == "critical"
        assert self._determine_health(0, 11) == "critical"

    def test_critical_overrides_warning(self):
        """Critical takes precedence when both conditions met."""
        assert self._determine_health(3, 11) == "critical"


# ---------------------------------------------------------------------------
# Generator initialization tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Cross-asset relative value JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Performance JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Stats JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Alerts JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Health JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Incident lifecycle JSON tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Broker data tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ML signals tests
# ---------------------------------------------------------------------------

class TestMLSignals:
    """Test _generate_ml_signals."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "available" in signals
        assert signals["available"] is False
        gen.conn.close()

    def test_features_file_detected(self, tmp_path):
        """Features file makes signals available."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "timestamp": datetime.now().isoformat(),
            "momentum_12m": 0.15, "volatility": 0.18,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()

    def test_stale_feature_rows_publish_source_freshness_metadata(self, tmp_path):
        """Feature predictions expose feature as-of time and stale status."""
        gen, _ = _make_generator(tmp_path)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY",
            "timestamp": old_timestamp,
            "vix_level": 15,
            "trend_direction": 1,
            "price_vs_sma20": 0.05,
            "return_5d": 0.01,
            "spy_correlation_20d": 0.4,
        }) + "\n")

        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()

        assert signals["available"] is True
        assert signals["timestamp"] is not None
        assert signals["generated_at"] == signals["timestamp"]
        assert signals["feature_source_artifact"] == "features.jsonl"
        assert signals["feature_as_of"] == old_timestamp
        assert signals["feature_freshness_status"] == "stale"
        assert signals["feature_staleness_days"] >= 30
        assert signals["prediction_source_mode"] == "stale_features"
        assert signals["predictions"]["SPY"]["feature_timestamp"] == old_timestamp
        assert signals["predictions"]["SPY"]["feature_freshness_status"] == "stale"
        assert signals["predictions"]["SPY"]["source_artifact"] == "features.jsonl"
        assert signals["execution_role"]["routed"] is False
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Run integration test
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GARCH-CVaR data edge cases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entropy data edge cases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ML signals edge cases
# ---------------------------------------------------------------------------

class TestMlSignalsEdgeCases:
    """Test _generate_ml_signals edge cases."""

    def test_no_features_file(self, tmp_path):
        """Returns available=False when features file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        assert "features" in signals
        assert "predictions" in signals
        gen.conn.close()

    def test_empty_features_file(self, tmp_path):
        """Empty features file returns available=False."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        gen.conn.close()

    def test_malformed_line_skipped(self, tmp_path):
        """Malformed JSON line in features file is skipped."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            "not valid json\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "timestamp": "2026-01-01"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()

    def test_vix_crisis_predictions(self, tmp_path):
        """VIX >25 yields bearish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 30, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bear"
        assert pred["confidence"] == 0.5
        assert pred["probabilities"]["bear"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_trend_bull_predictions(self, tmp_path):
        """Positive trend and price above SMA yields bullish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 1,
            "price_vs_sma20": 0.05, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bull"
        assert pred["probabilities"]["bull"] == 0.6
        gen.conn.close()

    def test_trend_bear_predictions(self, tmp_path):
        """Negative trend yields bearish-leaning prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": -1,
            "price_vs_sma20": -0.03, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.4
        assert pred["probabilities"]["neutral"] == 0.4
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_default_predictions(self, tmp_path):
        """Default probabilities when vix <=20 and no trend signal."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_grid_search_results(self, tmp_path):
        """Grid search results loaded from JSONL file."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "allocations": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
            "sharpe": 0.85,
            "volatility": 0.12,
        }) + "\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        gs = signals["grid_search"]
        assert gs["available"] is True
        assert gs["sharpe"] == 0.85
        assert gs["top_allocation"] == {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        gen.conn.close()

    def test_grid_search_results_publish_frozen_benchmark_semantics(self, tmp_path):
        """Grid-search metrics disclose source artifact and frozen benchmark status."""
        gen, _ = _make_generator(tmp_path)
        grid_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text(json.dumps({
            "timestamp": grid_timestamp,
            "allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "sharpe": 0.95,
            "volatility": 0.11,
        }) + "\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY",
            "vix_level": 15,
            "trend_direction": 0,
            "price_vs_sma20": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()

        gs = signals["grid_search"]
        assert gs["available"] is True
        assert gs["source_artifact"] == "grid_search_results.jsonl"
        assert gs["benchmark_timestamp"] == grid_timestamp
        assert gs["observation_semantics"] == "frozen_benchmark_not_live_snapshot"
        assert gs["freshness_status"] == "frozen_benchmark"
        assert gs["staleness_days"] >= 45
        assert gs["live_authoritative"] is False
        gen.conn.close()

    def test_multi_symbol_features(self, tmp_path):
        """Multiple symbols produce separate predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
            + json.dumps({"symbol": "GLD", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.02, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "SPY" in signals["predictions"]
        assert "GLD" in signals["predictions"]
        assert signals["predictions"]["SPY"]["predicted_regime"] == "bear"
        assert signals["predictions"]["GLD"]["predicted_regime"] == "bull"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve edge cases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Broker data edge cases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Performance JSON edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Signals JSON edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Health JSON edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Stats JSON edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Alerts JSON edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Run edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Analytics JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Graduation JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Overlay JSON tests
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# GARCH-CVaR edge cases (continued)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entropy edge cases (continued)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Signals JSON edge cases (continued)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ML signals edge cases (continued)
# ---------------------------------------------------------------------------

class TestMlSignalsAdditionalEdgeCases:
    """Additional _generate_ml_signals edge cases."""

    def test_grid_search_empty_file(self, tmp_path):
        """Empty grid search results file returns empty grid_search dict."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["grid_search"] == {}
        gen.conn.close()

    def test_vix_at_vol_spike_boundary_ml_prediction(self, tmp_path):
        """VIX at 21 (just above 20) triggers vol_spike probability distribution."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 21, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3
        assert pred["probabilities"]["neutral"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON edge cases (continued)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Constants validation (continued)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sector momentum signals tests (completely untested method)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Signals JSON — regime composite integration tests
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Signals JSON — positions, orders, and paper portfolio state
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Signals JSON — smart rebalance
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Signals JSON — alternative data
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Stats JSON — paper portfolio and SPY comparison (core logic, untested)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Health JSON — signal health testing
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Performance JSON — regime data and paper portfolio
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Explainability JSON freshness
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Yield curve — missing keys and malformed data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generator init — edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Run — overlay and signals edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ML signals — edge cases continued
# ---------------------------------------------------------------------------

class TestMlSignalsGridSearch:
    """Test ML signals grid search edge cases."""

    def test_grid_search_malformed_line(self, tmp_path):
        """Malformed line in grid search file is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("not valid json\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["grid_search"] == {}
        gen.conn.close()

    def test_multiple_features_keeps_latest(self, tmp_path):
        """Multiple entries for same symbol keep the latest by timestamp."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"}) + "\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.05, "timestamp": "2026-01-02T00:00:00"}) + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["features"]["SPY"]["vix_level"] == 15  # Latest
        gen.conn.close()

    def test_missing_vix_in_features(self, tmp_path):
        """Features missing vix_level key defaults to 20 in predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # Default probabilities: vix <=20 and no trend
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()


# ---------------------------------------------------------------------------
# VIX regime detection — boundary values at exact thresholds
# ---------------------------------------------------------------------------

# Graduation JSON — additional edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------

class TestAllExports:
    """Test __all__ exports match module's public API."""

    def test_all_defined(self):
        """__all__ is defined and contains expected names."""
        from src.dashboard.generator import __all__
        assert isinstance(__all__, list)
        assert "DashboardGenerator" in __all__
        assert "PUBLIC_DIR" in __all__
        assert "DB_PATH" in __all__
        assert len(__all__) >= 3

    def test_all_names_importable(self):
        """Every name in __all__ can be imported from the module."""
        import src.dashboard.generator as gen_mod
        from src.dashboard.generator import __all__

        for name in __all__:
            assert hasattr(gen_mod, name), f"{name} missing from module"
            assert getattr(gen_mod, name) is not None, f"{name} should not be None"


# ---------------------------------------------------------------------------
# Extended field type / dataclass validation
# ---------------------------------------------------------------------------

class TestOutputFieldTypes:
    """Validate field types in all generated JSON outputs."""

    def test_signals_json_field_types(self, tmp_path):
        """All signals.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)

        # Top-level scalar fields
        assert isinstance(data["generated_at"], str), "generated_at should be str"
        assert isinstance(data["cash"], (int, float)), "cash should be numeric"
        assert isinstance(data["total_value"], (int, float)), "total_value should be numeric"

        # Regime section
        regime = data["regime"]
        assert isinstance(regime, dict)
        assert isinstance(regime["regime"], str)
        assert regime["vix"] is None or isinstance(regime["vix"], (int, float))
        assert regime["detected"] is None or isinstance(regime["detected"], str)

        # Target allocations
        target = data["target_allocations"]
        assert isinstance(target, dict)
        for sym, weight in target.items():
            assert isinstance(sym, str)
            assert isinstance(weight, (int, float))
            assert 0 <= weight <= 1.0, f"Weight {weight} out of range [0, 1]"

        # Latest prices
        prices = data["latest_prices"]
        assert isinstance(prices, dict)
        for sym, price in prices.items():
            assert isinstance(sym, str)
            assert isinstance(price, (int, float)), f"Price for {sym} should be numeric"

        # Positions
        assert isinstance(data["current_positions"], list)
        for pos in data["current_positions"]:
            assert isinstance(pos["symbol"], str)
            assert isinstance(pos["shares"], (int, float))
            assert isinstance(pos["value"], (int, float))
            assert isinstance(pos["weight"], (int, float))
            assert isinstance(pos["unrealized"], (int, float))

        # ML signals
        ml = data["ml_signals"]
        assert isinstance(ml, dict)
        assert isinstance(ml["available"], bool)
        assert ml["timestamp"] is None or isinstance(ml["timestamp"], str)
        assert isinstance(ml["predictions"], dict)
        assert isinstance(ml["features"], dict)
        assert isinstance(ml["grid_search"], dict)
        marl = data["marl_status"]
        assert isinstance(marl, dict)
        assert marl["schema_version"] == "marl-runtime-status/v1"
        assert isinstance(marl["available"], bool)
        assert isinstance(marl["runtime"], dict)
        assert marl["execution_role"]["routed"] is False
        assert marl["execution_role"]["role"] == "research_shadow_non_routed"
        gen.conn.close()

    def test_health_json_field_types(self, tmp_path):
        """All health.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["system_status"], str)
        assert data["system_status"] in ("healthy", "warning", "critical", "degraded")
        assert isinstance(data["cron_jobs"], list)
        assert isinstance(data["data_freshness"], dict)
        assert isinstance(data["signal_health"], dict)
        assert isinstance(data["generated_at"], str)

        for sym, freshness in data["data_freshness"].items():
            assert isinstance(sym, str)
            assert isinstance(freshness, dict)
            assert "last_update" in freshness
            assert "days_stale" in freshness
            assert "status" in freshness
            assert freshness["status"] in ("fresh", "stale", "critical")
            assert isinstance(freshness["days_stale"], int)

        for job in data["cron_jobs"]:
            assert isinstance(job["name"], str)
            assert isinstance(job["status"], str)
        gen.conn.close()

    def test_stats_json_field_types(self, tmp_path):
        """All stats.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["generated_at"], str)
        assert isinstance(data["asset_stats"], dict)

        for sym, stat in data["asset_stats"].items():
            assert isinstance(sym, str)
            assert isinstance(stat["30d_return"], (int, float))
            assert isinstance(stat["volatility"], (int, float))
            assert isinstance(stat["current"], (int, float))

        assert isinstance(data["paper_portfolio"], dict)
        assert data["spy_comparison"] is None or isinstance(data["spy_comparison"], dict)
        if data.get("spy_comparison"):
            sc = data["spy_comparison"]
            for key in ("portfolio_value", "spy_value", "relative_return", "correlation_30d", "beta"):
                assert key in sc, f"spy_comparison missing '{key}'"
        gen.conn.close()

    def test_alerts_json_field_types(self, tmp_path):
        """All alerts.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["alerts"], list)
        assert isinstance(data["count"], int)
        assert isinstance(data["generated_at"], str)

        for alert in data["alerts"]:
            assert isinstance(alert["level"], str)
            assert isinstance(alert["type"], str)
            assert isinstance(alert["title"], str)
            assert isinstance(alert["message"], str)
            assert isinstance(alert["requires_action"], bool)
            assert alert["level"] in ("success", "warning", "error", "info")
        gen.conn.close()

    def test_broker_data_field_types(self, tmp_path):
        """_load_broker_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert isinstance(broker["connected"], bool)
        assert isinstance(broker["positions"], list)
        assert isinstance(broker["drift"], list)
        assert isinstance(broker["recent_orders"], list)
        assert broker["last_sync"] is None or isinstance(broker["last_sync"], str)
        assert isinstance(broker["kill_switch"], bool)
        gen.conn.close()

    def test_garch_cvar_field_types(self, tmp_path):
        """_load_garch_cvar_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            garch = gen._load_garch_cvar_data()
        assert isinstance(garch["cvar_95"], (int, float))
        assert isinstance(garch["cvar_95_garch"], (int, float))
        assert isinstance(garch["var_95"], (int, float))
        assert isinstance(garch["var_95_garch"], (int, float))
        assert isinstance(garch["cvar_ratio"], (int, float))
        assert isinstance(garch["garch_active"], bool)
        assert isinstance(garch["current_volatility"], (int, float))
        assert isinstance(garch["forecast_volatility"], (int, float))
        assert isinstance(garch["volatility_clustering"], str)
        assert garch["volatility_clustering"] in ("normal", "elevated", "high")
        gen.conn.close()

    def test_entropy_data_field_types(self, tmp_path):
        """_load_entropy_data dict has correct field types (nulls allowed when uncomputed)."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            entropy = gen._load_entropy_data()
        for key in (
            "shannon_entropy",
            "effective_n",
            "max_possible",
            "normalized_score",
            "hhi_index",
            "correlation_entropy",
            "participation_ratio",
        ):
            assert entropy[key] is None or isinstance(entropy[key], (int, float))
        assert isinstance(entropy["concentration_risk"], str)
        assert entropy["concentration_risk"] in (
            "good", "low", "medium", "high", "critical", "unknown",
        )
        assert entropy["correlation_metrics_status"] in ("ok", "unavailable", "partial")
        gen.conn.close()


# ---------------------------------------------------------------------------
# Additional computation edge cases — boundary values, zero/negative, large
# ---------------------------------------------------------------------------

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




class TestMLSignalsEdgeCasesExtended:
    """Additional _generate_ml_signals prediction edge cases."""

    def test_vix_exactly_25_classification(self, tmp_path):
        """VIX exactly 25 falls into vol_spike branch (>20, not >25)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 25, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3  # vol_spike probs
        assert pred["probabilities"]["neutral"] == 0.5
        gen.conn.close()

    def test_vix_exactly_20_classification(self, tmp_path):
        """VIX exactly 20 falls into normal branch (not >20, not <15)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 20, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["neutral"] == 0.6
        gen.conn.close()

    def test_vix_extremely_high(self, tmp_path):
        """Very high VIX (e.g., 80) does not crash."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 80, "trend_direction": -5,
            "price_vs_sma20": -0.2, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.5  # crisis probs dominate
        assert pred["predicted_regime"] == "bear"
        gen.conn.close()

    def test_trend_direction_zero_price_vs_sma_zero(self, tmp_path):
        """trend_direction=0 and price_vs_sma20=0 with vix <=20 defaults."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_missing_trend_fields_defaults(self, tmp_path):
        """Missing trend_direction and price_vs_sma20 fields use default 0."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()

    def test_price_vs_sma_positive_trend_zero(self, tmp_path):
        """price_vs_sma20 > 0 but trend_direction is 0 → default branch."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0.1, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # trend == 0 and price_vs_sma > 0 does NOT match trend > 0 condition
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()






# ---------------------------------------------------------------------------
# Constants validation — extended
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI main() function test
# ---------------------------------------------------------------------------



class TestZeroDTEClosingAuctionHonesty:
    """Dead overlay surfaces must not publish silent empty {}."""

    def test_unavailable_zero_dte_payload_has_schema_fields(self):
        payload = DashboardGenerator._unavailable_zero_dte_payload()
        assert payload["positions"] == []
        assert payload["config"] is None
        assert payload["weekly_trades_used"] == 0
        assert payload["status"] == "unavailable"
        assert payload["runtime_status"] == "unavailable_no_producer"
        assert payload["live_authoritative"] is False
        assert payload["active"] is False
        assert "generated_at" in payload

    def test_unavailable_closing_auction_payload_has_schema_fields(self):
        payload = DashboardGenerator._unavailable_closing_auction_payload()
        assert payload["signals"] == []
        assert payload["last_update"] is None
        assert payload["market_open"] is False
        assert payload["status"] == "unavailable"
        assert payload["runtime_status"] == "unavailable_no_producer"
        assert payload["live_authoritative"] is False

    def test_empty_dict_not_populated(self):
        assert DashboardGenerator._is_populated_overlay_section({}) is False
        assert DashboardGenerator._is_populated_overlay_section(None) is False
        assert DashboardGenerator._is_populated_overlay_section(
            DashboardGenerator._unavailable_zero_dte_payload()
        ) is False

    def test_real_producer_payload_is_populated(self):
        assert DashboardGenerator._is_populated_overlay_section(
            {"positions": [{"id": "x"}], "active": True}
        ) is True
        assert DashboardGenerator._is_populated_overlay_section(
            {"signals": [{"should_trade": False}], "market_open": True}
        ) is True

    def test_get_overlay_data_never_returns_silent_empty_surfaces(self, monkeypatch):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeOverlay:
            def generate(self):
                return self

            def to_dict(self):
                # No zero_dte / closing_auction keys — historical producer gap
                return {
                    "collar": {"active": False},
                    "crypto": {},
                    "calendar": {},
                    "kurtosis": {},
                    "bond_duration": {},
                    "unified": {},
                }

        monkeypatch.setattr(
            "src.dashboard.overlay_dashboard.OverlayDashboardGenerator",
            FakeOverlay,
            raising=False,
        )

        # Avoid real VIX generator / file IO
        class FakeVixGen:
            def generate_signal(self):
                class S:
                    def to_dict(self_inner):
                        return {"regime": "contango", "vix_spot": 18.0}

                return S()

        import sys
        import types

        monkeypatch.setitem(
            sys.modules,
            "src.signals.vix_term_structure",
            types.SimpleNamespace(VIXTermStructureSignalGenerator=FakeVixGen),
        )

        with patch("src.dashboard.generator.DATA_DIR", Path("/tmp/no-vix-overlay-state")):
            data = gen._get_overlay_data()

        assert data["zero_dte"]["runtime_status"] == "unavailable_no_producer"
        assert data["zero_dte"] != {}
        assert data["closing_auction"]["runtime_status"] == "unavailable_no_producer"
        assert data["closing_auction"] != {}
        assert data["zero_dte"]["positions"] == []
        assert data["closing_auction"]["signals"] == []

    def test_get_overlay_data_passes_through_real_producer(self, monkeypatch):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeOverlay:
            def generate(self):
                return self

            def to_dict(self):
                return {
                    "collar": {},
                    "crypto": {},
                    "calendar": {},
                    "kurtosis": {},
                    "bond_duration": {},
                    "unified": {},
                    "zero_dte": {
                        "positions": [{"id": "p1"}],
                        "config": None,
                        "weekly_trades_used": 1,
                        "total_premium_collected_mtd": 10.0,
                        "active": True,
                    },
                    "closing_auction": {
                        "signals": [{"should_trade": True}],
                        "last_update": "2026-07-20T12:00:00",
                        "market_open": True,
                    },
                }

        monkeypatch.setattr(
            "src.dashboard.overlay_dashboard.OverlayDashboardGenerator",
            FakeOverlay,
            raising=False,
        )

        class FakeVixGen:
            def generate_signal(self):
                class S:
                    def to_dict(self_inner):
                        return {}

                return S()

        import sys
        import types

        monkeypatch.setitem(
            sys.modules,
            "src.signals.vix_term_structure",
            types.SimpleNamespace(VIXTermStructureSignalGenerator=FakeVixGen),
        )

        with patch("src.dashboard.generator.DATA_DIR", Path("/tmp/no-vix-overlay-state")):
            data = gen._get_overlay_data()

        assert data["zero_dte"]["positions"][0]["id"] == "p1"
        assert data["closing_auction"]["market_open"] is True
        assert data["zero_dte"].get("runtime_status") != "unavailable_no_producer"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestTwoStageRegimeUnavailableHonesty:
    """Optional regime sections must not silently disappear when generators return None."""

    def test_two_stage_none_publishes_unavailable_section(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        import types
        from src.dashboard import generator as generator_module

        gen, _ = _make_generator(tmp_path)
        monkeypatch.setattr(generator_module, "DATA_DIR", tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        monkeypatch.setattr(
            "src.dashboard.generator.validate_signal",
            lambda _name, signal: signal,
        )
        monkeypatch.setattr(gen, "_generate_two_stage_regime", lambda: None)
        monkeypatch.setattr(gen, "_generate_bocd_regime", lambda: None)
        monkeypatch.setattr(gen, "_run_spc_monitor", lambda output: {"status": "ok"})
        monkeypatch.setattr(gen, "_record_ic_data", lambda output: None)
        # Empty regime history → transition also unavailable
        fake_cursor = types.SimpleNamespace(
            execute=lambda *a, **k: None,
            fetchall=lambda: [],
        )
        output = {
            "ensemble_voting": {
                "generated_at": fresh,
                "regime": "normal",
                "source_breakdown": [],
            },
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated_at": fresh},
        }
        try:
            result = gen._apply_signal_postprocessors(
                output,
                {"cursor": fake_cursor, "current_regime": "normal"},
            )
        finally:
            gen.conn.close()

        assert "two_stage_regime" in result
        ts = result["two_stage_regime"]
        assert ts.get("status") == "unavailable" or ts.get("runtime_status") == "unavailable"
        # Unavailable: null metric slots (not fake 0.0 confidence / UNKNOWN as calibrated)
        assert ts.get("regime") is None
        assert ts.get("confidence") is None

        assert "regime_transition" in result
        rt = result["regime_transition"]
        assert rt.get("status") == "unavailable" or rt.get("runtime_status") == "unavailable"


def test_asset_stats_tags_non_champion_symbols():
    """QQQ/VIX must be role-tagged, not undifferentiated held assets."""
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    # generate_stats_json must split held vs context and tag roles
    assert "held_asset_stats" in src
    assert "context_asset_stats" in src
    assert "champion_symbols" in src
    assert '"role": "held"' in src or "'role': 'held'" in src or 'role": "held"' in src
    assert "benchmark_or_context" in src
    assert "not_in_portfolio" in src


def test_load_entropy_data_no_hardcoded_correlation(tmp_path, monkeypatch):
    """Absent health entropy metrics must not invent 0.95 / 2.5 correlation quality."""
    from src.dashboard import generator as gen_mod
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)
    # no .health_report.json
    data = DashboardGenerator._load_entropy_data(DashboardGenerator.__new__(DashboardGenerator))
    assert data.get("correlation_entropy") is None
    assert data.get("participation_ratio") is None
    assert data.get("correlation_metrics_status") == "unavailable"
    assert data.get("correlation_entropy") != 0.95
    assert data.get("participation_ratio") != 2.5


def test_factor_rotation_signal_is_canonical_not_dual_authority():
    """Canonical factor_rotation carries authority tags; dashboard is alias-only."""
    src = Path("src/dashboard/signal_section_builder.py").read_text(encoding="utf-8")
    assert 'alias_of": "factor_rotation"' in src or "alias_of" in src
    assert "live_authoritative" in src
    assert "research_caveats" in src
    # No silent strength rounding fork on dashboard branch
    assert 'round(factor_rotation_result.get("signal_strength"' not in src


def test_configured_source_status_has_effective_and_active_weights():
    """Stale/missing sources get effective_weight 0; actives renorm to sum≈1."""
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "alternative_data", "weight": 0.2},
            {"source": "cross_asset_rv", "weight": 0.2},
        ],
    )
    # Force google_trends stale-like missing with disclosure
    by = {r["source"]: r for r in statuses}
    # At least some configured rows exist
    assert len(statuses) >= 2
    for row in statuses:
        assert "effective_weight" in row
        assert "active_weight" in row
        assert "configured_weight" in row
        if not row.get("contributing"):
            assert row["effective_weight"] == 0.0
            assert row["active_weight"] == 0.0
    rollup = DashboardGenerator._ensemble_active_weights_rollup(statuses)
    assert "active_weights" in rollup
    assert "dropped_weight_mass" in rollup
    if rollup["active_weights"]:
        assert abs(rollup["active_weights_sum"] - 1.0) < 0.02


def test_garch_cvar_demotes_when_coverage_fails(tmp_path, monkeypatch):
    """coverage_pass false must demote garch_active for primary risk use."""
    gen, _ = _make_generator(tmp_path)
    # Seed health report with filter_active true
    (tmp_path / ".health_report.json").write_text(json.dumps({
        "garch_filtered": True,
        "filter_active": True,
        "cvar_95": -2.0,
        "var_95": -1.5,
        "cvar_ratio": 1.3,
        "conditional_volatility_current": 1.2,
        "garch_persistence": 0.9,
    }))
    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        with patch(
            "src.monitor.conformal_risk.conformal_coverage_diagnostics",
            return_value={"coverage_pass": False, "kupiec_pass": False},
        ):
            with patch("src.monitor.conformal_risk.conformal_cvar", return_value=-0.02):
                with patch("src.monitor.conformal_risk.conformal_var", return_value=-0.01):
                    data = gen._load_garch_cvar_data()
    assert data["garch_active"] is False
    assert data.get("runtime_role") == "advisory_degraded"
    assert "coverage" in (data.get("garch_active_reason") or "").lower()
    gen.conn.close()


def test_generator_data_dir_isolated_by_autouse_without_explicit_patch(
    tmp_path, monkeypatch
):
    """Generator DATA_DIR reads are isolated by autouse, not opt-in patch().

    Retro (P1): missing/ignored generator fixture inventory caused host-only noise.
    Root cause: ``_isolate_live_ensemble_and_ic_health`` stubs compute but never
    rebinds ``src.dashboard.generator.DATA_DIR``, so any test that forgets the
    explicit ``patch("src.dashboard.generator.DATA_DIR", tmp_path)`` reads live
    host state (``data/.health_report.json``, ``performance.jsonl``, etc).

    This test must pass WITHOUT an explicit DATA_DIR patch: it seeds a sentinel
    ``.health_report.json`` under ``tmp_path`` and asserts the generator reads the
    sentinel (isolated), not the live host file.
    """
    gen, _ = _make_generator(tmp_path)
    # Sentinel: a value the live host file does not hold (host has cvar_95=-2.21).
    (tmp_path / ".health_report.json").write_text(json.dumps({
        "garch_filtered": True,
        "filter_active": True,
        "cvar_95": -9.99,
        "var_95": -8.88,
        "cvar_ratio": 1.11,
        "conditional_volatility_current": 1.0,
        "garch_persistence": 0.90,
    }))
    # Intentionally NO patch("src.dashboard.generator.DATA_DIR", tmp_path).
    # The autouse fixture must rebind DATA_DIR to tmp_path for this to pass.
    try:
        garch = gen._load_garch_cvar_data()
    finally:
        gen.conn.close()
    # If DATA_DIR leaked to the live host, cvar_95 would be -2.21 (host value).
    # Sentinel -9.99 / 100 -> -0.0999 (abs>1 branch divides by 100).
    assert garch["cvar_95"] == -0.0999, (
        f"DATA_DIR not isolated by autouse: cvar_95={garch['cvar_95']} "
        "indicates live host .health_report.json leaked into the test"
    )
