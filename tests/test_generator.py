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



# ---------------------------------------------------------------------------
# Broker data tests
# ---------------------------------------------------------------------------



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



# ---------------------------------------------------------------------------
# Run integration test
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# GARCH-CVaR data edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Entropy data edge cases
# ---------------------------------------------------------------------------



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



# ---------------------------------------------------------------------------
# Broker data edge cases
# ---------------------------------------------------------------------------



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



# ---------------------------------------------------------------------------
# Entropy edge cases (continued)
# ---------------------------------------------------------------------------



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
