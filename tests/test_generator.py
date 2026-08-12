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



# ---------------------------------------------------------------------------
# VIX regime detection — boundary values at exact thresholds
# ---------------------------------------------------------------------------

# Graduation JSON — additional edge cases
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Extended field type / dataclass validation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Additional computation edge cases — boundary values, zero/negative, large
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# Constants validation — extended
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CLI main() function test
# ---------------------------------------------------------------------------





