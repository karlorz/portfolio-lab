#!/usr/bin/env python3
"""
Generator regime tests — VIX regime detection classes
(TEST-GENERATOR-SPLIT s4, 2026-08-12).

Moved verbatim from tests/test_generator.py (TestVIXRegimeDetection,
TestVIXRegimeBoundaries) — no tests renamed or weakened. Shared helpers live
in tests/helpers.py (plain module; the autouse fixture below is duplicated
verbatim per split file — never move it to conftest.py, it would pollute the
full ~15k-test suite).
"""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator


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
# VIX regime detection tests
# ---------------------------------------------------------------------------


class TestVIXRegimeDetection:
    """Test VIX-based regime classification logic."""

    def _classify_vix(self, vix_level):
        """Extract VIX regime classification logic."""
        if vix_level > 25:
            return "crisis"
        elif vix_level > 20:
            return "vol_spike"
        elif vix_level < 15:
            return "low_vol"
        else:
            return "normal"

    def test_crisis_regime(self):
        assert self._classify_vix(30) == "crisis"
        assert self._classify_vix(26) == "crisis"

    def test_vol_spike_regime(self):
        assert self._classify_vix(22) == "vol_spike"
        assert self._classify_vix(21) == "vol_spike"

    def test_low_vol_regime(self):
        assert self._classify_vix(12) == "low_vol"
        assert self._classify_vix(14) == "low_vol"

    def test_normal_regime(self):
        assert self._classify_vix(18) == "normal"
        assert self._classify_vix(15) == "normal"
        assert self._classify_vix(20) == "normal"

    def test_composite_regime_vix_overrides(self):
        """VIX crisis/vol_spike overrides trend regime."""
        # If VIX says crisis, it overrides regardless of trend
        vix_regime = "crisis"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        else:
            current_regime = trend_regime
        assert current_regime == "crisis"

    def test_composite_regime_low_vol_with_normal_trend(self):
        """Low vol + normal trend → low_vol."""
        vix_regime = "low_vol"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "low_vol"

    def test_composite_regime_normal_uses_trend(self):
        """Normal VIX + trend regime → uses trend."""
        vix_regime = "normal"
        trend_regime = "bull"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "bull"


# ---------------------------------------------------------------------------
# VIX regime detection — boundary values at exact thresholds
# ---------------------------------------------------------------------------

class TestVIXRegimeBoundaries:
    """VIX regime at exact boundary values."""

    def test_vix_exactly_15(self):
        """VIX exactly 15 is normal regime."""
        _ = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", Path("/tmp")):
            # Extract the classify logic
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(15) == "normal"
            assert classify(20) == "normal"
            assert classify(25) == "vol_spike"  # >20 not >=20

    def test_vix_vol_spike_upper(self):
        """VIX exactly 25 is vol_spike (>20, not >25)."""
        _ = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", Path("/tmp")):
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(25) == "vol_spike"

