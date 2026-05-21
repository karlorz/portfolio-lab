#!/usr/bin/env python3
"""Tests for v8.07 Portfolio Explainability Dashboard."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.portfolio_explainability import (
    CATEGORY_EMOJI,
    SIGNAL_SOURCE_META,
    DecisionExplanation,
    ExplainabilityReport,
    PortfolioExplainability,
    SignalContribution,
    SignalDeepDive,
    _get_latest_ensemble_votes,
    _get_source_readings_for_vote,
    main,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_vote():
    """Sample ensemble vote dict."""
    return {
        "timestamp": "2026-05-16T13:19:20.871111",
        "regime": "normal",
        "consensus": "bullish",
        "agreement_ratio": 0.85,
        "equity_bias": 0.637,
        "duration_bias": -0.724,
        "gold_bias": 0.577,
        "action": "INCREASE_EQUITY",
        "confidence": 0.64,
        "reasoning": "Regime: normal. Sources: 6. Consensus bullish.",
    }


@pytest.fixture
def sample_reading():
    """Sample source reading."""
    return {
        "source": "factor_rotation",
        "value": 0.34,
        "confidence": 0.70,
        "weight": 0.05,
        "regime_fit": "neutral",
        "explanation": "Quality-momentum blend bullish",
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with minimal state files."""
    # Create attribution subdir
    attr_dir = tmp_path / "attribution"
    attr_dir.mkdir()

    attr_report = {
        "timestamp": datetime.now().isoformat(),
        "sources": {
            "multi_speed_momentum": {
                "hit_rate": 0.52,
                "sharpe_contribution": 0.3,
                "avg_return_bps": 1.5,
                "display_name": "Multi-Speed Momentum",
                "category": "trend",
                "total_readings": 10,
                "active_days": 5,
                "win_rate": 0.55,
                "avg_correlation": 0.1,
                "avg_weight": 0.15,
                "max_consecutive_losses": 2,
            }
        },
        "best_source": "multi_speed_momentum",
        "worst_source": "multi_speed_momentum",
        "avg_hit_rate": 0.50,
        "avg_correlation": 0.1,
    }
    with open(attr_dir / "attribution_2026-05-17.json", "w") as f:
        json.dump(attr_report, f)

    return tmp_path


# ─────────────────────────────────────────────
#  Data Model Tests
# ─────────────────────────────────────────────


class TestSignalContribution:
    def test_creation(self):
        sig = SignalContribution(
            source="test_source",
            display_name="Test Signal",
            category="trend",
            value=0.5,
            confidence=0.8,
            weight=0.2,
            regime_fit="all",
            explanation="Bullish momentum",
            contribution_pct=25.0,
            is_tiebreaking=False,
        )
        assert sig.source == "test_source"
        assert sig.value == 0.5
        assert sig.contribution_pct == 25.0
        assert sig.is_tiebreaking is False

    def test_tiebreaking_flag(self):
        sig = SignalContribution(
            source="ts", display_name="TS", category="trend",
            value=0.5, confidence=0.8, weight=0.2, regime_fit="all",
            explanation="", contribution_pct=45.0, is_tiebreaking=True,
        )
        assert sig.is_tiebreaking is True


class TestDecisionExplanation:
    def test_creation(self):
        sig = SignalContribution(
            source="a", display_name="A", category="trend",
            value=0.3, confidence=0.7, weight=0.1, regime_fit="all",
            explanation="", contribution_pct=30.0, is_tiebreaking=False,
        )
        dec = DecisionExplanation(
            timestamp="2026-05-17T00:00:00",
            period="2026-05-17",
            regime="normal",
            action="rebalance",
            confidence=0.7,
            reasoning="Multi-signal consensus",
            asset_changes={"SPY": 0.5, "GLD": -0.3},
            current_allocation={"SPY": 46.0, "GLD": 38.0, "TLT": 16.0},
            total_signals=1,
            consensus_direction="bullish",
            agreement_ratio=0.8,
            signals=[sig],
            top_drivers=["A"],
            top_opposers=[],
        )
        assert dec.period == "2026-05-17"
        assert dec.regime == "normal"
        assert len(dec.signals) == 1
        assert dec.asset_changes["SPY"] == 0.5

    def test_with_attribution(self):
        dec = DecisionExplanation(
            timestamp="2026-05-17T00:00:00",
            period="2026-05-17",
            regime="normal", action="hold",
            confidence=0.5, reasoning="",
            asset_changes={}, current_allocation={},
            total_signals=0, consensus_direction="neutral",
            agreement_ratio=0.0, signals=[],
            top_drivers=[], top_opposers=[],
            attribution_summary={"sources": {"mom": {"hit_rate": 0.55}}},
        )
        assert dec.attribution_summary is not None
        assert dec.attribution_summary["sources"]["mom"]["hit_rate"] == 0.55


class TestExplainabilityReport:
    def test_creation(self):
        report = ExplainabilityReport(
            timestamp="2026-05-17T00:00:00",
            analysis_date="2026-05-17",
        )
        assert report.latest_decision is None
        assert report.recent_decisions == []
        assert report.signal_deep_dives == {}
        assert report.decision_quality == "unknown"

    def test_with_data(self):
        sig = SignalContribution(
            source="a", display_name="A", category="trend",
            value=0.3, confidence=0.7, weight=0.1, regime_fit="all",
            explanation="", contribution_pct=30.0, is_tiebreaking=False,
        )
        dec = DecisionExplanation(
            timestamp="2026-05-17T00:00:00", period="2026-05-17",
            regime="normal", action="rebalance", confidence=0.7,
            reasoning="test", asset_changes={}, current_allocation={},
            total_signals=1, consensus_direction="bullish",
            agreement_ratio=0.8, signals=[sig],
            top_drivers=["A"], top_opposers=[],
        )
        dive = SignalDeepDive(
            source="a", display_name="A", category="trend",
            total_observations=10, avg_value=0.3, avg_confidence=0.7,
            avg_weight=0.15, hit_rate=0.6, sharpe_contribution=0.5,
            avg_return_bps=2.0, recent_trend="improving",
            regime_fit_distribution={"all": 10},
            correlation_with_peers=0.1,
        )
        report = ExplainabilityReport(
            timestamp="2026-05-17T00:00:00",
            analysis_date="2026-05-17",
            latest_decision=dec,
            recent_decisions=[dec],
            signal_deep_dives={"a": dive},
            top_sources_today=["A"],
            decision_quality="good",
        )
        assert report.latest_decision is not None
        assert len(report.recent_decisions) == 1
        assert len(report.signal_deep_dives) == 1
        assert report.decision_quality == "good"


# ─────────────────────────────────────────────
#  Data Loading Tests
# ─────────────────────────────────────────────


class TestGetLatestEnsembleVotes:
    def test_no_db_returns_empty(self):
        """Should return empty when no DB exists."""
        with patch("src.monitor.portfolio_explainability.DATA_DIR", Path("/nonexistent")):
            result = _get_latest_ensemble_votes(5)
            assert result == []


class TestGetSourceReadings:
    def test_no_db_returns_empty(self):
        with patch("src.monitor.portfolio_explainability.DATA_DIR", Path("/nonexistent")):
            result = _get_source_readings_for_vote("2026-05-16T13:00:00")
            assert result == []


# ─────────────────────────────────────────────
#  Explainability Engine Tests
# ─────────────────────────────────────────────


class TestPortfolioExplainability:
    def test_init(self, temp_data_dir):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        assert explainer is not None

    @patch("src.monitor.portfolio_explainability._get_latest_ensemble_votes")
    def test_explain_latest_no_votes(self, mock_votes, temp_data_dir):
        mock_votes.return_value = []
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        result = explainer.explain_latest_decision()
        assert result is None

    @patch("src.monitor.portfolio_explainability._get_latest_ensemble_votes")
    @patch("src.monitor.portfolio_explainability._get_source_readings_for_vote")
    def test_explain_latest_with_vote(self, mock_readings, mock_votes,
                                       temp_data_dir, sample_vote, sample_reading):
        mock_votes.return_value = [sample_vote]
        mock_readings.return_value = [sample_reading]

        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        decision = explainer.explain_latest_decision()

        assert decision is not None
        assert decision.action == "INCREASE_EQUITY"
        assert decision.regime == "normal"
        assert decision.consensus_direction in ("bullish", "bearish", "neutral")
        assert len(decision.signals) > 0
        assert len(decision.top_drivers) > 0

    @patch("src.monitor.portfolio_explainability._get_latest_ensemble_votes")
    @patch("src.monitor.portfolio_explainability._get_source_readings_for_vote")
    def test_explain_recent(self, mock_readings, mock_votes,
                             temp_data_dir, sample_vote, sample_reading):
        mock_votes.return_value = [sample_vote] * 3
        mock_readings.return_value = [sample_reading]

        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        decisions = explainer.explain_recent_decisions(3)

        assert len(decisions) <= 3
        for d in decisions:
            assert d.action == "INCREASE_EQUITY"

    def test_signal_deep_dive_unknown(self, temp_data_dir):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        result = explainer.signal_deep_dive("nonexistent_signal")
        assert result is None

    @patch("src.monitor.portfolio_explainability._get_all_source_readings")
    def test_signal_deep_dive_known_no_data(self, mock_readings, temp_data_dir):
        mock_readings.return_value = {}
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        result = explainer.signal_deep_dive("multi_speed_momentum")
        # Should match "multi_speed_momentum" from SIGNAL_SOURCE_META key check
        # but find no readings
        assert result is None

    @patch("src.monitor.portfolio_explainability._get_all_source_readings")
    def test_signal_deep_dive_with_data(self, mock_readings, temp_data_dir, sample_reading):
        sample_reading["source"] = "multi_speed_momentum"
        # 25 readings — enough for >=20 threshold, all same value → "stable" trend
        readings = [dict(sample_reading) for _ in range(25)]
        for i, r in enumerate(readings):
            r["timestamp"] = f"2026-05-{1 + i:02d}T00:00:00"
        mock_readings.return_value = {"multi_speed_momentum": readings}

        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        dive = explainer.signal_deep_dive("multi_speed_momentum")

        assert dive is not None
        assert dive.source == "multi_speed_momentum"
        assert dive.total_observations == 25
        assert dive.avg_value == pytest.approx(0.34, abs=0.01)
        assert dive.recent_trend == "stable"

    def test_generate_report_empty(self, temp_data_dir):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        report = explainer.generate_report()

        assert report.analysis_date == datetime.now().strftime("%Y-%m-%d")
        assert isinstance(report, ExplainabilityReport)

    def test_save_report(self, temp_data_dir):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        report = explainer.generate_report()
        path = explainer.save_report(report)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["analysis_date"] == report.analysis_date

    def test_to_json(self, temp_data_dir):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        report = explainer.generate_report()
        json_str = explainer.to_json(report)

        data = json.loads(json_str)
        assert data["analysis_date"] == report.analysis_date


# ─────────────────────────────────────────────
#  Signal Contribution Building
# ─────────────────────────────────────────────


class TestSignalContributionBuilding:
    def test_build_from_readings(self):
        readings = [
            {"source": "factor_rotation", "value": 0.34, "confidence": 0.7,
             "weight": 0.05, "regime_fit": "neutral", "explanation": "bullish"},
            {"source": "macro_momentum", "value": -0.15, "confidence": 0.6,
             "weight": 0.09, "regime_fit": "expansion", "explanation": "bearish"},
        ]

        explainer = PortfolioExplainability(data_dir=Path("/tmp"))
        signals = explainer._build_signal_contributions(readings)

        assert len(signals) == 2
        # Sorted by contribution_pct descending
        assert signals[0].contribution_pct >= signals[1].contribution_pct
        # Factor rotation (0.34) should have higher abs contribution than macro (-0.15)
        assert signals[0].source == "factor_rotation"

    def test_empty_readings(self):
        explainer = PortfolioExplainability(data_dir=Path("/tmp"))
        signals = explainer._build_signal_contributions([])
        assert signals == []

    def test_single_reading(self):
        readings = [{"source": "cta_trend", "value": 0.5, "confidence": 0.8,
                     "weight": 0.15, "regime_fit": "all", "explanation": ""}]
        explainer = PortfolioExplainability(data_dir=Path("/tmp"))
        signals = explainer._build_signal_contributions(readings)

        assert len(signals) == 1
        assert signals[0].contribution_pct == 100.0
        assert signals[0].is_tiebreaking is False  # Only 1 signal, no tie to break


class TestAssetChanges:
    def test_estimate_changes(self, temp_data_dir, sample_vote):
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        changes = explainer._estimate_asset_changes(sample_vote, {})

        assert "SPY" in changes
        assert "GLD" in changes
        assert "TLT" in changes
        assert changes["SPY"] == 0.64
        assert changes["TLT"] == -0.72

    def test_no_bias(self, temp_data_dir):
        vote = {"some_key": "value"}
        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        changes = explainer._estimate_asset_changes(vote, {})

        assert changes == {}


# ─────────────────────────────────────────────
#  CLI Tests
# ─────────────────────────────────────────────


class TestCLI:
    def test_main_no_args(self):
        """Should print help with no args."""
        with patch.object(sys, "argv", ["portfolio_explainability.py"]):
            try:
                main()
            except SystemExit:
                pass

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_explain(self, mock_explainer):
        mock_instance = MagicMock()
        mock_instance.explain_latest_decision.return_value = None
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "explain"]):
            main()
            mock_instance.explain_latest_decision.assert_called_once()

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_today(self, mock_explainer):
        mock_instance = MagicMock()
        mock_instance.generate_report.return_value = MagicMock(
            analysis_date="2026-05-17",
            latest_decision=None,
        )
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "today"]):
            main()
            mock_instance.generate_report.assert_called_once()

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_signal_found(self, mock_explainer):
        mock_instance = MagicMock()
        dive = SignalDeepDive(
            source="factor_rotation", display_name="Factor Rotation",
            category="factor", total_observations=15,
            avg_value=0.3, avg_confidence=0.7, avg_weight=0.05,
            hit_rate=0.55, sharpe_contribution=0.4, avg_return_bps=1.2,
            recent_trend="stable",
            regime_fit_distribution={"neutral": 15},
            correlation_with_peers=0.05,
        )
        mock_instance.signal_deep_dive.return_value = dive
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "signal", "factor_rotation"]):
            main()
            mock_instance.signal_deep_dive.assert_called_once_with("factor_rotation")

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_signal_not_found(self, mock_explainer):
        mock_instance = MagicMock()
        mock_instance.signal_deep_dive.return_value = None
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "signal", "nonexistent"]):
            main()
            mock_instance.signal_deep_dive.assert_called_once_with("nonexistent")

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_history(self, mock_explainer):
        mock_instance = MagicMock()
        mock_instance.explain_recent_decisions.return_value = []
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "history", "--n", "3"]):
            main()
            mock_instance.explain_recent_decisions.assert_called_once_with(3)

    @patch("src.monitor.portfolio_explainability.PortfolioExplainability")
    def test_main_all(self, mock_explainer):
        mock_instance = MagicMock()
        mock_instance.generate_report.return_value = MagicMock(
            latest_decision=None,
            recent_decisions=[],
            signal_deep_dives={},
        )
        mock_explainer.return_value = mock_instance

        with patch.object(sys, "argv", ["portfolio_explainability.py", "all"]):
            main()
            mock_instance.generate_report.assert_called_once()

    def test_main_save_flag(self):
        """Verify save flag is accepted by subparsers."""
        # Test that --save is accepted with explain command
        with patch.object(sys, "argv", ["portfolio_explainability.py", "explain", "--save"]):
            with patch("src.monitor.portfolio_explainability.PortfolioExplainability") as me:
                inst = MagicMock()
                inst.explain_latest_decision.return_value = None
                me.return_value = inst
                main()
                inst.explain_latest_decision.assert_called_once()

    def test_main_all_save(self):
        with patch.object(sys, "argv", ["portfolio_explainability.py", "all", "--save"]):
            with patch("src.monitor.portfolio_explainability.PortfolioExplainability") as me:
                inst = MagicMock()
                inst.generate_report.return_value = MagicMock(
                    latest_decision=None, recent_decisions=[], signal_deep_dives={},
                )
                me.return_value = inst
                main()
                inst.generate_report.assert_called_once()


# ─────────────────────────────────────────────
#  Edge Cases & Error Handling
# ─────────────────────────────────────────────


class TestEdgeCases:
    @patch("src.monitor.portfolio_explainability._get_latest_ensemble_votes")
    @patch("src.monitor.portfolio_explainability._get_source_readings_for_vote")
    def test_very_negative_consensus(self, mock_readings, mock_votes,
                                      temp_data_dir, sample_vote):
        """Verify bearish consensus detection."""
        vote = dict(sample_vote)
        mock_votes.return_value = [vote]
        mock_readings.return_value = [
            {"source": "factor_rotation", "value": -0.8, "confidence": 0.9,
             "weight": 0.2, "regime_fit": "crisis", "explanation": "bearish"},
        ]

        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        dec = explainer.explain_latest_decision()

        assert dec is not None
        assert dec.consensus_direction == "bearish"

    @patch("src.monitor.portfolio_explainability._get_latest_ensemble_votes")
    @patch("src.monitor.portfolio_explainability._get_source_readings_for_vote")
    def test_mixed_signals_neutral(self, mock_readings, mock_votes,
                                    temp_data_dir, sample_vote):
        """Verify neutral consensus when signals cancel."""
        vote = dict(sample_vote)
        mock_votes.return_value = [vote]
        mock_readings.return_value = [
            {"source": "macro_momentum", "value": 0.02, "confidence": 0.3,
             "weight": 0.01, "regime_fit": "neutral", "explanation": ""},
            {"source": "mean_reversion", "value": -0.01, "confidence": 0.2,
             "weight": 0.01, "regime_fit": "neutral", "explanation": ""},
        ]

        explainer = PortfolioExplainability(data_dir=temp_data_dir)
        dec = explainer.explain_latest_decision()

        assert dec is not None
        assert dec.consensus_direction == "neutral"

    def test_category_emoji_complete(self):
        """All categories in SIGNAL_SOURCE_META should have an emoji."""
        for meta in SIGNAL_SOURCE_META.values():
            cat = meta.get("category", "other")
            assert cat in CATEGORY_EMOJI, f"Missing emoji for category: {cat}"

    def test_signal_deep_dive_trend_improving(self):
        readings = []
        for i in range(25):
            readings.append({
                "source": "test_sig", "value": 0.1 * (i + 1),  # Increasing
                "confidence": 0.7, "weight": 0.1,
                "regime_fit": "all", "explanation": "",
                "timestamp": f"2026-05-{1 + i:02d}T00:00:00",
            })
        # Reverse so most recent (highest values) come first
        readings.reverse()

        with patch("src.monitor.portfolio_explainability._get_all_source_readings") as mr:
            mr.return_value = {"test_sig": readings}
            # Patch SIGNAL_SOURCE_META to include test_sig
            with patch.dict("src.monitor.portfolio_explainability.SIGNAL_SOURCE_META",
                            {"test_sig": {"name": "Test Signal", "category": "trend"}}):
                explainer = PortfolioExplainability(data_dir=Path("/tmp"))
                dive = explainer.signal_deep_dive("test_sig")

                assert dive is not None
                assert dive.recent_trend == "improving"


# ─────────────────────────────────────────────
#  Performance Tests
# ─────────────────────────────────────────────


class TestPerformance:
    def test_large_reading_set(self):
        """Should handle many readings without error."""
        readings = [
            {"source": f"sig_{i % 20}", "value": (i % 10) * 0.1,
             "confidence": 0.5 + (i % 5) * 0.1, "weight": 0.01 * (i % 10 + 1),
             "regime_fit": "all", "explanation": "auto"}
            for i in range(1000)
        ]

        explainer = PortfolioExplainability(data_dir=Path("/tmp"))
        signals = explainer._build_signal_contributions(readings)

        assert len(signals) <= 1000
        assert all(s.contribution_pct >= 0 for s in signals)

    def test_report_generation_speed(self, temp_data_dir):
        """Report generation should be fast."""
        import time
        explainer = PortfolioExplainability(data_dir=temp_data_dir)

        start = time.time()
        report = explainer.generate_report()
        elapsed = time.time() - start

        assert isinstance(report, ExplainabilityReport)
        assert elapsed < 2.0  # Should generate in under 2 seconds
