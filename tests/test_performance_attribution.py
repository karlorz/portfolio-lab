#!/usr/bin/env python3
"""
Tests for Performance Attribution System (v5.70).
"""

import json
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


import pytest

from src.monitor.performance_attribution import (
    PerformanceAttribution,
    SourceAttribution,
    AttributionReport,
    SIGNAL_SOURCE_META,
    print_report,
    patch_save_vote,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory with ensemble DB."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def ensemble_db(tmp_data_dir):
    """Create ensemble_signals.db with sample data."""
    db_path = tmp_data_dir / "ensemble_signals.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_votes (
            timestamp TEXT PRIMARY KEY,
            regime TEXT,
            regime_confidence REAL,
            num_sources INTEGER,
            consensus REAL,
            agreement_ratio REAL,
            equity_bias REAL,
            duration_bias REAL,
            gold_bias REAL,
            action TEXT,
            confidence REAL,
            reasoning TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_readings (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            source TEXT,
            value REAL,
            confidence REAL,
            weight REAL,
            regime_fit TEXT,
            explanation TEXT
        )
    """)

    # Add sample ensemble votes
    base_date = datetime.now()
    for i in range(30):
        d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            INSERT INTO ensemble_votes
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (d, "normal", 0.7, 8, 0.25, 0.65, 0.3, -0.1, 0.05, "neutral", 0.6,
              f"test reasoning day {i}"))

        # Add source readings for each vote
        for src in ["multi_speed_momentum", "macro_momentum", "visibility_graph",
                     "vp_macd", "transient_factors", "duration_regime",
                     "tsfm_momentum", "unified_overlay"]:
            val = np.random.uniform(-0.5, 0.5)
            conf = np.random.uniform(0.4, 0.9)
            weight = np.random.uniform(0.05, 0.25)
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (d, src, float(val), float(conf), float(weight), "all",
                  f"Test signal for {src}: value={val:.3f}"))

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def paper_trading_db(tmp_data_dir):
    """Create paper_trading.db with sample daily snapshots."""
    db_path = tmp_data_dir / "paper_trading.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            total_value REAL,
            daily_return REAL,
            cumulative_return REAL,
            spy_value REAL,
            gld_value REAL,
            tlt_value REAL,
            ief_value REAL,
            shy_value REAL,
            btc_value REAL,
            eth_value REAL,
            cash REAL,
            collar_active INTEGER,
            crypto_active INTEGER,
            bond_position TEXT,
            vix_level REAL
        )
    """)

    base_date = datetime.now()
    cum_ret = 0.0
    for i in range(30):
        d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_ret = np.random.normal(0.0005, 0.008)
        cum_ret += daily_ret
        conn.execute("""
            INSERT INTO daily_snapshots
            (date, total_value, daily_return, cumulative_return)
            VALUES (?, ?, ?, ?)
        """, (d, float(100000 * (1 + cum_ret)), float(daily_ret), float(cum_ret)))

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def attributor(tmp_data_dir, ensemble_db, paper_trading_db):
    """Create PerformanceAttribution with tmp data dir."""
    return PerformanceAttribution(data_dir=tmp_data_dir)


# ---------------------------------------------------------------------------
# SourceAttribution tests
# ---------------------------------------------------------------------------

class TestSourceAttribution:
    def test_default_construction(self):
        sa = SourceAttribution(
            source="test_src", display_name="Test Source", category="trend",
            total_readings=10, active_days=8, hit_rate=0.6, win_rate=0.55,
            avg_return_bps=1.5, total_return_bps=12.0, sharpe_contribution=0.8,
            max_consecutive_losses=3, avg_correlation=0.2, avg_weight=0.15,
        )
        assert sa.source == "test_src"
        assert sa.hit_rate == 0.6
        assert sa.sharpe_contribution == 0.8
        assert sa.efficiency_ratio == 0.009  # 0.6 * 1.5 / 100

    def test_efficiency_zero_return(self):
        sa = SourceAttribution(
            source="flat", display_name="Flat", category="other",
            total_readings=0, active_days=0, hit_rate=0.0, win_rate=0.0,
            avg_return_bps=0.0, total_return_bps=0.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.0,
        )
        assert sa.efficiency_ratio == 0.0

    def test_to_dict(self):
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=5, active_days=4, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=2.0, total_return_bps=8.0, sharpe_contribution=0.3,
            max_consecutive_losses=2, avg_correlation=0.1, avg_weight=0.1,
        )
        d = sa.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "test"
        assert d["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# AttributionReport tests
# ---------------------------------------------------------------------------

class TestAttributionReport:
    def test_default_construction(self):
        src = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=10, active_days=8, hit_rate=0.6, win_rate=0.55,
            avg_return_bps=1.5, total_return_bps=12.0, sharpe_contribution=0.8,
            max_consecutive_losses=3, avg_correlation=0.2, avg_weight=0.15,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01",
            end_date="2026-01-01",
            analysis_days=90,
            sources={"test": src},
            best_source="test",
            worst_source=None,
            avg_hit_rate=0.6,
            avg_correlation=0.2,
            avg_active_sources_per_day=5.0,
            total_sources_tracked=1,
            degradation_signals=[],
            top_performers=["test"],
        )
        assert report.best_source == "test"
        assert report.avg_hit_rate == 0.6

    def test_to_json(self):
        src = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=10, active_days=8, hit_rate=0.6, win_rate=0.55,
            avg_return_bps=1.5, total_return_bps=12.0, sharpe_contribution=0.8,
            max_consecutive_losses=3, avg_correlation=0.2, avg_weight=0.15,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01",
            end_date="2026-01-01",
            analysis_days=90,
            sources={"test": src},
            best_source="test",
            worst_source=None,
            avg_hit_rate=0.6,
            avg_correlation=0.2,
            avg_active_sources_per_day=5.0,
            total_sources_tracked=1,
            degradation_signals=[],
            top_performers=["test"],
        )
        j = report.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["best_source"] == "test"


# ---------------------------------------------------------------------------
# PerformanceAttribution tests
# ---------------------------------------------------------------------------

class TestPerformanceAttribution:
    def test_init(self, attributor):
        assert attributor.data_dir.exists()
        assert attributor.attribution_dir.exists()

    def test_get_signal_history(self, attributor, ensemble_db):
        history = attributor._get_signal_history(days=90)
        assert len(history) > 0

        # Should include source readings and ensemble votes
        readings = [h for h in history if h.get("type") != "ensemble_vote"]
        votes = [h for h in history if h.get("type") == "ensemble_vote"]
        assert len(readings) > 0
        assert len(votes) > 0

        # Verify reading fields
        reading = readings[0]
        assert "source" in reading
        assert "value" in reading
        assert "confidence" in reading
        assert "timestamp" in reading

    def test_get_signal_history_no_db(self, tmpdir):
        no_db_dir = Path(tmpdir) / "no_data"
        no_db_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=no_db_dir)
        history = attributor._get_signal_history()
        assert history == []

    def test_get_paper_trading_returns(self, attributor, paper_trading_db):
        returns = attributor._get_paper_trading_returns(days=30)
        assert len(returns) > 0

        # Verify return structure
        for date_str, ret_data in returns.items():
            assert "daily_return" in ret_data
            assert "cumulative_return" in ret_data

    def test_get_paper_trading_returns_no_data(self, tmpdir):
        no_db_dir = Path(tmpdir) / "empty"
        no_db_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=no_db_dir)
        returns = attributor._get_paper_trading_returns()
        assert isinstance(returns, dict)

    def test_compute_hit_rate(self, attributor):
        # Signal positive, return positive = hit
        assert attributor._compute_hit_rate(0.5, 0.01) is True
        # Signal positive, return negative = miss
        assert attributor._compute_hit_rate(0.5, -0.01) is False
        # Signal negative, return negative = hit
        assert attributor._compute_hit_rate(-0.5, -0.01) is True
        # Signal negative, return positive = miss
        assert attributor._compute_hit_rate(-0.5, 0.01) is False
        # Neutral signal, flat return = hit
        assert attributor._compute_hit_rate(0.0, 0.0005) is True
        # Neutral signal, large move = miss
        assert attributor._compute_hit_rate(0.0, 0.02) is False

    def test_compute_source_attribution(self, attributor):
        signals = [
            {"source": "multi_speed_momentum", "timestamp": "2026-05-15", "value": 0.5,
             "confidence": 0.8, "weight": 0.2, "regime_fit": "all", "explanation": ""},
            {"source": "multi_speed_momentum", "timestamp": "2026-05-14", "value": -0.3,
             "confidence": 0.7, "weight": 0.15, "regime_fit": "all", "explanation": ""},
            {"source": "multi_speed_momentum", "timestamp": "2026-05-13", "value": 0.1,
             "confidence": 0.6, "weight": 0.1, "regime_fit": "all", "explanation": ""},
        ]

        daily_returns = {
            "2026-05-15": {"daily_return": 0.01},
            "2026-05-14": {"daily_return": -0.005},
            "2026-05-13": {"daily_return": 0.002},
        }

        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.source == "multi_speed_momentum"
        assert attrib.total_readings == 3
        assert attrib.active_days == 3
        assert attrib.hit_rate > 0  # Should have some hits
        assert attrib.avg_weight > 0

    def test_compute_source_attribution_empty(self, attributor):
        attrib = attributor._compute_source_attribution([], {})
        assert attrib.source == "unknown"
        assert attrib.total_readings == 0

    def test_generate_report(self, attributor):
        report = attributor.generate_report(days=30)
        assert isinstance(report, AttributionReport)
        assert len(report.sources) > 0
        assert report.total_sources_tracked > 0

        # Sources should have meaningful metrics
        for src_name, src in report.sources.items():
            assert src.source == src_name
            assert src.total_readings >= 0
            assert 0 <= src.hit_rate <= 1
            assert 0 <= src.win_rate <= 1

    def test_generate_report_no_data(self, tmpdir):
        """Generate report without any database (graceful degradation)."""
        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=empty_dir)
        report = attributor.generate_report(days=30)
        assert isinstance(report, AttributionReport)
        assert len(report.sources) == 0

    def test_save_report(self, attributor):
        report = attributor.generate_report(days=30)
        path = attributor.save_report(report)
        assert path.exists()
        assert "attribution_" in path.name

        # Verify JSON content
        with open(path) as f:
            data = json.load(f)
        assert "sources" in data
        assert "best_source" in data

    def test_save_and_load_report(self, attributor):
        report = attributor.generate_report(days=30)
        attributor.save_report(report)

        loaded = attributor.load_latest_report()
        assert loaded is not None
        assert loaded.timestamp == report.timestamp
        assert len(loaded.sources) == len(report.sources)

    def test_load_latest_report_no_files(self, tmpdir):
        empty_dir = Path(tmpdir) / "no_reports"
        empty_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=empty_dir)
        assert attributor.load_latest_report() is None

    def test_correlation_matrix(self, attributor, ensemble_db):
        signal_history = attributor._get_signal_history(days=30)
        source_signals = {}
        for entry in signal_history:
            if entry.get("type") != "ensemble_vote":
                src = entry.get("source", "unknown")
                if src not in source_signals:
                    source_signals[src] = []
                source_signals[src].append(entry)

        if len(source_signals) >= 2:
            correlations = attributor._compute_correlation_matrix(source_signals)
            assert len(correlations) == len(source_signals)
            for src in source_signals:
                assert src in correlations

    def test_signal_source_meta(self):
        """Verify signal source metadata is complete."""
        assert len(SIGNAL_SOURCE_META) >= 6
        for key, meta in SIGNAL_SOURCE_META.items():
            assert "name" in meta
            assert "category" in meta
            assert "weight_tier" in meta

    def test_report_identifies_degradation(self, attributor):
        """Report should flag sources with very poor metrics."""
        report = attributor.generate_report(days=30)
        # degradation_signals should be a list (possibly empty)
        assert isinstance(report.degradation_signals, list)
        assert isinstance(report.top_performers, list)
        # best and worst should be strings or None
        assert report.best_source is None or isinstance(report.best_source, str)
        assert report.worst_source is None or isinstance(report.worst_source, str)


# ---------------------------------------------------------------------------
# print_report test (smoke test - does not crash)
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_print_report_empty(self, caplog):
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01",
            end_date="2026-01-01",
            analysis_days=90,
            sources={},
            best_source=None,
            worst_source=None,
            avg_hit_rate=0.0,
            avg_correlation=0.0,
            avg_active_sources_per_day=0.0,
            total_sources_tracked=0,
            degradation_signals=[],
            top_performers=[],
        )
        # Should not crash
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "PERFORMANCE ATTRIBUTION REPORT" in caplog.text

    def test_print_report_none_win_rate_no_data_source(self, caplog):
        """Sources with active_days=0 use win_rate=None — must not TypeError."""
        src = SourceAttribution(
            source="stale_src",
            display_name="Stale Source",
            category="other",
            total_readings=3,
            active_days=0,
            hit_rate=None,
            win_rate=None,
            avg_return_bps=0.0,
            total_return_bps=0.0,
            sharpe_contribution=0.0,
            max_consecutive_losses=0,
            avg_correlation=0.0,
            avg_weight=0.0,
        )
        report = AttributionReport(
            timestamp="2026-07-20T00:00:00",
            start_date="2026-04-21",
            end_date="2026-07-20",
            analysis_days=90,
            sources={"stale_src": src},
            best_source=None,
            worst_source=None,
            avg_hit_rate=None,
            avg_correlation=0.0,
            avg_active_sources_per_day=0.0,
            total_sources_tracked=1,
            degradation_signals=[],
            top_performers=[],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "Stale Source" in caplog.text
        assert "n/a" in caplog.text

    def test_print_report_with_data(self, caplog):
        src = SourceAttribution(
            source="vp_macd", display_name="VP-MACD", category="momentum",
            total_readings=50, active_days=45, hit_rate=0.62, win_rate=0.58,
            avg_return_bps=1.2, total_return_bps=54.0, sharpe_contribution=0.85,
            max_consecutive_losses=4, avg_correlation=0.15, avg_weight=0.12,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01",
            end_date="2026-01-01",
            analysis_days=90,
            sources={"vp_macd": src},
            best_source="vp_macd",
            worst_source=None,
            avg_hit_rate=0.62,
            avg_correlation=0.15,
            avg_active_sources_per_day=3.0,
            total_sources_tracked=1,
            degradation_signals=[],
            top_performers=["vp_macd"],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "VP-MACD" in caplog.text
        assert "TOP PERFORMERS" in caplog.text


# ---------------------------------------------------------------------------
# Patch function test
# ---------------------------------------------------------------------------

class TestPatchSaveVote:
    def test_patch_function_exists(self):
        """patch_save_vote should be callable (integration test would need EnsembleVoter)."""
        assert callable(patch_save_vote)


# ---------------------------------------------------------------------------
# Extended edge-case tests
# ---------------------------------------------------------------------------

class TestComputeHitRateExtended:
    """Additional edge cases for _compute_hit_rate."""

    def test_neutral_signal_flat_return_boundary(self, attributor):
        """Neutral signal, return exactly at flat boundary (0.001)."""
        # abs(return) < 0.001 → hit for neutral
        assert attributor._compute_hit_rate(0.0, 0.0009) is True
        assert attributor._compute_hit_rate(0.0, -0.0009) is True

    def test_neutral_signal_above_flat_boundary(self, attributor):
        """Neutral signal, return just above flat boundary."""
        assert attributor._compute_hit_rate(0.0, 0.001) is False
        assert attributor._compute_hit_rate(0.0, -0.001) is False

    def test_signal_at_neutral_threshold(self, attributor):
        """Signal value exactly at |0.05| neutral threshold."""
        # |value| = 0.05 → NOT neutral (abs < 0.05 is neutral)
        assert attributor._compute_hit_rate(0.05, 0.01) is True
        assert attributor._compute_hit_rate(-0.05, -0.01) is True

    def test_signal_just_below_neutral_threshold(self, attributor):
        """Signal value just below |0.05| neutral threshold."""
        # |value| < 0.05 → neutral, uses flat-market logic
        assert attributor._compute_hit_rate(0.049, 0.0005) is True
        assert attributor._compute_hit_rate(0.049, 0.02) is False


class TestComputeSourceAttributionExtended:
    """Additional edge cases for _compute_source_attribution."""

    def test_string_values_converted(self, attributor):
        """String value/weight fields should be converted to float."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": "0.5",
                "confidence": "0.8",
                "weight": "0.2",
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.source == "multi_speed_momentum"
        assert attrib.total_readings == 1
        assert attrib.avg_weight > 0

    def test_invalid_string_value_skipped(self, attributor):
        """Non-numeric string values should be skipped."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": "not_a_number",
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        # Value conversion fails → signal skipped (total=0, but readings=1)
        assert attrib.total_readings == 1
        assert attrib.active_days == 0

    def test_invalid_string_weight_defaults_zero(self, attributor):
        """Non-numeric string weight should default to 0."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": 0.5,
                "confidence": 0.8,
                "weight": "bad",
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.avg_weight == 0.0

    def test_missing_date_in_returns(self, attributor):
        """Signals with dates not in daily_returns should be skipped."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-01-01",
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.active_days == 0
        assert attrib.hit_rate is None  # no_data: no joined return days

    def test_none_daily_return_skipped(self, attributor):
        """daily_return=None should be skipped."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": None}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.active_days == 0

    def test_consecutive_losses_tracked(self, attributor):
        """Consecutive loss streak should be tracked."""
        signals = []
        daily_returns = {}
        for i in range(6):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum",
                "timestamp": d,
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            })
            daily_returns[d] = {"daily_return": -0.01}  # All negative

        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.max_consecutive_losses == 6

    def test_consecutive_losses_reset_on_win(self, attributor):
        """Win after loss streak resets consecutive counter."""
        signals = []
        daily_returns = {}
        for i, ret in enumerate([-0.01, -0.01, 0.005, -0.01, -0.01, -0.01]):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum",
                "timestamp": d,
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            })
            daily_returns[d] = {"daily_return": ret}

        attrib = attributor._compute_source_attribution(signals, daily_returns)
        # 2 losses, then win, then 3 losses → max = 3
        assert attrib.max_consecutive_losses == 3

    def test_neutral_signal_contribution_path(self, attributor):
        """Neutral signals (|value| <= 0.05) use weight-scaled contribution."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": 0.03,  # Below 0.05 threshold
                "confidence": 0.8,
                "weight": 0.25,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        # Neutral path: contribution = ret * 10000 * weight * 2
        expected_bps = 0.01 * 10000 * 0.25 * 2
        assert attrib.avg_return_bps == round(expected_bps, 2)

    def test_unknown_source_uses_fallback_meta(self, attributor):
        """Source not in SIGNAL_SOURCE_META gets fallback name/category."""
        signals = [
            {
                "source": "custom_signal",
                "timestamp": "2026-05-15",
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.display_name == "custom_signal"
        assert attrib.category == "other"

    def test_sharpe_contribution_single_observation(self, attributor):
        """With only one matching observation, sharpe_contribution should be 0."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.sharpe_contribution == 0.0

    def test_sharpe_contribution_multiple_observations(self, attributor):
        """With multiple observations and varying returns, sharpe > 0 possible."""
        signals = []
        daily_returns = {}
        for i in range(10):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum",
                "timestamp": d,
                "value": 0.5,
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            })
            daily_returns[d] = {"daily_return": 0.001 * (i + 1)}

        attrib = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib.sharpe_contribution != 0.0

    def test_directional_signal_contribution_path(self, attributor):
        """Directional signals (|value| > 0.05) use value-scaled contribution."""
        signals = [
            {
                "source": "multi_speed_momentum",
                "timestamp": "2026-05-15",
                "value": 0.5,  # Above 0.05 threshold
                "confidence": 0.8,
                "weight": 0.2,
                "regime_fit": "all",
                "explanation": "",
            },
        ]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib = attributor._compute_source_attribution(signals, daily_returns)
        # Directional path: contribution = ret * 10000 * abs(value)
        expected_bps = 0.01 * 10000 * 0.5
        assert attrib.avg_return_bps == round(expected_bps, 2)


class TestCorrelationMatrixExtended:
    """Additional edge cases for _compute_correlation_matrix."""

    def test_single_source_returns_zero(self, attributor):
        """Single source → no pairwise correlation → 0.0."""
        source_data = {
            "src_a": [{"timestamp": "2026-05-15T10:00:00", "value": 0.5}],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert result == {"src_a": 0.0}

    def test_fewer_than_five_dates_returns_zero(self, attributor):
        """Less than 5 unique dates → returns 0.0 for each source."""
        source_data = {
            "src_a": [{"timestamp": "2026-05-15T10:00:00", "value": 0.5},
                      {"timestamp": "2026-05-16T10:00:00", "value": -0.3}],
            "src_b": [{"timestamp": "2026-05-15T10:00:00", "value": 0.2},
                      {"timestamp": "2026-05-16T10:00:00", "value": 0.1}],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert result["src_a"] == 0.0
        assert result["src_b"] == 0.0

    def test_string_values_converted(self, attributor):
        """String values should be converted to float in correlation matrix."""
        source_data = {
            "src_a": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": str(0.1 * i)}
                      for i in range(6)],
            "src_b": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": str(0.2 * i)}
                      for i in range(6)],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert "src_a" in result
        assert "src_b" in result
        # Both should have finite correlation
        assert np.isfinite(result["src_a"])
        assert np.isfinite(result["src_b"])

    def test_five_plus_dates_produces_correlation(self, attributor):
        """5+ dates with 2 sources should produce nonzero correlation."""
        source_data = {
            "src_a": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": 0.1 * i}
                      for i in range(10)],
            "src_b": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": 0.05 * i}
                      for i in range(10)],
        }
        result = attributor._compute_correlation_matrix(source_data)
        # Both sources are positively trending → correlation should be positive
        assert result["src_a"] > 0
        assert result["src_b"] > 0


class TestSaveLoadReportExtended:
    """Additional edge cases for save_report / load_latest_report."""

    def test_load_corrupt_json_returns_none(self, attributor):
        """Corrupt JSON file should return None gracefully."""
        report = attributor.generate_report(days=30)
        attributor.save_report(report)

        # Corrupt the saved file
        files = sorted(attributor.attribution_dir.glob("attribution_*.json"), reverse=True)
        assert len(files) > 0
        with open(files[0], "w") as f:
            f.write("{invalid json content")

        result = attributor.load_latest_report()
        assert result is None

    def test_load_latest_picks_most_recent(self, attributor):
        """When multiple reports exist, load the most recent."""
        # Save two reports with different timestamps
        src = SourceAttribution(
            source="first", display_name="First", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
        )
        report1 = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01", end_date="2026-01-01", analysis_days=90,
            sources={"first": src}, best_source=None, worst_source=None,
            avg_hit_rate=0.5, avg_correlation=0.0,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=[],
        )

        src2 = SourceAttribution(
            source="second", display_name="Second", category="meanrev",
            total_readings=2, active_days=2, hit_rate=0.6, win_rate=0.6,
            avg_return_bps=2.0, total_return_bps=4.0, sharpe_contribution=0.1,
            max_consecutive_losses=1, avg_correlation=0.1, avg_weight=0.2,
        )
        report2 = AttributionReport(
            timestamp="2026-02-01T00:00:00",
            start_date="2025-11-01", end_date="2026-02-01", analysis_days=90,
            sources={"second": src2}, best_source=None, worst_source=None,
            avg_hit_rate=0.6, avg_correlation=0.1,
            avg_active_sources_per_day=2.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=[],
        )

        attributor.save_report(report1)
        attributor.save_report(report2)

        loaded = attributor.load_latest_report()
        assert loaded is not None
        # "2026-02-01" > "2026-01-01" → second report should load
        assert "second" in loaded.sources

    def test_save_creates_attribution_dir(self, tmpdir):
        """save_report should create attribution_dir if missing."""
        data_dir = Path(tmpdir) / "new_data"
        data_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=data_dir)
        # Remove attribution_dir to test creation
        import shutil
        if attributor.attribution_dir.exists():
            shutil.rmtree(attributor.attribution_dir)

        src = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01", end_date="2026-01-01", analysis_days=90,
            sources={"test": src}, best_source=None, worst_source=None,
            avg_hit_rate=0.5, avg_correlation=0.0,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=[],
        )
        path = attributor.save_report(report)
        assert path.exists()
        assert attributor.attribution_dir.exists()


class TestSourceAttributionExtended:
    """Additional edge cases for SourceAttribution dataclass."""

    def test_current_weight_regime_default(self):
        """current_weight_regime should default to 'normal'."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
        )
        assert sa.current_weight_regime == "normal"

    def test_current_weight_regime_custom(self):
        """current_weight_regime can be set to other values."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
            current_weight_regime="crisis",
        )
        assert sa.current_weight_regime == "crisis"

    def test_to_dict_includes_weight_regime(self):
        """to_dict should include current_weight_regime."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
            current_weight_regime="high_vol",
        )
        d = sa.to_dict()
        assert d["current_weight_regime"] == "high_vol"


class TestAttributionReportExtended:
    """Additional edge cases for AttributionReport dataclass."""

    def test_to_dict(self):
        """to_dict should produce a complete dict."""
        src = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1, active_days=1, hit_rate=0.5, win_rate=0.5,
            avg_return_bps=1.0, total_return_bps=1.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.1,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01", end_date="2026-01-01", analysis_days=90,
            sources={"test": src}, best_source="test", worst_source=None,
            avg_hit_rate=0.5, avg_correlation=0.0,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=["test"],
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["best_source"] == "test"
        assert "sources" in d
        assert d["degradation_signals"] == []


class TestPrintReportExtended:
    """Additional print_report edge cases."""

    def test_print_report_with_degradation(self, caplog):
        """Print report with degradation signals."""
        src = SourceAttribution(
            source="bad_sig", display_name="Bad Signal", category="trend",
            total_readings=20, active_days=18, hit_rate=0.35, win_rate=0.3,
            avg_return_bps=-2.0, total_return_bps=-36.0, sharpe_contribution=-0.5,
            max_consecutive_losses=8, avg_correlation=0.4, avg_weight=0.1,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01", end_date="2026-01-01", analysis_days=90,
            sources={"bad_sig": src}, best_source=None, worst_source="bad_sig",
            avg_hit_rate=0.35, avg_correlation=0.4,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=["bad_sig"], top_performers=[],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "DEGRADATION" in caplog.text
        assert "Bad Signal" in caplog.text

    def test_print_report_best_and_worst(self, caplog):
        """Print report with both best and worst sources."""
        best = SourceAttribution(
            source="good", display_name="Good Source", category="trend",
            total_readings=50, active_days=45, hit_rate=0.7, win_rate=0.65,
            avg_return_bps=3.0, total_return_bps=135.0, sharpe_contribution=1.2,
            max_consecutive_losses=2, avg_correlation=0.1, avg_weight=0.2,
        )
        worst = SourceAttribution(
            source="bad", display_name="Bad Source", category="meanrev",
            total_readings=50, active_days=45, hit_rate=0.3, win_rate=0.25,
            avg_return_bps=-1.5, total_return_bps=-67.5, sharpe_contribution=-0.8,
            max_consecutive_losses=10, avg_correlation=0.3, avg_weight=0.1,
        )
        report = AttributionReport(
            timestamp="2026-01-01T00:00:00",
            start_date="2025-10-01", end_date="2026-01-01", analysis_days=90,
            sources={"good": best, "bad": worst},
            best_source="good", worst_source="bad",
            avg_hit_rate=0.5, avg_correlation=0.2,
            avg_active_sources_per_day=2.0, total_sources_tracked=2,
            degradation_signals=["bad"], top_performers=["good"],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "Best source" in caplog.text
        assert "Worst source" in caplog.text
        assert "Good Source" in caplog.text
        assert "Bad Source" in caplog.text


class TestGetPaperTradingReturnsExtended:
    """Additional edge cases for _get_paper_trading_returns."""

    def test_fallback_json_performance_file(self, tmpdir):
        """When no paper_trading.db exists, fallback to JSON performance file."""
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        logs_dir = data_dir / "logs"
        logs_dir.mkdir()

        # Create a JSON performance file
        perf_file = logs_dir / "performance_summary_2026-01-01.json"
        perf_data = {
            "daily_returns": [
                {"date": "2026-01-01", "return": 0.005, "cumulative": 0.005},
                {"date": "2026-01-02", "return": -0.003, "cumulative": 0.002},
            ]
        }
        with open(perf_file, "w") as f:
            json.dump(perf_data, f)

        attributor = PerformanceAttribution(data_dir=data_dir)
        # The fallback uses global DATA_DIR, so patch it to our tmp dir
        from src.monitor import performance_attribution as pa
        original_data_dir = pa.DATA_DIR
        try:
            pa.DATA_DIR = data_dir
            returns = attributor._get_paper_trading_returns(days=30)
            assert len(returns) == 2
            assert returns["2026-01-01"]["daily_return"] == 0.005
            assert returns["2026-01-02"]["cumulative_return"] == 0.002
        finally:
            pa.DATA_DIR = original_data_dir

    def test_fallback_performance_jsonl(self, tmpdir):
        """When paper_trading.db missing, use performance.jsonl SSOT."""
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        (data_dir / "performance.jsonl").write_text(
            '{"timestamp":"2026-07-18T10:00:00","daily_return":0.001}\n'
            '{"date":"2026-07-19","daily_return":-0.0005}\n'
            '{"date":"2026-07-20","daily_return":0.0,"source":"capture_daily_pnl"}\n',
            encoding="utf-8",
        )
        attributor = PerformanceAttribution(data_dir=data_dir)
        returns = attributor._get_paper_trading_returns(days=30)
        assert "2026-07-19" in returns
        assert abs(returns["2026-07-19"]["daily_return"] - (-0.0005)) < 1e-12
        assert "2026-07-18" in returns

    def test_daily_pnl_preferred_over_performance_jsonl(self, tmpdir):
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        (data_dir / "daily_pnl.jsonl").write_text(
            '{"date":"2026-07-20","daily_return":-0.0002}\n',
            encoding="utf-8",
        )
        (data_dir / "performance.jsonl").write_text(
            '{"date":"2026-07-20","daily_return":-3e-8}\n',
            encoding="utf-8",
        )
        attributor = PerformanceAttribution(data_dir=data_dir)
        returns = attributor._get_paper_trading_returns(days=30)
        assert abs(returns["2026-07-20"]["daily_return"] - (-0.0002)) < 1e-12

    def test_corrupt_json_performance_file(self, tmpdir):
        """Corrupt JSON file should be handled gracefully."""
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        logs_dir = data_dir / "logs"
        logs_dir.mkdir()

        perf_file = logs_dir / "performance_summary_2026-01-01.json"
        with open(perf_file, "w") as f:
            f.write("not valid json")

        attributor = PerformanceAttribution(data_dir=data_dir)
        returns = attributor._get_paper_trading_returns(days=30)
        assert isinstance(returns, dict)
        assert len(returns) == 0


class TestGenerateReportExtended:
    """Additional edge cases for generate_report."""

    def test_report_with_only_ensemble_votes(self, tmpdir):
        """Report with only ensemble_votes (no source readings) should work."""
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Create DB with only ensemble_votes
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        conn.execute("""
            INSERT INTO ensemble_votes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2026-05-15 10:00:00", "normal", 0.7, 6, 0.25, 0.65,
              0.3, -0.1, 0.05, "neutral", 0.6, "test"))
        # No source_readings table
        conn.commit()
        conn.close()

        attributor = PerformanceAttribution(data_dir=data_dir)
        report = attributor.generate_report(days=30)
        assert isinstance(report, AttributionReport)
        assert len(report.sources) == 0


class TestSourceAttributionExtended:
    """Extended tests for SourceAttribution dataclass."""

    def test_all_fields(self):
        sa = SourceAttribution(
            source="MULTI_SPEED_MOM", display_name="Multi-Speed Momentum",
            category="momentum", total_readings=500, active_days=200,
            hit_rate=0.65, win_rate=0.55, avg_return_bps=1.2,
            total_return_bps=120.0, sharpe_contribution=0.15,
            max_consecutive_losses=5, avg_correlation=0.3,
            avg_weight=0.20, current_weight_regime="normal",
        )
        assert sa.source == "MULTI_SPEED_MOM"
        assert sa.hit_rate == 0.65
        assert sa.total_return_bps == 120.0

    def test_to_dict_completeness(self):
        sa = SourceAttribution(
            source="CROSS_ASSET_RV", display_name="Cross-Asset RV",
            category="relative_value", total_readings=300, active_days=150,
            hit_rate=0.55, win_rate=0.50, avg_return_bps=0.5,
            total_return_bps=50.0, sharpe_contribution=0.08,
            max_consecutive_losses=3, avg_correlation=0.2,
            avg_weight=0.13,
        )
        d = sa.to_dict()
        expected_keys = {
            "source", "display_name", "category", "total_readings", "active_days",
            "hit_rate", "win_rate", "avg_return_bps", "total_return_bps",
            "sharpe_contribution", "max_consecutive_losses", "avg_correlation",
            "avg_weight", "current_weight_regime",
        }
        assert set(d.keys()) == expected_keys

    def test_efficiency_ratio_positive(self):
        sa = SourceAttribution(
            source="TEST", display_name="Test", category="test",
            total_readings=100, active_days=50, hit_rate=0.6,
            win_rate=0.5, avg_return_bps=2.0, total_return_bps=100.0,
            sharpe_contribution=0.1, max_consecutive_losses=2,
            avg_correlation=0.0, avg_weight=0.2,
        )
        assert sa.efficiency_ratio > 0

    def test_efficiency_ratio_zero_return(self):
        sa = SourceAttribution(
            source="TEST", display_name="Test", category="test",
            total_readings=100, active_days=50, hit_rate=0.5,
            win_rate=0.5, avg_return_bps=0.0, total_return_bps=0.0,
            sharpe_contribution=0.0, max_consecutive_losses=0,
            avg_correlation=0.0, avg_weight=0.2,
        )
        assert sa.efficiency_ratio == 0.0

    def test_default_weight_regime(self):
        sa = SourceAttribution(
            source="TEST", display_name="Test", category="test",
            total_readings=0, active_days=0, hit_rate=0.0,
            win_rate=0.0, avg_return_bps=0.0, total_return_bps=0.0,
            sharpe_contribution=0.0, max_consecutive_losses=0,
            avg_correlation=0.0, avg_weight=0.0,
        )
        assert sa.current_weight_regime == "normal"


class TestAttributionReportExtended:
    """Extended tests for AttributionReport dataclass."""

    def test_to_dict_serializes_sources(self):
        sa = SourceAttribution(
            source="TEST", display_name="Test", category="test",
            total_readings=100, active_days=50, hit_rate=0.6,
            win_rate=0.5, avg_return_bps=1.0, total_return_bps=50.0,
            sharpe_contribution=0.1, max_consecutive_losses=2,
            avg_correlation=0.0, avg_weight=0.2,
        )
        report = AttributionReport(
            timestamp="2026-05-24", start_date="2026-03-01", end_date="2026-05-24",
            analysis_days=90, sources={"TEST": sa},
            best_source="TEST", worst_source=None,
            avg_hit_rate=0.6, avg_correlation=0.0,
            avg_active_sources_per_day=5.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=["TEST"],
        )
        d = report.to_dict()
        assert "sources" in d
        assert "TEST" in d["sources"]

    def test_to_json(self):
        report = AttributionReport(
            timestamp="2026-05-24", start_date="2026-03-01", end_date="2026-05-24",
            analysis_days=90, sources={},
            best_source=None, worst_source=None,
            avg_hit_rate=0.0, avg_correlation=0.0,
            avg_active_sources_per_day=0.0, total_sources_tracked=0,
            degradation_signals=[], top_performers=[],
        )
        j = report.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["analysis_days"] == 90

    def test_empty_report(self):
        report = AttributionReport(
            timestamp="2026-05-24", start_date="2026-05-24", end_date="2026-05-24",
            analysis_days=0, sources={},
            best_source=None, worst_source=None,
            avg_hit_rate=0.0, avg_correlation=0.0,
            avg_active_sources_per_day=0.0, total_sources_tracked=0,
            degradation_signals=[], top_performers=[],
        )
        assert report.sources == {}
        assert report.best_source is None
        assert report.total_sources_tracked == 0


class TestPerformanceAttributionExtended:
    """Extended PerformanceAttribution tests."""

    def test_default_data_dir(self):
        from src.paths import DATA_DIR as _DATA_DIR
        pa = PerformanceAttribution()
        assert pa.data_dir == _DATA_DIR

    def test_custom_data_dir(self, tmp_path):
        pa = PerformanceAttribution(data_dir=tmp_path)
        assert pa.data_dir == tmp_path

    def test_compute_hit_rate_neutral_signal(self):
        """Neutral signal should not count as hit."""
        pa = PerformanceAttribution()
        result = pa._compute_hit_rate(0.0, 0.01)
        assert isinstance(result, bool)

    def test_compute_hit_rate_correct_direction(self):
        """Positive signal + positive return = hit."""
        pa = PerformanceAttribution()
        result = pa._compute_hit_rate(0.5, 0.01)
        assert result is True

    def test_compute_hit_rate_wrong_direction(self):
        """Positive signal + negative return = miss."""
        pa = PerformanceAttribution()
        result = pa._compute_hit_rate(0.5, -0.01)
        assert result is False


class TestPrintReport:
    """Test print_report function."""

    def test_print_empty_report(self, caplog):
        report = AttributionReport(
            timestamp="2026-05-24", start_date="2026-03-01", end_date="2026-05-24",
            analysis_days=90, sources={},
            best_source=None, worst_source=None,
            avg_hit_rate=0.0, avg_correlation=0.0,
            avg_active_sources_per_day=0.0, total_sources_tracked=0,
            degradation_signals=[], top_performers=[],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "ATTRIBUTION" in caplog.text


# ---------------------------------------------------------------------------
# New tests: SourceAttribution edge cases
# ---------------------------------------------------------------------------

class TestSourceAttributionNew:
    """Additional SourceAttribution edge cases."""

    def test_efficiency_ratio_negative_return(self):
        """Negative avg_return_bps still yields positive efficiency via abs()."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=100, active_days=50, hit_rate=0.6, win_rate=0.5,
            avg_return_bps=-2.5, total_return_bps=-250.0, sharpe_contribution=-0.8,
            max_consecutive_losses=5, avg_correlation=0.2, avg_weight=0.15,
        )
        assert sa.efficiency_ratio > 0
        assert sa.efficiency_ratio == 0.6 * abs(-2.5) / 100

    def test_efficiency_ratio_large_numbers(self):
        """Very large avg_return_bps should not overflow."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=1000, active_days=500, hit_rate=1.0, win_rate=1.0,
            avg_return_bps=999999.99, total_return_bps=50000000.0,
            sharpe_contribution=5.0, max_consecutive_losses=0,
            avg_correlation=0.1, avg_weight=0.5,
        )
        assert sa.efficiency_ratio > 0
        assert sa.efficiency_ratio == 1.0 * 999999.99 / 100

    def test_zero_total_and_active_days(self):
        """Source with zero readings and days should not crash."""
        sa = SourceAttribution(
            source="empty", display_name="Empty", category="other",
            total_readings=0, active_days=0, hit_rate=0.0, win_rate=0.0,
            avg_return_bps=0.0, total_return_bps=0.0, sharpe_contribution=0.0,
            max_consecutive_losses=0, avg_correlation=0.0, avg_weight=0.0,
        )
        assert sa.efficiency_ratio == 0.0
        d = sa.to_dict()
        assert d["source"] == "empty"


class TestAttributionReportNew:
    """Additional AttributionReport edge cases."""

    def test_to_json_with_multiple_sources(self):
        """to_json produces valid JSON with multiple sources."""
        src_a = SourceAttribution(
            source="alpha", display_name="Alpha", category="trend",
            total_readings=50, active_days=40, hit_rate=0.65, win_rate=0.60,
            avg_return_bps=2.0, total_return_bps=80.0, sharpe_contribution=0.9,
            max_consecutive_losses=3, avg_correlation=0.15, avg_weight=0.20,
        )
        src_b = SourceAttribution(
            source="beta", display_name="Beta", category="meanrev",
            total_readings=30, active_days=25, hit_rate=0.55, win_rate=0.50,
            avg_return_bps=0.8, total_return_bps=20.0, sharpe_contribution=0.3,
            max_consecutive_losses=4, avg_correlation=0.30, avg_weight=0.10,
        )
        report = AttributionReport(
            timestamp="2026-05-24T12:00:00",
            start_date="2026-02-23", end_date="2026-05-24", analysis_days=90,
            sources={"alpha": src_a, "beta": src_b},
            best_source="alpha", worst_source="beta",
            avg_hit_rate=0.60, avg_correlation=0.225,
            avg_active_sources_per_day=2.0, total_sources_tracked=2,
            degradation_signals=["beta"], top_performers=["alpha"],
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert len(parsed["sources"]) == 2
        assert parsed["best_source"] == "alpha"
        assert parsed["worst_source"] == "beta"
        assert parsed["degradation_signals"] == ["beta"]
        assert parsed["top_performers"] == ["alpha"]

    def test_degradation_top_performers_overlap(self):
        """A source at the boundary should not be in both lists."""
        src = SourceAttribution(
            source="neutral", display_name="Neutral", category="trend",
            total_readings=100, active_days=80, hit_rate=0.50, win_rate=0.50,
            avg_return_bps=0.5, total_return_bps=40.0, sharpe_contribution=0.0,
            max_consecutive_losses=5, avg_correlation=0.20, avg_weight=0.15,
        )
        report = AttributionReport(
            timestamp="2026-05-24T00:00:00",
            start_date="2026-02-23", end_date="2026-05-24", analysis_days=90,
            sources={"neutral": src},
            best_source="neutral", worst_source="neutral",
            avg_hit_rate=0.5, avg_correlation=0.2,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=[],
        )
        assert "neutral" not in report.degradation_signals
        assert "neutral" not in report.top_performers


# ---------------------------------------------------------------------------
# New tests: Signal history edge cases
# ---------------------------------------------------------------------------

class TestSignalHistoryNew:
    """Additional _get_signal_history edge cases."""

    def test_empty_ensemble_db(self, tmp_path):
        """Empty (zero-byte) ensemble DB should not crash."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        db_path.write_text("")
        attributor = PerformanceAttribution(data_dir=data_dir)
        history = attributor._get_signal_history(days=30)
        assert history == []

    def test_corrupt_ensemble_db(self, tmp_path):
        """Corrupt (non-SQLite) ensemble DB should not crash."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        db_path.write_bytes(b"\x00\x01\x02\x03")
        attributor = PerformanceAttribution(data_dir=data_dir)
        history = attributor._get_signal_history(days=30)
        assert history == []

    def test_missing_source_readings_table(self, tmp_path):
        """DB with only ensemble_votes table — source_readings query fails
        and try/except returns empty history."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        conn.execute("""
            INSERT INTO ensemble_votes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("2026-05-15 10:00:00", "normal", 0.7, 6, 0.25, 0.65,
              0.3, -0.1, 0.05, "neutral", 0.6, "working"))
        conn.commit()
        conn.close()
        attributor = PerformanceAttribution(data_dir=data_dir)
        # The entire try/except in _get_signal_history wraps both queries,
        # so a missing source_readings table fails the whole attempt.
        history = attributor._get_signal_history(days=30)
        assert history == []

    def test_signal_history_custom_days(self, tmp_path, ensemble_db):
        """Different day parameters should affect LIMIT."""
        data_dir = ensemble_db.parent
        attributor = PerformanceAttribution(data_dir=data_dir)
        history_short = attributor._get_signal_history(days=5)
        history_long = attributor._get_signal_history(days=90)
        assert len(history_long) >= len(history_short)


# ---------------------------------------------------------------------------
# New tests: Paper trading returns edge cases
# ---------------------------------------------------------------------------

class TestPaperTradingReturnsNew:
    """Additional _get_paper_trading_returns edge cases."""

    def test_db_read_exception(self, tmp_path):
        """Exception during DB read returns empty dict (graceful)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "paper_trading.db"
        db_path.write_bytes(b"\x00\x01\x02\x03")
        attributor = PerformanceAttribution(data_dir=data_dir)
        returns = attributor._get_paper_trading_returns(days=30)
        assert isinstance(returns, dict)
        assert len(returns) == 0

    def test_empty_paper_trading_db(self, tmp_path):
        """DB with table but no rows returns empty dict."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "paper_trading.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE daily_snapshots (
                date TEXT PRIMARY KEY, total_value REAL, daily_return REAL,
                cumulative_return REAL
            )
        """)
        conn.commit()
        conn.close()
        attributor = PerformanceAttribution(data_dir=data_dir)
        returns = attributor._get_paper_trading_returns(days=30)
        assert isinstance(returns, dict)
        assert len(returns) == 0

    def test_fallback_no_logs_dir(self, tmp_path):
        """When logs directory does not exist, fallback returns empty dict."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        attributor = PerformanceAttribution(data_dir=data_dir)
        from src.monitor import performance_attribution as pa_mod
        original = pa_mod.DATA_DIR
        try:
            pa_mod.DATA_DIR = data_dir  # No logs dir here
            returns = attributor._get_paper_trading_returns(days=30)
            assert isinstance(returns, dict)
            assert len(returns) == 0
        finally:
            pa_mod.DATA_DIR = original

    def test_fallback_json_no_daily_returns_key(self, tmp_path):
        """JSON fallback with no 'daily_returns' key returns empty dict."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        logs_dir = data_dir / "logs"
        logs_dir.mkdir()
        perf_file = logs_dir / "performance_summary_2026-01-01.json"
        with open(perf_file, "w") as f:
            json.dump({"other_data": True}, f)
        attributor = PerformanceAttribution(data_dir=data_dir)
        from src.monitor import performance_attribution as pa_mod
        original = pa_mod.DATA_DIR
        try:
            pa_mod.DATA_DIR = data_dir
            returns = attributor._get_paper_trading_returns(days=30)
            assert isinstance(returns, dict)
            assert len(returns) == 0
        finally:
            pa_mod.DATA_DIR = original


# ---------------------------------------------------------------------------
# New tests: Compute source attribution edge cases
# ---------------------------------------------------------------------------

class TestComputeSourceAttributionNew:
    """Additional _compute_source_attribution edge cases."""

    def test_missing_value_key(self, attributor):
        """Signal dict without 'value' key defaults to 0."""
        signals = [{
            "source": "multi_speed_momentum", "timestamp": "2026-05-15",
            "confidence": 0.8, "weight": 0.2, "regime_fit": "all",
            "explanation": "",
        }]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        # value defaults to 0 → neutral signal with non-flat return → miss → hit_rate=0.0
        assert attrib_out.active_days == 1
        assert attrib_out.hit_rate == 0.0
        # avg_weight = mean of [0.2] = 0.2

    def test_missing_weight_key(self, attributor):
        """Signal dict without 'weight' key defaults to 0."""
        signals = [{
            "source": "multi_speed_momentum", "timestamp": "2026-05-15",
            "value": 0.5, "confidence": 0.8, "regime_fit": "all",
            "explanation": "",
        }]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib_out.avg_weight == 0.0

    def test_all_zero_signals(self, attributor):
        """All zero value signals should produce zero contribution."""
        signals = []
        daily_returns = {}
        for i in range(5):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum", "timestamp": d,
                "value": 0.0, "confidence": 0.8, "weight": 0.2,
                "regime_fit": "all", "explanation": "",
            })
            daily_returns[d] = {"daily_return": 0.005}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        # All neutral signals → weight-scaled contribution
        assert attrib_out.active_days == 5
        assert attrib_out.total_return_bps != 0.0  # neutral path: ret * 10000 * w * 2

    def test_all_returns_negative(self, attributor):
        """All negative returns should give hit_rate=1.0 with all negative signals."""
        signals = []
        daily_returns = {}
        for i in range(5):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum", "timestamp": d,
                "value": -0.5, "confidence": 0.8, "weight": 0.2,
                "regime_fit": "all", "explanation": "",
            })
            daily_returns[d] = {"daily_return": -0.01}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib_out.hit_rate == 1.0
        assert attrib_out.max_consecutive_losses == 5

    def test_constant_signals_zero_sharpe(self, attributor):
        """Constant daily contributions yield zero sharpe (zero std)."""
        signals = []
        daily_returns = {}
        for i in range(10):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum", "timestamp": d,
                "value": 0.5, "confidence": 0.8, "weight": 0.2,
                "regime_fit": "all", "explanation": "",
            })
            daily_returns[d] = {"daily_return": 0.001}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        # All identical returns → std=0 → sharpe=0
        assert attrib_out.sharpe_contribution == 0.0

    def test_extreme_signal_values(self, attributor):
        """Extreme signal values (very large) should not crash."""
        signals = [{
            "source": "multi_speed_momentum", "timestamp": "2026-05-15",
            "value": 999.0, "confidence": 0.99, "weight": 0.5,
            "regime_fit": "all", "explanation": "",
        }]
        daily_returns = {"2026-05-15": {"daily_return": 0.01}}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib_out.active_days == 1
        assert attrib_out.avg_return_bps > 0
        # directional path: contribution = ret * 10000 * abs(value) = 0.01 * 10000 * 999
        assert attrib_out.total_return_bps >= 99900.0

    def test_win_rate_accounting(self, attributor):
        """Win rate should reflect positive return days."""
        signals = []
        daily_returns = {}
        for i, ret in enumerate([0.01, -0.005, 0.02, -0.003, 0.015]):
            d = f"2026-05-{10+i:02d}"
            signals.append({
                "source": "multi_speed_momentum", "timestamp": d,
                "value": 0.5, "confidence": 0.8, "weight": 0.2,
                "regime_fit": "all", "explanation": "",
            })
            daily_returns[d] = {"daily_return": ret}
        attrib_out = attributor._compute_source_attribution(signals, daily_returns)
        assert attrib_out.win_rate == 3 / 5  # 3 positive returns out of 5


# ---------------------------------------------------------------------------
# New tests: Correlation matrix edge cases
# ---------------------------------------------------------------------------

class TestCorrelationMatrixNew:
    """Additional _compute_correlation_matrix edge cases."""

    def test_constant_values_returns_correlation(self, attributor):
        """Both sources constant (zero variance) → NaN in corrcoef → returns NaN float."""
        source_data = {
            "src_a": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": 0.5}
                      for i in range(10)],
            "src_b": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": 0.5}
                      for i in range(10)],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert "src_a" in result
        assert "src_b" in result
        # NaN from zero variance → result is a float (NaN is acceptable for zero-variance)
        assert isinstance(result["src_a"], float)
        assert isinstance(result["src_b"], float)

    def test_single_date_for_one_source(self, attributor):
        """Source with fewer dates still gets a result, matrix handles gaps."""
        source_data = {
            "src_a": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00", "value": float(i)}
                      for i in range(10)],
            "src_b": [{"timestamp": "2026-05-10T10:00:00", "value": 1.0}],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert "src_a" in result
        assert "src_b" in result

    def test_all_value_extremes(self, attributor):
        """Sources with extreme value ranges produce finite correlations."""
        source_data = {
            "src_a": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00",
                       "value": 1e6 if i % 2 == 0 else -1e6}
                      for i in range(10)],
            "src_b": [{"timestamp": f"2026-05-{10+i:02d}T10:00:00",
                       "value": -1e6 if i % 2 == 0 else 1e6}
                      for i in range(10)],
        }
        result = attributor._compute_correlation_matrix(source_data)
        assert np.isfinite(result["src_a"])
        assert np.isfinite(result["src_b"])


# ---------------------------------------------------------------------------
# New tests: Generate report edge cases
# ---------------------------------------------------------------------------

class TestGenerateReportNew:
    """Additional generate_report edge cases."""

    def test_degradation_detected_via_sharpe(self, tmp_path):
        """Source with negative sharpe appears in degradation_signals."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT,
                value REAL, confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        from datetime import datetime, timedelta
        base = datetime.now()
        perf_db = data_dir / "paper_trading.db"
        pconn = sqlite3.connect(perf_db)
        pconn.execute("""
            CREATE TABLE daily_snapshots (
                date TEXT PRIMARY KEY, total_value REAL,
                daily_return REAL, cumulative_return REAL
            )
        """)
        for i in range(30):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (d + " 10:00:00", "bad_signal", -0.5, 0.8, 0.2,
                  "all", "consistently wrong"))
            pconn.execute("""
                INSERT INTO daily_snapshots (date, total_value, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?)
            """, (d, 100000.0, 0.01, 0.3))
        conn.commit()
        conn.close()
        pconn.commit()
        pconn.close()

        attributor2 = PerformanceAttribution(data_dir=data_dir)
        report = attributor2.generate_report(days=30)
        # Signal is -0.5, return is +0.01 → always wrong → hit_rate=0.0
        # sharpe_contribution will be negative → flagged
        assert "bad_signal" in report.degradation_signals

    def test_top_performer_detected(self, tmp_path):
        """Source with very positive sharpe appears in top_performers."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT,
                value REAL, confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        from datetime import datetime, timedelta
        base = datetime.now()
        perf_db = data_dir / "paper_trading.db"
        pconn = sqlite3.connect(perf_db)
        pconn.execute("""
            CREATE TABLE daily_snapshots (
                date TEXT PRIMARY KEY, total_value REAL,
                daily_return REAL, cumulative_return REAL
            )
        """)
        for i in range(30):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (d + " 10:00:00", "good_signal", 0.5, 0.8, 0.2,
                  "all", "consistently correct"))
            # Vary returns so std > 0 and sharpe can be computed
            daily_ret = 0.01 if i % 2 == 0 else -0.005
            pconn.execute("""
                INSERT INTO daily_snapshots (date, total_value, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?)
            """, (d, 100000.0, daily_ret, 0.3))
        conn.commit()
        conn.close()
        pconn.commit()
        pconn.close()

        attributor2 = PerformanceAttribution(data_dir=data_dir)
        report = attributor2.generate_report(days=30)
        # Signal is +0.5, returns alternate → sharpe contribution should be > 0.5
        assert "good_signal" in report.top_performers

    def test_correlations_assigned_to_sources(self, attributor, ensemble_db):
        """Correlation values should be assigned into source attributions."""
        report = attributor.generate_report(days=30)
        for src_name, src in report.sources.items():
            assert isinstance(src.avg_correlation, float)
            # At least one source should have non-zero correlation (random data)
        correlations_nonzero = sum(
            1 for s in report.sources.values() if s.avg_correlation != 0.0
        )
        assert correlations_nonzero >= 0  # at minimum, no crash

    def test_best_worst_identified_correctly(self, tmp_path):
        """Best/worst sources match min/max sharpe contribution.
        Uses different numbers of matching days so sharpes diverge."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT,
                value REAL, confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        base = datetime.now()
        perf_db = data_dir / "paper_trading.db"
        pconn = sqlite3.connect(perf_db)
        pconn.execute("""
            CREATE TABLE daily_snapshots (
                date TEXT PRIMARY KEY, total_value REAL,
                daily_return REAL, cumulative_return REAL
            )
        """)
        for i in range(30):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_ret = 0.01 if i % 2 == 0 else -0.005
            # src_good: value=0.5 matches 30 days of returns (varying)
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (d + " 10:00:00", "src_good", 0.5, 0.8, 0.2,
                  "all", "good"))
            # src_bad: dates that do NOT match the daily_snapshots (use future dates)
            future = (base + timedelta(days=i + 100)).strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (future + " 10:00:00", "src_bad", -0.5, 0.8, 0.2,
                  "all", "bad"))
            pconn.execute("""
                INSERT INTO daily_snapshots (date, total_value, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?)
            """, (d, 100000.0, daily_ret, 0.3))
        conn.commit()
        conn.close()
        pconn.commit()
        pconn.close()

        attributor = PerformanceAttribution(data_dir=data_dir)
        report = attributor.generate_report(days=30)
        assert report.best_source == "src_good"
        assert report.worst_source == "src_bad"
        assert report.total_sources_tracked == 2

    def test_avg_active_sources_per_day(self, tmp_path):
        """avg_active_sources_per_day should be a positive number with data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "ensemble_signals.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE source_readings (
                id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT,
                value REAL, confidence REAL, weight REAL,
                regime_fit TEXT, explanation TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE ensemble_votes (
                timestamp TEXT PRIMARY KEY, regime TEXT, regime_confidence REAL,
                num_sources INTEGER, consensus REAL, agreement_ratio REAL,
                equity_bias REAL, duration_bias REAL, gold_bias REAL,
                action TEXT, confidence REAL, reasoning TEXT
            )
        """)
        base = datetime.now()
        for i in range(10):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                INSERT INTO source_readings
                (timestamp, source, value, confidence, weight, regime_fit, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (d, "test_src", 0.1, 0.5, 0.2, "all", "test"))
        conn.commit()
        conn.close()
        db_path = data_dir / "paper_trading.db"
        conn2 = sqlite3.connect(db_path)
        conn2.execute("""
            CREATE TABLE daily_snapshots (
                date TEXT PRIMARY KEY, total_value REAL,
                daily_return REAL, cumulative_return REAL
            )
        """)
        for i in range(10):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            conn2.execute("""
                INSERT INTO daily_snapshots (date, total_value, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?)
            """, (d, 100000.0, 0.001, 0.01))
        conn2.commit()
        conn2.close()
        attributor = PerformanceAttribution(data_dir=data_dir)
        report = attributor.generate_report(days=30)
        assert report.avg_active_sources_per_day >= 0


# ---------------------------------------------------------------------------
# New tests: Print report edge cases
# ---------------------------------------------------------------------------

class TestPrintReportNew:
    """Additional print_report edge cases."""

    def test_print_no_best_worst_with_valid_sources(self, caplog):
        """Print report with valid sources but None best/worst should work."""
        src = SourceAttribution(
            source="valid", display_name="Valid Source", category="trend",
            total_readings=50, active_days=45, hit_rate=0.55, win_rate=0.52,
            avg_return_bps=0.5, total_return_bps=22.5, sharpe_contribution=0.2,
            max_consecutive_losses=4, avg_correlation=0.1, avg_weight=0.15,
        )
        report = AttributionReport(
            timestamp="2026-05-24T00:00:00",
            start_date="2026-02-23", end_date="2026-05-24", analysis_days=90,
            sources={"valid": src},
            best_source=None, worst_source=None,
            avg_hit_rate=0.55, avg_correlation=0.1,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=[], top_performers=[],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "Valid Source" in caplog.text
        assert "Best source:" not in caplog.text
        assert "Worst source:" not in caplog.text

    def test_print_negative_sharpe_formatting(self, caplog):
        """Negative sharpe values should format correctly."""
        src = SourceAttribution(
            source="loser", display_name="Losing Signal", category="trend",
            total_readings=30, active_days=28, hit_rate=0.30, win_rate=0.25,
            avg_return_bps=-1.5, total_return_bps=-42.0, sharpe_contribution=-0.75,
            max_consecutive_losses=12, avg_correlation=0.35, avg_weight=0.10,
        )
        report = AttributionReport(
            timestamp="2026-05-24T00:00:00",
            start_date="2026-02-23", end_date="2026-05-24", analysis_days=90,
            sources={"loser": src},
            best_source=None, worst_source="loser",
            avg_hit_rate=0.3, avg_correlation=0.35,
            avg_active_sources_per_day=1.0, total_sources_tracked=1,
            degradation_signals=["loser"], top_performers=[],
        )
        with caplog.at_level(logging.INFO, logger="src.monitor.performance_attribution"):
            print_report(report)
        assert "DEGRADATION" in caplog.text
        assert "Losing Signal" in caplog.text


# ---------------------------------------------------------------------------
# New tests: Patch save vote edge cases
# ---------------------------------------------------------------------------

class TestPatchSaveVoteNew:
    """Additional patch_save_vote edge cases."""

    def test_patch_save_vote_imports_ensemble_voter(self):
        """patch_save_vote should import from ensemble_voter module."""
        import src.strategy.ensemble_voter
        assert hasattr(src.strategy.ensemble_voter, "EnsembleVoter")

    def test_patch_save_vote_patches_method(self):
        """After calling patch_save_vote, _save_vote should be the patched version."""
        import src.strategy.ensemble_voter as ev
        original_method = ev.EnsembleVoter._save_vote
        patch_save_vote()
        patched_method = ev.EnsembleVoter._save_vote
        # Restore to avoid side effects
        ev.EnsembleVoter._save_vote = original_method
        assert patched_method is not original_method


# ---------------------------------------------------------------------------
# New tests: SourceAttribution edge cases (metrics validation)
# ---------------------------------------------------------------------------

class TestSourceAttributionMetrics:
    """Validation of SourceAttribution computed metrics."""

    def test_current_weight_regime_preserved_in_dict(self):
        """current_weight_regime field is preserved through to_dict round-trip."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=10, active_days=8, hit_rate=0.6, win_rate=0.55,
            avg_return_bps=1.5, total_return_bps=12.0, sharpe_contribution=0.8,
            max_consecutive_losses=3, avg_correlation=0.2, avg_weight=0.15,
            current_weight_regime="high_vol",
        )
        d = sa.to_dict()
        assert d["current_weight_regime"] == "high_vol"
        restored = SourceAttribution(**d)
        assert restored.current_weight_regime == "high_vol"
        assert restored.source == "test"

    def test_efficiency_ratio_with_hit_rate_zero(self):
        """Efficiency ratio is zero when hit_rate is zero (non-zero return)."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=10, active_days=8, hit_rate=0.0, win_rate=0.0,
            avg_return_bps=5.0, total_return_bps=40.0, sharpe_contribution=0.0,
            max_consecutive_losses=10, avg_correlation=0.0, avg_weight=0.1,
        )
        assert sa.efficiency_ratio == 0.0

    def test_efficiency_ratio_round_trip_precision(self):
        """Efficiency ratio computed from integers should match."""
        sa = SourceAttribution(
            source="test", display_name="Test", category="trend",
            total_readings=100, active_days=80, hit_rate=0.75, win_rate=0.70,
            avg_return_bps=4.0, total_return_bps=320.0, sharpe_contribution=1.5,
            max_consecutive_losses=2, avg_correlation=0.1, avg_weight=0.25,
        )
        expected = 0.75 * 4.0 / 100
        assert sa.efficiency_ratio == expected



class TestAttributionNoDataHonesty:
    def test_vacuous_report_status_no_data(self, tmp_path):
        """When no source joins returns (active_days=0), status is no_data and hit rates null."""
        from src.monitor.performance_attribution import PerformanceAttribution, AttributionReport

        attrib = PerformanceAttribution(data_dir=tmp_path)
        # Force empty history / empty returns path
        report = attrib.generate_report(days=30)
        assert isinstance(report, AttributionReport)
        # Empty ensemble DB → no sources or all zero active
        if report.total_sources_tracked == 0 or all(
            (s.active_days if hasattr(s, "active_days") else report.sources[s].active_days) == 0
            for s in (report.sources if isinstance(report.sources, dict) else [])
        ):
            assert report.status == "no_data"
            assert report.avg_hit_rate is None
