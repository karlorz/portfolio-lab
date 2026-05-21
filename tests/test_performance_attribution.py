#!/usr/bin/env python3
"""
Tests for Performance Attribution System (v5.70).
"""

import sys
import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    def test_print_report_empty(self, capsys):
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
        print_report(report)
        captured = capsys.readouterr()
        assert "PERFORMANCE ATTRIBUTION REPORT" in captured.out

    def test_print_report_with_data(self, capsys):
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
        print_report(report)
        captured = capsys.readouterr()
        assert "VP-MACD" in captured.out
        assert "TOP PERFORMERS" in captured.out


# ---------------------------------------------------------------------------
# Patch function test
# ---------------------------------------------------------------------------

class TestPatchSaveVote:
    def test_patch_function_exists(self):
        """patch_save_vote should be callable (integration test would need EnsembleVoter)."""
        assert callable(patch_save_vote)
