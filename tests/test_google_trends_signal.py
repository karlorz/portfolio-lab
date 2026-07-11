"""Tests for Google Trends sentiment signal.

Replaces the net-negative behavioral sentiment signal (VIX-proxy, -0.216 Sharpe,
65.8% false positive rate) with search volume data for macro fear indicators.

Data source: data/google_trends.json (cached, populated by fetch script or cron).
Signal construction: Z-score of 7-day rolling search volume relative to 90-day mean.
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.signals.google_trends_signal import (
    GoogleTrendsSignal,
    TREND_TERMS,
    FEAR_THRESHOLD,
    GREED_THRESHOLD,
)


def _make_trend_data(days=100, base_volume=50, recent_volume=None):
    """Helper to create mock trend data."""
    data = {}
    today = datetime.now()
    for i in range(days):
        date = (today - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        if recent_volume and i >= days - 7:
            vol = recent_volume
        else:
            vol = base_volume + (i % 10)  # Some variation
        data[date] = vol
    return data


class TestGoogleTrendsConstants:
    """Test module constants."""

    def test_trend_terms_not_empty(self):
        """Must track at least one search term."""
        assert len(TREND_TERMS) >= 1

    def test_fear_threshold_positive(self):
        """Fear threshold should be positive Z-score."""
        assert FEAR_THRESHOLD > 0

    def test_greed_threshold_negative(self):
        """Greed threshold should be negative Z-score."""
        assert GREED_THRESHOLD < 0


class TestGoogleTrendsSignal:
    """Test GoogleTrendsSignal class."""

    def test_init_no_file(self):
        """Signal initializes even when data file doesn't exist."""
        signal = GoogleTrendsSignal(data_path="/nonexistent/path.json")
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is False
        assert snapshot.value == 0.0

    def test_init_with_data(self, tmp_path):
        """Signal initializes and reads data from file."""
        data_file = tmp_path / "google_trends.json"
        trend_data = _make_trend_data()
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        assert signal._data is not None

    def test_snapshot_with_no_data(self, tmp_path):
        """Empty data file returns inactive snapshot."""
        data_file = tmp_path / "google_trends.json"
        data_file.write_text("{}")

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is False

    def test_snapshot_with_normal_data(self, tmp_path):
        """Normal search volume returns near-zero signal."""
        data_file = tmp_path / "google_trends.json"
        # Steady volume = low Z-score
        trend_data = _make_trend_data(days=100, base_volume=50, recent_volume=52)
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is True
        assert abs(snapshot.value) < 0.5  # Near zero = neutral

    def test_snapshot_with_fear_spike(self, tmp_path):
        """Spiking search volume produces negative (fear) signal."""
        data_file = tmp_path / "google_trends.json"
        # Recent volume 3x normal = fear spike
        trend_data = _make_trend_data(days=100, base_volume=50, recent_volume=150)
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is True
        assert snapshot.value < 0  # Fear = negative signal

    def test_snapshot_with_greed(self, tmp_path):
        """Very low search volume produces positive (greed) signal."""
        data_file = tmp_path / "google_trends.json"
        # Recent volume much lower than average = complacency
        trend_data = _make_trend_data(days=100, base_volume=50, recent_volume=10)
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is True
        assert snapshot.value > 0  # Low fear = positive signal

    def test_snapshot_source_field(self, tmp_path):
        """Snapshot source identifies the signal."""
        data_file = tmp_path / "google_trends.json"
        trend_data = _make_trend_data()
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.source == "google_trends"

    def test_snapshot_value_bounded(self, tmp_path):
        """Signal value is bounded to [-1, 1]."""
        data_file = tmp_path / "google_trends.json"
        # Extreme spike
        trend_data = _make_trend_data(days=100, base_volume=50, recent_volume=999)
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert -1.0 <= snapshot.value <= 1.0

    def test_multiple_terms_averaged(self, tmp_path):
        """Multiple search terms are averaged into one signal."""
        data_file = tmp_path / "google_trends.json"
        recession_data = _make_trend_data(days=100, base_volume=50, recent_volume=50)
        inflation_data = _make_trend_data(days=100, base_volume=50, recent_volume=50)
        data_file.write_text(json.dumps({
            "recession": recession_data,
            "inflation": inflation_data,
        }))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is True

    def test_insufficient_data_returns_inactive(self, tmp_path):
        """Less than 14 days of data returns inactive."""
        data_file = tmp_path / "google_trends.json"
        short_data = _make_trend_data(days=5)
        data_file.write_text(json.dumps({"recession": short_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is False

    def test_stale_data_returns_inactive(self, tmp_path):
        """Data older than 14 days returns inactive."""
        data_file = tmp_path / "google_trends.json"
        old_data = {}
        for i in range(100):
            date = (datetime.now() - timedelta(days=30 + i)).strftime("%Y-%m-%d")
            old_data[date] = 50
        data_file.write_text(json.dumps({"recession": old_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert snapshot.is_active is False

    def test_stale_data_snapshot_exposes_machine_readable_reason(self, tmp_path):
        """Inactive stale data snapshots include a reason dashboards can surface."""
        data_file = tmp_path / "google_trends.json"
        old_data = {}
        for i in range(100):
            date = (datetime.now() - timedelta(days=30 + i)).strftime("%Y-%m-%d")
            old_data[date] = 50
        data_file.write_text(json.dumps({"recession": old_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()

        assert snapshot.is_active is False
        assert snapshot.metadata["inactive_reason"].startswith("Data is ")
        assert snapshot.metadata["inactive_reason"].endswith("days old (max 14)")
        assert snapshot.metadata["inactive_category"] == "stale"

    def test_confidence_reflects_data_quality(self, tmp_path):
        """Confidence is higher with more data points and recent data."""
        data_file = tmp_path / "google_trends.json"
        trend_data = _make_trend_data(days=100, base_volume=50, recent_volume=80)
        data_file.write_text(json.dumps({"recession": trend_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snapshot = signal.get_signal_snapshot()
        assert 0.0 < snapshot.confidence <= 1.0

    def test_reload_on_new_data(self, tmp_path):
        """Signal reloads data when file is updated."""
        data_file = tmp_path / "google_trends.json"
        # First read: normal
        normal_data = _make_trend_data(days=100, base_volume=50, recent_volume=50)
        data_file.write_text(json.dumps({"recession": normal_data}))

        signal = GoogleTrendsSignal(data_path=str(data_file))
        snap1 = signal.get_signal_snapshot()

        # Update file: fear spike
        fear_data = _make_trend_data(days=100, base_volume=50, recent_volume=200)
        data_file.write_text(json.dumps({"recession": fear_data}))

        snap2 = signal.get_signal_snapshot()
        # Signal should change after file update
        assert snap2.value != snap1.value
